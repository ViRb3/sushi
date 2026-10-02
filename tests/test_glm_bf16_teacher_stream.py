import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from capture_glm_bf16_teacher import Store, routed
from glm_teacher_io import BoundedLru


class StreamTests(unittest.TestCase):
    def test_selected_experts_match_resident(self):
        mx.random.seed(53)
        weights = {}
        resident = {}
        for proj, shape in [('gate_proj',(5,8)),('up_proj',(5,8)),('down_proj',(8,5))]:
            resident[proj] = (mx.random.normal((4,*shape))*0.2).astype(mx.bfloat16)
            for expert in range(4):
                weights[f'model.language_model.layers.3.mlp.experts.{expert}.{proj}.weight'] = resident[proj][expert]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            mx.save_safetensors(str(p/'weights.safetensors'), weights)
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'weights.safetensors' for k in weights}}))
            store = Store(p)
            try:
                key = next(iter(weights))
                self.assertTrue(mx.array_equal(store.read(key).view(mx.uint16), weights[key].view(mx.uint16)).item())
                store.cache = BoundedLru(160)
                store.cache_enabled = True
                for ids in ([0,1], [1,2], [0,3], [3,3]):
                    got=store.experts(3,ids,'gate_proj')
                    mx.eval(got)
                    np.testing.assert_array_equal(np.array(got.view(mx.uint16)),np.array(resident['gate_proj'][mx.array(ids)].view(mx.uint16)))
                    self.assertLessEqual(store.cache.bytes,store.cache.limit)
                self.assertGreater(store.cache_hits,0)
                for rows in (1,3):
                    x = mx.random.normal((1,rows,8)).astype(mx.bfloat16)
                    ids = mx.array([[[3,1],[1,2],[0,3]]],dtype=mx.uint32)[:,:rows]
                    scores = mx.array([[[0.25,0.75]]]*rows).reshape(1,rows,2)
                    store.cache_enabled=False
                    uncached = routed(store,3,x,ids,scores,10.0)
                    store.cache_enabled=True
                    actual = routed(store,3,x,ids,scores,10.0)
                    np.testing.assert_array_equal(np.array(uncached.view(mx.uint16)),np.array(actual.view(mx.uint16)))
                    expanded = mx.expand_dims(x,(-2,-3))
                    g = mx.gather_mm(expanded,resident['gate_proj'].swapaxes(-1,-2),rhs_indices=ids)
                    u = mx.gather_mm(expanded,resident['up_proj'].swapaxes(-1,-2),rhs_indices=ids)
                    a = nn.silu(mx.minimum(g,10.0))*mx.clip(u,-10.0,10.0)
                    expected = mx.gather_mm(a,resident['down_proj'].swapaxes(-1,-2),rhs_indices=ids)
                    expected = (expected.squeeze(-2)*scores[...,None]).sum(axis=-2).astype(x.dtype)
                    mx.eval(actual,expected)
                    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))
            finally:
                store.close()

    def test_bad_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            mx.save_safetensors(str(p/'weights.safetensors'),{'x':mx.ones((2,2),dtype=mx.uint32)})
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'weights.safetensors'}}))
            store=Store(p)
            try:
                with self.assertRaisesRegex(ValueError,'Unexpected reference dtype'):
                    store.read('x')
            finally:
                store.close()


if __name__=='__main__':
    unittest.main()
