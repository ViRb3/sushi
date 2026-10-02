"""Diagnostic lossless BF16 expert streaming over oMLX's GLM forward.

This is a reference sanity runner, not the native Sushi implementation.
"""
import argparse
import concurrent.futures
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
from glm_teacher_io import BoundedLru, Journal, atomic_json
import struct
import sys
import time
import types

os.environ["MLX_ENABLE_TF32"] = "0"
os.environ["OMLX_GLM_HC_PREFILL"] = "0"
os.environ["OMLX_GLM53_KDA_PREFILL_FUSED"] = "0"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

OMLX = None
MODEL = None


def language_module():
    # This oMLX revision imports an optional fast_ops module absent from its
    # pinned mlx-vlm. Use the existing HC reference operation for that helper.
    name = 'mlx_vlm.models.fast_ops'
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        def exact_hc_norm(connection, norm, x, mixes):
            collapsed, post, comb = connection(x)
            return norm(collapsed), post, comb
        module.exact_hc_norm = exact_hc_norm
        sys.modules[name] = module
    name = 'mlx_vlm.models.linear'
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        module.DECODE_BLOCK_SIZE = 8
        sys.modules[name] = module
    from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
    if not apply_mlx_vlm_glm5_next_compat_patch():
        raise RuntimeError('Cannot register the oMLX GLM model')
    import mlx_vlm.models.glm5_next.language as language
    # The reference arm cannot trace file I/O inside a compiled FFN.
    language._DECODE_FUSION = False
    return language


class Store:
    def __init__(self, path):
        self.path = Path(path)
        index = json.loads((self.path / 'model.safetensors.index.json').read_text())
        self.entries = {}
        self.fds = {}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.cache = BoundedLru(0)
        self.cache_enabled = False
        self.cache_hits = 0
        self.budget = 100 * 1024**3
        self.bytes_read = 0
        self.read_seconds = 0.0
        for name in sorted(set(index['weight_map'].values())):
            p = self.path / name
            fd = os.open(p, os.O_RDONLY)
            self.fds[name] = fd
            n = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            if n > 128 * 1024**2:
                raise ValueError('Invalid safetensors header')
            header = json.loads(os.pread(fd, n, 8))
            size = os.fstat(fd).st_size
            for key, e in header.items():
                if key == '__metadata__':
                    continue
                a, b = e['data_offsets']
                if index['weight_map'].get(key) != name or not 0 <= a <= b <= size - 8 - n:
                    raise ValueError(f'Invalid tensor span: {key}')
                self.entries[key] = (fd, 8+n+a, b-a, e['shape'], e['dtype'])
        if set(self.entries) != set(index['weight_map']):
            raise ValueError('Index/header mismatch')

    def read(self, key):
        fd, offset, count, shape, dtype = self.entries[key]
        if dtype not in ('BF16', 'F32'):
            raise ValueError(f'Unexpected reference dtype {dtype}: {key}')
        blob = os.pread(fd, count, offset)
        if len(blob) != count:
            raise IOError(f'Short tensor read: {key}')
        numpy_dtype = {'BF16': np.uint16, 'F32': np.float32, 'F16': np.float16}[dtype]
        a = np.frombuffer(blob, dtype=numpy_dtype).reshape(shape)
        value = mx.array(a)
        if dtype == 'BF16':
            value = value.view(mx.bfloat16)
        self.bytes_read += count
        return value

    def experts(self, layer, ids, projection):
        keys = [f'model.language_model.layers.{layer}.mlp.experts.{i}.{projection}.weight' for i in ids]
        if self.cache_enabled:
            values=[]
            for key in keys:
                value=self.cache.get(key)
                if value is None:
                    size=self.entries[key][2]
                    if self.entries[key][4] != 'BF16':raise ValueError('Non-BF16 routed expert')
                    self.cache.reserve(size)
                    value=self.read(key);mx.eval(value)
                    self.cache.put(key,value,size)
                else:self.cache_hits+=1
                values.append(value)
            result=mx.stack(values);mx.eval(result)
            return result
        specs = [self.entries[k] for k in keys]
        shape = specs[0][3]
        if any(s[3] != shape or s[4] != 'BF16' for s in specs):
            raise ValueError('Expert tensor shape/dtype mismatch')
        out = np.empty((len(ids), *shape), dtype=np.uint16)
        def fill(pair):
            i, (fd, off, count, _, _) = pair
            # preadv fills an allocated slice directly; no float conversion.
            view = memoryview(out[i]).cast('B')
            done = 0
            while done < count:
                n = os.preadv(fd, [view[done:]], off + done) if hasattr(os, 'preadv') else 0
                if not hasattr(os, 'preadv'):
                    blob = os.pread(fd, count-done, off+done)
                    n = len(blob)
                    view[done:done+n] = blob
                if n <= 0:
                    raise IOError('Short expert read')
                done += n
        start = time.perf_counter()
        list(self.pool.map(fill, enumerate(specs)))
        self.read_seconds += time.perf_counter() - start
        self.bytes_read += sum(s[2] for s in specs)
        return mx.array(out).view(mx.bfloat16)

    def close(self):
        self.cache = BoundedLru(0)
        self.pool.shutdown()
        for fd in self.fds.values():
            os.close(fd)


def routed(store, layer, x, ids, scores, limit):
    mx.eval(x, ids, scores)
    unique, inverse = np.unique(np.array(ids), return_inverse=True)
    local = mx.array(inverse.reshape(ids.shape).astype(np.uint32))
    route_x = mx.expand_dims(x, (-2, -3))
    gate = store.experts(layer, unique.tolist(), 'gate_proj')
    up = store.experts(layer, unique.tolist(), 'up_proj')
    g = mx.gather_mm(route_x, gate.swapaxes(-1, -2), rhs_indices=local)
    u = mx.gather_mm(route_x, up.swapaxes(-1, -2), rhs_indices=local)
    activation = nn.silu(mx.minimum(g, limit)) * mx.clip(u, -limit, limit)
    mx.eval(activation)
    del gate, up, g, u
    mx.clear_cache()
    down = store.experts(layer, unique.tolist(), 'down_proj')
    y = mx.gather_mm(activation, down.swapaxes(-1, -2), rhs_indices=local)
    y = (y.squeeze(-2) * scores[..., None]).sum(axis=-2).astype(x.dtype)
    mx.eval(y)
    del down, activation
    mx.clear_cache()
    return y


class StreamMoE(nn.Module):
    def __init__(self, old, store, layer, limit):
        super().__init__()
        self.gate = old.gate
        self.shared_experts = old.shared_experts
        self._store = store
        self._layer = layer
        self._limit = limit

    def __call__(self, x):
        self._store.cache_enabled = x.shape[1] == 1
        ids, scores = self.gate(x)
        result = routed(self._store, self._layer, x, ids, scores, self._limit)
        if self.shared_experts is not None:
            result = result + self.shared_experts(x)
        mx.eval(result)
        if x.shape[1] > 1:
            print(f'PREFILL_LAYER {self._layer} tokens={x.shape[1]} read_GiB={self._store.bytes_read/2**30:.3f}', flush=True)
        return result


class LMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def __call__(self, inputs, cache=None):
        return self.inner(inputs, cache=cache, num_logits_to_keep=1).logits

    def make_cache(self):
        return self.inner.make_cache()

    @property
    def layers(self):
        return self.inner.layers


def load_model(store):
    language = language_module()
    config = language.TextConfig.from_dict(json.loads((MODEL/'config.json').read_text())['text_config'])
    model = language.LanguageModel(config)
    for i, layer in enumerate(model.layers):
        layer.compile_ffn = False
        if isinstance(layer.mlp, language.Glm5NextMoE):
            layer.mlp = StreamMoE(layer.mlp, store, i, config.swiglu_limit)
    weights = {}
    prefix = 'model.language_model.'
    for key in store.entries:
        if key.startswith(prefix):
            nk = 'model.' + key[len(prefix):]
            if key.startswith(prefix+'layers.'):
                if '.mlp.experts.' in key or int(key.split('.')[3]) >= config.num_hidden_layers:
                    continue
        elif key.startswith('lm_head.'):
            nk = key
        else:
            continue
        weights[nk] = store.read(key)
    source_parameter_dtypes={}
    for value in weights.values():
        name=str(value.dtype);source_parameter_dtypes[name]=source_parameter_dtypes.get(name,0)+1
    weights = model.sanitize(weights)
    sanitized_parameter_dtypes={}
    for value in weights.values():
        if value.dtype not in (mx.bfloat16,mx.float32):raise ValueError('Teacher sanitize changed weight precision')
        name=str(value.dtype);sanitized_parameter_dtypes[name]=sanitized_parameter_dtypes.get(name,0)+1
    store.trunk_dtype_audit=dict(source=source_parameter_dtypes,sanitized=sanitized_parameter_dtypes,
        lossless_promotions='oMLX sanitize widens HC/router BF16 toFP32; conv weights concatenate/transpose only')
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    del weights
    gc.collect()
    mx.clear_cache()
    return LMWrapper(model)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory(store):
    active=mx.get_active_memory();cached=mx.get_cache_memory()
    peak=mx.get_peak_memory();rss_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':rss_peak*=1024
    if active+cached>store.budget or rss_peak>store.budget:
        raise MemoryError(f'Teacher exceeded {store.budget} byte budget: active={active}, cache={cached}, peakRSS={rss_peak}')
    return dict(active_bytes=active,allocator_cache_bytes=cached,mlx_peak_bytes=peak,rss_peak_bytes=rss_peak,
                expert_cache_bytes=store.cache.bytes,expert_cache_limit=store.cache.limit,expert_cache_hits=store.cache_hits)


def main():
    global MODEL, OMLX
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--omlx',type=Path,required=True)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ssd-budget-gb',type=int,default=100,choices=[100])
    parser.add_argument('--prefix-chunk',type=int,default=512,choices=[512])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();MODEL=args.model.resolve();OMLX=args.omlx.resolve();out=args.out.resolve()
    sys.path.insert(0,str(OMLX));out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False,local_files_only=True)
    prompts=json.loads(args.prompts.read_text())
    if len(prompts)!=4 or sorted(p['category'] for p in prompts)!=['code','code','prose','prose']:
        raise ValueError('Exactly two code and two prose prompts required')
    if len(set(p['id'] for p in prompts))!=4 or any(not p['id'].replace('-','').isalnum() for p in prompts):raise ValueError('Invalid prompt identifiers')
    for p in prompts:
        p['ids']=tokenizer.encode(p['text'],add_special_tokens=False)
        if not 1<=len(p['ids'])<=512:raise ValueError('Raw seed must contain1..512tokens')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OMLX,text=True).strip()
    source=Store(MODEL)
    try:
        counts={};total=0
        for key,entry in source.entries.items():
            dtype=entry[4];counts[dtype]=counts.get(dtype,0)+1;total+=entry[2]
            if dtype not in ('BF16','F32'):raise ValueError(f'Lossless BF16 teacher rejects {dtype}: {key}')
            if '.mlp.experts.' in key and dtype!='BF16':raise ValueError(f'Expert is not BF16: {key}')
        config=json.loads((MODEL/'config.json').read_text());text_cfg=config.get('text_config',config)
        vocab=text_cfg['vocab_size'];eos=text_cfg.get('eos_token_id',config.get('eos_token_id',[]))
        eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
        import importlib.metadata
        identity=dict(model=str(MODEL),config_sha256=sha(MODEL/'config.json'),index_sha256=sha(MODEL/'model.safetensors.index.json'),
                      tokenizer_sha256=sha(MODEL/'tokenizer.json'),prompts_sha256=sha(args.prompts),script_sha256=sha(__file__),
                      io_sha256=sha(Path(__file__).with_name('glm_teacher_io.py')),omlx_revision=revision,
                      mlx_version=importlib.metadata.version('mlx'),transformers_version=importlib.metadata.version('transformers'),python_version=sys.version,tokens_per_prompt=512,prefix_chunk=args.prefix_chunk,
                      no_template=True,kv_cache_format='bf16',kda_state_format='float32',quantization='none',mtp=False,
                      mlx_enable_tf32='0',omlx_glm_hc_prefill='0',omlx_glm53_kda_prefill_fused='0',decode_fusion=False,ssd_budget_gb=args.ssd_budget_gb,budget_bytes=args.ssd_budget_gb*1024**3)
        prior=out/'identity.json'
        if prior.exists() and json.loads(prior.read_text())!=identity:raise ValueError('Capture identity changed; choose a new output directory')
        atomic_json(prior,identity)
        atomic_json(out/'prepared.json',dict(identity=identity,source_tensor_dtypes=counts,source_tensor_bytes=total,
                                           indexed_shards=len(source.fds),prompts=prompts,eos_token_ids=eos,complete=False))
        if args.prepare_only:
            print(json.dumps(dict(status='prepared',out=str(out),source_dtypes=counts,prompt_tokens=[len(p['ids']) for p in prompts])),flush=True);return
        source.budget=identity['budget_bytes']
        mx.set_memory_limit(source.budget);mx.set_cache_limit(0)
        recommended=mx.device_info().get('max_recommended_working_set_size',source.budget)
        mx.set_wired_limit(min(source.budget,recommended))
        print('LOAD_BEGIN',json.dumps(identity),flush=True)
        started=time.perf_counter();model=load_model(source)
        loaded=mx.get_active_memory();reserve=24*1024**3
        if loaded+reserve>=source.budget:raise MemoryError('Trunk leaves insufficient teacher working reserve')
        source.cache=BoundedLru(source.budget-loaded-reserve)
        atomic_json(out/'loaded.json',dict(memory=memory(source),trunk_dtype_audit=source.trunk_dtype_audit))
        print('MODEL_LOADED',json.dumps(memory(source)),flush=True)
        records=[]
        for index,p in enumerate(prompts):
            directory=out/'prompts'/f"{index:02d}_{p['id']}"
            journal=Journal(directory,identity,p['ids'],vocab)
            if len(journal.tokens)>512:raise ValueError('Too many committed continuation rows')
            for name,value in [('id.txt',p['id']),('prompt.txt',p['text']),('rendered_prompt.txt',p['text']),('prompt_tokens.txt',','.join(map(str,p['ids']))+'\n')]:
                (directory/name).write_text(value)
            if len(journal.tokens)<512:
                cache=model.make_cache()
                for begin in range(0,len(p['ids']),args.prefix_chunk):
                    logits=model(mx.array([p['ids'][begin:begin+args.prefix_chunk]],dtype=mx.uint32),cache=cache)
                    mx.eval(logits)
                replay=len(journal.tokens)
                for step in range(512):
                    row=np.asarray(logits.reshape(-1,vocab)[-1].astype(mx.float32),dtype='<f4')
                    if not np.isfinite(row).all():raise ValueError('Nonfinite BF16 teacher logits')
                    chosen=int(np.argmax(row));blob=row.tobytes()
                    wide=row.astype(np.float64);maximum=float(wide.max());nll=maximum+float(np.log(np.exp(wide-maximum).sum()))-float(wide[chosen])
                    if step<replay:journal.verify(step,blob,chosen)
                    else:journal.append(blob,chosen,nll)
                    status=dict(phase='replay' if step<replay else 'capture',prompt=index,prompt_id=p['id'],category=p['category'],
                                completed_rows=len(journal.tokens),total_rows=512,completed_prompts=len(records),elapsed_seconds=time.perf_counter()-started,
                                source_read_bytes=source.bytes_read,memory=memory(source),complete=False)
                    atomic_json(out/'progress.json',status)
                    if step%16==0 or step==511:print('PROGRESS',json.dumps(status),flush=True)
                    if step==511:break
                    logits=model(mx.array([[chosen]],dtype=mx.uint32),cache=cache);mx.eval(logits)
                del cache,logits;gc.collect();mx.clear_cache()
            (directory/'generated_tokens.txt').write_text(','.join(map(str,journal.tokens))+'\n')
            nll=journal.nll_sum/512
            record=dict(id=p['id'],dir=str(directory.relative_to(out)),prompt_tokens=len(p['ids']),generated_tokens=512,
                        strict_nll_mean=nll,strict_perplexity=float(np.exp(nll)),category=p['category'],
                        first_eos_position=next((i for i,v in enumerate(journal.tokens) if v in eos),None))
            records.append(record);atomic_json(directory/'complete.json',record)
            atomic_json(out/'partial-baseline.json',dict(schema='mlx-serve-kld-baseline-v1',complete=False,prompts=records))
        result=dict(schema='mlx-serve-kld-baseline-v1',tool='sushi',label='glm53-lossless-bf16-stream-4x512-raw',model=str(MODEL),
                    run='glm53-lossless-bf16-stream-4x512-raw',kv_cache_format='bf16',inference_profile='greedy',ssd_budget_gb=args.ssd_budget_gb,
                    prompt_set=str(args.prompts.resolve()),tokens_per_prompt=512,top_k=10,elapsed_secs=time.perf_counter()-started,prompts=records,
                    complete=True,provenance=identity,memory=memory(source),source_tensor_dtypes=counts)
        atomic_json(out/'baseline.json',result);atomic_json(out/'progress.json',dict(phase='complete',complete=True,prompts=4,rows=2048,memory=memory(source)))
        print('COMPLETE',json.dumps(result),flush=True)
    except BaseException as error:
        atomic_json(out/'failure.json',dict(complete=False,error=type(error).__name__,message=str(error)))
        raise
    finally:source.close()

if __name__=='__main__':main()
