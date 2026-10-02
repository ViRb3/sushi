import tempfile
import unittest
from pathlib import Path
from glm_teacher_io import Journal, BoundedLru

class TeacherIoTests(unittest.TestCase):
    def test_resume_discards_uncommitted_tail_and_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            j=Journal(p, {'source':'fixed'}, [7,8], 3)
            j.append(b'\0'*12, 2, 0.5)
            with j.data_path.open('ab') as f:f.write(b'x'*12)
            recovered=Journal(p, {'source':'fixed'}, [7,8], 3)
            self.assertEqual(recovered.tokens,[2])
            self.assertEqual(recovered.data_path.stat().st_size,12)
            with self.assertRaises(ValueError):Journal(p, {'source':'changed'}, [7,8], 3)
    def test_cache_never_exceeds_budget_and_hits_change_eviction_order(self):
        cache=BoundedLru(8)
        cache.put('a',b'aaaa',4);cache.put('b',b'bbbb',4)
        self.assertEqual(cache.get('a'),b'aaaa')
        cache.put('c',b'cccc',4)
        self.assertIsNone(cache.get('b'));self.assertEqual(cache.bytes,8)
        cache.put('oversize',b'x'*9,9)
        self.assertIsNone(cache.get('oversize'));self.assertLessEqual(cache.bytes,8)
if __name__=='__main__':unittest.main()
