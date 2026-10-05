import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from experiments.numerical_qk_reuse.v29_lse_merge import merge_reference, validate_order


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.o = np.arange(2*5*3*7, dtype=np.float32).reshape(2, 5, 3, 7)
        self.lse = np.zeros((2, 3, 5), dtype=np.float32)
        self.order = np.broadcast_to(np.arange(5), (3, 5)).copy()

    def test_identity_average(self):
        np.testing.assert_array_equal(merge_reference(self.o,self.lse,self.order)[0], self.o.mean(0))

    def test_different_head_maps_restore_rows(self):
        self.order[0] = [2,4,0,1,3]
        self.order[1] = [4,3,2,1,0]
        result = merge_reference(self.o,self.lse,self.order)[0]
        for head in range(3):
            np.testing.assert_array_equal(result[self.order[head],head], self.o[:, :, head].mean(0))

    def test_one_empty_split(self):
        self.lse[0] = -np.inf
        np.testing.assert_array_equal(merge_reference(self.o,self.lse,self.order)[0], self.o[1])

    def test_extreme_stable_lse(self):
        self.lse[0] = 10000
        self.lse[1] = -10000
        np.testing.assert_array_equal(merge_reference(self.o,self.lse,self.order)[0], self.o[0])

    def test_common_lse_shift_invariance(self):
        self.lse[0] = -3
        np.testing.assert_allclose(merge_reference(self.o,self.lse,self.order), merge_reference(self.o,self.lse+8000,self.order))

    def test_duplicate_and_float_map_rejected(self):
        for value in (np.zeros_like(self.order),self.order.astype(float)):
            with self.assertRaises(ValueError):validate_order(value,3,5)

    def test_lse_invalid_rejected(self):
        for value in (np.full_like(self.lse,-np.inf),np.full_like(self.lse,np.inf),np.full_like(self.lse,np.nan),self.lse[:1]):
            with self.assertRaises(ValueError):merge_reference(self.o,value,self.order)

    def test_nonfinite_partial_rejected(self):
        self.o[0,0,0,0] = np.nan
        with self.assertRaises(ValueError):merge_reference(self.o,self.lse,self.order)

    def test_invalid_partial_geometry(self):
        for value in (self.o[:1],self.o[:,:,:,:0],self.o[0]):
            with self.assertRaises(ValueError):merge_reference(value,self.lse,self.order)

    def test_bench_rejects_overwrite_before_selection_or_gpu(self):
        from scripts.v29_regroup_merge_bench import main
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json'
            output.write_text('retain')
            with patch('scripts.v29_regroup_merge_bench.held.select_accepted') as select:
                with self.assertRaises(ValueError):main([folder,str(output)])
                select.assert_not_called()
            self.assertEqual(output.read_text(),'retain')

    def test_incremental_interpretation(self):
        from scripts.v29_regroup_merge_bench import interpretation
        self.assertIn('held_fused only to natural_fused',interpretation())


if __name__ == '__main__':unittest.main()
