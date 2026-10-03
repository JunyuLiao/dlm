"""CPU contract tests only; no torch/CUDA import or device qualification."""
import unittest
import numpy as np
from experiments.numerical_qk_reuse.v29_sparse_q_cycles import plan_cycles, apply_reference, amortization_receipt

class TestSparseQCycles(unittest.TestCase):
    def test_identity_no_programs_or_bytes(self):
        plan=plan_cycles(np.tile(np.arange(8),(2,1)))
        self.assertEqual(plan.cycles,())
        self.assertEqual(plan.statistics()['restoration_inclusive_logical_read_write_bytes'],0)
        self.assertEqual(plan.statistics()['permutation_launches_per_restored_call'],0)

    def check_order(self,order):
        order=np.asarray(order,dtype=np.int32);h,q=order.shape
        query=np.arange(h*q*7,dtype=np.int16).reshape(h,q,7);original=query.copy()
        plan=plan_cycles(order);actual=apply_reference(query,plan)
        np.testing.assert_array_equal(actual,query[np.arange(h)[:,None],order])
        np.testing.assert_array_equal(apply_reference(actual,plan,inverse=True),original)
        np.testing.assert_array_equal(query,original)
        owned={(head,row) for head,cycle in plan.cycles for row in cycle}
        self.assertEqual(len(owned),plan.moved_rows)
        self.assertEqual(owned,set(zip(*np.where(order!=np.arange(q)[None]))))
        self.assertEqual(plan.moved_rows,np.count_nonzero(order!=np.arange(q)[None]))

    def test_two_cycle(self):self.check_order([[1,0,2,3]])
    def test_long_cycle(self):self.check_order([[1,2,3,0]])
    def test_disjoint_cycles(self):self.check_order([[1,0,3,2]])
    def test_per_head_different_cycles(self):self.check_order([[2,0,1,3],[0,2,3,1]])

    def test_random_per_head_cycles_and_nonoverlap(self):
        rng=np.random.default_rng(2931);order=np.stack([rng.permutation(128) for _ in range(16)])
        plan=plan_cycles(order);query=rng.integers(-32768,32767,(16,128,13),dtype=np.int16)
        np.testing.assert_array_equal(apply_reference(query,plan),query[np.arange(16)[:,None],order])
        np.testing.assert_array_equal(apply_reference(apply_reference(query,plan),plan,True),query)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):plan_cycles(np.array([[0,0]]))
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):plan_cycles(np.array([[-1,0]]))
    def test_float_rejected(self):
        with self.assertRaises(ValueError):plan_cycles(np.array([[0.,1.]]))
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):plan_cycles(np.empty((1,0),int))
    def test_dimension_rejected(self):
        with self.assertRaises(ValueError):plan_cycles(np.arange(4))

    def test_geometry_and_byte_scope(self):
        plan=plan_cycles(np.array([[1,0,2],[0,2,1]]));stats=plan.statistics()
        self.assertEqual(stats['moved_rows'],4)
        self.assertEqual(stats['one_direction_logical_read_write_bytes'],4*512*2*2)
        self.assertEqual(stats['restoration_inclusive_logical_read_write_bytes'],4*512*2*4)
        with self.assertRaises(ValueError):apply_reference(np.zeros((1,3,8)),plan)

    def test_nonpositive_saving_has_no_break_even(self):
        self.assertIsNone(amortization_receipt(1,1,1)['optimistic_break_even_calls'])
        self.assertFalse(amortization_receipt(1,1,2,16)['breaks_even_within_measured_lifetime'])

    def test_positive_saving_does_not_invent_reuse(self):
        result=amortization_receipt(1,.5,.48)
        self.assertAlmostEqual(result['total_component_saving_seconds_per_call'],.00002)
        self.assertGreaterEqual(result['optimistic_break_even_calls'],50000)
        self.assertIsNone(result['breaks_even_within_measured_lifetime'])
        self.assertFalse(amortization_receipt(1,.5,.48,16)['breaks_even_within_measured_lifetime'])

    def test_invalid_costs_rejected(self):
        for args in ((-1,1,1),(1,float('nan'),1),(1,1,1,0)):
            with self.assertRaises(ValueError):amortization_receipt(*args)

if __name__=='__main__':unittest.main()
