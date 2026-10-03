"""CPU contracts for independent same-state cost diagnostics; no CUDA imports."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts.v29_homogeneous_cost_bench import (Settings, check_split_builds,
    dense_prefix_reference, fixed_support, frozen_settings, main, rotate, snapshot_contract,
    tail_decision_reference, zero_counter_deltas)


class HomogeneousCostContracts(unittest.TestCase):
    def test_fixed_mask_protects_first_boundary_and_canvas(self):
        for prefix in (64,65,32768):
            kept=fixed_support(prefix,.01,9)
            self.assertTrue(kept[...,0].all())
            self.assertTrue(kept[...,prefix//64:].all())
            self.assertEqual(kept.shape,(1,16,2,(prefix+256+63)//64))
            np.testing.assert_array_equal(kept,fixed_support(prefix,.01,9))

    def test_allkept_endpoint_and_partial_canvas(self):
        self.assertTrue(fixed_support(65,1.,3).all())
        for density in (0.,-1.,1.1,float('nan')):
            with self.assertRaises(ValueError):fixed_support(64,density,3)

    def test_native_and_alias_parallelism_are_disclosed_not_equalized_to_s1(self):
        spec=frozen_settings(Settings(),'synthetic')
        self.assertEqual(spec['scale'],1.)
        self.assertEqual(spec['native']['num_splits'],0)
        self.assertTrue(spec['native']['causal'])
        self.assertFalse(spec['native']['dynamic_causal'])
        self.assertEqual(spec['sparse']['alias_splits'],2)
        self.assertEqual(spec['sparse']['inner_num_splits'],1)
        self.assertFalse(spec['request_speed_claim_allowed'])
        self.assertEqual(spec['settings']['output_atol'],.02)
        self.assertEqual(spec['settings']['dp_atol'],.0002)

    def test_bad_geometry_timing_or_tolerances_fail_closed(self):
        for kwargs in (dict(prefixes=()),dict(prefixes=(0,)),dict(prefixes=(106496,)),
                       dict(prefixes=(64,64)),dict(repeats=3),dict(warm=0),dict(seed=-1),
                       dict(keep_fraction=float('inf')),dict(output_atol=0.)):
            with self.assertRaises(ValueError):Settings(**kwargs).validate()

    def test_balanced_rotations_and_warmed_build_guard(self):
        names=['a','b','c','d']
        self.assertEqual({rotate(names,i)[0] for i in range(4)},set(names))
        self.assertEqual(rotate(names,4),names)
        check_split_builds({'a':2},{'a':2})
        with self.assertRaises(RuntimeError):check_split_builds({'a':2},{'a':3})

    def test_snapshot_absolute_page_mapping_bounds_and_identity(self):
        query=np.empty((256,16,512),dtype=np.int8)
        cache=np.empty((9,2,64,1024),dtype=np.int8)
        # Prefix65 + canvas256 uses6 pages. Physical order need not be affine.
        table=np.array([8,0,6,2,4,1],dtype=np.int32)
        snapshot_contract(query,cache,table,65,1.,False)
        for bad in (np.array([0,0,1,2,3,4]),np.array([-1,0,1,2,3,4]),
                    np.arange(5),np.arange(4,10),table.astype(float)):
            with self.assertRaises(ValueError):snapshot_contract(query,cache,bad,65,1.,False)
        for scale,dynamic in ((512**-.5,False),(1.,True)):
            with self.assertRaises(ValueError):snapshot_contract(query,cache,table,65,scale,dynamic)

    def test_independent_dense_prefix_two_tile_hand_calculation(self):
        z=np.array([[0.,0.]])
        mu=np.array([[[0.,0.],[2.,0.]]])
        risk,mass,means=dense_prefix_reference(z,mu)
        self.assertTrue(np.isposinf(risk[0,0]))
        self.assertAlmostEqual(risk[0,1],0.) # alpha=.5, delta=(1,0)
        np.testing.assert_allclose(means,[[[0.,0.],[1.,0.]]])
        self.assertAlmostEqual(mass[0,1],np.log(2))
        # Mass shift leaves normalized mean/risk unchanged.
        shifted=dense_prefix_reference(z+17.,mu)
        np.testing.assert_allclose(shifted[0],risk)
        np.testing.assert_allclose(shifted[2],means)

    def test_dense_prefix_zero_delta_and_nonfinite_guard(self):
        risk,_,_=dense_prefix_reference(np.zeros((2,3)),np.ones((2,3,4)))
        self.assertTrue(np.isneginf(risk[:,1:]).all())
        with self.assertRaises(ValueError):dense_prefix_reference([[float('nan')]],np.zeros((1,1,2)))

    def test_tail_partial_tile_and_strict_threshold_independent_case(self):
        # Identical projected mean gives exact zero risk, hence both tiles drop.
        got=tail_decision_reference(np.zeros((256,65)),np.ones((65,32)),
                                   np.zeros(256),np.ones((256,32)),1.,-3.)
        self.assertEqual(got.shape,(2,2))
        self.assertTrue(got.all())
        # One-key first tail: alpha=.5, ||mu-prev||=2, norm=1, log risk=0.
        sketch=np.zeros((1,32));sketch[0,0]=2.
        tied=tail_decision_reference(np.zeros((256,1)),sketch,np.zeros(256),np.zeros((256,32)),1.,0.)
        self.assertFalse(tied.any()) # strict '<', not '<='
        with self.assertRaises(ValueError):tail_decision_reference(np.full((256,1),np.nan),sketch,np.zeros(256),np.zeros((256,32)),1.,0.)

    def test_new_jit_capture_and_incomplete_counter_receipt_rejected(self):
        self.assertEqual(zero_counter_deltas({'jit':2,'capture':0},{'jit':2,'capture':0}),{'jit':0,'capture':0})
        for before,after in (({'jit':2},{'jit':3}),({'capture':0},{'capture':1}),
                             ({'jit':0},{}),({'jit':0},{'jit':False})):
            with self.assertRaises(RuntimeError):zero_counter_deltas(before,after)

    def test_spec_template_is_cpu_only(self):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream): result=main(['--print-spec'])
        self.assertEqual(json.loads(stream.getvalue()),result)
        self.assertEqual(result['settings']['prefixes'],[32768])

    def test_spec_or_existing_output_rejected_before_gpu_import(self):
        with TemporaryDirectory() as td:
            root=Path(td);report=root/'report.json';spec=root/'spec.json'
            spec.write_text(json.dumps({'incorrect':'spec'}),encoding='utf-8')
            args=[str(report),'--source-commit','a'*40,'--spec',str(spec)]
            with self.assertRaisesRegex(ValueError,'frozen spec'):main(args)
            report.write_text('user-owned',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'report must be new'):main(args)
            self.assertEqual(report.read_text(encoding='utf-8'),'user-owned')


if __name__=='__main__':unittest.main()
