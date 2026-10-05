"""Stdlib CPU compatibility/receipt checks for the planned fusion qualification.

No real model, GPU, SSH, private inputs, or performance inference.
"""
from copy import deepcopy
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from scripts import v29_expanded_panel as run
from scripts.v28_jit_receipts import COUNT_FIELDS
from tests.test_v29_expanded_panel import fake_receipt

ROOT=Path(__file__).resolve().parents[1]
SPEC_DIR=ROOT/'results/v29_20261002/specs'

def family():return json.loads((SPEC_DIR/'v29_engineering_fusion_family001.json').read_text())
def read_spec(entry):return json.loads((ROOT/entry['spec']).read_text())


class FusionSpecTests(unittest.TestCase):
    def test_all_six_specs_use_existing_validator(self):
        f=family();self.assertEqual(len(f['variants']),6)
        for entry in f['variants'].values():
            spec=read_spec(entry);self.assertIs(run.validate_spec(spec),spec)
            self.assertEqual(entry['protocol_id'],spec['protocol_id'])

    def test_qualification_is40_warm40_timed_not_full_stage(self):
        f=family();self.assertFalse(f['formal_launch_enabled']);self.assertIsNone(f['deployment_commit'])
        self.assertEqual(f['initial_stage'],'qualification_only')
        self.assertEqual(f['qualification_inventory']['warm_requests'],40)
        self.assertEqual(f['qualification_inventory']['timed_requests'],40)
        self.assertEqual(sum(x['qualification_workers'] for x in f['variants'].values()),40)

    def test_full_inventory_compatibility_does_not_trim_items(self):
        expected={'longbench':1888,'aime':960,'humaneval':5248}
        for entry in family()['variants'].values():
            spec=read_spec(entry);self.assertEqual(len(run.old.expected_inventory(spec)),expected[entry['suite']])
            self.assertEqual([b['engine_seed'] for b in spec['blocks']],list(range(30001,30009)))
            self.assertTrue(all(x['budget']==8192 for x in spec['task_contracts'].values()))

    def test_pairs_differ_only_declared_backend_and_identity(self):
        f=family()
        for a,b in f['comparison_pairs']:
            first,second=read_spec(f['variants'][a]),read_spec(f['variants'][b])
            self.assertNotEqual(first['protocol_id'],second['protocol_id'])
            for spec in (first,second):
                for key in ('name','protocol_id'):spec.pop(key)
                spec['engineering_optimization'].pop('variant')
                spec['adapter_settings'].update(kv_copy_backend='toy',merge_backend='toy')
            self.assertEqual(first,second)

    def test_same_suite_single_host_and_order(self):
        f=family();expected={'longbench':'dllm','aime':'dlm2','humaneval':'mpk'}
        for entry in f['variants'].values():
            spec=read_spec(entry)
            self.assertEqual({b['host'] for b in spec['blocks']},{expected[entry['suite']]})
            self.assertEqual(spec['adapter_settings']['lifecycle'],'request_clear')
            self.assertEqual(spec['adapter_settings']['canvas_buffers'],'legacy')
            self.assertEqual(spec['adapter_settings']['alias_splits'],2)

    def test_original_main_no_gate_q128_unchanged(self):
        for entry in family()['variants'].values():
            self.assertEqual(read_spec(entry)['primary_receipt_method'],run.METHOD)
            self.assertEqual(read_spec(entry)['primary_receipt_method']['min_route_keys'],0)

    def test_private_free_public_specs(self):
        for value in [family()]+[read_spec(x) for x in family()['variants'].values()]:
            raw=json.dumps(value)
            for bad in ('/home/','/media/','E:/','prompt_tokens','gold_sha256','completion_text','gpu_uuid'):
                self.assertNotIn(bad,raw)

    def test_gpu_queue_after_all_strict_score(self):
        f=family();gate=f['stage_gate']
        self.assertTrue(gate['mpk_cpu_scorer_idle']);self.assertTrue(gate['no_overlap_with_existing_formal_or_scoring'])
        self.assertIn('strict-scored',gate['required_completion'])

    def test_real_full_receipts_accept_torch_and_fused(self):
        for entry in family()['variants'].values():
            settings=run.execution_settings(read_spec(entry))
            for arm in run.ARMS:
                receipt=fake_receipt(arm,settings)
                if entry['variant']=='fused' and arm in ('method','allkept'):
                    receipt['adapter'].update(triton_kv_copy_calls=50,triton_kv_copy_elements=1024,triton_lse_merge_calls=50)
                run.validate_receipt(arm,receipt,10,run.METHOD,settings)

    def test_fused_missing_copy_or_merge_rejected_on_both_matched_arms(self):
        settings=run.execution_settings(read_spec(family()['variants']['aime_fused']))
        for arm in ('allkept','method'):
            for missing in ('triton_kv_copy_calls','triton_kv_copy_elements','triton_lse_merge_calls'):
                receipt=fake_receipt(arm,settings);receipt['adapter'].update(triton_kv_copy_calls=1,triton_kv_copy_elements=2,triton_lse_merge_calls=1)
                del receipt['adapter'][missing]
                with self.assertRaises(ValueError):run.validate_receipt(arm,receipt,10,run.METHOD,settings)

    def test_torch_receipts_reject_accidental_fused_path(self):
        settings=run.execution_settings(read_spec(family()['variants']['aime_legacy']))
        for arm in ('allkept','method'):
            for field in ('triton_kv_copy_calls','triton_kv_copy_elements','triton_lse_merge_calls'):
                receipt=fake_receipt(arm,settings);receipt['adapter'][field]=1
                with self.assertRaises(ValueError):run.validate_receipt(arm,receipt,10,run.METHOD,settings)

    def test_fused_backend_constructor_and_context_restore(self):
        class Adapter:
            def __init__(self,kv_copy_backend='torch',merge_backend='torch'):
                self.kv_copy_backend=kv_copy_backend;self.merge_backend=merge_backend
        module=types.SimpleNamespace(VllmMethodAdapter=Adapter)
        settings=run.execution_settings(read_spec(family()['variants']['aime_fused']))
        original=run.panel.validate_receipts
        with patch.object(run.qualify,'adapter_variant',side_effect=lambda base,*args:base):
            with run.instrument_runner(module,settings,1):
                adapter=module.VllmMethodAdapter(kv_copy_backend='torch',merge_backend='torch')
                self.assertEqual((adapter.kv_copy_backend,adapter.merge_backend),('triton','triton'))
        self.assertIs(module.VllmMethodAdapter,Adapter);self.assertIs(run.panel.validate_receipts,original)

    def test_unapproved_margin_or_gate_cannot_be_added(self):
        for mutate in (lambda s:s['accuracy_analysis'].update(noninferiority_margin=.05),lambda s:s['primary_receipt_method'].update(min_route_keys=2048)):
            spec=read_spec(family()['variants']['aime_fused']);mutate(spec)
            with self.assertRaises(ValueError):run.validate_spec(spec)

    def test_task_budget_cannot_be_shrunk_for_qualification(self):
        spec=read_spec(family()['variants']['humaneval_fused']);spec['task_contracts']['humaneval']['budget']=2048
        with self.assertRaises(ValueError):run.validate_spec(spec)

    def test_jit_event_in_timed_row_still_rejected(self):
        delta={'jit_monitor_'+key:0 for key in COUNT_FIELDS}
        delta['jit_monitor_triton_events']=1
        with self.assertRaises(ValueError):run.qualify.validate_jit_deltas([dict(compilation_deltas=delta)],True)


if __name__=='__main__':unittest.main()
