"""CPU diagnostic contract tests; no torch/vLLM or GPU imports required."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from scripts import v29_dense_reference_diag as d


def base_spec():
    return json.loads((Path(__file__).resolve().parents[1]/'results/v28_20261002/specs/v28_q128_request_clear_seed4.json').read_text())


class Array:
    def __init__(self,value):self.value=value
    def tolist(self):return [self.value]


def fixtures():
    prep=[dict(mode='NONE',draft=False,tokens=32),dict(mode='FULL',draft=True,tokens=256),
          dict(mode='FULL',draft=True,tokens=256),dict(mode='PIECEWISE',draft=True,tokens=25)]
    snapshots=[(False,32,Array(0)),(True,256,Array(0)),(True,256,Array(0)),(True,25,Array(25))]
    counts=dict(prefill_steps=1,denoising_forwards=2,commit_forwards=1,scheduler_denoising_forwards=1,
                scheduler_commit_forwards=1,speculative_unused_denoising=1,prefill_s=.2,decode_span_s=.8)
    return prep,snapshots,counts


class Protocol(unittest.TestCase):
    def test_each_alias_maps_to_original_worker_and_exact_graph_mode(self):
        base=base_spec();before=copy.deepcopy(base)
        for alias,(arm,mode) in d.VARIANTS.items():
            spec=d.make_spec(base,alias,[0])
            self.assertEqual(d.validate_spec(spec,alias,1,1),arm)
            self.assertEqual(spec['arm_settings'][arm]['cudagraph_mode'],mode)
            self.assertEqual(len(spec['launch_order']),12)
            self.assertEqual(spec['control_indices'],{'longbench_v2_32k':[0]})
        self.assertEqual(base,before)

    def test_prior_event_profiler_settings_are_removed(self):
        base=base_spec();base.update(cuda_events=True,profile_ordinal=1,cuda_event_leaf_scopes=['x'],trace_export={'enabled':True})
        spec=d.make_spec(base,'default_dense',[0])
        self.assertFalse(spec['cuda_events'])
        self.assertNotIn('profile_ordinal',spec)
        d.validate_spec(spec,'default_dense',0,0)

    def test_mutated_graph_warm_jit_or_rng_contract_fails_before_gpu(self):
        original=d.make_spec(base_spec(),'piecewise_dense_nohook',[0])
        edits=[lambda s:s['arm_settings']['dense'].update(cudagraph_mode='default'),
               lambda s:s.update(control_indices={'longbench_v2_32k':[]}),
               lambda s:s.update(jit_event_receipts=False),
               lambda s:s['dense_reference_diagnostic'].update(reseed_or_restore_rng=True),
               lambda s:s.update(cuda_events=True)]
        for edit in edits:
            s=copy.deepcopy(original);edit(s)
            with self.assertRaises(ValueError):d.validate_spec(s,'piecewise_dense_nohook',0,0)

    def test_invalid_seed_inventory_and_engine_repeat_rejected(self):
        for seeds in ((1,1),(1,),(-1,2)):
            with self.assertRaises(ValueError):d.make_spec(base_spec(),'default_dense',[0],seeds)
        spec=d.make_spec(base_spec(),'default_dense',[0])
        with self.assertRaises(ValueError):d.validate_spec(spec,'default_dense',0,2)
        with self.assertRaises(ValueError):d.make_spec(base_spec(),'default_dense',[0,0])


class ActualExecution(unittest.TestCase):
    def test_unused_forward_and_partial_commit_keep_actual_runtime_modes(self):
        prep,snapshots,counts=fixtures()
        rows=d.classify_modes(prep,snapshots,counts)
        self.assertEqual([x['phase'] for x in rows],['prefill','denoise','denoise','commit'])
        self.assertEqual(rows[-1]['emitted_token_count'],25)
        self.assertEqual(rows[1]['cudagraph_mode'],'FULL')
        self.assertEqual(counts['scheduler_denoising_forwards'],1)

    def test_missing_or_misaligned_snapshot_and_initializer_mode_fail(self):
        prep,snapshots,counts=fixtures()
        cases=[(prep[:-1],snapshots),(prep,[(True,32,Array(0)),*snapshots[1:]]),
               ([dict(prep[0],mode='FULL_AND_PIECEWISE'),*prep[1:]],snapshots),
               (prep,[(False,32,Array(1)),*snapshots[1:]])]
        for p,s in cases:
            with self.assertRaises(ValueError):d.classify_modes(p,s,counts)

    def test_capture_and_other_request_ignored_multi_request_rejected(self):
        session=d.BoundarySession(lambda:'a');session.start();session.tracker=NS(rid='r')
        batch=NS(req_ids=['r'],num_reqs=1,num_tokens=256,num_draft_tokens=256)
        session.prepare(batch,'FULL',True)
        session.prepare(NS(**dict(vars(batch),req_ids=['other'])),'FULL',False)
        self.assertEqual(session.preparations,[])
        with self.assertRaises(ValueError):session.prepare(NS(**dict(vars(batch),num_reqs=2,req_ids=['r','s'])),'FULL',False)

    def test_warm_counts_output_length_and_rng_boundary_chain_retained(self):
        rng=iter(('a','b','b','c'))
        session=d.BoundarySession(lambda:next(rng))
        prep,snapshots,counts=fixtures();tracker=NS(rid='r',_execution_snapshots=snapshots)
        for _ in range(2):
            session.start();session.tracker=tracker;session.preparations=copy.deepcopy(prep)
            session.output([NS(finished=True,outputs=[NS(token_ids=range(17),finish_reason='stop')])])
            session.finish(tracker,counts,1.,2.)
        self.assertIsNone(session.rows[0]['rng_before_equals_previous_after'])
        self.assertTrue(session.rows[1]['rng_before_equals_previous_after'])
        self.assertEqual(session.rows[0]['output_tokens'],17)
        self.assertEqual(session.rows[0]['actual_counts']['denoising_forwards'],2)
        self.assertNotIn('cuda_rng_before_private',session.rows[0])

    def test_missing_output_length_fails_closed(self):
        session=d.BoundarySession(lambda:'a');session.start();session.tracker=NS(rid='r')
        with self.assertRaisesRegex(ValueError,'output length'):session.finish(session.tracker,fixtures()[2])

    def test_wrappers_restore_exact_functions_on_exception(self):
        original=lambda *args,**kwargs:object()
        dg=NS(DiffusionGemmaModelState=type('State',(),{'prepare_attn':original}))
        panel=NS(add_tracked_request=original)
        metrics=NS(PhaseTracker=type('Tracker',(),{'finalize':original}))
        vllm=NS(LLM=original)
        with self.assertRaises(RuntimeError):
            with d.instrument(d.BoundarySession(lambda:'a'),panel,metrics,dg,vllm):
                raise RuntimeError('test')
        self.assertIs(dg.DiffusionGemmaModelState.prepare_attn,original)
        self.assertIs(panel.add_tracked_request,original)
        self.assertIs(metrics.PhaseTracker.finalize,original)
        self.assertIs(vllm.LLM,original)

    def test_existing_output_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.json';path.write_text('original')
            with self.assertRaises(FileExistsError):d.write_new(path,{})
            self.assertEqual(path.read_text(),'original')


class ClosedValidation(unittest.TestCase):
    def receipt(self,directory):
        root=Path(directory)
        spec=d.make_spec(base_spec(),'default_dense',[0])
        spec_path=root/'spec.json';d.write_new(spec_path,spec)
        binding=dict(spec=str(spec_path),deploy_commit='1'*40)
        rng=iter(('a'*64,'b'*64,'b'*64,'c'*64))
        session=d.BoundarySession(lambda:next(rng))
        prep,snapshots,counts=fixtures();tracker=NS(rid='r',_execution_snapshots=snapshots)
        for _ in range(2):
            session.start();session.tracker=tracker;session.preparations=prep
            session.output([NS(finished=True,outputs=[NS(token_ids=range(17),finish_reason='stop')])])
            session.finish(tracker,counts,1.,2.)
        rows=[dict(row,warm=index==0) for index,row in enumerate(session.rows)]
        report=dict(complete=True,diagnostic_validation_passed=True,source_commit='1'*40,
                    protocol_id=spec['protocol_id'],variant='default_dense',engine_repeat=0,
                    engine_seed=28001,performance_claim_allowed=False,profiler_enabled=False,
                    rng_reset_or_restore=False,initialized_graph_modes=['FULL_AND_PIECEWISE'],requests=rows,
                    jit_monitor_activation=dict(active=True,mode='warn',cute_hook_installed=True,
                          triton_hook_callable=True,activated_before_component_imports=True))
        record=dict(denoise_forward_count=2,commit_forward_count=1,prefill_steps=1,output_tokens=17,
                    graph_captures_timed=0,compilation_deltas={
                      'jit_monitor_'+key:0 for key in ('triton_events','cute_events','unknown_events','all_events')})
        d.write_new(root/'dense_reference_diagnostic.json',report)
        d.write_new(root/'terminal.json',dict(complete=True,arm='dense',block=0,protocol_id=spec['protocol_id']))
        d.write_new(root/'rng_boundaries.private.json',dict(private=True,not_for_publication=True,requests=session.private_rows))
        (root/'records.jsonl').write_text(json.dumps(record)+'\n')
        return binding,report,record

    def test_complete_boundary_receipts_validate_even_when_rng_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            binding,report,_=self.receipt(directory)
            status=d.validate_run(directory,binding,'default_dense',0,0)
            self.assertTrue(status['diagnostic_validation_passed'])
            self.assertFalse(report['requests'][0]['rng_before_equals_after'])
            self.assertNotIn('requests',status)

    def test_missing_or_incomplete_monitor_activation_fails_closed(self):
        for key in ('active','mode','cute_hook_installed','triton_hook_callable','activated_before_component_imports'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                binding,report,_=self.receipt(directory)
                report['jit_monitor_activation'].pop(key)
                (Path(directory)/'dense_reference_diagnostic.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'monitor receipt'):
                    d.validate_run(directory,binding,'default_dense',0,0)

    def test_missing_runtime_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            binding,report,_=self.receipt(directory)
            report['requests'][1]['execution'].pop()
            (Path(directory)/'dense_reference_diagnostic.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):d.validate_run(directory,binding,'default_dense',0,0)

    def test_timed_jit_or_output_mismatch_rejected(self):
        for kind in ('jit','output'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                binding,_,record=self.receipt(directory)
                if kind=='jit':record['compilation_deltas']['jit_monitor_all_events']=1
                else:record['output_tokens']=18
                (Path(directory)/'records.jsonl').write_text(json.dumps(record)+'\n')
                with self.assertRaises(ValueError):d.validate_run(directory,binding,'default_dense',0,0)


if __name__=='__main__':unittest.main()
