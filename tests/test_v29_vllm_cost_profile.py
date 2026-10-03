"""CPU-only diagnostic accounting and lifecycle checks; no torch needed."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.v29_vllm_cost_profile import (PREFIX, ProfileSession, attention_label,
    buffer_label, host_category, instrument, nearest_label, summarize_events,
    validate_32k, wrapped, profile_counts, phase_attention_label, summarize_cuda_events, validate_event_settings)


def event(name, parent=None, kernels=(), device='CPU', cpu=0., self_cpu=0., gpu=0.):
    return NS(name=name, cpu_parent=parent, kernels=[NS(duration=x) for x in kernels],
              device_type=NS(name=device), cpu_time_total=cpu, self_cpu_time_total=self_cpu,
              time_range=NS(elapsed_us=lambda: gpu))


class CostProfileTests(unittest.TestCase):
    def test_official_shared_gemma4_attention_prefixes(self):
        # vLLM0.30.0 diffusion_gemma.py:237-240 + gemma4.py:606,532,1079.
        for index in range(30):
            expected = 'global_attention' if index in (5,11,17,23,29) else 'local_attention'
            self.assertEqual(attention_label(f'model.layers.{index}.self_attn.attn'), expected)

    def test_unrelated_and_encoder_prefixes_are_not_layer_receipts(self):
        for name in ('model.encoder.layers.5.self_attn.attn',
                     'model.decoder.layers.5.self_attn.attn',
                     'model.layers.5.mlp', 'model.layers.5.self_attn',
                     'other.model.layers.5.self_attn.attn',
                     'model.layers.30.self_attn.attn', 'model.layers.-1.self_attn.attn',
                     'model.layers.5.self_attn.attn.extra', 'private-arbitrary'):
            with self.subTest(name=name):
                self.assertEqual(attention_label(name), 'attention_other')

    def test_copy_first_build_refresh_and_new_canvas(self):
        adapter = NS(buffers={})
        self.assertEqual(buffer_label(adapter, 5, 32768, 256), 'kv_prefix_build')
        adapter.buffers[5] = dict(prefix=32768, nk=33024)
        self.assertEqual(buffer_label(adapter, 5, 32768, 256), 'kv_canvas_refresh')
        self.assertEqual(buffer_label(adapter, 5, 33024, 256), 'kv_prefix_build')
        self.assertEqual(buffer_label(adapter, 5, 32768, 128), 'kv_prefix_build')

    def test_phase_tags_reuse_existing_exact_context_without_gpu_read(self):
        va=NS(_ACTIVE=NS(bound=True, step_ctx={'encoder':False}))
        self.assertEqual(phase_attention_label(va,'model.layers.5.self_attn.attn'), 'denoise_global_attention')
        va._ACTIVE.step_ctx={'encoder':True}
        self.assertEqual(phase_attention_label(va,'model.layers.5.self_attn.attn'), 'encoder_global_attention')
        va._ACTIVE=None
        self.assertEqual(phase_attention_label(va,'model.layers.5.self_attn.attn'), 'global_attention')

    def test_nearest_scope_kernel_ownership_avoids_inclusive_double_count(self):
        outer = event(PREFIX+'global_attention', cpu=900)
        inner = event(PREFIX+'selector', outer, cpu=300)
        op = event('sensitive_operator_name', inner, kernels=(40, 60))
        unrelated = event('other', kernels=(20,))
        cuda = event('private-kernel', device='CUDA', gpu=120)
        result = summarize_events([outer, inner, op, unrelated, cuda], {'global_attention':1, 'selector':1})
        self.assertAlmostEqual(result['directly_associated_gpu_ms']['selector'], .1)
        self.assertNotIn('global_attention', result['directly_associated_gpu_ms'])
        self.assertAlmostEqual(result['raw_cuda_activity_sum_ms'], .12)
        self.assertEqual(result['directly_associated_kernel_records'], 3)
        self.assertNotIn('sensitive_operator_name', str(result))
        self.assertNotIn('private-kernel', str(result))

    def test_host_wait_copy_api_and_d2h_are_separate(self):
        parent = event(PREFIX+'prepare_metadata')
        events = [event('cudaStreamSynchronize', parent, cpu=110, self_cpu=100),
                  event('aten::_local_scalar_dense', parent, cpu=130, self_cpu=20),
                  event('cudaMemcpyAsync', parent, cpu=9, self_cpu=9),
                  event('Memcpy DtoH', device='CUDA', gpu=4)]
        result = summarize_events(events, {'prepare_metadata':1})
        self.assertEqual(result['d2h_activity'], dict(calls=1, sum_ms=.004))
        self.assertAlmostEqual(result['host_ops']['host_sync']['cpu_self_ms'], .1)
        self.assertAlmostEqual(result['host_ops']['host_scalar_extract']['cpu_self_ms'], .02)
        self.assertAlmostEqual(result['host_ops']['host_copy_api']['cpu_self_ms'], .009)
        self.assertIn('implicit blocking', result['interpretation'])
        self.assertNotIn('host_copy_enqueue', result['host_ops'])
        self.assertIsNone(host_category('aten::copy_'))  # Not proof of host waiting or D2H.

    def test_unknown_scope_and_bad_durations_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize_events([], {'raw-private-scope':1})
        with self.assertRaises(ValueError):
            summarize_events([event('kernel', device='CUDA', gpu=float('nan'))], {})
        with self.assertRaises(ValueError):
            summarize_events([event('op', kernels=(-1,))], {})
        cyc = event('cyclic'); cyc.cpu_parent = cyc
        with self.assertRaises(ValueError):
            nearest_label(cyc)

    def test_full_graph_missing_ranges_is_visible_not_false_zero_claim(self):
        report = summarize_events([event('graph_kernel', device='CUDA', gpu=1000)], {})
        self.assertEqual(report['missing_eager_attention_scopes'], ['global_attention', 'local_attention'])
        self.assertEqual(report['raw_cuda_activity_sum_ms'], 1.)

    def test_session_starts_after_warm_and_stops_without_explicit_sync(self):
        trace = []
        class Prof:
            def __enter__(self): trace.append('enter')
            def __exit__(self, *_): trace.append('exit')
            def events(self): return []
        torch = NS(profiler=NS(profile=lambda **_: Prof(), record_function=lambda _: nullcontext(),
                              ProfilerActivity=NS(CPU=1, CUDA=2)))
        session = ProfileSession(torch, 1)
        session.request_start()
        with session.scope('selector'): pass
        self.assertEqual(trace, [])
        self.assertEqual(dict(session.calls), {})
        session.request_start()
        with session.scope('selector'): pass
        session.stop(); session.stop()
        self.assertEqual(trace, ['enter', 'exit'])
        self.assertTrue(session.finished)
        self.assertEqual(session.calls['selector'], 1)
        self.assertEqual(wrapped(session, lambda x: x+1, 'consumer')(2), 3)
        with self.assertRaises(ValueError): ProfileSession(torch, 0)

    def test_cuda_spans_record_entry_stream_and_read_only_after_boundary(self):
        trace = []; current = ['main']
        class Ev:
            def record(self, stream): trace.append(('record', stream))
            def elapsed_time(self, end): trace.append(('elapsed',)); return 2.5
        class Prof:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def events(self): return []
        torch = NS(cuda=NS(current_stream=lambda: current[0], Event=lambda **_: Ev()),
                   profiler=NS(profile=lambda **_: Prof(), record_function=lambda _: nullcontext(),
                               ProfilerActivity=NS(CPU=1, CUDA=2)))
        session = ProfileSession(torch,1,cuda_events=True)
        session.request_start(); session.request_start()
        with session.scope('denoise_global_attention'):
            current[0] = 'side'
            with session.scope('dp_build'): pass
        self.assertEqual(trace, [('record','main'),('record','side'),('record','side'),('record','main')])
        session.stop(boundary_completed=True)
        self.assertEqual(sum(x == ('elapsed',) for x in trace),2)
        self.assertEqual(session.event_summary['spans']['dp_build'], dict(calls=1,sum_ms=2.5))
        self.assertIn('overlap',session.event_summary['interpretation'])

    def test_cuda_spans_failure_or_missing_boundary_never_reads_events(self):
        class Unreadable:
            def elapsed_time(self, end): raise AssertionError('premature event read')
        self.assertFalse(summarize_cuda_events([('dp_route',Unreadable(),Unreadable())],
                                              boundary_completed=False)['available'])
        with self.assertRaisesRegex(ValueError,'invalid profiler duration'):
            summarize_cuda_events([('fused_observe',NS(elapsed_time=lambda _:float('nan')),NS())],
                                  boundary_completed=True)
        with self.assertRaisesRegex(ValueError,'unknown CUDA event scope'):
            summarize_cuda_events([('private-unrecognized',NS(),NS())],boundary_completed=True)

    def test_default_cuda_spans_are_disabled_and_do_not_touch_cuda(self):
        class Prof:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def events(self): return []
        torch = NS(profiler=NS(profile=lambda **_: Prof(),record_function=lambda _:nullcontext(),
                              ProfilerActivity=NS(CPU=1,CUDA=2)))
        session = ProfileSession(torch,1)
        session.request_start(); session.request_start()
        with session.scope('dp_build'): pass
        session.stop(boundary_completed=True)
        self.assertIsNone(session.event_summary)
        self.assertEqual(session.event_pairs,[])

    def test_cuda_event_switch_and_leaf_scopes_require_frozen_diagnostic(self):
        validate_event_settings({},False)
        spec=dict(cuda_events=True,diagnostic_only=True,
                  cuda_event_leaf_scopes=['fused_observe','dp_build','dp_route'])
        validate_event_settings(spec,True)
        with self.assertRaises(ValueError): validate_event_settings(spec,False)
        with self.assertRaises(ValueError): validate_event_settings(dict(spec,diagnostic_only=False),True)
        with self.assertRaises(ValueError): validate_event_settings(dict(spec,cuda_event_leaf_scopes=[]),True)
        with self.assertRaises(ValueError): validate_event_settings(dict(cuda_events=1),True)

    def test_32k_binding_validation_does_not_return_private_identifiers(self):
        objects = dict(cells=[dict(dataset='32k', id='private_id')], manifest=[dict(id='private_id', prompt_tokens=[0]*32768)])
        read = objects.__getitem__
        binding = dict(cells='cells', manifests={'32k':'manifest'})
        self.assertEqual(validate_32k(binding, read), 1)
        objects['manifest'][0]['prompt_tokens'] = [0]*65536
        with self.assertRaisesRegex(ValueError, 'only frozen 32K'):
            validate_32k(binding, read)

    def test_actual_counts_keep_unretired_forward_and_omit_private_fields(self):
        phase=dict(denoising_forwards=350, scheduler_denoising_forwards=349,
                   speculative_unused_denoising=1, commit_forwards=23,
                   scheduler_commit_forwards=23, prefill_steps=6, private_request='hidden')
        self.assertEqual(profile_counts(phase)['denoising_forwards'], 350)
        self.assertNotIn('private_request', profile_counts(phase))
        phase['speculative_unused_denoising']=0
        with self.assertRaises(ValueError): profile_counts(phase)

    def test_instrumentation_preserves_results_and_restores_on_exception(self):
        class Adapter:
            buffers = {}
            def forward(self, *_): return 'adapter'
            def _buffers(self, *_): return 'buffer'
            def sparse_lists(self, *_): return 'consumer'
            def _split(self, *_): return 'lists'
            def on_prepare(self, *_): return None
            def on_sample(self, *_): return None
        class Model:
            def prepare_attn(self, *_): return 'prepare'
        class Flash:
            def forward(self, *_): return 'flash'
        class Tracker:
            def finalize(self, *_): return 'final'
        class Session:
            def __init__(self): self.stops=0; self.starts=0; self.labels=[]
            def scope(self, label): self.labels.append(label); return nullcontext()
            def request_start(self): self.starts+=1
            def stop(self, **_): self.stops+=1
        panel = NS(add_tracked_request=lambda *_: 'rid')
        metrics=NS(PhaseTracker=Tracker)
        va=NS(VllmMethodAdapter=Adapter, install_vllm_patches=lambda _: None)
        dg=NS(DiffusionGemmaModelState=Model, _compiled_sample_step=lambda: 'sample')
        fa=NS(FlashAttentionImpl=Flash)
        original_forward, original_prepare, original_add = Flash.forward, Model.prepare_attn, panel.add_tracked_request
        session=Session()
        attention=type('Attention',(),{name:lambda *_:None for name in ('_route','_route_dense_prefix','_observe','_fused_bootstrap_observation','_consume')})
        dp=NS(build=lambda:'build',route=lambda:'route'); consumer=NS(fused_observe=lambda:'fused')
        originals=(dp.build,dp.route,consumer.fused_observe)
        modules={'experiments.numerical_qk_reuse.v27_dense_prefix':dp,
                 'experiments.numerical_qk_reuse.v27_consumer64':consumer}
        with self.assertRaisesRegex(RuntimeError, 'intentional'), patch.dict('sys.modules',modules):
            with instrument(session,panel,metrics,va,dg,fa,NS(Attention=attention),'dense'):
                self.assertEqual(Flash().forward(NS(layer_name='model.layers.5.self_attn.attn')), 'flash')
                self.assertEqual(Model().prepare_attn(), 'prepare')
                self.assertEqual(panel.add_tracked_request(), 'rid')
                self.assertEqual(Tracker().finalize(), 'final')
                self.assertEqual((dp.build(),dp.route(),consumer.fused_observe()),('build','route','fused'))
                raise RuntimeError('intentional')
        self.assertIs(Flash.forward, original_forward)
        self.assertIs(Model.prepare_attn, original_prepare)
        self.assertIs(panel.add_tracked_request, original_add)
        self.assertEqual(session.starts, 1)
        self.assertEqual(session.stops, 2)
        self.assertEqual((dp.build,dp.route,consumer.fused_observe),originals)
        for name in ('dp_build','dp_route','fused_observe'): self.assertIn(name,session.labels)


if __name__ == '__main__':
    unittest.main()
