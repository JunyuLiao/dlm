"""CPU contract tests for v20 direct replay; no model or CUDA import required."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def load_profile():
    torch = types.ModuleType('torch')
    torch.is_tensor = lambda value: False
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    harness = types.ModuleType('scripts.replay_harness')
    harness.StepSnapshot = type('StepSnapshot', (), {'digest': staticmethod(lambda kwargs, controller=None: str(kwargs['cur_step']))})
    for name in ('assert_native_path', 'dispatch_spy', 'execution_context', 'output_digest', 'reset_router'):
        setattr(harness, name, lambda *args, **kwargs: None)
    v9 = types.ModuleType('scripts.v9_step_replay_profile')
    v9.decoder_call = lambda model, kwargs: model.forward(**kwargs)
    v9.native_registry = lambda model: ({'sdpa': object()}, object())
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'v20_profile.py'
    spec = importlib.util.spec_from_file_location('v20_profile_tested', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch, 'scripts.replay_harness': harness,
                                  'scripts.v9_step_replay_profile': v9}):
        spec.loader.exec_module(module)
    return module


P = load_profile()


class V20ProfileContract(unittest.TestCase):
    def config(self):
        return dict(model='pinned', revision='sha', manifest='gold-free.json', reps=3,
                    targets=[dict(id='a', dataset='RULER4K', canvas=0, call_index=2,
                                  fallback_call_index=1)],
                    arms=[dict(name='D_native', condition='native_dense', config={}),
                          dict(name='M3', condition='v20_global_M3_R3',
                               plugin='experiments.numerical_qk_reuse.v20:install', config={})])

    def test_config_requires_native_bracket_and_three_blocks(self):
        cfg = self.config()
        P.validate_config(cfg)
        cfg['reps'] = 2
        with self.assertRaisesRegex(ValueError, 'three'):
            P.validate_config(cfg)
        cfg['reps'] = 3
        cfg['arms'] = cfg['arms'][::-1]
        with self.assertRaisesRegex(ValueError, 'native_dense'):
            P.validate_config(cfg)

    def test_staged_boundaries_and_lengths(self):
        cfg = self.config()
        cfg['boundaries'] = ['model_forward']
        cfg['sequence_lengths'] = [4]
        P.validate_config(cfg)
        cfg['sequence_lengths'] = [8]
        with self.assertRaisesRegex(ValueError, 'sequence_lengths'):
            P.validate_config(cfg)

    def test_missing_later_call_and_predeclared_fallback(self):
        target = self.config()['targets'][0]
        got = P.resolve_target(target, {0: [{'call_index': 0}, {'call_index': 1}]})
        self.assertEqual(got['selected_call'], 1)
        self.assertTrue(got['fallback_used'])
        self.assertEqual(got['sequence_lengths']['16'], 2)
        self.assertTrue(P.resolve_target(target, {0: [{'call_index': 0}]})['missing'])

    def test_task_contract_keeps_thinking_and_budget(self):
        base = types.ModuleType('dllm.models')
        base.GenerationRequest = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {'dllm': types.ModuleType('dllm'), 'dllm.models': base}):
            request = P.request_for(dict(prompt='ruler', thinking=False, generation_budget=512), 101)
            self.assertEqual(request['max_new_tokens'], 512)
            self.assertEqual(request['extra']['thinking'], False)
            self.assertEqual(request['temperature'], 0.0)

    def test_native_snapshot_keeps_arm_method(self):
        state = types.SimpleNamespace(method='kernel_dense', fast_t=False,
                                      m_ref=7.0, beta=2.0, gamma=.25)

        class Snapshot:
            def prepare(self, controller=None):
                controller.method = 'T'
                controller.fast_t = True
                controller.m_ref = 14.258454322814941
                return {'cur_step': 48}

        self.assertEqual(P.prepare_step(Snapshot(), state), {'cur_step': 48})
        self.assertEqual((state.method, state.fast_t, state.m_ref),
                         ('kernel_dense', False, 7.0))

    def test_model_forward_begins_state_outside_timer_and_router_evolves(self):
        events = []

        class State:
            def begin(self, cur_step, canvas):
                events.append(('begin', cur_step))

        class Snapshot:
            def __init__(self, cur_step):
                self.cur_step = cur_step

            def prepare(self, controller=None):
                events.append(('prepare', self.cur_step))
                return dict(cur_step=self.cur_step, current_canvas='canvas')

        class Model:
            def forward(self, **kwargs):
                events.append(('forward', kwargs['cur_step']))
                return 'logits'

        router = types.SimpleNamespace(counts={})
        runtime = dict(state=State(), router=router)
        sequence = [dict(snapshot=Snapshot(48-i), call_index=i, cur_step=48-i)
                    for i in range(3)]
        with patch.object(P, 'reset_arm', lambda runtime: events.append(('reset', None))), \
             patch.object(P, '_measure', lambda fn: (fn(), 1.0, 2.0)), \
             patch.object(P, 'output_digest', lambda result: result):
            rows = P.replay(Model(), sequence, runtime, 'model_forward')
        self.assertEqual(len(rows), 3)
        self.assertEqual(events, [('reset', None), ('prepare', 48), ('begin', 48), ('forward', 48),
                                  ('prepare', 47), ('begin', 47), ('forward', 47),
                                  ('prepare', 46), ('begin', 46), ('forward', 46)])

    def test_checkpoint_survives_later_failure_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}', encoding='utf-8')
            out = root / 'profile.json'
            def fail_later(cfg, checkpoint):
                checkpoint({'targets': {'first': {'ok': True}}})
                raise RuntimeError('later failed')
            with patch.object(P, 'profile', fail_later):
                with self.assertRaisesRegex(RuntimeError, 'later failed'):
                    P.main(['--config', str(config), '--out', str(out)])
            self.assertTrue((root / 'profile.json.partial.json').exists())
            self.assertTrue((root / 'profile.json.failure.json').exists())
            with self.assertRaises(FileExistsError):
                P.main(['--config', str(config), '--out', str(out)])

    def test_compile_guard_records_miss_and_restores_hook(self):
        jit = types.ModuleType('triton.runtime.jit')
        jit.JITFunction = type('JITFunction', (), {'cache_hook': None, 'compiled_hook': None})
        with patch.dict(sys.modules, {'triton': types.ModuleType('triton'),
                                      'triton.runtime': types.ModuleType('triton.runtime'),
                                      'triton.runtime.jit': jit}):
            with P.no_compile_during_accepted() as misses:
                jit.JITFunction.cache_hook(fn=types.SimpleNamespace(name='kernel'))
            self.assertEqual(misses, ['kernel'])
            self.assertIsNone(jit.JITFunction.cache_hook)

    def test_preload_identity_checks_manifest_sources_model_and_gpu_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            metadata = model / 'config.json'
            metadata.write_bytes(b'model-config')
            source = root / 'method.py'
            source.write_bytes(b'qualified-source')
            manifest = root / 'ruler.json'
            manifest.write_bytes(b'gold-free-manifest')
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            cfg = self.config()
            cfg.update(model=str(model), manifests={'ruler4k': str(manifest)},
                       manifest_sha256={'ruler4k': sha(manifest)}, gpu_uuid='GPU-pinned')
            for arm in cfg['arms']:
                arm['config'] = dict(model=str(model), revision='sha',
                                     manifest_sha256_by_dataset=cfg['manifest_sha256'],
                                     source_hashes={str(source): sha(source)},
                                     model_metadata_hashes={'config.json': sha(metadata)})
            with patch.object(P.subprocess, 'check_output', return_value='GPU-pinned\n'):
                receipt = P.preflight_identity(cfg, cfg['manifests'])
                self.assertEqual(receipt['gpu_uuid'], 'GPU-pinned')
                manifest.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError, 'manifest byte identity'):
                    P.preflight_identity(cfg, cfg['manifests'])
                manifest.write_bytes(b'gold-free-manifest')
                source.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError, 'source byte identity'):
                    P.preflight_identity(cfg, cfg['manifests'])
                source.write_bytes(b'qualified-source')
                metadata.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError, 'metadata byte identity'):
                    P.preflight_identity(cfg, cfg['manifests'])
                metadata.write_bytes(b'model-config')
            with patch.object(P.subprocess, 'check_output', return_value='GPU-other\n'):
                with self.assertRaisesRegex(ValueError, 'GPU UUID'):
                    P.preflight_identity(cfg, cfg['manifests'])

    def test_counter_replay_default_off_and_floor_requires_twin(self):
        cfg = self.config()
        cfg['counter_twins'] = False
        cfg['prepared_support_floor'] = False
        P.validate_config(cfg)
        cfg['prepared_support_floor'] = True
        with self.assertRaisesRegex(ValueError, 'requires an untimed counter'):
            P.validate_config(cfg)
        cfg['counter_twins'] = True
        cfg['floor_reps'] = 2
        with self.assertRaisesRegex(ValueError, 'at least three'):
            P.validate_config(cfg)

    def test_control_phase_counters_are_visible_in_per_call_delta(self):
        values = dict(attention_calls=7, bootstrap_calls=1,
                      bitmap_observation_calls=1, held_decision_calls=5,
                      noncumulative='ignore')
        runtime = dict(counters=lambda: values)
        before = P.counter_snapshot(runtime)
        values.update(attention_calls=8, bitmap_observation_calls=2)
        after = P.counter_snapshot(runtime)
        self.assertEqual(after['bitmap_observation_calls'] - before['bitmap_observation_calls'], 1)
        self.assertEqual(after['bootstrap_calls'] - before['bootstrap_calls'], 0)
        self.assertNotIn('noncumulative', after)


if __name__ == '__main__':
    unittest.main()
