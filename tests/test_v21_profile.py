"""CPU-only v21 profile wrapper contract; no torch or model launch."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def load_module():
    base = types.ModuleType('scripts.v20_profile')
    base.validate_config = lambda config: None
    base.preflight_identity = lambda config, paths: (config, paths)
    base.counter_replay = lambda *a, **kw: ({'status': 'qualified'}, [])
    base.atomic_json = lambda path, obj: Path(path).write_text(json.dumps(obj), encoding='utf-8')
    base.profile = lambda config, checkpoint=None: {'schema': 'v20', 'source_sha256': {}, 'targets': {}}
    scripts = types.ModuleType('scripts')
    scripts.v20_profile = base
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'v21_profile.py'
    spec = importlib.util.spec_from_file_location('v21_profile_tested', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'scripts': scripts, 'scripts.v20_profile': base}):
        spec.loader.exec_module(module)
    return module, base


P, B = load_module()


def config():
    common = dict(model='/model', revision='sha', manifest_sha256_by_dataset={'*': 'hash'},
                  source_hashes={'/source': 'hash'}, model_metadata_hashes={'config.json': 'hash'},
                  v20_arm='M3_R3_A8_current_output', v20_scope='GLOBAL_ONLY_NATIVE_LOCAL',
                  decision_interval=3, score_refresh_period=8, condition='v20_global_M3_R3')
    arms = [dict(name='D_native', condition='native_dense', config={'model': '/model'})]
    for name, (precision, layout) in P.MODES.items():
        wrapper = dict(plugin=P.PLUGIN, parent_kind='v20_method', parent_config=dict(common),
                       condition=common['condition'], output_score_precision=precision,
                       output_layout=layout)
        arms.append(dict(name=name, plugin=P.PLUGIN, condition=common['condition'], config=wrapper))
    return dict(scope='GLOBAL_ONLY_NATIVE_LOCAL', counter_twins=True, arms=arms,
                targets=[dict(id='a', dataset='ruler4k', canvas=0, call_index=0)],
                boundaries=['model_forward', 'denoising_step'], sequence_lengths=[4, 16])


class V21ProfileContract(unittest.TestCase):
    def test_exact_grid_and_parent_clock(self):
        cfg = config()
        P.validate_config(cfg)
        cfg['arms'][2]['config']['output_layout'] = 'head_major'
        with self.assertRaisesRegex(ValueError, 'mode/name'):
            P.validate_config(cfg)
        cfg = config()
        cfg['arms'][3]['config']['parent_config']['decision_interval'] = 2
        with self.assertRaisesRegex(ValueError, 'R3/A8'):
            P.validate_config(cfg)
        cfg = config()
        cfg['counter_twins'] = False
        with self.assertRaisesRegex(ValueError, 'counter pass'):
            P.validate_config(cfg)

    def test_parent_preflight_flatten_but_installer_retains_wrapper(self):
        cfg = config()
        flat = P._flatten_for_preflight(cfg)
        self.assertEqual(flat['arms'][1]['config']['v20_arm'], 'M3_R3_A8_current_output')
        self.assertEqual(cfg['arms'][1]['config']['parent_kind'], 'v20_method')
        seen = []
        fake_v21 = types.ModuleType('experiments.numerical_qk_reuse.v21')
        fake_v21.validate_effective = lambda wrapper, condition: seen.append((wrapper, condition))
        with patch.dict(sys.modules, {'experiments': types.ModuleType('experiments'),
                                      'experiments.numerical_qk_reuse': types.ModuleType('experiments.numerical_qk_reuse'),
                                      'experiments.numerical_qk_reuse.v21': fake_v21}):
            result = P.preflight_identity(cfg, {'*': 'manifest'})
        self.assertEqual(len(seen), 4)
        self.assertEqual(result[0]['arms'][2]['config']['v20_scope'], cfg['scope'])

    def test_profile_uses_v20_replay_and_restores_hooks(self):
        cfg = config()
        fake_v21 = types.ModuleType('experiments.numerical_qk_reuse.v21')
        fake_v21.validate_effective = lambda wrapper, condition: None
        visited = []
        def fake_profile(config, checkpoint=None):
            visited.append(B.preflight_identity(config, {'*': 'manifest'}))
            physical, _ = B.counter_replay('model', [], 'runtime', 'model_forward', [])
            self.assertIn('copy_allocation_trace', physical)
            return {'schema': 'v20', 'source_sha256': {}, 'targets': {}}
        with patch.dict(sys.modules, {'experiments': types.ModuleType('experiments'),
                                      'experiments.numerical_qk_reuse': types.ModuleType('experiments.numerical_qk_reuse'),
                                      'experiments.numerical_qk_reuse.v21': fake_v21}), \
             patch.object(B, 'profile', fake_profile), \
             patch.object(P, 'copy_allocation_trace', lambda *a: {'status': 'qualified'}):
            report = P.profile(cfg)
        self.assertEqual(report['schema'], 'v21_direct_output_modes_v1')
        self.assertEqual(len(visited), 1)
        self.assertEqual(B.preflight_identity.__name__, '<lambda>')

    def test_failure_receipt_is_durable_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conf = root / 'config.json'
            conf.write_text('{}', encoding='utf-8')
            out = root / 'out.json'
            def failure(config, checkpoint):
                checkpoint({'targets': {'first': {'status': 'done'}}})
                raise RuntimeError('later failure')
            with patch.object(P, 'profile', failure):
                with self.assertRaisesRegex(RuntimeError, 'later failure'):
                    P.main(['--config', str(conf), '--out', str(out)])
            self.assertTrue((root / 'out.json.partial.json').exists())
            self.assertTrue((root / 'out.json.failure.json').exists())
            with self.assertRaises(FileExistsError):
                P.main(['--config', str(conf), '--out', str(out)])


if __name__ == '__main__':
    unittest.main()
