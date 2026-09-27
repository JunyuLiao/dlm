"""CPU binder tests: frozen IDs, task contracts, arm grid and LF byte identity."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import v20_bind as B


class V20BindTests(unittest.TestCase):
    def fixture(self, root):
        manifests = root / 'manifests'
        manifests.mkdir()
        protocol = dict(schema='v20_fan_panel_v1', planned_executions=700,
                        model_revision=B.REVISION, ids={}, generation_manifest_sha256={},
                        block_assignments={})
        for dataset in B.DATASETS:
            rows = []
            ids = [f'{dataset}/{i}' for i in range(2)]
            for index, id_ in enumerate(ids):
                prompt = f'{dataset} prompt {index}'
                rows.append(dict(id=id_, prompt=prompt, prompt_hash=B.sha_bytes(prompt.encode()),
                                 prompt_tokens=[1, 2], prompt_token_count=2,
                                 generation_budget=32 if dataset == 'ruler4k' else 8192,
                                 thinking=dataset != 'ruler4k'))
            path = manifests / f'{dataset}_generation_manifest.json'
            B.atomic_new(path, rows)
            protocol['ids'][dataset] = ids
            protocol['generation_manifest_sha256'][dataset] = B.sha_bytes(path.read_bytes())
        for index, dataset in enumerate(B.DATASETS):
            for local in range(2):
                block = index * 2 + local
                protocol['block_assignments'][str(block)] = dict(dataset=dataset,
                    id=protocol['ids'][dataset][local], seed=101,
                    host=f'host{local}', gpu_uuid=f'GPU-{local}')
        protocol_path = root / 'protocol.json'
        B.atomic_new(protocol_path, protocol)
        calibration_path = root / 'calibration.json'
        B.atomic_new(calibration_path, dict(T50={'policy': {'local': {'log_threshold': -1.},
                                                       'global': {'log_threshold': -3.}}},
                                              T60={'policy': {'local': {'log_threshold': -.7},
                                                       'global': {'log_threshold': -2.8}}}))
        return protocol_path, manifests, calibration_path

    @staticmethod
    def args(protocol_path, calibration_path, host=None):
        return SimpleNamespace(model=Path('/model'), library=Path('/library.so'),
                               torch_library=Path('/torch_library.so'),
                               support_build=Path('/support.json'), consumer='hopper',
                               source_commit='abc123', host=host, gpu_uuid=('GPU-0' if host else None),
                               protocol_sha256=B.sha_bytes(protocol_path.read_bytes()),
                               calibration_sha256=B.sha_bytes(calibration_path.read_bytes()))

    def test_frozen_screen_host_filter_and_arm_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, manifests_dir, calibration_path = self.fixture(root)
            protocol, policies, rows, manifests, chosen = B.read_frozen(
                protocol_path, manifests_dir, calibration_path)
            args = self.args(protocol_path, calibration_path, host='host0')
            identity = dict(model={'config.json': 'a'}, control={'x': 'a'}, method={'y': 'b'})
            screen = B.screen_config(protocol, policies, rows, manifests, chosen, args, identity)
            self.assertEqual((screen['host'], screen['gpu_uuid']), ('host0', 'GPU-0'))
            self.assertEqual(len(screen['targets']), 6)
            self.assertEqual(len(screen['arms']), 26)
            self.assertEqual((screen['boundaries'], screen['sequence_lengths']),
                             (['model_forward'], [4]))
            self.assertTrue(all(t['prompt_hash'] and 'prompt' not in t for t in screen['targets']))
            method = next(a for a in screen['arms'] if a['name'].endswith('P0_M3_R3_A8_current_output'))
            held = next(a for a in screen['arms'] if a['name'].endswith('P0_B_A8_matched'))
            self.assertEqual(method['config']['selector'], 'prefix_block_summary')
            self.assertEqual(held['config']['selector'], 'legacy_recompute')
            self.assertEqual(method['config']['decision_interval'], 3)
            native = screen['arms'][0]
            self.assertEqual(native['name'], 'D_native')
            self.assertEqual(native['config']['model'], str(args.model.resolve()))
            self.assertEqual(native['config']['revision'], B.REVISION)
            self.assertEqual(native['config']['manifest_sha256_by_dataset'],
                             screen['manifest_sha256'])
            historical = next(a for a in screen['arms'] if a['name'] == 'G75L30_nativeQ128')
            self.assertEqual(historical['config']['v20_scope'], 'ALL_NATIVE_LEGAL')
            alternatives = [a for a in screen['arms'] if a.get('diagnostic_consumer_alternative')]
            self.assertEqual(len(alternatives), 4)
            self.assertTrue(all(a['config']['consumer'] == 'triton' for a in alternatives))
            self.assertTrue(screen['counter_twins'])
            self.assertFalse(screen['prepared_support_floor'])

    def test_binder_screen_passes_profile_preflight_before_model_load(self):
        from tests.test_v20_profile import P as profile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, manifests_dir, calibration_path = self.fixture(root)
            protocol, policies, rows, manifests, chosen = B.read_frozen(
                protocol_path, manifests_dir, calibration_path)
            args = self.args(protocol_path, calibration_path, host='host0')
            args.model = root / 'model'
            args.model.mkdir()
            (args.model / 'config.json').write_bytes(b'{}')
            source_path = root / 'source.py'
            source_path.write_bytes(b'pass\n')
            source = {str(source_path): B.sha_bytes(source_path.read_bytes())}
            identity = dict(model={'config.json': B.sha_bytes(b'{}')},
                            control=source, method=source)
            screen = B.screen_config(protocol, policies, rows, manifests, chosen, args, identity)
            with patch.object(profile.subprocess, 'check_output', return_value='GPU-0\n'):
                proof = profile.preflight_identity(screen, screen['manifests'])
            self.assertEqual(proof['gpu_uuid'], 'GPU-0')

    def test_generation_configs_and_lf_atomic_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, manifests_dir, calibration_path = self.fixture(root)
            protocol, policies, rows, manifests, chosen = B.read_frozen(
                protocol_path, manifests_dir, calibration_path)
            args = self.args(protocol_path, calibration_path)
            identity = dict(model={'config.json': 'a'}, control={'x': 'a'}, method={'y': 'b'})
            configs = B.generation_configs(protocol, policies, manifests, args, identity,
                                           'GLOBAL_ONLY_NATIVE_LOCAL', 'P1')
            self.assertEqual({d: len(v) for d, v in configs.items()}, {d: 8 for d in B.DATASETS})
            self.assertFalse(configs['ruler4k']['D_native']['thinking'])
            self.assertTrue(configs['aime26']['T_scope']['thinking'])
            self.assertTrue(configs['longbench_v2']['M3_R3_A8_current_output']['timing_events'])
            self.assertEqual(configs['longbench_v2']['G75L30_nativeQ128']['v20_scope'],
                             'ALL_NATIVE_LEGAL')
            path = root / 'out.json'
            B.atomic_new(path, {'x': 1})
            self.assertEqual(path.read_bytes()[-1:], b'\n')
            self.assertNotIn(b'\r\n', path.read_bytes())
            with self.assertRaises(FileExistsError):
                B.atomic_new(path, {'x': 2})


if __name__ == '__main__':
    unittest.main()
