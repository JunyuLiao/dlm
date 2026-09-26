import unittest
import tempfile
from pathlib import Path

from scripts.v18_evaluate import (aime_token_overlay, canonical_arm_hash, native_phase_evidence,
                                  prioritized_schedule, run_complete_blocks, strict_warm)
from scripts.v18_protocol import sha


class EvaluateTests(unittest.TestCase):
    def test_deadline_is_checked_between_complete_blocks(self):
        schedule = [{'block': block, 'cell_id': f'cell{block}', 'role': role, 'repeat': repeat}
                    for block in (0, 1) for role, repeat in (('attempt0', 0), ('warm', 1))]
        now = [0.0]
        seen = []

        def execute(entries):
            for entry in entries:
                seen.append(entry['role'])
                now[0] += 4.0
            return 'complete'

        status = run_complete_blocks(schedule, set(), deadline_epoch=10, gpu_budget_s=10,
                                     remaining_requests=4, block_guard_s=3,
                                     process_started=0, execute_block=execute,
                                     wall_clock=lambda: now[0], process_clock=lambda: now[0])
        self.assertEqual(status, 'stopped_before_block')
        self.assertEqual(seen, ['attempt0', 'warm'])

    def test_priority_starts_with_complete_warm_block(self):
        arms = {name: str(i) for i, name in enumerate(('D_native', 'D_matched', 'U50', 'U60', 'T50', 'T60'))}
        schedule = prioritized_schedule('test', 'revision', arms, ['a', 'b'], ['b'], 9)
        first = [e for e in schedule if e['block'] == 0]
        self.assertEqual({e['id'] for e in first}, {'b'})
        self.assertEqual(len(first), 12)
        self.assertEqual({e['role'] for e in first}, {'attempt0', 'warm'})
        self.assertEqual([e['index'] for e in schedule], list(range(len(schedule))))

    def test_missing_native_phase_is_unaccepted(self):
        receipt = {'per_canvas': [{'decoder_calls': 2, 'schedule_steps': [48, 47],
                                   'native_stop_final_call': True, 'iteration_cap_final_call': False}],
                   'total_decoder_calls': 2, 'generation_gpu_timeline_seconds': None}
        self.assertIsNone(native_phase_evidence(receipt))
        first = {'ok': True, 'completion_token_hash': 'hash', 'per_canvas_calls': [2],
                 'termination': 'eos', 'phase_evidence': None}
        warm = dict(first, triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[])
        self.assertFalse(strict_warm(first, warm)['accepted'])

    def test_strict_warm_checks_tokens_steps_stop_and_compile(self):
        receipt = {'per_canvas': [{'decoder_calls': 2, 'schedule_steps': [48, 47],
                                   'native_stop_final_call': True, 'iteration_cap_final_call': False}],
                   'total_decoder_calls': 2, 'generation_gpu_timeline_seconds': 0.2}
        phase = native_phase_evidence(receipt)
        self.assertIsNotNone(phase)
        self.assertEqual(native_phase_evidence(receipt, sparse=True)['phase'], 'fresh_sparse_decoder')
        first = {'ok': True, 'completion_token_hash': 'hash', 'per_canvas_calls': [2],
                 'termination': 'eos', 'phase_evidence': phase}
        warm = dict(first, triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[])
        self.assertTrue(strict_warm(first, warm)['accepted'])
        self.assertFalse(strict_warm(first, dict(warm, completion_token_hash='other'))['accepted'])
        self.assertFalse(strict_warm(first, dict(warm, triton_misses=1))['accepted'])
        changed = dict(phase, per_canvas=[dict(phase['per_canvas'][0], native_stop=False)])
        self.assertFalse(strict_warm(first, dict(warm, phase_evidence=changed))['accepted'])

    def test_aime_overlay_freezes_exact_cpu_tokens_in_new_manifest(self):
        class Adapter:
            def encode_prompt(self, prompt, extra):
                self_flag = extra['thinking']
                return [len(prompt), int(self_flag)]
        rows = [{'id': f'aime26/{i}', 'prompt': f'q{i}', 'prompt_hash': sha(f'q{i}'),
                 'thinking': True, 'generation_budget': 8192} for i in range(1, 31)]
        draft = {'dataset': 'aime26', 'protocol_id': 'draft',
                 'prompt_hashes': {r['id']: r['prompt_hash'] for r in rows}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'private_v3' / 'aime.json'
            updated = aime_token_overlay(draft, rows, Adapter(), path)
            self.assertEqual(len(updated['prompt_token_hashes']), 30)
            self.assertEqual(updated['generation_manifest_sha256'], sha(path.read_bytes()))
            with self.assertRaisesRegex(ValueError, 'gold'):
                aime_token_overlay(draft, [dict(r, answer='secret') for r in rows], Adapter(),
                                   Path(folder) / 'other.json')

    def test_canonical_arm_identity_ignores_host_paths_but_checks_math(self):
        config = {'condition': 'native_legal_all_layers', 'method': 'T', 'frontier_arm': 'T60',
                  'policy': {'local': {'log_threshold': 1.0}, 'global': {'log_threshold': -1.0}},
                  'source_hashes': {'/mpk/code/a.py': 'a' * 64, '/mpk/lib.so': 'b' * 64},
                  'model_metadata_hashes': {'config.json': 'c' * 64},
                  'manifest_sha256': 'd' * 64, 'policy_sha256': 'e' * 64}
        binary = {'kernel': 'b' * 64, 'bridge': 'f' * 64}
        same = dict(config, source_hashes={'/dllm/deploy/a.py': 'a' * 64,
                                           '/dllm/bridge/lib.so': 'b' * 64})
        self.assertEqual(canonical_arm_hash(config, binary, 'f' * 64),
                         canonical_arm_hash(same, binary, 'f' * 64))
        changed = dict(same, method='unweighted')
        self.assertNotEqual(canonical_arm_hash(config, binary, 'f' * 64),
                            canonical_arm_hash(changed, binary, 'f' * 64))
        repacked = dict(config, model_metadata_hashes={'config.json': 'c' * 64,
                                                       'model.safetensors.index.json': '0' * 64})
        self.assertEqual(canonical_arm_hash(config, binary, 'f' * 64),
                         canonical_arm_hash(repacked, binary, 'f' * 64))
        self.assertNotEqual(canonical_arm_hash(config, binary, 'f' * 64),
                            canonical_arm_hash(repacked, binary, 'e' * 64))


if __name__ == '__main__':
    unittest.main()
