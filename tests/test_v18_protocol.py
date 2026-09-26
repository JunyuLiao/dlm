import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.v18_protocol import RULER_ARMS, build, generation_rows, sha, strip_gold
from scripts.v18_frontier import advance_calibration, midpoint_policy
from experiments.numerical_qk_reuse.runner import _rows, request_budget


class ProtocolTests(unittest.TestCase):
    def test_manifest_strips_gold_and_seed_without_changing_prompt(self):
        row = {'id': 'q', 'prompt': 'hello', 'prompt_hash': sha('hello'), 'outputs': ['secret'],
               'expected': 'secret', 'seed': 42, 'generation_budget': 8192}
        clean = generation_rows(strip_gold([row]))[0]
        self.assertEqual(clean['prompt'], 'hello')
        self.assertNotIn('outputs', clean)
        self.assertNotIn('expected', clean)
        self.assertNotIn('seed', clean)

    def test_prompt_tamper_and_duplicate_rejected(self):
        row = {'id': 'q', 'prompt': 'hello', 'prompt_hash': sha('other')}
        with self.assertRaisesRegex(ValueError, 'prompt hash'):
            strip_gold([row])
        row['prompt_hash'] = sha('hello')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            strip_gold([row, row])

    def test_schedule_has_distinct_seed_cells_and_selected_warm_only(self):
        rows = [{'id': k, 'prompt_hash': sha(k)} for k in ('a', 'b')]
        p = build('synthetic', rows, RULER_ARMS[:2], ['a'], schedule_seed=7,
                  authorization='test', source_sha='0' * 64, construction='test')
        self.assertEqual(len(p['schedule']), 2 * 3 * 2 + 1 * 3 * 2)
        self.assertEqual(len({(e['arm'], e['id'], e['seed'], e['role']) for e in p['schedule']}), len(p['schedule']))
        self.assertTrue(all(e['id'] == 'a' for e in p['schedule'] if e['role'] == 'warm'))

    def test_sixty_policy_is_only_a_calibration_start(self):
        frozen = {'policies': {'T_s50': {'local': {'log_threshold': 0.0}, 'global': {'log_threshold': -2.0}},
                               'T_s70': {'local': {'log_threshold': 2.0}, 'global': {'log_threshold': 0.0}}}}
        self.assertEqual(midpoint_policy(frozen, 'T60'),
                         {'local': {'log_threshold': 1.0}, 'global': {'log_threshold': -1.0}})

    def test_legacy_and_ruler_request_budgets(self):
        self.assertEqual(request_budget({}, {}), 8192)
        self.assertEqual(request_budget({'generation_budget': 8192}, {'max_new_tokens': 8192}), 8192)
        for budget in (30, 32, 50, 120, 128):
            self.assertEqual(request_budget({'generation_budget': budget}, {'max_new_tokens': 8192}), budget)
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            manifest.write_text(json.dumps([{'id': 'ruler/q', 'prompt': 'question',
                                             'generation_budget': 30, 'thinking': True}]))
            with self.assertRaisesRegex(ValueError, '8192-token'):
                _rows(manifest)
            self.assertEqual(_rows(manifest, allow_task_budgets=True)[0]['generation_budget'], 30)

    def test_calibration_resume_after_history_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            policy = {'local': {'log_threshold': -1.0}, 'global': {'log_threshold': -2.0}}
            proto = {'point_index': 0, 'ids': ['q'], 'manifest': str(root / 'manifest.json'),
                     'policy_file': str(root / 'policy.json'), 'model_path': str(root / 'model'),
                     'authorization': 'test', 'arms': {'U50': {'policy': policy}},
                     'schedule': [{'arm': 'U50', 'id': 'q', 'seed': 42}]}
            protocol_path = root / 'calibration_p0_protocol.json'
            protocol_path.write_text(json.dumps(proto))
            config_path = root / 'configs/p0/U50.json'
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({'library': 'lib.so', 'torch_library': 'bridge.so'}))
            receipt_path = root / 'receipt.json'
            receipt_path.write_text(json.dumps({'id': 'q', 'seed': 42, 'routing': [
                {'attention_type': 'local', 'eligible': 10, 'skipped': 3},
                {'attention_type': 'global', 'eligible': 10, 'skipped': 3}]}))
            ledger_path = root / 'ledger.jsonl'
            ledger_path.write_text(json.dumps({'event': 'run', 'arm': 'U50', 'id': 'q', 'seed': 42,
                                               'ok': True, 'private_receipt': str(receipt_path)}) + '\n')
            with patch('scripts.v18_frontier.plan_calibration', side_effect=RuntimeError('crash after history')):
                with self.assertRaisesRegex(RuntimeError, 'crash after history'):
                    advance_calibration(protocol_path, ledger_path, root)
            first_history = json.loads((root / 'calibration_history.json').read_text())
            self.assertEqual(len(first_history['U50']), 1)
            with patch('scripts.v18_frontier.plan_calibration', return_value={'next': True}) as plan:
                self.assertEqual(advance_calibration(protocol_path, ledger_path, root), {'next': True})
                planned_policy = plan.call_args.kwargs['policies']
            self.assertEqual(len(json.loads((root / 'calibration_history.json').read_text())['U50']), 1)
            with patch('scripts.v18_frontier.plan_calibration', return_value={'next': True}) as plan:
                advance_calibration(protocol_path, ledger_path, root)
                self.assertEqual(plan.call_args.kwargs['policies'], planned_policy)
            history = json.loads((root / 'calibration_history.json').read_text())
            history['U50'].append(dict(point=1, policy={'local': {'log_threshold': 9}, 'global': {'log_threshold': 9}},
                                      actual={'whole': 1, 'local': 1, 'global': 1}, counts={}))
            (root / 'calibration_history.json').write_text(json.dumps(history))
            with patch('scripts.v18_frontier.plan_calibration', return_value={'next': True}) as plan:
                advance_calibration(protocol_path, ledger_path, root)
                self.assertEqual(plan.call_args.kwargs['policies'], planned_policy)
            self.assertEqual(len(json.loads((root / 'calibration_history.json').read_text())['U50']), 2)


if __name__ == '__main__':
    unittest.main()
