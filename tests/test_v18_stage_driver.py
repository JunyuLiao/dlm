import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.v13_seed_runs import execution_key
from scripts.v18_stage_driver import gpu_seconds_by_host, stage_plan


def item(phase, block, host):
    return dict(index=block, block=block, arm='D_native', id=f'{phase}{block}', seed=101,
                role='attempt0', repeat=0, cell_id=f'{phase}{block}', host=host)


class DriverTests(unittest.TestCase):
    def test_crashed_worker_is_conservatively_charged_before_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'ledger.jsonl'
            ledger.write_text(''.join(json.dumps(row) + '\n' for row in [
                {'event': 'start', 'host': 'mpk', 'when': 10},
                {'event': 'start', 'host': 'mpk', 'when': 30},
                {'event': 'worker_end', 'host': 'mpk', 'gpu_process_seconds': 5},
            ]))
            self.assertEqual(gpu_seconds_by_host([ledger], 40), {'mpk': 25.0})

    def test_two_host_stage_gate_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ruler_entries = [item('r', 0, 'mpk'), item('r', 1, 'dllm'), item('r', 2, 'mpk')]
            aime_entries = [item('a', 0, 'mpk'), item('a', 1, 'dllm')]

            def protocol(name, entries):
                return dict(status='frozen', logical_protocol_sha256='digest', protocol_id=name,
                            schedule=entries, initial_prefix_blocks=2,
                            assigned_hosts=[{'host': h} for h in ('mpk', 'dllm')],
                            block_assignments={str(e['block']): {'host': e['host'], 'gpu_uuid': e['host'] + '-gpu'}
                                               for e in entries})

            ruler, aime = protocol('ruler', ruler_entries), protocol('aime', aime_entries)
            rledger, aledger = root / 'r.jsonl', root / 'a.jsonl'
            budget = dict(gpu_cap_s_by_host={'mpk': 100, 'dllm': 100},
                          baseline_gpu_s_by_host={'mpk': 10, 'dllm': 10},
                          baseline_requests=10, baseline_requests_by_host={'mpk': 5, 'dllm': 5},
                          request_cap=30, request_cap_by_host={'mpk': 15, 'dllm': 15},
                          block_guard_s=5, deadline_epoch=1000, initial_soft_deadline_epoch=300,
                          scoring_minutes_reserved=1)

            def write(path, entries):
                path.write_text(''.join(json.dumps(dict(e, event='run', execution_key=execution_key(e),
                                                       gpu_uuid=e['host'] + '-gpu')) + '\n' for e in entries))

            with patch('scripts.v18_stage_driver.logical_protocol_digest', return_value='digest'):
                write(rledger, ruler_entries[:1])
                with self.assertRaisesRegex(ValueError, 'both hosts'):
                    stage_plan('aime', ruler, aime, [rledger], [aledger], budget, host='mpk', now=100)
                write(rledger, ruler_entries[:2])
                plan = stage_plan('aime', ruler, aime, [rledger], [aledger], budget, host='mpk', now=100)
                self.assertTrue(plan['allowed'])
                self.assertEqual(plan['next_block_requests'], 1)
                with self.assertRaisesRegex(ValueError, 'both hosts'):
                    stage_plan('remainder', ruler, aime, [rledger], [aledger], budget, host='mpk', now=100)
                write(aledger, aime_entries)
                final = stage_plan('remainder', ruler, aime, [rledger], [aledger], budget, host='mpk', now=100)
                self.assertTrue(final['allowed'])
                self.assertEqual(final['next_block_requests'], 1)

    def test_clean_soft_stop_advances_to_aime_but_partial_block_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'r.jsonl'
            block = [dict(index=i, block=0, arm=arm, id='q', seed=101,
                          role='attempt0', repeat=0, cell_id=f'cell{arm}')
                     for i, arm in enumerate(('D_native', 'U50'))]
            ruler = dict(status='frozen', logical_protocol_sha256='digest', protocol_id='r',
                         schedule=block, initial_prefix_blocks=1,
                         assigned_hosts=[{'host': 'mpk'}],
                         block_assignments={'0': {'host': 'mpk', 'gpu_uuid': 'g'}})
            aime = dict(ruler, protocol_id='a', schedule=[])
            budget = dict(gpu_cap_s_by_host={'mpk': 100}, baseline_gpu_s_by_host={'mpk': 0},
                          baseline_requests=0, baseline_requests_by_host={'mpk': 0},
                          request_cap=10, request_cap_by_host={'mpk': 10}, block_guard_s=5,
                          deadline_epoch=1000, initial_soft_deadline_epoch=300,
                          scoring_minutes_reserved=1)
            start = json.dumps({'event': 'start', 'host': 'mpk', 'when': 90}) + '\n'
            ledger.write_text(start + json.dumps({'event': 'worker_end', 'host': 'mpk',
                                          'gpu_process_seconds': 1,
                                          'status': 'stopped_before_block'}) + '\n')
            with patch('scripts.v18_stage_driver.logical_protocol_digest', return_value='digest'):
                self.assertTrue(stage_plan('aime', ruler, aime, [ledger], [], budget,
                                           host='mpk', now=100)['ruler_initial_gate_ready'])
                recovery = stage_plan('remainder', ruler, aime, [ledger], [], budget,
                                      host='mpk', now=100)
                self.assertTrue(recovery['allowed'])
                self.assertEqual(recovery['next_block_requests'], 2)
                self.assertEqual(recovery['eval_stage'], 'all')
                row = dict(block[0], event='run', execution_key=execution_key(block[0]),
                           host='mpk', gpu_uuid='g')
                ledger.write_text(start + json.dumps(row) + '\n' + json.dumps({'event': 'worker_end',
                                  'host': 'mpk', 'gpu_process_seconds': 1,
                                  'status': 'stopped_before_block'}) + '\n')
                with self.assertRaisesRegex(ValueError, 'partial initial block'):
                    stage_plan('aime', ruler, aime, [ledger], [], budget, host='mpk', now=100)


if __name__ == '__main__':
    unittest.main()
