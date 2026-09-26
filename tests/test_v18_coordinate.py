import unittest
from unittest.mock import patch
import json
from pathlib import Path
import sys
import shlex
import hashlib
from types import SimpleNamespace
import tempfile

from scripts.v18_coordinate import aime_complete, launch, marker_state, wait_known_writers_quiet
from scripts import v18_coordinate


class CoordinatorTests(unittest.TestCase):
    def completion_args(self, root):
        budget = root / 'campaign_budget_extension_20260926.json'
        budget_data = dict(deadline_is_gpu_cutoff=True, request_cap=7000,
                           request_cap_by_host={'mpk': 3500, 'dllm': 3500},
                           gpu_cap_s_by_host={'mpk': 86400, 'dllm': 86400},
                           deadline_epoch=1790573520, scoring_minutes_reserved=30)
        budget.write_text(json.dumps(budget_data))
        hook = root / 'hook.json'
        hook.write_text(json.dumps(['python', 'cpu_finalizer.py']))
        args = SimpleNamespace(start_at='aime', continuation_deploy='bounded_deploy',
                               budget=budget, outcome=root / 'outcome.json',
                               poll_seconds=60, after_stages_command_file=hook)
        return args, budget_data

    def test_completion_restart_adopts_recorded_segment_before_ledger_read(self):
        with tempfile.TemporaryDirectory() as folder:
            args, budget = self.completion_args(Path(folder))
            sha = hashlib.sha256(args.budget.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            args.outcome.write_text(json.dumps(dict(
                schema='v18_completion_coordinator_outcome_v1',
                original_deploy=v18_coordinate.DEPLOY,
                continuation_deploy=args.continuation_deploy,
                bounded_budget_sha256=sha, segments=[dict(segment='aime_c001',
                hosts=['mpk'], status='planned')])))
            order = []
            def stage(name, **kw):
                order.append(('stage', kw.get('segment') or name, kw.get('adopt')))
            def sync(_staging):
                order.append(('sync',))
                return []
            complete = dict(complete=True, completed=1080, planned=1080)
            with patch('scripts.v18_coordinate.wait_stage', side_effect=stage), \
                 patch('scripts.v18_coordinate.discovered_segments',
                       return_value={'mpk': {'aime_c001'}, 'dllm': set()}), \
                 patch('scripts.v18_coordinate.sync_all_ledgers', side_effect=sync), \
                 patch('scripts.v18_coordinate.aime_complete', return_value=complete), \
                 patch('scripts.v18_coordinate.panel_complete', return_value=complete), \
                 patch('scripts.v18_coordinate.wait_known_writers_quiet', return_value={}), \
                 patch('scripts.v18_coordinate.subprocess.run'):
                v18_coordinate.completion_main(args, budget)
            self.assertEqual(order[:3], [('stage', 'aime', True),
                                         ('stage', 'aime_c001', True), ('sync',)])
            self.assertIn(('stage', 'remainder', False), order)

    def test_uncertain_numbered_dispatch_keeps_intent_and_defers_finalizer(self):
        with tempfile.TemporaryDirectory() as folder:
            args, budget = self.completion_args(Path(folder))
            order = []
            def stage(name, **kw):
                order.append(kw.get('segment') or name)
                if kw.get('segment') == 'aime_c001':
                    raise v18_coordinate.UncertainDispatch('unknown SSH result')
            partial = dict(complete=False, completed=10, planned=1080,
                           partial_blocks=[], missing_blocks=[1], missing_hosts=['mpk'],
                           clean_stop=True)
            sha = hashlib.sha256(args.budget.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            with patch('scripts.v18_coordinate.wait_stage', side_effect=stage), \
                 patch('scripts.v18_coordinate.discovered_segments',
                       return_value={'mpk': set(), 'dllm': set()}), \
                 patch('scripts.v18_coordinate.sync_all_ledgers', return_value=[]), \
                 patch('scripts.v18_coordinate.aime_complete', return_value=partial), \
                 patch('scripts.v18_coordinate.wait_known_writers_quiet', return_value={}), \
                 patch('scripts.v18_coordinate.remote', return_value=sha), \
                 patch('scripts.v18_coordinate.orphan_private_receipts', return_value=0), \
                 patch('scripts.v18_coordinate.subprocess.run') as finalizer:
                with self.assertRaises(SystemExit):
                    v18_coordinate.completion_main(args, budget)
            finalizer.assert_not_called()
            saved = json.loads(args.outcome.read_text())
            self.assertEqual(saved['segments'][0]['segment'], 'aime_c001')
            self.assertEqual(saved['segments'][0]['hosts'], ['mpk'])
            self.assertEqual(saved['finalization'], 'deferred_uncertain_writer')
    def test_completion_requires_per_host_whole_block_prefix_and_clean_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(arm='T60', id=f'q{i}', seed=101, role='attempt0', repeat=0,
                         cell_id=f'cell{i}', block=i) for i in range(3)]
            protocol = dict(protocol_id='frozen-aime', model_revision='revision',
                            schedule=rows, block_assignments={str(i):
                            dict(host='mpk', gpu_uuid='GPU-A') for i in range(3)})
            mpk, dllm = root / 'mpk.jsonl', root / 'dllm.jsonl'
            dllm.write_text('')
            start = dict(event='start', host='mpk', protocol_id='frozen-aime',
                         model_revision='revision')
            end = dict(event='worker_end', host='mpk', status='stopped_before_block')
            def run(index):
                return dict(rows[index], event='run', execution_key=f'cell{index}:attempt0:0',
                            host='mpk', gpu_uuid='GPU-A', ok=True)
            mpk.write_text('\n'.join(json.dumps(x) for x in (start, run(0), end)) + '\n')
            result = v18_coordinate.completion_detail(protocol, [mpk, dllm])
            self.assertTrue(result['clean_stop'])
            self.assertEqual(result['missing_hosts'], ['mpk'])
            mpk.write_text('\n'.join(json.dumps(x) for x in (start, run(0), run(2), end)) + '\n')
            result = v18_coordinate.completion_detail(protocol, [mpk, dllm])
            self.assertFalse(result['clean_stop'])
            self.assertEqual(result['prefix_violations'], [2])
            mpk.write_text('\n'.join(json.dumps(x) for x in (
                dict(start, protocol_id='wrong'), run(0), end)) + '\n')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                v18_coordinate.completion_detail(protocol, [mpk, dllm])

    def test_numbered_segment_launches_only_missing_host(self):
        seen = []
        def marked(host, _stage, suffix):
            seen.append(host)
            return {'start': 10, 'pid': 123} if suffix == 'started' else {'start': 10, 'rc': 0}
        with patch('scripts.v18_coordinate.marker', side_effect=marked):
            v18_coordinate.wait_stage('aime', adopt=True, poll_seconds=60,
                                      deadline_epoch=10**12, segment='aime_c001',
                                      hosts=['mpk'])
        self.assertEqual(set(seen), {'mpk'})

    def test_orphan_check_command_is_cpu_only_and_syntactically_valid(self):
        seen = []
        with patch('scripts.v18_coordinate.remote', side_effect=lambda _host, cmd: seen.append(cmd) or '0'):
            self.assertEqual(v18_coordinate.orphan_private_receipts('mpk'), 0)
        argv = shlex.split(seen[0])
        self.assertEqual(argv[1], '-c')
        compile(argv[2], '<remote orphan check>', 'exec')
        self.assertNotIn('torch', argv[2])
    def test_launch_waits_for_marker_without_relaunching(self):
        with patch('scripts.v18_coordinate.subprocess.run') as run, \
             patch('scripts.v18_coordinate.marker', side_effect=[None, None, {'start': 10}]) as marker, \
             patch('scripts.v18_coordinate.time.sleep'):
            launch('mpk', 'aime')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(marker.call_count, 3)

    def test_marker_states_never_retry_failure(self):
        started = {'start': 10, 'pid': 1}
        self.assertEqual(marker_state(None, None), 'absent')
        self.assertEqual(marker_state(started, None), 'running')
        self.assertEqual(marker_state(started, {'start': 10, 'rc': 0}), 'complete')
        self.assertEqual(marker_state(started, {'start': 10, 'rc': 1}), 'failed')
        self.assertEqual(marker_state(started, {'start': 11, 'rc': 0}), 'failed')
        self.assertEqual(marker_state(None, {'start': 10, 'rc': 0}), 'failed')

    def test_quiet_gate_waits_for_known_worker_even_with_done_marker(self):
        def marked(_host, _stage, suffix):
            return {'start': 10, 'pid': 123} if suffix == 'started' else {'start': 10, 'rc': 0}
        with patch('scripts.v18_coordinate.marker', side_effect=marked), \
             patch('scripts.v18_coordinate.open_worker_pids', return_value=[dict(pid=456, dataset='aime')]), \
             patch('scripts.v18_coordinate.own_process_alive', side_effect=lambda _h, pid, _t: pid == 456), \
             patch('scripts.v18_coordinate.time.time', return_value=100):
            with self.assertRaisesRegex(RuntimeError, 'known GPU writers remain live'):
                wait_known_writers_quiet(['aime'], poll_seconds=60, deadline_epoch=99)

    def test_failure_finalizes_partial_ledgers_without_downstream_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = root / 'budget.json'
            budget.write_text(json.dumps(dict(deadline_epoch=2000000000,
                                              scoring_minutes_reserved=30)))
            command = root / 'final.json'
            command.write_text(json.dumps(['python', '-m', 'cpu_finalizer']))
            outcome = root / 'outcome.json'
            calls = []
            def fail_stage(stage, **_kw):
                calls.append(('stage', stage))
                raise RuntimeError('aime worker failed')
            def quiet(_stages, **_kw):
                calls.append(('quiet', None))
                return {'mpk': {}, 'dllm': {}}
            def sync(_staging):
                calls.append(('sync', None))
                return [dict(source='mpk', dataset='ruler', sha256='abc')]
            def finalize(*_args, **_kw):
                calls.append(('finalize', None))
            args = ['v18_coordinate', '--start-at', 'aime', '--budget', str(budget),
                    '--outcome', str(outcome), '--after-stages-command-file', str(command)]
            with patch.object(sys, 'argv', args), \
                 patch('scripts.v18_coordinate.wait_stage', side_effect=fail_stage), \
                 patch('scripts.v18_coordinate.wait_known_writers_quiet', side_effect=quiet), \
                 patch('scripts.v18_coordinate.sync_all_ledgers', side_effect=sync), \
                 patch('scripts.v18_coordinate.subprocess.run', side_effect=finalize):
                with self.assertRaises(SystemExit) as stopped:
                    v18_coordinate.main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(calls, [('stage', 'aime'), ('quiet', None),
                                     ('sync', None), ('finalize', None)])
            saved = json.loads(outcome.read_text())
            self.assertEqual(saved['gpu_stop_reason'], 'stage_failure')
            self.assertEqual(saved['finalization'], 'complete')
            self.assertTrue(saved['writers_quiet'])

    def test_clean_partial_aime_skips_remainder_but_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = root / 'budget.json'
            budget.write_text(json.dumps(dict(deadline_epoch=2000000000,
                                              scoring_minutes_reserved=30)))
            command = root / 'final.json'
            command.write_text(json.dumps(['python', 'cpu_finalizer.py']))
            outcome = root / 'outcome.json'
            stages = []
            args = ['v18_coordinate', '--start-at', 'aime', '--budget', str(budget),
                    '--outcome', str(outcome), '--after-stages-command-file', str(command)]
            with patch.object(sys, 'argv', args), \
                 patch('scripts.v18_coordinate.wait_stage', side_effect=lambda stage, **_: stages.append(stage)), \
                 patch('scripts.v18_coordinate.sync_ledger', return_value={'sha256': 'abc'}), \
                 patch('scripts.v18_coordinate.aime_complete', return_value=dict(complete=False,
                                                      completed=100, planned=1080)), \
                 patch('scripts.v18_coordinate.wait_known_writers_quiet', return_value={}), \
                 patch('scripts.v18_coordinate.sync_all_ledgers', return_value=[]), \
                 patch('scripts.v18_coordinate.subprocess.run') as finalizer:
                with self.assertRaises(SystemExit):
                    v18_coordinate.main()
            self.assertEqual(stages, ['aime'])
            finalizer.assert_called_once()
            saved = json.loads(outcome.read_text())
            self.assertEqual(saved['gpu_stop_reason'], 'clean_partial_aime')
            self.assertEqual(saved['finalization'], 'complete')

    def test_unreadable_writer_state_never_runs_final_hook(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = root / 'budget.json'
            budget.write_text(json.dumps(dict(deadline_epoch=2000000000,
                                              scoring_minutes_reserved=30)))
            command = root / 'final.json'
            command.write_text(json.dumps(['python', 'cpu_finalizer.py']))
            outcome = root / 'outcome.json'
            args = ['v18_coordinate', '--start-at', 'aime', '--budget', str(budget),
                    '--outcome', str(outcome), '--after-stages-command-file', str(command)]
            with patch.object(sys, 'argv', args), \
                 patch('scripts.v18_coordinate.wait_stage', side_effect=RuntimeError('AIME failed')), \
                 patch('scripts.v18_coordinate.wait_known_writers_quiet',
                       side_effect=OSError('remote state unreadable')), \
                 patch('scripts.v18_coordinate.subprocess.run') as finalizer:
                with self.assertRaises(SystemExit):
                    v18_coordinate.main()
            finalizer.assert_not_called()
            saved = json.loads(outcome.read_text())
            self.assertEqual(saved['finalization'], 'failed')
            self.assertIn('remote state unreadable', saved['finalization_error']['message'])
            self.assertNotIn('writers_quiet', saved)

    def test_aime_completion_requires_every_frozen_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(arm='T60', id=f'q{i}', seed=101, role='attempt0', repeat=0,
                         cell_id=f'cell{i}', block=i) for i in range(2)]
            protocol = dict(protocol_id='frozen-aime', schedule=rows,
                            block_assignments={str(i): dict(host='mpk', gpu_uuid='GPU-A')
                                               for i in range(2)})
            payload = json.dumps(protocol)
            import hashlib
            digest = hashlib.sha256(payload.encode()).hexdigest()
            ledger = root / 'mpk_aime.jsonl'
            (root / 'dllm_aime.jsonl').write_text('')
            def scp(command, **_kw):
                Path(command[-1]).write_text(payload)
            with patch('scripts.v18_coordinate.subprocess.run', side_effect=scp), \
                 patch('scripts.v18_coordinate.remote', return_value=digest):
                ledger.write_text(json.dumps(dict(rows[0], event='run',
                                                  execution_key='cell0:attempt0:0',
                                                  host='mpk', gpu_uuid='GPU-A')) + '\n')
                self.assertFalse(aime_complete(root)['complete'])
                ledger.write_text('\n'.join(json.dumps(dict(row, event='run',
                                       execution_key=f"cell{i}:attempt0:0",
                                       host='mpk', gpu_uuid='GPU-A'))
                                           for i, row in enumerate(rows)) + '\n')
                self.assertTrue(aime_complete(root)['complete'])


if __name__ == '__main__':
    unittest.main()
