import hashlib
import json
import tempfile
import unittest
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts import v18_secondary_pipeline as cp


class FakeTransport:
    def __init__(self):
        self.files = {}
        self.calls = []
        self.config = {'hosts': {h: {'root': '/' + h,
                                    'primary_protocol': '/' + h + '/ruler_protocol.json',
                                    'aime_protocol': '/' + h + '/aime_protocol.json'}
                                 for h in cp.EXPECTED}}

    def path(self, host, *parts):
        return '/' + host + '/' + '/'.join(parts)

    def read(self, host, path):
        return self.files.get((host, path))

    def write_immutable(self, host, path, data):
        old = self.read(host, path)
        if old is not None and old != data:
            raise ValueError('identity drift')
        self.files[(host, path)] = data

    def ssh(self, host, command, timeout=60):
        self.calls.append((host, command))
        return 'yes'


def primary_fixture(transport):
    for host in cp.EXPECTED:
        for stage in ('initial', 'aime', 'remainder'):
            prefix = f'/{host}/evaluation/status/{host}_{stage}'
            transport.files[(host, prefix + '.started.json')] = b'{"start":1}'
            transport.files[(host, prefix + '.done.json')] = b'{"start":1,"rc":0}'
    for dataset, stage in (('ruler', 'ruler4k_primary'), ('aime', 'aime26_primary')):
        summary = dict(schema='v18_redacted_summary_v1', stage=stage, complete=True,
                       protocol_id=stage + '_id', recorded_executions=10,
                       planned_executions=10)
        transport.files[('mpk', f'/mpk/scoring/{dataset}_redacted_summary.json')] = json.dumps(summary).encode()
        transport.files[('mpk', '/mpk/' + dataset + '_protocol.json')] = json.dumps(
            dict(stage=stage, protocol_id=stage + '_id', schedule=[{}] * 10)).encode()


class PipelineTests(unittest.TestCase):
    def test_read_only_ssh_retries_then_returns_exact_bytes(self):
        t = cp.Transport({'hosts': {'mpk': {'root': '/mpk', 'ssh': cp.EXPECTED['mpk']}}})
        calls = []
        def ssh(_host, _command, timeout=60):
            calls.append(timeout)
            if len(calls) < 3:
                raise subprocess.TimeoutExpired('ssh', timeout)
            return 'YWJj'
        t.ssh = ssh
        with patch.object(cp.time, 'sleep') as sleep:
            self.assertEqual(t.read('mpk', '/mpk/status.json'), b'abc')
        self.assertEqual(len(calls), 3)
        self.assertEqual([x.args[0] for x in sleep.call_args_list], [2, 4])

    def test_exhausted_read_is_bounded_and_mutation_is_not_retried(self):
        t = cp.Transport({'hosts': {'mpk': {'root': '/mpk', 'ssh': cp.EXPECTED['mpk']}}})
        calls = []
        def fail(_host, _command, timeout=60):
            calls.append(1)
            raise subprocess.CalledProcessError(255, 'ssh')
        t.ssh = fail
        with patch.object(cp.time, 'sleep'):
            with self.assertRaises(subprocess.CalledProcessError):
                t.read('mpk', '/mpk/status.json')
        self.assertEqual(len(calls), 3)
        t.read = lambda *_args: None
        calls.clear()
        with self.assertRaises(subprocess.CalledProcessError):
            t.write_immutable('mpk', '/mpk/new.json', b'{}')
        self.assertEqual(len(calls), 1)

    def test_read_cutoff_prevents_another_ssh_attempt(self):
        t = cp.Transport({'hosts': {'mpk': {'root': '/mpk', 'ssh': cp.EXPECTED['mpk']}}})
        t.read_deadline = 10
        with patch.object(cp.time, 'time', return_value=10), \
             patch.object(t, 'ssh') as ssh:
            with self.assertRaises(cp.PrimaryPendingDeadline):
                t.read('mpk', '/mpk/status.json')
        ssh.assert_not_called()

    def test_primary_read_outage_waits_until_original_deadline(self):
        t = FakeTransport()
        status = []
        error = subprocess.TimeoutExpired('ssh', 60)
        with patch.object(cp, 'primary_ready', side_effect=error), \
             patch.object(cp.time, 'time', side_effect=[0, 0, 0, 95]), \
             patch.object(cp.time, 'sleep'):
            with self.assertRaises(cp.PrimaryPendingDeadline):
                cp.run_pipeline(t, {'deadline_epoch': 100, 'block_guard_s': 10},
                                poll=60, folder=Path('/unused'),
                                status_callback=lambda **fields: status.append(fields))
        self.assertEqual(len(status), 2)
        self.assertTrue(all(x['connection_pending'] for x in status))

    def test_primary_requires_closed_workers_and_both_offline_summaries(self):
        t = FakeTransport()
        primary_fixture(t)
        self.assertTrue(cp.primary_ready(t))
        del t.files[('mpk', '/mpk/scoring/aime_redacted_summary.json')]
        self.assertFalse(cp.primary_ready(t))
        primary_fixture(t)
        t.files[('dllm', '/dllm/evaluation/status/dllm_remainder.done.json')] = b'{"start":1,"rc":1}'
        with self.assertRaisesRegex(RuntimeError, 'failed primary'):
            cp.primary_ready(t)

    def test_uncertain_dispatch_never_repeats(self):
        t = FakeTransport()
        t.config['deploy'] = 'frozen'
        t.config['hosts']['mpk'].update(python='/python', env={})
        def fail(host, command, timeout=60):
            if command.startswith('mkdir -p '):
                return ''
            raise TimeoutError('unknown SSH result')
        t.ssh = fail
        with self.assertRaisesRegex(RuntimeError, 'dispatch uncertain'):
            cp.launch_once(t, 'ruler_secondary70', 'mpk', ['-m', 'worker'])

    def test_logical_protocol_drift_rejected(self):
        t = FakeTransport()
        for host in cp.EXPECTED:
            t.files[(host, f'/{host}/secondary/ruler_secondary70_protocol.json')] = json.dumps(
                dict(stage='ruler_secondary70', status='frozen', protocol_id='phase',
                     logical_protocol_sha256='same' if host == 'mpk' else 'other')).encode()
        with self.assertRaisesRegex(ValueError, 'drift'):
            cp.assert_same_protocol(t, 'ruler_secondary70')

    def test_frozen_config_rejects_budget_or_qualification_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            budget_path = Path(temp, 'budget.json')
            budget = dict(schema='v18_frozen_campaign_budget_v1', hard_timeout_required=True,
                          block_guard_s=1800, deadline_epoch=1000000)
            budget_path.write_text(json.dumps(budget))
            gate_path = Path(temp, 'gate.json')
            hosts = {}
            for host, ssh in cp.EXPECTED.items():
                hosts[host] = dict(ssh=ssh, hostname=host, gpu_uuid=host + '-gpu',
                                   **{k: '/path' for k in ('root', 'python', 'model', 'library',
                                         'torch_library', 'draft', 'calibration_manifest',
                                         'policy_file', 'primary_protocol', 'aime_protocol',
                                         'primary_calibration', 'old_scope_policies',
                                         'ledger_inventory')})
            config = dict(hosts=hosts, deploy='fixed', budget_local=str(budget_path),
                          budget_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),
                          cpu_qualification=str(gate_path), torch_version='2.6.0+cu124',
                          torch_cuda='12.4', qualification_sources={name: 'a' * 64 for name in (
                              'scripts.v18_secondary', 'scripts.v18_secondary_coordinate',
                              'experiments.value_direction_hopper.frontier_scope',
                              'transformers.integrations.sdpa_attention')})
            cp.validate_config(config, 'fixed', budget, 0)
            budget_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'budget byte drift'):
                cp.validate_config(config, 'fixed', budget, 0)
            budget_path.write_text(json.dumps(budget))
            config['budget_sha256'] = hashlib.sha256(budget_path.read_bytes()).hexdigest()
            config['qualification_sources']['scripts.v18_secondary'] = 'short'
            with self.assertRaisesRegex(ValueError, 'qualification plan'):
                cp.validate_config(config, 'fixed', budget, 0)

    def test_resume_completed_worker_does_not_launch(self):
        t = FakeTransport()
        for host in cp.EXPECTED:
            stage = 'ruler_secondary70'
            path = f'/{host}/secondary/status/{host}_{stage}'
            t.files[(host, path + '.started.json')] = b'{"start":2,"pid":33}'
            t.files[(host, path + '.done.json')] = b'{"start":2,"rc":0}'
        cp.await_workers(t, 'ruler_secondary70', {h: ['-m', 'worker'] for h in cp.EXPECTED},
                         deadline=10**12, poll=60)
        self.assertEqual(t.calls, [])

    def test_one_complete_one_running_waits_for_second(self):
        t = FakeTransport()
        ticks = [0]
        def state(_t, _stage, host):
            return 'complete' if host == 'mpk' or ticks[0] else 'running'
        def advance(_seconds):
            ticks[0] += 1
        with patch.object(cp, 'worker_state', side_effect=state), \
             patch.object(cp.time, 'sleep', side_effect=advance):
            cp.await_workers(t, 'ruler_secondary70', {h: ['-m', 'worker'] for h in cp.EXPECTED},
                             deadline=10**12, poll=60)
        self.assertEqual(ticks, [1])

    def test_partial_closed_stage_is_scored_without_unlocking_controls(self):
        t = FakeTransport()
        t.files[('mpk', '/mpk/secondary/ruler_secondary70_protocol.json')] = b'{}'
        t.ssh = lambda *_args, **_kwargs: ''
        def state(_t, stage, host):
            return 'failed' if stage == 'ruler_secondary70' and host == 'dllm' else (
                'complete' if stage == 'ruler_secondary70' else 'absent')
        with patch.object(cp, 'worker_state', side_effect=state):
            self.assertEqual(cp.closed_stages(t), ['ruler_secondary70'])

    def test_private_archive_waits_for_supervisor_exit(self):
        t = FakeTransport()
        stage = 'ruler_secondary70'
        t.files[('mpk', '/mpk/secondary/' + stage + '_protocol.json')] = b'{}'
        t.files[('mpk', f'/mpk/secondary/status/mpk_{stage}.started.json')] = b'{"pid":123,"start":1}'
        t.files[('mpk', f'/mpk/secondary/status/mpk_{stage}.done.json')] = b'{"start":1,"rc":0}'
        t.ssh = lambda _host, command, **_kwargs: 'S' if command.startswith('ps ') else ''
        with self.assertRaisesRegex(RuntimeError, 'has not exited'):
            cp.closed_stages(t)

    def test_controls_spec_inherits_frozen_70_stage(self):
        t = FakeTransport()
        t.config['hosts']['mpk'].update(
            calibration_manifest='/cal26', old_scope_policies='/old.json',
            primary_calibration='/primary_cal.json', primary_protocol='/ruler.json',
            aime_protocol='/aime.json', gpu_uuid='GPU-1', model='/model',
            library='/kernel.so', torch_library='/torch.so', policy_file='/policy.json')
        source = cp.make_input(t, 'controls')
        self.assertEqual(source['group'], 'controls')
        self.assertEqual(source['preceding_secondary'],
                         '/mpk/secondary/ruler_secondary70_protocol.json')
        self.assertEqual(len(source['preceding_ledgers']), 2)
        self.assertNotIn('initial_policies', source)

    def test_inventory_charges_all_future_calibration_points_once(self):
        inventory = cp.ledger_inventory('/root')
        self.assertEqual(set(inventory), {'mpk', 'dllm'})
        self.assertEqual(len(inventory['mpk']), 14)  # primary 2, calibration 10, eval 2
        self.assertEqual(len(inventory['dllm']), 4)
        self.assertEqual(len(set(inventory['mpk'] + inventory['dllm'])), 18)
        self.assertTrue(all('/calibration_controls/' in path or '/calibration_core70/' in path
                            for path in inventory['mpk'][2:12]))

    def test_terminal_failure_persists_partial_scoring_deferred(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            budget = base / 'budget.json'
            budget.write_text('{"deadline_epoch":9999999999}')
            config = base / 'config.json'
            config.write_text(json.dumps(dict(budget_local=str(budget), deploy='frozen')))
            argv = ['pipeline', '--deploy', 'frozen', '--config', str(config)]
            with patch.object(sys, 'argv', argv), \
                 patch.object(cp, 'validate_config'), \
                 patch.object(cp, 'run_pipeline', side_effect=RuntimeError('stage failed')), \
                 patch.object(cp, 'score_secondary', side_effect=RuntimeError('writer active')):
                with self.assertRaisesRegex(RuntimeError, 'stage failed'):
                    cp.main()
            outcome = json.loads(config.with_suffix('.outcome.json').read_text())
            self.assertEqual(outcome['status'], 'error')
            self.assertEqual(outcome['partial_scoring'], 'deferred')
            self.assertEqual(outcome['partial_scoring_reason'], 'RuntimeError')
            self.assertEqual(outcome['deploy'], 'frozen')

    def test_primary_deadline_never_calls_partial_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            budget = base / 'budget.json'
            budget.write_text('{"deadline_epoch":100}')
            config = base / 'config.json'
            config.write_text(json.dumps(dict(budget_local=str(budget), deploy='frozen')))
            with patch.object(sys, 'argv', ['pipeline', '--deploy', 'frozen', '--config', str(config)]), \
                 patch.object(cp, 'validate_config'), \
                 patch.object(cp, 'run_pipeline', side_effect=cp.PrimaryPendingDeadline('expired')), \
                 patch.object(cp, 'score_secondary') as scorer:
                with self.assertRaises(cp.PrimaryPendingDeadline):
                    cp.main()
            scorer.assert_not_called()
            outcome = json.loads(config.with_suffix('.outcome.json').read_text())
            self.assertEqual(outcome['partial_scoring'], 'not_started')


if __name__ == '__main__':
    unittest.main()
