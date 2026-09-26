"""Dependency-free checks for secondary launch and calibration gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.v13_seed_runs import execution_key
from scripts.v18_evaluate import logical_protocol_digest
from scripts.v18_secondary import inherit_block_assignments, prerequisite_gate, stage_completion
from scripts.v18_secondary_coordinate import (budget_plan, calibrate_group, calibration_quota,
                                              primary_quiet_gate)


def protocol(stage):
    row = dict(index=0, block=0, arm='T60', id='q', seed=101, role='attempt0',
               repeat=0, cell_id=stage)
    value = dict(stage=stage, status='frozen', protocol_id=stage, schedule=[row],
                 block_assignments={'0': dict(host='old', gpu_uuid='GPU-A')})
    for key in ('ids', 'seeds', 'arm_hashes', 'manifest_sha256', 'calibration_sha256',
                'binary_hashes', 'model_tensor_identity_sha256', 'assigned_hosts',
                'bridge_receipt_sha256', 'initial_prefix_blocks'):
        value[key] = None
    value['logical_protocol_sha256'] = logical_protocol_digest(value)
    return value


class SecondaryStdlibTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ruler = protocol('ruler4k_primary')
        self.aime = protocol('aime26_primary')

    def ledger(self, item, name):
        path = self.root / f'{name}.jsonl'
        row = item['schedule'][0]
        path.write_text('\n'.join((json.dumps(dict(event='start', when=10.)),
                                   json.dumps(dict(row, event='run', execution_key=execution_key(row),
                                                   host='old', gpu_uuid='GPU-A')),
                                   json.dumps(dict(event='worker_end', when=15., host='old',
                                                   gpu_process_seconds=6.)))) + '\n')
        return path

    def summaries(self):
        result = {}
        for item in (self.ruler, self.aime):
            path = self.root / f"{item['stage']}_summary.json"
            path.write_text(json.dumps(dict(schema='v18_redacted_summary_v1',
                                            stage=item['stage'], protocol_id=item['protocol_id'],
                                            complete=True, planned_executions=1,
                                            recorded_executions=1)))
            result[item['stage']] = path
        return result

    def test_full_primary_identity_duplicate_and_finalizer_gate(self):
        ruler_ledger = self.ledger(self.ruler, 'ruler')
        aime_ledger = self.ledger(self.aime, 'aime')
        prior = {'ruler4k_primary': (self.ruler, [ruler_ledger]),
                 'aime26_primary': (self.aime, [aime_ledger])}
        secondary = dict(stage='ruler_secondary70', prerequisite_protocol_ids={
            stage: value[0]['protocol_id'] for stage, value in prior.items()})
        self.assertTrue(prerequisite_gate(secondary, *prior.values())['ready'])
        self.assertTrue(primary_quiet_gate(prior, self.summaries(),
                                           {'old': [ruler_ledger, aime_ledger]}, gpu_idle=True)['ready'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            stage_completion(self.ruler, [ruler_ledger, ruler_ledger])
        drift = dict(self.ruler, protocol_id='wrong')
        with self.assertRaisesRegex(ValueError, 'logical'):
            stage_completion(drift, [ruler_ledger])
        incomplete = self.summaries()
        incomplete['aime26_primary'].write_text(json.dumps(dict(schema='v18_redacted_summary_v1',
                                        stage='aime26_primary', protocol_id='aime26_primary',
                                        complete=False, planned_executions=1, recorded_executions=1)))
        with self.assertRaisesRegex(ValueError, 'completion'):
            primary_quiet_gate(prior, incomplete, {'old': [ruler_ledger]}, gpu_idle=True)

    def test_inherited_assignment_and_campaign_cutoff(self):
        primary = dict(schedule=[dict(id='a', seed=101, block=0),
                                 dict(id='b', seed=101, block=1)],
                       block_assignments={'0': dict(host='old', gpu_uuid='GPU-A'),
                                          '1': dict(host='new', gpu_uuid='GPU-B')})
        secondary = [dict(id='b', seed=101, block=0), dict(id='a', seed=101, block=1)]
        self.assertEqual(inherit_block_assignments(primary, secondary)['0']['host'], 'new')
        budget = dict(gpu_cap_s_by_host={'old': 100, 'new': 100},
                      request_cap_by_host={'old': 5, 'new': 5},
                      baseline_gpu_s_by_host={'old': 10, 'new': 0},
                      baseline_requests_by_host={'old': 1, 'new': 0},
                      baseline_requests=1, request_cap=10, deadline_epoch=200,
                      block_guard_s=5, scoring_minutes_reserved=30)
        quota = calibration_quota(budget, {'old': [], 'new': []}, 'old', 2, now=100)
        self.assertTrue(quota['allowed'])
        self.assertEqual(quota['deadline_epoch'], 200)  # already excludes scoring reserve
        self.assertFalse(calibration_quota(budget, {'old': [], 'new': []}, 'old', 5,
                                           now=100)['allowed'])
        with self.assertRaisesRegex(ValueError, 'budget'):
            calibration_quota(dict(budget, request_cap=9), {'old': [], 'new': []}, 'old', 1,
                              now=100)

    def test_calibration_never_freezes_before_completed_primary_and_70(self):
        core = protocol('ruler_secondary70')
        ruler_ledger = self.ledger(self.ruler, 'ruler')
        aime_ledger = self.ledger(self.aime, 'aime')
        prior = {'ruler4k_primary': (self.ruler, [ruler_ledger]),
                 'aime26_primary': (self.aime, [aime_ledger]),
                 'ruler_secondary70': (core, [])}
        spec = dict(group='controls', ids=[f'q{i}' for i in range(26)],
                    initial_policies={'T60_shuffled': {}, 'T60_uniform': {}},
                    prerequisite_protocol_ids={k: v[0]['protocol_id'] for k, v in prior.items()})
        with patch('scripts.v18_secondary_coordinate.freeze_calibration_point') as freeze:
            result = calibrate_group(spec, prior, {'old': []}, {}, execute=False)
            self.assertEqual(result['status'], 'prerequisite_incomplete')
            freeze.assert_not_called()

    def test_calibration_resume_dry_run_is_frozen_and_gpu_free(self):
        ruler_ledger = self.ledger(self.ruler, 'ruler')
        aime_ledger = self.ledger(self.aime, 'aime')
        prior = {'ruler4k_primary': (self.ruler, [ruler_ledger]),
                 'aime26_primary': (self.aime, [aime_ledger])}
        policy = {'local': {'log_threshold': -1.}, 'global': {'log_threshold': -2.}}
        output = self.root / 'cal70'
        spec = dict(group='core70', ids=[f'q{i}' for i in range(26)],
                    initial_policies={'U70': policy, 'T70': policy},
                    prerequisite_protocol_ids={k: v[0]['protocol_id'] for k, v in prior.items()},
                    prerequisites={k: dict(protocol=str(self.root / f'{k}.json'),
                                           ledgers=[str(p) for p in value[1]])
                                   for k, value in prior.items()},
                    primary_summaries={k: str(v) for k, v in self.summaries().items()},
                    calibration_host='old', calibration_gpu_uuid='GPU-A', out=str(output),
                    manifest='manifest', policy_file='policy', model='model', library='kernel',
                    torch_library='bridge', authorization='frozen', lock='lock')
        budget = dict(gpu_cap_s_by_host={'old': 100}, request_cap_by_host={'old': 20},
                      baseline_gpu_s_by_host={'old': 0}, baseline_requests_by_host={'old': 0},
                      baseline_requests=0, request_cap=20, deadline_epoch=200,
                      block_guard_s=5)
        schedule = [dict(arm='U70', id='q0', seed=42, role='attempt0', repeat=0,
                         cell_id='cal', block=0)]
        with (patch('scripts.v18_secondary_coordinate.socket.gethostname', return_value='old'),
              patch('scripts.v18_secondary_coordinate.time.time', return_value=100.),
              patch('scripts.v18_secondary_coordinate.freeze_calibration_point',
                    return_value=dict(schedule=schedule)) as freeze):
            kwargs = dict(execute=False, gpu_idle=True)
            first = calibrate_group(spec, prior, {'old': [ruler_ledger, aime_ledger]},
                                    budget, **kwargs)
            second = calibrate_group(spec, prior, {'old': [ruler_ledger, aime_ledger]},
                                     budget, **kwargs)
        self.assertEqual(first['status'], second['status'])
        self.assertEqual(first['status'], 'ready')
        self.assertEqual(freeze.call_count, 2)
        self.assertTrue((output / 'calibration_state.json').is_file())
        with patch('scripts.v18_secondary_coordinate.socket.gethostname', return_value='old'):
            with self.assertRaisesRegex(ValueError, 'identity drift'):
                calibrate_group(dict(spec, authorization='changed'), prior,
                                {'old': [ruler_ledger, aime_ledger]}, budget,
                                execute=False, gpu_idle=True)


if __name__ == '__main__':
    unittest.main()
