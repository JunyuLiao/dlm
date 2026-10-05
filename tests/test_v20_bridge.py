"""CPU-only bridge target and exact cross-host comparison tests."""
from __future__ import annotations

import unittest

from scripts import v20_bridge as B


class V20BridgeTests(unittest.TestCase):
    def test_bridge_target_is_first_frozen_ruler_seed(self):
        protocol = {'block_assignments': {'0': {'dataset': 'ruler4k', 'seed': 101,
                                               'id': 'ruler4k/a'}},
                    'ids': {'ruler4k': ['ruler4k/a']}}
        self.assertEqual(B.bridge_target(protocol), ('ruler4k/a', 101))
        protocol['block_assignments']['0']['seed'] = 202
        with self.assertRaisesRegex(ValueError, 'seed101'):
            B.bridge_target(protocol)

    def test_cross_host_comparison_flags_output_work_and_source(self):
        common = dict(protocol_sha256='p', binding_sha256='b', id='ruler4k/a',
                      seed=101, model_revision='rev')
        row = dict(arm='D_native', status='ok', completion_token_hash='x',
                   per_canvas_calls=[3], termination='eos',
                   router_phase_evidence={'phase': 'native_dense'},
                   source_value_hash='s', config_fingerprint='host-specific')
        a = dict(common, host='host0', rows=[row])
        b = dict(common, host='host1', rows=[dict(row, config_fingerprint='other')])
        compared = B.compare_rows(a, b)
        self.assertTrue(compared['exact_all'])
        self.assertFalse(compared['per_arm']['D_native']['config_fingerprint_match'])
        b['rows'][0]['per_canvas_calls'] = [4]
        self.assertFalse(B.compare_rows(a, b)['exact_all'])

    def test_failure_marks_remaining_arms_not_run(self):
        rows = B.failure_tail(['D_native', 'M1', 'M3'], 'M1', {'seed': 101})
        self.assertEqual(rows, [{'seed': 101, 'arm': 'M3', 'status': 'not_run',
                                 'reason': 'prior_bridge_failure'}])


if __name__ == '__main__':
    unittest.main()
