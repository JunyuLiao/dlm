"""Small adversarial checks for the CP0 evidence gate."""
import unittest

from scripts.v18_evidence_audit import require_equal_lengths, unique_ids, validate_warm


class EvidenceAuditTests(unittest.TestCase):
    def test_duplicate_id_rejected_even_if_set_covers_expected(self):
        issues = []
        unique_ids([{'id': 'a'}, {'id': 'a'}, {'id': 'b'}], ['a', 'b'], issues)
        self.assertIn('duplicate_manifest_id', [i['code'] for i in issues])

    def test_absent_fields_do_not_match_as_none(self):
        first = {'ok': True, 'completion_token_hash': None, 'per_canvas_calls': None, 'termination': None}
        warm = dict(first, triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[])
        reasons = validate_warm(first, warm)
        self.assertIn('attempt0_completion_token_hash_absent', reasons)
        self.assertIn('warm_phases_absent', reasons)

    def test_tampered_token_and_termination_rejected(self):
        first = {'ok': True, 'completion_token_hash': 'a', 'per_canvas_calls': [1], 'termination': 'eos',
                 'phases': {'anchor': 0, 'reselect': 0, 'held': 0, 'routed_calls': 1}}
        warm = dict(first, completion_token_hash='b', termination='length', triton_misses=0,
                    triton_disk_entries_added=0, new_shared_objects=[])
        reasons = validate_warm(first, warm)
        self.assertIn('completion_token_hash_mismatch', reasons)
        self.assertIn('termination_mismatch', reasons)

    def test_scorer_input_truncation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scorer_input_length_mismatch'):
            require_equal_lengths(['a', 'b'], ['A'], ['eos', 'eos'])


if __name__ == '__main__':
    unittest.main()
