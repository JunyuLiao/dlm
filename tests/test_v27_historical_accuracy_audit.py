"""Public toy-only tests; runnable with the standard library unittest runner."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import v27_historical_accuracy_audit as audit


class HistoricalAccuracyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rows(self):
        return [dict(dataset='toy_private_label', id='private_item_alpha', seed=str(seed),
                     arm=arm, first_status='success', quality_eligible='True', scored_first='True',
                     strict_correct=str(correct), host='toy_host', gpu_uuid='toy_gpu', cell_id=str(seed))
                for seed, a, b in [(1, True, False), (2, False, False)]
                for arm, correct in [('method', a), ('dense', b)]]

    def write(self, rows, name='toy.csv'):
        path = self.root / name
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read(self, rows):
        return audit.read_pairs([self.write(rows)], 'method', 'dense')

    def test_pairs_and_aggregate_counts(self):
        result = audit.summarize(self.read(self.rows()), bootstrap_reps=100)
        row = result['datasets'][0]
        self.assertEqual((row['cells'], row['items'], row['arm_correct'], row['base_correct']), (2, 1, 1, 0))
        self.assertEqual(row['accuracy_difference'], .5)
        self.assertEqual(row['accuracy_difference_item_cluster_ci95'], [.5, .5])

    def test_input_row_order_does_not_change_bootstrap(self):
        forward = audit.summarize(self.read(self.rows()), bootstrap_reps=100)
        reverse = audit.summarize(self.read(list(reversed(self.rows()))), bootstrap_reps=100)
        self.assertEqual(forward, reverse)

    def test_privacy_unknown_dataset_and_extra_columns(self):
        rows = self.rows()
        for row in rows:
            row.update(prompt='private_prompt', gold='private_gold', prompt_hash='private_hash')
        text = json.dumps(audit.summarize(self.read(rows), bootstrap_reps=100))
        for forbidden in ['toy_private_label', 'private_item_alpha', 'private_prompt', 'private_gold',
                          'private_hash', str(self.root), 'toy_host', 'toy_gpu']:
            self.assertNotIn(forbidden, text)
        self.assertIn('anonymous_dataset_001', text)

    def test_duplicate_in_file(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            self.read(rows + [rows[0]])

    def test_duplicate_across_files(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            audit.read_pairs([self.write(rows), self.write(rows, 'copy.csv')], 'method', 'dense')

    def test_missing_pair(self):
        with self.assertRaises(ValueError):
            self.read(self.rows()[:-1])

    def test_bad_qualification_and_boolean(self):
        for field, value in [('first_status', 'failure'), ('quality_eligible', 'False'),
                             ('scored_first', 'False'), ('strict_correct', '1')]:
            rows = self.rows()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(rows)

    def test_empty_or_bad_seed(self):
        for value in ['', 'not_integer']:
            rows = self.rows()
            rows[0]['seed'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(rows)

    def test_missing_columns_or_no_selected_arms(self):
        rows = self.rows()
        for row in rows:
            del row['scored_first']
        with self.assertRaises(ValueError):
            self.read(rows)
        with self.assertRaises(ValueError):
            audit.read_pairs([self.write(self.rows())], 'other', 'another')

    def test_environment_mismatch(self):
        for field in ['host', 'gpu_uuid']:
            rows = self.rows()
            rows[0][field] = 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(rows)

    def test_arm_execution_cell_ids_can_differ(self):
        rows = self.rows()
        rows[0]['cell_id'] = 'toy_method_execution'
        self.assertEqual(len(self.read(rows)['toy_private_label']), 2)

    def test_exact_sign_flip_public_examples(self):
        self.assertEqual(audit.exact_item_sign_flip([[1], [1]]), .5)
        self.assertEqual(audit.exact_item_sign_flip([[1], [-1]]), 1)
        self.assertEqual(audit.exact_item_sign_flip([[0], [0]]), 1)
        self.assertIsNone(audit.exact_item_sign_flip([[1]] * 17))
        # Repeats stay together: these are two items, not four independent cells.
        self.assertEqual(audit.exact_item_sign_flip([[1, 1], [1, 1]]), .5)

    def test_bootstrap_determinism_and_unequal_cluster_weight(self):
        groups = [[1, 1], [-1], [0, 0, 0]]
        self.assertEqual(audit.item_bootstrap_ci(groups, 100, 7), audit.item_bootstrap_ci(groups, 100, 7))
        self.assertEqual(audit.item_bootstrap_ci([[1, 0]], 100), [.5, .5])
        self.assertEqual(audit.item_bootstrap_ci([[1], [-1]], 100), [-1, 1])

    def test_invalid_statistical_parameters(self):
        for groups, reps in [([], 100), ([[]], 100), ([[1]], 0)]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                audit.item_bootstrap_ci(groups, reps)
        with self.assertRaises(ValueError):
            audit.exact_item_sign_flip([[1]], 21)

    def test_exploratory_mcnemar(self):
        self.assertEqual(audit.exploratory_mcnemar(0, 0), 1)
        self.assertEqual(audit.exploratory_mcnemar(2, 0), .5)
        self.assertAlmostEqual(audit.exploratory_mcnemar(29, 12), .011507789735333063)

    def test_cli_output_and_public_dataset(self):
        rows = self.rows()
        for row in rows:
            row['dataset'] = 'longbench_v2_96k'
        output = self.root / 'aggregate.json'
        audit.main(['--scored', str(self.write(rows)), '--arm', 'method', '--base', 'dense',
                    '--out', str(output), '--bootstrap-reps', '100'])
        result = json.loads(output.read_text())
        self.assertEqual(result['datasets'][0]['dataset'], 'longbench_v2_96k')
        self.assertIn('exploratory', result['interpretation'])

    def test_cli_error_does_not_echo_private_path(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
            audit.main(['--scored', str(self.root / 'private_missing.csv'), '--arm', 'method', '--base', 'dense'])
        self.assertNotIn(str(self.root), error.getvalue())
        self.assertNotIn('private_missing', error.getvalue())

    def test_cli_cannot_overwrite_scored_input(self):
        path = self.write(self.rows())
        original = path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.main(['--scored', str(path), '--arm', 'method', '--base', 'dense', '--out', str(path)])
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
