"""Standard-library tests; no Torch, CUDA or pytest required."""
import math
import unittest

from scripts import v30_closed_length_accounting as audit


def row(index, arm, w, s, n, *, dataset='longbench_v2_64k'):
    return dict(dataset=dataset, index=index, repeat=0, arm=arm, wall_s=w,
                decode_span_s=s, denoise_forward_count=n, commit_forward_count=2,
                output_tokens=100, host='own', gpu_uuid='gpu', engine_seed=7,
                deploy_commit='pinned', protocol_id='closed')


class ClosedLengthAccountingTests(unittest.TestCase):
    def test_pair_uses_each_cell_before_geomean_and_preserves_subset(self):
        a = [row(1, 'method', 4, 2, 4), row(2, 'method', 18, 12, 3)]
        b = [row(2, 'dense', 9, 6, 6), row(1, 'dense', 8, 4, 2), row(3, 'dense', 20, 10, 9)]
        report = audit.paired(a, b)[0]
        self.assertEqual(report['cells'], 2)
        self.assertEqual(report['questions'], 2)
        self.assertAlmostEqual(report['ratios']['W'], 1)
        self.assertAlmostEqual(report['ratios']['N'], 1)
        self.assertAlmostEqual(report['ratios']['SN'], math.sqrt(.25 * 4))
        self.assertNotIn('index', str(report))
        self.assertNotIn('own', str(report))
        with self.assertRaisesRegex(ValueError, 'inventories'):
            audit.paired(a, b, require_equal=True)

    def test_duplicate_or_execution_identity_rejected(self):
        a, b = row(1, 'method', 4, 2, 4), row(1, 'dense', 8, 4, 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.paired([a, a], [b])
        for field in ('host', 'gpu_uuid', 'engine_seed', 'deploy_commit', 'protocol_id'):
            with self.subTest(field=field):
                altered = dict(b)
                altered[field] = 'changed'
                with self.assertRaisesRegex(ValueError, 'identity'):
                    audit.paired([a], [altered])

    def test_length_strata_are_not_pooled(self):
        rows = [row(1, 'method', 4, 2, 4), row(2, 'method', 8, 4, 4, dataset='longbench_v2_96k')]
        baseline = [row(1, 'dense', 8, 4, 4), row(2, 'dense', 4, 2, 4, dataset='longbench_v2_96k')]
        reports = audit.paired(rows, baseline, require_equal=True)
        self.assertEqual([(r['dataset'], r['ratios']['W']) for r in reports], [('longbench_v2_64k', .5), ('longbench_v2_96k', 2)])

    def test_invalid_timing_or_counts_fail(self):
        for values in ([], [0], [-1], [float('nan')], [float('inf')], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.geomean(values)

    def test_no_paired_cells_fails(self):
        with self.assertRaisesRegex(ValueError, 'no paired'):
            audit.paired([row(1, 'method', 4, 2, 4)], [row(2, 'dense', 8, 4, 2)])


if __name__ == '__main__':
    unittest.main()
