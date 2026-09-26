import unittest

from scripts.v18_summarize import bootstrap_question_clusters, score_one


class SummaryTests(unittest.TestCase):
    def test_aime_requires_completed_final_answer(self):
        try:
            import torch  # noqa: F401 - pinned scorer import dependency
        except ImportError:
            self.skipTest('pinned task scorer requires the remote torch runtime')
        receipt = {'raw_completion': '<channel|>final\\boxed{42}<turn|>'}
        gold = {'expected': '42'}
        good = score_one('aime26', gold, receipt, {'termination': 'eos'})
        capped = score_one('aime26', gold, receipt, {'termination': 'length'})
        thought = score_one('aime26', gold, {'raw_completion': 'Thinking 42'}, {'termination': 'eos'})
        self.assertTrue(good['correct'])
        self.assertEqual(capped['score'], 0)
        self.assertEqual(thought['score'], 0)

    def test_question_bootstrap_stratifies_and_preserves_seed_pairing(self):
        cells = {}
        for rid, task, value in [('q1', 'task1', 1), ('q2', 'task2', 0)]:
            for seed in (101, 202, 303):
                for arm in ('U50', 'D_native'):
                    cells[(arm, rid, seed)] = dict(attempt0={'host': 'h', 'gpu_uuid': 'g'},
                        quality={'score': float(value if arm == 'U50' else 0)},
                        warm_s=1.0 if arm == 'U50' else 2.0,
                        work={'decoder_calls': 4 if arm == 'U50' else 8,
                              'canvases': 1, 'native_stop_canvases': 1})
        result = bootstrap_question_clusters(cells, 'U50', 'D_native',
                                             task_by_id={'q1': 'task1', 'q2': 'task2'},
                                             resamples=100, seed=7)
        self.assertEqual(result['questions'], 2)
        self.assertEqual(result['task_strata'], {'task1': 1, 'task2': 1})
        self.assertEqual(result['quality_difference'], 0.5)
        self.assertAlmostEqual(result['time_geomean_ratio'], 0.5)
        self.assertAlmostEqual(result['host_paired_decomposition']['h']['total_time_ratio'], 0.5)
        self.assertAlmostEqual(result['host_paired_decomposition']['h']['decoder_call_ratio'], 0.5)
        self.assertAlmostEqual(result['host_paired_decomposition']['h']['wall_per_call_ratio'], 1.0)
        self.assertEqual(result['per_seed_quality_difference'], {'101': 0.5, '202': 0.5, '303': 0.5})
        cells[('D_native', 'q1', 101)]['attempt0']['host'] = 'other'
        with self.assertRaisesRegex(ValueError, 'same host'):
            bootstrap_question_clusters(cells, 'U50', 'D_native', resamples=2)

    def test_timing_subset_has_separate_cluster_interval(self):
        cells = {}
        for rid in ('timed', 'untimed'):
            for seed in (101, 202, 303):
                for arm in ('U50', 'D_native'):
                    cells[(arm, rid, seed)] = dict(attempt0={'host': 'h', 'gpu_uuid': 'g'},
                        quality={'score': 0.0},
                        warm_s=(1.0 if arm == 'U50' else 2.0) if rid == 'timed' else None)
        result = bootstrap_question_clusters(cells, 'U50', 'D_native', resamples=10)
        self.assertEqual(result['questions'], 2)
        self.assertEqual(result['timed_questions'], 1)
        self.assertAlmostEqual(result['time_geomean_ratio'], 0.5)
        self.assertEqual(result['time_geomean_ratio_95'], [0.5, 0.5])


if __name__ == '__main__':
    unittest.main()
