import unittest
from dataclasses import replace
from experiments.numerical_qk_reuse.cache import Identity, ScoreCache


def identity():
    return Identity(0, 0, 1, 5, 1, 16, 2, 256, 128, 512, 100, 0, 128,
                    1.0, "bfloat16", "cuda:0", (False, 0), 123)


class LifecycleTest(unittest.TestCase):
    def test_score_and_decision_clocks_are_distinct(self):
        key = identity(); cache = ScoreCache(decision_interval=2)
        refresh = []
        decision = []
        for step in range(12):
            plan = cache.plan(key, step)
            if plan.score_refresh:
                refresh.append(step); cache.publish_scores(key, step, object())
            if plan.decision_refresh:
                decision.append(step); cache.publish_decision(key, step, object())
        self.assertEqual(refresh, [0, 8])
        self.assertEqual(decision, [0, 2, 4, 6, 8, 10])
        self.assertEqual(cache.get(key).score_step, 8)

    def test_r1_is_m1_and_r3_refresh_restarts_clock(self):
        for interval, expected in [(1, list(range(10))), (3, [0, 3, 6, 8])]:
            cache = ScoreCache(decision_interval=interval); key = identity(); got = []
            for step in range(10):
                p = cache.plan(key, step)
                if p.score_refresh: cache.publish_scores(key, step, object())
                if p.decision_refresh:
                    got.append(step); cache.publish_decision(key, step, object())
            self.assertEqual(got, expected)

    def test_reset_and_geometry_never_inherit_scores(self):
        key = identity(); cache = ScoreCache(); cache.publish_scores(key, 0, object())
        for field, value in [("canvas", 1), ("request", 1), ("encoder_epoch", 2),
                             ("query_start", 101), ("key_start", 64), ("scale", .5),
                             ("prefix_owner", 456), ("mask_signature", (True, 0))]:
            self.assertTrue(cache.plan(replace(key, **{field: value}), 1).score_refresh)
        cache.clear(); self.assertEqual(cache.storage_bytes, 0)

    def test_planning_does_not_publish_and_budget_checked_before_allocation(self):
        key = identity(); cache = ScoreCache(max_bytes=key.storage_bytes-1)
        self.assertTrue(cache.plan(key, 0).score_refresh)
        self.assertEqual(cache.entries, {})
        with self.assertRaises(MemoryError): cache.reserve(key)

    def test_invalid_mask_forces_real_observation_and_backward_step_rejected(self):
        key = identity(); cache = ScoreCache(); cache.publish_scores(key, 3, object())
        cache.publish_decision(key, 3, object())
        self.assertTrue(cache.plan(key, 4, force_refresh=True).score_refresh)
        with self.assertRaises(ValueError): cache.plan(key, 2)


if __name__ == "__main__": unittest.main()
