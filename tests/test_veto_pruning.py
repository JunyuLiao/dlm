import unittest

import numpy as np

from analysis.veto_pruning import (
    analytical_safe_mask,
    apply_group_thresholds,
    calibrated_thresholds,
    online_attention_mass,
    wilson_upper,
)


class VetoPruningAnalysisTest(unittest.TestCase):
    def test_online_mass_is_one_for_first_tile(self):
        local = np.array([[2.0, 3.0]], dtype=np.float32)
        running_max = np.full_like(local, -np.inf)
        running_sum = np.zeros_like(local)
        np.testing.assert_allclose(
            online_attention_mass(local, running_max, running_sum), np.ones_like(local)
        )

    def test_safe_mask_protects_new_max_and_masked_error(self):
        error = np.array([[0.001, 0.01], [0.001, 0.001]], dtype=np.float32)
        states = np.array([[0, 0], [0, 3]], dtype=np.uint8)
        valid = np.ones_like(states, dtype=bool)
        safe = analytical_safe_mask(error, states, valid, np.array([False, True]))
        np.testing.assert_array_equal(safe, [False, False])

    def test_calibration_is_group_disjoint_and_bounded(self):
        score = np.arange(600, dtype=np.float32)
        safe = np.ones(600, dtype=bool)
        eligible = np.ones(600, dtype=bool)
        group = np.repeat([0, 1], 300)
        calibration = np.ones(600, dtype=bool)
        thresholds = calibrated_thresholds(
            score,
            safe,
            eligible,
            group,
            calibration,
            max_wilson_upper=0.02,
            minimum_support=256,
        )
        selected = apply_group_thresholds(score, group, thresholds)
        self.assertEqual(int(selected.sum()), 600)
        self.assertLess(wilson_upper(0, 300), 0.02)


if __name__ == "__main__":
    unittest.main()
