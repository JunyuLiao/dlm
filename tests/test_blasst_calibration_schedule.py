import pathlib
import unittest

from blasst.calibration import (
    calibration_matches_runtime_defaults,
    load_physical_calibration,
    select_largest_eligible_lambda,
)
from blasst.triton_bidirectional import DiffusionLambdaSchedule


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CalibrationScheduleRegressionTest(unittest.TestCase):
    def test_calibrated_defaults_and_boundaries(self):
        schedule = DiffusionLambdaSchedule()
        self.assertEqual(schedule.threshold(0.9), 0.04858582466840744)
        self.assertEqual(schedule.threshold(0.5), 0.4334796965122223)
        self.assertEqual(schedule.threshold(0.15), 1.0)
        self.assertEqual(schedule.threshold(0.75), schedule.high_noise_lambda)
        self.assertEqual(schedule.threshold(0.25), schedule.mid_noise_lambda)
        self.assertEqual(schedule.threshold(0.249999), schedule.low_noise_lambda)

    def test_configuration_round_trip_is_exact_and_zero_remains_available(self):
        schedule = DiffusionLambdaSchedule()
        restored = DiffusionLambdaSchedule.from_dict(schedule.to_dict())
        self.assertEqual(restored, schedule)
        self.assertEqual(DiffusionLambdaSchedule(high_noise_lambda=0.0).threshold(0.9), 0.0)

    def test_exact_selection_has_no_hidden_epsilon(self):
        rejected = 0.9499999938
        self.assertEqual(select_largest_eligible_lambda([(0.5, rejected)]), 0.0)
        self.assertEqual(select_largest_eligible_lambda([(0.4, 0.95), (0.5, rejected)]), 0.4)

    def test_runtime_defaults_match_existing_calibration_artifact(self):
        calibration = load_physical_calibration(ROOT / "outputs" / "blasst_physical_calibration_4096.json")
        schedule = DiffusionLambdaSchedule()
        self.assertTrue(
            calibration_matches_runtime_defaults(
                calibration,
                high_noise_lambda=schedule.high_noise_lambda,
                mid_noise_lambda=schedule.mid_noise_lambda,
                low_noise_lambda=schedule.low_noise_lambda,
            )
        )
        expected = {
            0.15: (0.4838, 0.9692),
            0.5: (0.2923, 0.9586),
            0.9: (0.0986, 0.9551),
        }
        for ratio, (sparsity, agreement) in expected.items():
            self.assertAlmostEqual(calibration.physical_sparsity[ratio], sparsity, delta=5e-5)
            self.assertAlmostEqual(calibration.dense_agreement[ratio], agreement, delta=5e-5)
            self.assertGreaterEqual(calibration.dense_agreement[ratio], 0.95)
        self.assertEqual(calibration.skipped_tiles, 3_668_691)
        self.assertEqual(calibration.total_tiles, 12_582_912)
        self.assertAlmostEqual(calibration.overall_physical_sparsity, 0.2916, delta=5e-5)


if __name__ == "__main__":
    unittest.main()
