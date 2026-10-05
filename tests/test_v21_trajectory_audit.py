import unittest

from scripts.v21_trajectory_audit import (DEFAULT_RECORDS, build_cells, quantiles, repetition,
                                          select_text_cells, similarity, stop_category, verify_manifest)


class TrajectoryAuditTests(unittest.TestCase):
    def test_disjoint_stopping(self):
        base = {"native_stop_final_call": False, "iteration_cap_final_call": False}
        self.assertEqual(stop_category(base), "neither")
        self.assertEqual(stop_category({**base, "native_stop_final_call": True}), "native_stop_only")
        self.assertEqual(stop_category({**base, "iteration_cap_final_call": True}), "cap_only")
        self.assertEqual(stop_category({**base, "native_stop_final_call": True,
                                        "iteration_cap_final_call": True}), "both")

    def test_quantile_and_similarity_are_descriptive(self):
        self.assertEqual(quantiles([1, 3, 5, 7, 9])["p90"], 8.2)
        self.assertEqual(similarity([1, 2, 4], [1, 2, 3])["common_prefix_tokens"], 2)
        self.assertEqual(repetition([1] * 20, 16), .8)

    def test_selection_is_metadata_only_bounded_and_deterministic(self):
        cells = {}
        arms = ("D_native", "D_matched", "T_scope", "B_A8_matched", "M1_R1_A8_current_output",
                "M3_R2_A8_current_output", "M3_R3_A8_current_output", "G75L30_nativeQ128")
        for dataset in ("longbench_v2", "aime26"):
            for n in range(6):
                for arm in arms:
                    calls = 10 + (n - 3 if arm == "M3_R2_A8_current_output" else
                                  10 - n if arm == "D_matched" else 0)
                    cells[(dataset, f"q{n}", 101, arm)] = {"attempt0": {"generated": {"total_decoder_calls": calls}}}
        selected = select_text_cells(cells)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(r["dataset"] == "longbench_v2" for r in selected), 4)
        self.assertEqual(sum(r["dataset"] == "aime26" for r in selected), 2)
        self.assertEqual(selected, select_text_cells(cells))

    def test_published_inventory_and_warm_trajectories(self):
        manifest, records, sha = verify_manifest(DEFAULT_RECORDS)
        cells = build_cells(records)
        self.assertEqual((manifest["first_records"], manifest["warm_records"], len(cells)), (400, 400, 400))
        self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main()
