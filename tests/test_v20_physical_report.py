import json
import tempfile
import unittest
from pathlib import Path

from scripts.v20_physical_report import _rates, digest, render, summarize


def counts(eligible=100, skipped=20):
    return dict(eligible_pairs=eligible, skipped_qk_pairs=skipped, executed_qk_pairs=eligible - skipped,
                skipped_pv_pairs=skipped, executed_pv_pairs=eligible - skipped,
                executed_qk_multiply_accumulates=eligible - skipped,
                executed_pv_multiply_accumulates=eligible - skipped)


class PhysicalReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.profile = root / "profile.json"
        self.cost = root / "cost.json"
        physical = dict(accepted_timing=False, projection_skipping_measured=False, calls=1,
                        rows=[dict(phase="D")],
                        by_phase_kind={"D/local": {s: counts() for s in ("whole", "static_prefix", "current_canvas")}})
        arm = dict(name="M1", config=dict(v20_scope="ALL_NATIVE_LEGAL", policy_name="P0",
                                          policy_sha256="a" * 64, source_commit="abc"))
        sequence = dict(arms={"M1": dict(summary=dict(input_output_digests=[["in", "out"]]))},
                        diagnostic_replays={"M1": dict(status="qualified", input_output_digests=[["in", "out"]],
                                                       physical=physical)})
        self.data = dict(schema="v20_direct_full_forward_v1", counter_twins=True,
                         runtime_identity=dict(hostname="h1", gpu_uuid="GPU-1"),
                         selected_boundaries=["model_forward"], selected_sequence_lengths=[4],
                         source_sha256={"counter.py": "b" * 64}, arms=[arm],
                         targets={"one": dict(dataset="aime26", boundaries=dict(model_forward=dict(N4=sequence)))})
        self.profile.write_text(json.dumps(self.data))
        self.cost.write_text(json.dumps(dict(no_quality_based_selection=True, profiles=[digest(self.profile)],
                                             rows=[dict(profile_sha256=digest(self.profile), dataset="aime26",
                                                        arm="M1", status="measured")])))

    def test_matched_twin_counts_and_source_binding(self):
        report = summarize({"h1": self.profile}, self.cost)
        self.assertEqual(report["group_count"], 1)
        row = report["groups"][0]
        self.assertEqual(row["status"], "qualified")
        self.assertEqual(row["phase_attention_calls"], {"D": 1})
        self.assertAlmostEqual(row["all_kinds"]["whole"]["qk_skipped_fraction"], .2)
        self.assertEqual(row["measured_cost_rows"], 1)
        self.assertEqual(report["profiles"][0]["source_sha256"]["counter.py"], "b" * 64)
        self.assertIn("no policy selection", render(report))

    def test_digest_drift_and_pair_conservation_fail_closed(self):
        self.data["targets"]["one"]["boundaries"]["model_forward"]["N4"]["diagnostic_replays"]["M1"]["input_output_digests"] = [["wrong", "out"]]
        self.profile.write_text(json.dumps(self.data))
        self.cost.write_text(json.dumps(dict(no_quality_based_selection=True, profiles=[digest(self.profile)], rows=[])))
        with self.assertRaisesRegex(ValueError, "input/output digests"):
            summarize({"h1": self.profile}, self.cost)
        with self.assertRaisesRegex(ValueError, "conservation"):
            _rates(counts(100, 20) | {"executed_qk_pairs": 79})


if __name__ == "__main__":
    unittest.main()
