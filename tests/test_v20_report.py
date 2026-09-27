import json
import tempfile
import unittest
from pathlib import Path

from scripts.v20_report import _phase, canonical_arm, extract_profile, render


def sample(event_sum, epoch, phase=None, jit=0):
    return dict(summary=dict(event_sum_median_ms=event_sum, wall_sum_median_ms=event_sum + 1,
                             per_call_event_median_ms=[event_sum / 4] * 4,
                             phase_deltas=phase or [{}] * 4),
                direct_epoch=dict(event_median_ms=epoch, wall_median_ms=epoch + 1),
                blocks=[dict(block=i, triton_misses=jit, triton_specializations_before={},
                             triton_specializations_after={}, triton_disk_entries_before=0,
                             triton_disk_entries_after=0, peak_allocated_bytes=2 * 1024**3) for i in range(3)])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "profile.json"
        self.profile = dict(schema="v20_direct_full_forward_v1",
                            selected_boundaries=["model_forward"], selected_sequence_lengths=[4],
                            runtime_identity=dict(hostname="mpk", gpu_uuid="GPU-123", gpu="H100",
                                                  torch="2.6.0+cu124"),
                            arms=[dict(name="D_native", condition="native_dense", config={}),
                                  dict(name="M1_R1_A8_current_output", condition="M1",
                                       plugin="experiments.numerical_qk_reuse.v20:install",
                                       config=dict(v20_arm="M1_R1_A8_current_output", condition="M1",
                                                   v20_scope="ALL_NATIVE_LEGAL", policy_sha256="a" * 64))],
                            targets={"one": dict(dataset="aime26", id="aime26/2", canvas=0, requested_call=1,
                                                  resolution=dict(missing=False, selected_call=1, fallback_used=False),
                                                  boundaries=dict(model_forward=dict(N4=dict(reached_calls=4, requested_calls=4,
                                                      native_bracket_drift=[dict(open_median_ms=10, close_median_ms=11)],
                                                      arms={"D_native": sample(40, 38),
                                                            "M1_R1_A8_current_output": sample(32, 30, [
                                                                dict(attention_calls=30, score_refresh_calls=30, decision_refresh_calls=30, held_decision_calls=0),
                                                                dict(attention_calls=30, score_refresh_calls=0, decision_refresh_calls=30, held_decision_calls=0),
                                                                dict(attention_calls=30, score_refresh_calls=0, decision_refresh_calls=0, held_decision_calls=30),
                                                                dict(attention_calls=30, score_refresh_calls=0, decision_refresh_calls=0, held_decision_calls=30)])}))))})
        self.path.write_text(json.dumps(self.profile))

    def test_direct_calls_epoch_phase_and_ratios_remain_distinct(self):
        rows = extract_profile(self.path)
        self.assertEqual(len(rows), 2)
        m1 = next(r for r in rows if r.get("arm", "").startswith("M1"))
        self.assertEqual(m1["status"], "measured")
        self.assertEqual((m1["host"], m1["gpu_uuid"]), ("mpk", "GPU-123"))
        self.assertEqual(m1["canonical_arm"], "M1_R1_A8_current_output")
        self.assertAlmostEqual(m1["within_gpu_complete_call_ratio_to_native"], .8)
        self.assertAlmostEqual(m1["within_gpu_direct_epoch_ratio_to_native"], 30 / 38)
        self.assertEqual(m1["phase_complete_call_costs"]["H"]["calls"], 2)
        self.assertEqual(m1["phase_complete_call_costs"]["A"]["calls"], 1)
        self.assertEqual(m1["phase_complete_call_costs"]["D"]["calls"], 1)
        self.assertAlmostEqual(m1["native_bracket_max_abs_drift"], .1)
        text = render(rows)
        self.assertIn("not natural request speedups", text)
        self.assertIn("32.000", text)
        self.assertIn("30.000", text)

    def test_mixed_layer_counts_and_control_phases(self):
        mixed = dict(attention_calls=30, score_refresh_calls=5,
                     decision_refresh_calls=20, held_decision_calls=10)
        self.assertEqual(_phase(mixed, "M1_R1_A8_current_output"), "MIXED")
        self.assertEqual(_phase(dict(attention_calls=30), "T_scope"), "FRESH")
        self.assertEqual(_phase(dict(attention_calls=30, bootstrap_calls=30,
                                     bitmap_observation_calls=0, held_decision_calls=0), "D_matched"), "FRESH")
        self.assertEqual(_phase(dict(attention_calls=30, bootstrap_calls=0,
                                     bitmap_observation_calls=30, held_decision_calls=0), "G75L30_nativeQ128"), "A")
        self.assertEqual(_phase(dict(attention_calls=30, bootstrap_calls=0,
                                     bitmap_observation_calls=0, held_decision_calls=30), "G75L30_nativeQ128"), "H")
        self.assertIsNone(_phase(dict(attention_calls=30), "D_native"))
        arm = self.profile["targets"]["one"]["boundaries"]["model_forward"]["N4"]["arms"]["M1_R1_A8_current_output"]
        arm["summary"]["phase_deltas"][1] = mixed
        self.path.write_text(json.dumps(self.profile))
        row = next(r for r in extract_profile(self.path) if r.get("arm", "").startswith("M1"))
        self.assertEqual(row["phase_complete_call_costs"]["MIXED"]["calls"], 1)
        self.assertEqual(row["phase_complete_call_costs"]["MIXED"]["counter_totals"]["score_refresh_calls"], 5)
        self.assertIn("MIXED:", render([row]))

    def test_prefixed_screen_names_use_frozen_method_identity(self):
        method = dict(name="GLOBAL_ONLY_NATIVE_LOCAL_P1_M3_R3_A8_current_output",
                      condition="v20_global_M3_R3",
                      plugin="experiments.numerical_qk_reuse.v20:install",
                      config=dict(v20_arm="M3_R3_A8_current_output",
                                  condition="v20_global_M3_R3"))
        anchor = dict(attention_calls=5, score_refresh_calls=5,
                      decision_refresh_calls=5, held_decision_calls=0)
        held = dict(attention_calls=5, score_refresh_calls=0,
                    decision_refresh_calls=0, held_decision_calls=5)
        self.assertEqual(canonical_arm(method), "M3_R3_A8_current_output")
        self.assertEqual(_phase(anchor, method), "A")
        self.assertEqual(_phase(held, method), "H")
        b = dict(method, name="ALL_NATIVE_LEGAL_P0_B_A8_matched",
                 condition="M3", config=dict(v20_arm="B_A8_matched", condition="M3"))
        self.assertEqual(_phase(held | {'attention_calls': 30, 'held_decision_calls': 30}, b), "H")
        dense = dict(name="ALL_NATIVE_LEGAL_D_matched", condition="v20_dense_consumer",
                     plugin="experiments.numerical_qk_reuse.v20_controls:install", config={})
        self.assertEqual(_phase(dict(attention_calls=30, bootstrap_calls=30,
                                     bitmap_observation_calls=0, held_decision_calls=0), dense), "FRESH")
        self.assertIsNone(canonical_arm(dict(name="M3_fake", condition="unknown", config={})))

    def test_new_jit_disables_ratio(self):
        self.profile["targets"]["one"]["boundaries"]["model_forward"]["N4"]["arms"]["M1_R1_A8_current_output"]["blocks"][0]["triton_misses"] = 1
        self.path.write_text(json.dumps(self.profile))
        m1 = next(r for r in extract_profile(self.path) if r.get("arm", "").startswith("M1"))
        self.assertEqual(m1["status"], "invalid_new_jit_or_missing_blocks")
        self.assertIsNone(m1["within_gpu_complete_call_ratio_to_native"])

    def test_native_new_jit_disables_all_relative_ratios(self):
        self.profile["targets"]["one"]["boundaries"]["model_forward"]["N4"]["arms"]["D_native"]["blocks"][0]["triton_misses"] = 1
        self.path.write_text(json.dumps(self.profile))
        m1 = next(r for r in extract_profile(self.path) if r.get("arm", "").startswith("M1"))
        self.assertEqual(m1["status"], "measured")
        self.assertIsNone(m1["within_gpu_complete_call_ratio_to_native"])

    def test_missing_state_kept_missing(self):
        self.profile["targets"]["one"]["resolution"]["missing"] = True
        self.path.write_text(json.dumps(self.profile))
        rows = extract_profile(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "missing_native_state")


if __name__ == "__main__":
    unittest.main()
