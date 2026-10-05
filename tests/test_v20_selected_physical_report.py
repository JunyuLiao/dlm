import json
import tempfile
import unittest
from pathlib import Path

from scripts.v20_selected_physical_report import _operator, _physical, csv_text, summarize


def counts():
    return dict(eligible_pairs=100, skipped_qk_pairs=20, executed_qk_pairs=80,
                skipped_pv_pairs=25, executed_pv_pairs=75,
                executed_qk_multiply_accumulates=80, executed_pv_multiply_accumulates=75)


class SelectedPhysicalTests(unittest.TestCase):
    def test_physical_twin_age_and_operator_scope(self):
        row = dict(phase="D", kind="global", layer=5, numeric_age=2, decision_age=1,
                   by_segment={s: counts() for s in ("whole", "static_prefix", "current_canvas")})
        physical = dict(calls=1, rows=[row], accepted_timing=False, projection_skipping_measured=False,
                        by_phase_kind={"D/global": {s: counts() for s in row["by_segment"]}})
        diagnostic = dict(status="qualified", input_output_digests=[["in", "out"]], physical=physical)
        summary = _physical(diagnostic, dict(summary=dict(input_output_digests=[["in", "out"]])))
        self.assertEqual(summary["phase_attention_calls"], dict(A=0, D=1, H=0))
        self.assertEqual(summary["numeric_age"], dict(min=2, max=2))
        self.assertAlmostEqual(summary["all_kinds"]["whole"]["qk_skipped_fraction"], .2)
        probe = dict(status="qualified", rows=[dict(layer=5, actual_output_max_abs_error=.1,
            actual_output_relative_l2_error=.01, prepared_consumer_event_median_ms=.2,
            v11_envelope_pass=True)])
        self.assertEqual(_operator(probe)["by_layer"][0]["layer"], 5)
        probe["rows"][0]["layer"] = 7
        with self.assertRaisesRegex(ValueError, "layers 0/5"):
            _operator(probe)
        diagnostic["input_output_digests"] = [["wrong", "out"]]
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            _physical(diagnostic, dict(summary=dict(input_output_digests=[["in", "out"]])))

    def test_ruler_missing_n16_preserved_and_aliases_not_pooled(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for stage in ("selected_001", "ruler_selected_001"):
                for host in ("mpk", "dllm"):
                    arms = [dict(name=f"A{i}", config=dict(v20_scope="GLOBAL_ONLY_NATIVE_LOCAL",
                                                       policy_name="P0", policy_sha256="a" * 64,
                                                       source_commit="cp3")) for i in range(8)]
                    targets = {}
                    for call in (0, 2):
                        seq = None
                        if stage == "ruler_selected_001":
                            row = dict(phase="A", kind="global", layer=5, numeric_age=0, decision_age=0,
                                       by_segment={s: counts() for s in ("whole", "static_prefix", "current_canvas")})
                            physical = dict(calls=1, rows=[row], accepted_timing=False,
                                projection_skipping_measured=False,
                                by_phase_kind={"A/global": {s: counts() for s in row["by_segment"]}})
                            seq = dict(reached_calls=4, requested_calls=4,
                                arms={f"A{i}": dict(summary=dict(input_output_digests=[["in", "out"]])) for i in range(8)},
                                diagnostic_replays={f"A{i}": (dict(status="qualified",
                                    input_output_digests=[["in", "out"]], physical=physical,
                                    operator_probe=dict(status="N/A")) if i == 0 else
                                    dict(status="N/A", reason="no numerical router")) for i in range(8)})
                        targets[str(call)] = dict(dataset="ruler4k", id="gold-free-question", canvas=0,
                            requested_call=call, resolution=dict(selected_call=call, fallback_used=False),
                            boundaries=({b: {"N4": seq} for b in ("model_forward", "denoising_step")}
                                        if seq else {}))
                    profile = dict(schema="v20_direct_full_forward_v1", counter_twins=True,
                        runtime_identity=dict(hostname=host, gpu_uuid=f"GPU-{host}"),
                        selected_boundaries=["model_forward", "denoising_step"],
                        selected_sequence_lengths=[16 if stage == "selected_001" else 4],
                        source_sha256={"v20_counter.py": "b" * 64}, arms=arms, targets=targets)
                    path = Path(folder) / f"{stage}-{host}.json"
                    path.write_text(json.dumps(profile))
                    paths[stage, host] = path
            report = summarize(paths)
            self.assertEqual(report["missing_sequence_rows"], 32)
            self.assertEqual(report["measured_rows"], 4)
            self.assertEqual(report["distinct_captured_state_keys"], 2)
            self.assertEqual({r["same_canvas_window_count"] for r in report["rows"]}, {2})
            self.assertEqual({tuple(x["requested_call"] for x in r["requested_call_aliases"])
                              for r in report["rows"]}, {(0, 2)})
            self.assertIn("numeric_age_min", csv_text(report))
            path = paths["ruler_selected_001", "mpk"]
            changed = json.loads(path.read_text())
            changed["targets"]["2"]["boundaries"]["model_forward"]["N4"]["diagnostic_replays"]["A0"]["operator_probe"]["reason"] = "different replay"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "alias has different diagnostic evidence"):
                summarize(paths)


if __name__ == "__main__":
    unittest.main()
