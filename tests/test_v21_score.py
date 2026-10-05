import json
import tempfile
import unittest
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v18_protocol import sha
from scripts.v21_run import strict_v21_warm
from scripts.v21_score import (build_cells, load_records, paired_ratios, score, summarize,
                               write_redacted, _stop_counts)
from tests.test_v21_run import diagnostic_panel, panel


def first_record():
    phase = dict(phase="v21_current_output_decoder", fresh_decoder_calls=3,
                 per_canvas=[dict(decoder_calls=3, native_stop=True, iteration_cap=False)],
                 initial_prefill_end_observed=True, prefill_end_to_finish_gpu_s=1.2)
    return dict(ok=True, completion_token_hash="token-hash", per_canvas_calls=[3], termination="eos",
                decoder_calls=3, canvases=1, output_tokens=8, api_wall_s=2., phase_evidence=phase,
                router_phase_evidence=dict(phase="M1_M3_cached", A=1, D=1, H=1, attention_calls=3),
                host="hostA", gpu_uuid="gpuA")


def quality():
    return dict(score=1., correct=True, task_correct=True, strict_correct=True,
                parsed=True, capped=False, eos=True)


class V21ScoreTests(unittest.TestCase):
    def test_eligible_partial_keeps_first_quality_and_only_strict_warm_time(self):
        p = panel()
        first_spec, warm_spec = p["schedule"][0], p["schedule"][3]
        self.assertEqual((first_spec["arm"], first_spec["cell_id"]), (warm_spec["arm"], warm_spec["cell_id"]))
        first = first_record()
        warm = dict(first, triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[])
        warm["acceptance"] = strict_v21_warm(first, warm)
        rows = {execution_key(first_spec): first, execution_key(warm_spec): warm}
        q = {first_spec["cell_id"]: quality()}
        result = summarize(p, rows, q, protocol_sha="a" * 64, binding_sha="b" * 64)
        self.assertFalse(result["executions_complete"])
        self.assertIn(0, result["block_completeness"]["partial_block_ids"])
        group = next(g for g in result["groups"] if g["dataset"] == first_spec["dataset"] and
                     g["host"] == "hostA" and g["arm"] == first_spec["arm"])
        self.assertEqual((group["first_success"], group["scored_first"], group["warm_accepted"]), (1, 1, 1))
        self.assertEqual(group["exact_work"]["pooled_calls_per_canvas"], 3.)
        self.assertEqual(group["warm_whole_request_s"]["mean"], 2.)
        self.assertEqual(group["stop_categories"], dict(native_only=1, cap_only=0, both=0,
                                                         neither=0, unrecorded=0))
        self.assertEqual(group["router_phase_layer_calls"]["A"], 1)
        self.assertEqual(group["output_tokens_by_termination"]["eos"]["mean"], 8)
        self.assertEqual(group["unique_question_count"], 4)
        self.assertEqual(len(result["cells"]), 48)
        success_row = next(r for r in result["cells"] if r["cell_id"] == first_spec["cell_id"])
        self.assertEqual((success_row["first_status"], success_row["warm_status"],
                          success_row["first_score"]), ("success", "accepted", 1.))
        self.assertEqual(sum(r["first_status"] == "missing" for r in result["cells"]), 47)
        with self.assertRaisesRegex(ValueError, "score inventory"):
            summarize(p, rows, {}, protocol_sha="a" * 64, binding_sha="b" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "redacted"
            write_redacted(result, prefix)
            self.assertTrue(prefix.with_suffix(".json").exists())
            self.assertNotIn("token-hash", prefix.with_suffix(".json").read_text())
            self.assertNotIn("prompt", prefix.with_suffix(".csv").read_text())
            self.assertEqual(len(prefix.with_suffix(".csv").read_text().splitlines()), 49)
            self.assertTrue(prefix.with_name(prefix.name + ".groups.csv").exists())

    def test_rejected_warm_never_enters_time_or_pair(self):
        p = panel()
        first_spec, warm_spec = p["schedule"][0], p["schedule"][3]
        first = first_record()
        warm = dict(first, triton_misses=1, triton_disk_entries_added=0, new_shared_objects=[])
        warm["acceptance"] = strict_v21_warm(first, warm)
        rows = {execution_key(first_spec): first, execution_key(warm_spec): warm}
        result = summarize(p, rows, {first_spec["cell_id"]: quality()},
                           protocol_sha="a" * 64, binding_sha="b" * 64)
        group = next(g for g in result["groups"] if g["arm"] == first_spec["arm"] and
                     g["dataset"] == first_spec["dataset"] and g["host"] == "hostA")
        self.assertEqual(group["scored_first"], 1)
        self.assertEqual(group["warm_accepted"], 0)
        self.assertIsNone(group["warm_whole_request_s"])
        self.assertIn(0, result["block_completeness"]["failed_block_ids"])

    def test_same_gpu_pair_uses_accepted_warm_and_first_work(self):
        p = panel()
        specs = p["schedule"][:4]
        records, scores = {}, {}
        for j, first_spec in enumerate(specs[:2]):
            first = first_record()
            first["decoder_calls"] = 3 * (j + 1)
            first["per_canvas_calls"] = [3 * (j + 1)]
            first["phase_evidence"] = dict(first["phase_evidence"], fresh_decoder_calls=3 * (j + 1),
                                           per_canvas=[dict(decoder_calls=3 * (j + 1), native_stop=True,
                                                            iteration_cap=False)])
            first["router_phase_evidence"] = dict(phase="M1_M3_cached", A=j + 1, D=j + 1,
                                                   H=j + 1, attention_calls=3 * (j + 1))
            first["api_wall_s"] = 2. * (j + 1)
            warm_spec = next(e for e in specs[2:] if e["cell_id"] == first_spec["cell_id"])
            warm = dict(first, triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[])
            warm["acceptance"] = strict_v21_warm(first, warm)
            records[execution_key(first_spec)] = first
            records[execution_key(warm_spec)] = warm
            scores[first_spec["cell_id"]] = quality()
        cells = build_cells(p, records, scores)
        ratios = paired_ratios(p, cells)
        pair = next(r for r in ratios if r["dataset"] == specs[0]["dataset"] and
                    r["arm"] == specs[1]["arm"] and r["reference"] == specs[0]["arm"])
        self.assertEqual(pair["paired_accepted_warm_cells"], 1)
        self.assertEqual(pair["by_host"]["hostA"]["warm_wall_ratio_of_totals"], 2.)
        self.assertEqual(pair["by_host"]["hostA"]["decoder_call_ratio_of_totals"], 2.)

    def test_diagnostic_is_ineligible_and_first_only(self):
        p = diagnostic_panel()
        spec = p["schedule"][0]
        first = first_record()
        first["phase_evidence"] = None  # diagnostic timing-events-off receipt has no CUDA phase span
        first["per_canvas_stopping"] = [{"native_stop": True, "iteration_cap": False}]
        rows = {execution_key(spec): first}
        result = summarize(p, rows, {}, protocol_sha="a" * 64, binding_sha="b" * 64)
        self.assertFalse(result["quality_eligible"])
        self.assertFalse(result["timing_eligible"])
        self.assertEqual(result["paired_ratios"], [])
        group = next(g for g in result["groups"] if g["arm"] == spec["arm"] and g["host"] == "hostA")
        self.assertIsNone(group["strict_correct"])
        self.assertIsNone(group["warm_accepted"])
        self.assertEqual(group["stop_categories"]["native_only"], 1)
        self.assertEqual(group["stop_categories"]["unrecorded"], 0)
        with self.assertRaisesRegex(ValueError, "may not enter"):
            summarize(p, rows, {spec["cell_id"]: quality()}, protocol_sha="a" * 64,
                      binding_sha="b" * 64)
        self.assertEqual(_stop_counts({"canvases": 1, "phase_evidence": None})["unrecorded"], 1)

    def test_ledger_identity_and_missing_failure_preserved(self):
        p = diagnostic_panel()
        spec = p["schedule"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = root / "binding.json"
            binding.write_text("{}")
            ledger = root / "ledger.jsonl"
            start = dict(event="start", protocol_id=p["protocol_id"],
                         binding_sha256=sha(binding.read_bytes()), model_revision=p["model_revision"],
                         host=spec["host"], gpu_uuid=spec["gpu_uuid"])
            row = dict(spec, event="run", execution_key=execution_key(spec), generation_seed=spec["seed"],
                       quality_eligible=False, timing_eligible=False, ok=False, error="test failure")
            end = dict(event="worker_end", host=spec["host"], gpu_uuid=spec["gpu_uuid"])
            ledger.write_text("\n".join(json.dumps(x) for x in (start, row, end)) + "\n")
            records = load_records(p, binding, [ledger])
            self.assertEqual(records[execution_key(spec)]["error"], "test failure")
            self.assertEqual(len(records), 1)
            ledger.write_text("\n".join(json.dumps(x) for x in (start, row)) + "\n")
            with self.assertRaisesRegex(ValueError, "remains open"):
                load_records(p, binding, [ledger])


if __name__ == "__main__":
    unittest.main()
