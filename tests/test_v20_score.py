import unittest
import json
import tempfile
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v20_score import _first_receipt, cluster_interval, summarize_subset


def event(index, role, arm, qid="q1", seed=101, block=0):
    return dict(index=index, block=block, dataset="aime26", id=qid, seed=seed,
                role=role, repeat=0 if role == "attempt0" else 1, arm=arm,
                host="h1", gpu_uuid="GPU-1", cell_id=f"{qid}-{seed}-{arm}")


def record(spec, *, seconds=1.0, calls=4, ok=True):
    phase = dict(phase="native_dense_decoder", fresh_decoder_calls=calls,
                 per_canvas=[dict(decoder_calls=calls, iteration_cap=False)], initial_prefill_end_observed=True)
    base = dict(ok=ok, host="h1", gpu_uuid="GPU-1", termination="eos",
                completion_token_hash="h", per_canvas_calls=[calls], decoder_calls=calls,
                canvases=1, output_tokens=10, phase_evidence=phase,
                router_phase_evidence=dict(phase="native_dense"),
                triton_misses=0, triton_disk_entries_added=0, new_shared_objects=[],
                api_wall_s=seconds)
    if spec["role"] == "warm":
        base["acceptance"] = dict(accepted=True, reasons=[])
    return base


class ScoreTests(unittest.TestCase):
    def test_private_archive_host_remap_and_receipt_parity(self):
        from scripts.v18_protocol import sha
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cell = root / "cells" / "cell1"
            cell.mkdir(parents=True)
            receipt = dict(id="q1", seed=101, fingerprint="cfg", prompt_token_hash="prompt",
                           completion_tokens=[2, 3], per_canvas=[dict(decoder_calls=2)],
                           termination_reason="eos", total_decoder_calls=2, output_tokens=2)
            (cell / "attempt00.json").write_text(json.dumps(receipt))
            record = dict(host="h1", private_receipt="/remote/cells/cell1/attempt00.json",
                          fingerprint="cfg", prompt_token_hash="prompt",
                          completion_token_hash=sha(json.dumps([2, 3], separators=(",", ":"))), per_canvas_calls=[2],
                          termination="eos", decoder_calls=2, output_tokens=2)
            self.assertEqual(_first_receipt(record, dict(id="q1", seed=101, cell_id="cell1"), {"h1": root}), receipt)

    def test_question_cluster_keeps_seed_repeats_together(self):
        point, ci = cluster_interval({"q1": [0., 1.], "q2": [1., 1.]}, seed=9, resamples=500)
        self.assertAlmostEqual(point, .75)
        self.assertGreaterEqual(ci[0], .5)
        self.assertLessEqual(ci[1], 1.)

    def protocol(self):
        schedule = [event(0, "attempt0", "D_native"), event(1, "attempt0", "T_scope"),
                    event(2, "warm", "T_scope"), event(3, "warm", "D_native")]
        return dict(ids={"aime26": ["q1"]}, schedule=schedule)

    def test_same_host_complete_pair_ratio(self):
        p = self.protocol()
        records = {execution_key(e): record(e, seconds=0.8 if e["arm"] == "T_scope" else 1.0)
                   for e in p["schedule"]}
        # T phase must agree with its own first, but the native and T labels may differ.
        for e in p["schedule"]:
            if e["arm"] == "T_scope":
                records[execution_key(e)]["phase_evidence"]["phase"] = "fresh_T_decoder"
                records[execution_key(e)]["router_phase_evidence"] = dict(phase="fresh_T", attention_calls=4)
        quality = {"q1-101-D_native": dict(score=1., correct=True, parsed=True),
                   "q1-101-T_scope": dict(score=0., correct=False, parsed=True)}
        summary = summarize_subset(p, records, quality, first84=False)
        ratio = summary["datasets"]["aime26"]["paired_request_time_ratios"]["T_scope/D_native"]
        self.assertEqual(ratio["paired_cells"], 1)
        self.assertAlmostEqual(ratio["geometric_ratio"], .8)
        self.assertEqual(summary["datasets"]["aime26"]["arms"]["T_scope"]["score_mean"], 0.)

    def test_missing_warm_never_implies_speed(self):
        p = self.protocol()
        first = [e for e in p["schedule"] if e["role"] == "attempt0"]
        records = {execution_key(e): record(e) for e in first}
        quality = {e["cell_id"]: dict(score=1., correct=True, parsed=True) for e in first}
        summary = summarize_subset(p, records, quality, first84=False)
        ratio = summary["datasets"]["aime26"]["paired_request_time_ratios"]["T_scope/D_native"]
        self.assertEqual(ratio["paired_cells"], 0)
        self.assertIsNone(ratio["geometric_ratio"])
        self.assertFalse(summary["executions_complete"])

    def test_wrong_gpu_pair_rejected(self):
        p = self.protocol()
        records = {execution_key(e): record(e) for e in p["schedule"]}
        for e in p["schedule"]:
            if e["arm"] == "T_scope":
                records[execution_key(e)]["phase_evidence"]["phase"] = "fresh_T_decoder"
                records[execution_key(e)]["router_phase_evidence"] = dict(phase="fresh_T", attention_calls=4)
        first_t = p["schedule"][1]
        records[execution_key(first_t)]["host"] = "h2"
        quality = {e["cell_id"]: dict(score=1., correct=True, parsed=True)
                   for e in p["schedule"] if e["role"] == "attempt0"}
        with self.assertRaisesRegex(ValueError, "crossed GPUs"):
            summarize_subset(p, records, quality, first84=False)


if __name__ == "__main__":
    unittest.main()
