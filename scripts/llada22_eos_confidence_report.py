"""Audit the retired LLaDA EOS-confidence ablation against saved baselines."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from scripts.v27_humaneval import extract_code
from scripts.llada22_v2_humaneval_diagnose import token_metrics


NEW_ARMS = ("dense_native_eos_conf0",
            "value_v2_gaussian32_reuse_r4_eos_conf0")
BASE_ARMS = ("dense_native", "allkept_fa4", "mass_topk_reuse_r4",
             "value_v2_gaussian32_reuse_r4", "value_v3a_singleton_reuse_r4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cluster_ci(values, seed=738, repeats=10000):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(len(x), size=(repeats, len(x)))].mean(1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def describe(rows, private_by_key, gold_by_id):
    counts = collections.Counter()
    lengths = []
    complete = collections.Counter()
    outputs = {}
    score = 0
    for row in rows:
        counts.update(row["counts"])
        length = row["output_tokens"]
        lengths.append(length)
        score += row["quality"]
        outputs[row["id"]] = row["quality"]
        private = private_by_key[row["id"], row["seed"]]
        _, reason = extract_code("<channel|>" + private["text"],
                                 gold_by_id[row["id"]]["entry_point"])
        complete[reason] += 1
        tokens = token_metrics(private["text"], gold_by_id[row["id"]]["entry_point"])
        for key in ("entry_point_present", "entry_point_docstring_only",
                    "entry_point_stub_only", "entry_point_has_substantive_body"):
            complete[key] += tokens[key]
        eos = row["finished"]
        complete["eos_finished"] += bool(eos)
        complete["cap_hits"] += bool(not eos and length == row["generation_cap"])
        complete["length_le_64"] += length <= 64
        complete["length_le_128"] += length <= 128
        complete["length_gt_192"] += length > 192
    eligible = counts["eligible_tiles"]
    kept = counts["kept_tiles"]
    n, c = counts["denoising_forwards"], counts["canvases"]
    return {
        "requests": len(rows), "correct": score, "accuracy": score / len(rows),
        "mean_output_tokens": statistics.mean(lengths),
        "median_output_tokens": statistics.median(lengths),
        "max_output_tokens": max(lengths),
        "N": n, "C": c, "N_per_request": n / len(rows),
        "C_per_request": c / len(rows), "N_per_C": n / c,
        "T": sum(lengths), "consumer_sparsity": 1 - kept / eligible,
        "eligible_tiles": eligible, "kept_tiles": kept,
        "selection_layer_calls": counts["selection_layer_calls"],
        "held_layer_calls": counts["held_layer_calls"],
        "completeness": dict(complete),
        "quality_by_id": outputs,
    }


def paired(a, b):
    ids = sorted(set(a["quality_by_id"]) & set(b["quality_by_id"]))
    delta = [a["quality_by_id"][i] - b["quality_by_id"][i] for i in ids]
    return {"questions": len(ids), "wins": sum(d > 0 for d in delta),
            "losses": sum(d < 0 for d in delta), "ties": sum(d == 0 for d in delta),
            "accuracy_delta": statistics.mean(delta),
            "accuracy_delta_ci95": cluster_ci(delta)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    base, run, out = a.base.resolve(), a.run.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "receipt.json").exists() or (out / "final_complete.json").exists():
        raise RuntimeError("report output already finalized")
    errors = []
    marker = json.loads((run / "generation_complete.json").read_text())
    assert marker["status"] == "complete" and marker["requests_per_arm"] == 164
    protocol = json.loads((run / "protocol.json").read_text())
    if protocol["runner_sha256"] != sha(Path(__file__).with_name("llada22_eos_confidence_eval.py")):
        errors = ["runner source hash differs from frozen validation protocol"]
    else:
        errors = []
    dataset = json.loads((base / "private/data/llada22/humaneval.json").read_text())
    expected = {(r["id"], seed) for r in dataset for seed in r["seeds"]}
    if len(expected) != 164:
        errors.append("base HumanEval cohort is not 164 seed-paired items")
    gold_by_id = {r["id"]: r["gold"] for r in dataset}
    baseline_rows = read_jsonl(base / "analysis_scope30_20261102_attempt001/scored_records.jsonl")
    baseline = {arm: [r for r in baseline_rows if (r["model"], r["suite"], r["arm"]) ==
                       ("llada22", "humaneval", arm)] for arm in BASE_ARMS}
    new_analysis = read_jsonl(run / "analysis_eos_conf0_20261102_attempt001/scored_records.jsonl")
    new = {arm: [r for r in new_analysis if (r["model"], r["suite"], r["arm"]) ==
                 ("llada22", "humaneval", arm)] for arm in NEW_ARMS}
    summaries = {}
    raw_by_arm = {}
    for arm, rows in {**baseline, **new}.items():
        raw_root = base if arm in BASE_ARMS else run
        raw_arm = arm
        raw_rows = [r for path in (raw_root / "runs/llada22/humaneval" / raw_arm).glob(
            "shard*/attempt001/private/generation.private.jsonl")
                    for r in read_jsonl(path)]
        private_by_key = {(r["id"], r["seed"]): r for r in raw_rows}
        if set(private_by_key) != expected:
            errors.append(f"private raw coverage {arm}: {len(private_by_key)}/{len(expected)}")
        keys = {(r["id"], r["seed"]) for r in rows}
        if len(rows) != 164 or keys != expected:
            errors.append(f"coverage {arm}: {len(rows)}/{len(expected)}")
        summaries[arm] = describe(rows, private_by_key, gold_by_id)
        if arm in NEW_ARMS and summaries[arm]["completeness"]["eos_finished"] != 164:
            errors.append(f"EOS stop count changed unexpectedly: {arm}")
        raw_by_arm[arm] = {(r["id"], r["seed"]): r for r in rows}
    for new_arm in NEW_ARMS:
        old_arm = "dense_native" if new_arm.startswith("dense_") else BASE_ARMS[3]
        for key in sorted(expected):
            left, right = raw_by_arm[new_arm][key], raw_by_arm[old_arm][key]
            for field in ("prompt_sha256", "prompt_tokens", "generation_cap"):
                if left[field] != right[field]:
                    errors.append(f"paired identity {new_arm}/{key}/{field}")
    contrasts = {}
    comparisons = (
        (NEW_ARMS[0], "dense_native"),
        (NEW_ARMS[1], BASE_ARMS[3]),
        (NEW_ARMS[1], BASE_ARMS[2]),
        (NEW_ARMS[1], "dense_native"),
        (NEW_ARMS[1], NEW_ARMS[0]),
    )
    for x, y in comparisons:
        contrasts[f"{x}_vs_{y}"] = paired(summaries[x], summaries[y])

    # Reconcile the new scored rows to raw records and source-bound shard receipts.
    sources = []
    raw_zeroed_by_arm = collections.Counter()
    for marker_path in sorted((run / "runs/llada22/humaneval").glob(
            "*/shard*/attempt001/final_complete.json")):
        leaf = marker_path.parent
        config = json.loads((leaf / "config.json").read_text())
        receipt = json.loads((leaf / "receipt.json").read_text())
        raw_rows = read_jsonl(leaf / "records.jsonl")
        raw_zeroed_by_arm[config["arm"]] += sum(r["eos_confidences_zeroed"] for r in raw_rows)
        if receipt["records_sha256"] != sha(leaf / "records.jsonl"):
            errors.append(f"record hash {leaf}")
        scored_by_key = {(r["id"], r["seed"]): r for r in new[config["arm"]]}
        for raw in raw_rows:
            scored_row = scored_by_key.get((raw["id"], raw["seed"]))
            if scored_row is None:
                errors.append(f"missing scored row {leaf}/{raw['id']}")
                continue
            for field in ("counts", "output_tokens", "output_sha256", "prompt_sha256",
                          "prompt_tokens", "finished", "generation_cap"):
                if raw[field] != scored_row[field]:
                    errors.append(f"raw/scored mismatch {leaf}/{raw['id']}/{field}")
        sources.append({"arm": config["arm"], "path": str(leaf), "config_sha256": sha(leaf / "config.json"),
                        "records_sha256": sha(leaf / "records.jsonl"),
                        "eos_confidences_zeroed": receipt["eos_confidences_zeroed"]})
    for arm in NEW_ARMS:
        receipted_zeroed = sum(s["eos_confidences_zeroed"] for s in sources if s["arm"] == arm)
        if raw_zeroed_by_arm[arm] != receipted_zeroed or receipted_zeroed <= 0:
            errors.append(f"EOS confidence counters inconsistent: {arm}")
    if len(sources) != 8:
        errors.append("expected eight complete 41-cell shards")
    output = {
        "status": "pass" if not errors else "fail", "errors": errors,
        "source_sha256": sha(__file__), "run_protocol_sha256": sha(run / "protocol.json"),
        "run_generation_complete_sha256": sha(run / "generation_complete.json"),
        "official_scoring_identity": json.loads((run / "analysis_eos_conf0_20261102_attempt001/scoring_identity.json").read_text()),
        "source_shards": sources, "summaries": summaries, "paired": contrasts,
        "run_arms": list(NEW_ARMS),
        "sparsity_target": [.48, .52],
        "sparsity_target_met": {arm: (.48 <= summaries[arm]["consumer_sparsity"] <= .52
                                       if "reuse" in arm else None)
                                for arm in NEW_ARMS},
        "no_threshold_recalibration": True,
        "bootstrap": {"resamples": 10000, "seed": 738, "unit": "HumanEval problem"},
    }
    (out / "receipt.json").write_text(json.dumps(output, indent=2) + "\n")
    summary = ["# LLaDA2.2 HumanEval EOS-confidence validation", "",
               "Output token IDs and logits are unchanged by the intervention; only confidence-based selection zeros the confidence for predicted EOS. Native forced progress and EOS termination remain enabled.", "",
               "| Arm | Correct /164 | Mean T | Median T | Max T | EOS finished | Cap hits | N/request | C/request | N/C | Consumer sparsity | Syntax errors | Parsed funcs | Substantive funcs | EOS confidence zeros |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, result in summaries.items():
        comp = result["completeness"]
        zeros = sum(x["eos_confidences_zeroed"] for x in sources if x["arm"] == arm)
        summary.append(f"| {arm} | {result['correct']}/164 ({100*result['accuracy']:.2f}%) | {result['mean_output_tokens']:.1f} | {result['median_output_tokens']:.1f} | {result['max_output_tokens']} | {comp['eos_finished']} | {comp['cap_hits']} | {result['N_per_request']:.1f} | {result['C_per_request']:.2f} | {result['N_per_C']:.3f} | {100*result['consumer_sparsity']:.2f}% | {comp['syntax_error']} | {comp['entry_point_present']} | {comp['entry_point_has_substantive_body']} | {zeros} |")
    summary.extend(["", "Paired accuracy contrasts (difference and 95% problem-bootstrap CI):"])
    for name in (f"{NEW_ARMS[0]}_vs_dense_native",
                 f"{NEW_ARMS[1]}_vs_value_v2_gaussian32_reuse_r4",
                 f"{NEW_ARMS[1]}_vs_{NEW_ARMS[0]}",
                 f"{NEW_ARMS[1]}_vs_mass_topk_reuse_r4",
                 f"{NEW_ARMS[1]}_vs_dense_native"):
        c = contrasts[name]
        lo, hi = c["accuracy_delta_ci95"]
        summary.append(f"- `{name}`: {100*c['accuracy_delta']:+.2f} percentage points [{100*lo:+.2f}, {100*hi:+.2f}], {c['wins']} wins / {c['losses']} losses / {c['ties']} ties.")
    dense_new = summaries[NEW_ARMS[0]]
    v2_new = summaries[NEW_ARMS[1]]
    v2_old = summaries[BASE_ARMS[3]]
    dense_old = summaries["dense_native"]
    v2_contrast = contrasts[f"{NEW_ARMS[1]}_vs_{BASE_ARMS[3]}"]
    dense_contrast = contrasts[f"{NEW_ARMS[0]}_vs_dense_native"]
    summary.extend([
        "",
        "The EOS-confidence-zero column counts EOS predictions whose confidence value was zeroed across all model sampling calls; it is not the number of EOS tokens emitted in final answers.",
        "",
        f"V2 changes from {v2_old['correct']:.0f}/164 at {v2_old['mean_output_tokens']:.1f} mean tokens, {v2_old['N_per_request']:.1f} N/request, and {100*v2_old['consumer_sparsity']:.2f}% sparsity to {v2_new['correct']:.0f}/164 at {v2_new['mean_output_tokens']:.1f} tokens, {v2_new['N_per_request']:.1f} N/request, and {100*v2_new['consumer_sparsity']:.2f}% sparsity. The paired improvement is {100*v2_contrast['accuracy_delta']:.2f}pp, but generation work rises substantially and realized sparsity moves outside the 48-52% target. This partially rescues V2 on this fixed threshold; it is not a matched-sparsity result or a complete repair.",
        f"Dense changes from {dense_old['correct']:.0f}/164 to {dense_new['correct']:.0f}/164 ({100*dense_contrast['accuracy_delta']:+.2f}pp), with mean output length {dense_old['mean_output_tokens']:.1f} vs {dense_new['mean_output_tokens']:.1f}. The larger gain is specific to the tested V2 trajectory, though the V2 sparsity/work shift prevents attributing it to EOS confidence alone at equal attention work.",
        "",
        "The confidence mask leaves EOS logits and native stopping enabled. EOS still ends all164 requests, so it delays/changes EOS selection rather than removing the stop token. Follow-up should calibrate this combined V2+mask policy on separate non-benchmark long-code prompts to restore 48-52% physical sparsity, then test on a fresh code-completion cohort or explicitly label any repeated HumanEval use as transductive. Include syntax/entry-point/completeness checks so sparsity calibration cannot select a short-answer failure.",
        "",
        "Intervals resample HumanEval problems (10,000 draws); one seed/problem means they quantify item uncertainty, not seed variability. The no-retuning, fixed-threshold panel is exploratory; do not claim matched-work superiority.", ""])
    (out / "summary.md").write_text("\n".join(summary))
    if errors:
        raise RuntimeError("EOS-confidence audit failed: " + repr(errors[:20]))
    (out / "final_complete.json").write_text(json.dumps({
        "status": "complete", "models": 1, "benchmarks": 1,
        "new_arms": 2, "requests": 328,
        "receipt_sha256": sha(out / "receipt.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
