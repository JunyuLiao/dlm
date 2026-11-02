"""Verify and summarize the user-retained scope and count-matched V2 reruns."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


ARM = "value_v2_gaussian32_reuse_r4"
ARMS = ("dense_native", "allkept_fa4", "mass_topk_reuse_r4", ARM,
        "value_v3a_singleton_reuse_r4")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def paired(rows, reference):
    ref = {(r["id"], r["seed"]): r for r in reference}
    clusters = collections.defaultdict(list)
    wins = losses = 0
    for r in rows:
        s = ref[r["id"], r["seed"]]
        delta = r["quality"] - s["quality"]
        wins += delta > 0
        losses += delta < 0
        n, c = r["counts"]["denoising_forwards"], r["counts"]["canvases"]
        on, oc = s["counts"]["denoising_forwards"], s["counts"]["canvases"]
        clusters[r["id"]].append((delta, n - on, c - oc,
                                 r["output_tokens"] - s["output_tokens"],
                                 np.log(n / on), np.log((n / c) / (on / oc))))
    m = np.array([np.mean(x, axis=0) for _, x in sorted(clusters.items())])
    rng = np.random.default_rng(738)
    b = m[rng.integers(len(m), size=(10000, len(m)))].mean(1)
    result = {"questions": len(m), "wins": int(wins), "losses": int(losses)}
    for i, name in enumerate(("accuracy_delta", "N_delta", "C_delta", "T_delta")):
        result[name] = float(m[:, i].mean())
        result[name + "_ci95"] = np.quantile(b[:, i], [.025, .975]).tolist()
    for i, name in ((4, "paired_N_ratio"), (5, "paired_N_per_C_ratio")):
        result[name] = float(np.exp(m[:, i].mean()))
        result[name + "_ci95"] = np.exp(np.quantile(b[:, i], [.025, .975])).tolist()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("rematch", type=Path)
    p.add_argument("--original-analysis", required=True)
    p.add_argument("--rerun-analysis", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    errors = []
    for model in ("llada22", "trado"):
        marker = a.rematch / model / "final_complete.json"
        if not marker.exists():
            raise RuntimeError("missing accepted fresh rerun: " + model)
        d = json.loads(marker.read_text())
        assert d["requests"] == 90 and .48 <= d["sparsity"] <= .52
    if not (a.rematch / "generation_complete.json").exists():
        raise RuntimeError("two-model generation not complete")
    orig_path = a.base / a.original_analysis / "scored_records.jsonl"
    rerun_path = a.rematch / a.rerun_analysis / "scored_records.jsonl"
    original = load_rows(orig_path)
    reruns = load_rows(rerun_path)
    if len(original) != 4010 or len(reruns) != 180:
        raise RuntimeError("scored record coverage mismatch")
    tables = collections.defaultdict(list)
    history = []
    for r in original:
        if r["suite"] == "aime26" and r["arm"] == ARM:
            history.append(r)
        else:
            tables[r["model"], r["suite"], r["arm"]].append(r)
    for r in reruns:
        assert r["suite"] == "aime26" and r["arm"] == ARM
        tables[r["model"], r["suite"], r["arm"]].append(r)
    expected = {(m, s, arm) for m, suites in
                (("llada22", ("aime26", "humaneval", "ruler128k")),
                 ("trado", ("aime26", "humaneval")), ("ultrallada", ("humaneval",)))
                for s in suites for arm in ARMS}
    if set(tables) != expected:
        errors.append("retained panel scope mismatch")
    summaries, contrasts, sources = {}, {}, []
    for key, rows in sorted(tables.items()):
        model, suite, arm = key
        count = 90 if suite == "aime26" else 164 if suite == "humaneval" else 130
        ids = {(r["id"], r["seed"]) for r in rows}
        if len(rows) != count or len(ids) != count:
            errors.append(f"coverage: {key}")
        totals = collections.Counter()
        for r in rows:
            totals.update(r["counts"])
        eligible, kept = totals["eligible_tiles"], totals["kept_tiles"]
        sparse = 1 - kept / eligible
        if "reuse" in arm and not .48 <= sparse <= .52:
            errors.append(f"sparsity target: {key}/{sparse}")
        if arm in ARMS[:2] and eligible != kept:
            errors.append(f"dense control skips: {key}")
        reference = {(r["id"], r["seed"]): r for r in tables[model, suite, "dense_native"]}
        for r in rows:
            s = reference[r["id"], r["seed"]]
            for field in ("prompt_sha256", "prompt_tokens", "generation_cap", "model_revision"):
                if r[field] != s[field]:
                    errors.append(f"fairness: {key}/{field}")
        label = "/".join(key)
        summaries[label] = {
            "requests": count, "accuracy": float(np.mean([r["quality"] for r in rows])),
            "full_credit": sum(r["quality"] == 1 for r in rows),
            "N": totals["denoising_forwards"], "C": totals["canvases"],
            "N_per_C": totals["denoising_forwards"] / totals["canvases"],
            "mean_request_N_per_C": float(np.mean([r["counts"]["denoising_forwards"] / r["counts"]["canvases"] for r in rows])),
            "T": sum(r["output_tokens"] for r in rows),
            "eligible_tiles": eligible, "kept_tiles": kept, "consumer_sparsity": sparse,
            "no_eos_at_cap": sum(not r["finished"] and r["output_tokens"] == r["generation_cap"] for r in rows),
            "counts": dict(totals), "transductively_recalibrated": suite == "aime26" and arm == ARM,
        }
        for ref_arm in ("mass_topk_reuse_r4", "dense_native", "allkept_fa4"):
            if arm != ref_arm:
                contrasts[label + "_vs_" + ref_arm] = paired(rows, tables[model, suite, ref_arm])
    # Verify raw rerun shard identities and original numerical/consumer sources.
    for marker in sorted((a.rematch / "runs").glob("*/*/*/shard*/attempt*/final_complete.json")):
        leaf = marker.parent
        config = json.loads((leaf / "config.json").read_text())
        receipt = json.loads((leaf / "receipt.json").read_text())
        original_config = a.base / leaf.relative_to(a.rematch) / "config.json"
        ref = json.loads(original_config.read_text())
        for field in ("manifest_sha256", "source_sha256", "runtime_sha256", "attention_source_sha256",
                      "protocol_sha256", "model_revision", "host_gpu_uuid"):
            if config[field] != ref[field]:
                errors.append(f"numerical binding: {leaf}/{field}")
        if receipt["records_sha256"] != sha(leaf / "records.jsonl"):
            errors.append(f"rerun records hash: {leaf}")
        sources.append({"path": str(leaf), "config_sha256": sha(leaf / "config.json"),
                        "records_sha256": sha(leaf / "records.jsonl")})
    if len(sources) != 8:
        errors.append("expected eight fresh rerun shards")
    receipt = {"status": "pass" if not errors else "fail", "errors": errors,
               "retained_panels": len(summaries), "retained_requests": sum(x["requests"] for x in summaries.values()),
               "original_scored_sha256": sha(orig_path), "rerun_scored_sha256": sha(rerun_path),
               "source_sha256": sha(__file__), "sources": sources,
               "cancelled": ["ultrallada/aime26", "ultrallada/ruler128k"],
               "transductive_count_calibration": ["llada22/aime26/V2", "trado/aime26/V2"]}
    for name, value in (("summary.json", summaries), ("paired.json", contrasts),
                        ("receipt.json", receipt)):
        (a.out / name).write_text(json.dumps(value, indent=2) + "\n")
    # Scored rows contain no tokens or text; keep old target misses separate.
    (a.out / "scored_records.jsonl").write_text("".join(json.dumps(r) + "\n" for rs in tables.values() for r in rs))
    (a.out / "historical_target_miss_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    labels = dict(zip(ARMS, ("Dense", "All-kept", "V31", "V2", "V3a")))
    lines = ["# Retained multi-model evaluation", "", "All retained sparse arms must achieve48–52% consumer sparsity. AIME V2 is transductively count-calibrated; other arms are frozen primary panels. Ultra AIME/RULER were cancelled; TraDo RULER is unsupported. No clean latency claim.", "",
             "| Model/task | Arm | Accuracy | N | C | N/C | T | Sparsity |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, r in summaries.items():
        m, s, arm = label.split("/")
        lines.append(f"| {m}/{s} | {labels[arm]} | {100*r['accuracy']:.2f}% | {r['N']} | {r['C']} | {r['N_per_C']:.3f} | {r['T']} | {100*r['consumer_sparsity']:.2f}% |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    if errors:
        raise RuntimeError("final audit failed: " + repr(errors))
    (a.out / "final_complete.json").write_text(json.dumps({"status": "complete_user_retained_scope",
          "panels": 30, "requests": 4010, "fresh_v2_replacement_requests": 180,
          "receipt_sha256": sha(a.out / "receipt.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
