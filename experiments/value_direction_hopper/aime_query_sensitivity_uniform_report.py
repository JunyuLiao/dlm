"""Offline comparison of the causal AIME26 run with archived controls.

All denoising quantiles here are computed over canvases, not over the total
number of calls in a question. No model, calibration, or routing is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


SEEDS = (42, 43, 44)
KINDS = ("whole", "global", "local")
PHASES = ("call1", "call2", "late")
CONTROLS = ("dense", "gaussian32", "temporal")


def summarize(rows):
    """Pool tile counts and canvas iterations without averaging percentages."""
    counts = {kind: {key: sum(r["counts"][kind][key] for r in rows)
                     for key in ("eligible", "skipped")} for kind in KINDS}
    steps = [int(c["iterations"]) for r in rows for c in r["canvases"]]
    result = dict(n=len(rows), correct=sum(int(r["score"]) for r in rows),
                  accuracy=statistics.mean(r["score"] for r in rows),
                  mean_canvas_steps=statistics.mean(steps),
                  median_canvas_steps=statistics.median(steps),
                  p90_canvas_steps=float(np.quantile(steps, .9)),
                  total_canvases=len(steps), cap_canvases=sum(s >= 48 for s in steps),
                  executed_tiles=counts["whole"]["eligible"]-counts["whole"]["skipped"])
    for kind, label in zip(KINDS, ("overall", "global", "local")):
        result[f"{label}_sparsity"] = counts[kind]["skipped"] / max(1, counts[kind]["eligible"])
    for phase in PHASES:
        for kind in KINDS:
            parts = [r.get("phase_counts", {}).get(phase, {}).get(kind) for r in rows]
            # Dense archives can lack the call-1/call-2 split. No tiles skipped.
            if any(part is None for part in parts):
                value = 0. if all(r["counts"][kind]["skipped"] == 0 for r in rows) else None
            else:
                value = sum(p["skipped"] for p in parts) / max(1, sum(p["eligible"] for p in parts))
            result[f"{phase}_{kind}_sparsity"] = value
    return result


def question_disagreements(method, rows, seeds=SEEDS):
    questions = {}
    for row in rows:
        scores = questions.setdefault(row["id"], {})
        if row["seed"] in scores:
            raise ValueError(f"duplicate question/seed: {method} {row['id']} {row['seed']}")
        scores[row["seed"]] = int(row["score"])
    output = []
    for question, scores in sorted(questions.items(), key=lambda x: int(x[0].split("/")[-1])):
        if set(scores) != set(seeds):
            raise ValueError(f"incomplete question: {method} {question}")
        values = [scores[s] for s in seeds]
        output.append(dict(method=method, id=question,
                           **{f"correct_seed{s}": scores[s] for s in seeds},
                           correct_seed_count=sum(values),
                           seed_disagreement=int(len(set(values)) > 1)))
    return output


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root, controls):
    root, controls = Path(root), Path(controls)
    completion = json.loads((root / "final_complete.json").read_text())
    if not completion["passed"] or set(completion["seeds"]) != set(SEEDS):
        raise ValueError("the complete three-seed evaluation is required")
    cfg = json.loads((root / "configuration.json").read_text())
    expected_ids = {r["id"] for r in json.loads((root / "final_manifest.json").read_text())}
    frozen = {method: json.loads((root / "thresholds" / f"{method}_s50.json").read_text())
              for method in cfg["methods"]}
    methods = list(CONTROLS) + cfg["methods"]
    by_method, per_seed, pooled, disagreements = {}, [], [], []
    prompt_hashes = {}
    for method in methods:
        source = controls if method in CONTROLS else root
        all_rows = []
        seed_accuracies = []
        for seed in SEEDS:
            condition = f"dense_seed{seed}" if method == "dense" else f"{method}_s50_seed{seed}"
            rows = [json.loads(p.read_text()) for p in (source / "final" / condition).glob("*.json")]
            if len(rows) != 30 or len({r["id"] for r in rows}) != 30:
                raise ValueError(f"incomplete or duplicate {condition}: {len(rows)}")
            if {r["id"] for r in rows} != expected_ids:
                raise ValueError(f"question manifest mismatch: {condition}")
            for row in rows:
                if row["seed"] != seed:
                    raise ValueError(f"incorrect seed in {condition}")
                old = prompt_hashes.setdefault(row["id"], row["prompt_hash"])
                if old != row["prompt_hash"]:
                    raise ValueError(f"prompt mismatch: {method} {row['id']}")
                if method in frozen and row["policy"] != frozen[method]["policy"]:
                    raise ValueError(f"evaluation policy changed: {condition} {row['id']}")
            metrics = summarize(rows)
            per_seed.append(dict(method=method, seed=seed, **metrics))
            seed_accuracies.append(metrics["accuracy"])
            all_rows.extend(rows)
        by_method[method] = all_rows
        question_rows = question_disagreements(method, all_rows)
        if len(question_rows) != 30:
            raise ValueError(f"question manifest differs: {method}")
        disagreements.extend(question_rows)
        pooled.append(dict(method=method, **summarize(all_rows),
                           accuracy_std=statistics.stdev(seed_accuracies),
                           disagreement_questions=sum(r["seed_disagreement"] for r in question_rows)))
    _csv(root / "comparison_per_seed.csv", per_seed)
    _csv(root / "comparison_pooled.csv", pooled)
    _csv(root / "comparison_question_disagreement.csv", disagreements)

    sensitivity = []
    for method in cfg["methods"]:
        groups = {}
        for row in by_method[method]:
            first_calls = 0
            for step in row["step_records"]:
                if step["threshold_phase"] != "uniform":
                    raise ValueError(f"nonuniform threshold selection: {method} {row['id']}")
                if step["iteration"] == 1:
                    first_calls += 1
                    if abs(step["sensitivity_mean"] - (1 + cfg["beta"])) > 1e-6:
                        raise ValueError(f"first call did not use maximum sensitivity: {method}")
                groups.setdefault(step["iteration"], []).append(step["sensitivity_mean"])
            if first_calls != row["total_canvases"]:
                raise ValueError(f"missing canvas initialization: {method} {row['id']}")
        for iteration, values in sorted(groups.items()):
            sensitivity.append(dict(method=method, call=iteration, canvases=len(values),
                                    mean_sensitivity=statistics.mean(values)))
    _csv(root / "sensitivity_by_call.csv", sensitivity)

    threshold_rows = []
    for method in cfg["methods"]:
        data = frozen[method]
        policy = data["policy"]
        if not policy["call1"] == policy["call2"] == policy["late"]:
            raise ValueError(f"nonuniform policy: {method}")
        if data["status"] != "attained" or data["max_error"] > .02:
            raise ValueError(f"unattained calibration: {method}")
        threshold_rows.append(dict(method=method,
            local_log_threshold=policy["late"]["local"]["log_threshold"],
            global_log_threshold=policy["late"]["global"]["log_threshold"],
            max_error=data["max_error"], status=data["status"],
            **{f"calibration_{k}_sparsity": data["selected_metrics"]["sparsity"][k] for k in KINDS}))
    _csv(root / "comparison_calibration.csv", threshold_rows)
    (root / "final_audit.json").write_text(json.dumps(dict(
        passed=True, new_evaluations=30 * len(SEEDS) * len(cfg["methods"]),
        archived_control_evaluations=30 * len(SEEDS) * len(CONTROLS),
        identical_prompts=True, identical_frozen_policies=True,
        uniform_threshold_selection=True, maximum_sensitivity_every_first_call=True), indent=2) + "\n")

    lines = ["# AIME26 causal sensitivity: final comparison", "",
        "Six methods × 30 prompts × seeds 42/43/44 = 540 new evaluations. "
        "Dense, Gaussian32 s50, and original temporal s50 are archived controls. "
        "All sparse methods below target 50% physical sparsity; actual sparsity is reported.", "",
        "Calibration IDs **2, 8, 14, 20, 23, 30** overlap the 30-question final manifest. "
        "These are development results. Thresholds were frozen before the final runs. "
        "The three seeds repeat the same 30 questions; 90 runs are not 90 independent questions.", "",
        "β=3, flip EMA γ=0.5, trajectory γ=0.5 except T_smooth γ=0.8, "
        f"m_ref={cfg['m_ref']:.6f}, Gaussian rank 32, physical tiles 128×64. "
        "Every new method starts each canvas at sensitivity 4 and uses exactly one local/global log threshold pair for every call.", "",
        "## Accuracy and seed variation", "",
        "SD is the sample standard deviation across three seed accuracies (ddof=1). "
        "Disagreement counts questions whose correctness differs across seeds.", "",
        "| Method | Seed 42 | Seed 43 | Seed 44 | Pooled | SD (pp) | Disagreement |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in pooled:
        accuracies = [r["accuracy"] for r in per_seed if r["method"] == row["method"]]
        lines.append(f"| {row['method']} | " + " | ".join(f"{100*a:.2f}%" for a in accuracies) +
                     f" | {100*row['accuracy']:.2f}% ({row['correct']}/90) | {100*row['accuracy_std']:.2f} | {row['disagreement_questions']}/30 |")
    lines += ["", "## Physical sparsity and denoising", "",
        "Sparsity pools skipped/eligible tile counts. Mean, median, and P90 are all over "
        "256-token canvases, not question totals. Capped canvases reached 48 calls. "
        "The native dense archive has no physical tile counter, so its executed-tile count is unavailable.", "",
        "| Method | Overall / global / local sparsity | Mean / median / P90 calls | Capped / total canvases | Executed tiles |",
        "|---|---:|---:|---:|---:|"]
    for row in pooled:
        sparsity = " / ".join(f"{100*row[f'{k}_sparsity']:.2f}%" for k in ("overall", "global", "local"))
        steps = " / ".join(f"{row[f'{k}_canvas_steps']:.2f}" for k in ("mean", "median", "p90"))
        executed = "N/A" if row["method"] == "dense" else f"{row['executed_tiles']:,}"
        lines.append(f"| {row['method']} | {sparsity} | {steps} | {row['cap_canvases']} / {row['total_canvases']} | {executed} |")
    lines += ["", "## Phase sparsity", "",
        "Each cell is overall / global / local. Calls are indexed within each canvas.", "",
        "| Method | Call 1 | Call 2 | Later calls |", "|---|---:|---:|---:|"]
    for row in pooled:
        phases = [" / ".join("N/A" if row[f"{p}_{k}_sparsity"] is None else
                            f"{100*row[f'{p}_{k}_sparsity']:.2f}%" for k in KINDS) for p in PHASES]
        lines.append(f"| {row['method']} | " + " | ".join(phases) + " |")
    lines += ["", "## Calibration and frozen thresholds", "",
        "Error is max absolute deviation from 50% over overall, global, and local sparsity. "
        "Every selected pair passed the 2 percentage point calibration tolerance.", "",
        "| Method | Local log threshold | Global log threshold | Calibration O/G/L | Max error (pp) |",
        "|---|---:|---:|---:|---:|"]
    for row in threshold_rows:
        s = " / ".join(f"{100*row[f'calibration_{k}_sparsity']:.2f}%" for k in KINDS)
        lines.append(f"| {row['method']} | {row['local_log_threshold']:.6f} | {row['global_log_threshold']:.6f} | {s} | {100*row['max_error']:.2f} |")
    lines += ["", "## Artifacts", "",
        "- `comparison_per_seed.csv`: all metrics for every method and seed.",
        "- `comparison_pooled.csv`: pooled metrics, seed SD, and phase sparsity.",
        "- `comparison_question_disagreement.csv`: each question's correctness for each seed, including controls.",
        "- `comparison_calibration.csv`: full-precision thresholds and calibration error.", ""]
    lines += ["- `sensitivity_by_call.csv`: observed mean coefficients by canvas call (later calls include only surviving canvases).",
              "- `final_audit.json`: checks manifests, frozen policies, uniform selection, and every canvas's first-call coefficient.", ""]
    (root / "causal_analysis.md").write_text("\n".join(lines))
    return pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--controls", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/query_adaptive_aime_temporal_v14")
    args = parser.parse_args()
    report(args.root, args.controls)


if __name__ == "__main__":
    main()
