"""Build a source-bound scientific report from the complete replicated study."""
import argparse
import csv
import json
from pathlib import Path

from scripts.v31_verify_replicated_results import read, require, sha, verify


SUITES = {"aime26": "AIME26 (30 problems × 3 seeds)",
          "longbench_v2": "LongBench-v2 (64 cells)",
          "math500": "MATH (100 cells)",
          "ruler128k": "RULER128K (13 tasks × 8 cells)"}
NAMES = {"current_v31_control": "V31 control", "full_swap8": "Full-value swaps",
         "residual_direction_guard": "Direction-guarded residual", "residual": "Raw residual",
         "full_swap8_direction_guard": "Swaps + direction guard",
         "dense_full_fix51994": "Dense FULL fix51994", "dense_piecewise": "Dense PIECEWISE",
         "allkept_fa4": "All-kept FA4"}
METHODS = list(NAMES)[1:5]


def save_json(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(reports, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    for ax, (suite, title) in zip(axes.flat, SUITES.items()):
        ax.axhline(0, color="0.65", linewidth=.8)
        ax.axvline(0, color="0.65", linewidth=.8)
        for method, color, marker in zip(METHODS, colors, markers):
            item = next(x for x in reports[suite]["comparisons"] if x["arm"] == method)
            result = item["problem_or_task_cluster"]
            y, x = result["delta"][:2]
            cy, cx = result["ci95"][:2]
            ax.errorbar(x, y, xerr=[[x-cx[0]], [cx[1]-x]],
                        yerr=[[y-cy[0]], [cy[1]-y]], color=color,
                        marker=marker, markersize=6, linewidth=1.2, capsize=3, alpha=.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Change in denoising calls / trial (lower is better)", fontsize=9)
        ax.set_ylabel("Change in official score (points)", fontsize=9)
        ax.grid(alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", label=NAMES[a])
               for a, c, m in zip(METHODS, colors, markers)]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Fixed-budget value refinements relative to same-engine V31\n"
                 "95% problem/task-cluster intervals; exploratory development results", fontsize=12)
    fig.savefig(output / "accuracy_calls.png", dpi=180)
    fig.savefig(output / "accuracy_calls.svg")
    fig.savefig(output / "accuracy_calls.pdf")
    plt.close(fig)


def build(study, output):
    complete = read(study / "final_complete.json")
    require(complete["status"] == "complete", "study incomplete")
    require(complete["protocol_sha256"] == sha(study / "protocol.json"), "changed protocol")
    require(complete["development_promotion_sha256"] == sha(study / "development_promotion.json"),
            "changed promotion")
    reports, checks = {}, {}
    for suite in SUITES:
        require(complete["report_sha256"][suite] == sha(study / "reports" / suite / "summary.json"),
                "changed workload report")
        checks[suite] = verify(study, suite)
        reports[suite] = read(study / "reports" / suite / "summary.json")
    output.mkdir(parents=True, exist_ok=False)
    rows, contrasts = [], []
    text = ["# Fixed-budget value refinements: completed replicated evaluation", "",
            "Official scoring is complete for 358 input/seed cells across four workloads: "
            "5,370 generation trials, including references. Each candidate has two trials per cell; "
            "V31 has four control bookends. This is repeated evaluation, not pass@2 or ensembling.", "",
            "Value changes are confined to GLOBAL layer 29. Every query head and Q128 tile keeps "
            "the same eligible-prefix K (27 for AIME/MATH; 128 for LongBench/RULER), plus shared "
            "mandatory/carry blocks. LOCAL remains dense and native sampling is unchanged. "
            "Clean timing remains paused; audit durations and call counts are not speedup.", "",
            "Inputs were previously examined development data. Intervals are separate problem/task "
            "and paired engine-shard sensitivity analyses, not simultaneous confidence regions or "
            "noninferiority guarantees. All structurally valid outcomes, including control drift, remain.", "",
            "![Accuracy and denoising-call changes](accuracy_calls.png)", ""]
    for suite, title in SUITES.items():
        report = reports[suite]
        text += [f"## {title}", "", "RULER uses the official task-average 0–100 score; other scores are accuracy percentages." if suite == "ruler128k" else "Official accuracy; calls and lengths are averages per generation trial.", "",
                 "| Method | Score | Calls N | N/C | Canvases C | Tokens T | p95 N | Cap % |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, name in NAMES.items():
            result = report["summaries"][arm]
            work, count = result["work"], result["problem_trials"]
            extra = result["extra_refinement_work"]
            row = dict(suite=suite, arm=arm, score=result["score_points"], trials=count,
                       N=work["mean_N"], N_per_C=work["N_per_C"], C=work["mean_C"], T=work["mean_T"],
                       p95_N=result["p95_N"], cap_rate=result["cap_rate"],
                       global_eligible_per_trial=work["global_eligible_tiles"]/count,
                       global_kept_per_trial=work["global_kept_tiles"]/count,
                       global_sparsity=work["global_sparsity"],
                       local_eligible_per_trial=work["local_eligible_tiles"]/count,
                       local_kept_per_trial=work["local_kept_tiles"]/count,
                       local_skipped_per_trial=work["local_skipped_tiles"]/count,
                       overall_eligible_per_trial=work["overall_eligible_rectangles"]/count,
                       overall_kept_per_trial=work["overall_kept_rectangles"]/count,
                       overall_sparsity=work["overall_sparsity"],
                       extra_full_qk_per_trial=extra.get("anchor_full_qk_rectangles", 0)/count,
                       extra_full_pv_per_trial=extra.get("anchor_full_pv_rectangles", 0)/count,
                       correction_path_calls_per_trial=extra.get("corrected_calls", 0)/count)
            rows.append(row)
            text.append(f"| {name} | {row['score']:.3f} | {row['N']:.3f} | {row['N_per_C']:.3f} | {row['C']:.3f} | {row['T']:.1f} | {row['p95_N']:.2f} | {100*row['cap_rate']:.2f} |")
        text += ["", "| Candidate | Score difference [95% interval] | N difference [95% interval] | N change % | Local screen |",
                 "|---|---:|---:|---:|---|"]
        baseline = report["summaries"]["current_v31_control"]["work"]["mean_N"]
        for item in report["comparisons"]:
            estimate, interval = item["problem_or_task_cluster"]["delta"], item["problem_or_task_cluster"]["ci95"]
            engine = item["paired_engine_block"]["ci95"]
            row = dict(suite=suite, arm=item["arm"], score_delta=estimate[0],
                       score_ci_low=interval[0][0], score_ci_high=interval[0][1],
                       N_delta=estimate[1], N_ci_low=interval[1][0], N_ci_high=interval[1][1],
                       N_change_percent=100*estimate[1]/baseline,
                       engine_score_ci_low=engine[0][0], engine_score_ci_high=engine[0][1],
                       engine_N_ci_low=engine[1][0], engine_N_ci_high=engine[1][1],
                       C_delta=estimate[2], T_delta=estimate[3], passes_local_screen=item["passes_screen"])
            contrasts.append(row)
            text.append(f"| {NAMES[item['arm']]} | {estimate[0]:+.3f} [{interval[0][0]:.3f}, {interval[0][1]:.3f}] | {estimate[1]:+.3f} [{interval[1][0]:.3f}, {interval[1][1]:.3f}] | {row['N_change_percent']:+.2f}% | {'Pass' if item['passes_screen'] else 'Fail'} |")
        noise = report["control_noise"]
        text += ["", f"Control bookends: {noise['bookend_hash_mismatches']}/{noise['bookend_pairs']} output-hash differences; "
                 f"{checks[suite]['bookend_score_disagreements']} score disagreements. First-control hashes differ across "
                 f"engines on {noise['cross_engine_first_control_hash_mismatches']}/{noise['cross_engine_pairs']} cells. "
                 "Dense/all-kept references use separate engines; incremental comparisons use same-engine control pairs.", ""]
    promotion = read(study / "development_promotion.json")
    text += ["## Promotion and work accounting", "",
             f"Full503 candidates under the predeclared all-workload screen: `{promotion['full503_candidates']}`. "
             "No further accuracy panel or timing run is launched by this report.", "",
             "`results.csv` contains per-trial GLOBAL, LOCAL and overall physical work, extra full-QK/PV "
             "discovery and correction-path counts. The overall denominator is decode GLOBAL full-support "
             "plus LOCAL window-intersecting H × Q128 × KV64 rectangles. Correction-path calls do not "
             "imply a nonzero correction on every row. Warmup requests are excluded from measured trials; "
             "the initial dense phases within measured requests remain in their accounting.", "",
             "`comparisons.csv` includes both problem/task and engine-block intervals. Per-workload JSON reports "
             "retain phase accounting, repeat-specific contrasts and all control drift. Paired trajectory diagnostics "
             "give C-versus-N/C decomposition and outcome-conditioned groups; those groups and tail trimming "
             "are descriptive, selection-biased analyses rather than causal effects.", ""]
    write_csv(output / "results.csv", rows)
    write_csv(output / "comparisons.csv", contrasts)
    (output / "summary.md").write_text("\n".join(text))
    save_json(output / "verification.json", checks)
    make_plot(reports, output)
    save_json(output / "final_complete.json", {
        "status": "complete", "source_sha256": sha(Path(__file__)),
        "study_final_complete_sha256": sha(study / "final_complete.json"),
        "protocol_sha256": sha(study / "protocol.json"),
        "outputs": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        "timing": "paused"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.study.resolve(), args.output.resolve())
