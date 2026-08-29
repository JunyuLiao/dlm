"""Audit and final three-question report for the canonical result bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .plots import generate_plots
from .fitting import REQUIRED_DESIGN


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _strict_parity_passed(metadata: dict[str, Any]) -> bool:
    parity = metadata.get("parity", {})
    prompts = parity.get("prompts", [])
    return bool(parity.get("passed")) and bool(prompts) and all(
        row.get("bitwise_identical")
        and row.get("completion_tokens_identical")
        and row.get("text_identical")
        and row.get("attention_output_guard", {}).get("passed")
        and row.get("native_noop_parity", {}).get("bitwise_identical")
        and row.get("coverage", {}).get("complete")
        and row.get("coverage", {}).get("missing_call_layer_head_combinations") == 0
        for row in prompts
    )


def build_report(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    collections = []
    shard_audit = []
    for metadata_path in sorted((output_dir / "collection").glob("*/*/metadata.json")):
        metadata = _load(metadata_path) or {}
        collections.append(metadata)
        adapter, corpus = metadata_path.parts[-3], metadata_path.parts[-2]
        shards = list((output_dir / "shards" / adapter / corpus).glob("*.npz"))
        region_counts = [
            (_load(path.with_suffix(".json")) or {}).get("coverage", {}).get("kv_region", {})
            for path in shards
        ]
        regions_separated = bool(region_counts) and all(
            row.get("prefix_tiles", 0) > 0 and row.get("canvas_tiles", 0) > 0
            for row in region_counts
        )
        shard_audit.append({
            "adapter": adapter,
            "corpus": corpus,
            "expected": metadata.get("num_prompts"),
            "present": len(shards),
            "complete": len(shards) == metadata.get("num_prompts") and all(path.with_suffix(".json").exists() for path in shards),
            "prefix_canvas_regions_separated": regions_separated,
            "strict_native_output_parity": _strict_parity_passed(metadata),
        })
    fit = _load(output_dir / "fit" / "threshold_models.json")
    math = _load(output_dir / "math500" / "summary.json")
    plot_paths = generate_plots(output_dir) if list((output_dir / "shards").glob("*/*/*.npz")) else []
    stable = bool(collections) and all(
        bool(item.get("convergence", {}).get("passed")) for item in collections
    )
    observed_design = {(row["adapter"], row["corpus"]) for row in shard_audit}
    trustworthy = stable and observed_design == REQUIRED_DESIGN and all(
        row["complete"] and row["prefix_canvas_regions_separated"] and row["strict_native_output_parity"]
        for row in shard_audit
    )
    profiling = bool(fit and fit.get("simple_profiling_passed"))
    safe = []
    if math:
        safe = [row for row in math["conditions"] if row["condition"] != "dense" and row["accuracy_safe_at_minus_5pp"]]
        safe.sort(key=lambda row: (row["logical_density"] if row["logical_density"] is not None else 2.0, row["condition"]))
    if not collections:
        answer1 = "Not yet measured: no final prompt shards are present."
    elif trustworthy:
        answer1 = "Yes. All corpus/model collections passed native-output parity, shard completeness, and the predeclared prompt-convergence gate."
    else:
        failed = []
        for item in collections:
            label = f"{item.get('adapter', 'unknown')}/{item.get('corpus', 'unknown')}"
            reasons = []
            if not item.get("parity", {}).get("passed"):
                reasons.append("native-output parity")
            if not item.get("convergence", {}).get("passed"):
                reasons.append("prompt convergence")
            if reasons:
                failed.append(f"{label} ({', '.join(reasons)})")
        incomplete = [f"{row['adapter']}/{row['corpus']} (shard completeness)" for row in shard_audit if not row["complete"]]
        failed.extend(incomplete)
        failed.extend(
            f"{row['adapter']}/{row['corpus']} (prefix/canvas separation)"
            for row in shard_audit if not row["prefix_canvas_regions_separated"]
        )
        failed.extend(
            f"{row['adapter']}/{row['corpus']} (strict native-output parity)"
            for row in shard_audit if not row["strict_native_output_parity"]
        )
        answer1 = "No. Failed gate(s): " + "; ".join(failed) + "."
    if fit is None:
        answer2 = "Not yet measured: fit has not been run."
    elif profiling:
        selected = fit["selected"]
        answer2 = f"Yes. The least-granular passing table is {selected['family']} at scope {selected['scope']}."
    else:
        answer2 = "No. No portable model/model-attention table passed every held-out density gate; the selected empirical result is diagnostic only."
    if math is None:
        answer3 = "Not yet measured: the fixed 50-problem sparse-accuracy sweep has not been run."
    elif safe:
        best = safe[0]
        answer3 = f"{best['condition']} is the lowest-realized-density condition whose paired 95% delta lower bound exceeds -5 percentage points."
    else:
        answer3 = "None of the sparse conditions met the predeclared -5 percentage-point paired equivalence criterion."
    audit = {
        "all_prompt_shards_present": bool(shard_audit) and all(row["complete"] for row in shard_audit),
        "all_four_model_corpus_pairs_present": observed_design == REQUIRED_DESIGN,
        "all_strict_native_output_parity_passed": bool(shard_audit) and all(row["strict_native_output_parity"] for row in shard_audit),
        "all_prefix_canvas_regions_separated": bool(shard_audit) and all(row["prefix_canvas_regions_separated"] for row in shard_audit),
        "fitting_uses_calibration_prompts_only": bool(fit and fit.get("calibration_only_audit")),
        "math500_conditions_share_prompts_and_seeds": bool(math and math.get("all_conditions_share_problem_ids_and_seeds")),
        "plots_derived_from_final_shards": bool(plot_paths),
        "measured_speedup_claimed": False,
        "shards": shard_audit,
    }
    result = {
        "distribution_stable_and_trustworthy": answer1,
        "small_table_predicts_density": answer2,
        "accuracy_preserving_choice": answer3,
        "audit": audit,
        "plots": [str(path.relative_to(output_dir)) for path in plot_paths],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_text = "not available" if not fit or not fit.get("selected") else f"`{fit['selected']['family']}` / `{fit['selected']['scope']}` ({fit['decision']})"
    math_rows = [] if not math else math["conditions"]
    table = "\n".join(
        f"| {row['condition']} | {row['symbolic_pass_at_1']:.1%} | {row['paired_delta_vs_dense']:+.1%} [{row['paired_delta_ci95_lower']:+.1%}, {row['paired_delta_ci95_upper']:+.1%}] | "
        f"{row['logical_density']:.1%} | {row['physical_density']:.1%} | {row['retained_dense_attention_mass']:.1%} | "
        f"{row['attention_output_relative_error']:.3g} | {row['mcnemar_dense_wrong_sparse_right']}/{row['mcnemar_dense_right_sparse_wrong']} | "
        f"{'yes' if row['accuracy_safe_at_minus_5pp'] else 'no'} |"
        for row in math_rows
    ) or "| _pending_ | — | — | — | — | — | — | — | — |"
    text = f"""# Distribution-first sparse-routing study

This bundle models the Sol-Attn mean-pooled 64x64 proxy and evaluates only freshly recomputed canvas-KV routing. Prefix tiles are dense sinks. Cross-step reuse, selective refresh, and skipped-block approximation are outside this study.

## Answers

1. **Is the estimated aggregate distribution stable and trustworthy?** {answer1}
2. **Can a small profiling table predict routing density across corpora?** {answer2}
3. **Which threshold model and density, if any, preserve Math500 accuracy?** {answer3}

Selected threshold result: {selected_text}.

## Math500 symbolic pass@1

| Condition | Accuracy | Paired delta vs dense (95% CI) | Logical density | Physical density | Retained mass | Output rel. error | McNemar +/− | Safe at -5pp |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
{table}

## Audit

- Prompt shards complete: `{audit['all_prompt_shards_present']}`.
- Strict native-output parity and complete call/layer/head coverage: `{audit['all_strict_native_output_parity_passed']}`.
- Prefix and canvas KV regions separated in every prompt shard: `{audit['all_prefix_canvas_regions_separated']}`.
- Fits use calibration prompts only: `{audit['fitting_uses_calibration_prompts_only']}`.
- Math500 prompt/seed pairing: `{audit['math500_conditions_share_prompts_and_seeds']}`.
- Plots regenerated from final compact shards: `{audit['plots_derived_from_final_shards']}`.
- Logical and physical density are routing measurements from a dense reference implementation. They are not measured kernel speedups; no speedup claim is made.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")
    return result
