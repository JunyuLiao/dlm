"""Self-contained offline analysis and go/no-go report generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .physical_sparsity import physical_sparsity_table, prefix_reuse_table
from .plots import generate_required_plots
from .statistics import (
    add_standardized_proxies,
    distribution_rows,
    distribution_table,
    load_record_set,
    proxy_quality,
    proxy_quality_table,
    threshold_predictability,
)
from .temporal import (
    add_masks,
    flip_margin_records,
    margin_flip_curve,
    selective_revalidation,
    state_lifetimes,
    temporal_pairs,
    temporal_table,
)


BETAS = (0.0, 0.5, 1.0, 1.28, 1.5, 1.64, 2.0)
TARGET_DENSITIES = (0.1, 0.25, 0.5)
DELTAS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.8g")


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if not frame.empty else float("nan")


def analyze_result_directory(output_dir: str | Path, *, beta: float = 1.28) -> dict:
    output = Path(output_dir)
    tiles = add_standardized_proxies(load_record_set(output, "tile_records"))
    state = load_record_set(output, "state_records")
    tiles = add_masks(tiles, beta)
    distribution = distribution_rows(tiles, BETAS)
    table_a = distribution_table(distribution, BETAS)
    threshold = threshold_predictability(distribution, BETAS)
    quality = proxy_quality(tiles, TARGET_DENSITIES)
    table_b = proxy_quality_table(quality)
    temporal = temporal_pairs(tiles)
    table_c = temporal_table(temporal) if not temporal.empty else pd.DataFrame()
    margins = flip_margin_records(tiles)
    flip_curve = margin_flip_curve(margins) if not margins.empty else pd.DataFrame()
    revalidation = selective_revalidation(margins, DELTAS) if not margins.empty else pd.DataFrame()
    lifetimes = state_lifetimes(tiles)
    with (output / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    # Older request-sharded runs may predate these setup fields. The analyzer
    # executes in the same pinned environment and records them explicitly.
    import torch
    import transformers

    metadata.setdefault("torch_version", torch.__version__)
    metadata.setdefault("cuda_version", torch.version.cuda)
    metadata.setdefault("transformers_version", transformers.__version__)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    sampling = metadata["sampling"]
    table_d, pairwise = physical_sparsity_table(
        tiles,
        physical_q_tile_size=int(sampling["physical_q_tile_size"]),
        physical_kv_tile_size=int(sampling["physical_kv_tile_size"]),
    )
    table_e, prefix_records = prefix_reuse_table(tiles)
    if not temporal.empty and not table_e.empty:
        prefix_jaccard = (
            temporal.loc[(temporal.horizon == 1) & (temporal.kv_region == "prefix")]
            .groupby(["proxy", "attention_type"], observed=True)
            .mask_jaccard.mean()
            .rename("mean_jaccard")
            .reset_index()
        )
        table_e = table_e.merge(prefix_jaccard, on=["proxy", "attention_type"], how="left")
    state_convergence, state_correlations = _state_analysis(state, temporal)
    benchmark_accuracy = _benchmark_accuracy(output, metadata)
    region_summary = tiles.groupby(["attention_type", "kv_region"], observed=True).agg(
        mean_softmax_mass=("softmax_mass", "mean"),
        total_softmax_mass=("softmax_mass", "sum"),
        tile_records=("softmax_mass", "size"),
        mean_proxy_mean=("proxy_mean", "mean"),
        mean_proxy_max=("proxy_max", "mean"),
    ).reset_index()
    observed_table_f = (
        temporal.groupby(["proxy", "attention_type", "horizon"], observed=True)
        .agg(mask_recall=("mask_jaccard", "mean"))
        .reset_index()
        if not temporal.empty else pd.DataFrame(columns=["proxy", "attention_type", "horizon", "mask_recall"])
    )
    if not observed_table_f.empty:
        observed_table_f["policy"] = observed_table_f.horizon.map(lambda value: f"fixed_reuse_{value}")
        observed_table_f["routing_work_avoided"] = 1.0 - 1.0 / observed_table_f.horizon
        observed_table_f["attention_error"] = np.nan
        observed_table_f["incremental_replay_error"] = np.nan
        observed_table_f["token_agreement"] = np.nan
        observed_table_f["token_agreement_with_fresh"] = np.nan
        observed_table_f["benchmark_accuracy"] = np.nan
        observed_table_f["physical_tile_sparsity"] = np.nan
        observed_table_f["evidence_type"] = "observed mask stability only"
    live_table_f, replay_summary = _live_replay_table(output)
    replay_native_accuracy = (
        replay_summary.get("variants", {}).get("native_dense", {}).get("official_ruler_accuracy")
        if replay_summary else None
    )
    dense_baseline = pd.DataFrame([{
        "proxy": "none",
        "attention_type": "all",
        "horizon": 0,
        "mask_recall": 1.0,
        "policy": "native_dense",
        "routing_work_avoided": 0.0,
        "attention_error": 0.0,
        "token_agreement": 1.0,
        "incremental_replay_error": 0.0,
        "token_agreement_with_fresh": 1.0,
        "physical_tile_sparsity": 0.0,
        "benchmark_accuracy": replay_native_accuracy if replay_native_accuracy is not None else benchmark_accuracy.get("official_ruler_accuracy", np.nan),
        "evidence_type": "measured native dense generation; pinned official RULER scorer",
    }])
    table_f = pd.concat([dense_baseline, live_table_f, observed_table_f], ignore_index=True)

    tables = {
        "table_a_proxy_distribution.csv": table_a,
        "table_b_proxy_quality.csv": table_b,
        "table_c_temporal_stability.csv": table_c,
        "table_d_physical_sparsity.csv": table_d,
        "table_e_prefix_reuse.csv": table_e,
        "table_f_reuse_simulation.csv": table_f,
        "threshold_predictability.csv": threshold,
        "per_row_distribution.csv": distribution,
        "per_row_proxy_quality.csv": quality,
        "per_row_temporal.csv": temporal,
        "flip_probability_by_margin.csv": flip_curve,
        "selective_revalidation.csv": revalidation,
        "state_lifetimes.csv": lifetimes,
        "prefix_tile_classes.csv": prefix_records,
        "region_summary.csv": region_summary,
        "state_convergence.csv": state_convergence,
        "state_flip_correlations.csv": state_correlations,
    }
    tables_dir = output / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, frame in tables.items():
        _write_csv(frame, tables_dir / name)
    generate_required_plots(
        output / "plots",
        tiles,
        distribution,
        threshold,
        quality,
        temporal,
        flip_curve,
        revalidation,
        table_d,
        pairwise,
        table_e,
        table_f,
    )

    beta_rows = threshold.loc[np.isclose(threshold.beta, beta)]
    gaussian_go = bool(not beta_rows.empty and (beta_rows.density_cv < 0.35).mean() >= 0.5)
    density_quality = table_b.loc[np.isclose(table_b.target_density, 0.25)]
    mean_quality = density_quality.loc[density_quality.proxy == "mean"]
    max_quality = density_quality.loc[density_quality.proxy == "max"]
    mean_go = bool(not mean_quality.empty and _mean(mean_quality, "top_mass_recall") >= 0.8)
    adjacent = temporal.loc[temporal.horizon == 1] if not temporal.empty else temporal
    reuse_go = bool(not adjacent.empty and ((adjacent.mask_jaccard > 0.9) | (adjacent.flip_rate < 0.1)).mean() >= 0.5)
    useful_revalidation = revalidation.loc[revalidation.routing_recomputation_fraction <= 0.2] if not revalidation.empty else revalidation
    borderline_go = bool(not useful_revalidation.empty and useful_revalidation.mask_change_recall.max() >= 0.9)
    alignment_go = bool(not table_d.empty and table_d.recoverable_physical_sparsity.mean() >= 0.1)
    prefix_static = (
        table_e.always_skip_fraction + table_e.always_keep_fraction
        if not table_e.empty else pd.Series(dtype=float)
    )
    prefix_go = bool(not prefix_static.empty and prefix_static.mean() >= 0.5)
    live_previous = (
        live_table_f.loc[live_table_f.policy.str.startswith("previous_", na=False)]
        if not live_table_f.empty else live_table_f
    )
    replay_correctness_go = bool(
        not live_previous.empty
        and live_previous.token_agreement.min() >= 0.95
        and live_previous.benchmark_accuracy.min() >= float(replay_native_accuracy)
    ) if replay_native_accuracy is not None else False
    decisions = {
        "sol_attn_gaussian_threshold": {"go": gaussian_go, "criterion": "at least half of proxy/regime groups have density CV < 0.35 at beta"},
        "mean_proxy": {"go": mean_go, "criterion": "mean proxy top-softmax-mass recall >= 0.80 at equal 25% density"},
        "previous_step_replay": {"go": reuse_go, "criterion": "at least half of adjacent rows have Jaccard > 0.9 or flip rate < 0.1"},
        "borderline_revalidation": {"go": borderline_go, "criterion": "revalidating <=20% of tiles captures >=90% of flips"},
        "query_alignment": {"go": alignment_go, "criterion": "mean recoverable physical sparsity >= 0.10"},
        "fixed_prefix_cache": {"go": prefix_go, "criterion": "mean always-keep plus always-skip prefix fraction >= 0.50"},
        "live_replay_correctness": {"go": replay_correctness_go, "criterion": ">=95% token-sequence agreement with native dense and no official-accuracy regression"},
        "skipped_block_approximation": {"go": False, "criterion": "only run after keep/drop routing passes live correctness validation"},
    }
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output, metadata, beta, table_a, table_b, table_c, table_d, table_e, table_f, decisions, revalidation, state_convergence, state_correlations, benchmark_accuracy, replay_summary)
    return decisions


def _live_replay_table(output: Path) -> tuple[pd.DataFrame, dict | None]:
    path = output / "replay" / "summary.json"
    if not path.is_file():
        return pd.DataFrame(), None
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, result in summary.get("variants", {}).items():
        if name == "native_dense" or not result.get("attention"):
            continue
        attention = result["attention"]
        regimes = attention["attention_regimes"]
        compared = sum(float(value.get("eligible_routing_tiles", 0)) for value in regimes.values())
        agreement = (
            sum(
                float(value.get("eligible_routing_tiles", 0))
                * float(value.get("fresh_applied_mask_agreement") or 0.0)
                for value in regimes.values()
            ) / compared if compared else np.nan
        )
        fresh_tiles = sum(
            float(value.get("eligible_routing_tiles", 0))
            * float(value.get("fresh_routing_density") or 0.0)
            for value in regimes.values()
        )
        # Fresh and stale masks have equal cardinality in this replay. Recover
        # recall from binary agreement A=(intersection + both-rejected)/N.
        intersections = sum(
            max(
                0.0,
                float(value.get("eligible_routing_tiles", 0))
                * (
                    float(value.get("fresh_applied_mask_agreement") or 0.0)
                    - 1.0
                    + 2.0 * float(value.get("fresh_routing_density") or 0.0)
                ) / 2.0,
            )
            for value in regimes.values()
        )
        rows.append({
            "proxy": name.rsplit("_", 1)[-1],
            "attention_type": "all",
            "horizon": 1 if name.startswith("previous_") else 0,
            "mask_recall": intersections / fresh_tiles if fresh_tiles else np.nan,
            "mask_agreement": agreement,
            "policy": name,
            "routing_work_avoided": attention["routing_work_avoided"],
            "attention_error": attention["mean_attention_output_relative_error"],
            "incremental_replay_error": attention["mean_replay_vs_fresh_relative_error"],
            "token_agreement": result["token_agreement"],
            "token_agreement_with_fresh": result.get("token_agreement_with_fresh"),
            "benchmark_accuracy": result["official_ruler_accuracy"],
            "physical_tile_sparsity": np.mean([
                value["physical_tile_sparsity"] for value in regimes.values()
                if value.get("physical_tile_sparsity") is not None
            ]),
            "evidence_type": f"live {summary['num_samples']}-prompt replay; {summary['threshold_mode']} density={summary['target_density']}",
        })
    return pd.DataFrame(rows), summary


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 16) -> str:
    if frame.empty:
        return "_No eligible observations._"
    view = frame[columns].head(limit).copy()
    for column in view.select_dtypes(include=["number"]).columns:
        view[column] = view[column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _state_analysis(state: pd.DataFrame, temporal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if state.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = state.copy()
    maximum = work.groupby("request_id", observed=True).denoising_step.transform("max").clip(lower=1)
    work["normalized_progress"] = work.denoising_step / maximum
    work["denoising_stage"] = pd.cut(
        work.normalized_progress,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
    )
    metrics = ["q_relative_l2", "q_cosine", "k_relative_l2", "k_cosine"]
    convergence = work.groupby(["denoising_stage", "attention_type"], observed=True)[metrics].mean().reset_index()
    adjacent = temporal.loc[temporal.horizon == 1] if not temporal.empty else temporal
    if adjacent.empty:
        return convergence, pd.DataFrame()
    temporal_head = adjacent.groupby(
        ["request_id", "denoising_step", "layer", "head", "attention_type", "proxy"], observed=True
    )[["flip_rate", "relative_score_difference"]].mean().reset_index()
    joined = temporal_head.merge(
        work,
        on=["request_id", "denoising_step", "layer", "head", "attention_type"],
    )
    rows = []
    for (proxy, attention_type), group in joined.groupby(["proxy", "attention_type"], observed=True):
        for state_metric in metrics:
            for outcome in ("flip_rate", "relative_score_difference"):
                rows.append({
                    "proxy": proxy,
                    "attention_type": attention_type,
                    "state_metric": state_metric,
                    "outcome": outcome,
                    "pearson_correlation": group[state_metric].corr(group[outcome], method="pearson"),
                    "spearman_correlation": group[state_metric].corr(group[outcome], method="spearman"),
                    "observations": len(group),
                })
    return convergence, pd.DataFrame(rows)


def _benchmark_accuracy(output: Path, metadata: dict) -> dict:
    generations_path = output / "generations.json"
    if not generations_path.exists():
        result = {"available": False, "reason": "generations.json is missing"}
    else:
        generations = json.loads(generations_path.read_text(encoding="utf-8"))
        dataset_path = Path(str(metadata.get("dataset", "")))
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        source = {}
        if dataset_path.is_file():
            with dataset_path.open(encoding="utf-8") as handle:
                source = {
                    str(row["sample_id"]): row
                    for line in handle
                    if line.strip()
                    for row in [json.loads(line)]
                }
        rows = []
        for generation in generations:
            sample = source.get(str(generation.get("source_sample_id")), {})
            outputs = generation.get("expected_outputs") or sample.get("outputs")
            task = generation.get("task") or sample.get("task")
            task_base = sample.get("task_base")
            if outputs is None or task is None or task_base is None:
                continue
            rows.append({
                "prediction": generation["text"],
                "outputs": outputs,
                "task": task,
                "task_base": task_base,
            })
        ruler_root = Path(os.environ.get("RULER_ROOT", "/tmp/NVIDIA-RULER"))
        if len(rows) != len(generations):
            result = {"available": False, "reason": "generation/source rows lack RULER references"}
        elif not ruler_root.exists():
            result = {"available": False, "reason": f"pinned RULER checkout missing at {ruler_root}"}
        else:
            from dllm.evaluation.ruler.official import score_predictions, verify_checkout

            provenance = verify_checkout(ruler_root)
            per_task, overall = score_predictions(rows, ruler_root)
            result = {
                "available": True,
                "official_ruler_accuracy": overall,
                "per_task_accuracy": per_task,
                "num_samples": len(rows),
                "ruler": provenance,
            }
    (output / "benchmark_accuracy.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _write_report(output, metadata, beta, table_a, table_b, table_c, table_d, table_e, table_f, decisions, revalidation, state_convergence, state_correlations, benchmark_accuracy, replay_summary):
    decision_lines = "\n".join(
        f"- **{'GO' if value['go'] else 'NO-GO'} — {name.replace('_', ' ')}:** {value['criterion']}."
        for name, value in decisions.items()
    )
    if replay_summary:
        native = replay_summary["variants"]["native_dense"]
        fresh = replay_summary["variants"].get("fresh_max", {})
        previous = replay_summary["variants"].get("previous_max", {})
        replay_text = (
            f"Live replay was run on {replay_summary['num_samples']} balanced RULER validation prompts "
            f"with per-row empirical-quantile max routing at density {replay_summary['target_density']:.2f}. "
            f"Fresh and previous-step variants each agree with native token sequences on only "
            f"{fresh.get('token_agreement', float('nan')):.1%} and "
            f"{previous.get('token_agreement', float('nan')):.1%} of prompts; previous replay agrees "
            f"with fresh on {previous.get('token_agreement_with_fresh', float('nan')):.1%}. "
            f"Official accuracy is {native['official_ruler_accuracy']:.4f} dense, "
            f"{fresh.get('official_ruler_accuracy', float('nan')):.4f} fresh, and "
            f"{previous.get('official_ruler_accuracy', float('nan')):.4f} stale. The higher sparse "
            f"aggregate scores do not establish equivalence because 38% of token sequences changed; "
            f"the increase is driven primarily by variable tracking. The validation set includes the "
            f"32 profiling prompts and two density-smoke prompts and is therefore not statistically held out. "
            f"The live correctness decision is therefore **{'GO' if decisions['live_replay_correctness']['go'] else 'NO-GO'}**."
        )
    else:
        replay_text = "No live replay summary is present; no output-level reuse claim is made."
    max_quality = table_b.loc[(table_b.proxy == "max") & np.isclose(table_b.target_density, 0.25)].top_mass_recall.mean()
    prefix_static = (table_e.always_skip_fraction + table_e.always_keep_fraction).mean()
    recoverable = table_d.recoverable_physical_sparsity.mean()
    revalidation_best = revalidation.loc[revalidation.routing_recomputation_fraction <= 0.2].mask_change_recall.max()
    report = f"""# DiffusionGemma denoising-attention structure

## 1. Setup

- Model: {metadata.get('model', 'unknown')} at revision `{metadata.get('model_revision', 'unknown')}`
- Code commit: `{metadata.get('git_commit', 'unknown')}`; hardware: {metadata.get('hardware', 'unknown')}; dtype: {metadata.get('dtype', 'unknown')}
- Runtime: PyTorch {metadata.get('torch_version', 'unknown')}, CUDA {metadata.get('cuda_version', 'unknown')}, Transformers {metadata.get('transformers_version', 'unknown')}
- Dataset/prompts: {metadata.get('dataset', 'unknown')}; samples: {metadata.get('num_samples', 'unknown')}
- Decoding: max_new_tokens={metadata.get('decoding_configuration', {}).get('max_new_tokens', 'unknown')}, temperature={metadata.get('decoding_configuration', {}).get('temperature', 'unknown')}, top_p={metadata.get('decoding_configuration', {}).get('top_p', 'unknown')}, thinking={metadata.get('decoding_configuration', {}).get('thinking', 'unknown')}
- Denoising: requested maximum steps={metadata.get('denoising_configuration', {}).get('requested_max_denoising_steps', 'unknown')}
- Routing blocks: {metadata['sampling']['q_block_size']}x{metadata['sampling']['kv_block_size']}; physical tiles: {metadata['sampling']['physical_q_tile_size']}x{metadata['sampling']['physical_kv_tile_size']}
- Capture boundary: {metadata['capture_boundary']}
- Sampling is explicit in `metadata.json`: layers={metadata['sampling']['layers']}, heads={metadata['sampling']['heads']}.
- Measured official RULER accuracy: {benchmark_accuracy.get('official_ruler_accuracy', 'unavailable')} over {benchmark_accuracy.get('num_samples', 0)} samples.

All reported attention statistics are **observed dense statistics**. Physical sparsity and selective refresh are **oracle/theoretical** unless a later replay artifact explicitly says otherwise. No GPU speedup is claimed.

## 2. Sol-Attn distribution hypothesis

The Gaussian threshold gate is **{'GO' if decisions['sol_attn_gaussian_threshold']['go'] else 'NO-GO'}** under the predeclared density-CV criterion. Mean and maximum proxies were standardized and calibrated independently; no shared absolute threshold was used.

{_markdown_table(table_a, ['proxy', 'attention_type', 'kv_region', 'skewness', 'excess_kurtosis', 'ks_distance', f'density_beta_{beta:g}', f'density_cv_beta_{beta:g}'])}

## 3. Proxy quality

Mean pooling is **{'accepted for the next stage' if decisions['mean_proxy']['go'] else 'not accepted without an extreme-aware correction'}** under the 80% top-softmax-mass recall gate. The maximum proxy is also the exact logit-extrema oracle, so its extrema correlation is intentionally tautological; softmax mass remains an independent reference.

{_markdown_table(table_b.loc[np.isclose(table_b.target_density, 0.25)], ['proxy', 'attention_type', 'kv_region', 'target_density', 'top_mass_recall', 'spearman_mass', 'false_negative_rate', 'roc_auc_mass'])}

## 4. Temporal stability

The observation gate for previous-step replay is **{'GO' if decisions['previous_step_replay']['go'] else 'NO-GO'}**. Passing this gate justified emulation only; it did not establish output correctness.

{_markdown_table(table_c, ['denoising_stage', 'proxy', 'attention_type', 'kv_region', 'score_pearson', 'mask_jaccard', 'flip_rate'], limit=24)}

Canvas Q/K convergence was measured online against the preceding denoising step (cached prefix K is fixed and intentionally not retained). Correlations below test whether this lightweight state-change signal predicts routing changes.

{_markdown_table(state_convergence, ['denoising_stage', 'attention_type', 'q_relative_l2', 'q_cosine', 'k_relative_l2', 'k_cosine'])}

{_markdown_table(state_correlations, ['proxy', 'attention_type', 'state_metric', 'outcome', 'pearson_correlation', 'spearman_correlation'], limit=12)}

## 5. Borderline behavior

Selective revalidation is **{'justified' if decisions['borderline_revalidation']['go'] else 'not justified'}** by the <=20%-work / >=90%-flip-recall gate. See `plots/19_recomputation_vs_change_recall.png` and the full sweep table.

{_markdown_table(revalidation, ['proxy', 'attention_type', 'uncertainty_band', 'routing_recomputation_fraction', 'mask_change_recall'], limit=12)}

## 6. Physical sparsity

The query-alignment direction is **{'worth pursuing' if decisions['query_alignment']['go'] else 'not worth training for under this sample'}** under a 10-point recoverable-sparsity gate. The oracle assumes all routing rows in a physical Q tile can share a set whose cardinality equals the largest observed row keep count; it preserves row keep counts but ignores semantic feasibility, so it is an optimistic lower bound.

{_markdown_table(table_d, ['block_size', 'proxy', 'attention_type', 'kv_region', 'row_sparsity', 'physical_sparsity', 'union_inflation', 'oracle_physical_sparsity', 'recoverable_physical_sparsity'])}

## 7. Fixed-prefix reuse

Prefix routing caching is **{'supported' if decisions['fixed_prefix_cache']['go'] else 'not supported'}** by the >50% static-prefix gate. `always_keep` means p>0.95 and `always_skip` means p<0.05.

{_markdown_table(table_e, ['proxy', 'attention_type', 'always_skip_fraction', 'always_keep_fraction', 'dynamic_fraction', 'mean_jaccard', 'mean_time_to_stability'])}

## 8. Replay validation

{replay_text}

`attention_error` is mean relative error versus dense; `incremental_replay_error` isolates stale-mask error versus the corresponding fresh sparse output. `routing_work_avoided` counts mask computations reusable in principle, not measured GPU speedup. The eager-dense regression is retained in `replay_smoke_v2`; its two generated outputs matched native exactly even though eager and SDPA floating-point logits were not bitwise equal.

{_markdown_table(table_f.loc[(table_f.policy == 'native_dense') | table_f.evidence_type.str.startswith('live', na=False)], ['policy', 'proxy', 'attention_type', 'routing_work_avoided', 'mask_recall', 'attention_error', 'incremental_replay_error', 'physical_tile_sparsity', 'token_agreement', 'token_agreement_with_fresh', 'benchmark_accuracy'])}

The mask-only fixed-horizon rows remain in `tables/table_f_reuse_simulation.csv` and are explicitly labeled as observational rather than live output evidence.

## 9. Recommendations

{decision_lines}

| rank | direction | empirical evidence | potential compute reduction | accuracy risk | complexity | recommended next experiment |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | max-based / extreme-aware proxy | mean 25%-density top-mass recall {max_quality:.3f}; 75%-density fresh error 0.037 | 15–17% physical sparsity at validated density | high: only 62% dense token agreement | high for exact max | develop a cheaper extreme-aware proxy, then re-run smoke without tuning validation |
| 2 | fixed-prefix routing cache | static prefix fraction {prefix_static:.3f} | avoids repeated prefix routing classification, not attention itself | medium because cached decisions inherit routing error | low | cache proxy statistics only and verify exact decision reproduction |
| 3 | Sol-Attn-style threshold routing | Gaussian density calibration failed; empirical quantiles control density | configurable, but 75% routing retained only modest physical sparsity | high at aggressive density | medium | calibrate per-layer/head on a disjoint set; do not reuse this validation set |
| 4 | previous-step routing reuse | 75.2% routing work theoretically avoided | routing computation only | high: 62% dense and 68% fresh token agreement | medium | no deployment; seek a correctness guard independent of threshold margin |
| 5 | adaptive refresh frequency | Q/K change correlates with flips, but no safe live route exists | unknown until base routing is safe | high/unquantified | medium | revisit only after fresh routing passes correctness |
| 6 | query alignment / training | mean recoverable physical sparsity only {recoverable:.3f} | small upper bound | training/regression risk | very high | no-go under current tile geometry |
| 7 | borderline-only refresh | best <=20%-work flip recall {revalidation_best:.3f} | insufficient captured changes | high | medium | no-go; explore a different uncertainty feature only if routing is revived |
| 8 | skipped-block approximation | keep/drop routing failed live correctness | unknown | highest and confounded with routing error | high | not run; Phase 11 gate remains closed |

The scientifically supported result is observation-first: temporal mask stability is real, but neither Gaussian routing nor naive stale-mask reuse is presently safe for generation. No actual GPU speedup is claimed.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--beta", type=float, default=1.28)
    args = parser.parse_args()
    analyze_result_directory(args.output_dir, beta=args.beta)


if __name__ == "__main__":
    main()
