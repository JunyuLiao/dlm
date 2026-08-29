"""Required plots for the DiffusionGemma attention investigation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


COLORS = {"local": "#1f77b4", "global": "#d62728"}


def _save(output: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(output / f"{name}.png", dpi=180)
    plt.close()


def _empty(output: Path, name: str, title: str, message: str = "No eligible observations") -> None:
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    _save(output, name)


def generate_required_plots(
    output_dir: str | Path,
    tiles: pd.DataFrame,
    distribution: pd.DataFrame,
    threshold: pd.DataFrame,
    quality: pd.DataFrame,
    temporal: pd.DataFrame,
    flip_curve: pd.DataFrame,
    revalidation: pd.DataFrame,
    physical: pd.DataFrame,
    pairwise: pd.DataFrame,
    prefix: pd.DataFrame,
    reuse_simulation: pd.DataFrame,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Standardized score histograms.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    grid = np.linspace(-4, 4, 200)
    for axis, proxy in zip(axes, ("mean", "max")):
        for attention_type, group in tiles.groupby("attention_type", observed=True):
            axis.hist(group[f"proxy_z_{proxy}"].clip(-4, 4), bins=60, density=True, alpha=0.4, label=attention_type, color=COLORS.get(attention_type))
        axis.plot(grid, stats.norm.pdf(grid), "k--", label="N(0,1)")
        axis.set(title=f"{proxy} proxy", xlabel="row-standardized score", ylabel="density")
        axis.legend()
    _save(output, "01_standardized_proxy_histograms")

    # 2. Representative Q-Q plots.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    representative = tiles.groupby(["attention_type", "request_id", "denoising_step", "layer", "head", "query_block", "kv_region"], observed=True, sort=False)
    for axis, proxy in zip(axes, ("mean", "max")):
        for attention_type in sorted(tiles.attention_type.unique()):
            group = next((part for keys, part in representative if keys[0] == attention_type and len(part) >= 4), None)
            if group is None:
                continue
            values = np.sort(group[f"proxy_z_{proxy}"].to_numpy())
            theoretical = stats.norm.ppf((np.arange(len(values)) + 0.5) / len(values))
            axis.scatter(theoretical, values, s=12, label=attention_type, color=COLORS.get(attention_type))
        axis.axline((0, 0), slope=1, color="black", linestyle="--")
        axis.set(title=f"Representative {proxy} rows", xlabel="Gaussian quantile", ylabel="empirical quantile")
        axis.legend()
    _save(output, "02_proxy_qq_plots")

    # 3. Actual keep ratio vs Gaussian prediction.
    plt.figure(figsize=(7, 5))
    for (proxy, attention_type, kv_region), group in threshold.groupby(["proxy", "attention_type", "kv_region"], observed=True):
        plt.plot(group.gaussian_predicted_density, group.mean_density, marker="o", label=f"{proxy}/{attention_type}/{kv_region}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
    plt.plot([0, 0.55], [0, 0.55], "k:")
    plt.xlabel("Gaussian-predicted density")
    plt.ylabel("Observed mean density")
    plt.legend()
    _save(output, "03_actual_vs_gaussian_keep_ratio")

    # 4. Keep-ratio variance.
    plt.figure(figsize=(8, 4))
    labels = threshold.apply(lambda row: f"{row.proxy}/{row.attention_type}/{row.kv_region}/β={row.beta:g}", axis=1)
    plt.bar(np.arange(len(threshold)), threshold.density_cv, color=[COLORS.get(value, "gray") for value in threshold.attention_type])
    plt.xticks(np.arange(len(threshold)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("coefficient of variation")
    _save(output, "04_keep_ratio_variance")

    # 5-7 proxy comparisons.
    sampled = tiles.sample(min(len(tiles), 50_000), random_state=0) if len(tiles) else tiles
    for name, x, y, title in (
        ("05_mean_vs_max_proxy_correlation", "proxy_mean", "proxy_max", "Mean proxy vs block maximum"),
        ("06_proxy_vs_softmax_mass_correlation", "proxy_mean", "softmax_mass", "Mean proxy vs dense softmax mass"),
    ):
        plt.figure(figsize=(7, 5))
        for attention_type, group in sampled.groupby("attention_type", observed=True):
            plt.scatter(group[x], group[y], s=5, alpha=0.25, label=attention_type, color=COLORS.get(attention_type))
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.legend()
        _save(output, name)
    plt.figure(figsize=(8, 4))
    quality_mean = quality.groupby(["proxy", "attention_type", "target_density"], observed=True).false_negative_rate.mean().reset_index()
    for (proxy, attention_type), group in quality_mean.groupby(["proxy", "attention_type"], observed=True):
        plt.plot(group.target_density, group.false_negative_rate, marker="o", label=f"{proxy}/{attention_type}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
    plt.xlabel("equal target density")
    plt.ylabel("false-negative rate vs top softmax mass")
    plt.legend()
    _save(output, "07_false_negative_rate_equal_density")

    adjacent = temporal.loc[temporal.horizon == 1]
    for number, metric, ylabel in (
        (8, "score_pearson", "score Pearson correlation"),
        (9, "mask_jaccard", "mask Jaccard"),
        (10, "flip_rate", "tile flip rate"),
    ):
        plt.figure(figsize=(8, 4))
        summary = adjacent.groupby(["proxy", "attention_type", "normalized_progress"], observed=True)[metric].mean().reset_index()
        for (proxy, attention_type), group in summary.groupby(["proxy", "attention_type"], observed=True):
            plt.plot(group.normalized_progress, group[metric], marker=".", label=f"{proxy}/{attention_type}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
        plt.xlabel("normalized denoising progress")
        plt.ylabel(ylabel)
        plt.legend()
        _save(output, f"{number:02d}_{metric}_vs_progress")

    # 11. Flip probability by distance from threshold.
    if flip_curve.empty:
        _empty(output, "11_flip_probability_vs_threshold_distance", "Flip probability vs threshold distance")
    else:
        plt.figure(figsize=(9, 5))
        curve = flip_curve.groupby(["proxy", "attention_type", "distance_bin"], observed=True).apply(lambda value: np.average(value.flip_probability, weights=value.tiles), include_groups=False).rename("probability").reset_index()
        curve["distance"] = curve.distance_bin.map(lambda value: value.mid if np.isfinite(value.right) else value.left * 1.5)
        for (proxy, attention_type), group in curve.groupby(["proxy", "attention_type"], observed=True):
            plt.plot(group.distance, group.probability, marker="o", label=f"{proxy}/{attention_type}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
        plt.xlabel("absolute normalized distance from previous threshold")
        plt.ylabel("P(classification flips next step)")
        plt.legend()
        _save(output, "11_flip_probability_vs_threshold_distance")

    # 12. Reuse horizons.
    plt.figure(figsize=(8, 4))
    horizon = temporal.groupby(["proxy", "attention_type", "horizon"], observed=True).mask_jaccard.mean().reset_index()
    for (proxy, attention_type), group in horizon.groupby(["proxy", "attention_type"], observed=True):
        plt.plot(group.horizon, group.mask_jaccard, marker="o", label=f"{proxy}/{attention_type}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
    plt.xlabel("reuse horizon (steps)")
    plt.ylabel("mask Jaccard")
    plt.legend()
    _save(output, "12_mask_reuse_horizon")

    # 13-16 physical sparsity.
    x = np.arange(len(physical))
    labels = physical.apply(lambda row: f"{row.proxy}/{row.attention_type}/{row.kv_region}", axis=1)
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.18, physical.row_sparsity, width=0.36, label="row")
    plt.bar(x + 0.18, physical.physical_sparsity, width=0.36, label="physical")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("sparsity")
    plt.legend()
    _save(output, "13_row_vs_physical_sparsity")
    if pairwise.empty:
        _empty(output, "14_within_query_block_jaccard", "Within-query-block Jaccard")
    else:
        plt.figure(figsize=(8, 4))
        groups = list(pairwise.groupby(["proxy", "attention_type"], observed=True))
        plt.boxplot([group.within_query_block_jaccard for _, group in groups], tick_labels=[f"{keys[0]}/{keys[1]}" for keys, _ in groups])
        plt.ylabel("pairwise Jaccard")
        _save(output, "14_within_query_block_jaccard")
    for number, column, title in (
        (15, "union_inflation", "Union inflation factor"),
        (16, "oracle_physical_sparsity", "Oracle-aligned physical sparsity"),
    ):
        plt.figure(figsize=(8, 4))
        plt.bar(x, physical[column], color=[COLORS.get(value, "gray") for value in physical.attention_type])
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(column.replace("_", " "))
        plt.title(title)
        _save(output, f"{number:02d}_{column}")

    # 17. Prefix vs canvas stability.
    plt.figure(figsize=(8, 4))
    region = adjacent.groupby(["proxy", "attention_type", "kv_region"], observed=True).mask_jaccard.mean().reset_index()
    region_labels = region.apply(lambda row: f"{row.proxy}/{row.attention_type}/{row.kv_region}", axis=1)
    plt.bar(np.arange(len(region)), region.mask_jaccard, color=[COLORS.get(value, "gray") for value in region.attention_type])
    plt.xticks(np.arange(len(region)), region_labels, rotation=60, ha="right")
    plt.ylabel("adjacent-step mask Jaccard")
    _save(output, "17_prefix_vs_canvas_mask_stability")

    if prefix.empty:
        _empty(output, "18_prefix_routing_classes", "Prefix routing classes")
    else:
        prefix_plot = prefix.set_index(["proxy", "attention_type"])[["always_skip_fraction", "always_keep_fraction", "dynamic_fraction"]]
        prefix_plot.plot(kind="bar", stacked=True, figsize=(8, 4))
        plt.ylabel("fraction of fixed-prefix tiles")
        plt.legend(loc="upper right")
        _save(output, "18_prefix_routing_classes")

    if revalidation.empty:
        _empty(output, "19_recomputation_vs_change_recall", "Selective revalidation")
    else:
        plt.figure(figsize=(7, 5))
        for (proxy, attention_type), group in revalidation.groupby(["proxy", "attention_type"], observed=True):
            plt.plot(group.routing_recomputation_fraction, group.mask_change_recall, marker="o", label=f"{proxy}/{attention_type}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
        plt.xlabel("routing recomputation fraction")
        plt.ylabel("true mask changes captured")
        plt.legend()
        _save(output, "19_recomputation_vs_change_recall")

    live = reuse_simulation.loc[reuse_simulation.evidence_type.str.startswith("live", na=False)]
    plt.figure(figsize=(8, 4))
    if live.empty:
        for (proxy, attention_type), group in horizon.groupby(["proxy", "attention_type"], observed=True):
            plt.plot(1 - 1 / group.horizon, group.mask_jaccard, marker="o", label=f"{proxy}/{attention_type}", color=COLORS.get(attention_type), linestyle="-" if proxy == "mean" else "--")
        plt.ylabel("stale/fresh mask Jaccard (observation only)")
    else:
        plt.scatter(live.routing_work_avoided, live.token_agreement, s=70)
        for row in live.itertuples():
            plt.annotate(f"{row.policy}\nattn err={row.attention_error:.3f}", (row.routing_work_avoided, row.token_agreement), xytext=(5, 5), textcoords="offset points")
        plt.ylabel("token-sequence agreement with native dense")
        plt.ylim(0, 1.05)
    plt.xlabel("routing work avoided")
    plt.legend() if live.empty else None
    _save(output, "20_routing_reuse_vs_degradation")
