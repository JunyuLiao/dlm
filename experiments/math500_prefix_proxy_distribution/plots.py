"""Plots and weighted summaries for row-standardized prefix proxy logits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("diffusion_gemma", "fast_dllm_v2")
ATTENTION_TYPES = ("global", "local")


STANDARDIZED_EDGES = np.linspace(-8.0, 8.0, 321)
ROW_MEAN_EDGES = np.linspace(-256.0, 256.0, 257)


def _hist_quantile(hist: np.ndarray, edges: np.ndarray, probability: float) -> float:
    total = float(hist.sum())
    if total <= 0:
        return float("nan")
    target = probability * total
    index = int(np.searchsorted(np.cumsum(hist), target, side="left"))
    index = max(0, min(index, len(hist) - 1))
    return float((edges[index] + edges[index + 1]) * 0.5)


def _load_group(root: Path, adapter: str, attention_type: str) -> dict[str, Any]:
    standardized_hist = np.zeros(len(STANDARDIZED_EDGES) - 1, dtype=np.float64)
    row_mean_hist = np.zeros(len(ROW_MEAN_EDGES) - 1, dtype=np.float64)
    standardized_observations = 0
    standardized_weight = 0.0
    standardized_sum = 0.0
    standardized_sum_sq = 0.0
    standardized_sum_cube = 0.0
    standardized_sum_fourth = 0.0
    standardized_tail_weight = {
        "gt_2": 0.0, "lt_minus_2": 0.0,
        "gt_3": 0.0, "lt_minus_3": 0.0,
    }
    row_mean_observations = 0
    row_count = 0
    prefix_tiles = 0
    standardized_rows = 0
    zero_variance_rows = 0
    problems: set[str] = set()
    for path in sorted((root / "shards" / adapter).glob("*.npz")):
        with np.load(path) as payload:
            mask = payload["attention_type"].astype(str) == attention_type
            if not mask.any():
                continue
            reservoir_all = payload["prefix_reservoir"]
            row_weight_all = payload["hierarchical_row_weight"].astype(np.float64, copy=False)
            means_all = payload["prefix_mean"].astype(np.float64, copy=False)
            stds_all = payload["prefix_std"].astype(np.float64, copy=False)
            selected = np.flatnonzero(mask)
            for start in range(0, len(selected), 8192):
                indices = selected[start:start + 8192]
                reservoir = reservoir_all[indices].astype(np.float64, copy=False)
                row_weight = row_weight_all[indices]
                valid = np.isfinite(reservoir)
                counts = valid.sum(axis=1)
                per_value = np.divide(row_weight, counts, out=np.zeros_like(row_weight), where=counts > 0)
                means = means_all[indices]
                stds = stds_all[indices]
                positive_std = np.isfinite(stds) & (stds > 0.0)
                standardized_rows += int(positive_std.sum())
                zero_variance_rows += int((~positive_std).sum())
                row_mean_hist += np.histogram(means, bins=ROW_MEAN_EDGES, weights=row_weight)[0]
                row_mean_observations += len(means)
                if positive_std.any():
                    standardized = (reservoir[positive_std] - means[positive_std, None]) / stds[positive_std, None]
                    standardized_valid = np.isfinite(standardized)
                    tile_weights = np.broadcast_to(per_value[positive_std, None], standardized.shape)
                    values = standardized[standardized_valid]
                    weights = tile_weights[standardized_valid]
                    standardized_hist += np.histogram(values, bins=STANDARDIZED_EDGES, weights=weights)[0]
                    standardized_observations += int(len(values))
                    standardized_weight += float(weights.sum())
                    standardized_sum += float(np.dot(values, weights))
                    standardized_sum_sq += float(np.dot(values * values, weights))
                    standardized_sum_cube += float(np.dot(values**3, weights))
                    standardized_sum_fourth += float(np.dot(values**4, weights))
                    standardized_tail_weight["gt_2"] += float(weights[values > 2.0].sum())
                    standardized_tail_weight["lt_minus_2"] += float(weights[values < -2.0].sum())
                    standardized_tail_weight["gt_3"] += float(weights[values > 3.0].sum())
                    standardized_tail_weight["lt_minus_3"] += float(weights[values < -3.0].sum())
            row_count += int(mask.sum())
            prefix_tiles += int(payload["prefix_tiles"][mask].sum())
            problems.update(str(x) for x in payload["request_id"][mask])
    if row_count == 0:
        return {
            "standardized_hist": standardized_hist, "row_mean_hist": row_mean_hist,
            "standardized_observations": 0, "standardized_weight": 0.0,
            "standardized_sum": 0.0, "standardized_sum_sq": 0.0,
            "row_mean_observations": 0,
            "rows": 0, "prefix_tiles": 0, "problems": 0,
            "standardized_rows": 0, "zero_variance_rows": 0,
        }
    return {
        "standardized_hist": standardized_hist, "row_mean_hist": row_mean_hist,
        "standardized_observations": standardized_observations,
        "standardized_weight": standardized_weight,
        "standardized_sum": standardized_sum, "standardized_sum_sq": standardized_sum_sq,
        "standardized_sum_cube": standardized_sum_cube,
        "standardized_sum_fourth": standardized_sum_fourth,
        "standardized_tail_weight": standardized_tail_weight,
        "rows": row_count,
        "prefix_tiles": prefix_tiles,
        "problems": len(problems),
        "row_mean_observations": row_mean_observations,
        "standardized_rows": standardized_rows,
        "zero_variance_rows": zero_variance_rows,
    }


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_plots(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = {(model, attention): _load_group(root, model, attention) for model in MODELS for attention in ATTENTION_TYPES}
    summary: dict[str, Any] = {
        "schema_version": 2,
        "population": "prefix_only",
        "score_transform": "row-standardized raw pre-softmax block-proxy logits z=(s-mu_row)/sigma_row",
        "raw_score_transform": "raw pre-softmax block-proxy logits",
        "groups": {},
    }
    for (model, attention), group in groups.items():
        key = f"{model}|{attention}"
        if group["standardized_weight"] > 0:
            hist = group["standardized_hist"]
            mean = group["standardized_sum"] / group["standardized_weight"]
            variance = max(0.0, group["standardized_sum_sq"] / group["standardized_weight"] - mean * mean)
            raw_third = group["standardized_sum_cube"] / group["standardized_weight"]
            raw_fourth = group["standardized_sum_fourth"] / group["standardized_weight"]
            central_third = raw_third - 3.0 * mean * (
                group["standardized_sum_sq"] / group["standardized_weight"]
            ) + 2.0 * mean**3
            central_fourth = raw_fourth - 4.0 * mean * raw_third + 6.0 * mean**2 * (
                group["standardized_sum_sq"] / group["standardized_weight"]
            ) - 3.0 * mean**4
            summary["groups"][key] = {
                "problems": group["problems"], "rows": group["rows"], "prefix_tiles": group["prefix_tiles"],
                "reservoir_observations": group["standardized_observations"],
                "standardized_rows": group["standardized_rows"],
                "zero_variance_rows": group["zero_variance_rows"],
                "weighted_quantiles": {str(q): _hist_quantile(hist, STANDARDIZED_EDGES, q) for q in (.01, .05, .25, .5, .75, .95, .99)},
                "weighted_mean": mean, "weighted_std": float(np.sqrt(variance)),
                "weighted_skewness": float(central_third / variance**1.5) if variance > 0 else None,
                "weighted_excess_kurtosis": float(central_fourth / variance**2 - 3.0) if variance > 0 else None,
                "asymmetric_tail_probabilities": {
                    name: float(value / group["standardized_weight"])
                    for name, value in group["standardized_tail_weight"].items()
                },
                "gaussian_tail_densities": {
                    name: float(hist[((STANDARDIZED_EDGES[:-1] + STANDARDIZED_EDGES[1:]) * .5) >= beta].sum() / hist.sum())
                    for name, beta in {
                        "beta_-0.674_rho75": -0.6744897501960817,
                        "beta_0_rho50": 0.0,
                        "beta_0.674_rho25": 0.6744897501960817,
                        "beta_1.282_rho10": 1.2815515655446004,
                    }.items()
                },
            }
        else:
            summary["groups"][key] = {
                "problems": group["problems"], "rows": group["rows"], "prefix_tiles": group["prefix_tiles"],
                "reservoir_observations": 0, "standardized_rows": group["standardized_rows"],
                "zero_variance_rows": group["zero_variance_rows"],
            }

    # Row-standardized tile-proxy distributions, one panel per model/type.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, model in enumerate(MODELS):
        for col, attention in enumerate(ATTENTION_TYPES):
            ax = axes[row, col]
            group = groups[(model, attention)]
            if group["standardized_weight"] <= 0:
                ax.text(0.5, 0.5, "no observed layers", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{model} / {attention}")
                continue
            hist = group["standardized_hist"]
            density = hist / (hist.sum() * np.diff(STANDARDIZED_EDGES))
            ax.stairs(density, STANDARDIZED_EDGES, color="tab:blue")
            centers = (STANDARDIZED_EDGES[:-1] + STANDARDIZED_EDGES[1:]) * .5
            gaussian = np.exp(-.5 * centers**2) / np.sqrt(2 * np.pi)
            ax.plot(centers, gaussian, color="black", ls="--", lw=1, label="N(0,1)")
            q = summary["groups"][f"{model}|{attention}"]["weighted_quantiles"]
            ax.axvline(q["0.5"], color="tab:red", ls="--", lw=1, label="weighted median")
            ax.set_title(f"{model} / {attention}")
            ax.set_xlabel("row-standardized prefix proxy z")
            ax.set_ylabel("row-balanced weighted density")
            ax.legend(fontsize=7, frameon=False)
            ax.text(0.02, 0.97, f"10 problems; {group['rows']:,} rows\n{group['prefix_tiles']:,} prefix tiles\nz=(s−μ)/σ; plot range: 0.5–99.5%", transform=ax.transAxes, va="top", fontsize=7)
    path1 = plot_dir / "01_standardized_prefix_proxy_distributions.png"
    _save(fig, path1)

    # Row-mean distributions make the aggregation unit explicit.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    for col, model in enumerate(MODELS):
        ax = axes[0, col]
        for attention, color in (("global", "tab:blue"), ("local", "tab:orange")):
            group = groups[(model, attention)]
            if group["row_mean_observations"] == 0:
                continue
            hist = groups[(model, attention)]["row_mean_hist"]
            density = hist / (hist.sum() * np.diff(ROW_MEAN_EDGES))
            ax.stairs(density, ROW_MEAN_EDGES, color=color, label=attention)
        ax.set_title(f"{model}: row-mean prefix proxy")
        ax.set_xlabel("row mean raw proxy"); ax.set_ylabel("row-balanced density")
        if ax.lines or ax.patches:
            ax.legend(frameon=False)
    path2 = plot_dir / "02_row_mean_prefix_proxy_distributions.png"
    _save(fig, path2)
    summary["plots"] = [str(path1.relative_to(root)), str(path2.relative_to(root))]
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary
