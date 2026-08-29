"""Distribution and proxy-quality statistics for compact tile observations."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


STANDARDIZE_ROW_KEYS = [
    "request_id",
    "denoising_step",
    "layer",
    "head",
    "attention_type",
    "query_block",
]

# Region-specific summaries are useful, but routing thresholds are calibrated
# over the complete KV row.  Keeping these keys distinct prevents a one-block
# prefix from acquiring sigma=0 and being classified as skipped by definition.
ROW_KEYS = [
    *STANDARDIZE_ROW_KEYS,
    "kv_region",
]


def load_npz_frame(path) -> pd.DataFrame:
    with np.load(path) as payload:
        return pd.DataFrame({name: payload[name] for name in payload.files})


def load_record_set(output_dir, stem: str) -> pd.DataFrame:
    """Load a consolidated NPZ or request shards from a result directory."""
    from pathlib import Path

    output = Path(output_dir)
    paths = []
    consolidated = output / f"{stem}.npz"
    if consolidated.exists():
        paths.append(consolidated)
    paths.extend(sorted((output / "raw").glob(f"{stem}_*.npz")) if (output / "raw").exists() else [])
    if not paths:
        raise FileNotFoundError(f"no {stem} NPZ records found under {output}")
    frames = [load_npz_frame(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return _safe_pearson(stats.rankdata(left), stats.rankdata(right))


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    return float(np.dot(left_centered, right_centered) / denominator) if denominator else math.nan


def _binary_auc_and_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan, math.nan
    ranks = stats.rankdata(scores)
    auc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(scores)[::-1]
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    average_precision = float(precision[ordered == 1].mean())
    return float(auc), average_precision


def add_standardized_proxies(tiles: pd.DataFrame) -> pd.DataFrame:
    result = tiles.copy()
    for proxy in ("mean", "max"):
        column = f"proxy_{proxy}"
        grouped = result.groupby(STANDARDIZE_ROW_KEYS, observed=True)[column]
        mean = grouped.transform("mean")
        sigma = grouped.transform(lambda values: values.std(ddof=0))
        result[f"proxy_z_{proxy}"] = (result[column] - mean) / (sigma + 1.0e-8)
    return result


def distribution_rows(tiles: pd.DataFrame, betas: Iterable[float]) -> pd.DataFrame:
    output: list[dict] = []
    for keys, group in tiles.groupby(ROW_KEYS, observed=True, sort=False):
        common = dict(zip(ROW_KEYS, keys))
        for proxy in ("mean", "max"):
            values = group[f"proxy_{proxy}"].to_numpy(dtype=np.float64)
            full_row_z = group[f"proxy_z_{proxy}"].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            sigma = float(values.std())
            z = (values - mean) / (sigma + 1.0e-8)
            ordered = np.sort(z)
            if len(ordered):
                empirical = np.arange(1, len(ordered) + 1) / len(ordered)
                ks_distance = float(np.max(np.abs(empirical - stats.norm.cdf(ordered))))
                wasserstein = float(stats.wasserstein_distance(ordered, stats.norm.ppf((np.arange(len(ordered)) + 0.5) / len(ordered))))
            else:
                ks_distance = math.nan
                wasserstein = math.nan
            row = {
                **common,
                "proxy": proxy,
                "tile_count": len(values),
                "row_mean": mean,
                "row_sigma": sigma,
                "skewness": float(stats.skew(values, bias=False)) if len(values) >= 3 else math.nan,
                "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)) if len(values) >= 4 else math.nan,
                "ks_distance": ks_distance,
                "wasserstein_distance": wasserstein,
            }
            for beta in betas:
                # z was calibrated over the complete prefix+canvas row. Region
                # summaries report that same routing decision rather than
                # silently recalibrating an independent prefix threshold.
                row[f"density_beta_{beta:g}"] = float(np.mean(full_row_z > beta))
            output.append(row)
    return pd.DataFrame(output)


def distribution_table(rows: pd.DataFrame, betas: Iterable[float]) -> pd.DataFrame:
    aggregations = {
        "skewness": "mean",
        "excess_kurtosis": "mean",
        "ks_distance": "mean",
        "wasserstein_distance": "mean",
    }
    for beta in betas:
        aggregations[f"density_beta_{beta:g}"] = "mean"
    table = rows.groupby(["proxy", "attention_type", "kv_region"], observed=True).agg(aggregations).reset_index()
    for beta in betas:
        column = f"density_beta_{beta:g}"
        variation = rows.groupby(["proxy", "attention_type", "kv_region"], observed=True)[column].agg(
            lambda values: values.std(ddof=0) / max(values.mean(), 1.0e-12)
        )
        table = table.merge(variation.rename(f"density_cv_beta_{beta:g}").reset_index())
    return table


def threshold_predictability(rows: pd.DataFrame, betas: Iterable[float]) -> pd.DataFrame:
    output = []
    for (proxy, attention_type, kv_region), group in rows.groupby(["proxy", "attention_type", "kv_region"], observed=True):
        for beta in betas:
            values = group[f"density_beta_{beta:g}"].to_numpy(dtype=float)
            output.append(
                {
                    "proxy": proxy,
                    "attention_type": attention_type,
                    "kv_region": kv_region,
                    "beta": beta,
                    "gaussian_predicted_density": float(stats.norm.sf(beta)),
                    "mean_density": float(np.mean(values)),
                    "median_density": float(np.median(values)),
                    "p10_density": float(np.quantile(values, 0.1)),
                    "p90_density": float(np.quantile(values, 0.9)),
                    "density_cv": float(np.std(values) / max(np.mean(values), 1.0e-12)),
                }
            )
    return pd.DataFrame(output)


def proxy_quality(tiles: pd.DataFrame, target_densities: Iterable[float]) -> pd.DataFrame:
    output = []
    for keys, group in tiles.groupby(ROW_KEYS, observed=True, sort=False):
        common = dict(zip(ROW_KEYS, keys))
        mass = group["softmax_mass"].to_numpy(dtype=float)
        max_reference = group["proxy_max"].to_numpy(dtype=float)
        blasst = group["blasst_margin"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        count = len(group)
        for proxy in ("mean", "max"):
            values = group[f"proxy_{proxy}"].to_numpy(dtype=float)
            pearson_mass = _safe_pearson(values, mass)
            spearman_mass = _safe_spearman(values, mass)
            pearson_max = _safe_pearson(values, max_reference)
            spearman_max = _safe_spearman(values, max_reference)
            finite_blasst = np.isfinite(blasst)
            spearman_blasst = _safe_spearman(values[finite_blasst], blasst[finite_blasst])
            for density in target_densities:
                keep = max(1, min(count, int(math.ceil(count * density))))
                proxy_top = set(np.argsort(values)[-keep:])
                mass_top = set(np.argsort(mass)[-keep:])
                max_top = set(np.argsort(max_reference)[-keep:])
                labels = np.zeros(count, dtype=int)
                labels[list(mass_top)] = 1
                roc_auc, pr_auc = _binary_auc_and_ap(labels, values)
                output.append(
                    {
                        **common,
                        "proxy": proxy,
                        "target_density": density,
                        "top_mass_recall": len(proxy_top & mass_top) / keep,
                        "top_max_recall": len(proxy_top & max_top) / keep,
                        "false_negative_rate": 1.0 - len(proxy_top & mass_top) / keep,
                        "pearson_mass": pearson_mass,
                        "spearman_mass": spearman_mass,
                        "pearson_max": pearson_max,
                        "spearman_max": spearman_max,
                        "spearman_blasst": spearman_blasst,
                        "roc_auc_mass": roc_auc,
                        "pr_auc_mass": pr_auc,
                    }
                )
    return pd.DataFrame(output)


def proxy_quality_table(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "top_mass_recall",
        "top_max_recall",
        "false_negative_rate",
        "spearman_mass",
        "spearman_max",
        "spearman_blasst",
        "roc_auc_mass",
        "pr_auc_mass",
    ]
    return rows.groupby(["proxy", "attention_type", "kv_region", "target_density"], observed=True)[metrics].mean().reset_index()
