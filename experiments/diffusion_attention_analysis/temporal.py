"""Cross-denoising stability, reuse horizons, and selective refresh oracles."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


IDENTITY = ["request_id", "layer", "head", "attention_type", "query_block", "kv_region", "kv_start"]
ROW_IDENTITY = ["request_id", "layer", "head", "attention_type", "query_block", "kv_region"]


def add_masks(tiles: pd.DataFrame, beta: float) -> pd.DataFrame:
    result = tiles.copy()
    for proxy in ("mean", "max"):
        result[f"margin_{proxy}"] = result[f"proxy_z_{proxy}"] - beta
        result[f"keep_{proxy}"] = result[f"margin_{proxy}"] > 0
    return result


def _correlation(left, right, method: str) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    function = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(function(left, right).statistic)


def temporal_pairs(tiles: pd.DataFrame, horizons: Iterable[int] = (1, 2, 4, 8)) -> pd.DataFrame:
    output = []
    indexed = tiles.set_index(IDENTITY + ["denoising_step"]).sort_index()
    for horizon in horizons:
        previous = indexed.reset_index().copy()
        previous["denoising_step"] += horizon
        joined = tiles.merge(
            previous,
            on=IDENTITY + ["denoising_step"],
            suffixes=("_current", "_previous"),
        )
        if joined.empty:
            continue
        group_keys = ROW_IDENTITY + ["denoising_step"]
        for keys, group in joined.groupby(group_keys, observed=True, sort=False):
            common = dict(zip(group_keys, keys))
            max_step = int(tiles.loc[tiles.request_id == common["request_id"], "denoising_step"].max())
            for proxy in ("mean", "max"):
                current_scores = group[f"proxy_{proxy}_current"].to_numpy(dtype=float)
                previous_scores = group[f"proxy_{proxy}_previous"].to_numpy(dtype=float)
                current = group[f"keep_{proxy}_current"].to_numpy(dtype=bool)
                prior = group[f"keep_{proxy}_previous"].to_numpy(dtype=bool)
                intersection = np.logical_and(current, prior).sum()
                union = np.logical_or(current, prior).sum()
                flips = current != prior
                output.append(
                    {
                        **common,
                        "proxy": proxy,
                        "horizon": horizon,
                        "normalized_progress": common["denoising_step"] / max(max_step, 1),
                        "score_pearson": _correlation(current_scores, previous_scores, "pearson"),
                        "score_spearman": _correlation(current_scores, previous_scores, "spearman"),
                        "relative_score_difference": float(np.linalg.norm(current_scores - previous_scores) / max(np.linalg.norm(previous_scores), 1.0e-12)),
                        "mask_jaccard": float(intersection / union) if union else 1.0,
                        "mask_precision": float(intersection / current.sum()) if current.sum() else 1.0,
                        "mask_recall": float(intersection / prior.sum()) if prior.sum() else 1.0,
                        "keep_to_skip_rate": float(np.logical_and(prior, ~current).mean()),
                        "skip_to_keep_rate": float(np.logical_and(~prior, current).mean()),
                        "flip_rate": float(flips.mean()),
                        "tile_count": len(group),
                    }
                )
    return pd.DataFrame(output)


def temporal_table(rows: pd.DataFrame) -> pd.DataFrame:
    adjacent = rows.loc[rows.horizon == 1].copy()
    adjacent["denoising_stage"] = pd.cut(
        adjacent.normalized_progress,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
    )
    return adjacent.groupby(
        ["denoising_stage", "proxy", "attention_type", "kv_region"], observed=True
    )[["score_pearson", "score_spearman", "mask_jaccard", "flip_rate"]].mean().reset_index()


def flip_margin_records(tiles: pd.DataFrame) -> pd.DataFrame:
    previous = tiles.copy()
    previous["denoising_step"] += 1
    joined = tiles.merge(previous, on=IDENTITY + ["denoising_step"], suffixes=("_current", "_previous"))
    output = []
    for proxy in ("mean", "max"):
        part = joined[IDENTITY + ["denoising_step", "sequence_length_current"]].copy()
        part["proxy"] = proxy
        part["previous_margin_abs"] = joined[f"margin_{proxy}_previous"].abs()
        part["flipped"] = joined[f"keep_{proxy}_current"] != joined[f"keep_{proxy}_previous"]
        max_step = part.groupby("request_id", observed=True).denoising_step.transform("max").clip(lower=1)
        part["normalized_progress"] = part.denoising_step / max_step
        output.append(part)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def margin_flip_curve(records: pd.DataFrame, bins: Iterable[float] = (0, 0.1, 0.25, 0.5, 1, 2, np.inf)) -> pd.DataFrame:
    result = records.copy()
    result["distance_bin"] = pd.cut(result.previous_margin_abs, bins=list(bins), include_lowest=True)
    result["denoising_stage"] = pd.cut(
        result.normalized_progress,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
    )
    return result.groupby(
        ["proxy", "attention_type", "denoising_stage", "distance_bin"], observed=True
    ).agg(flip_probability=("flipped", "mean"), tiles=("flipped", "size")).reset_index()


def selective_revalidation(records: pd.DataFrame, deltas: Iterable[float]) -> pd.DataFrame:
    output = []
    for (proxy, attention_type), group in records.groupby(["proxy", "attention_type"], observed=True):
        total_flips = int(group.flipped.sum())
        for delta in deltas:
            selected = group.previous_margin_abs < delta
            output.append(
                {
                    "proxy": proxy,
                    "attention_type": attention_type,
                    "uncertainty_band": delta,
                    "routing_recomputation_fraction": float(selected.mean()),
                    "mask_change_recall": float(group.loc[selected, "flipped"].sum() / total_flips) if total_flips else 1.0,
                    "total_flips": total_flips,
                }
            )
    return pd.DataFrame(output)


def state_lifetimes(tiles: pd.DataFrame) -> pd.DataFrame:
    output = []
    for proxy in ("mean", "max"):
        part = tiles[IDENTITY + ["denoising_step", f"keep_{proxy}"]].sort_values(
            IDENTITY + ["denoising_step"]
        ).copy()
        grouped = part.groupby(IDENTITY, observed=True, sort=False)
        previous_state = grouped[f"keep_{proxy}"].shift()
        previous_step = grouped.denoising_step.shift()
        starts = previous_state.isna() | (part[f"keep_{proxy}"] != previous_state) | (part.denoising_step != previous_step + 1)
        part["run_id"] = starts.groupby([part[column] for column in IDENTITY], observed=True).cumsum()
        runs = part.groupby(IDENTITY + ["run_id"], observed=True, sort=False).agg(
            state_value=(f"keep_{proxy}", "first"),
            start_step=("denoising_step", "min"),
            lifetime=("denoising_step", "size"),
        ).reset_index()
        runs["proxy"] = proxy
        runs["state"] = np.where(runs.pop("state_value"), "keep", "skip")
        output.append(runs.drop(columns="run_id"))
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()
