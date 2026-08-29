"""Physical QxKV tile alignment and fixed-prefix routing statistics."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def physical_sparsity_table(
    tiles: pd.DataFrame,
    *,
    physical_q_tile_size: int,
    physical_kv_tile_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    pairwise = []
    work = tiles.copy()
    work["physical_q_tile"] = work.query_start // physical_q_tile_size
    work["physical_kv_tile"] = work.kv_start // physical_kv_tile_size
    base_keys = ["request_id", "denoising_step", "layer", "head", "attention_type", "kv_region", "physical_q_tile"]
    for keys, group in work.groupby(base_keys, observed=True, sort=False):
        common = dict(zip(base_keys, keys))
        for proxy in ("mean", "max"):
            row_sets = []
            eligible_sets = []
            for _, row in group.groupby("query_block", observed=True):
                eligible = set(row.physical_kv_tile.astype(int))
                kept = set(row.loc[row[f"keep_{proxy}"], "physical_kv_tile"].astype(int))
                eligible_sets.append(eligible)
                row_sets.append(kept)
            eligible = set().union(*eligible_sets) if eligible_sets else set()
            union = set().union(*row_sets) if row_sets else set()
            row_counts = [len(value) for value in row_sets]
            oracle_count = max(row_counts, default=0)
            denominator = max(len(eligible), 1)
            records.append({
                **common,
                "proxy": proxy,
                "block_size": f"{physical_q_tile_size}x{physical_kv_tile_size}",
                "row_sparsity": 1.0 - np.mean(row_counts) / denominator if row_counts else 0.0,
                "physical_sparsity": 1.0 - len(union) / denominator,
                "union_inflation": (
                    len(union) / np.mean(row_counts)
                    if row_counts and np.mean(row_counts) > 0
                    else 1.0
                ),
                "oracle_physical_sparsity": 1.0 - oracle_count / denominator,
                "recoverable_physical_sparsity": (len(union) - oracle_count) / denominator,
                "query_rows": len(row_sets),
            })
            for left, right in itertools.combinations(row_sets, 2):
                pair_union = left | right
                pairwise.append({
                    **common,
                    "proxy": proxy,
                    "within_query_block_jaccard": len(left & right) / len(pair_union) if pair_union else 1.0,
                })
    rows = pd.DataFrame(records)
    table = rows.groupby(["block_size", "proxy", "attention_type", "kv_region"], observed=True)[
        ["row_sparsity", "physical_sparsity", "union_inflation", "oracle_physical_sparsity", "recoverable_physical_sparsity"]
    ].mean().reset_index()
    return table, pd.DataFrame(pairwise)


def prefix_reuse_table(tiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = tiles.loc[tiles.kv_region == "prefix"].copy()
    if prefix.empty:
        return pd.DataFrame(), pd.DataFrame()
    identity = ["request_id", "layer", "head", "attention_type", "query_block", "kv_start"]
    records = []
    for proxy in ("mean", "max"):
        grouped = prefix.groupby(identity, observed=True)
        probabilities = grouped[f"keep_{proxy}"].mean().rename("keep_probability").reset_index()
        ordered = prefix[identity + ["denoising_step", f"keep_{proxy}"]].sort_values(identity + ["denoising_step"]).copy()
        ordered_group = ordered.groupby(identity, observed=True, sort=False)
        previous = ordered_group[f"keep_{proxy}"].shift()
        ordered["change_step"] = ordered.denoising_step.where(previous.isna() | (ordered[f"keep_{proxy}"] != previous))
        stability = ordered.groupby(identity, observed=True).agg(
            first_step=("denoising_step", "min"),
            stable_step=("change_step", "max"),
        ).reset_index()
        stability["time_to_stability"] = stability.stable_step - stability.first_step
        probabilities = probabilities.merge(stability[identity + ["time_to_stability"]], on=identity)
        probabilities["proxy"] = proxy
        probabilities["classification"] = np.select(
            [probabilities.keep_probability > 0.95, probabilities.keep_probability < 0.05],
            ["always_keep", "always_skip"],
            default="dynamic",
        )
        records.append(probabilities)
    records = pd.concat(records, ignore_index=True)
    counts = records.groupby(["proxy", "attention_type", "classification"], observed=True).size().unstack(fill_value=0)
    for column in ("always_skip", "always_keep", "dynamic"):
        if column not in counts:
            counts[column] = 0
    fractions = counts.div(counts.sum(axis=1), axis=0).reset_index()
    fractions = fractions.rename(columns={
        "always_skip": "always_skip_fraction",
        "always_keep": "always_keep_fraction",
        "dynamic": "dynamic_fraction",
    })
    mean_stability = records.groupby(["proxy", "attention_type"], observed=True).time_to_stability.mean().rename("mean_time_to_stability").reset_index()
    fractions = fractions.merge(mean_stability, on=["proxy", "attention_type"])
    return fractions, records
