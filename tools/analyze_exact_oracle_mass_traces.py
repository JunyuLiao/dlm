#!/usr/bin/env python3
"""Prompt-disjoint analysis using a reconstructed dense final denominator.

Unlike the older veto analysis, this script does not use the stored ordinary-
BLASST final denominator.  It groups every head's 64 local tile
log-normalizers and recomputes ``logsumexp`` across all physical KV tiles.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.veto_pruning import online_attention_mass  # noqa: E402
from tracing.veto_trace_schema import iter_veto_trace_shards  # noqa: E402


FIELDS = (
    "metadata",
    "row_local_logsumexp",
    "row_running_max_before",
    "row_running_sum_before",
    "row_local_value_norm",
    "row_final_output_norm",
    "row_keep_vote",
    "row_token_state",
)


def block_changes(metadata: np.ndarray) -> np.ndarray:
    names = ("sample_id", "denoising_step", "layer", "query_tile")
    changed = np.zeros(len(metadata), dtype=bool)
    changed[0] = True
    for name in names:
        changed[1:] |= metadata[name][1:] != metadata[name][:-1]
    return np.flatnonzero(changed)


def row_max(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.max(np.where(selected, values, -np.inf), axis=1)


def row_q99(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.nanquantile(np.where(valid, values, np.nan), 0.99, axis=1)


def phase(ratio: float) -> str:
    return "high" if ratio >= 0.75 else "mid" if ratio >= 0.25 else "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="1,3,5")
    args = parser.parse_args()
    selected_steps = {int(value) for value in args.steps.split(",")}
    plan = json.loads((args.trace_dir / "collection_plan.json").read_text())
    context_splits = plan["context_splits"]
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    kept_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    new_max_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    layer_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    score_values: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    correlation: dict[str, list[np.ndarray]] = defaultdict(list)
    maximum_mass_sum_error = 0.0
    blocks = 0
    records = 0
    carry: dict[str, np.ndarray] | None = None

    def process(payload: dict[str, np.ndarray]) -> None:
        nonlocal maximum_mass_sum_error, blocks, records
        metadata = payload["metadata"]
        if not len(metadata):
            return
        heads = np.unique(metadata["head"])
        tiles = np.unique(metadata["kv_tile"])
        expected = len(heads) * len(tiles)
        if len(metadata) != expected:
            raise ValueError(
                f"incomplete exact-mass block: got {len(metadata)}, expected {expected}"
            )
        head_index = {int(value): index for index, value in enumerate(heads)}
        tile_index = {int(value): index for index, value in enumerate(tiles)}
        hi = np.array([head_index[int(value)] for value in metadata["head"]])
        ti = np.array([tile_index[int(value)] for value in metadata["kv_tile"]])
        local = payload["row_local_logsumexp"].astype(np.float32)
        valid_count = metadata["valid_query_rows"].astype(np.int64)
        valid = np.arange(local.shape[1])[None, :] < valid_count[:, None]
        grid = np.full((len(heads), len(tiles), local.shape[1]), -np.inf, dtype=np.float32)
        grid[hi, ti] = np.where(valid, local, -np.inf)
        maximum = np.max(grid, axis=1)
        dense_final = maximum + np.log(
            np.sum(np.exp(grid - maximum[:, None]), axis=1, dtype=np.float64)
        )
        exact_mass = np.exp(np.clip(local - dense_final[hi], -80.0, 0.0))
        exact_mass = np.where(valid, exact_mass, 0.0)
        mass_grid = np.zeros_like(grid)
        mass_grid[hi, ti] = exact_mass
        maximum_mass_sum_error = max(
            maximum_mass_sum_error,
            float(np.max(np.abs(mass_grid.sum(axis=1) - 1.0))),
        )
        online_mass = online_attention_mass(
            local,
            payload["row_running_max_before"].astype(np.float32),
            payload["row_running_sum_before"].astype(np.float32),
        )
        online_mass = np.where(valid, online_mass, 0.0)
        votes = payload["row_keep_vote"].astype(bool) & valid
        veto_count = votes.sum(axis=1)
        kept = ~metadata["exact_blasst_skip"].astype(bool)
        masked = valid & (payload["row_token_state"] == 0)
        output_norm = np.maximum(payload["row_final_output_norm"].astype(np.float32), 1e-6)
        effect = row_max(
            exact_mass
            * payload["row_local_value_norm"].astype(np.float32)
            / output_norm,
            valid,
        )
        exact_scores = {
            "all_max": row_max(exact_mass, valid),
            "veto_max": row_max(exact_mass, votes),
            "masked_max": row_max(exact_mass, masked),
            "q99": row_q99(exact_mass, valid),
            "sum_masked": np.sum(np.where(masked, exact_mass, 0.0), axis=1),
        }
        online_scores = {
            "all_max": row_max(online_mass, valid),
            "veto_max": row_max(online_mass, votes),
            "masked_max": row_max(online_mass, masked),
            "q99": row_q99(online_mass, valid),
            "sum_masked": np.sum(np.where(masked, online_mass, 0.0), axis=1),
        }
        split = context_splits[str(metadata["sample_id"][0])]
        noise = phase(float(metadata["remaining_mask_ratio"][0]))
        layer = int(metadata["layer"][0])
        kept_counts[(split, noise)] += int(kept.sum())
        total_counts[split] += len(metadata)
        for k in (1, 2, 4, 8, 16):
            candidate = kept & (veto_count >= 1) & (veto_count <= k)
            counts[(split, noise, k)] += int(candidate.sum())
            new_max_counts[(split, noise, k)] += int(
                (candidate & metadata["introduced_new_max"].astype(bool)).sum()
            )
            layer_counts[(split, layer, k)] += int(candidate.sum())
            selected_score = exact_scores["all_max"][candidate]
            if selected_score.size:
                score_values[(split, k)].append(selected_score.astype(np.float32))
        heldout_candidate = kept & (veto_count >= 1) & (veto_count <= 16)
        if split == "heldout":
            correlation["target"].append(effect[heldout_candidate].astype(np.float32))
            for name, values in exact_scores.items():
                correlation[f"exact_{name}"].append(values[heldout_candidate].astype(np.float32))
            for name, values in online_scores.items():
                correlation[f"online_{name}"].append(values[heldout_candidate].astype(np.float32))
        blocks += 1
        records += len(metadata)

    for shard, payload in iter_veto_trace_shards(args.trace_dir):
        keep = np.isin(payload["metadata"]["denoising_step"], list(selected_steps))
        if not keep.any():
            continue
        chunk = {name: payload[name][keep] for name in FIELDS}
        if carry is not None:
            chunk = {name: np.concatenate((carry[name], chunk[name])) for name in FIELDS}
        changes = block_changes(chunk["metadata"])
        if len(changes) == 1:
            carry = chunk
            continue
        for start, end in zip(changes[:-1], changes[1:]):
            process({name: values[start:end] for name, values in chunk.items()})
        last = int(changes[-1])
        carry = {name: values[last:] for name, values in chunk.items()}
        print(f"processed through {shard.name}: blocks={blocks}", flush=True)
    if carry is not None:
        process(carry)

    correlations = []
    target = np.concatenate(correlation.pop("target"))
    target_log = np.log10(np.maximum(target, 1e-12))
    for name, pieces in correlation.items():
        values = np.concatenate(pieces)
        finite = np.isfinite(values) & np.isfinite(target_log)
        transformed = np.log10(np.maximum(values[finite], 1e-12))
        correlations.append(
            {
                "predictor": name,
                "heldout_log_pearson_vs_mass_value_effect": float(
                    np.corrcoef(transformed, target_log[finite])[0, 1]
                ),
                "records": int(finite.sum()),
            }
        )
    correlations.sort(
        key=lambda row: row["heldout_log_pearson_vs_mass_value_effect"], reverse=True
    )
    headroom = []
    for split in ("calibration", "heldout", "final_benchmark"):
        for noise in ("high", "mid", "low"):
            kept = kept_counts[(split, noise)]
            for k in (1, 2, 4, 8, 16):
                value = counts[(split, noise, k)]
                headroom.append(
                    {
                        "split": split,
                        "phase": noise,
                        "k": k,
                        "candidate_tiles": value,
                        "fraction_of_kept_tiles": value / kept if kept else 0.0,
                        "fraction_introducing_new_max": (
                            new_max_counts[(split, noise, k)] / value if value else 0.0
                        ),
                    }
                )
    percentiles = []
    threshold_sweep = []
    for (split, k), pieces in score_values.items():
        values = np.concatenate(pieces)
        quantiles = np.quantile(values, (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99))
        percentiles.append(
            {
                "split": split,
                "k": k,
                "candidate_tiles": int(values.size),
                **{
                    f"p{int(q * 100):02d}": float(value)
                    for q, value in zip((0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99), quantiles)
                },
            }
        )
        for threshold in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            selected = int((values < threshold).sum())
            threshold_sweep.append(
                {
                    "split": split,
                    "k": k,
                    "threshold": threshold,
                    "selected_tiles": selected,
                    "fraction_of_candidates": selected / len(values),
                    "additional_physical_sparsity": selected / total_counts[split],
                }
            )
    layer_headroom = [
        {"split": split, "layer": layer, "k": k, "candidate_tiles": value}
        for (split, layer, k), value in sorted(layer_counts.items())
    ]
    result = {
        "schema": "blasst-exact-dense-final-mass-trace-analysis-v1",
        "trace_dir": str(args.trace_dir),
        "selected_steps": sorted(selected_steps),
        "records": records,
        "blocks": blocks,
        "contexts_by_split": {
            split: sum(value == split for value in context_splits.values())
            for split in ("calibration", "heldout", "final_benchmark")
        },
        "maximum_dense_mass_sum_error": maximum_mass_sum_error,
        "headroom": headroom,
        "candidate_all_max_mass_percentiles": percentiles,
        "all_max_threshold_sweep": threshold_sweep,
        "predictor_correlations": correlations,
        "layer_headroom": layer_headroom,
        "effect_definition": "max_i exact_dense_mass[i,j] * local_value_norm[i,j] / sparse_output_norm[i]",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "schema", "records", "blocks", "maximum_dense_mass_sum_error", "predictor_correlations"
    )}, indent=2))


if __name__ == "__main__":
    main()
