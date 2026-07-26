#!/usr/bin/env python3
"""Transparent analytical traffic/operation model for pre-QK skipping."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPRESENTATIONS = {
    "bit_mask": {"bytes_per_tile": 1 / 8, "predictor_ops": 1},
    "log_score_q4": {"bytes_per_tile": 1 / 2, "predictor_ops": 2},
    "log_score_q8": {"bytes_per_tile": 1, "predictor_ops": 2},
    "score_fp16": {"bytes_per_tile": 2, "predictor_ops": 2},
}


def estimate(
    *,
    sequence_length: int,
    layers: int,
    heads: int,
    head_dim: int,
    q_tile_rows: int,
    kv_tile_cols: int,
    pre_qk_sparsity: float,
    downstream_sparsity: float,
    sources: int,
    representation: str,
) -> dict[str, float | int | str]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    if not 0 <= pre_qk_sparsity <= 1 or not 0 <= downstream_sparsity <= 1:
        raise ValueError("sparsities must be in [0, 1]")
    q_tiles = math.ceil(sequence_length / q_tile_rows)
    kv_tiles = math.ceil(sequence_length / kv_tile_cols)
    physical_tiles = layers * heads * q_tiles * kv_tiles
    qk_flops_per_tile = 2 * q_tile_rows * kv_tile_cols * head_dim
    # A conservative lower bound: one BF16 K tile load per head/tile. It does
    # not credit Q reuse, V avoidance, L2 hits, or downstream softmax/PV work.
    k_bytes_per_tile = kv_tile_cols * head_dim * 2
    spec = REPRESENTATIONS[representation]
    metadata_bytes_per_tile = float(spec["bytes_per_tile"]) * sources
    metadata_total = metadata_bytes_per_tile * physical_tiles
    k_bytes_avoided = pre_qk_sparsity * physical_tiles * k_bytes_per_tile
    qk_flops_avoided = pre_qk_sparsity * physical_tiles * qk_flops_per_tile
    cache_bytes = float(spec["bytes_per_tile"]) * physical_tiles
    break_even = metadata_bytes_per_tile / k_bytes_per_tile
    return {
        "representation": representation,
        "sources": sources,
        "layers": layers,
        "physical_tiles": physical_tiles,
        "pre_qk_sparsity": pre_qk_sparsity,
        "qk_tile_mmas_avoided": pre_qk_sparsity * physical_tiles,
        "qk_flops_avoided": qk_flops_avoided,
        "estimated_k_tile_hbm_bytes_avoided": k_bytes_avoided,
        "downstream_blasst_sparsity": downstream_sparsity,
        "metadata_bytes_per_target_tile": metadata_bytes_per_tile,
        "metadata_bytes_loaded": metadata_total,
        "previous_step_cache_bytes": cache_bytes,
        "predictor_operations_per_tile": int(spec["predictor_ops"]) * sources,
        "hbm_break_even_pre_qk_sparsity": break_even,
        "net_k_metadata_bytes_saved": k_bytes_avoided - metadata_total,
        "net_k_traffic_saving_fraction": (k_bytes_avoided - metadata_total)
        / (physical_tiles * k_bytes_per_tile),
        "caveat": "analytical lower-bound traffic model; not a kernel speedup claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-tile-rows", type=int, default=128)
    parser.add_argument("--kv-tile-cols", type=int, default=64)
    parser.add_argument("--pre-qk-sparsity", type=float, required=True)
    parser.add_argument("--downstream-sparsity", type=float, default=0.0)
    parser.add_argument("--sources", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/proxy_cost.json"))
    args = parser.parse_args()
    rows = [
        estimate(
            sequence_length=args.sequence_length,
            layers=args.layers,
            heads=args.heads,
            head_dim=args.head_dim,
            q_tile_rows=args.q_tile_rows,
            kv_tile_cols=args.kv_tile_cols,
            pre_qk_sparsity=args.pre_qk_sparsity,
            downstream_sparsity=args.downstream_sparsity,
            sources=args.sources,
            representation=name,
        )
        for name in REPRESENTATIONS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"representations": rows}, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"representations": rows}, indent=2))


if __name__ == "__main__":
    main()
