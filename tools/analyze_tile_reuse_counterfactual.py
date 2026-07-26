#!/usr/bin/env python3
"""Exact one-tile cache-replacement effects from raw reuse trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.tile_reuse import AttentionTileStatistics, compose_attention_tile_statistics


PATTERN = re.compile(
    r"raw-(?P<request>.+)-step(?P<step>\d+)-layer(?P<layer>\d+)-qtile(?P<qtile>\d+)\.pt"
)
THRESHOLDS = (1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)


def one_tile_effect(current: dict[str, object], previous: dict[str, object]) -> torch.Tensor:
    """Maximum-row relative full-output error for replacing each tile alone."""

    cm = current["m"].float()  # type: ignore[union-attr]
    cl = current["l"].float()  # type: ignore[union-attr]
    cu = current["u"].float()  # type: ignore[union-attr]
    pm = previous["m"].float()  # type: ignore[union-attr]
    pl = previous["l"].float()  # type: ignore[union-attr]
    pu = previous["u"].float()  # type: ignore[union-attr]
    tiles = [
        AttentionTileStatistics(cm[:, :, tile], cl[:, :, tile], cu[:, :, tile])
        for tile in range(cm.shape[2])
    ]
    aggregate = compose_attention_tile_statistics(tiles)
    baseline = aggregate.normalized_output()
    global_m = aggregate.m[:, :, None]
    current_scale = torch.exp(cm - global_m)
    without_l = aggregate.l[:, :, None] - current_scale * cl
    without_u = aggregate.u[:, :, None] - current_scale[..., None] * cu
    replacement_m = torch.maximum(global_m, pm)
    remaining_scale = torch.exp(global_m - replacement_m)
    previous_scale = torch.exp(pm - replacement_m)
    mixed_l = remaining_scale * without_l + previous_scale * pl
    mixed_u = remaining_scale[..., None] * without_u + previous_scale[..., None] * pu
    mixed = mixed_u / mixed_l.clamp_min(1e-30)[..., None]
    relative = torch.linalg.vector_norm(mixed - baseline[:, :, None], dim=-1) / (
        torch.linalg.vector_norm(baseline[:, :, None], dim=-1) + 1e-6
    )
    return relative.amax(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.trace_dir / "collection_plan.json").read_text(encoding="utf-8"))
    split_map = plan["context_splits"]
    groups: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    for path in args.trace_dir.glob("context-*/raw-*.pt"):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("request"), int(match.group("layer")), int(match.group("qtile")))
        groups.setdefault(key, []).append((int(match.group("step")), path))
    records: list[dict[str, object]] = []
    for (request, layer, query_tile), paths in groups.items():
        previous = None
        for step, path in sorted(paths):
            current = torch.load(path, map_location="cpu", weights_only=False)
            if previous is not None:
                effect = one_tile_effect(current, previous)[0, 0].numpy()
                local_current = current["tile_normalized_output"].float()[0, 0]
                local_previous = previous["tile_normalized_output"].float()[0, 0]
                local = torch.linalg.vector_norm(local_current - local_previous, dim=-1) / (
                    torch.linalg.vector_norm(local_current, dim=-1) + 1e-6
                )
                local = local.amax(-1).numpy()
                for tile in range(len(effect)):
                    records.append(
                        {
                            "request_id": request,
                            "split": split_map[request],
                            "layer": layer,
                            "step": step,
                            "remaining_mask_ratio": float(current["remaining_mask_ratio"]),
                            "query_tile": query_tile,
                            "kv_tile": tile,
                            "local_tile_output_error": float(local[tile]),
                            "one_tile_full_output_error": float(effect[tile]),
                        }
                    )
            previous = current
    errors = np.asarray([row["one_tile_full_output_error"] for row in records])
    local = np.asarray([row["local_tile_output_error"] for row in records])
    layers = np.asarray([row["layer"] for row in records])
    splits = np.asarray([row["split"] for row in records])
    ratios = np.asarray([row["remaining_mask_ratio"] for row in records])
    result: dict[str, object] = {
        "records": len(records),
        "raw_layers": sorted(set(int(value) for value in layers)),
        "local_error_quantiles": {
            str(q): float(np.quantile(local, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "one_tile_full_output_error_quantiles": {
            str(q): float(np.quantile(errors, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "oracle_reuse": {
            str(threshold): float(np.mean(errors <= threshold)) for threshold in THRESHOLDS
        },
        "oracle_reuse_heldout": {
            str(threshold): float(np.mean(errors[splits == "heldout"] <= threshold))
            for threshold in THRESHOLDS
        },
        "by_layer": {},
        "by_phase": {},
    }
    for layer in np.unique(layers):
        selected = layers == layer
        result["by_layer"][str(int(layer))] = {  # type: ignore[index]
            "records": int(selected.sum()),
            "median_error": float(np.median(errors[selected])),
            "reuse_at_0.1pct": float(np.mean(errors[selected] <= 0.001)),
            "reuse_at_1pct": float(np.mean(errors[selected] <= 0.01)),
        }
    phase = np.where(ratios >= 0.75, "high", np.where(ratios >= 0.25, "mid", "low"))
    for name in ("high", "mid", "low"):
        selected = phase == name
        result["by_phase"][name] = {  # type: ignore[index]
            "records": int(selected.sum()),
            "median_error": float(np.median(errors[selected])),
            "reuse_at_0.1pct": float(np.mean(errors[selected] <= 0.001)),
            "reuse_at_1pct": float(np.mean(errors[selected] <= 0.01)),
        }
    result["warning"] = (
        "Single-tile counterfactuals are not additive: simultaneously reusing many "
        "individually safe tiles can produce larger cumulative error."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
