#!/usr/bin/env python3
"""Structural and numerical integrity checks for tile-reuse traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.trace_dir / "collection_plan.json").read_text(encoding="utf-8"))
    context_length = int(plan["context_length"])
    kv_tiles = math.ceil(context_length / 64)
    layers = 32 if plan["layers"] is None else len(plan["layers"])
    heads = len(plan["heads"])
    expected_records = int(plan["num_contexts"]) * (int(plan["steps"]) - 1) * layers * heads * kv_tiles
    chunks: dict[str, list[np.ndarray]] = {}
    for path in sorted(args.trace_dir.glob("context-*/tile-reuse-summary.npz")):
        with np.load(path) as payload:
            for name in payload.files:
                if name != "schema_version":
                    chunks.setdefault(name, []).append(payload[name])
    data = {name: np.concatenate(values) for name, values in chunks.items()}
    records = len(data["step"])
    identity = np.rec.fromarrays(
        [
            data[name]
            for name in ("request_id", "layer", "head", "step", "query_tile", "kv_tile")
        ],
        names="request,layer,head,step,query,kv",
    )
    unique = len(np.unique(identity))
    group_identity = np.rec.fromarrays(
        [data[name] for name in ("request_id", "layer", "head", "step", "query_tile")],
        names="request,layer,head,step,query",
    )
    _, counts = np.unique(group_identity, return_counts=True)
    float_fields = [name for name, value in data.items() if value.dtype.kind == "f"]
    nonfinite = {name: int((~np.isfinite(data[name])).sum()) for name in float_fields}
    q_state_sum = sum(data[f"query_state_{state}_fraction"] for state in range(5))
    kv_state_sum = sum(data[f"kv_state_{state}_fraction"] for state in range(5))
    raw_paths = list(args.trace_dir.glob("context-*/raw-*.pt"))
    raw_shapes_valid = True
    for path in raw_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["m"].shape[-2:] != (kv_tiles, 128):
            raw_shapes_valid = False
            break
        if payload["u"].shape[-3:] != (kv_tiles, 128, 128):
            raw_shapes_valid = False
            break
    expected_raw = int(plan["num_contexts"]) * int(plan["steps"]) * len(plan["raw_layers"])
    trajectories = list(args.trace_dir.glob("context-*/trajectory.json"))
    trajectory_steps_valid = all(
        len(json.loads(path.read_text(encoding="utf-8"))["steps"]) == int(plan["steps"])
        for path in trajectories
    )
    result = {
        "records": records,
        "expected_records": expected_records,
        "unique_identities": unique,
        "duplicate_identities": records - unique,
        "query_groups": len(counts),
        "expected_kv_tiles_per_group": kv_tiles,
        "incomplete_query_groups": int((counts != kv_tiles).sum()),
        "nonfinite_values": nonfinite,
        "maximum_query_state_fraction_sum_error": float(np.max(np.abs(q_state_sum - 1))),
        "maximum_kv_state_fraction_sum_error": float(np.max(np.abs(kv_state_sum - 1))),
        "raw_files": len(raw_paths),
        "expected_raw_files": expected_raw,
        "raw_shapes_valid": raw_shapes_valid,
        "trajectories": len(trajectories),
        "trajectory_steps_valid": trajectory_steps_valid,
    }
    result["valid"] = bool(
        records == expected_records
        and unique == records
        and not (counts != kv_tiles).any()
        and all(value == 0 for value in nonfinite.values())
        and np.max(np.abs(q_state_sum - 1)) < 1e-5
        and np.max(np.abs(kv_state_sum - 1)) < 1e-5
        and len(raw_paths) == expected_raw
        and raw_shapes_valid
        and len(trajectories) == int(plan["num_contexts"])
        and trajectory_steps_valid
    )
    output = args.output or args.trace_dir / "integrity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
