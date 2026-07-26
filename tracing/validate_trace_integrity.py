#!/usr/bin/env python3
"""Fail closed on duplicate or relation-incomplete trace collections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.proxy_relationships import joined_pairs
from tracing.trace_schema import load_trace_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = load_trace_directory(args.trace_dir)
    identity_fields = ["sample_id", "denoising_iteration", "layer", "head", "query_tile", "kv_tile"]
    spatial_fields = ["sample_id", "query_tile", "kv_tile"]
    _, identity_counts = np.unique(records[identity_fields], return_counts=True)
    spatial, spatial_counts = np.unique(records[spatial_fields], return_counts=True)
    incomplete = []
    for key, count in zip(spatial, spatial_counts):
        selected = records[records["sample_id"] == key["sample_id"]]
        expected = (
            len(np.unique(selected["denoising_iteration"]))
            * len(np.unique(selected["layer"]))
            * len(np.unique(selected["head"]))
        )
        if int(count) != expected:
            incomplete.append({
                "sample_id": str(key["sample_id"]),
                "query_tile": int(key["query_tile"]),
                "kv_tile": int(key["kv_tile"]),
                "records": int(count),
                "expected": expected,
            })
    previous_step_pairs = len(joined_pairs(records, "previous_step"))
    previous_layer_pairs = len(joined_pairs(records, "previous_layer"))
    result = {
        "records": int(len(records)),
        "duplicate_identities": int((identity_counts > 1).sum()),
        "sampled_spatial_coordinates": int(len(spatial)),
        "incomplete_spatial_coordinates": len(incomplete),
        "incomplete_examples": incomplete[:20],
        "previous_step_joined_pairs": previous_step_pairs,
        "previous_layer_joined_pairs": previous_layer_pairs,
        "valid": not bool((identity_counts > 1).any()) and not incomplete and len(records) > 0,
    }
    output = args.output or Path(args.trace_dir) / "integrity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
