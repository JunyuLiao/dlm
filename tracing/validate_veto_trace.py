#!/usr/bin/env python3
"""Validate identity, traversal completeness, and numerical veto-trace invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .veto_trace_schema import iter_veto_trace_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    identities: set[tuple[object, ...]] = set()
    group_counts: dict[tuple[object, ...], int] = {}
    duplicate_count = 0
    vote_mismatch = 0
    invalid_padding = 0
    records = 0
    expected_kv_tiles: set[int] = set()
    query_delta_by_step: dict[int, list[float]] = {}
    mass_state: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
    mass_failures = 0
    final_lse_mismatches = 0
    maximum_mass_sum_error = 0.0

    for _, payload in iter_veto_trace_shards(args.trace_dir):
        metadata = payload["metadata"]
        votes = payload["row_keep_vote"].astype(bool)
        states = payload["row_token_state"]
        delta = payload["row_query_delta"].astype(np.float32)
        for index, row in enumerate(metadata):
            identity = (
                str(row["sample_id"]),
                int(row["denoising_step"]),
                int(row["layer"]),
                int(row["head"]),
                int(row["query_tile"]),
                int(row["kv_tile"]),
            )
            if identity in identities:
                duplicate_count += 1
            identities.add(identity)
            group = identity[:-1]
            group_counts[group] = group_counts.get(group, 0) + 1
            row_num_kv_tiles = (int(row["sequence_length"]) + 63) // 64
            expected_kv_tiles.add(row_num_kv_tiles)
            valid = int(row["valid_query_rows"])
            if bool(row["exact_blasst_skip"]) != (int(votes[index, :valid].sum()) == 0):
                vote_mismatch += 1
            if np.any(states[index, valid:] != 255) or np.any(votes[index, valid:]):
                invalid_padding += 1
            finite = np.isfinite(delta[index, :valid])
            query_delta_by_step.setdefault(int(row["denoising_step"]), []).append(
                float(finite.mean())
            )
            local_lse = payload["row_local_logsumexp"][index, :valid].astype(np.float32)
            final_lse = payload["row_final_logsumexp"][index, :valid].astype(np.float32)
            if group not in mass_state:
                mass_state[group] = (np.zeros(valid, dtype=np.float64), final_lse.copy())
            mass_sum, expected_final_lse = mass_state[group]
            if not np.allclose(final_lse, expected_final_lse, atol=2e-3, rtol=0.0):
                final_lse_mismatches += 1
            if not bool(row["exact_blasst_skip"]):
                mass_sum += np.exp(np.clip(local_lse - final_lse, -80.0, 0.0))
            if int(row["traversal_index"]) == row_num_kv_tiles - 1:
                error = float(np.max(np.abs(mass_sum - 1.0), initial=0.0))
                maximum_mass_sum_error = max(maximum_mass_sum_error, error)
                mass_failures += int(error > 0.02)
                del mass_state[group]
            records += 1

    if len(expected_kv_tiles) != 1:
        raise SystemExit(f"mixed/invalid sequence lengths: {sorted(expected_kv_tiles)}")
    num_kv_tiles = next(iter(expected_kv_tiles), 0)
    incomplete = sum(count != num_kv_tiles for count in group_counts.values())
    result = {
        "records": records,
        "unique_identities": len(identities),
        "duplicate_identities": duplicate_count,
        "query_row_groups": len(group_counts),
        "expected_kv_tiles_per_group": num_kv_tiles,
        "incomplete_query_row_groups": incomplete,
        "exact_skip_vote_mismatches": vote_mismatch,
        "invalid_padding_records": invalid_padding,
        "normalized_mass_sum_failures": mass_failures,
        "final_logsumexp_mismatches": final_lse_mismatches,
        "maximum_normalized_mass_sum_error": maximum_mass_sum_error,
        "unfinished_mass_groups": len(mass_state),
        "finite_query_delta_fraction_by_step": {
            str(step): float(np.mean(values)) for step, values in query_delta_by_step.items()
        },
    }
    result["valid"] = not any(
        (
            duplicate_count,
            incomplete,
            vote_mismatch,
            invalid_padding,
            mass_failures,
            final_lse_mismatches,
            len(mass_state),
        )
    )
    output = args.output or args.trace_dir / "integrity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
