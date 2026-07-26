#!/usr/bin/env python3
"""Evaluate cumulative per-row error budgets on complete sampled Q traversals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.veto_pruning import online_attention_mass, wilson_upper
from tracing.veto_trace_schema import iter_veto_trace_shards


PROTECTIONS = ("all", "critical", "masked", "none")
ESTIMATORS = ("oracle_relative_effect", "online_relative_proxy")
RULES = ("sum", "rss", "max")
SPLITS = ("calibration", "heldout", "final_benchmark")
STATE_MULTIPLIERS = np.array((1.0, 1.5, 4.0, 10.0, 1.0), dtype=np.float32)
STATE_LIMITS = np.array((0.005, 0.0075, 0.02, 0.05, 0.005), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", default="0.001,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument(
        "--steps",
        default="1,3,5",
        help="comma-separated target steps; adjacent source-only states are omitted by default",
    )
    args = parser.parse_args()
    budgets = np.array([float(value) for value in args.budgets.split(",")], dtype=np.float32)
    selected_steps = {int(value) for value in args.steps.split(",")}
    if np.any(budgets <= 0):
        raise ValueError("budgets must be positive")

    plan = json.loads((args.trace_dir / "collection_plan.json").read_text(encoding="utf-8"))
    split_code = {name: index for index, name in enumerate(SPLITS)}
    context_splits = plan["context_splits"]
    shape = (len(SPLITS), len(PROTECTIONS), len(ESTIMATORS), len(RULES), len(budgets))
    additional = np.zeros(shape, dtype=np.int64)
    unsafe = np.zeros(shape, dtype=np.int64)
    newmax_removed = np.zeros(shape, dtype=np.int64)
    affected_masked = np.zeros(shape, dtype=np.int64)
    masked_error_sum = np.zeros(shape, dtype=np.float64)
    masked_error_max = np.zeros(shape, dtype=np.float32)
    total = np.zeros(len(SPLITS), dtype=np.int64)
    baseline_skipped = np.zeros(len(SPLITS), dtype=np.int64)
    # One active state per complete (context, step, layer, head, Q tile).
    active: dict[tuple[object, ...], np.ndarray] = {}

    for shard_path, payload in iter_veto_trace_shards(args.trace_dir):
        metadata = payload["metadata"]
        for index, row in enumerate(metadata):
            if int(row["denoising_step"]) not in selected_steps:
                continue
            sample = str(row["sample_id"])
            split = split_code[context_splits[sample]]
            total[split] += 1
            baseline_skip = bool(row["exact_blasst_skip"])
            baseline_skipped[split] += int(baseline_skip)
            valid_count = int(row["valid_query_rows"])
            valid = np.arange(payload["row_token_state"].shape[1]) < valid_count
            states = payload["row_token_state"][index].astype(np.int16)
            safe_states = np.where(valid, states, 0)
            local_lse = payload["row_local_logsumexp"][index].astype(np.float32)
            final_lse = payload["row_final_logsumexp"][index].astype(np.float32)
            oracle_mass = np.exp(np.clip(local_lse - final_lse, -80.0, 0.0))
            online_mass = online_attention_mass(
                local_lse[None, :],
                payload["row_running_max_before"][index].astype(np.float32)[None, :],
                payload["row_running_sum_before"][index].astype(np.float32)[None, :],
            )[0]
            output_norm = np.maximum(
                payload["row_final_output_norm"][index].astype(np.float32), 1e-6
            )
            local_value = payload["row_local_value_norm"][index].astype(np.float32)
            errors = np.stack(
                (
                    oracle_mass * local_value / output_norm,
                    online_mass * local_value / output_norm,
                )
            )
            errors[:, ~valid] = 0.0
            actual_error = errors[0]
            row_newmax = valid & (
                payload["row_log_score"][index].astype(np.float32) > 0.0
            )
            protected_newmax = np.array(
                (
                    row_newmax.any(),
                    (
                        row_newmax
                        & ((states == 0) | (states == 1) | (states == 4))
                    ).any(),
                    (row_newmax & ((states == 0) | (states == 4))).any(),
                    False,
                ),
                dtype=bool,
            )
            row_safe = (~valid) | (actual_error <= STATE_LIMITS[safe_states])
            analytically_safe = row_safe.all() & ~protected_newmax
            group = (
                sample,
                int(row["denoising_step"]),
                int(row["layer"]),
                int(row["head"]),
                int(row["query_tile"]),
            )
            accumulated = active.setdefault(
                group,
                np.zeros(
                    (len(PROTECTIONS), len(ESTIMATORS), len(RULES), len(budgets), valid.size),
                    dtype=np.float32,
                ),
            )
            if not baseline_skip:
                for estimator in range(len(ESTIMATORS)):
                    error = errors[estimator][None, None, :]
                    previous = accumulated[:, estimator]
                    proposed = np.empty_like(previous)
                    proposed[:, 0] = previous[:, 0] + error
                    proposed[:, 1] = np.sqrt(previous[:, 1] ** 2 + error**2)
                    proposed[:, 2] = np.maximum(previous[:, 2], error)
                    per_row_budget = budgets[:, None] * STATE_MULTIPLIERS[safe_states][None, :]
                    permitted = ((proposed <= per_row_budget[None, None, :, :]) | ~valid).all(-1)
                    permitted &= ~protected_newmax[:, None, None]
                    accumulated[:, estimator] = np.where(
                        permitted[..., None], proposed, previous
                    )
                    for protection in range(len(PROTECTIONS)):
                        for rule in range(len(RULES)):
                            chosen = permitted[protection, rule]
                            if not chosen.any():
                                continue
                            target = (split, protection, estimator, rule)
                            additional[target][chosen] += 1
                            unsafe[target][chosen] += int(not analytically_safe[protection])
                            newmax_removed[target][chosen] += int(row_newmax.any())
                            masked = valid & (states == 0)
                            affected = int((masked & (actual_error > 1e-8)).sum())
                            affected_masked[target][chosen] += affected
                            masked_error_sum[target][chosen] += float(actual_error[masked].sum())
                            masked_error_max[target][chosen] = np.maximum(
                                masked_error_max[target][chosen],
                                float(actual_error[masked].max(initial=0.0)),
                            )
            if int(row["traversal_index"]) == int(row["sequence_length"]) // 64 - 1:
                del active[group]
        print(f"processed {shard_path.name}: active_groups={len(active)}", flush=True)

    if active:
        raise RuntimeError(f"{len(active)} incomplete query traversals remain")
    rows: list[dict[str, object]] = []
    for split, split_name in enumerate(SPLITS):
        kept = int(total[split] - baseline_skipped[split])
        for protection, protection_name in enumerate(PROTECTIONS):
            for estimator, estimator_name in enumerate(ESTIMATORS):
                for rule, rule_name in enumerate(RULES):
                    for budget_index, budget in enumerate(budgets):
                        key = (split, protection, estimator, rule, budget_index)
                        added = int(additional[key])
                        reduction = added / kept if kept else 0.0
                        rows.append(
                            {
                                "split": split_name,
                                "new_max_protection": protection_name,
                                "estimator": estimator_name,
                                "accumulation_rule": rule_name,
                                "base_budget": float(budget),
                                "total_tiles": int(total[split]),
                                "baseline_skipped_tiles": int(baseline_skipped[split]),
                                "additional_skipped_tiles": added,
                                "additional_physical_sparsity": added / max(int(total[split]), 1),
                                "downstream_work_reduction": reduction,
                                "unsafe_removals": int(unsafe[key]),
                                "unsafe_removal_rate": int(unsafe[key]) / max(added, 1),
                                "unsafe_wilson95_upper": wilson_upper(int(unsafe[key]), added),
                                "new_max_removals": int(newmax_removed[key]),
                                "affected_masked_rows": int(affected_masked[key]),
                                "mean_masked_relative_error": float(
                                    masked_error_sum[key] / max(int(affected_masked[key]), 1)
                                ),
                                "maximum_masked_relative_error": float(masked_error_max[key]),
                                "half_downstream_kernel_speedup": 1.0
                                / max(1.0 - 0.5 * reduction, 1e-12),
                                "passes_offline_work_gate": bool(
                                    (
                                        added / max(int(total[split]), 1) >= 0.20
                                        or reduction >= 0.20
                                    )
                                    and 1.0 / max(1.0 - 0.5 * reduction, 1e-12) >= 1.05
                                ),
                            }
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "heldout_gate_passes": sum(
            row["split"] == "heldout" and row["passes_offline_work_gate"] for row in rows
        ),
        "state_multipliers": STATE_MULTIPLIERS.tolist(),
        "selected_steps": sorted(selected_steps),
        "note": (
            "oracle_relative_effect is an offline upper bound; online_relative_proxy uses "
            "the final output norm only for cross-layer normalization and is not directly online-computable."
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
