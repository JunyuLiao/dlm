#!/usr/bin/env python3
"""Analyze rich veto traces and calibrate offline approximation policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.veto_pruning import (
    STATE_MASKED,
    STATE_PREFIX,
    STATE_STABLE_VISIBLE,
    analytical_safe_mask,
    apply_group_thresholds,
    calibrated_thresholds,
    online_attention_mass,
    policy_metrics,
    state_weight,
)
from tracing.veto_trace_schema import TOKEN_STATE_NAMES, iter_veto_trace_shards


VETO_BINS = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
)


def row_quantile(values: np.ndarray, q: float, valid: np.ndarray) -> np.ndarray:
    # The first traversed tile legitimately has +inf row score.  Finite
    # sentinels preserve ordering while avoiding inf-inf interpolation warnings.
    finite = np.nan_to_num(values, nan=np.nan, posinf=80.0, neginf=-80.0)
    work = np.where(valid, finite, np.nan)
    return np.nanquantile(work, q, axis=1).astype(np.float32)


def row_max(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.max(np.where(selected, values, -np.inf), axis=1).astype(np.float32)


def row_sum(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.sum(np.where(selected, values, 0.0), axis=1, dtype=np.float64).astype(np.float32)


def concatenate(items: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values) for name, values in items.items()}


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-unsafe-wilson", type=float, default=0.01)
    parser.add_argument("--minimum-calibration-support", type=int, default=256)
    parser.add_argument(
        "--steps",
        default="1,3,5",
        help="adjacent target steps; source-only states are omitted by default",
    )
    parser.add_argument(
        "--new-max-protection",
        choices=("all", "critical", "masked", "none"),
        default="all",
        help="rows whose new running maxima may never be removed",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_steps = {int(value) for value in args.steps.split(",")}

    plan = json.loads((args.trace_dir / "collection_plan.json").read_text(encoding="utf-8"))
    split_names = ("calibration", "heldout", "final_benchmark")
    split_code = {name: index for index, name in enumerate(split_names)}
    context_splits = plan["context_splits"]

    base_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    score_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    direct_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    records = 0

    for shard_path, payload in iter_veto_trace_shards(args.trace_dir):
        keep_records = np.isin(payload["metadata"]["denoising_step"], list(selected_steps))
        if not keep_records.any():
            continue
        payload = {
            name: values[keep_records]
            for name, values in payload.items()
        }
        metadata = payload["metadata"]
        count = metadata.size
        if not count:
            continue
        valid_rows = metadata["valid_query_rows"].astype(np.int64)
        row_indices = np.arange(payload["row_token_state"].shape[1])[None, :]
        valid = row_indices < valid_rows[:, None]
        states = payload["row_token_state"].astype(np.uint8)
        votes = payload["row_keep_vote"].astype(bool) & valid
        baseline_skip = metadata["exact_blasst_skip"].astype(bool)
        baseline_keep = ~baseline_skip
        new_max = metadata["introduced_new_max"].astype(bool)
        veto_count = votes.sum(1).astype(np.int16)

        local_lse = payload["row_local_logsumexp"].astype(np.float32)
        final_lse = payload["row_final_logsumexp"].astype(np.float32)
        oracle_mass = np.exp(np.clip(local_lse - final_lse, -80.0, 0.0))
        oracle_mass = np.where(valid, oracle_mass, 0.0)
        online_mass = online_attention_mass(
            local_lse,
            payload["row_running_max_before"].astype(np.float32),
            payload["row_running_sum_before"].astype(np.float32),
        )
        online_mass = np.where(valid, online_mass, 0.0)
        output_norm = np.maximum(
            payload["row_final_output_norm"].astype(np.float32), 1e-6
        )
        local_value_norm = payload["row_local_value_norm"].astype(np.float32)
        centroid_error_norm = payload["row_centroid_error_norm"].astype(np.float32)
        relative_error = oracle_mass * local_value_norm / output_norm
        relative_centroid_error = oracle_mass * centroid_error_norm / output_norm
        masked = valid & (states == STATE_MASKED)
        protected = valid & ((states == STATE_MASKED) | (states == STATE_PREFIX))
        nonstable = valid & (states != STATE_STABLE_VISIBLE)
        row_new_max = valid & (payload["row_log_score"].astype(np.float32) > 0.0)
        if args.new_max_protection == "all":
            protected_new_max = new_max
        elif args.new_max_protection == "critical":
            protected_new_max = (
                row_new_max
                & (
                    (states == STATE_MASKED)
                    | (states == 1)
                    | (states == STATE_PREFIX)
                )
            ).any(1)
        elif args.new_max_protection == "masked":
            protected_new_max = (row_new_max & protected).any(1)
        else:
            protected_new_max = np.zeros(count, dtype=bool)
        safe = analytical_safe_mask(relative_error, states, valid, protected_new_max)
        masked_error_max = np.max(np.where(masked, relative_error, 0.0), axis=1)
        masked_error_sum = row_sum(relative_error, masked)
        affected_masked_rows = (masked & (relative_error > 1e-8)).sum(1).astype(np.int16)
        veto_state_counts = np.stack(
            [(votes & (states == state)).sum(1) for state in range(5)], axis=1
        ).astype(np.int16)

        samples = np.array([str(value) for value in metadata["sample_id"]])
        splits = np.array([split_code[context_splits[value]] for value in samples], dtype=np.uint8)
        phases = np.where(
            metadata["remaining_mask_ratio"] >= 0.75,
            0,
            np.where(metadata["remaining_mask_ratio"] >= 0.25, 1, 2),
        ).astype(np.uint8)
        layer_bucket = (metadata["layer"].astype(np.uint8) // 8).astype(np.uint8)
        calibration_group = (phases * 4 + layer_bucket).astype(np.uint8)

        base_values = {
            "split": splits,
            "phase": phases,
            "layer": metadata["layer"].astype(np.uint8),
            "head": metadata["head"].astype(np.uint8),
            "step": metadata["denoising_step"].astype(np.uint8),
            "group": calibration_group,
            "baseline_skip": baseline_skip,
            "new_max": new_max,
            "protected_new_max": protected_new_max,
            "safe": safe,
            "veto_count": veto_count,
            "masked_error_max": masked_error_max.astype(np.float32),
            "masked_error_sum": masked_error_sum.astype(np.float32),
            "affected_masked_rows": affected_masked_rows,
            "veto_mass_max": row_max(oracle_mass, votes),
            "v_max_norm": metadata["v_max_row_norm"].astype(np.float32),
            "v_variance": metadata["v_per_dimension_variance"].astype(np.float32),
            "veto_state_counts": veto_state_counts,
        }
        for name, values in base_values.items():
            base_lists[name].append(values)

        log_score = payload["row_log_score"].astype(np.float32)
        log_work = np.where(
            valid,
            np.nan_to_num(log_score, nan=np.nan, posinf=80.0, neginf=-80.0),
            np.nan,
        )
        requested_quantiles = (0.90, 0.95, 0.97, 0.98, 0.99)
        log_quantiles = np.nanquantile(log_work, requested_quantiles, axis=1)
        for quantile_index, q in enumerate(requested_quantiles):
            score_lists[f"quantile_q{int(q * 100)}"].append(
                log_quantiles[quantile_index].astype(np.float32)
            )

        for prefix, mass in (("oracle_mass", oracle_mass), ("online_mass", online_mass)):
            score_lists[f"{prefix}_max"].append(row_max(mass, valid))
            mass_quantiles = np.nanquantile(
                np.where(valid, mass, np.nan), (0.95, 0.99), axis=1
            )
            score_lists[f"{prefix}_q95"].append(mass_quantiles[0].astype(np.float32))
            score_lists[f"{prefix}_q99"].append(mass_quantiles[1].astype(np.float32))
            score_lists[f"{prefix}_masked_max"].append(row_max(mass, protected))
            score_lists[f"{prefix}_stable_relaxed_max"].append(row_max(mass, nonstable))
            for beta in (0.0, 0.1, 0.25, 0.5, 1.0):
                weights = state_weight(states, beta) * valid
                score_lists[f"{prefix}_weighted_sum_b{beta:g}"].append(
                    np.sum(mass * weights, axis=1, dtype=np.float64).astype(np.float32)
                )

        for mass_name, mass in (("oracle", oracle_mass), ("online", online_mass)):
            mass_max = row_max(mass, valid)
            masked_mass_max = row_max(mass, protected)
            for v_name, field in (
                ("max", "v_max_row_norm"),
                ("mean", "v_mean_row_norm"),
                ("fro", "v_frobenius_per_sqrt_rows"),
            ):
                v_summary = metadata[field].astype(np.float32)
                score_lists[f"{mass_name}_mass_v_{v_name}_max"].append(mass_max * v_summary)
                score_lists[f"{mass_name}_mass_v_{v_name}_masked"].append(
                    masked_mass_max * v_summary
                )
            residual = metadata["v_residual_max_norm"].astype(np.float32)
            score_lists[f"{mass_name}_variance_bound"].append(
                row_max(mass / output_norm, valid) * residual
            )

        score_lists["oracle_exact_output_effect"].append(row_max(relative_error, valid))
        score_lists["oracle_exact_masked_effect"].append(row_max(relative_error, protected))
        score_lists["oracle_centroid_error"].append(row_max(relative_centroid_error, valid))
        score_lists["masked_global_budget"].append(masked_error_sum)

        protection = ~protected_new_max
        for k in (1, 2, 4, 8, 16):
            candidate = baseline_keep & protection & (veto_count <= k)
            direct_lists[f"k_veto_{k}"].append(candidate)
            direct_lists[f"k_veto_{k}_visible_only"].append(
                candidate & (veto_state_counts[:, 0] == 0) & (veto_state_counts[:, 4] == 0)
            )
            direct_lists[f"k_veto_{k}_nonmasked_only"].append(
                candidate & (veto_state_counts[:, 0] == 0)
            )
            direct_lists[f"k_veto_{k}_stable_only"].append(
                candidate & (veto_state_counts[:, 3] == veto_count)
            )
        records += count
        print(f"derived {shard_path.name}: {count} records", flush=True)

    base = concatenate(base_lists)
    scores = concatenate(score_lists)
    direct = concatenate(direct_lists)
    calibration = base["split"] == split_code["calibration"]
    eligible = (~base["baseline_skip"]) & ~base["protected_new_max"]

    threshold_payload: dict[str, dict[str, float]] = {}
    selections: dict[str, np.ndarray] = dict(direct)
    for name, score in scores.items():
        thresholds = calibrated_thresholds(
            score,
            base["safe"],
            eligible,
            base["group"],
            calibration,
            max_wilson_upper=args.max_unsafe_wilson,
            minimum_support=args.minimum_calibration_support,
        )
        threshold_payload[name] = {str(key): value for key, value in thresholds.items()}
        selections[name] = eligible & apply_group_thresholds(score, base["group"], thresholds)

    policy_rows: list[dict[str, object]] = []
    for name, selected in selections.items():
        for split_name in split_names:
            metrics = policy_metrics(
                name,
                split_name,
                selected,
                base["split"] == split_code[split_name],
                base["baseline_skip"],
                base["safe"],
                base["new_max"],
                base["masked_error_max"],
                base["masked_error_sum"],
                base["affected_masked_rows"],
            )
            row = metrics.to_dict()
            row["passes_offline_work_gate"] = bool(
                (
                    metrics.additional_physical_sparsity >= 0.20
                    or metrics.downstream_work_reduction >= 0.20
                )
                and metrics.half_downstream_kernel_speedup >= 1.05
            )
            policy_rows.append(row)

    veto_rows: list[dict[str, object]] = []
    kept = ~base["baseline_skip"]
    for dimension, values, labels in (
        ("all", np.zeros(records, dtype=np.int16), {0: "all"}),
        ("phase", base["phase"], {0: "high", 1: "mid", 2: "low"}),
        ("layer", base["layer"], None),
        ("head", base["head"], None),
        ("step", base["step"], None),
    ):
        for value in np.unique(values):
            scope = kept & (values == value)
            for label, lower, upper in VETO_BINS:
                selected = scope & (base["veto_count"] >= lower) & (base["veto_count"] <= upper)
                count = int(selected.sum())
                veto_rows.append(
                    {
                        "dimension": dimension,
                        "value": labels.get(int(value), str(int(value))) if labels else int(value),
                        "veto_bin": label,
                        "kept_tiles": int(scope.sum()),
                        "tiles": count,
                        "fraction_of_kept": count / max(int(scope.sum()), 1),
                        "mean_veto_attention_mass": float(
                            np.mean(base["veto_mass_max"][selected]) if count else 0.0
                        ),
                        "new_max_fraction": float(np.mean(base["new_max"][selected]) if count else 0.0),
                        "mean_v_max_norm": float(np.mean(base["v_max_norm"][selected]) if count else 0.0),
                        "mean_v_variance": float(np.mean(base["v_variance"][selected]) if count else 0.0),
                    }
                )

    state_rows: list[dict[str, object]] = []
    for state, state_name in TOKEN_STATE_NAMES.items():
        counts = base["veto_state_counts"][:, state]
        state_rows.append(
            {
                "token_state": state_name,
                "veto_rows": int(counts[kept].sum()),
                "fraction_of_veto_rows": float(
                    counts[kept].sum() / max(base["veto_count"][kept].sum(), 1)
                ),
                "tiles_with_state_veto": int((kept & (counts > 0)).sum()),
            }
        )

    k_headroom = []
    for k in (1, 2, 4, 8, 16):
        selected = kept & (base["veto_count"] <= k)
        k_headroom.append(
            {
                "k": k,
                "tiles": int(selected.sum()),
                "fraction_of_kept_tiles": float(selected.sum() / max(kept.sum(), 1)),
                "additional_physical_sparsity": float(selected.sum() / records),
                "fraction_introducing_new_max": float(
                    np.mean(base["new_max"][selected]) if selected.any() else 0.0
                ),
            }
        )

    heldout_rows = [row for row in policy_rows if row["split"] == "heldout"]
    heldout_rows.sort(
        key=lambda row: (
            bool(row["passes_offline_work_gate"]),
            float(row["additional_physical_sparsity"]),
        ),
        reverse=True,
    )
    correlation_scope = (
        (base["split"] == split_code["heldout"])
        & ~base["baseline_skip"]
        & np.isfinite(scores["oracle_exact_output_effect"])
    )
    target_log = np.log10(
        np.maximum(scores["oracle_exact_output_effect"][correlation_scope], 1e-12)
    )
    predictor_correlations = []
    for name, score in scores.items():
        values = score[correlation_scope]
        finite = np.isfinite(values) & np.isfinite(target_log)
        if finite.sum() < 2:
            correlation = 0.0
        else:
            transformed = (
                values[finite]
                if np.any(values[finite] < 0)
                else np.log10(np.maximum(values[finite], 1e-12))
            )
            correlation = float(
                np.corrcoef(transformed, target_log[finite])[0, 1]
            )
        predictor_correlations.append(
            {
                "predictor": name,
                "heldout_log_pearson_vs_exact_output_effect": correlation,
                "records": int(finite.sum()),
            }
        )
    predictor_correlations.sort(
        key=lambda row: row["heldout_log_pearson_vs_exact_output_effect"], reverse=True
    )
    summary = {
        "trace_dir": str(args.trace_dir),
        "records": records,
        "selected_steps": sorted(selected_steps),
        "contexts_by_split": {
            name: sum(value == name for value in context_splits.values()) for name in split_names
        },
        "calibration": {
            "grouping": "noise phase x 8-layer bucket",
            "maximum_unsafe_wilson95_upper": args.max_unsafe_wilson,
            "minimum_support_per_group": args.minimum_calibration_support,
            "new_max_protection": args.new_max_protection,
            "analytical_safe_definition": {
                "masked_relative_tile_error": 0.005,
                "newly_revealed": 0.0075,
                "previously_revealed": 0.02,
                "stable_visible": 0.05,
                "prefix": 0.005,
                "warning": "This is a trace-level screening bound, not a model-quality result.",
            },
        },
        "k_veto_headroom": k_headroom,
        "predictor_correlations": predictor_correlations,
        "top_heldout_policies": heldout_rows[:30],
        "offline_gate_passes": sum(bool(row["passes_offline_work_gate"]) for row in heldout_rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "calibrated_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "policy_sweep.csv", policy_rows)
    write_csv(args.output_dir / "veto_distribution.csv", veto_rows)
    write_csv(args.output_dir / "veto_token_states.csv", state_rows)

    try:
        import matplotlib.pyplot as plt

        all_veto = [row for row in veto_rows if row["dimension"] == "all"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(
            [str(row["veto_bin"]) for row in all_veto],
            [100.0 * float(row["fraction_of_kept"]) for row in all_veto],
        )
        axes[0].set_ylabel("Kept physical tiles (%)")
        axes[0].set_xlabel("Veto-row count")
        axes[0].set_title("Why ordinary BLASST retains physical tiles")
        x = [100.0 * float(row["additional_physical_sparsity"]) for row in heldout_rows]
        y = [100.0 * float(row["unsafe_removal_rate"]) for row in heldout_rows]
        colors = ["tab:green" if row["passes_offline_work_gate"] else "tab:blue" for row in heldout_rows]
        axes[1].scatter(x, y, c=colors, alpha=0.65, s=20)
        axes[1].axvline(20.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Additional physical sparsity (percentage points)")
        axes[1].set_ylabel("Analytically unsafe removals (%)")
        axes[1].set_title("Held-out policy trade-off")
        figure.tight_layout()
        figure.savefig(args.output_dir / "veto_policy_summary.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
