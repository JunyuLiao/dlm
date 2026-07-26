#!/usr/bin/env python3
"""Replay exact oracle mass pruning on raw LLaDA attention traces.

The raw trace format stores exact per-tile ``(m, l, u)`` sufficient
statistics.  Consequently this script applies all selected removals at once
and exactly renormalizes the remaining attention distribution without needing
the original Q/K tensors or the 8B model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.oracle_mass_pruning import (  # noqa: E402
    ExactTileStatistics,
    aggregate_tile_mass,
    attention_error_metrics,
    candidate_tiles,
    new_maximum_protected_tiles,
    select_cumulative_budget,
    select_independent,
    simulate_ordinary_blasst,
)
from blasst.triton_bidirectional import DiffusionLambdaSchedule  # noqa: E402


STARTING_THRESHOLDS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
BUDGETS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
K_VALUES = (1, 2, 4, 8, 16)
AGGREGATIONS = (
    "all_max",
    "veto_max",
    "masked_max",
    "q90",
    "q95",
    "q99",
    "sum_all",
    "mean_all",
    "sum_masked",
    "mean_masked",
    "sum_mixed",
    "mean_mixed",
)
CUMULATIVE_AGGREGATIONS = ("all_max", "veto_max", "masked_max", "q99", "sum_masked")


@dataclass(frozen=True)
class Configuration:
    mode: str
    k: int
    aggregation: str
    value: float
    scope: str = "all"
    order: str = "score"
    new_max_protection: str = "none"
    oracle_refresh: str = "current_step"

    @property
    def name(self) -> str:
        return (
            f"{self.mode}:k{self.k}:{self.aggregation}:{self.value:g}:"
            f"{self.scope}:{self.order}:newmax-{self.new_max_protection}:"
            f"{self.oracle_refresh}"
        )


@dataclass
class Record:
    path: Path
    step: int
    ratio: float
    phase: str
    masked_rows: torch.Tensor
    statistics: ExactTileStatistics
    dense_output: torch.Tensor


def phase(ratio: float) -> str:
    return "high" if ratio >= 0.75 else "mid" if ratio >= 0.25 else "low"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _query_tile_masks(trace_dir: Path) -> dict[int, torch.Tensor]:
    trajectory = json.loads((trace_dir / "trajectory.json").read_text(encoding="utf-8"))
    query_start = int(trajectory["query_tile"]) * 128
    visible = set(range(int(trajectory["prompt_length"])))
    result: dict[int, torch.Tensor] = {}
    for item in trajectory["steps"]:
        positions = range(query_start, query_start + 128)
        result[int(item["step"])] = torch.tensor(
            [[position not in visible for position in positions]], dtype=torch.bool
        )
        visible.update(int(value) for value in item["revealed_positions"])
    return result


def load_records(trace_dir: Path) -> list[Record]:
    masks = _query_tile_masks(trace_dir)
    result: list[Record] = []
    for path in sorted(trace_dir.glob("raw-*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        m = payload["m"].float()
        l = payload["l"].float()
        u = payload["u"].float()
        step = int(payload["step"])
        ratio = float(payload["remaining_mask_ratio"])
        result.append(
            Record(
                path=path,
                step=step,
                ratio=ratio,
                phase=phase(ratio),
                masked_rows=masks[step],
                statistics=ExactTileStatistics(m, l, u),
                dense_output=payload["final_attention_output"].float(),
            )
        )
    if not result:
        raise ValueError(f"no raw trace files found in {trace_dir}")
    return result


def configurations(args: argparse.Namespace) -> list[Configuration]:
    result: list[Configuration] = []
    if not args.skip_independent:
        for k in K_VALUES:
            for aggregation in AGGREGATIONS:
                for threshold in STARTING_THRESHOLDS:
                    result.append(Configuration("independent", k, aggregation, threshold))
    if not args.skip_cumulative:
        for k in K_VALUES:
            for aggregation in CUMULATIVE_AGGREGATIONS:
                for budget in BUDGETS:
                    for order in ("score", "traversal", "masked_mass"):
                        result.append(
                            Configuration("cumulative", k, aggregation, budget, "all", order)
                        )
        # Budget-scope and running-maximum ablations on the conservative score.
        for k in K_VALUES:
            for budget in BUDGETS:
                for scope in ("masked", "mixed"):
                    result.append(
                        Configuration("cumulative", k, "all_max", budget, scope, "score")
                    )
                for protection in ("all", "masked"):
                    result.append(
                        Configuration(
                            "cumulative", k, "all_max", budget, "all", "score", protection
                        )
                    )
        # Fixed and one-step-stale masks isolate staleness from current-step
        # oracle quality.
        for budget in BUDGETS:
            for refresh in ("fixed", "previous_step"):
                result.append(
                    Configuration(
                        "cumulative", 8, "all_max", budget, "all", "score", "none", refresh
                    )
                )
    return result


def add_empirical_thresholds(
    configs: list[Configuration], records: list[Record]
) -> list[Configuration]:
    """Add observed candidate-score percentiles without replacing log points."""
    empirical: list[Configuration] = []
    schedule = DiffusionLambdaSchedule()
    for k in K_VALUES:
        for aggregation in AGGREGATIONS:
            values: list[torch.Tensor] = []
            for record in records:
                decisions = simulate_ordinary_blasst(
                    record.statistics, schedule.threshold(record.ratio)
                )
                candidate = candidate_tiles(decisions, k)
                score = aggregate_tile_mass(
                    record.statistics.final_mass,
                    decisions,
                    aggregation,  # type: ignore[arg-type]
                    masked_rows=record.masked_rows,
                )
                selected = score[candidate & torch.isfinite(score)]
                if selected.numel():
                    values.append(selected)
            if not values:
                continue
            joined = torch.cat(values)
            for value in torch.quantile(
                joined, torch.tensor((0.05, 0.25, 0.50, 0.75, 0.95))
            ).tolist():
                empirical.append(Configuration("independent", k, aggregation, float(value)))
    unique = {config.name: config for config in (*configs, *empirical)}
    return list(unique.values())


def select_for_record(
    config: Configuration,
    record: Record,
    decisions,
    *,
    fixed: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mass = record.statistics.final_mass
    candidates = candidate_tiles(decisions, config.k)
    protected = new_maximum_protected_tiles(
        decisions, config.new_max_protection, record.masked_rows
    )
    score = aggregate_tile_mass(
        mass,
        decisions,
        config.aggregation,  # type: ignore[arg-type]
        masked_rows=record.masked_rows,
    )
    stale = fixed.get(config.name)
    if config.mode == "independent":
        current = select_independent(candidates, score, config.value, protected=protected)
    else:
        current, _ = select_cumulative_budget(
            candidates,
            mass,
            budget=config.value,
            scope=config.scope,  # type: ignore[arg-type]
            masked_rows=record.masked_rows,
            order=config.order,  # type: ignore[arg-type]
            score=score,
            traversal=decisions.traversal,
            protected=protected,
        )
    if config.oracle_refresh == "fixed":
        if stale is None:
            fixed[config.name] = current.clone()
        selected = fixed[config.name].clone() & decisions.keep
    elif config.oracle_refresh == "previous_step":
        selected = current if stale is None else stale.clone() & decisions.keep
        fixed[config.name] = current.clone()
    else:
        selected = current
    return selected, score


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run(records: list[Record], configs: list[Configuration]) -> tuple[list[dict], list[dict], dict]:
    schedule = DiffusionLambdaSchedule()
    fixed: dict[str, torch.Tensor] = {}
    per_step: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []
    for record in records:
        threshold = schedule.threshold(record.ratio)
        decisions = simulate_ordinary_blasst(record.statistics, threshold)
        baseline = record.statistics.compose(decisions.keep)
        dense_reconstructed = record.statistics.compose(torch.ones_like(decisions.keep))
        integrity.append(
            {
                "step": record.step,
                "ratio": record.ratio,
                "phase": record.phase,
                "dense_trace_max_abs": float(
                    (dense_reconstructed - record.dense_output).abs().amax()
                ),
                "baseline_physical_sparsity": float((~decisions.keep).float().mean()),
                "baseline_vs_dense_mean_relative": attention_error_metrics(
                    baseline, dense_reconstructed, masked_rows=record.masked_rows
                ).mean_relative_error,
            }
        )
        for config in configs:
            selected, score = select_for_record(config, record, decisions, fixed=fixed)
            output = record.statistics.compose(decisions.keep & ~selected)
            metrics = attention_error_metrics(
                output, baseline, masked_rows=record.masked_rows
            ).to_dict()
            candidate = candidate_tiles(decisions, config.k)
            removed = int(selected.sum())
            total = decisions.keep.numel()
            kept = int(decisions.keep.sum())
            row = {
                "configuration": config.name,
                **asdict(config),
                "step": record.step,
                "mask_ratio": record.ratio,
                "phase": record.phase,
                "blasst_threshold": threshold,
                "total_tiles": total,
                "baseline_kept_tiles": kept,
                "baseline_skipped_tiles": total - kept,
                "candidate_tiles": int(candidate.sum()),
                "additional_skipped_tiles": removed,
                "additional_physical_sparsity": removed / total,
                "fraction_of_baseline_kept_removed": removed / kept if kept else 0.0,
                "removed_new_maximum_tiles": int(
                    (selected & decisions.introduced_new_max_rows.any(-1)).sum()
                ),
                "maximum_selected_score": float(score[selected].amax()) if removed else 0.0,
                "maximum_removed_row_mass": float(
                    record.statistics.final_mass[selected].amax()
                ) if removed else 0.0,
                **metrics,
            }
            per_step.append(row)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in per_step:
        grouped.setdefault(str(row["configuration"]), []).append(row)
    summary: list[dict[str, object]] = []
    for name, rows in grouped.items():
        first = rows[0]
        total = sum(int(row["total_tiles"]) for row in rows)
        kept = sum(int(row["baseline_kept_tiles"]) for row in rows)
        removed = sum(int(row["additional_skipped_tiles"]) for row in rows)
        summary.append(
            {
                "configuration": name,
                **{key: first[key] for key in asdict(configs[0])},
                "steps": len(rows),
                "total_tiles": total,
                "baseline_kept_tiles": kept,
                "additional_skipped_tiles": removed,
                "additional_physical_sparsity": removed / total,
                "fraction_of_baseline_kept_removed": removed / kept if kept else 0.0,
                "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in rows),
                "mean_relative_error": _mean(float(row["mean_relative_error"]) for row in rows),
                "worst_mean_row_cosine": min(float(row["mean_row_cosine"]) for row in rows),
                "minimum_row_cosine": min(float(row["minimum_row_cosine"]) for row in rows),
                "masked_mean_relative_error": _mean(
                    float(row["masked_mean_relative_error"]) for row in rows
                ),
                "visible_mean_relative_error": _mean(
                    float(row["visible_mean_relative_error"]) for row in rows
                ),
                "masked_worst_mean_row_cosine": min(
                    float(row["masked_mean_row_cosine"]) for row in rows
                ),
                "removed_new_maximum_tiles": sum(
                    int(row["removed_new_maximum_tiles"]) for row in rows
                ),
                "first_nonzero_error_step": next(
                    (
                        int(row["step"])
                        for row in rows
                        if float(row["maximum_absolute_error"]) > 0
                    ),
                    -1,
                ),
            }
        )
    return per_step, summary, {"steps": integrity}


def pareto(rows: list[dict], *, error_key: str = "mean_relative_error") -> list[dict]:
    ordered = sorted(rows, key=lambda row: (-float(row["additional_physical_sparsity"]), float(row[error_key])))
    result = []
    best_error = math.inf
    for row in ordered:
        error = float(row[error_key])
        if error < best_error:
            result.append(row)
            best_error = error
    return sorted(result, key=lambda row: float(row["additional_physical_sparsity"]))


def plot_results(summary: list[dict], per_step: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    current = [row for row in summary if row["oracle_refresh"] == "current_step"]
    fig, axis = plt.subplots(figsize=(8, 5))
    for mode, marker in (("independent", "o"), ("cumulative", "s")):
        points = [row for row in current if row["mode"] == mode and row["aggregation"] == "all_max"]
        axis.scatter(
            [100 * float(row["additional_physical_sparsity"]) for row in points],
            [float(row["mean_relative_error"]) for row in points],
            s=12,
            alpha=0.55,
            marker=marker,
            label=mode,
        )
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_xlabel("additional physical sparsity (%)")
    axis.set_ylabel("mean relative attention-output error")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_sparsity.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for aggregation in ("all_max", "veto_max", "masked_max", "q99", "sum_masked"):
        points = pareto(
            [
                row
                for row in current
                if row["mode"] == "cumulative"
                and row["aggregation"] == aggregation
                and row["scope"] == "all"
                and row["order"] == "score"
                and row["new_max_protection"] == "none"
            ]
        )
        axis.plot(
            [100 * float(row["additional_physical_sparsity"]) for row in points],
            [float(row["mean_relative_error"]) for row in points],
            marker=".",
            label=aggregation,
        )
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_xlabel("additional physical sparsity (%)")
    axis.set_ylabel("mean relative attention-output error")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregation_pareto.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for k in K_VALUES:
        points = pareto(
            [
                row
                for row in current
                if row["aggregation"] == "all_max"
                and int(row["k"]) == k
                and row["new_max_protection"] == "none"
            ]
        )
        axis.plot(
            [100 * float(row["additional_physical_sparsity"]) for row in points],
            [float(row["mean_relative_error"]) for row in points],
            marker=".",
            label=f"K={k}",
        )
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_xlabel("additional physical sparsity (%)")
    axis.set_ylabel("mean relative attention-output error")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "k_pareto.png", dpi=180)
    plt.close(fig)

    phase_groups: dict[tuple[str, str], list[dict]] = {}
    for row in per_step:
        if (
            row["aggregation"] == "all_max"
            and int(row["k"]) == 8
            and row["new_max_protection"] == "none"
            and row["oracle_refresh"] == "current_step"
        ):
            phase_groups.setdefault((str(row["configuration"]), str(row["phase"])), []).append(row)
    fig, axis = plt.subplots(figsize=(8, 5))
    for phase_name in ("high", "mid", "low"):
        points = []
        for (configuration, value), items in phase_groups.items():
            if value != phase_name:
                continue
            points.append(
                (
                    100 * _mean(float(item["additional_physical_sparsity"]) for item in items),
                    _mean(float(item["mean_relative_error"]) for item in items),
                )
            )
        points.sort()
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=12,
            alpha=0.6,
            label=phase_name,
        )
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_xlabel("additional physical sparsity within phase (%)")
    axis.set_ylabel("mean relative attention-output error")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_pareto.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("artifacts/certified_omission_pilot_h100_v2/context-000"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/oracle_mass_experiment"))
    parser.add_argument("--skip-independent", action="store_true")
    parser.add_argument("--skip-cumulative", action="store_true")
    args = parser.parse_args()
    records = load_records(args.trace_dir)
    configs = configurations(args)
    if not args.skip_independent:
        configs = add_empirical_thresholds(configs, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_step, summary, integrity = run(records, configs)
    _write_csv(args.output_dir / "per_step.csv", per_step)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "pareto.csv", pareto(summary))
    (args.output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "blasst-oracle-mass-experiment-v1",
        "source": str(args.trace_dir),
        "records": len(records),
        "configurations": len(configs),
        "thresholds": STARTING_THRESHOLDS,
        "budgets": BUDGETS,
        "k_values": K_VALUES,
        "limitations": [
            "raw trace covers one development prompt, layer 0, head 0, and one query tile",
            "attention-output replay cannot reconstruct hidden states, logits, or generated tokens",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(summary, per_step, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
