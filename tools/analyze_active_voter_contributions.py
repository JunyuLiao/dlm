#!/usr/bin/env python3
"""Analyze removed row/tile contributions and Active-Voter policy headroom."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracing.veto_trace_schema import TOKEN_STATE_NAMES, iter_veto_trace_shards  # noqa: E402


PERCENTILES = (0.50, 0.90, 0.95, 0.99, 0.999)


@dataclass
class LogHistogram:
    low: float = 1e-12
    high: float = 1e4
    bins: int = 4096
    counts: np.ndarray = field(default_factory=lambda: np.zeros(4096, dtype=np.int64))
    zeros: int = 0
    total: int = 0
    value_sum: float = 0.0
    maximum: float = 0.0

    def __post_init__(self) -> None:
        self.edges = np.geomspace(self.low, self.high, self.bins + 1)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values) & (values >= 0)]
        if not values.size:
            return
        self.total += int(values.size)
        self.value_sum += float(values.sum(dtype=np.float64))
        self.maximum = max(self.maximum, float(values.max(initial=0.0)))
        positive = values[values > 0]
        self.zeros += int(values.size - positive.size)
        if positive.size:
            self.counts += np.histogram(positive, bins=self.edges)[0]
            self.counts[-1] += int((positive > self.high).sum())

    def quantile(self, q: float) -> float:
        if not self.total:
            return 0.0
        target = math.ceil(q * self.total)
        if target <= self.zeros:
            return 0.0
        index = int(np.searchsorted(np.cumsum(self.counts), target - self.zeros, side="left"))
        index = min(index, self.bins - 1)
        return float(math.sqrt(self.edges[index] * self.edges[index + 1]))

    def summary(self) -> dict[str, object]:
        return {
            "count": self.total,
            "mean": self.value_sum / self.total if self.total else 0.0,
            **{f"p{100*q:g}": self.quantile(q) for q in PERCENTILES},
            "maximum": self.maximum,
            "quantile_method": f"log histogram with {self.bins} bins over [{self.low:g}, {self.high:g}]",
        }


@dataclass
class ContributionGroup:
    mass: LogHistogram = field(default_factory=LogHistogram)
    output_norm: LogHistogram = field(default_factory=LogHistogram)
    relative_output_norm: LogHistogram = field(default_factory=LogHistogram)

    def update(self, mass: np.ndarray, output: np.ndarray, relative: np.ndarray) -> None:
        self.mass.update(mass)
        self.output_norm.update(output)
        self.relative_output_norm.update(relative)

    def summary(self) -> dict[str, object]:
        return {
            "attention_mass": self.mass.summary(),
            "output_norm": self.output_norm.summary(),
            "relative_output_norm": self.relative_output_norm.summary(),
        }


@dataclass
class PolicyAccumulator:
    family: str
    parameter: float
    baseline_retained_row_work: int = 0
    candidate_tiles: int = 0
    candidate_row_work: int = 0
    selected_rows: int = 0
    removed_rows: int = 0
    removed_relative_sum: float = 0.0
    removed_relative_max: float = 0.0
    removed_over_1e3: int = 0
    removed_over_1e2: int = 0
    removed_over_5e2: int = 0
    active_count_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(33, dtype=np.int64)
    )

    def update(
        self,
        valid: np.ndarray,
        baseline_keep: np.ndarray,
        selected: np.ndarray,
        relative: np.ndarray,
    ) -> None:
        valid_count = valid.sum(1)
        self.baseline_retained_row_work += int(valid_count[baseline_keep].sum())
        counts = selected.sum(1)
        candidate = baseline_keep & (counts > 0) & (counts <= 32)
        if not candidate.any():
            return
        self.candidate_tiles += int(candidate.sum())
        self.candidate_row_work += int(valid_count[candidate].sum())
        self.selected_rows += int(counts[candidate].sum())
        np.add.at(self.active_count_histogram, counts[candidate], 1)
        removed = candidate[:, None] & valid & ~selected
        values = relative[removed]
        self.removed_rows += int(values.size)
        if values.size:
            self.removed_relative_sum += float(values.sum(dtype=np.float64))
            self.removed_relative_max = max(
                self.removed_relative_max, float(values.max(initial=0.0))
            )
            self.removed_over_1e3 += int((values > 1e-3).sum())
            self.removed_over_1e2 += int((values > 1e-2).sum())
            self.removed_over_5e2 += int((values > 5e-2).sum())

    def summary(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameter": self.parameter,
            "candidate_tiles": self.candidate_tiles,
            "candidate_row_work": self.candidate_row_work,
            "candidate_work_fraction": (
                self.candidate_row_work / self.baseline_retained_row_work
                if self.baseline_retained_row_work
                else 0.0
            ),
            "selected_rows": self.selected_rows,
            "mean_active_rows": self.selected_rows / self.candidate_tiles if self.candidate_tiles else 0.0,
            "selected_fraction_within_candidates": (
                self.selected_rows / self.candidate_row_work if self.candidate_row_work else 0.0
            ),
            "removed_rows": self.removed_rows,
            "mean_removed_relative_output_norm": (
                self.removed_relative_sum / self.removed_rows if self.removed_rows else 0.0
            ),
            "maximum_removed_relative_output_norm": self.removed_relative_max,
            "removed_fraction_over_0.001": self.removed_over_1e3 / max(self.removed_rows, 1),
            "removed_fraction_over_0.01": self.removed_over_1e2 / max(self.removed_rows, 1),
            "removed_fraction_over_0.05": self.removed_over_5e2 / max(self.removed_rows, 1),
            "active_count_histogram": {
                str(index): int(value)
                for index, value in enumerate(self.active_count_histogram)
                if value
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="0,2,4")
    parser.add_argument("--margins", default="0,0.5,1,2,4,6,8")
    parser.add_argument("--mass-epsilons", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2")
    parser.add_argument("--v-epsilons", default="1e-4,3e-4,1e-3,3e-3,1e-2")
    args = parser.parse_args()
    selected_steps = {int(value) for value in args.steps.split(",")}
    margins = [float(value) for value in args.margins.split(",")]
    mass_epsilons = [float(value) for value in args.mass_epsilons.split(",")]
    v_epsilons = [float(value) for value in args.v_epsilons.split(",")]
    plan = json.loads((args.trace_dir / "collection_plan.json").read_text())
    sample_ids = sorted(plan["context_splits"])
    sample_index = {value: index for index, value in enumerate(sample_ids)}
    step_values = sorted(selected_steps)
    step_index = {value: index for index, value in enumerate(step_values)}

    groups: dict[tuple[str, str], ContributionGroup] = defaultdict(ContributionGroup)
    policies: list[PolicyAccumulator] = []
    for value in margins:
        policies.append(PolicyAccumulator("gap_margin", value))
    for value in mass_epsilons:
        policies.append(PolicyAccumulator("oracle_final_mass", value))
    for value in v_epsilons:
        policies.append(PolicyAccumulator("oracle_mass_times_vmax", value))
        policies.append(PolicyAccumulator("oracle_exact_output_norm", value))

    shape = (len(sample_ids), len(step_values), 32, 32, 128)
    cumulative_relative = np.zeros(shape, dtype=np.float32)
    cumulative_tiles = np.zeros(shape, dtype=np.uint8)
    raw_candidate_tiles = 0
    raw_negative_rows = 0
    records = 0

    for shard_path, payload in iter_veto_trace_shards(args.trace_dir):
        keep = np.isin(payload["metadata"]["denoising_step"], list(selected_steps))
        if not keep.any():
            continue
        metadata = payload["metadata"][keep]
        arrays = {name: values[keep] for name, values in payload.items() if name != "metadata"}
        rows = np.arange(128)[None, :]
        valid = rows < metadata["valid_query_rows"].astype(np.int64)[:, None]
        baseline_keep = ~metadata["exact_blasst_skip"].astype(bool)
        log_score = arrays["row_log_score"].astype(np.float32)
        local_lse = arrays["row_local_logsumexp"].astype(np.float32)
        final_lse = arrays["row_final_logsumexp"].astype(np.float32)
        mass = np.where(valid, np.exp(np.clip(local_lse - final_lse, -80.0, 0.0)), 0.0)
        local_value_norm = arrays["row_local_value_norm"].astype(np.float32)
        output_contribution = mass * local_value_norm
        final_output_norm = np.maximum(arrays["row_final_output_norm"].astype(np.float32), 1e-6)
        relative = output_contribution / final_output_norm

        votes = arrays["row_keep_vote"].astype(bool) & valid
        vote_count = votes.sum(1)
        raw_candidate = baseline_keep & (vote_count > 0) & (vote_count <= 32)
        negative = raw_candidate[:, None] & valid & ~votes
        raw_candidate_tiles += int(raw_candidate.sum())
        raw_negative_rows += int(negative.sum())
        raw_mass = mass[negative]
        raw_output = output_contribution[negative]
        raw_relative = relative[negative]
        groups[("all", "all")].update(raw_mass, raw_output, raw_relative)

        dimensions = (
            ("layer", metadata["layer"]),
            ("head", metadata["head"]),
            ("step", metadata["denoising_step"]),
            ("remaining_mask_ratio", metadata["remaining_mask_ratio"]),
        )
        for dimension, values in dimensions:
            for value in np.unique(values):
                row_scope = values == value
                scope = negative & row_scope[:, None]
                groups[(dimension, str(value))].update(
                    mass[scope], output_contribution[scope], relative[scope]
                )
        states = arrays["row_token_state"].astype(np.uint8)
        for state in np.unique(states[negative]):
            scope = negative & (states == state)
            groups[("token_state", TOKEN_STATE_NAMES[int(state)])].update(
                mass[scope], output_contribution[scope], relative[scope]
            )

        samples = np.array([sample_index[str(value)] for value in metadata["sample_id"]])
        steps = np.array([step_index[int(value)] for value in metadata["denoising_step"]])
        relative_removed = np.where(negative, relative, 0.0)
        tile_removed = negative.astype(np.uint8)
        np.add.at(
            cumulative_relative,
            (samples, steps, metadata["layer"], metadata["head"]),
            relative_removed,
        )
        np.add.at(
            cumulative_tiles,
            (samples, steps, metadata["layer"], metadata["head"]),
            tile_removed,
        )

        threshold = np.log(np.maximum(metadata["threshold"].astype(np.float32), 1e-30))
        policy_index = 0
        for margin in margins:
            selected = valid & (log_score >= threshold[:, None] - margin)
            policies[policy_index].update(valid, baseline_keep, selected, relative)
            policy_index += 1
        for epsilon in mass_epsilons:
            selected = valid & (mass >= epsilon)
            policies[policy_index].update(valid, baseline_keep, selected, relative)
            policy_index += 1
        vmax = metadata["v_max_row_norm"].astype(np.float32)[:, None]
        for epsilon in v_epsilons:
            selected = valid & (mass * vmax >= epsilon)
            policies[policy_index].update(valid, baseline_keep, selected, relative)
            policy_index += 1
            selected = valid & (output_contribution >= epsilon)
            policies[policy_index].update(valid, baseline_keep, selected, relative)
            policy_index += 1
        records += int(metadata.size)
        print(f"processed {shard_path.name}: {metadata.size} records", flush=True)

    accumulated = cumulative_relative[cumulative_tiles > 0]
    accumulated_tiles = cumulative_tiles[cumulative_tiles > 0]
    accumulated_summary = ContributionGroup()
    accumulated_summary.relative_output_norm.update(accumulated)
    tile_count_histogram = np.bincount(accumulated_tiles, minlength=65)
    breakdown = [
        {"dimension": dimension, "value": value, **group.summary()}
        for (dimension, value), group in sorted(groups.items())
    ]
    payload = {
        "trace_dir": str(args.trace_dir),
        "trace_plan": plan,
        "selected_steps": step_values,
        "records": records,
        "raw_gap_tau32": {
            "candidate_tiles": raw_candidate_tiles,
            "negative_voter_rows": raw_negative_rows,
            "removed_contribution_distribution": groups[("all", "all")].summary(),
            "cumulative_relative_norm_upper_bound_per_row_layer": (
                accumulated_summary.relative_output_norm.summary()
            ),
            "removed_tiles_per_affected_row_layer_histogram": {
                str(index): int(value)
                for index, value in enumerate(tile_count_histogram)
                if value
            },
        },
        "breakdown": breakdown,
        "policy_sweep": [policy.summary() for policy in policies],
        "interpretation_limits": [
            "The trace contains one deterministic query tile per context/layer, all KV tiles, 12 contexts, and real nested masked states.",
            "Per-contribution norms do not capture vector cancellation; cumulative relative norms are conservative sums of norms.",
            "Oracle selectors establish statistical headroom only and are not online implementations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "records": records,
        "raw_candidate_tiles": raw_candidate_tiles,
        "negative_voter_rows": raw_negative_rows,
        "policies": len(policies),
    }, indent=2))


if __name__ == "__main__":
    main()
