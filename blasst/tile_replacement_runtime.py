"""Slow full-model reference runtime for compact tile replacement.

This module is a quality harness.  It deliberately performs the complete QK
pass and exact FP32 sufficient-statistic accumulation; it is not a latency or
kernel implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .oracle_mass_pruning import (
    ExactTileStatistics,
    candidate_tiles,
    select_cumulative_budget,
    simulate_ordinary_blasst,
)
from .tile_replacement_predictors import (
    normalized_mass_from_log_z,
    proxy_query_summaries,
    summary_predictions,
    tiled_logits,
)
from .tile_replacement_reference import (
    ReplacementComponents,
    compose_replacement_state,
    exact_conditional_value,
    exact_log_z,
    replacement_rows,
)


ReplacementPolicy = Literal[
    "none",
    "all_candidates",
    "exact_allmax",
    "exact_vetomax",
    "predicted_maxmass",
    "predicted_error",
    "cumulative_exact_mass",
    "cumulative_predicted_error",
]
MassMethod = Literal["exact", "calibrated_maximum"]
ValueMethod = Literal["exact", "proxy_mean_q"]
Scope = Literal["veto_only", "all_rows"]
NewMaximumProtection = Literal["none", "veto_rows", "all_rows"]


@dataclass(frozen=True)
class ReplacementRuntimeConfig:
    k: int = 8
    scope: Scope = "all_rows"
    mass_method: MassMethod = "calibrated_maximum"
    value_method: ValueMethod = "proxy_mean_q"
    summary_slots: int = 8
    policy: ReplacementPolicy = "all_candidates"
    qualification: float = 0.03
    new_maximum_protection: NewMaximumProtection = "none"
    calibrated_maximum_bias: float = 0.9463811033763883
    q_block_size: int = 128
    kv_block_size: int = 64
    protected_layers: tuple[int, ...] = ()


@dataclass
class ReplacementRuntimeStats:
    physical_tiles: int = 0
    baseline_kept_tiles: int = 0
    candidate_tiles: int = 0
    replaced_tiles: int = 0
    replaced_new_maximum_tiles: int = 0
    calls: int = 0
    by_layer: dict[int, int] = field(default_factory=dict)
    by_step: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "physical_replacement_fraction": self.replaced_tiles
            / max(self.physical_tiles, 1),
            "kept_replacement_fraction": self.replaced_tiles
            / max(self.baseline_kept_tiles, 1),
        }


class TileReplacementRuntime:
    def __init__(self, config: ReplacementRuntimeConfig) -> None:
        self.config = config
        self.stats = ReplacementRuntimeStats()
        self.step = 0
        self.remaining_mask_ratio = 1.0

    def set_step(self, step: int, remaining_mask_ratio: float) -> None:
        self.step = int(step)
        self.remaining_mask_ratio = float(remaining_mask_ratio)

    def _select(
        self,
        statistics: ExactTileStatistics,
        decisions,
        candidates: torch.Tensor,
        predicted_log_z: torch.Tensor,
        predicted_u: torch.Tensor,
        baseline_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.policy == "none" or not bool(candidates.any()):
            return torch.zeros_like(candidates)
        exact_mass = statistics.final_mass
        predicted_mass = normalized_mass_from_log_z(predicted_log_z)
        predicted_row_error = predicted_mass * torch.linalg.vector_norm(
            predicted_u.float(), dim=-1
        ) / (torch.linalg.vector_norm(baseline_output.float(), dim=-1)[:, :, None] + 1e-8)
        predicted_tile_error = predicted_row_error.amax(-1)
        policy = self.config.policy
        limit = self.config.qualification
        if policy == "all_candidates":
            selected = candidates
        elif policy == "exact_allmax":
            selected = candidates & (exact_mass.amax(-1) < limit)
        elif policy == "exact_vetomax":
            veto_max = torch.where(decisions.veto_rows, exact_mass, -torch.inf).amax(-1)
            selected = candidates & (veto_max < limit)
        elif policy == "predicted_maxmass":
            selected = candidates & (predicted_mass.amax(-1) < limit)
        elif policy == "predicted_error":
            selected = candidates & (predicted_tile_error < limit)
        elif policy == "cumulative_exact_mass":
            selected, _ = select_cumulative_budget(
                candidates,
                exact_mass,
                budget=limit,
                order="score",
                score=predicted_tile_error,
            )
        elif policy == "cumulative_predicted_error":
            selected, _ = select_cumulative_budget(
                candidates,
                predicted_row_error,
                budget=limit,
                order="score",
                score=predicted_tile_error,
            )
        else:
            raise ValueError(f"unknown replacement policy: {policy}")
        if self.config.new_maximum_protection == "veto_rows":
            protected = (
                decisions.introduced_new_max_rows & decisions.veto_rows
            ).any(-1)
            selected = selected & ~protected
        elif self.config.new_maximum_protection == "all_rows":
            selected = selected & ~decisions.introduced_new_max_rows.any(-1)
        return selected

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        blasst_threshold: float,
        layer: int,
        row_masked: torch.Tensor | None,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if causal:
            raise ValueError("tile-replacement runtime currently targets bidirectional LLaDA")
        if k.shape[1] % self.config.kv_block_size:
            raise ValueError("runtime currently requires complete KV tiles")
        chunks = []
        for q_start in range(0, q.shape[1], self.config.q_block_size):
            q_end = min(q_start + self.config.q_block_size, q.shape[1])
            q_chunk = q[:, q_start:q_end]
            logits = tiled_logits(
                q_chunk,
                k,
                v,
                softmax_scale=softmax_scale,
                kv_block_size=self.config.kv_block_size,
            )
            scores = logits.scores
            m = scores.amax(-1)
            probability = torch.exp(scores - m[..., None])
            probability = torch.where(
                torch.isfinite(probability), probability, torch.zeros_like(probability)
            )
            l = probability.sum(-1)
            u = torch.einsum("bhtrw,bhtwd->bhtrd", probability, logits.values)
            statistics = ExactTileStatistics(m, l, u)
            decisions = simulate_ordinary_blasst(statistics, blasst_threshold)
            candidates = candidate_tiles(decisions, self.config.k)
            baseline = compose_replacement_state(statistics, decisions.keep)

            if self.config.policy == "none" or layer in self.config.protected_layers:
                selected = torch.zeros_like(candidates)
                output = baseline.output
            else:
                if self.config.mass_method == "exact":
                    predicted_log_z = exact_log_z(statistics)
                else:
                    predicted_log_z = statistics.m + self.config.calibrated_maximum_bias
                if self.config.value_method == "exact":
                    predicted_u = exact_conditional_value(statistics)
                else:
                    summaries = proxy_query_summaries(
                        q_chunk,
                        k,
                        v,
                        slots=self.config.summary_slots,
                        softmax_scale=softmax_scale,
                        kv_block_size=self.config.kv_block_size,
                    )
                    _, predicted_u = summary_predictions(
                        q_chunk, summaries, softmax_scale=softmax_scale
                    )
                selected = self._select(
                    statistics,
                    decisions,
                    candidates,
                    predicted_log_z,
                    predicted_u,
                    baseline.output,
                )
                rows = replacement_rows(decisions, selected, self.config.scope)
                replacement = ReplacementComponents(predicted_log_z, predicted_u, rows)
                output = compose_replacement_state(
                    statistics,
                    decisions.keep & ~selected,
                    replacement,
                ).output

            self.stats.physical_tiles += candidates.numel()
            self.stats.baseline_kept_tiles += int(decisions.keep.sum())
            self.stats.candidate_tiles += int(candidates.sum())
            count = int(selected.sum())
            self.stats.replaced_tiles += count
            self.stats.replaced_new_maximum_tiles += int(
                (selected & decisions.introduced_new_max_rows.any(-1)).sum()
            )
            self.stats.calls += 1
            self.stats.by_layer[layer] = self.stats.by_layer.get(layer, 0) + count
            self.stats.by_step[self.step] = self.stats.by_step.get(self.step, 0) + count
            chunks.append(output.transpose(1, 2))
        return torch.cat(chunks, dim=1).to(q.dtype)
