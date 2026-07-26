"""Training-free reference policies for post-QK physical-tile pruning.

These policies remain separate from the Triton kernel.  They are used for
counterfactual quality evaluation until an offline policy clears the work and
safety gates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Mapping

import torch

from .flash_attention import VetoDecisionContext


Restriction = Literal["any", "visible_only", "nonmasked_only", "stable_only"]
NewMaxProtection = Literal["all", "critical", "masked", "none"]


@dataclass
class VetoPolicyStats:
    total_tiles: int = 0
    baseline_skipped_tiles: int = 0
    additional_skipped_tiles: int = 0
    additional_new_max_tiles: int = 0
    veto_histogram: Counter[int] = field(default_factory=Counter)
    additional_by_layer: Counter[int] = field(default_factory=Counter)

    @property
    def baseline_sparsity(self) -> float:
        return self.baseline_skipped_tiles / self.total_tiles if self.total_tiles else 0.0

    @property
    def additional_sparsity(self) -> float:
        return self.additional_skipped_tiles / self.total_tiles if self.total_tiles else 0.0

    @property
    def downstream_work_reduction(self) -> float:
        retained = self.total_tiles - self.baseline_skipped_tiles
        return self.additional_skipped_tiles / retained if retained else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_tiles": self.total_tiles,
            "baseline_skipped_tiles": self.baseline_skipped_tiles,
            "additional_skipped_tiles": self.additional_skipped_tiles,
            "baseline_physical_sparsity": self.baseline_sparsity,
            "additional_physical_sparsity": self.additional_sparsity,
            "new_physical_sparsity": self.baseline_sparsity + self.additional_sparsity,
            "downstream_work_reduction": self.downstream_work_reduction,
            "additional_new_max_tiles": self.additional_new_max_tiles,
            "veto_histogram": dict(sorted(self.veto_histogram.items())),
            "additional_by_layer": dict(sorted(self.additional_by_layer.items())),
        }


class VetoPolicy:
    def __init__(self, *, new_max_protection: NewMaxProtection = "all") -> None:
        self.new_max_protection = new_max_protection
        self.stats = VetoPolicyStats()

    @staticmethod
    def _protected_new_max(
        context: VetoDecisionContext, mode: NewMaxProtection
    ) -> torch.Tensor:
        new_max = context.row_introduced_new_max
        states = context.row_token_state
        if mode == "all":
            return new_max.any(-1)
        if mode == "none":
            return torch.zeros(new_max.shape[:-1], dtype=torch.bool, device=new_max.device)
        if states is None:
            raise ValueError(f"{mode} new-max protection requires row token states")
        if mode == "critical":
            protected = (states == 0) | (states == 1) | (states == 4)
        elif mode == "masked":
            protected = (states == 0) | (states == 4)
        else:
            raise ValueError(f"unknown new-max protection: {mode}")
        return (new_max & protected).any(-1)

    def _record(
        self,
        layer: int,
        context: VetoDecisionContext,
        decision: torch.Tensor,
    ) -> torch.Tensor:
        baseline = context.baseline_physical_skip
        extra = decision.bool() & ~baseline
        self.stats.total_tiles += int(context.valid_rows.any(-1).sum().item())
        self.stats.baseline_skipped_tiles += int(baseline.sum().item())
        self.stats.additional_skipped_tiles += int(extra.sum().item())
        self.stats.additional_new_max_tiles += int(
            (extra & context.row_introduced_new_max.any(-1)).sum().item()
        )
        self.stats.additional_by_layer[layer] += int(extra.sum().item())
        return extra


class KVetoPolicy(VetoPolicy):
    def __init__(
        self,
        k: int,
        *,
        log_threshold: float | Mapping[int, float],
        restriction: Restriction = "any",
        new_max_protection: NewMaxProtection = "all",
    ) -> None:
        super().__init__(new_max_protection=new_max_protection)
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.log_threshold = log_threshold
        self.restriction = restriction

    def _threshold(self, layer: int) -> float:
        if isinstance(self.log_threshold, Mapping):
            return float(self.log_threshold[layer])
        return float(self.log_threshold)

    def __call__(
        self, layer: int, query_tile: int, kv_tile: int, context: VetoDecisionContext
    ) -> torch.Tensor:
        del query_tile, kv_tile
        votes = context.valid_rows & (context.row_log_score >= self._threshold(layer))
        count = votes.sum(-1)
        candidate = (count > 0) & (count <= self.k) & ~context.baseline_physical_skip
        states = context.row_token_state
        if self.restriction != "any":
            if states is None:
                raise ValueError("token-state restriction requires row token states")
            if self.restriction == "visible_only":
                eligible_rows = (states == 1) | (states == 2) | (states == 3)
            elif self.restriction == "nonmasked_only":
                eligible_rows = states != 0
            elif self.restriction == "stable_only":
                eligible_rows = states == 3
            else:
                raise ValueError(f"unknown restriction: {self.restriction}")
            candidate &= ((~votes) | eligible_rows).all(-1)
        candidate &= ~self._protected_new_max(context, self.new_max_protection)
        extra = self._record(layer, context, candidate)
        for value in count[extra].detach().cpu().tolist():
            self.stats.veto_histogram[int(value)] += 1
        return extra


class ScoreThresholdPolicy(VetoPolicy):
    """Quantile, normalized-mass, V-aware, or masked-budget policy."""

    def __init__(
        self,
        threshold: float | Mapping[int, float],
        *,
        score: Literal[
            "quantile",
            "online_mass",
            "online_mass_v_max",
            "online_mass_v_mean",
            "online_variance_bound",
            "masked_total",
        ] = "online_mass",
        quantile: float = 0.99,
        aggregator: Literal["max", "quantile", "weighted_sum"] = "max",
        row_scope: Literal["all", "masked", "nonstable"] = "all",
        beta: float = 1.0,
        new_max_protection: NewMaxProtection = "all",
    ) -> None:
        super().__init__(new_max_protection=new_max_protection)
        self.threshold = threshold
        self.score = score
        self.quantile = quantile
        self.aggregator = aggregator
        self.row_scope = row_scope
        self.beta = beta

    def _threshold(self, layer: int) -> float:
        if isinstance(self.threshold, Mapping):
            return float(self.threshold[layer])
        return float(self.threshold)

    def _aggregate(
        self, values: torch.Tensor, context: VetoDecisionContext
    ) -> torch.Tensor:
        valid = context.valid_rows
        if self.row_scope != "all":
            if context.row_token_state is None:
                raise ValueError("row-scoped aggregation requires token states")
            if self.row_scope == "masked":
                valid = valid & (
                    (context.row_token_state == 0) | (context.row_token_state == 4)
                )
            elif self.row_scope == "nonstable":
                valid = valid & (context.row_token_state != 3)
            else:
                raise ValueError(f"unknown row scope: {self.row_scope}")
        if self.score == "masked_total":
            if context.row_token_state is None:
                raise ValueError("masked-total policy requires token states")
            masked = (context.row_token_state == 0) | (context.row_token_state == 4)
            return torch.where(valid & masked, values, 0.0).sum(-1)
        if self.aggregator == "max":
            return torch.where(valid, values, -torch.inf).amax(-1)
        if self.aggregator == "quantile":
            work = torch.where(valid, values, torch.inf)
            return torch.quantile(work, self.quantile, dim=-1)
        if context.row_token_state is None:
            raise ValueError("weighted aggregation requires token states")
        weights = torch.full_like(values, self.beta)
        protected = (context.row_token_state == 0) | (context.row_token_state == 4)
        weights = torch.where(protected, torch.ones_like(weights), weights)
        return torch.where(valid, values * weights, 0.0).sum(-1)

    def __call__(
        self, layer: int, query_tile: int, kv_tile: int, context: VetoDecisionContext
    ) -> torch.Tensor:
        del query_tile, kv_tile
        if self.score == "quantile":
            score = self._aggregate(context.row_log_score, context)
        else:
            running_lse = torch.where(
                context.row_running_sum_before > 0,
                context.row_running_max_before
                + context.row_running_sum_before.clamp_min(1e-30).log(),
                -torch.inf,
            )
            denominator = torch.logaddexp(running_lse, context.row_local_logsumexp)
            values = torch.exp(context.row_local_logsumexp - denominator)
            if self.score in ("online_mass_v_max", "online_mass_v_mean", "online_variance_bound"):
                v = context.v_tile.float()
                if self.score == "online_mass_v_max":
                    magnitude = torch.linalg.vector_norm(v, dim=-1).amax(-1)
                elif self.score == "online_mass_v_mean":
                    magnitude = torch.linalg.vector_norm(v, dim=-1).mean(-1)
                else:
                    centroid = v.mean(-2)
                    magnitude = torch.linalg.vector_norm(
                        v - centroid[..., None, :], dim=-1
                    ).amax(-1)
                values = values * magnitude[..., None]
            score = self._aggregate(values, context)
        candidate = (
            (score < self._threshold(layer))
            & ~context.baseline_physical_skip
            & ~self._protected_new_max(context, self.new_max_protection)
        )
        return self._record(layer, context, candidate)


class CumulativeBudgetPolicy(VetoPolicy):
    """Per-layer online mass × V-residual error budget."""

    def __init__(
        self,
        base_budget: float,
        *,
        rule: Literal["sum", "rss", "max"] = "sum",
        state_multipliers: tuple[float, float, float, float, float] = (
            1.0,
            1.5,
            4.0,
            10.0,
            1.0,
        ),
        new_max_protection: NewMaxProtection = "all",
    ) -> None:
        super().__init__(new_max_protection=new_max_protection)
        self.base_budget = base_budget
        self.rule = rule
        self.state_multipliers = state_multipliers
        self._accumulated: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, layer: int, query_tile: int, kv_tile: int, context: VetoDecisionContext
    ) -> torch.Tensor:
        if context.row_token_state is None:
            raise ValueError("cumulative budget requires token states")
        key = (layer, query_tile)
        accumulated = self._accumulated.setdefault(
            key, torch.zeros_like(context.row_local_logsumexp)
        )
        running_lse = torch.where(
            context.row_running_sum_before > 0,
            context.row_running_max_before
            + context.row_running_sum_before.clamp_min(1e-30).log(),
            -torch.inf,
        )
        mass = torch.exp(
            context.row_local_logsumexp
            - torch.logaddexp(running_lse, context.row_local_logsumexp)
        )
        v = context.v_tile.float()
        centroid = v.mean(-2)
        residual = torch.linalg.vector_norm(v - centroid[..., None, :], dim=-1).amax(-1)
        error = mass * residual[..., None]
        if self.rule == "sum":
            proposed = accumulated + error
        elif self.rule == "rss":
            proposed = torch.sqrt(accumulated.square() + error.square())
        else:
            proposed = torch.maximum(accumulated, error)
        multipliers = torch.tensor(
            self.state_multipliers, dtype=error.dtype, device=error.device
        )
        budgets = self.base_budget * multipliers[context.row_token_state.long()]
        candidate = (
            ((proposed <= budgets) | ~context.valid_rows).all(-1)
            & ~context.baseline_physical_skip
            & ~self._protected_new_max(context, self.new_max_protection)
        )
        accumulated.copy_(torch.where(candidate[..., None], proposed, accumulated))
        extra = self._record(layer, context, candidate)
        if kv_tile == 0:
            del self._accumulated[key]
        return extra
