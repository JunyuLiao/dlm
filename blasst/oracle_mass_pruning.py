"""Exact post-pass oracle pruning for complete physical attention tiles.

This module is deliberately a correctness reference.  It first computes (or
accepts) exact per-tile online-softmax sufficient statistics, then makes one
physical decision for every ``[query tile, KV tile]`` pair.  Selected tiles are
removed simultaneously and the remaining probability distribution is exactly
renormalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import torch
import torch.nn.functional as F


Aggregation = Literal[
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
]
BudgetScope = Literal["all", "masked", "mixed"]
TileOrder = Literal["score", "traversal", "masked_mass"]
NewMaximumProtection = Literal["none", "all", "masked"]
SelectionMode = Literal["independent", "cumulative"]


@dataclass(frozen=True)
class ExactTileStatistics:
    """Per-tile exact softmax statistics for one physical query tile.

    ``m`` and ``l`` have shape ``[batch, heads, KV tiles, query rows]`` and
    ``u`` has one additional output-dimension axis.  For a tile, ``m`` is its
    row maximum, ``l = sum(exp(score - m))``, and
    ``u = sum(exp(score - m) * V)``.
    """

    m: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor

    def __post_init__(self) -> None:
        if self.m.ndim != 4 or self.m.shape != self.l.shape:
            raise ValueError("m and l must have shape [batch, heads, tiles, rows]")
        if self.u.ndim != 5 or self.u.shape[:-1] != self.m.shape:
            raise ValueError("u must have shape [batch, heads, tiles, rows, value_dim]")
        if bool((self.l < 0).any()):
            raise ValueError("tile softmax denominators must be non-negative")

    @property
    def final_mass(self) -> torch.Tensor:
        """Exact full-sequence normalized mass ``alpha`` for every tile/row."""
        valid = self.l > 0
        global_m = torch.where(valid, self.m, -torch.inf).amax(dim=2)
        scales = torch.where(valid, torch.exp(self.m - global_m[:, :, None]), 0.0)
        denominator = (scales * self.l).sum(dim=2).clamp_min(1e-30)
        return scales * self.l / denominator[:, :, None]

    def compose(self, active_tiles: torch.Tensor) -> torch.Tensor:
        """Compose and renormalize the output using complete active tiles.

        ``active_tiles`` has shape ``[batch, heads, KV tiles]``.  Subtracting
        selected *unnormalized sufficient statistics* from a common reference
        is mathematically identical to masking logits and recomputing softmax;
        the returned result always divides by the new denominator.
        """
        if active_tiles.shape != self.m.shape[:-1]:
            raise ValueError("active tile mask must have shape [batch, heads, tiles]")
        if bool((~active_tiles).all(dim=2).any()):
            raise ValueError("at least one KV tile must remain active per batch/head")
        active = active_tiles[..., None] & (self.l > 0)
        active_m = torch.where(active, self.m, -torch.inf)
        global_m = active_m.amax(dim=2)
        scale = torch.where(active, torch.exp(self.m - global_m[:, :, None]), 0.0)
        denominator = (scale * self.l).sum(dim=2).clamp_min(1e-30)
        numerator = (scale[..., None] * self.u).sum(dim=2)
        return numerator / denominator[..., None]


@dataclass(frozen=True)
class BlasstTileDecisions:
    keep: torch.Tensor
    veto_rows: torch.Tensor
    veto_count: torch.Tensor
    introduced_new_max_rows: torch.Tensor
    traversal: tuple[int, ...]


@dataclass(frozen=True)
class AttentionErrorMetrics:
    maximum_absolute_error: float
    mean_relative_error: float
    mean_row_cosine: float
    minimum_row_cosine: float
    masked_mean_relative_error: float
    visible_mean_relative_error: float
    masked_mean_row_cosine: float
    visible_mean_row_cosine: float

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class OraclePolicyConfig:
    """One oracle policy point on the accuracy/sparsity frontier."""

    k: int = 8
    aggregation: Aggregation = "all_max"
    selection: SelectionMode = "cumulative"
    threshold: float | None = None
    budget: float = 1e-3
    budget_scope: BudgetScope = "all"
    order: TileOrder = "score"
    new_maximum_protection: NewMaximumProtection = "none"
    visible_weight: float = 0.25
    visible_budget_multiplier: float = 4.0
    protected_layers: tuple[int, ...] = ()


@dataclass
class OraclePruningStats:
    total_tiles: int = 0
    baseline_skipped_tiles: int = 0
    candidate_tiles: int = 0
    additional_skipped_tiles: int = 0
    removed_new_maximum_tiles: int = 0
    calls: int = 0
    by_layer: dict[int, int] = field(default_factory=dict)
    by_layer_head: dict[str, int] = field(default_factory=dict)

    @property
    def additional_physical_sparsity(self) -> float:
        return self.additional_skipped_tiles / self.total_tiles if self.total_tiles else 0.0

    @property
    def fraction_of_baseline_kept_removed(self) -> float:
        kept = self.total_tiles - self.baseline_skipped_tiles
        return self.additional_skipped_tiles / kept if kept else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "additional_physical_sparsity": self.additional_physical_sparsity,
            "fraction_of_baseline_kept_removed": self.fraction_of_baseline_kept_removed,
        }


class OracleMassPruner:
    """Stateful selector supporting current-step and fixed-mask oracle studies."""

    def __init__(self, config: OraclePolicyConfig, *, reuse_mask: bool = False) -> None:
        self.config = config
        self.reuse_mask = reuse_mask
        self.stats = OraclePruningStats()
        self._cached_masks: dict[object, torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._cached_masks.clear()

    def select(
        self,
        statistics: ExactTileStatistics,
        decisions: BlasstTileDecisions,
        *,
        masked_rows: torch.Tensor | None,
        cache_key: object | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        if layer in self.config.protected_layers:
            result = torch.zeros_like(decisions.keep)
        elif self.reuse_mask and cache_key in self._cached_masks:
            result = self._cached_masks[cache_key].to(statistics.m.device)
        else:
            candidates = candidate_tiles(decisions, self.config.k)
            protected = new_maximum_protected_tiles(
                decisions, self.config.new_maximum_protection, masked_rows
            )
            score = aggregate_tile_mass(
                statistics.final_mass,
                decisions,
                self.config.aggregation,
                masked_rows=masked_rows,
                visible_weight=self.config.visible_weight,
            )
            if self.config.selection == "independent":
                if self.config.threshold is None:
                    raise ValueError("independent selection requires a threshold")
                result = select_independent(
                    candidates, score, self.config.threshold, protected=protected
                )
            else:
                if self.config.threshold is not None:
                    candidates = candidates & torch.isfinite(score) & (
                        score < self.config.threshold
                    )
                result, _ = select_cumulative_budget(
                    candidates,
                    statistics.final_mass,
                    budget=self.config.budget,
                    scope=self.config.budget_scope,
                    masked_rows=masked_rows,
                    visible_budget_multiplier=self.config.visible_budget_multiplier,
                    order=self.config.order,
                    score=score,
                    traversal=decisions.traversal,
                    protected=protected,
                )
            if self.reuse_mask and cache_key is not None:
                self._cached_masks[cache_key] = result.detach().cpu()
        candidates = candidate_tiles(decisions, self.config.k)
        self.stats.total_tiles += decisions.keep.numel()
        self.stats.baseline_skipped_tiles += int((~decisions.keep).sum())
        self.stats.candidate_tiles += int(candidates.sum())
        self.stats.additional_skipped_tiles += int(result.sum())
        self.stats.removed_new_maximum_tiles += int(
            (result & decisions.introduced_new_max_rows.any(-1)).sum()
        )
        self.stats.calls += 1
        self.stats.by_layer[layer] = self.stats.by_layer.get(layer, 0) + int(result.sum())
        for head, count in enumerate(result.sum(dim=(0, 2)).tolist()):
            if count:
                key = f"{layer}:{head}"
                self.stats.by_layer_head[key] = self.stats.by_layer_head.get(key, 0) + int(count)
        return result


def collect_exact_tile_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kv_block_size: int = 64,
    causal_query_offset: int | None = None,
) -> ExactTileStatistics:
    """Collect exact final-normalized mass inputs from a complete QK pass.

    Inputs use FlashAttention layout ``[batch, sequence, heads, dim]``.  ``q``
    is one physical query tile (normally 128 rows).  Partial final KV tiles are
    padded internally and are fully supported.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-four tensors")
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]:
        raise ValueError("incompatible q, k, v batch/sequence/head dimensions")
    if q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]:
        raise ValueError("incompatible Q/K dimensions or grouped-query heads")
    if kv_block_size <= 0:
        raise ValueError("kv_block_size must be positive")
    repeats = q.shape[2] // k.shape[2]
    if repeats != 1:
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    qh = q.transpose(1, 2).float()
    tile_m: list[torch.Tensor] = []
    tile_l: list[torch.Tensor] = []
    tile_u: list[torch.Tensor] = []
    for start in range(0, k.shape[1], kv_block_size):
        end = min(start + kv_block_size, k.shape[1])
        kh = k[:, start:end].transpose(1, 2).float()
        vh = v[:, start:end].transpose(1, 2).float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        if causal_query_offset is not None:
            q_pos = torch.arange(
                causal_query_offset,
                causal_query_offset + q.shape[1],
                device=q.device,
            )[:, None]
            k_pos = torch.arange(start, end, device=q.device)[None, :]
            scores = scores.masked_fill(k_pos > q_pos, -torch.inf)
        local_m = scores.amax(dim=-1)
        probabilities = torch.exp(scores - local_m[..., None])
        probabilities = torch.where(
            torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities)
        )
        tile_m.append(local_m)
        tile_l.append(probabilities.sum(dim=-1))
        tile_u.append(torch.matmul(probabilities, vh))
    return ExactTileStatistics(
        torch.stack(tile_m, dim=2),
        torch.stack(tile_l, dim=2),
        torch.stack(tile_u, dim=2),
    )


def simulate_ordinary_blasst(
    statistics: ExactTileStatistics,
    threshold: float | torch.Tensor,
    *,
    traversal: tuple[int, ...] | None = None,
) -> BlasstTileDecisions:
    """Replay ordinary BLASST's running-maximum veto decisions exactly."""
    if isinstance(threshold, torch.Tensor):
        if threshold.shape not in (statistics.m.shape[:1], statistics.m.shape[:2]):
            raise ValueError("tensor threshold must have shape [batch] or [batch, heads]")
        if bool(((threshold < 0) | (threshold > 1)).any()):
            raise ValueError("threshold must be in [0, 1]")
        log_threshold = torch.where(
            threshold > 0, threshold.log(), torch.full_like(threshold, -torch.inf)
        ).float()
        if log_threshold.ndim == 1:
            log_threshold = log_threshold[:, None]
    else:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        log_threshold = math.log(threshold) if threshold else -math.inf
    tile_count = statistics.m.shape[2]
    order = traversal or tuple(range(tile_count - 1, -1, -1))
    if tuple(sorted(order)) != tuple(range(tile_count)):
        raise ValueError("traversal must contain every KV tile exactly once")
    running_max = torch.full_like(statistics.m[:, :, 0], -torch.inf)
    keep = torch.zeros(statistics.m.shape[:-1], dtype=torch.bool, device=statistics.m.device)
    veto_rows = torch.zeros_like(statistics.m, dtype=torch.bool)
    new_max_rows = torch.zeros_like(veto_rows)
    for tile in order:
        local_m = statistics.m[:, :, tile]
        valid = (statistics.l[:, :, tile] > 0) & torch.isfinite(local_m)
        gap = local_m - running_max
        current_veto = valid & (
            gap >= (log_threshold[..., None] if isinstance(log_threshold, torch.Tensor) else log_threshold)
        )
        physical_keep = current_veto.any(dim=-1)
        veto_rows[:, :, tile] = current_veto
        keep[:, :, tile] = physical_keep
        current_new_max = valid & (local_m > running_max)
        new_max_rows[:, :, tile] = current_new_max
        running_max = torch.where(
            physical_keep[..., None] & valid,
            torch.maximum(running_max, local_m),
            running_max,
        )
    return BlasstTileDecisions(
        keep=keep,
        veto_rows=veto_rows,
        veto_count=veto_rows.sum(dim=-1),
        introduced_new_max_rows=new_max_rows,
        traversal=tuple(order),
    )


def candidate_tiles(decisions: BlasstTileDecisions, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    return decisions.keep & (decisions.veto_count >= 1) & (decisions.veto_count <= k)


def new_maximum_protected_tiles(
    decisions: BlasstTileDecisions,
    mode: NewMaximumProtection,
    masked_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "none":
        return torch.zeros_like(decisions.keep)
    new_max = decisions.introduced_new_max_rows
    if mode == "all":
        return new_max.any(dim=-1)
    if masked_rows is None:
        raise ValueError("masked new-maximum protection requires masked_rows")
    mask = _expand_rows(masked_rows, new_max)
    return (new_max & mask).any(dim=-1)


def _expand_rows(rows: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if rows.ndim == 2 and rows.shape == (reference.shape[0], reference.shape[-1]):
        return rows[:, None, None, :].expand_as(reference)
    if rows.ndim == 3 and rows.shape == (
        reference.shape[0], reference.shape[1], reference.shape[-1]
    ):
        return rows[:, :, None, :].expand_as(reference)
    if rows.shape == reference.shape:
        return rows
    raise ValueError("row mask has incompatible shape")


def aggregate_tile_mass(
    mass: torch.Tensor,
    decisions: BlasstTileDecisions,
    rule: Aggregation,
    *,
    masked_rows: torch.Tensor | None = None,
    visible_weight: float = 0.25,
) -> torch.Tensor:
    """Aggregate exact row masses into physical-tile oracle scores."""
    if mass.shape != decisions.veto_rows.shape:
        raise ValueError("mass and BLASST decisions must have identical row shapes")
    if visible_weight < 0:
        raise ValueError("visible_weight must be non-negative")
    if rule == "all_max":
        return mass.amax(dim=-1)
    if rule == "veto_max":
        return torch.where(decisions.veto_rows, mass, -torch.inf).amax(dim=-1)
    if rule in ("q90", "q95", "q99"):
        return torch.quantile(mass, int(rule[1:]) / 100.0, dim=-1)
    if rule == "masked_max":
        if masked_rows is None:
            raise ValueError("masked aggregation requires masked_rows")
        selected = _expand_rows(masked_rows, mass)
        return torch.where(selected, mass, -torch.inf).amax(dim=-1)
    if rule.endswith("_masked") or rule.endswith("_mixed"):
        if masked_rows is None:
            raise ValueError(f"{rule} requires masked_rows")
        selected = _expand_rows(masked_rows, mass)
        if rule.endswith("_masked"):
            weights = selected.to(mass.dtype)
        else:
            weights = torch.where(
                selected, torch.ones_like(mass), torch.full_like(mass, visible_weight)
            )
    else:
        weights = torch.ones_like(mass)
    weighted = (mass * weights).sum(dim=-1)
    return weighted / weights.sum(dim=-1).clamp_min(1.0) if rule.startswith("mean") else weighted


def select_independent(
    candidate: torch.Tensor,
    score: torch.Tensor,
    threshold: float,
    *,
    protected: torch.Tensor | None = None,
) -> torch.Tensor:
    if candidate.shape != score.shape:
        raise ValueError("candidate and score shapes differ")
    result = candidate & torch.isfinite(score) & (score < threshold)
    return result if protected is None else result & ~protected


def select_cumulative_budget(
    candidate: torch.Tensor,
    mass: torch.Tensor,
    *,
    budget: float,
    scope: BudgetScope = "all",
    masked_rows: torch.Tensor | None = None,
    visible_budget_multiplier: float = 4.0,
    order: TileOrder = "score",
    score: torch.Tensor | None = None,
    traversal: tuple[int, ...] | None = None,
    protected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily select physical tiles under exact cumulative row-mass limits."""
    if budget < 0 or budget >= 1:
        raise ValueError("budget must be in [0, 1)")
    if mass.shape[:-1] != candidate.shape:
        raise ValueError("mass and candidate shapes differ")
    if scope != "all" and masked_rows is None:
        raise ValueError(f"{scope} budget requires masked_rows")
    if visible_budget_multiplier < 1:
        raise ValueError("visible budget multiplier must be at least one")
    eligible = candidate if protected is None else candidate & ~protected
    selected = torch.zeros_like(candidate)
    cumulative = torch.zeros_like(mass[:, :, 0])
    batch, heads, tiles = candidate.shape
    if masked_rows is None:
        masked = torch.ones_like(cumulative, dtype=torch.bool)
    elif masked_rows.ndim == 2:
        masked = masked_rows[:, None].expand(batch, heads, mass.shape[-1])
    elif masked_rows.shape == cumulative.shape:
        masked = masked_rows
    else:
        raise ValueError("masked_rows has incompatible shape")
    if scope == "all":
        checked = torch.ones_like(masked)
        row_budget = torch.full_like(cumulative, budget)
    elif scope == "masked":
        checked = masked
        row_budget = torch.full_like(cumulative, budget)
    else:
        checked = torch.ones_like(masked)
        row_budget = torch.where(
            masked,
            torch.full_like(cumulative, budget),
            torch.full_like(cumulative, budget * visible_budget_multiplier),
        )
    if order == "traversal":
        base_order = traversal or tuple(range(tiles - 1, -1, -1))
        orders = torch.tensor(base_order, device=mass.device).expand(batch, heads, tiles)
    else:
        if order == "score":
            if score is None:
                raise ValueError("score ordering requires a tile score")
            ranking = torch.where(eligible, score, torch.inf)
        else:
            masked_mass = torch.where(masked[:, :, None], mass, 0.0).amax(dim=-1)
            ranking = torch.where(eligible, masked_mass, torch.inf)
        orders = torch.argsort(ranking, dim=-1, stable=True)
    flat_mass = mass.reshape(batch * heads, tiles, mass.shape[-1])
    flat_eligible = eligible.reshape(batch * heads, tiles)
    flat_selected = selected.reshape(batch * heads, tiles)
    flat_cumulative = cumulative.reshape(batch * heads, mass.shape[-1])
    flat_checked = checked.reshape(batch * heads, mass.shape[-1])
    flat_budget = row_budget.reshape(batch * heads, mass.shape[-1])
    flat_order = orders.reshape(batch * heads, tiles)
    groups = torch.arange(batch * heads, device=mass.device)
    for rank in range(tiles):
        tile = flat_order[:, rank]
        addition = flat_mass[groups, tile]
        allowed = flat_eligible[groups, tile] & (
            ((flat_cumulative + addition <= flat_budget) | ~flat_checked).all(dim=-1)
        )
        flat_selected[groups[allowed], tile[allowed]] = True
        flat_cumulative = torch.where(
            allowed[:, None], flat_cumulative + addition, flat_cumulative
        )
    return flat_selected.reshape_as(selected), flat_cumulative.reshape_as(cumulative)


def attention_error_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    masked_rows: torch.Tensor | None = None,
) -> AttentionErrorMetrics:
    if candidate.shape != reference.shape or candidate.ndim != 4:
        raise ValueError("outputs must share shape [batch, heads, rows, dim]")
    difference = candidate.float() - reference.float()
    row_relative = torch.linalg.vector_norm(difference, dim=-1) / (
        torch.linalg.vector_norm(reference.float(), dim=-1) + 1e-8
    )
    cosine = F.cosine_similarity(candidate.float(), reference.float(), dim=-1)
    if masked_rows is None:
        masked = torch.zeros_like(row_relative, dtype=torch.bool)
    elif masked_rows.ndim == 2:
        masked = masked_rows[:, None].expand_as(row_relative)
    elif masked_rows.shape == row_relative.shape:
        masked = masked_rows
    else:
        raise ValueError("masked_rows has incompatible shape")

    def mean_or_one(values: torch.Tensor, selection: torch.Tensor, default: float) -> float:
        return float(values[selection].mean()) if bool(selection.any()) else default

    return AttentionErrorMetrics(
        maximum_absolute_error=float(difference.abs().amax()),
        mean_relative_error=float(row_relative.mean()),
        mean_row_cosine=float(cosine.mean()),
        minimum_row_cosine=float(cosine.amin()),
        masked_mean_relative_error=mean_or_one(row_relative, masked, 0.0),
        visible_mean_relative_error=mean_or_one(row_relative, ~masked, 0.0),
        masked_mean_row_cosine=mean_or_one(cosine, masked, 1.0),
        visible_mean_row_cosine=mean_or_one(cosine, ~masked, 1.0),
    )


def oracle_mass_pruned_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    blasst_threshold: float | torch.Tensor,
    pruner: OracleMassPruner,
    layer: int = 0,
    row_masked: torch.Tensor | None = None,
    q_block_size: int = 128,
    kv_block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = False,
    cache_namespace: object | None = None,
) -> torch.Tensor:
    """Dense/reference oracle attention for one complete attention invocation.

    Exact final mass is recomputed from every KV tile before selection.  The
    result is ordinary sparse BLASST plus additional whole-tile removals.
    """
    if q_block_size <= 0:
        raise ValueError("q_block_size must be positive")
    if row_masked is not None and row_masked.shape != q.shape[:2]:
        raise ValueError("row_masked must have shape [batch, query sequence]")
    chunks: list[torch.Tensor] = []
    for q_start in range(0, q.shape[1], q_block_size):
        q_end = min(q_start + q_block_size, q.shape[1])
        statistics = collect_exact_tile_statistics(
            q[:, q_start:q_end],
            k,
            v,
            softmax_scale=softmax_scale,
            kv_block_size=kv_block_size,
            causal_query_offset=q_start if causal else None,
        )
        decisions = simulate_ordinary_blasst(statistics, blasst_threshold)
        masked_tile = None if row_masked is None else row_masked[:, q_start:q_end]
        removed = pruner.select(
            statistics,
            decisions,
            masked_rows=masked_tile,
            cache_key=(cache_namespace, layer, q_start // q_block_size),
            layer=layer,
        )
        chunks.append(statistics.compose(decisions.keep & ~removed).transpose(1, 2))
    return torch.cat(chunks, dim=1).to(q.dtype)
