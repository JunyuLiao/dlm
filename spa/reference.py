"""Correctness-first Shared-Page Attention references.

The code in this module deliberately materializes exact scores.  It is an
oracle and quality-evaluation path, not a performance kernel.  In particular,
the sparse reference renormalizes softmax over the retained keys and never
uses the dense denominator after omitting values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

SupportFormat = Literal["columns", "pages", "hybrid"]
Selector = Literal["oracle-mean", "oracle-tail", "oracle-coverage"]
CoverageObjective = Literal["minimum", "p1", "p5", "mean-min"]


@dataclass(frozen=True)
class SPAConfig:
    group_size: int = 64
    page_size: int = 64
    support: SupportFormat = "pages"
    density: float = 0.5
    selector: Selector = "oracle-mean"
    tail_beta: float = 1.0
    tail_quantile: float = 0.95
    coverage_objective: CoverageObjective = "p1"
    sink_pages: int = 0
    mandatory_local: bool = False
    mandatory_first: bool = False
    mandatory_last: bool = False
    share_heads_experimental: bool = False

    def __post_init__(self) -> None:
        if self.group_size <= 0 or self.page_size <= 0:
            raise ValueError("group_size and page_size must be positive")
        if not 0 < self.density <= 1:
            raise ValueError("density must be in (0, 1]")
        if self.support not in ("columns", "pages", "hybrid"):
            raise ValueError(f"unsupported SPA format: {self.support}")
        if self.selector not in ("oracle-mean", "oracle-tail", "oracle-coverage"):
            raise ValueError(f"unsupported SPA selector: {self.selector}")
        if self.coverage_objective not in ("minimum", "p1", "p5", "mean-min"):
            raise ValueError(f"unsupported coverage objective: {self.coverage_objective}")
        if self.tail_beta < 0 or not 0 <= self.tail_quantile <= 1:
            raise ValueError("tail beta and quantile are invalid")
        if self.sink_pages < 0:
            raise ValueError("sink_pages must be nonnegative")
        if self.support != "hybrid" and self.sink_pages:
            raise ValueError("sink pages are only valid for hybrid support")


@dataclass(frozen=True)
class SPASupport:
    """Group-shared support, isolated by request and attention head.

    ``key_mask`` is indexed ``[batch, head, query_group, key]``.  Ordinary
    physical pages and packed sink columns are retained separately so logical
    and hardware-oriented budgets cannot be conflated.
    """

    key_mask: torch.Tensor
    ordinary_page_mask: torch.Tensor | None
    sink_column_mask: torch.Tensor | None
    group_size: int
    page_size: int
    format: SupportFormat

    def __post_init__(self) -> None:
        if self.key_mask.ndim != 4 or self.key_mask.dtype != torch.bool:
            raise ValueError("key_mask must be boolean [batch, head, group, key]")


@dataclass(frozen=True)
class SPAMetrics:
    mean_retained_mass: float
    p1_retained_mass: float
    p5_retained_mass: float
    p50_retained_mass: float
    minimum_retained_mass: float
    output_relative_l2: float
    output_cosine_similarity: float
    selected_key_density: float
    ordinary_page_density: float
    selected_pages_per_group: float
    sink_columns_per_head: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor | None = None) -> None:
    if q.ndim != 4 or k.ndim != 4 or (v is not None and v.ndim != 4):
        raise ValueError("q, k, and v must be [batch, sequence, heads, dimension]")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError("SPA currently requires request/head-isolated MHA Q and K")
    if v is not None and (v.shape[:3] != k.shape[:3]):
        raise ValueError("K and V batch, sequence, and head dimensions must match")


def _valid_mask(mask: torch.Tensor | None, batch: int, length: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, length), dtype=torch.bool, device=device)
    if mask.shape != (batch, length):
        raise ValueError(f"validity mask must have shape {(batch, length)}")
    return mask.to(device=device, dtype=torch.bool)


def _attention_probabilities(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    key_valid: torch.Tensor,
    query_valid: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    scores.masked_fill_(~key_valid[:, None, None, :], -torch.inf)
    if causal:
        q_positions = torch.arange(q.shape[1], device=q.device)[:, None]
        k_positions = torch.arange(k.shape[1], device=q.device)[None, :]
        scores.masked_fill_(k_positions > q_positions, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(torch.isfinite(probabilities), probabilities, 0.0)
    return probabilities * query_valid[:, None, :, None]


def dense_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_valid_mask: torch.Tensor | None = None,
    query_valid_mask: torch.Tensor | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    _validate_qkv(q, k, v)
    key_valid = _valid_mask(key_valid_mask, q.shape[0], k.shape[1], q.device)
    query_valid = _valid_mask(query_valid_mask, q.shape[0], q.shape[1], q.device)
    probabilities = _attention_probabilities(
        q, k, key_valid=key_valid, query_valid=query_valid,
        causal=causal, softmax_scale=softmax_scale,
    )
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float()).to(q.dtype)


def _page_mass(probabilities: torch.Tensor, page_size: int) -> torch.Tensor:
    padding = (-probabilities.shape[-1]) % page_size
    if padding:
        probabilities = F.pad(probabilities, (0, padding))
    return probabilities.reshape(*probabilities.shape[:-1], -1, page_size).sum(-1)


def _mandatory_pages(
    group: int,
    num_pages: int,
    config: SPAConfig,
) -> list[int]:
    result: set[int] = set()
    if config.mandatory_first:
        result.add(0)
    if config.mandatory_last:
        result.add(num_pages - 1)
    if config.mandatory_local:
        first_query = group * config.group_size
        last_query = first_query + config.group_size - 1
        first_page = min(first_query // config.page_size, num_pages - 1)
        last_page = min(last_query // config.page_size, num_pages - 1)
        result.update(range(first_page, last_page + 1))
    return sorted(result)


def _coverage_value(coverage: torch.Tensor, objective: CoverageObjective) -> tuple[float, float]:
    if objective == "minimum":
        return float(coverage.min()), float(coverage.mean())
    if objective == "p1":
        return float(torch.quantile(coverage, 0.01)), float(coverage.mean())
    if objective == "p5":
        return float(torch.quantile(coverage, 0.05)), float(coverage.mean())
    return float(coverage.min()), float(coverage.mean())


def _select_one(
    values: torch.Tensor,
    count: int,
    config: SPAConfig,
    mandatory: list[int],
) -> torch.Tensor:
    """Select columns/pages from per-row normalized mass ``[rows, items]``."""
    items = values.shape[-1]
    count = min(max(count, len(mandatory)), items)
    selected = torch.zeros(items, dtype=torch.bool, device=values.device)
    if mandatory:
        selected[torch.tensor(mandatory, device=values.device)] = True
    remaining = count - int(selected.sum())
    if remaining <= 0:
        return selected
    if config.selector == "oracle-mean":
        score = values.mean(0)
        score = score.masked_fill(selected, -torch.inf)
        selected[score.topk(remaining).indices] = True
        return selected
    if config.selector == "oracle-tail":
        score = values.mean(0) + config.tail_beta * torch.quantile(
            values, config.tail_quantile, dim=0
        )
        score = score.masked_fill(selected, -torch.inf)
        selected[score.topk(remaining).indices] = True
        return selected

    coverage = values[:, selected].sum(-1) if bool(selected.any()) else torch.zeros(
        values.shape[0], device=values.device
    )
    for _ in range(remaining):
        best_item = -1
        best_value = (-math.inf, -math.inf)
        for item in range(items):
            if bool(selected[item]):
                continue
            candidate = coverage + values[:, item]
            objective = _coverage_value(candidate, config.coverage_objective)
            if objective > best_value:
                best_item, best_value = item, objective
        if best_item < 0:
            break
        selected[best_item] = True
        coverage = coverage + values[:, best_item]
    return selected


def _pages_to_keys(page_mask: torch.Tensor, key_length: int, page_size: int) -> torch.Tensor:
    return page_mask.repeat_interleave(page_size, dim=-1)[..., :key_length]


def build_oracle_support(
    q: torch.Tensor,
    k: torch.Tensor,
    config: SPAConfig,
    *,
    key_valid_mask: torch.Tensor | None = None,
    query_valid_mask: torch.Tensor | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> tuple[SPASupport, torch.Tensor]:
    """Build current-step oracle support from exact normalized attention mass."""
    _validate_qkv(q, k)
    batch, query_length, heads, _ = q.shape
    key_length = k.shape[1]
    key_valid = _valid_mask(key_valid_mask, batch, key_length, q.device)
    query_valid = _valid_mask(query_valid_mask, batch, query_length, q.device)
    probabilities = _attention_probabilities(
        q, k, key_valid=key_valid, query_valid=query_valid,
        causal=causal, softmax_scale=softmax_scale,
    )
    support = build_oracle_support_from_probabilities(
        probabilities,
        config,
        key_valid_mask=key_valid,
        query_valid_mask=query_valid,
    )
    return support, probabilities


def build_oracle_support_from_probabilities(
    probabilities: torch.Tensor,
    config: SPAConfig,
    *,
    key_valid_mask: torch.Tensor | None = None,
    query_valid_mask: torch.Tensor | None = None,
) -> SPASupport:
    """Select support from exact probabilities already computed for analysis.

    This helper avoids repeating dense QK during multi-policy oracle sweeps.
    The returned support is still applied by
    :func:`exact_shared_support_attention` for correctness evaluation.
    """
    if probabilities.ndim != 4:
        raise ValueError("probabilities must be [batch, head, query, key]")
    batch, heads, query_length, key_length = probabilities.shape
    key_valid = _valid_mask(key_valid_mask, batch, key_length, probabilities.device)
    query_valid = _valid_mask(query_valid_mask, batch, query_length, probabilities.device)
    groups = math.ceil(query_length / config.group_size)
    num_pages = math.ceil(key_length / config.page_size)
    # Density is a hard retained-key budget.  Whole pages/columns are rounded
    # down so a nominal 70% policy cannot silently exceed the 70% gate.
    page_count = min(num_pages, max(1, math.floor(config.density * num_pages)))
    column_count = min(key_length, max(1, math.floor(config.density * key_length)))
    ordinary_pages = torch.zeros(
        (batch, heads, groups, num_pages), dtype=torch.bool, device=probabilities.device
    )
    key_mask = torch.zeros(
        (batch, heads, groups, key_length), dtype=torch.bool, device=probabilities.device
    )

    masses = _page_mass(probabilities, config.page_size)
    # The production-scale oracle shape has thousands of independent
    # (request, head, group) selections. Keep the general path below for
    # partial/invalid rows and mandatory support, but vectorize the common
    # all-valid ablation path to avoid Python/GPU synchronization per group.
    fast_path = (
        bool(query_valid.all())
        and query_length % config.group_size == 0
        and not config.share_heads_experimental
        and not config.mandatory_local
        and not config.mandatory_first
        and not config.mandatory_last
    )
    if fast_path:
        if config.support == "columns":
            values = probabilities.reshape(
                batch, heads, groups, config.group_size, key_length
            )
            if config.selector == "oracle-mean":
                scores = values.mean(-2)
            elif config.selector == "oracle-tail":
                scores = values.mean(-2) + config.tail_beta * torch.quantile(
                    values, config.tail_quantile, dim=-2
                )
            else:
                scores = None
            if scores is not None:
                key_mask.scatter_(-1, scores.topk(column_count, dim=-1).indices, True)
                key_mask &= key_valid[:, None, None, :]
        else:
            values = masses.reshape(
                batch, heads, groups, config.group_size, num_pages
            )
            if config.selector == "oracle-mean":
                scores = values.mean(-2)
                ordinary_pages.scatter_(-1, scores.topk(page_count, dim=-1).indices, True)
            elif config.selector == "oracle-tail":
                scores = values.mean(-2) + config.tail_beta * torch.quantile(
                    values, config.tail_quantile, dim=-2
                )
                ordinary_pages.scatter_(-1, scores.topk(page_count, dim=-1).indices, True)
            else:
                coverage = torch.zeros(
                    (batch, heads, groups, config.group_size),
                    device=probabilities.device,
                )
                for _ in range(page_count):
                    candidate = coverage[..., None] + values
                    if config.coverage_objective == "minimum":
                        score = candidate.amin(-2)
                    elif config.coverage_objective == "p1":
                        score = torch.quantile(candidate, 0.01, dim=-2)
                    elif config.coverage_objective == "p5":
                        score = torch.quantile(candidate, 0.05, dim=-2)
                    else:
                        score = candidate.amin(-2) + 1e-4 * candidate.mean(-2)
                    score.masked_fill_(ordinary_pages, -torch.inf)
                    chosen = score.argmax(-1, keepdim=True)
                    ordinary_pages.scatter_(-1, chosen, True)
                    selected_values = values.gather(
                        -1,
                        chosen[..., None, :].expand(
                            batch, heads, groups, config.group_size, 1
                        ),
                    ).squeeze(-1)
                    coverage += selected_values
            key_mask = _pages_to_keys(ordinary_pages, key_length, config.page_size)
            key_mask &= key_valid[:, None, None, :]

    if not fast_path or (
        config.support == "columns" and config.selector == "oracle-coverage"
    ):
        ordinary_pages.zero_()
        key_mask.zero_()
        for batch_index in range(batch):
            for group in range(groups):
                start = group * config.group_size
                end = min(start + config.group_size, query_length)
                rows_valid = query_valid[batch_index, start:end]
                if not bool(rows_valid.any()):
                    continue
                if config.share_heads_experimental:
                    if config.support == "columns":
                        values = probabilities[batch_index, :, start:end][:, rows_valid].reshape(
                            -1, key_length
                        )
                        mandatory_columns: list[int] = []
                        for page in _mandatory_pages(group, num_pages, config):
                            mandatory_columns.extend(
                                range(
                                    page * config.page_size,
                                    min((page + 1) * config.page_size, key_length),
                                )
                            )
                        chosen = _select_one(values, column_count, config, mandatory_columns)
                        key_mask[batch_index, :, group] = chosen & key_valid[batch_index]
                    else:
                        values = masses[batch_index, :, start:end][:, rows_valid].reshape(
                            -1, num_pages
                        )
                        chosen = _select_one(
                            values, page_count, config, _mandatory_pages(group, num_pages, config)
                        )
                        ordinary_pages[batch_index, :, group] = chosen
                        key_mask[batch_index, :, group] = (
                            _pages_to_keys(chosen, key_length, config.page_size)
                            & key_valid[batch_index]
                        )
                    continue
                for head in range(heads):
                    if config.support == "columns":
                        values = probabilities[batch_index, head, start:end][rows_valid]
                        mandatory_columns: list[int] = []
                        for page in _mandatory_pages(group, num_pages, config):
                            mandatory_columns.extend(
                                range(
                                    page * config.page_size,
                                    min((page + 1) * config.page_size, key_length),
                                )
                            )
                        chosen = _select_one(values, column_count, config, mandatory_columns)
                        key_mask[batch_index, head, group] = chosen & key_valid[batch_index]
                    else:
                        values = masses[batch_index, head, start:end][rows_valid]
                        chosen = _select_one(
                            values, page_count, config, _mandatory_pages(group, num_pages, config)
                        )
                        ordinary_pages[batch_index, head, group] = chosen
                        key_mask[batch_index, head, group] = (
                            _pages_to_keys(chosen, key_length, config.page_size)
                            & key_valid[batch_index]
                        )

    sink_columns: torch.Tensor | None = None
    if config.support == "hybrid" and config.sink_pages:
        sink_count = min(key_length, config.sink_pages * config.page_size)
        sink_columns = torch.zeros(
            (batch, heads, key_length), dtype=torch.bool, device=probabilities.device
        )
        for batch_index in range(batch):
            if config.share_heads_experimental:
                missed_score = torch.zeros(key_length, device=probabilities.device)
                for head in range(heads):
                    for group in range(groups):
                        start = group * config.group_size
                        end = min(start + config.group_size, query_length)
                        rows_valid = query_valid[batch_index, start:end]
                        if not bool(rows_valid.any()):
                            continue
                        missed = ~key_mask[batch_index, head, group]
                        missed_score += (
                            probabilities[batch_index, head, start:end][rows_valid].mean(0)
                            * missed
                        )
                missed_score.masked_fill_(~key_valid[batch_index], -torch.inf)
                chosen = missed_score.topk(sink_count).indices
                sink_columns[batch_index, :, chosen] = True
                continue
            for head in range(heads):
                missed_score = torch.zeros(key_length, device=probabilities.device)
                for group in range(groups):
                    start = group * config.group_size
                    end = min(start + config.group_size, query_length)
                    rows_valid = query_valid[batch_index, start:end]
                    if not bool(rows_valid.any()):
                        continue
                    missed = ~key_mask[batch_index, head, group]
                    missed_score += probabilities[batch_index, head, start:end][rows_valid].mean(0) * missed
                missed_score.masked_fill_(~key_valid[batch_index], -torch.inf)
                chosen = missed_score.topk(sink_count).indices
                sink_columns[batch_index, head, chosen] = True
        key_mask |= sink_columns[:, :, None, :]

    return SPASupport(
        key_mask=key_mask,
        ordinary_page_mask=None if config.support == "columns" else ordinary_pages,
        sink_column_mask=sink_columns,
        group_size=config.group_size,
        page_size=config.page_size,
        format=config.support,
    )


def exact_shared_support_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    support: SPASupport,
    *,
    key_valid_mask: torch.Tensor | None = None,
    query_valid_mask: torch.Tensor | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Apply group support and renormalize softmax over retained keys only."""
    _validate_qkv(q, k, v)
    batch, query_length, heads, _ = q.shape
    if support.key_mask.shape[:2] != (batch, heads) or support.key_mask.shape[-1] != k.shape[1]:
        raise ValueError("support batch/head/key dimensions do not match Q/K/V")
    groups = math.ceil(query_length / support.group_size)
    if support.key_mask.shape[2] != groups:
        raise ValueError("support query-group count does not match Q")
    key_valid = _valid_mask(key_valid_mask, batch, k.shape[1], q.device)
    query_valid = _valid_mask(query_valid_mask, batch, query_length, q.device)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    output = torch.zeros((batch, query_length, heads, v.shape[-1]), device=q.device, dtype=torch.float32)
    for group in range(groups):
        start = group * support.group_size
        end = min(start + support.group_size, query_length)
        for batch_index in range(batch):
            valid_rows = query_valid[batch_index, start:end]
            if not bool(valid_rows.any()):
                continue
            for head in range(heads):
                selected = support.key_mask[batch_index, head, group] & key_valid[batch_index]
                selected_positions = selected.nonzero().flatten()
                if not selected_positions.numel():
                    continue
                query = q[batch_index, start:end, head].float()
                selected_k = k[batch_index, selected_positions, head].float()
                scores = torch.matmul(query, selected_k.T) * scale
                allowed = valid_rows[:, None].expand(-1, selected_positions.numel()).clone()
                if causal:
                    query_positions = torch.arange(start, end, device=q.device)[:, None]
                    allowed &= selected_positions[None, :] <= query_positions
                scores.masked_fill_(~allowed, -torch.inf)
                probabilities = torch.softmax(scores, -1)
                probabilities = torch.where(torch.isfinite(probabilities), probabilities, 0.0)
                selected_v = v[batch_index, selected_positions, head].float()
                output[batch_index, start:end, head] = torch.matmul(probabilities, selected_v)
    return output.to(q.dtype)


def evaluate_support(
    support: SPASupport,
    dense_probabilities: torch.Tensor,
    sparse_output: torch.Tensor,
    dense_output: torch.Tensor,
    *,
    query_valid_mask: torch.Tensor | None = None,
) -> SPAMetrics:
    batch, heads, query_length, key_length = dense_probabilities.shape
    query_valid = _valid_mask(query_valid_mask, batch, query_length, dense_probabilities.device)
    group_ids = torch.arange(query_length, device=dense_probabilities.device) // support.group_size
    row_support = support.key_mask[:, :, group_ids, :]
    retained = (dense_probabilities * row_support).sum(-1)
    selected = query_valid[:, None, :].expand(-1, heads, -1)
    values = retained[selected].float()
    if not values.numel():
        raise ValueError("at least one valid query row is required")
    output_selected = query_valid[:, :, None, None]
    difference = torch.where(output_selected, sparse_output.float() - dense_output.float(), 0.0)
    reference = torch.where(output_selected, dense_output.float(), 0.0)
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    cosine = F.cosine_similarity(
        torch.where(output_selected, sparse_output.float(), 0.0).flatten(),
        reference.flatten(), dim=0,
    )
    key_density = support.key_mask.float().mean()
    if support.ordinary_page_mask is None:
        page_density = torch.tensor(0.0, device=key_density.device)
        pages_per_group = torch.tensor(0.0, device=key_density.device)
    else:
        page_density = support.ordinary_page_mask.float().mean()
        pages_per_group = support.ordinary_page_mask.sum(-1).float().mean()
    sink_columns = (
        torch.tensor(0.0, device=key_density.device)
        if support.sink_column_mask is None
        else support.sink_column_mask.sum(-1).float().mean()
    )
    return SPAMetrics(
        mean_retained_mass=float(values.mean()),
        p1_retained_mass=float(torch.quantile(values, 0.01)),
        p5_retained_mass=float(torch.quantile(values, 0.05)),
        p50_retained_mass=float(torch.quantile(values, 0.50)),
        minimum_retained_mass=float(values.min()),
        output_relative_l2=float(relative_l2),
        output_cosine_similarity=float(cosine),
        selected_key_density=float(key_density),
        ordinary_page_density=float(page_density),
        selected_pages_per_group=float(pages_per_group),
        sink_columns_per_head=float(sink_columns),
    )


def mean_support_jaccard(left: SPASupport, right: SPASupport) -> float:
    if left.key_mask.shape != right.key_mask.shape:
        raise ValueError("support shapes must match")
    intersection = (left.key_mask & right.key_mask).sum(-1).float()
    union = (left.key_mask | right.key_mask).sum(-1).float()
    return float(torch.where(union > 0, intersection / union, torch.ones_like(union)).mean())
