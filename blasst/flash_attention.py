"""Minimal BLASST modification of FlashAttention's online-softmax loop.

Only Algorithm 1 is implemented: no pipeline, warp, batched-load, or
architecture-specific optimizations. Shapes match ``flash_attn_func``:
``[batch, sequence, heads, head_dim]``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

import torch


@dataclass
class BlasstStats:
    skipped_blocks: int = 0  # Physically skippable (batch, head, Q tile, KV tile).
    total_blocks: int = 0
    skipped_row_blocks: int = 0
    total_row_blocks: int = 0
    skipped_blocks_per_sequence: tuple[int, ...] = ()
    total_blocks_per_sequence: tuple[int, ...] = ()

    @property
    def sparsity_ratio(self) -> float:
        return self.skipped_blocks / self.total_blocks if self.total_blocks else 0.0

    def reset(self) -> None:
        self.skipped_blocks = self.total_blocks = 0
        self.skipped_row_blocks = self.total_row_blocks = 0
        self.skipped_blocks_per_sequence = self.total_blocks_per_sequence = ()


_STATS = BlasstStats()


RowMaskVariant = Literal["none", "full", "output"]


@dataclass
class ActiveVoterStats:
    """Work counters for the opt-in row-masked reference path."""

    retained_tiles: int = 0
    candidate_tiles: int = 0
    candidate_row_work: int = 0
    selected_rows: int = 0
    removed_rows: int = 0
    non_candidate_row_work: int = 0

    @property
    def candidate_work_fraction(self) -> float:
        return self.candidate_row_work / self.retained_row_work if self.retained_row_work else 0.0

    @property
    def retained_row_work(self) -> int:
        return self.candidate_row_work + self.non_candidate_row_work

    @property
    def selected_fraction(self) -> float:
        return self.selected_rows / self.candidate_row_work if self.candidate_row_work else 0.0

    def reset(self) -> None:
        self.retained_tiles = 0
        self.candidate_tiles = 0
        self.candidate_row_work = 0
        self.selected_rows = 0
        self.removed_rows = 0
        self.non_candidate_row_work = 0


_ACTIVE_VOTER_STATS = ActiveVoterStats()


def collect_active_voter_stats(*, reset: bool = False) -> ActiveVoterStats:
    result = ActiveVoterStats(**_ACTIVE_VOTER_STATS.__dict__)
    if reset:
        _ACTIVE_VOTER_STATS.reset()
    return result


@dataclass(frozen=True)
class PhysicalTileTrace:
    """Vectorized exact statistics for one (Q tile, KV tile).

    Every tensor is indexed ``[batch, query_head]``.  This object is emitted
    only by the exact/reference path and is deliberately not accepted by the
    proxy policy API.
    """

    score: torch.Tensor
    log_score: torch.Tensor
    exact_skip: torch.Tensor
    introduced_new_max: torch.Tensor
    row_max: torch.Tensor
    row_q50: torch.Tensor
    row_q90: torch.Tensor
    row_q99: torch.Tensor
    valid_query_rows: int


@dataclass(frozen=True)
class RichPhysicalTileTrace:
    """Per-row statistics for offline veto-row policy evaluation.

    Tensors whose name starts with ``row_`` are indexed
    ``[batch, query_head, valid_query_rows]``.  V-tile summaries and physical
    decisions are indexed ``[batch, query_head]``.  The callback is deliberately
    opt-in: constructing these values adds sampled diagnostic work and is not
    part of either production BLASST kernel.
    """

    row_score: torch.Tensor
    row_log_score: torch.Tensor
    row_local_max: torch.Tensor
    row_running_max_before: torch.Tensor
    row_running_sum_before: torch.Tensor
    row_local_logsumexp: torch.Tensor
    row_final_logsumexp: torch.Tensor
    row_final_output_norm: torch.Tensor
    row_local_value_norm: torch.Tensor
    row_centroid_error_norm: torch.Tensor
    row_query_delta: torch.Tensor
    row_keep_vote: torch.Tensor
    exact_skip: torch.Tensor
    introduced_new_max: torch.Tensor
    v_mean_row_norm: torch.Tensor
    v_max_row_norm: torch.Tensor
    v_frobenius_per_sqrt_rows: torch.Tensor
    v_centroid_norm: torch.Tensor
    v_residual_max_norm: torch.Tensor
    v_per_dimension_variance: torch.Tensor
    valid_query_rows: int


@dataclass(frozen=True)
class VetoDecisionContext:
    """Current-tile quantities legally available after QK and before P@V."""

    row_log_score: torch.Tensor
    row_local_logsumexp: torch.Tensor
    row_running_max_before: torch.Tensor
    row_running_sum_before: torch.Tensor
    row_introduced_new_max: torch.Tensor
    row_token_state: torch.Tensor | None
    valid_rows: torch.Tensor
    v_tile: torch.Tensor
    baseline_physical_skip: torch.Tensor


def collect_blasst_stats(*, reset: bool = False) -> BlasstStats:
    result = BlasstStats(
        _STATS.skipped_blocks,
        _STATS.total_blocks,
        _STATS.skipped_row_blocks,
        _STATS.total_row_blocks,
        _STATS.skipped_blocks_per_sequence,
        _STATS.total_blocks_per_sequence,
    )
    if reset:
        _STATS.reset()
    return result


def blasst_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    blasst_lambda: float | torch.Tensor = 0.001,
    q_block_size: int = 128,
    kv_block_size: int = 64,
    physical_score_callback: Callable[[int, int, torch.Tensor], None] | None = None,
    tile_trace_callback: Callable[[int, int, PhysicalTileTrace], None] | None = None,
    rich_tile_trace_callback: Callable[[int, int, RichPhysicalTileTrace], None] | None = None,
    veto_decision_callback: Callable[[int, int, VetoDecisionContext], torch.Tensor] | None = None,
    tile_trace_filter: Callable[[int, int], bool] | None = None,
    query_delta: torch.Tensor | None = None,
    row_token_state: torch.Tensor | None = None,
    row_mask_variant: RowMaskVariant = "none",
    row_lambda: float | torch.Tensor | None = None,
    max_active_rows: int = 32,
) -> torch.Tensor:
    """Online-softmax attention with BLASST block thresholding.

    Lambda 0 is the exact dense baseline. The 0.001 default is calibrated on
    native 4096-token masked LLaDA inputs, requiring at least 95% top-1
    agreement with the dense path at every tested diffusion noise level.
    """
    del deterministic
    if dropout_p != 0.0:
        raise NotImplementedError("BLASST inference does not support dropout")
    if window_size != (-1, -1) or softcap != 0.0 or alibi_slopes is not None:
        raise NotImplementedError("windowing, softcap, and ALiBi are outside this experiment")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported")
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be [batch, sequence, heads, dim]")
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible q, k, v shapes")
    if query_delta is not None and query_delta.shape != q.shape[:-1]:
        raise ValueError("query_delta must have shape [batch, sequence, query_heads]")
    if row_token_state is not None and row_token_state.shape != q.shape[:2]:
        raise ValueError("row_token_state must have shape [batch, sequence]")
    if row_mask_variant not in ("none", "full", "output"):
        raise ValueError("row_mask_variant must be none, full, or output")
    if max_active_rows <= 0:
        raise ValueError("max_active_rows must be positive")
    if row_mask_variant != "none" and veto_decision_callback is not None:
        raise ValueError("row masking and physical-tile veto cannot be combined")

    batch, query_len, query_heads, head_dim = q.shape
    key_len, kv_heads = k.shape[1], k.shape[2]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query_heads != kv_heads:
        repeats = query_heads // kv_heads
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)

    if isinstance(blasst_lambda, torch.Tensor):
        if blasst_lambda.device != q.device or blasst_lambda.shape not in ((batch,), (batch, query_heads)):
            raise ValueError("tensor lambda must be on q.device with shape [batch] or [batch, heads]")
        if bool(((blasst_lambda < 0) | (blasst_lambda > 1)).any()):
            raise ValueError("lambda values must be in [0, 1]")
        thresholds = blasst_lambda[:, None] if blasst_lambda.ndim == 1 else blasst_lambda
        log_threshold = torch.where(
            thresholds > 0, thresholds.log(), torch.full_like(thresholds, -torch.inf)
        ).float()
    else:
        if not 0.0 <= blasst_lambda <= 1.0:
            raise ValueError("lambda must be in [0, 1]")
        log_threshold = math.log(blasst_lambda) if blasst_lambda else -math.inf
    if row_lambda is None:
        row_log_threshold = log_threshold
    elif isinstance(row_lambda, torch.Tensor):
        if row_lambda.device != q.device or row_lambda.shape not in ((batch,), (batch, query_heads)):
            raise ValueError("tensor row_lambda must be on q.device with shape [batch] or [batch, heads]")
        if bool(((row_lambda < 0) | (row_lambda > 1)).any()):
            raise ValueError("row_lambda values must be in [0, 1]")
        row_thresholds = row_lambda[:, None] if row_lambda.ndim == 1 else row_lambda
        row_log_threshold = torch.where(
            row_thresholds > 0,
            row_thresholds.log(),
            torch.full_like(row_thresholds, -torch.inf),
        ).float()
    else:
        if not 0.0 <= row_lambda <= 1.0:
            raise ValueError("row_lambda must be in [0, 1]")
        row_log_threshold = math.log(row_lambda) if row_lambda else -math.inf
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    output = torch.empty((batch, query_len, query_heads, v.shape[-1]), dtype=q.dtype, device=q.device)

    skipped_per_sequence = torch.zeros(batch, dtype=torch.int64, device=q.device)
    total_per_sequence = torch.zeros_like(skipped_per_sequence)

    for q_start in range(0, query_len, q_block_size):
        q_end = min(q_start + q_block_size, query_len)
        q_tile = q[:, q_start:q_end].transpose(1, 2).float()
        rows = q_end - q_start
        running_max = torch.full((batch, query_heads, rows), -torch.inf, device=q.device)
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros((batch, query_heads, rows, v.shape[-1]), device=q.device)
        rich_pending: list[tuple[int, RichPhysicalTileTrace]] = []

        # Match FlashAttention 2.8's forward traversal: highest KV tile first.
        last_k_start = ((key_len - 1) // kv_block_size) * kv_block_size
        for k_start in range(last_k_start, -1, -kv_block_size):
            k_end = min(k_start + kv_block_size, key_len)
            k_tile = k[:, k_start:k_end].transpose(1, 2).float()
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)).mul_(scale)
            if causal:
                q_pos = torch.arange(q_start, q_end, device=q.device)[:, None]
                k_pos = torch.arange(k_start, k_end, device=q.device)[None, :]
                scores.masked_fill_(k_pos > q_pos, -torch.inf)

            block_max = scores.amax(dim=-1)
            valid_rows = torch.isfinite(block_max)
            # R_j = max_i exp(local_max_ij - running_max_i).  Keep this in log
            # space for the decision and only exponentiate when calibrating.
            row_gap = block_max - running_max
            tile_log_score = torch.where(valid_rows, row_gap, -torch.inf).amax(dim=-1)
            if physical_score_callback is not None:
                physical_score_callback(q_start // q_block_size, k_start // kv_block_size, tile_log_score.exp())
            threshold = log_threshold[..., None] if isinstance(log_threshold, torch.Tensor) else log_threshold
            skip = valid_rows & (row_gap < threshold)
            _STATS.skipped_row_blocks += int(skip.sum().item())
            _STATS.total_row_blocks += int(valid_rows.sum().item())
            physical_skip = (skip | ~valid_rows).all(dim=-1) & valid_rows.any(dim=-1)
            q_tile_index = q_start // q_block_size
            kv_tile_index = k_start // kv_block_size
            baseline_physical_skip = physical_skip
            if veto_decision_callback is not None:
                token_state_tile = (
                    None
                    if row_token_state is None
                    else row_token_state[:, q_start:q_end, None]
                    .transpose(1, 2)
                    .expand(batch, query_heads, rows)
                )
                extra_skip = veto_decision_callback(
                    q_tile_index,
                    kv_tile_index,
                    VetoDecisionContext(
                        row_log_score=row_gap,
                        row_local_logsumexp=torch.logsumexp(scores, dim=-1),
                        row_running_max_before=running_max,
                        row_running_sum_before=running_sum,
                        row_introduced_new_max=valid_rows & (block_max > running_max),
                        row_token_state=token_state_tile,
                        valid_rows=valid_rows,
                        v_tile=v[:, k_start:k_end].transpose(1, 2),
                        baseline_physical_skip=baseline_physical_skip,
                    ),
                )
                if extra_skip.shape != physical_skip.shape or extra_skip.device != q.device:
                    raise ValueError("veto decision must return a device-local [batch, heads] tensor")
                physical_skip = physical_skip | (extra_skip.bool() & ~physical_skip)
            active_rows = valid_rows
            row_mask_candidate = torch.zeros_like(physical_skip)
            if row_mask_variant != "none":
                active_threshold = (
                    row_log_threshold[..., None]
                    if isinstance(row_log_threshold, torch.Tensor)
                    else row_log_threshold
                )
                row_votes = valid_rows & (row_gap >= active_threshold)
                active_count = row_votes.sum(dim=-1)
                row_mask_candidate = (
                    ~physical_skip & (active_count > 0) & (active_count <= max_active_rows)
                )
                active_rows = torch.where(row_mask_candidate[..., None], row_votes, valid_rows)
                retained = ~physical_skip
                valid_count = valid_rows.sum(dim=-1)
                _ACTIVE_VOTER_STATS.retained_tiles += int(retained.sum().item())
                _ACTIVE_VOTER_STATS.candidate_tiles += int(row_mask_candidate.sum().item())
                candidate_work = valid_count[row_mask_candidate]
                selected = active_count[row_mask_candidate]
                _ACTIVE_VOTER_STATS.candidate_row_work += int(candidate_work.sum().item())
                _ACTIVE_VOTER_STATS.selected_rows += int(selected.sum().item())
                _ACTIVE_VOTER_STATS.removed_rows += int((candidate_work - selected).sum().item())
                _ACTIVE_VOTER_STATS.non_candidate_row_work += int(
                    valid_count[retained & ~row_mask_candidate].sum().item()
                )
            should_trace = tile_trace_filter is None or tile_trace_filter(q_tile_index, kv_tile_index)
            if tile_trace_callback is not None and should_trace:
                row_values = torch.where(valid_rows, row_gap.exp(), torch.nan)
                # Noncausal LLaDA tiles have the same valid-row count for all
                # heads. nanquantile also keeps the hook correct for causal
                # smoke tests without leaking masked rows into summaries.
                quantile_values = torch.where(
                    torch.isposinf(row_values),
                    torch.finfo(row_values.dtype).max,
                    row_values,
                )
                quantiles = torch.nanquantile(
                    quantile_values,
                    torch.tensor((0.5, 0.9, 0.99), device=q.device),
                    dim=-1,
                )
                tile_trace_callback(
                    q_tile_index,
                    kv_tile_index,
                    PhysicalTileTrace(
                        score=tile_log_score.exp().detach(),
                        log_score=tile_log_score.detach(),
                        exact_skip=physical_skip.detach(),
                        introduced_new_max=(valid_rows & (block_max > running_max)).any(-1).detach(),
                        row_max=torch.where(torch.isnan(row_values), -torch.inf, row_values).amax(-1).detach(),
                        row_q50=quantiles[0].detach(),
                        row_q90=quantiles[1].detach(),
                        row_q99=quantiles[2].detach(),
                        valid_query_rows=rows,
                    ),
                )
            if rich_tile_trace_callback is not None and should_trace:
                # These diagnostics are intentionally evaluated even for a
                # tile ordinary BLASST skips.  They establish offline oracle
                # headroom without changing the online-softmax state below.
                v_tile_for_trace = v[:, k_start:k_end].transpose(1, 2).float()
                local_probabilities = torch.softmax(scores, dim=-1)
                local_probabilities = torch.where(
                    torch.isfinite(local_probabilities),
                    local_probabilities,
                    torch.zeros_like(local_probabilities),
                )
                local_value = torch.matmul(local_probabilities, v_tile_for_trace)
                centroid = v_tile_for_trace.mean(dim=-2)
                residual = v_tile_for_trace - centroid[..., None, :]
                v_row_norms = torch.linalg.vector_norm(v_tile_for_trace, dim=-1)
                q_delta_tile = (
                    torch.full_like(row_gap, torch.nan)
                    if query_delta is None
                    else query_delta[:, q_start:q_end].transpose(1, 2).float()
                )
                rich_pending.append(
                    (
                        kv_tile_index,
                        RichPhysicalTileTrace(
                            row_score=row_gap.exp().detach(),
                            row_log_score=row_gap.detach(),
                            row_local_max=block_max.detach(),
                            row_running_max_before=running_max.detach(),
                            row_running_sum_before=running_sum.detach(),
                            row_local_logsumexp=torch.logsumexp(scores, dim=-1).detach(),
                            # Filled after the complete Q-tile traversal.
                            row_final_logsumexp=torch.empty(0, device=q.device),
                            row_final_output_norm=torch.empty(0, device=q.device),
                            row_local_value_norm=torch.linalg.vector_norm(local_value, dim=-1).detach(),
                            row_centroid_error_norm=torch.linalg.vector_norm(
                                local_value - centroid[..., None, :], dim=-1
                            ).detach(),
                            row_query_delta=q_delta_tile.detach(),
                            row_keep_vote=(valid_rows & ~skip).detach(),
                            exact_skip=physical_skip.detach(),
                            introduced_new_max=(valid_rows & (block_max > running_max)).any(-1).detach(),
                            v_mean_row_norm=v_row_norms.mean(-1).detach(),
                            v_max_row_norm=v_row_norms.amax(-1).detach(),
                            v_frobenius_per_sqrt_rows=torch.sqrt(
                                v_tile_for_trace.square().sum((-2, -1)) / max(k_end - k_start, 1)
                            ).detach(),
                            v_centroid_norm=torch.linalg.vector_norm(centroid, dim=-1).detach(),
                            v_residual_max_norm=torch.linalg.vector_norm(residual, dim=-1).amax(-1).detach(),
                            v_per_dimension_variance=residual.square().mean((-2, -1)).detach(),
                            valid_query_rows=rows,
                        ),
                    )
                )
            _STATS.skipped_blocks += int(physical_skip.sum().item())
            _STATS.total_blocks += int(valid_rows.any(dim=-1).sum().item())
            skipped_per_sequence += physical_skip.sum(dim=-1)
            total_per_sequence += valid_rows.any(dim=-1).sum(dim=-1)

            # The realizable unit is the complete [Q tile, KV tile] for one
            # sequence and head. A non-unanimous tile is evaluated for every
            # row; a unanimous skip leaves the online-softmax state unchanged.
            proposed_max = torch.maximum(running_max, block_max)
            softmax_rows = active_rows if row_mask_variant == "full" else valid_rows
            update_rows = ~physical_skip[..., None] & softmax_rows
            new_max = torch.where(update_rows, proposed_max, running_max)
            old_scale = torch.exp(running_max - new_max)
            old_scale = torch.where(torch.isfinite(old_scale), old_scale, torch.zeros_like(old_scale))
            probabilities = torch.exp(scores - new_max[..., None])
            probabilities = torch.where(torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities))
            probabilities.masked_fill_(~update_rows[..., None], 0.0)
            running_sum = running_sum * old_scale + probabilities.sum(dim=-1)
            accumulator.mul_(old_scale[..., None])
            if not bool(physical_skip.all().item()):
                v_tile = v[:, k_start:k_end].transpose(1, 2).float()
                output_probabilities = probabilities
                if row_mask_variant == "output":
                    output_probabilities = probabilities * active_rows[..., None]
                accumulator.add_(torch.matmul(output_probabilities, v_tile))
            running_max = new_max

        if rich_pending:
            final_logsumexp = running_max + running_sum.clamp_min(1e-30).log()
            final_output_norm = torch.linalg.vector_norm(
                accumulator / running_sum.clamp_min(1e-20)[..., None], dim=-1
            )
            for kv_tile_index, trace in rich_pending:
                rich_tile_trace_callback(
                    q_start // q_block_size,
                    kv_tile_index,
                    RichPhysicalTileTrace(
                        **{
                            **trace.__dict__,
                            "row_final_logsumexp": final_logsumexp.detach(),
                            "row_final_output_norm": final_output_norm.detach(),
                        }
                    ),
                )
        normalized = accumulator / running_sum.clamp_min(1e-20)[..., None]
        output[:, q_start:q_end] = normalized.transpose(1, 2).to(q.dtype)
    previous_skipped = torch.tensor(_STATS.skipped_blocks_per_sequence, device=q.device)
    previous_total = torch.tensor(_STATS.total_blocks_per_sequence, device=q.device)
    if previous_skipped.numel() == 0:
        previous_skipped = torch.zeros_like(skipped_per_sequence)
        previous_total = torch.zeros_like(total_per_sequence)
    if previous_skipped.shape != skipped_per_sequence.shape:
        # Aggregate counters remain meaningful across arbitrary calls, while
        # per-sequence counters can only be continued for a stable batch.
        previous_skipped = torch.zeros_like(skipped_per_sequence)
        previous_total = torch.zeros_like(total_per_sequence)
    _STATS.skipped_blocks_per_sequence = tuple((previous_skipped + skipped_per_sequence).cpu().tolist())
    _STATS.total_blocks_per_sequence = tuple((previous_total + total_per_sequence).cpu().tolist())
    return output


@contextmanager
def install_blasst(
    model: torch.nn.Module,
    *,
    blasst_lambda: float | torch.Tensor | Callable[[int], float | torch.Tensor] = 0.001,
    q_block_size: int = 128,
    kv_block_size: int = 64,
    physical_score_callback: Callable[[int, int, int, torch.Tensor], None] | None = None,
    tile_trace_callback: Callable[[int, int, int, PhysicalTileTrace], None] | None = None,
    rich_tile_trace_callback: Callable[[int, int, int, RichPhysicalTileTrace], None] | None = None,
    veto_decision_callback: Callable[[int, int, int, VetoDecisionContext], torch.Tensor] | None = None,
    tile_trace_filter: Callable[[int, int], bool] | None = None,
    query_history: dict[int, torch.Tensor] | None = None,
    row_token_state: torch.Tensor | Callable[[], torch.Tensor] | None = None,
    row_mask_variant: RowMaskVariant = "none",
    row_lambda: float | torch.Tensor | Callable[[int], float | torch.Tensor] | None = None,
    max_active_rows: int = 32,
) -> Iterator[None]:
    """Temporarily replace LLaDA blocks' FlashAttention call target."""
    replaced: list[tuple[torch.nn.Module, object]] = []

    for layer_index, module in enumerate(module for module in model.modules() if hasattr(module, "flash_attn_func")):
        def call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            _layer_index: int = layer_index,
            **kwargs: object,
        ) -> torch.Tensor:
            threshold = blasst_lambda(_layer_index) if callable(blasst_lambda) else blasst_lambda
            callback = None
            if physical_score_callback is not None:
                callback = lambda q_tile, kv_tile, scores: physical_score_callback(
                    _layer_index, q_tile, kv_tile, scores
                )
            trace_callback = None
            if tile_trace_callback is not None:
                trace_callback = lambda q_tile, kv_tile, trace: tile_trace_callback(
                    _layer_index, q_tile, kv_tile, trace
                )
            rich_trace_callback = None
            if rich_tile_trace_callback is not None:
                rich_trace_callback = lambda q_tile, kv_tile, trace: rich_tile_trace_callback(
                    _layer_index, q_tile, kv_tile, trace
                )
            decision_callback = None
            if veto_decision_callback is not None:
                decision_callback = lambda q_tile, kv_tile, context: veto_decision_callback(
                    _layer_index, q_tile, kv_tile, context
                )
            current_token_state = row_token_state() if callable(row_token_state) else row_token_state
            current_row_lambda = row_lambda(_layer_index) if callable(row_lambda) else row_lambda
            query_delta = None
            if query_history is not None:
                previous = query_history.get(_layer_index)
                if previous is not None:
                    previous_float = previous.float()
                    query_delta = torch.linalg.vector_norm(q.float() - previous_float, dim=-1) / (
                        torch.linalg.vector_norm(previous_float, dim=-1) + 1e-6
                    )
                query_history[_layer_index] = q.detach().clone()
            return blasst_flash_attn_func(
                q,
                k,
                v,
                **kwargs,
                blasst_lambda=threshold,
                q_block_size=q_block_size,
                kv_block_size=kv_block_size,
                physical_score_callback=callback,
                tile_trace_callback=trace_callback,
                rich_tile_trace_callback=rich_trace_callback,
                veto_decision_callback=decision_callback,
                tile_trace_filter=tile_trace_filter,
                query_delta=query_delta,
                row_token_state=current_token_state,
                row_mask_variant=row_mask_variant,
                row_lambda=current_row_lambda,
                max_active_rows=max_active_rows,
            )  # type: ignore[arg-type]

        replaced.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
        module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module, original in replaced:
            module.flash_attn_func = original  # type: ignore[attr-defined]
