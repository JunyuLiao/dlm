"""Minimal BLASST modification of FlashAttention's online-softmax loop.

Only Algorithm 1 is implemented: no pipeline, warp, batched-load, or
architecture-specific optimizations. Shapes match ``flash_attn_func``:
``[batch, sequence, heads, head_dim]``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

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
            if physical_score_callback is not None:
                tile_log_score = torch.where(valid_rows, row_gap, -torch.inf).amax(dim=-1)
                physical_score_callback(q_start // q_block_size, k_start // kv_block_size, tile_log_score.exp())
            threshold = log_threshold[..., None] if isinstance(log_threshold, torch.Tensor) else log_threshold
            skip = valid_rows & (row_gap < threshold)
            _STATS.skipped_row_blocks += int(skip.sum().item())
            _STATS.total_row_blocks += int(valid_rows.sum().item())
            physical_skip = (skip | ~valid_rows).all(dim=-1) & valid_rows.any(dim=-1)
            _STATS.skipped_blocks += int(physical_skip.sum().item())
            _STATS.total_blocks += int(valid_rows.any(dim=-1).sum().item())
            skipped_per_sequence += physical_skip.sum(dim=-1)
            total_per_sequence += valid_rows.any(dim=-1).sum(dim=-1)

            # The realizable unit is the complete [Q tile, KV tile] for one
            # sequence and head. A non-unanimous tile is evaluated for every
            # row; a unanimous skip leaves the online-softmax state unchanged.
            proposed_max = torch.maximum(running_max, block_max)
            new_max = torch.where(physical_skip[..., None], running_max, proposed_max)
            old_scale = torch.exp(running_max - new_max)
            old_scale = torch.where(torch.isfinite(old_scale), old_scale, torch.zeros_like(old_scale))
            probabilities = torch.exp(scores - new_max[..., None])
            probabilities = torch.where(torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities))
            probabilities.masked_fill_(physical_skip[..., None, None], 0.0)
            running_sum = running_sum * old_scale + probabilities.sum(dim=-1)
            accumulator.mul_(old_scale[..., None])
            if not bool(physical_skip.all().item()):
                v_tile = v[:, k_start:k_end].transpose(1, 2).float()
                accumulator.add_(torch.matmul(probabilities, v_tile))
            running_max = new_max

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
            return blasst_flash_attn_func(
                q,
                k,
                v,
                **kwargs,
                blasst_lambda=threshold,
                q_block_size=q_block_size,
                kv_block_size=kv_block_size,
                physical_score_callback=callback,
            )  # type: ignore[arg-type]

        replaced.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
        module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module, original in replaced:
            module.flash_attn_func = original  # type: ignore[attr-defined]
