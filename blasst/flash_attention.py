"""Minimal BLASST modification of FlashAttention's online-softmax loop.

Only Algorithm 1 is implemented: no pipeline, warp, batched-load, or
architecture-specific optimizations. Shapes match ``flash_attn_func``:
``[batch, sequence, heads, head_dim]``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class BlasstStats:
    skipped_blocks: int = 0  # Physically skippable (batch, head, Q tile, KV tile).
    total_blocks: int = 0
    skipped_row_blocks: int = 0
    total_row_blocks: int = 0

    @property
    def sparsity_ratio(self) -> float:
        return self.skipped_blocks / self.total_blocks if self.total_blocks else 0.0

    def reset(self) -> None:
        self.skipped_blocks = self.total_blocks = 0
        self.skipped_row_blocks = self.total_row_blocks = 0


_STATS = BlasstStats()


def collect_blasst_stats(*, reset: bool = False) -> BlasstStats:
    result = BlasstStats(
        _STATS.skipped_blocks,
        _STATS.total_blocks,
        _STATS.skipped_row_blocks,
        _STATS.total_row_blocks,
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
    blasst_lambda: float = 0.001,
    q_block_size: int = 128,
    kv_block_size: int = 64,
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
    if not 0.0 <= blasst_lambda <= 1.0 or q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("lambda must be in [0, 1]; block sizes must be positive")
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

    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    log_threshold = math.log(blasst_lambda) if blasst_lambda else -math.inf
    output = torch.empty((batch, query_len, query_heads, v.shape[-1]), dtype=q.dtype, device=q.device)

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
            new_max = torch.maximum(running_max, block_max)
            valid_rows = torch.isfinite(block_max)
            skip = valid_rows & ((block_max - new_max) < log_threshold)
            _STATS.skipped_row_blocks += int(skip.sum().item())
            _STATS.total_row_blocks += int(valid_rows.sum().item())
            physical_skip = (skip | ~valid_rows).all(dim=-1) & valid_rows.any(dim=-1)
            _STATS.skipped_blocks += int(physical_skip.sum().item())
            _STATS.total_blocks += int(valid_rows.any(dim=-1).sum().item())

            old_scale = torch.exp(running_max - new_max)
            old_scale = torch.where(torch.isfinite(old_scale), old_scale, torch.zeros_like(old_scale))
            probabilities = torch.exp(scores - new_max[..., None])
            probabilities = torch.where(torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities))
            probabilities.masked_fill_(skip[..., None], 0.0)
            running_sum = running_sum * old_scale + probabilities.sum(dim=-1)
            accumulator.mul_(old_scale[..., None])
            if not bool(skip.all().item()):
                v_tile = v[:, k_start:k_end].transpose(1, 2).float()
                accumulator.add_(torch.matmul(probabilities, v_tile))
            running_max = new_max

        normalized = accumulator / running_sum.clamp_min(1e-20)[..., None]
        output[:, q_start:q_end] = normalized.transpose(1, 2).to(q.dtype)
    return output


@contextmanager
def install_blasst(
    model: torch.nn.Module,
    *,
    blasst_lambda: float = 0.001,
    q_block_size: int = 128,
    kv_block_size: int = 64,
) -> Iterator[None]:
    """Temporarily replace LLaDA blocks' FlashAttention call target."""
    replaced: list[tuple[torch.nn.Module, object]] = []

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return blasst_flash_attn_func(
            q,
            k,
            v,
            **kwargs,
            blasst_lambda=blasst_lambda,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
        )  # type: ignore[arg-type]

    for module in model.modules():
        if hasattr(module, "flash_attn_func"):
            replaced.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
            module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module, original in replaced:
            module.flash_attn_func = original  # type: ignore[attr-defined]
