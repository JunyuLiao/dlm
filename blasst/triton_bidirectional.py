"""Fused BLASST forward kernel for LLaDA's bidirectional self-attention.

The pruning unit is a physical 2D (query, KV) tile.  BMM1 (QK^T) always
runs.  A tile skips softmax and BMM2 (PV) only when every valid query row
votes that its local maximum is below the running maximum by log2(lambda).
"""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import triton
import triton.language as tl


@triton.jit
def _blasst_bidirectional_fwd(
    Q,
    K,
    V,
    O,
    STATS,
    softmax_scale,
    LOG_THRESHOLDS,
    stride_qb: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    seqlen: tl.constexpr,
    nheads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    q_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // nheads
    head = batch_head % nheads

    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    valid_m = offs_m < seqlen
    log_threshold = tl.load(LOG_THRESHOLDS + batch)

    q_ptrs = Q + batch * stride_qb + offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)

    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # FlashAttention 2 traverses noncausal KV tiles from right to left.
    for reverse_block in tl.range(0, tl.cdiv(seqlen, BLOCK_N), num_stages=PIPELINE_STAGES):
        start_n = (tl.cdiv(seqlen, BLOCK_N) - 1 - reverse_block) * BLOCK_N
        key_pos = start_n + offs_n
        valid_n = key_pos < seqlen
        k_ptrs = K + batch * stride_kb + key_pos[:, None] * stride_kn + head * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)

        # BLASST prefill design: preserve the normal V prefetch pipeline.  V
        # traffic is not the bottleneck in compute-bound full-sequence
        # attention; pruning saves softmax work and BMM2 without serializing
        # the K/BMM1 pipeline on a data-dependent load.
        v_ptrs = V + batch * stride_vb + key_pos[:, None] * stride_vn + head * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)

        # Keep the online-softmax state in base-2 units. Triton's exp2 maps
        # directly to the GPU approximation used by optimized FlashAttention;
        # multiplying both scores and the threshold by log2(e) is
        # mathematically equivalent to the natural-exponential formulation.
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * (softmax_scale * 1.4426950408889634)
        scores = tl.where(valid_m[:, None] & valid_n[None, :], scores, -float("inf"))
        local_max = tl.max(scores, axis=1)

        # Vote against the existing global maximum, as in BLASST. A unanimous
        # row vote is equivalent to reducing the largest gap. Delaying the
        # online-max update means rejected tiles do not compute it at all.
        max_gap = tl.max(tl.where(valid_m, local_max - running_max, -float("inf")), axis=0)
        tile_skip = max_gap < log_threshold

        if COLLECT_STATS:
            tl.atomic_add(STATS + 1, 1)
            if tile_skip:
                tl.atomic_add(STATS, 1)

        # Dynamic uniform branch: a skipped physical tile avoids exp, rowsum,
        # and P@V. V was deliberately prefetched above. Since all rows voted
        # to skip, online-softmax state is unchanged.
        if not tile_skip:
            next_max = tl.maximum(running_max, local_max)
            old_scale = tl.exp2(running_max - next_max)
            old_scale = tl.where(running_max == -float("inf"), 0.0, old_scale)
            probabilities = tl.exp2(scores - next_max[:, None])
            # Invalid KV scores are already -inf, hence exp2 is zero. Only
            # partial query tiles need an explicit mask to suppress -inf/-inf.
            probabilities = tl.where(valid_m[:, None], probabilities, 0.0)
            running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
            acc = acc * old_scale[:, None]

            acc += tl.dot(probabilities.to(tl.bfloat16), v, out_dtype=tl.float32)
            running_max = next_max

    output = acc / running_sum[:, None]
    o_ptrs = O + batch * stride_ob + offs_m[:, None] * stride_om + head * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, output, mask=valid_m[:, None])


@dataclass
class KernelStats:
    skipped_tiles: int
    total_tiles: int

    @property
    def sparsity_ratio(self) -> float:
        return self.skipped_tiles / self.total_tiles if self.total_tiles else 0.0


@dataclass(frozen=True)
class DiffusionLambdaSchedule:
    """Noise-aware thresholds calibrated for LLaDA's denoising trajectory."""

    high_noise_lambda: float = 0.03
    mid_noise_lambda: float = 0.3
    low_noise_lambda: float = 1.0
    high_noise_boundary: float = 0.75
    low_noise_boundary: float = 0.25

    def threshold(self, remaining_mask_ratio: float) -> float:
        if not 0.0 <= remaining_mask_ratio <= 1.0:
            raise ValueError("remaining_mask_ratio must be in [0, 1]")
        if remaining_mask_ratio >= self.high_noise_boundary:
            return self.high_noise_lambda
        if remaining_mask_ratio >= self.low_noise_boundary:
            return self.mid_noise_lambda
        return self.low_noise_lambda


@dataclass
class DiffusionKernelController:
    schedule: DiffusionLambdaSchedule
    current_threshold: float | torch.Tensor = 0.03

    def __post_init__(self) -> None:
        self.current_threshold = self.schedule.high_noise_lambda

    @property
    def blasst_lambda(self) -> float | torch.Tensor:
        return self.current_threshold

    def set_remaining_mask_ratio(self, ratio: float | torch.Tensor) -> None:
        # Called once per denoising step before the LLaDA forward.
        if isinstance(ratio, torch.Tensor):
            if ratio.ndim != 1 or bool(((ratio < 0) | (ratio > 1)).any()):
                raise ValueError("per-sequence mask ratios must be a vector in [0, 1]")
            self.current_threshold = torch.where(
                ratio >= self.schedule.high_noise_boundary,
                self.schedule.high_noise_lambda,
                torch.where(
                    ratio >= self.schedule.low_noise_boundary,
                    self.schedule.mid_noise_lambda,
                    self.schedule.low_noise_lambda,
                ),
            ).float()
        else:
            self.current_threshold = self.schedule.threshold(ratio)


_DEVICE_STATS: torch.Tensor | None = None
_SCALAR_LOG_THRESHOLD_CACHE: dict[tuple[int, int, float], torch.Tensor] = {}
_TENSOR_LOG_THRESHOLD_CACHE: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int, torch.Tensor]] = {}


def reset_kernel_stats(device: torch.device | str = "cuda") -> None:
    global _DEVICE_STATS
    _DEVICE_STATS = torch.zeros(2, dtype=torch.int64, device=device)


def get_kernel_stats(*, reset: bool = False) -> KernelStats:
    global _DEVICE_STATS
    if _DEVICE_STATS is None:
        return KernelStats(0, 0)
    values = _DEVICE_STATS.cpu().tolist()
    result = KernelStats(int(values[0]), int(values[1]))
    if reset:
        _DEVICE_STATS.zero_()
    return result


def blasst_bidirectional_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    blasst_lambda: float | torch.Tensor = 0.03,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
) -> torch.Tensor:
    """Drop-in inference replacement for LLaDA's ``flash_attn_func``."""
    del deterministic
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("LLaDA BLASST kernel requires CUDA BF16 tensors")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("LLaDA BLASST kernel is forward-only inference")
    if dropout_p != 0.0 or causal:
        raise ValueError("kernel supports dropout-free bidirectional attention only")
    if window_size != (-1, -1) or softcap != 0.0 or alibi_slopes is not None or return_attn_probs:
        raise NotImplementedError("windowing, softcap, ALiBi, and returned probabilities are unsupported")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("LLaDA kernel currently requires same-shape MHA tensors [B, S, H, D]")
    if q.shape[-1] != 128:
        raise ValueError("LLaDA kernel is specialized for head dimension 128")

    global _DEVICE_STATS
    if _DEVICE_STATS is None or _DEVICE_STATS.device != q.device:
        reset_kernel_stats(q.device)
    assert _DEVICE_STATS is not None

    batch, seqlen, nheads, head_dim = q.shape
    if isinstance(blasst_lambda, torch.Tensor):
        if blasst_lambda.shape != (batch,) or blasst_lambda.device != q.device:
            raise ValueError("tensor blasst_lambda must be a CUDA vector with one value per sequence")
        if bool(((blasst_lambda < 0) | (blasst_lambda > 1)).any()):
            raise ValueError("blasst_lambda values must be in [0, 1]")
        cached = _TENSOR_LOG_THRESHOLD_CACHE.get(id(blasst_lambda))
        if cached is not None and cached[0]() is blasst_lambda and cached[1] == blasst_lambda._version:
            log_thresholds = cached[2]
        else:
            log_thresholds = torch.where(
                blasst_lambda > 0,
                blasst_lambda.log2(),
                torch.full_like(blasst_lambda, -torch.inf),
            ).float()
            _TENSOR_LOG_THRESHOLD_CACHE[id(blasst_lambda)] = (
                weakref.ref(blasst_lambda),
                blasst_lambda._version,
                log_thresholds,
            )
    else:
        if not 0.0 <= blasst_lambda <= 1.0:
            raise ValueError("blasst_lambda must be in [0, 1]")
        cache_key = (q.device.index or 0, batch, float(blasst_lambda))
        log_thresholds = _SCALAR_LOG_THRESHOLD_CACHE.get(cache_key)
        if log_thresholds is None:
            value = math.log2(blasst_lambda) if blasst_lambda else -math.inf
            log_thresholds = torch.full((batch,), value, dtype=torch.float32, device=q.device)
            _SCALAR_LOG_THRESHOLD_CACHE[cache_key] = log_thresholds
    output = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    grid = (triton.cdiv(seqlen, 128), batch * nheads)
    _blasst_bidirectional_fwd[grid](
        q,
        k,
        v,
        output,
        _DEVICE_STATS,
        scale,
        log_thresholds,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        seqlen,
        nheads,
        BLOCK_M=128,
        BLOCK_N=64,
        HEAD_DIM=128,
        PIPELINE_STAGES=pipeline_stages,
        COLLECT_STATS=collect_stats,
        num_warps=num_warps,
        num_stages=pipeline_stages,
    )
    return output


@contextmanager
def install_bidirectional_blasst_kernel(
    model: torch.nn.Module,
    *,
    blasst_lambda: float | torch.Tensor | Callable[[], float | torch.Tensor] = 0.03,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
) -> Iterator[None]:
    """Install the fused kernel only at LLaDA FlashAttention call sites."""
    originals: list[tuple[torch.nn.Module, object]] = []

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        threshold = blasst_lambda() if callable(blasst_lambda) else blasst_lambda
        return blasst_bidirectional_flash_attn_func(
            q,
            k,
            v,
            **kwargs,
            blasst_lambda=threshold,
            collect_stats=collect_stats,
            num_warps=num_warps,
            pipeline_stages=pipeline_stages,
        )

    for module in model.modules():
        if hasattr(module, "flash_attn_func"):
            originals.append((module, module.flash_attn_func))  # type: ignore[attr-defined]
            module.flash_attn_func = call  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module, original in originals:
            module.flash_attn_func = original  # type: ignore[attr-defined]


@contextmanager
def install_diffusion_blasst_kernel(
    model: torch.nn.Module,
    *,
    schedule: DiffusionLambdaSchedule | None = None,
    collect_stats: bool = True,
    num_warps: int = 4,
    pipeline_stages: int = 2,
) -> Iterator[DiffusionKernelController]:
    """Install a noise-aware kernel and yield its per-step controller."""
    controller = DiffusionKernelController(schedule or DiffusionLambdaSchedule())
    with install_bidirectional_blasst_kernel(
        model,
        blasst_lambda=lambda: controller.blasst_lambda,
        collect_stats=collect_stats,
        num_warps=num_warps,
        pipeline_stages=pipeline_stages,
    ):
        yield controller
