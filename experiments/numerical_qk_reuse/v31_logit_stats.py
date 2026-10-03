"""v31 fused logit statistics for the sampler hook (execution variant; the selector's inputs are unchanged).

The method's per-step hook needs, per canvas position, the top-1 token (fast T), the top-1 probability (C gate) and
the sampler's entropy-bound acceptance mask (C gate). Recomputing them with torch ops reads the [CL, vocab] FP32
temperature-scaled logits (256 x 262144 = 268 MB) several times and allocates full-size temporaries; in the v31 step
profile this cost 1.9 ms per denoising step with the C gate. ``row_stats`` makes one pass per row (online max /
sum-exp / entropy, first-occurrence argmax) and returns argmax, max, logsumexp and entropy.

Equivalence: argmax is exact (first maximal index, as torch.argmax). max is exact. logsumexp and entropy are the same
quantities with a different FP32 summation order, so p_top = exp(max - lse) and the entropy agree to FP32 rounding;
the acceptance mask can only differ at an exact tie with the entropy bound. ``accepted_from_entropy`` applies the
official rule (vLLM diffusion_gemma._compiled_sample_step, phase 4) to the entropies.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class RowStats:
    argmax: torch.Tensor    # [rows] int64
    max: torch.Tensor       # [rows] fp32
    lse: torch.Tensor       # [rows] fp32
    entropy: torch.Tensor   # [rows] fp32


@triton.jit
def _row_stats(X, ARG, MX, LSE, ENT, V, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = X + row.to(tl.int64) * STRIDE
    offs0 = tl.arange(0, BLOCK)
    m = -float('inf')
    s0 = 0.
    s1 = 0.
    best = 0
    for start in range(0, V, BLOCK):
        offs = start + offs0
        ok = offs < V
        x = tl.load(base + offs, mask=ok, other=-float('inf')).to(tl.float32)
        bm = tl.max(x, 0)
        bi = tl.argmax(x, 0, tie_break_left=True)
        best = tl.where(bm > m, start + bi, best)               # strict: keeps the first maximal index
        new_m = tl.maximum(m, bm)
        alpha = tl.where(m > -float('inf'), tl.exp(m - new_m), 0.)
        shift = tl.where(m > -float('inf'), m - new_m, 0.)
        e = tl.where(ok, tl.exp(x - new_m), 0.)
        d = tl.where(ok, x - new_m, 0.)
        # s0 = sum exp(x - m), s1 = sum exp(x - m) * (x - m), both rescaled to the running max
        s1 = alpha * (s1 + shift * s0) + tl.sum(e * d, 0)
        s0 = alpha * s0 + tl.sum(e, 0)
        m = new_m
    tl.store(ARG + row, best.to(tl.int64))
    tl.store(MX + row, m)
    tl.store(LSE + row, m + tl.log(s0))
    tl.store(ENT + row, tl.log(s0) - s1 / s0)                  # -sum p log p = log s0 - s1 / s0


def row_stats(x: torch.Tensor, block: int = 4096) -> RowStats:
    """x [..., V] (any float dtype, last dim contiguous) -> per-row statistics over the leading dims."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    v = x.shape[-1]
    rows = x.reshape(-1, v)
    n = rows.shape[0]
    dev = x.device
    arg = torch.empty(n, device=dev, dtype=torch.int64)
    mx = torch.empty(n, device=dev, dtype=torch.float32)
    lse = torch.empty(n, device=dev, dtype=torch.float32)
    ent = torch.empty(n, device=dev, dtype=torch.float32)
    _row_stats[(n,)](rows, arg, mx, lse, ent, v, rows.stride(0), BLOCK=block, num_warps=8)
    shape = x.shape[:-1]
    return RowStats(arg.view(shape), mx.view(shape), lse.view(shape), ent.view(shape))


def accepted_from_entropy(entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    """The official entropy-bound acceptance rule on [B, CL] entropies."""
    s, idx = torch.sort(entropy, dim=-1)
    keep = (torch.cumsum(s, dim=-1) - torch.cummax(s, dim=-1).values) <= entropy_bound
    return torch.zeros_like(keep).scatter_(1, idx, keep)


def observe_logits_from_stats(state, stats: RowStats, accepted, cur_step):
    """``State.observe_logits`` on the fast-T path with precomputed row statistics (same updates, same order).
    Only valid for state.fast_t (T and C gate); the caller checks."""
    top = stats.argmax
    flip = None if state.previous_top is None else top != state.previous_top
    if state.cgate is not None:
        p_top = (stats.max - stats.lse).exp()
        acc = accepted.to(torch.float32)
        q = torch.ones_like(p_top) if state.cg_q is None else state.cg_q
        r = torch.zeros_like(p_top) if state.cg_r is None else state.cg_r
        state.cg_q = state.cgate['gamma_q'] * q + (1 - state.cgate['gamma_q']) * (1 - acc)
        stay = (r + 1) * acc
        state.cg_r = stay if flip is None else torch.where(flip, torch.zeros_like(stay), stay)
        state.cg_u = (1 - p_top).clamp_min(0).sqrt()
    if state.temporal is None:
        state.temporal = torch.zeros(top.shape, device=top.device, dtype=torch.float32)
    if flip is not None:
        state.temporal = state.gamma * state.temporal + (1 - state.gamma) * flip.float()
    state.previous_top = top.detach()
    state.t_history = True
