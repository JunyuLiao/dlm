"""Fused GPU scan for the value-aware online selectors V1 / V2.

V1/V2's routing state is sequential in the candidate tile: ``Z_i`` and ``o_hat_i`` advance with each
decision, so a thresholded decision cannot be replaced by a retrospective top-k mask. The reference
implementation (``v31_value_select.scan_online``) is therefore a loop over tiles in which every head,
unit and row is decided in one batched tensor step. This module runs the SAME loop inside one Triton
program per (query head, 128-row query block), which

  * removes the host-side loop over candidates entirely (no Python iteration per tile and no
    per-candidate CPU/GPU synchronization: the optional-slot quota lives in a register),
  * reads each tile's ``mu`` row block exactly once and contiguously, because ``ValueStats`` is
    tile-major,
  * and reproduces the reference's decisions (same FP32 reductions, same ``>=`` threshold comparison,
    same ``-inf`` handling of rows and tiles without support).

The kernel writes the keep map and its counters; it never writes an attention output. The later FA4
sparse consumer is unchanged and still receives the same ``[1, H, QB, KT]`` bool map.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from experiments.numerical_qk_reuse import v31_value_select as vs

try:                                                  # CPU reference/tests do not require Triton.
    _HAS_TRITON = True
except Exception:                                     # pragma: no cover
    _HAS_TRITON = False


@triton.jit
def _value_scan(MASS, MU, NU, ROWS, PROT, KEEP, COUNTS,
                SM_H, SM_P, SM_N, MU_H, MU_P, MU_N, PROT_H, PROT_B, KEEP_H, KEEP_B,
                CNT_H, CNT_B, NT, RANK: tl.constexpr, ROWS_PER_UNIT: tl.constexpr, THRESHOLD, BUDGET,
                KEEP_SKIPPED_MASS: tl.constexpr, EPS: tl.constexpr):
    """One program per (head, 128-row query block); ``NT`` optional tiles scanned in ascending order.

    ``MASS`` holds ``Z_ij`` already scaled by each row's largest prefix tile log-mass (0 where the
    tile has no legal key), exactly the tensor the reference derives from ``ValueStats.finite_mass``;
    ``MU`` the attention-weighted within-tile projected mean. ``KEEP`` receives 0/1 for the optional
    tiles; ``COUNTS`` receives (forced_keep, forced_skip, threshold_keep) per unit."""
    h = tl.program_id(0)
    b = tl.program_id(1)
    r = tl.arange(0, ROWS_PER_UNIT)
    ri = tl.arange(0, RANK)
    base_n = b * ROWS_PER_UNIT + r
    row_ok = tl.load(ROWS + h * ROWS_PER_UNIT + base_n) > 0
    nu = tl.maximum(tl.load(NU + h * ROWS_PER_UNIT + base_n, mask=row_ok, other=EPS), EPS)
    optional = BUDGET
    running = tl.zeros((ROWS_PER_UNIT,), tl.float32)
    num = tl.zeros((ROWS_PER_UNIT, RANK), tl.float32)
    has_support = 0
    forced = 0
    skipped = 0
    throught = 0
    for j in range(NT):
        m = tl.load(MASS + h * SM_H + j * SM_P + base_n, mask=row_ok, other=0.0)
        den = running + m
        eta = tl.where(den > 0, m / tl.maximum(den, EPS), 0.0)
        mu = tl.load(MU + h * MU_H + j * MU_P + base_n[:, None] * MU_N + ri[None, :],
                     mask=row_ok[:, None], other=0.0)
        est = num / tl.maximum(running, EPS)[:, None]
        delta = mu - est
        term = eta * tl.sqrt(tl.sum(delta * delta, axis=1)) / nu
        rho = tl.max(tl.where(row_ok, term, float('-inf')), 0)
        first = has_support == 0
        prot = tl.load(PROT + h * PROT_H + b * PROT_B + j) > 0
        remaining = NT - j
        take = tl.where(optional <= 0, 0,
                        tl.where(optional >= remaining, 1,
                                 tl.where((rho >= THRESHOLD) | first | prot, 1, 0)))
        take = tl.where(prot, 1, take)
        forced_t = tl.where(take == 1 and (optional <= 0 or optional >= remaining or first or prot), 1, 0)
        tl.store(KEEP + h * KEEP_H + b * KEEP_B + j, take)
        grow = tl.where((KEEP_SKIPPED_MASS) or (take == 1), 1.0, 0.0)
        num = num + grow[:, None] * m[:, None] * mu
        running = running + tl.where((KEEP_SKIPPED_MASS) or (take == 1), m, 0.0)
        optional = optional - take
        has_support = has_support | take
        forced += forced_t
        throught += take - forced_t
        skipped += 1 - take
    tl.store(COUNTS + h * CNT_H + b * CNT_B + 0, forced)
    tl.store(COUNTS + h * CNT_H + b * CNT_B + 1, skipped)
    tl.store(COUNTS + h * CNT_H + b * CNT_B + 2, throught)


def value_scan(stats, budget, threshold, keep_skipped_mass, protect=None, counters=None, num_warps=4):
    """Fused GPU V1/V2 selection. Returns ``(keep[1, H, QB, KT], counters)``."""
    z, mu, nu, rows = stats.z, stats.mu, stats.nu, stats.rows
    h, pt, n_pad = z.shape
    blocks = stats.blocks
    dev = z.device
    ptiles = min(int(budget), pt)
    if protect is not None:
        # protected tiles are mandatory AND charged against the optional budget, as in the control
        ptiles = max(0, ptiles - int(protect[:, :, :pt].sum(-1).sum()))
    keep = torch.zeros((h, blocks, stats.total_tiles), dtype=torch.int32, device=dev)
    counts = torch.zeros((h, blocks, 3), dtype=torch.int32, device=dev)
    if pt:
        mass = stats.finite_mass()
        mu = mu.contiguous()
        _value_scan[(h, blocks)](
            mass, mu, nu.contiguous(), rows.contiguous(),
            (protect if protect is not None else keep.bool()).contiguous(), keep, counts,
            mass.stride(0), mass.stride(1), mass.stride(2), mu.stride(0), mu.stride(1), mu.stride(2),
            keep.stride(0), keep.stride(1), keep.stride(0), keep.stride(1),
            counts.stride(0), counts.stride(1),
            int(pt), RANK=vs.RANK, ROWS_PER_UNIT=128, THRESHOLD=float(threshold), BUDGET=ptiles,
            KEEP_SKIPPED_MASS=bool(keep_skipped_mass), EPS=vs.EPS, num_warps=num_warps, num_stages=1)
    keep = keep.bool()
    keep[..., pt:] = True
    if protect is not None:
        keep |= protect
    c = counters or vs.Counters()
    forced_keep, forced_skip, thr = (int(x) for x in counts.sum(1).sum(0).tolist())
    c.candidates += int(pt)
    c.passes += 1
    c.forced_keep, c.forced_skip, c.threshold_keeps = forced_keep, forced_skip, thr
    c.extra['kernel'] = 'triton_value_scan'
    return keep.reshape(1, h, blocks, stats.total_tiles), c