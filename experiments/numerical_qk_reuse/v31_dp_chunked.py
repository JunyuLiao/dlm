"""v31 chunked dense-prefix build: the M1-DP state of ``v27_dense_prefix.build`` with a parallel prefix scan
(execution variant; same formulas, different FP32 summation order).

``v27_dense_prefix._dp_build`` scans the prefix tiles sequentially with one program per (query block, head): 32
programs for 1020 tiles at 65K keys, 3.2 ms per GLOBAL layer per canvas in the v31 CUPTI profile (more than all sparse
attention calls of that canvas together). The state it carries per query row is a running log-mass ``previous`` and
the mass-weighted mean ``projected`` of the eligible tiles' mu, i.e. an associative (logaddexp, weighted mean) pair.
Here:
  1. ``_chunk_pass(MODE=0)``: every chunk of CHUNK tiles runs the original recurrence from the empty state and stores
     its own (log-mass, weighted mean);
  2. ``_chunk_scan``: an exclusive scan over the chunks gives each chunk's start state and the final state;
  3. ``_chunk_pass(MODE=1)``: every chunk reruns the original per-tile body from its start state and writes lognorm /
     eligible / bad exactly as ``_dp_build`` does.
Decisions can differ from the sequential build only where a risk ties the threshold to FP32 rounding.
"""
from __future__ import annotations

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .cached_executor import _logadd
from .v27_dense_prefix import DensePrefixState


@tr.jit(do_not_specialize=['KT', 'PREFIX_TILES'])
def _chunk_pass(ZSUM, MUSUM, ACTSUM, BADSUM, POOLED, LOGN, DPELIG, DPBAD, CLSE, CPROJ,
                Q: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                QB: tl.constexpr, KT, PREFIX_TILES, NCH: tl.constexpr, CHUNK: tl.constexpr,
                COMPACT: tl.constexpr, MODE: tl.constexpr):
    qb, h, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch = 0
    kh = h // (H // HK)
    r128 = tl.arange(0, 128)
    ri = tl.arange(0, RP)
    rows = qb*128 + r128 < Q
    cstate = ((h*QB+qb)*NCH+c)*128 + r128
    if MODE == 0:
        previous = tl.full((128,), -float('inf'), tl.float32)
        projected = tl.full((128, RP), 0., tl.float32)
    else:
        previous = tl.load(CLSE+cstate, rows, other=-float('inf'))
        projected = tl.load(CPROJ+cstate[:, None]*RP+ri[None, :], rows[:, None] & (ri[None, :] < R), other=0.)
    lo = c*CHUNK
    hi = tl.minimum(lo+CHUNK, PREFIX_TILES)
    for j in range(lo, hi):
        base = (((batch*H+h)*QB+qb)*PREFIX_TILES+j)*128 + r128
        block_z = tl.load(ZSUM+base, rows, other=-float('inf'))
        if COMPACT:
            pooled = tl.load(POOLED+((batch*HK+kh)*KT+j)*RP+ri, ri < R, other=0.)
            mu = tl.zeros((128, RP), tl.float32) + pooled[None, :]
        else:
            mu = tl.load(MUSUM+base[:, None]*RP+ri[None, :], rows[:, None] & (ri[None, :] < R), other=0.)
        active = tl.load(ACTSUM+base, rows, other=0) != 0
        bad_row = tl.load(BADSUM+base, rows, other=0) != 0
        eligible = tl.sum((active | bad_row).to(tl.int32), 0) > 0
        combined = _logadd(previous, block_z)
        safe = tl.where(combined > -float('inf'), combined, 0.)
        alpha = tl.where(active, lib.exp(block_z-safe), 0.)
        if MODE == 1:
            delta = alpha[:, None]*(mu-projected)
            lognorm = lib.log(tl.sqrt(tl.sum(delta*delta, 1)))
            lognorm = tl.where(active, tl.where(previous > -float('inf'), lognorm, float('inf')), -float('inf'))
            lognorm = tl.where(bad_row, float('inf'), lognorm)
            tl.store(LOGN+base, lognorm, rows)
            tile = ((batch*H+h)*QB+qb)*PREFIX_TILES + j
            tl.store(DPELIG+tile, eligible.to(tl.int8))
            tl.store(DPBAD+tile, (tl.sum(bad_row.to(tl.int32), 0) > 0).to(tl.int8))
        old = lib.exp(previous-safe)
        projected = tl.where(eligible, old[:, None]*projected+alpha[:, None]*mu, projected)
        previous = tl.where(eligible, combined, previous)
    if MODE == 0:
        tl.store(CLSE+cstate, previous, rows)
        tl.store(CPROJ+cstate[:, None]*RP+ri[None, :], projected, rows[:, None] & (ri[None, :] < R))


@tr.jit
def _chunk_scan(CLSE, CPROJ, DPPREV, DPPROJ, Q: tl.constexpr, H: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                QB: tl.constexpr, NCH: tl.constexpr):
    """Exclusive scan over the chunk summaries, in place: chunk c's slot becomes the state before chunk c."""
    qb, h = tl.program_id(0), tl.program_id(1)
    r128 = tl.arange(0, 128)
    ri = tl.arange(0, RP)
    rows = qb*128 + r128 < Q
    m2 = rows[:, None] & (ri[None, :] < R)
    previous = tl.full((128,), -float('inf'), tl.float32)
    projected = tl.full((128, RP), 0., tl.float32)
    for c in range(NCH):
        cstate = ((h*QB+qb)*NCH+c)*128 + r128
        lse = tl.load(CLSE+cstate, rows, other=-float('inf'))
        proj = tl.load(CPROJ+cstate[:, None]*RP+ri[None, :], m2, other=0.)
        tl.store(CLSE+cstate, previous, rows)
        tl.store(CPROJ+cstate[:, None]*RP+ri[None, :], projected, m2)
        combined = _logadd(previous, lse)
        safe = tl.where(combined > -float('inf'), combined, 0.)
        projected = lib.exp(previous-safe)[:, None]*projected + lib.exp(lse-safe)[:, None]*proj
        previous = combined
    state = (h*QB+qb)*128 + r128
    tl.store(DPPREV+state, previous, rows)
    tl.store(DPPROJ+state[:, None]*RP+ri[None, :], projected, m2)


def build_chunked(summary, nq, kt, hk, pooled=None, identity=None, chunk=32, num_warps=4, num_stages=3):
    """Same contract and outputs as ``v27_dense_prefix.build`` (batch 1)."""
    b, h, qb, pt, _ = summary.z.shape
    if b != 1:
        raise ValueError('chunked dense-prefix build is written for batch 1')
    compact = pooled is not None
    if not compact and summary.mu.shape[-1] != 32:
        raise ValueError('M1-DP on exact mu needs the rank-32 summary')
    dev = summary.z.device
    state = DensePrefixState(
        lognorm=torch.empty((b, h, qb, pt, 128), device=dev, dtype=torch.float32),
        eligible=torch.empty((b, h, qb, pt), device=dev, dtype=torch.int8),
        bad=torch.empty((b, h, qb, pt), device=dev, dtype=torch.int8),
        previous=torch.empty((b, h, qb, 128), device=dev, dtype=torch.float32),
        projected=torch.zeros((b, h, qb, 128, 32), device=dev, dtype=torch.float32),
        prefix_tiles=pt, identity=identity if identity is not None else summary.identity)
    nch = max(1, -(-pt // chunk))
    clse = torch.empty((h, qb, nch, 128), device=dev, dtype=torch.float32)
    cproj = torch.zeros((h, qb, nch, 128, 32), device=dev, dtype=torch.float32)
    dummy = torch.empty((1,), device=dev, dtype=torch.float32)
    args = (summary.z, summary.mu if not compact else dummy, summary.active, summary.bad, pooled if compact else dummy,
            state.lognorm, state.eligible, state.bad, clse, cproj, nq, h, hk, 32, 32, qb, kt, pt, nch, chunk, compact)
    _chunk_pass[(qb, h, nch)](*args, 0, num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    _chunk_scan[(qb, h)](clse, cproj, state.previous, state.projected, nq, h, 32, 32, qb, nch,
                         num_warps=num_warps, enable_fp_fusion=False)
    _chunk_pass[(qb, h, nch)](*args, 1, num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=False)
    return state
