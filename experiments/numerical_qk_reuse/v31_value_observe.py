"""GPU statistics for the value-aware v31 selectors: the per-row projected within-tile mean.

WHY A SEPARATE PASS. At an initial/refresh call the observation output must be the model's native
current-step BF16 attention output on the existing path (FA4), so it is NOT replaced by a projection,
a dense-shaped masked PV or a cached old output. The value-aware selectors additionally need
``mu_ij = (sum_{u in j} exp(s_iu) z_u) / (sum_{u in j} exp(s_iu))`` per (query row, 64-key tile),
which the FA4 in-kernel observation (``v31_fa4_observe``) does not write. This module supplies that
statistic WITHOUT changing the output path and WITHOUT a second model forward:

  * ``observe_mu`` runs the repository's existing Triton fused observation with ``OUT=0`` (no V load,
    no PV, no output partials) and ``MU=1``. It is the same kernel, the same FP32 score tile, the same
    tile max / ``exp`` reductions and the same ``tl.dot(w, sketch, input_precision='tf32x3')`` used by
    the inherited online router's statistics, so ``z``/``mu`` agree with that path's bits. Its
    tail buffer is discarded and its z is replaced by the FA4 observation's own z.
  * ``tail_statistics`` reduces the canvas/boundary tiles the active control already computes in FP32
    for its tail mass, adding them to the same buffer so V3's reference covers the full support.

Cost is one QK pass plus one rank-32 product per tile, with no value load and no PV. It is charged to
the initial/refresh call only, never to a held call; the FA4 sparse consumer, its map type and its
output are untouched.
"""
from __future__ import annotations

import torch

from experiments.numerical_qk_reuse import v31_value_select as vs


def observe_mu(q, k, sketch, scale, prefix_tiles, summary, splits=2, mu_precision='tf32x3'):
    """Observation-only projected-mean pass over the prefix tiles.

    ``q`` [1, H, n, D], ``k`` [1, HK, nk, D], ``sketch`` [1, HK, nk, 32] (the BF16 frozen Gaussian32
    projection of the canvas's values), ``summary`` a ``cached_executor.PrefixSummary`` sized for
    ``prefix_tiles`` tiles. Returns the same ``summary`` (its ``z``/``mu``/``active``/``bad`` are
    written); the caller keeps FA4's own z and mu as the authority for z."""
    from experiments.numerical_qk_reuse import v27_consumer64
    # OUT=0: the kernel never loads V and never forms output partials, so K is passed in the V slot
    # only to supply the strides the wrapper forwards. Nothing here becomes an attention output.
    v27_consumer64.fused_observe(q, k, k, sketch, scale, prefix_tiles, summary,
                                 splits=splits, mu=True, mu_precision=mu_precision, output=False)
    return summary


def tail_statistics(tail_scores, sketch, need_mu=True):
    """``(z, mu)`` of the canvas/boundary tiles from the control's own FP32 tail logits.

    ``sketch`` is [1, HK, n_tail, 32]; the tail's key extent is the tail, not the whole context.
    ``need_mu=False`` returns ``(z, None)`` for a caller that never reads the sketch."""
    rows = torch.ones(tail_scores.shape[:2], dtype=torch.bool, device=tail_scores.device)
    z, mu = vs.tile_statistics(tail_scores, None if not need_mu else sketch[0].float(), rows,
                               need_mu=need_mu)
    return z, (mu if need_mu else None)


def build_value_stats(prefix_z, prefix_mu, tail_z, tail_mu, nu_kv, group, n, heads, prefix_tiles,
                      total_tiles, need_mu=True):
    """Assemble the tile-major ``ValueStats`` of one observation call.

    ``prefix_z`` [H, QB, PT, 128] is the FA4 observation's z (its own layout), ``prefix_mu``
    [H, QB, PT, 128, 32] the Triton statistics pass's mu, ``tail_z``/``tail_mu`` [H, N, ...] the
    canvas tiles, ``nu_kv`` [HK] the valid-KV reference scale. The result is tile-major
    [H, KT, QB*128, ...] with the padded rows masked.

    ``need_mu=False`` is for the selectors that read only ``z``/``nu`` (V1, V2). The per-row
    rank-32 sketch is [H, PT, QB*128, 32] FP32 and at a 120k-token context it is tens of GiB, so
    materializing it for a selector that never reads it is what pushes the engine into OOM.
    """
    h, qb, pt, r128 = prefix_z.shape
    if r128 != 128:
        raise ValueError('FA4 observation z must carry 128 rows per query block')
    n_pad = qb * 128
    rows = torch.zeros((h, n_pad), dtype=torch.bool, device=prefix_z.device)
    if n:
        rows[:, :n] = True
    z = torch.cat([prefix_z.permute(0, 2, 1, 3).reshape(h, pt, n_pad), tail_z], dim=1)
    if need_mu:
        # Copy the permuted source straight into a preallocated destination. A permute followed by
        # reshape materializes a second full-size tensor, which is the difference between fitting
        # and OOM here; copy_ does the permutation during the copy.
        kt = z.shape[1]
        mu = torch.zeros((h, kt, n_pad, vs.RANK), dtype=torch.float32, device=z.device)
        mu[:, :pt].view(h, pt, qb, 128, vs.RANK).copy_(prefix_mu.permute(0, 2, 1, 3))
        if tail_mu is not None and tail_mu.shape[1]:
            mu[:, pt:].view(h, kt - pt, tail_mu.shape[1], vs.RANK).copy_(
                tail_mu.permute(0, 2, 1, 3))
    else:
        mu = None
    nu = nu_kv[torch.arange(h, device=prefix_z.device) // group][:, None].expand(h, n_pad).contiguous()
    return vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=n, kv_heads=int(nu_kv.shape[0]), group=int(group),
                         prefix_tiles=int(prefix_tiles), total_tiles=int(total_tiles))


def protected_tiles(prefix_tiles, total_tiles, sink, recent, heads, blocks, device):
    return vs.protect_map(prefix_tiles, total_tiles, sink, recent, heads, blocks, device)