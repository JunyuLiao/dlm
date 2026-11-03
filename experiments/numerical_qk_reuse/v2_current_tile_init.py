"""V2 variant: initialize with the mandatory last KV tile, then scan 0..J-2.

Junyu's Gaussian32 preserve-mass recurrence and Yuhan's inherited map reuse
remain separate identities. This module changes only the recurrence order.
"""
import math

import torch


ARM = "value_v2_current_tile_init_gaussian32_reuse_r4"


def current_tile_reference(stats, threshold, mandatory):
    """Independent CPU oracle, including normalized-output-on-skip semantics."""
    if math.isnan(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if not bool(mandatory[:, -1].all()) or bool(mandatory[:, :-1].any()):
        raise ValueError("variant requires exactly one mandatory tile: the last")
    u, jt, q = stats.log_mass.shape
    keep = torch.zeros_like(mandatory)
    lm = torch.full_like(stats.log_mass[:, 0], -math.inf)
    state = torch.zeros_like(stats.mean[:, 0])
    support = torch.zeros((u, q), dtype=torch.bool, device=lm.device)
    risks = torch.full((u, jt), -math.inf, dtype=lm.dtype, device=lm.device)
    bad = stats.invalid.any((1, 2)) | (~torch.isfinite(stats.mean) & stats.valid[..., None]).any((1, 2, 3))
    for j in [jt - 1, *range(jt - 1)]:
        block = stats.log_mass[:, j]
        active = torch.isfinite(block)
        combined = torch.logaddexp(lm, block)
        eta = torch.where(active, torch.exp(block - torch.where(torch.isfinite(combined), combined, 0.)), 0.)
        rho = eta * (stats.mean[:, j] - state).norm(dim=-1) / stats.nu.clamp_min(1e-12)
        risk = torch.where(active, rho, 0.).amax(-1)
        take = active.any(-1) & (mandatory[:, j] | (active & ~support).any(-1) | bad | (risk >= threshold))
        keep[:, j] = take
        risks[:, j] = torch.where(active.any(-1), risk.log(), -math.inf)
        state = torch.where(take[:, None, None],
                            (1 - eta[..., None]) * state + eta[..., None] * stats.mean[:, j], state)
        lm = combined  # V2 preserves mass even when rejecting the tile.
        support |= active & take[:, None]
    keep[bad] = True
    return keep, state, lm, risks


def current_tile_cuda(stats, threshold, preserve, mandatory, held, sticky, row_weight=None):
    """Seed from the last tile; run the existing optimized kernel over its prefix.

    The mandatory tile is consumed once. No tensor permutation, full statistics
    copy, extra QK discovery, model forward, or host decision per tile is needed.
    """
    from .v31_value_kernels import _online_chunk

    if not preserve or sticky != 0. or row_weight is not None:
        raise ValueError("this frozen variant supports plain preserve-mass V2 only")
    if math.isnan(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative")
    torch._assert_async(mandatory[:, -1].all() & ~mandatory[:, :-1].any(),
                        "current initialization requires only the last tile mandatory")
    u, jt, q = stats.log_mass.shape
    if q != 128:
        raise ValueError("CUDA recurrence requires Q128 padding")
    keep = torch.zeros((u, jt), dtype=torch.bool, device=stats.log_mass.device)
    keep[:, -1] = True
    lm = stats.log_mass[:, -1].contiguous().clone()
    state = stats.mean[:, -1].contiguous().clone()
    support = torch.isfinite(lm)
    state.masked_fill_(~support[..., None], 0.)
    risk = torch.full((u, jt), -math.inf, dtype=torch.float32, device=lm.device)
    # Record the actual initialization risk, consistent with the CPU oracle.
    rho = torch.where(support, state.norm(dim=-1) / stats.nu.clamp_min(1e-12), 0.)
    risk[:, -1] = rho.amax(-1).log()
    weight = torch.ones_like(stats.nu)
    for j0 in range(0, jt - 1, 128):
        _online_chunk[(u,)](stats.log_mass, stats.mean, stats.nu, stats.invalid,
            mandatory, held, weight, keep, lm, state, support, risk, jt, j0,
            min(128, jt - 1 - j0), math.log(threshold) if threshold else -math.inf,
            True, 0., num_warps=8)
    keep |= stats.invalid.any((1, 2))[:, None]
    return keep, state, lm, risk
