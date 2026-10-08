"""Corrected value selectors inside Yuhan's inherited V31 reuse lifecycle.

Gaussian32 is Junyu's prior method family; these integrations are this study's
variants. Only maps are returned. Routing state never becomes model output.
The CPU routines are small trusted oracles. CUDA dispatch lives in
v31_value_kernels.py and has no host decisions per candidate.
"""
from dataclasses import dataclass
import math

import torch

SELECTORS = ('value_v1_online_discard_mass', 'value_v2_online_preserve_mass',
             'value_v3a_singleton_delete', 'value_v3b_greedy_exact',
             'value_v3b_approx_batch8')
EPS = 1e-12


@dataclass
class Stats:
    # U = query heads * query blocks, J = physical KV64 tiles, Q = 128.
    log_mass: torch.Tensor                 # U,J,Q
    mean: torch.Tensor                     # U,J,Q,R
    nu: torch.Tensor                       # U,Q
    invalid: torch.Tensor                  # U,J,Q
    prefix_tiles: int
    heads: int
    blocks: int

    def __post_init__(self):
        if self.log_mass.ndim != 3 or self.mean.ndim != 4:
            raise ValueError('statistics require U,J,Q and U,J,Q,R tensors')
        u, jt, q = self.log_mass.shape
        if (self.mean.shape[:3] != (u, jt, q) or self.nu.shape != (u, q)
                or self.invalid.shape != (u, jt, q) or self.invalid.dtype != torch.bool
                or u != self.heads*self.blocks or not 0 <= self.prefix_tiles <= jt):
            raise ValueError('invalid physical tile statistics geometry')
        if any(x.device != self.log_mass.device for x in (self.mean, self.nu, self.invalid)):
            raise ValueError('statistics must share one device')
        if self.log_mass.is_cuda and (q != 128 or self.mean.shape[-1] != 32):
            raise ValueError('CUDA selector requires Q128 and Gaussian32')

    @property
    def valid(self):
        return torch.isfinite(self.log_mass)

    @property
    def rows(self):
        return self.valid.any(1)

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.log_mass, self.mean, self.nu, self.invalid))


def statistics_reference(scores, sketch, nu, prefix_tiles, q_block=128, k_block=64):
    """Small score oracle, native GQA; -inf means a structurally illegal key.

    scores H,N,K, sketch HK,K,R, nu HK. Stable per-tile reductions only.
    NaN/+inf force dense retention of the affected physical query unit.
    """
    h, n, nk = scores.shape
    hk, kz, rank = sketch.shape
    if h % hk or nk != kz or nu.shape != (hk,):
        raise ValueError('GQA or reference geometry mismatch')
    qb, kt = math.ceil(n / q_block), math.ceil(nk / k_block)
    if not 0 <= prefix_tiles <= kt:
        raise ValueError('invalid prefix tile count')
    padded = torch.nn.functional.pad(scores, (0, kt*k_block-nk, 0, qb*q_block-n), value=-math.inf)
    tile = padded.reshape(h, qb, q_block, kt, k_block).permute(0, 1, 3, 2, 4)
    invalid = (torch.isnan(tile) | torch.isposinf(tile)).any(-1)
    legal = torch.isfinite(tile)
    clean = tile.masked_fill(~legal, -math.inf)
    lm = torch.logsumexp(clean, -1)
    safe = torch.where(torch.isfinite(lm), lm, 0.)
    weights = torch.where(legal, torch.exp(clean-safe[..., None]), 0.)
    zp = torch.nn.functional.pad(sketch, (0, 0, 0, kt*k_block-nk))
    zt = zp[torch.arange(h, device=scores.device)//(h//hk)].reshape(h, kt, k_block, rank)
    mean = torch.einsum('hbjqk,hjkr->hbjqr', weights, zt)
    ref = nu[torch.arange(h, device=scores.device)//(h//hk)]
    return Stats(lm.reshape(h*qb, kt, q_block), mean.reshape(h*qb, kt, q_block, rank),
                 ref[:, None, None].expand(h, qb, q_block).reshape(h*qb, q_block),
                 invalid.reshape(h*qb, kt, q_block), prefix_tiles, h, qb)


def mandatory_map(stats, sink=0, recent_tiles=0):
    u, kt, _ = stats.log_mass.shape
    keep = torch.zeros((u, kt), dtype=torch.bool, device=stats.log_mass.device)
    keep[:, stats.prefix_tiles:] = True
    keep[:, :min(sink, stats.prefix_tiles)] = True
    if recent_tiles:
        keep[:, max(0, stats.prefix_tiles-recent_tiles):stats.prefix_tiles] = True
    return keep


def online_reference(stats, threshold, preserve_mass=False, mandatory=None, held=None, sticky=0.):
    """Exact sequential V1/V2, no quota. Skip iff log(rho)+sticky < log(threshold).

    V2 advances log Z on skips and leaves the NORMALIZED output unchanged.
    Running log mass avoids loss of early support against a future large max.
    """
    if math.isnan(threshold) or threshold < 0:
        raise ValueError('threshold must be nonnegative')
    u, kt, q = stats.log_mass.shape
    rank = stats.mean.shape[-1]
    mandatory = mandatory_map(stats) if mandatory is None else mandatory
    held = torch.zeros_like(mandatory) if held is None else held
    keep = torch.zeros_like(mandatory)
    lm = stats.log_mass.new_full((u, q), -math.inf)
    output = stats.mean.new_zeros((u, q, rank))
    support = torch.zeros((u, q), dtype=torch.bool, device=lm.device)
    risks = lm.new_full((u, kt), -math.inf)
    bad = stats.invalid.any((1, 2)) | (~torch.isfinite(stats.mean) & stats.valid[..., None]).any((1, 2, 3))
    log_threshold = math.log(threshold) if threshold else -math.inf
    for j in range(kt):
        block = stats.log_mass[:, j]
        active = torch.isfinite(block)
        combined = torch.logaddexp(lm, block)
        safe = torch.where(torch.isfinite(combined), combined, 0.)
        eta = torch.where(active, torch.exp(block-safe), 0.)
        delta = stats.mean[:, j]-output
        rho = eta * delta.norm(dim=-1) / stats.nu.clamp_min(EPS)
        log_rho = torch.where(active, rho.log(), -math.inf).amax(-1)
        risks[:, j] = log_rho
        first = (active & ~support).any(-1)
        take = active.any(-1) & (mandatory[:, j] | first | bad |
                                 (log_rho + sticky*held[:, j] >= log_threshold))
        keep[:, j] = take | (mandatory[:, j] & bad)
        output = torch.where(take[:, None, None],
                             (1-eta[..., None])*output+eta[..., None]*stats.mean[:, j], output)
        lm = torch.where(take[:, None] | preserve_mass, combined, lm)
        support |= active & take[:, None]
    keep[bad] = True
    return keep, output, lm, risks


def full_support(stats):
    z = stats.log_mass
    normalizer = torch.logsumexp(z, 1, keepdim=True)
    alpha = torch.where(torch.isfinite(z), torch.exp(z-torch.where(torch.isfinite(normalizer), normalizer, 0.)), 0.)
    contribution = alpha[..., None] * stats.mean
    output = contribution.sum(1)
    g = contribution-alpha[..., None]*output[:, None]
    return alpha, contribution, output, g


def masked_output(alpha, contribution, keep):
    a = (alpha*keep[..., None]).sum(1)
    c = (contribution*keep[..., None, None]).sum(1)
    return torch.where(a[..., None] > 0, c/a.clamp_min(torch.finfo(a.dtype).tiny)[..., None], 0.)


def deletion_scores(stats, alpha, g, a, residual):
    remaining = a[:, None]-alpha
    live = stats.rows[:, None]
    error = (residual[:, None]-g).norm(dim=-1) / (remaining.clamp_min(torch.finfo(a.dtype).tiny)*stats.nu[:, None].clamp_min(EPS))
    error = torch.where(live, error, 0.)
    score = error.amax(-1)
    # A row's last support is inadmissible, even if its value residual is zero.
    score = score.masked_fill((live & (remaining <= 0)).any(-1), math.inf)
    return score


def singleton_reference(stats, budget, mandatory=None, held=None, sticky=0.):
    alpha, c, o, g = full_support(stats)
    mandatory = mandatory_map(stats) if mandatory is None else mandatory
    eligible = stats.valid.any(-1)
    score = deletion_scores(stats, alpha, g, alpha.sum(1), g.sum(1))
    if held is not None:
        score = score * torch.exp(sticky*held.to(score.dtype))
    # Mandatory prefix support is charged against k; tail support is outside k.
    forced = mandatory | torch.isposinf(score)
    pt = stats.prefix_tiles
    keep = mandatory.clone()
    for unit in range(keep.shape[0]):
        required = forced[unit, :pt] & eligible[unit, :pt]
        k = min(budget, int(eligible[unit, :pt].sum()))
        if int(required.sum()) > k:
            raise ValueError('mandatory row support exceeds prefix budget')
        rank = score[unit, :pt].masked_fill(~eligible[unit, :pt], -math.inf).masked_fill(required, math.inf)
        order = torch.argsort(rank, descending=True, stable=True)
        keep[unit, order[:k]] = True
    # Singleton rankings can collectively remove a row's support. Fail explicitly.
    if bool((stats.rows & ((alpha*keep[..., None]).sum(1) <= 0)).any()):
        raise ValueError('singleton ranking violates row support; map is inadmissible')
    bad = stats.invalid.any((1, 2))
    keep[bad] = True
    return keep, score


def greedy_reference(stats, budget, mandatory=None, held=None, sticky=0., batch=1):
    """Trusted backward greedy; batch>1 is explicitly an approximation.

    Sticky is an inherited log-score preference: held candidates pay exp(sticky)
    times the removal score. With sticky=0 this is precisely the stated G_j.
    """
    alpha, c, o, g = full_support(stats)
    mandatory = mandatory_map(stats) if mandatory is None else mandatory
    keep = stats.valid.any(-1) | mandatory
    pt = stats.prefix_tiles
    evaluations = 0
    for unit in range(keep.shape[0]):
        k = min(budget, int(keep[unit, :pt].sum()))
        if int((mandatory[unit, :pt] & keep[unit, :pt]).sum()) > k:
            raise ValueError('mandatory support exceeds prefix budget')
        while int(keep[unit, :pt].sum()) > k:
            a = (alpha[unit]*keep[unit, :, None]).sum(0)
            r = (g[unit]*keep[unit, :, None, None]).sum(0)
            remain = a[None]-alpha[unit]
            err = (r[None]-g[unit]).norm(dim=-1)/(remain.clamp_min(torch.finfo(a.dtype).tiny)*stats.nu[unit].clamp_min(EPS))
            score = torch.where(stats.rows[unit][None], err, 0.).amax(-1)
            score[(stats.rows[unit][None] & (remain <= 0)).any(-1)] = math.inf
            if held is not None:
                score *= torch.exp(sticky*held[unit].to(score.dtype))
            removable = keep[unit] & ~mandatory[unit]
            removable[pt:] = False
            score[~removable] = math.inf
            evaluations += int(removable.sum())
            count = min(batch, int(keep[unit, :pt].sum())-k)
            order = score.argsort(stable=True)[:count]
            if not bool(torch.isfinite(score[order]).all()):
                raise ValueError('no admissible deletion at requested budget')
            proposed = keep[unit].clone()
            proposed[order] = False
            if bool((stats.rows[unit] & ((alpha[unit]*proposed[:, None]).sum(0) <= 0)).any()):
                # Joint deletion can be inadmissible although individual deletions were legal.
                proposed = keep[unit].clone()
                proposed[order[0]] = False
            keep[unit] = proposed
    keep[stats.invalid.any((1, 2))] = True
    return keep, evaluations


def select(stats, selector, *, budget, threshold=None, mandatory=None, held=None, sticky=0.):
    if selector not in SELECTORS:
        raise ValueError(selector)
    if selector in SELECTORS[:2] and threshold is None:
        raise ValueError('V1/V2 require a frozen uniform GLOBAL threshold')
    if stats.log_mass.is_cuda:
        from .v31_value_kernels import select_cuda
        return select_cuda(stats, selector, budget, threshold, mandatory, held, sticky)
    if selector in SELECTORS[:2]:
        keep, _, _, _ = online_reference(stats, threshold, selector == SELECTORS[1], mandatory, held, sticky)
        evaluations = stats.log_mass.shape[0]*stats.log_mass.shape[1]
    elif selector == SELECTORS[2]:
        keep, _ = singleton_reference(stats, budget, mandatory, held, sticky)
        evaluations = stats.log_mass.shape[0]*stats.prefix_tiles
    else:
        keep, evaluations = greedy_reference(stats, budget, mandatory, held, sticky, 8 if selector == SELECTORS[4] else 1)
    return keep.reshape(1, stats.heads, stats.blocks, -1), {'candidate_evaluations': evaluations, 'kernel': 'cpu_oracle'}
