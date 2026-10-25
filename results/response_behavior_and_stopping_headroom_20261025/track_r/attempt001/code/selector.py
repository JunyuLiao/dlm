"""Current-QKV diagnostic selectors. No model calls or native RNG consumption.

The scores are local attention surrogates, not proven full-model contraction or
exact downstream margin influence. Previous queries and scalar logit margins
are allowed state; old K/V and attention outputs are never retained.
"""
import hashlib
import math

import torch


def removal(q, k, v, scale, indices, direction):
    """Exact current output bias and Q-direction differential of one tile cut."""
    p = (q @ k.T * scale).softmax(-1)
    o = p @ v
    ds = direction @ k.T * scale
    dp = p * (ds - (p * ds).sum(-1, keepdim=True))
    ja = dp @ v
    result = []
    for tile in indices:
        tiles = [tile] if isinstance(tile, int) else list(tile)
        positions = torch.tensor(tiles, device=q.device)[:, None]*64 + torch.arange(64, device=q.device)[None]
        positions = positions.flatten()
        dropped_p = p.index_select(1, positions)
        dropped_dp = dp.index_select(1, positions)
        dropped_v = v.index_select(0, positions)
        m = dropped_p.sum(-1)
        c = dropped_p @ dropped_v
        dm = dropped_dp.sum(-1)
        dc = dropped_dp @ dropped_v
        retained = (1 - m).clamp_min(1e-12)
        numerator = m[:, None] * o - c
        bias = numerator / retained[:, None]
        dbias = (dm[:, None] * o + m[:, None] * ja - dc) / retained[:, None]
        dbias += numerator * dm[:, None] / retained[:, None].square()
        result.append(dict(tile=tile, tiles=tiles, mass=m, bias=bias, J_A=ja,
                           J_S=ja + dbias, residual=dbias))
    return p, o, result


def candidate_tiles(mass, blocks, limit=12, unit=0):
    """Enumerate small banks; otherwise frozen mass/position/hash proposals."""
    if blocks <= limit:
        return list(range(blocks))
    ordered = torch.argsort(mass, stable=True).tolist()
    choices = ordered[:8] + [0, blocks - 1]
    hashed = sorted(range(blocks), key=lambda j: hashlib.sha256(f'7321|{unit}|{j}'.encode()).digest())
    for j in hashed + ordered:
        if j not in choices:
            choices.append(j)
        if len(set(choices)) >= limit:
            break
    return list(dict.fromkeys(choices))[:limit]


def candidate_supports(mass, blocks, drop_count, unit, limit=12):
    if drop_count == 1:
        return candidate_tiles(mass, blocks, limit, unit)
    order = torch.argsort(mass, stable=True).tolist()
    base = tuple(sorted(order[:drop_count]))
    supports = [base]
    # Equal-budget near/far, positional, and constrained hash swaps around the
    # low-mass deletion set. Proposal order is independent of native outcomes.
    boundary = order[drop_count:]
    alternatives = boundary[:4] + boundary[-2:]
    alternatives += sorted(boundary, key=lambda j: hashlib.sha256(f'7321|{unit}|{j}'.encode()).digest())
    for index, add in enumerate(alternatives):
        remove = order[(drop_count-1-index) % drop_count]
        candidate = tuple(sorted((set(base)-{remove}) | {add}))
        if candidate not in supports:
            supports.append(candidate)
        if len(supports) >= limit:
            break
    return supports


def mass_lower_bound(anchor_q, q, max_prefix_key_norm, prefix_logZ,
                     kept_prefix_logZ, current_tail_logZ, scale):
    """Conservative retained-mass bound under fixed PREFIX keys only.

    Whole current boundary/canvas log-partition is recomputed exactly, so
    changing canvas K is accounted for. Prefix K/V are native immutable cached
    context within a canvas. The certificate resets when the canvas changes.
    No old key/value tensor is kept for this bound.
    """
    drift = (q - anchor_q).norm(dim=-1) * max_prefix_key_norm * scale
    numerator = torch.logaddexp(kept_prefix_logZ - drift, current_tail_logZ)
    denominator = torch.logaddexp(prefix_logZ + drift, current_tail_logZ)
    return (numerator - denominator).exp().clamp(max=1)


def choose_unit(q, k, v, scale, blocks, previous_q, margin, stable, mode,
                unit, guards=(.2, .1), require_alternatives=1):
    """One rectangle deleted per feasible unit; scores share exactly one bank."""
    q, k, v = q.float(), k.float(), v.float()
    direction = q - previous_q.float() if previous_q is not None else torch.zeros_like(q)
    # A fixed coordinate probe when there is no observed motion. No random draw.
    if previous_q is None:
        direction[:, unit % q.shape[-1]] = q.norm(dim=-1) / math.sqrt(q.shape[-1])
    preliminary = (q @ k.T * scale).softmax(-1)
    tile_mass = preliminary[:, :blocks * 64].reshape(len(q), blocks, 64).sum(-1).amax(0)
    drop_count = max(1, blocks - math.ceil(.9*blocks))
    indices = candidate_supports(tile_mass, blocks, drop_count, unit)
    del preliminary
    p, o, candidates = removal(q, k, v, scale, indices, direction)
    norm = o.norm(dim=-1).clamp_min(1e-8)
    # Previous native margins prioritize vulnerable stable decisions. This is
    # attention displacement weighting, not an exact downstream influence oracle.
    if margin is None:
        weights = torch.ones(len(q), device=q.device)
        protected_empty = True
    else:
        protected_empty = not bool(stable.any())
        protected = torch.ones_like(stable) if protected_empty else stable
        weights = protected.float() / margin.float().clamp_min(.1)
    records = []
    for c in candidates:
        bias_norm = c['bias'].norm(dim=-1)
        error = float((bias_norm / norm).max())
        deleted_mass = float(c['mass'].max())
        records.append(dict(tile=c['tile'], tiles=c['tiles'], current_error=error, deleted_mass=deleted_mass,
            response=float((c['J_S'].norm(dim=-1) / norm).max()),
            residual_response=float((c['residual'].norm(dim=-1) / norm).max()),
            decision=float((weights * bias_norm / norm).max()),
            feasible=error <= guards[1] and deleted_mass <= guards[0]))
    feasible = [c for c in records if c['feasible']]
    chosen = None
    if len(feasible) >= require_alternatives:
        field = dict(current_error_fixed4='current_error', response_guard_qdirection='response',
                     margin_guard_weighted_bias='decision', validity_reuse_mass_certificate='current_error',
                     mass_fixed4='deleted_mass').get(mode)
        if mode == 'sham_fixed4':
            chosen = min(feasible, key=lambda c: hashlib.sha256(f'7321|{unit}|{c["tile"]}'.encode()).digest())
        else:
            chosen = min(feasible, key=lambda c: (c[field], tuple(c['tiles'])))
    keep = torch.ones(blocks, device=q.device, dtype=torch.bool)
    if chosen is not None:
        keep[chosen['tiles']] = False
    logits = q @ k.T * scale
    prefix_logits = logits[:, :blocks * 64]
    selected_logits = prefix_logits.masked_fill(~keep.repeat_interleave(64)[None], -torch.inf)
    certificate = dict(anchor_q=q.detach().clone(), prefix_logZ=prefix_logits.logsumexp(-1),
        kept_prefix_logZ=selected_logits.logsumexp(-1),
        max_prefix_key_norm=k[:blocks * 64].norm(dim=-1).max(), allkept=chosen is None)
    return keep, dict(candidates=records, feasible=len(feasible), chosen=chosen, deletion_budget=drop_count,
        ranking_identifying=len(feasible) >= 4, protected_set_empty=protected_empty), certificate
