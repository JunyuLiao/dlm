"""Fixed-shape execution of the existing full-value bounded-swap formula.

The independent reference remains in v31_value_decision.py. This backend takes
the known per-tile budget rather than synchronizing to discover it. Production
use requires source-bound same-state and native qualification.
"""
import torch

from .v31_value_decision import disjoint_pair_indices, mass_score, objective, squared_rows


def dense_output(m):
    """The all-true gather is redundant; preserve the reference reduction axis."""
    numerator = m.tail_numerator + m.numerator.sum(0)
    mass = m.tail_mass + m.mass.sum(0)
    return numerator / mass.clamp_min(torch.finfo(numerator.dtype).tiny)[:, None]


def selected_ids(anchor, budget):
    return torch.argsort((~anchor).to(torch.int8), stable=True)[:budget]


def selected_output(m, ids):
    num = m.tail_numerator + m.numerator.index_select(0, ids).sum(0)
    den = m.tail_mass + m.mass.index_select(0, ids).sum(0)
    return num, den


def rank_disjoint(pairs, losses, reference_loss, limit):
    if not losses.is_cuda:
        return disjoint_pair_indices(pairs, losses, reference_loss, limit)
    import triton
    from .v31_value_fast_kernel import disjoint_kernel
    indices = torch.empty(limit, device=losses.device, dtype=torch.int64)
    valid = torch.empty(limit, device=losses.device, dtype=torch.bool)
    disjoint_kernel[(1,)](pairs, losses, reference_loss, indices, valid,
                         N=len(pairs), LIMIT=limit, BLOCK=triton.next_power_of_2(len(pairs)))
    return indices, valid


@torch.inference_mode()
def bounded_proposal_device(m, anchor, budget, *, swaps=8, extra=32, shortlist=16,
                            gram=None, backend='exact', mass_guard=.025,
                            max_error_ratio=1.02):
    """Original singleton ranking and combined backoff, without host decisions.

    Exact uses the reference pair GEMMs. Factored moves the common Gram
    transform before pair formation; it is mathematically equivalent but has
    different FP32 rounding and is an explicitly separate candidate backend.
    Both recompute every combined proposal with the original direct objective.
    Returned swap counts remain device tensors until the audit receipt.
    """
    if backend not in ('exact', 'factored'):
        raise ValueError('unknown pair-loss backend')
    pt, qrows = m.mass.shape
    if not 0 <= budget <= pt:
        raise ValueError('invalid fixed budget')
    if budget in (0, pt) or swaps == 0:
        return anchor.clone(), {'swaps': anchor.new_zeros((), dtype=torch.int64), 'pairs': 0}
    ids = selected_ids(anchor, budget)
    score = mass_score(m)
    pool = torch.zeros_like(anchor)
    pool[torch.argsort(score, descending=True, stable=True)[:min(pt, budget+extra)]] = True
    nadd = min(shortlist, pt-budget)
    nremove = min(shortlist, budget)
    if nadd == 0:
        return anchor.clone(), {'swaps': anchor.new_zeros((), dtype=torch.int64), 'pairs': 0}
    remove = ids[torch.argsort(score[ids], stable=True)[:nremove]]
    priority = torch.where(pool & ~anchor, score, -torch.inf)
    outside = torch.argsort(priority, descending=True, stable=True)[:nadd]
    outside_valid = pool[outside] & ~anchor[outside]
    pairs = torch.stack((outside.repeat_interleave(nremove), remove.repeat(nadd)), -1)
    num, den = selected_output(m, ids)
    base = num / den[:, None]
    target = dense_output(m)
    weight = torch.ones_like(den)
    reference_loss = objective(base, target, weight, gram)
    reference_max = squared_rows(base-target, gram).max()
    losses = []
    if backend == 'exact':
        for batch in pairs.split(16):
            add, rem = batch[:, 0], batch[:, 1]
            pd = den + m.mass[add] - m.mass[rem]
            pn = num + m.numerator[add] - m.numerator[rem]
            po = pn / pd[:, :, None].clamp_min(1e-30)
            losses.append(objective(po, target, weight, gram))
        losses = torch.cat(losses)
    else:
        candidates = torch.cat((outside, remove))
        vectors = torch.cat((num[None], target[None], m.numerator[candidates]), 0)
        transformed = vectors if gram is None else vectors @ gram
        # Keep pair formation bounded in memory. The common Gram multiply is
        # performed on 34 vectors, rather than independently on all256 pairs.
        local_add = torch.arange(nadd, device=anchor.device).repeat_interleave(nremove)+2
        local_rem = torch.arange(nremove, device=anchor.device).repeat(nadd)+2+nadd
        for start in range(0, len(pairs), 16):
            add, rem = pairs[start:start+16].unbind(-1)
            den_new = (den + m.mass[add] - m.mass[rem]).clamp_min(1e-30)
            error = (num + m.numerator[add] - m.numerator[rem])/den_new[:, :, None]-target
            error_dual = (transformed[0]+transformed[local_add[start:start+16]]
                          -transformed[local_rem[start:start+16]])/den_new[:, :, None]-transformed[1]
            losses.append((error*error_dual).sum(-1).clamp_min(0).mean(-1))
        losses = torch.cat(losses)
    losses = torch.where(outside_valid.repeat_interleave(nremove), losses, torch.inf)
    limit = min(swaps, len(pairs))
    ranks, valid = rank_disjoint(pairs, losses, reference_loss, limit)
    chosen = pairs[ranks]
    add = torch.zeros((limit, pt), device=anchor.device, dtype=torch.bool)
    rem = torch.zeros_like(add)
    add.scatter_(1, chosen[:, :1], valid[:, None])
    rem.scatter_(1, chosen[:, 1:], valid[:, None])
    candidates = (anchor[None] | (add.cumsum(0)>0)) & ~(rem.cumsum(0)>0)
    candidate_ids = torch.argsort((~candidates).to(torch.int8), dim=-1, stable=True)[:, :budget]
    candidate_num = m.numerator.index_select(0, candidate_ids.flatten()).reshape(
        limit, budget, qrows, -1).sum(1)+m.tail_numerator
    candidate_den = m.mass.index_select(0, candidate_ids.flatten()).reshape(
        limit, budget, qrows).sum(1)+m.tail_mass
    output = candidate_num/candidate_den[:, :, None].clamp_min(torch.finfo(candidate_num.dtype).tiny)
    loss = objective(output, target, weight, gram)
    good = ((loss < reference_loss-1e-8) & (loss <= reference_loss+1e-6)
            & (squared_rows(output-target, gram).amax(-1) <= reference_max*max_error_ratio**2+1e-6)
            & ((den-candidate_den).amax(-1) <= mass_guard+1e-6))
    best = torch.where(good, torch.arange(limit, device=anchor.device), -1).max()
    best_candidate = candidates.index_select(0, best.clamp_min(0).reshape(1))[0]
    selected = torch.where(best >= 0, best_candidate, anchor)
    return selected, {'swaps': (anchor & ~selected).sum(), 'pairs': len(pairs),
                      'pair_losses': losses, 'reference_loss': reference_loss,
                      'ranks': ranks, 'valid': valid, 'admitted_counts': good}
