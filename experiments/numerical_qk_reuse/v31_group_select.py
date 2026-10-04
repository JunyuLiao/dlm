"""v31 group-shared target-sparsity selection for the method (named variant RISK_GROUP=kv on M1-DP top-k).

The method's fixed-fraction selector (``v27_dense_prefix.topk_skip``, e.g. k12 mass) ranks the prefix tiles of every
(batch, query head, 128-row block) by the block's worst-row value and drops the lowest (1 - keep) fraction. The keep
lists then differ between the 8 query heads of a KV head, which costs the FA4 block-sparse consumer the K/V tile reuse
that identical lists get (v31 profile: a held call costs 2.1x MAGE's for about 1.5x the kept tiles).

``grouped_topk_skip`` makes one decision per (batch, KV head, block) instead: a tile's group value is the max of its
worst-row value over the group's query heads, a tile is droppable only if it is droppable for every head of the group
(eligible, finite value), and the same (1 - keep) fraction of the prefix tiles is dropped. Every head of the group gets
that decision, so each head keeps the same number of tiles as before (same work) and the lists are identical inside a
group. Canvas / boundary (tail) tiles keep the core's own per-head decision.
"""
import math

import torch


def grouped_topk_skip(lognorm, eligible, sensitivity, reference, nq, keep):
    """Drop decision [B, H, QB, PT] (bool), identical for the query heads of each KV head of ``reference`` [B, HK]."""
    from experiments.numerical_qk_reuse.v27_dense_prefix import _worst_risk
    worst = _worst_risk(lognorm, sensitivity, reference, nq)                        # [B, H, QB, PT]
    candidate = (eligible != 0) & (worst < float('inf')) & ~torch.isnan(worst)
    b, h, qb, pt = worst.shape
    hk = reference.shape[1]
    if h % hk:
        raise ValueError('query heads must be a multiple of the KV heads')
    g = h // hk
    n_drop = min(pt, int(math.floor((1.0 - float(keep)) * pt + 1e-9)))
    if n_drop <= 0:
        return torch.zeros_like(candidate)
    cand_g = candidate.view(b, hk, g, qb, pt).all(2)                                # droppable for every head
    key = torch.where(cand_g, worst.view(b, hk, g, qb, pt).amax(2), torch.full_like(worst[:, :hk], float('inf')))
    order = torch.argsort(key, dim=-1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(pt, device=worst.device).expand_as(order))
    return (cand_g & (rank < n_drop)).repeat_interleave(g, dim=1)
