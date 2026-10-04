"""v31 pooled residual for the dropped prefix tiles (named variant, training-free, opt-in RESIDUAL=centroid).

A sparse GLOBAL call attends exactly to its kept 64-key tiles and drops the rest. The residual adds back an
approximation of every DROPPED wholly-prefix tile: its 64 keys are replaced by their centroid (mean key) and their
values by the mean value, so the tile contributes like 64 identical keys,

    z_t = scale * q . kbar_t + log 64,        o_drop = sum_t softmax(z)_t vbar_t,    lse_drop = logsumexp_t z_t,

and (o_drop, lse_drop) is merged with the exact sparse partials by the same log-sum-exp merge as the FA4 splits.
It is exact when the keys of a tile are equal; by Jensen it under-estimates a tile's mass otherwise (a conservative
correction), and it is accurate where attention is diffuse within a tile -- the aggregation regime (RULER cwe) that
top-k selection loses. Concentrated tiles are kept exactly by the selector. Cost per call: one [n, D] x [D, PT]
product and one [n, PT] x [PT, D] product per KV head (PT = prefix tiles, 1/64 of the prefix keys); centroids are
computed once per canvas and layer. Arm-agnostic: applies to the MAGE port and the method alike.
"""
import math

import torch

RESIDUAL_MODES = ('centroid',)


def tile_means(k, v, pt):
    """k, v [1, HK, nk, D] -> centroid keys and mean values of the wholly-prefix 64-key tiles, each [HK, PT, D] fp32."""
    hk, d = k.shape[1], k.shape[3]
    kb = k[0, :, :pt * 64].float().reshape(hk, pt, 64, d).mean(2)
    vb = v[0, :, :pt * 64].float().reshape(hk, pt, 64, d).mean(2)
    return kb, vb


def kept_from_lists(lists):
    """FA4 block-sparse lists (full blocks only; the first cnt entries of each permutation are kept) -> [H, QB, KT]."""
    idx, cnt = lists.full_block_idx, lists.full_block_cnt                     # [1, H, QB, KT], [1, H, QB]
    ar = torch.arange(idx.shape[-1], device=idx.device)
    kept = torch.zeros(idx.shape, dtype=torch.bool, device=idx.device)
    kept.scatter_(-1, idx.long(), ar < cnt[..., None])
    return kept[0]


def residual_partial(q, kb, vb, kept, pt, scale, q_block=128):
    """q [1, H, n, D]; kb, vb [HK, PT, D]; kept [H, QB, KT] -> the dropped tiles' partial: o [1, n, H, D] fp32 and
    lse [1, H, n] fp32 (-inf, weight 0 in the merge, for rows whose block dropped no prefix tile)."""
    _, h, n, d = q.shape
    hk = kb.shape[0]
    g = h // hk
    z = torch.bmm(q[0].float().reshape(hk, g * n, d), kb.transpose(1, 2)) * scale + math.log(64.0)
    z = z.view(h, n, pt)
    rows = torch.arange(n, device=q.device) // q_block
    z = z.masked_fill(kept[:, rows, :pt], float('-inf'))                     # keep only the dropped tiles
    lse = torch.logsumexp(z, -1)                                               # [H, n]
    p = torch.exp(z - lse[..., None]).nan_to_num(0.0)                          # all-kept rows: nan -> 0
    o = torch.bmm(p.view(hk, g * n, pt), vb).view(h, n, d)
    return o.transpose(0, 1)[None], lse[None]


def merge(o_parts, lse_parts, dtype):
    """Exact log-sum-exp merge: o_parts [P, n, H, D], lse_parts [P, H, n] -> [1, n, H, D] in dtype."""
    w = torch.softmax(lse_parts.float(), dim=0).permute(0, 2, 1)[..., None]
    return (o_parts.float() * w).sum(0, keepdim=True).to(dtype)
