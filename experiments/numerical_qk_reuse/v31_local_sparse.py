"""LOCAL FA4 routing on native paged K/V, with an unchanged sliding-window mask.

This is a named extension of Yuhan's GLOBAL reuse path. Selection observes exact
LOCAL attention, ranks the worst-row prefix mass per Q128/KV64 unit, and reuses
the map between observations. The current canvas and its mixed boundary tile
remain dense. There are no extra model forwards or changes to the sampler.
"""
from __future__ import annotations

import torch

WINDOW = (1023, 1023)


def geometry(prefix, n, window=WINDOW, qblock=128, ktile=64):
    """CPU geometry: eligible physical tiles, mandatory canvas tiles, per-block counts."""
    nk = prefix + n
    kt, qb = (nk + ktile - 1) // ktile, (n + qblock - 1) // qblock
    eligible, mandatory = [], []
    for b in range(qb):
        qlo, qhi = prefix + b * qblock, prefix + min(n, (b + 1) * qblock) - 1
        lo, hi = max(0, qlo - window[0]), min(nk - 1, qhi + window[1])
        tiles = list(range(lo // ktile, hi // ktile + 1))
        eligible.append(tiles)
        mandatory.append([t for t in tiles if t >= prefix // ktile])
    return eligible, mandatory, kt


def alias64(key, value, table, nk):
    """Zero-copy KV64 view plus logical table expansion of native KV128 pages.

    K/V remain interleaved with their actual strides. Cache rows must be adjacent
    across the native page boundary; refuse other layouts rather than copying.
    """
    page = key.shape[1]
    if page % 64 or key.shape != value.shape:
        raise ValueError('LOCAL sparse FA4 needs native pages divisible by 64')
    factor = page // 64
    if factor == 1:
        return key, value, table[:(nk + 63) // 64][None]
    for x in (key, value):
        if x.stride(0) != page * x.stride(1):
            raise ValueError('native cache cannot be viewed as contiguous KV64 pages')
    shape = (key.shape[0] * factor, 64, key.shape[2], key.shape[3])
    def view(x):
        return x.as_strided(shape, (64 * x.stride(1), *x.stride()[1:]))
    subtile = torch.arange(factor, device=table.device, dtype=table.dtype)
    expanded = (table[:(nk + page - 1) // page, None] * factor + subtile[None]).flatten()
    return view(key), view(value), expanded[:(nk + 63) // 64][None]


def select(z, prefix, n, budget, sticky=0.0, held=None, weights=None):
    """Window-masked log masses [H,QB,KT,128] -> eligible keep map and CPU counts.

    The budget counts wholly-prefix tokens; mandatory canvas/boundary tiles are
    additional. All-masked rows contribute -inf, never NaN or a top-k vote.
    """
    eligible, mandatory, kt = geometry(prefix, n)
    H, qb = z.shape[:2]
    if z.shape[2:] != (kt, 128) or qb != len(eligible) or budget < 64 or budget % 64:
        raise ValueError('LOCAL selection geometry/budget mismatch')
    share = z - z.logsumexp(2, keepdim=True)
    share = torch.where(torch.isfinite(z), share, float('-inf'))
    if weights is not None:
        w = torch.nn.functional.pad(weights[:n].float(), (0, qb * 128 - n))
        logw = w.log().view(qb, 128)
        weighted = (share + logw[None, :, None, :]).amax(-1)
        score = torch.where(torch.isfinite(logw).any(-1)[None, :, None], weighted, share.amax(-1))
    else:
        score = share.amax(-1)
    if held is not None and held.shape == (1, H, qb, kt):
        score = score + sticky * held[0].to(score.dtype)
    kept = torch.zeros((1, H, qb, kt), device=z.device, dtype=torch.bool)
    counts = []
    for b, tiles in enumerate(eligible):
        pre = [t for t in tiles if t < prefix // 64]
        k = min(budget // 64, len(pre))
        if k:
            positions = torch.tensor(pre, device=z.device)
            chosen = positions[score[:, b, positions].topk(k, dim=-1).indices]
            kept[0, :, b].scatter_(1, chosen, True)
        kept[0, :, b, mandatory[b]] = True
        counts.append(k + len(mandatory[b]))
    return kept, H * sum(counts), H * sum(map(len, eligible))


class LocalSparse:
    def __init__(self, budget=512):
        if budget < 64 or budget % 64:
            raise ValueError('local budget must be a positive multiple of 64')
        self.budget = budget
        self.clear()

    def clear(self):
        self.state = {}
        self.calls = self.observations = self.reuses = self.dense_calls = 0
        self.eligible = self.kept = 0
        self.sparse_eligible = self.sparse_kept = 0

    def account_dense(self, prefix, n, h):
        e, _, _ = geometry(prefix, n)
        count = h * sum(map(len, e))
        self.calls += 1
        self.dense_calls += 1
        self.eligible += count
        self.kept += count

    def forward(self, layer, query, key, value, table, prefix, n, scale, adapter):
        """None requests native dense warm-up; selection returns its exact output."""
        from experiments.numerical_qk_reuse import v27_fa4
        from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
        nk, H = prefix + n, query.shape[1]
        step = adapter._canvas_step - 1
        tag = (adapter.canvas_id, prefix, n)
        st = self.state.get(layer)
        if st is None or st['tag'] != tag:
            if step < adapter.mage_select_step:
                self.account_dense(prefix, n, H)
                return None
            key64, value64, table64 = alias64(key, value, table, nk)
            used = torch.tensor([nk], device=query.device, dtype=torch.int32)
            st = dict(tag=tag, key=key64, value=value64, table=table64, used=used, lists=None, kept=None)
            self.state[layer] = st
        due = st['lists'] is None or adapter._trig_at == (adapter.canvas_id, step)
        if adapter.mage_reselect is not None:
            due = due or step in adapter.mage_reselect
        if due:
            _, _, kt = geometry(prefix, n)
            z = torch.full((H, (n + 127) // 128, kt, 128), float('-inf'),
                           device=query.device, dtype=torch.float32)
            out = observe_dense(query.transpose(1, 2), st['key'], st['value'], scale, z,
                                page_table=st['table'], seqused_k=st['used'], num_splits=1,
                                window_size=WINDOW)
            kept, count, total = select(z, prefix, n, self.budget,
                                       sticky=adapter.mage_sticky or 0.0, held=st['kept'],
                                       weights=adapter._mage_row_w if adapter.mage_row_weight else None)
            st.update(lists=v27_fa4.block_sparse_tensors(kept), kept=kept, count=count, total=total)
            self.calls += 1
            self.observations += 1
            self.eligible += total
            self.kept += total
            return out
        self.calls += 1
        self.reuses += 1
        self.eligible += st['total']
        self.kept += st['count']
        self.sparse_eligible += st['total']
        self.sparse_kept += st['count']
        return v27_fa4.load()(query.transpose(1, 2), st['key'], st['value'], softmax_scale=scale,
                             causal=False, window_size_left=WINDOW[0], window_size_right=WINDOW[1],
                             page_table=st['table'], seqused_k=st['used'], block_sparse_tensors=st['lists'],
                             tile_mn=(128, 64), num_splits=1, pack_gqa=False)[0]

    def receipt(self):
        return dict(local_calls=self.calls, local_observations=self.observations,
                    local_reused_calls=self.reuses, local_native_dense_calls=self.dense_calls,
                    local_eligible_tiles=self.eligible, local_kept_tiles=self.kept,
                    local_sparsity=1 - self.kept / self.eligible if self.eligible else 0.0,
                    local_sparse_eligible_tiles=self.sparse_eligible, local_sparse_kept_tiles=self.sparse_kept,
                    local_sparse_only_sparsity=(1 - self.sparse_kept / self.sparse_eligible
                                                if self.sparse_eligible else 0.0),
                    local_budget=self.budget, local_window=WINDOW, local_carry_first=False)
