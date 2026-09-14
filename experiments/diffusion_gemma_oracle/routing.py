"""Physical 64x64 top-k/top-p masks and exact same-state output diagnostics."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dllm.attention.blasst.core import (
    _prepare_attention_scores, _finish_eager_attention, _attention_type,
    Blasst2DConfig, apply_blasst_2d,
)
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _policy_for_target


def select(importance, eligible, mode, amount):
    """Stable ties by KV index. top-k drops floor(s*n), preserving one tile.

    top-p retains the smallest ranked prefix covering p of nonnegative signal.
    It is an adaptive budget, not a claim of fixed physical sparsity.
    """
    n = eligible.sum(-1)
    order = importance.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order).scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    if mode == 'topk':
        count = (n - torch.floor(n * amount).long()).clamp_min(1)
    elif mode == 'topp':
        weights = importance.clamp_min(0).masked_fill(~eligible, 0).gather(-1, order)
        total = weights.sum(-1, keepdim=True)
        count = ((weights.cumsum(-1) - weights) < amount * total).sum(-1).clamp_min(1)
        count = torch.where(total.squeeze(-1) > 0, count, n)
    else:
        raise ValueError(mode)
    return eligible & (ranks < count[..., None])


def signals(scores, valid, values, width=64):
    """One logical Q block; B,H,Q,K inputs. Contribution uses vector P_tile V."""
    has = valid.any(-1)
    probs = torch.softmax(torch.where(has[..., None], scores.float(), 0), -1) * valid
    k = scores.shape[-1]; pad = (-k) % width; nt = (k + pad) // width
    pb = F.pad(probs, (0, pad)).reshape(*probs.shape[:-1], nt, width)
    vb = F.pad(valid, (0, pad)).reshape(*valid.shape[:-1], nt, width)
    sb = F.pad(scores.float(), (0, pad), value=-torch.inf).reshape(*scores.shape[:-1], nt, width)
    vv = F.pad(values.float(), (0, 0, 0, pad)).reshape(*values.shape[:2], nt, width, values.shape[-1])
    contrib = torch.einsum('bhqtk,bhtkd->bhqtd', pb, vv)
    row_mass = pb.sum(-1)
    row_max = sb.masked_fill(~vb, -torch.inf).amax(-1)
    running_before = F.pad(row_max.cummax(-1).values[..., :-1], (1, 0), value=-torch.inf)
    margin = row_max - running_before
    # Larger max margin means at least one valid row protects this whole tile.
    # First valid tile has +inf margin and must be retained at every lambda.
    margin = torch.where(vb.any(-1), margin, -torch.inf)
    eligible = vb.any((-1, -3))
    importance = dict(qk=row_max.amax(-2), mass=row_mass.sum(-2),
        contribution=contrib.square().sum((-1, -3)).sqrt(), blasst=margin.amax(-2))
    return importance, eligible, row_mass, contrib, has, vb


def masked_diagnostics(keep, row_mass, contrib, has):
    mass = (row_mass * keep[..., None, :]).sum(-1)
    dense = contrib.sum(-2)
    output = (contrib * keep[..., None, :, None]).sum(-2) / mass.clamp_min(1e-30)[..., None]
    delta = (output - dense) * has[..., None]
    return dict(mass_sum=(mass * has).sum(-1), rows=has.sum(-1),
        error_sq=delta.square().sum((-1, -2)), dense_sq=dense.square().sum((-1, -2)))


class OracleAttention:
    def __init__(self, signal='mass', mode='topk', amount=0., observer=False):
        self.signal, self.mode, self.amount, self.observer = signal, mode, amount, observer
        self.buckets = {}; self.rankings = []; self.calls = 0; self.observed = set()

    @property
    def records(self):
        return list(self.buckets.values())

    @torch.no_grad()
    def __call__(self, module, query, key, value, attention_mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        _, values, scores, valid = _prepare_attention_scores(module, query, key, value, attention_mask,
            scaling=scaling, is_causal=is_causal, sliding_window=sliding_window)
        kind = _attention_type(module, sliding_window); runtime = module._blasst_2d_runtime
        step = runtime.current_denoising_iteration
        observation = (int(module.layer_idx), int(step))
        if self.observer and (step not in (0, 1, 4, 12, 24, 47) or observation in self.observed):
            self.calls += 1
            return _finish_eager_attention(query, values, scores, valid, dropout, module.training)
        self.observed.add(observation)
        masked = scores.clone() if not self.observer else scores
        length = int(valid.any(dim=(1, 2)).sum(-1).item())
        blasst_masks = {}
        if self.observer:
            for target in (.25, .5, .75, .9):
                config = Blasst2DConfig(q_tile_size=64,kv_tile_size=64,length_aware_policy=_policy_for_target(target))
                lam = config.lambda_for(kind, step, length)
                _, decision = apply_blasst_2d(scores, valid, runtime.active_query_mask, config, blasst_lambda=lam)
                blasst_masks[target] = (~decision.skip_mask, lam)
        for qi, start in enumerate(range(0, query.shape[-2], 64)):
            ss, vv = scores[..., start:start+64, :], valid[..., start:start+64, :]
            imp, eligible, mass, contrib, has, vb = signals(ss, vv, values)
            keep = select(imp[self.signal], eligible, self.mode, self.amount)
            # A structural row may have support disjoint from the other rows.
            # Restore its best mass tile only if the proposed mask empties it.
            empty = has & ~(vb.any(-1) & keep[..., None, :]).any(-1)
            rescue = F.one_hot(mass.argmax(-1), eligible.shape[-1]).bool() & empty[..., None]
            keep |= rescue.any(-2)
            if not self.observer:
                token_keep = keep.repeat_interleave(64, -1)[..., :scores.shape[-1]]
                masked[..., start:start+64, :] = ss.masked_fill(~token_keep[..., None, :], -torch.inf)
            probes = {'dense' if self.observer else f'{self.signal}_{self.mode}_{self.amount}': eligible if self.observer else keep}
            if self.observer:
                for sig in ('qk', 'mass', 'contribution', 'blasst'):
                    for s in (.25, .5, .75, .9):
                        probes[f'{sig}_topk_{s}'] = select(imp[sig], eligible, 'topk', s)
                for s, (mask, lam) in blasst_masks.items(): probes[f'blasst_actual_{s}'] = mask[..., qi, :] & eligible
            for name, mask in probes.items():
                # Diagnostic-only masks use the same nonempty guard as execution.
                e = has & ~(vb.any(-1) & mask[..., None, :]).any(-1)
                mask = mask | (F.one_hot(mass.argmax(-1), eligible.shape[-1]).bool() & e[..., None]).any(-2)
                if not self.observer: e = e | empty
                d = masked_diagnostics(mask, mass, contrib, has)
                packed = torch.stack([eligible.sum(-1), (eligible & ~mask).sum(-1), d['mass_sum'],
                    d['rows'], d['error_sq'], d['dense_sq'], e.sum(-1)], -1).cpu().tolist()
                for b, heads in enumerate(packed):
                    for h, x in enumerate(heads):
                        index = (int(module.layer_idx), h, int(step), name)
                        record = self.buckets.setdefault(index, dict(layer=index[0], head=h, step=int(step),
                            attention_type=kind, probe=name, eligible=0., skipped=0., mass_sum=0., rows=0.,
                            error_sq=0., dense_sq=0., rescued_rows=0., query_blocks=0))
                        for field, v in zip(('eligible','skipped','mass_sum','rows','error_sq','dense_sq','rescued_rows'), x):
                            record[field] += v
                        record['query_blocks'] += 1
            # Representative ranks: every layer/head at the first Q block,
            # sampled calls (bounded per generation). Full metrics remain above.
            if self.observer and qi == 0:
                arrays = {n: t[0].cpu().numpy() for n,t in imp.items()}
                es = eligible[0].cpu().numpy()
                masks = {str(s): m[0][0, :, qi].cpu().numpy() for s,m in blasst_masks.items()}
                for h in range(query.shape[1]):
                    ids = np.flatnonzero(es[h]); ranks = {}
                    for n, a in arrays.items():
                        vals = a[h, ids]
                        # Store +/-inf scores as JSON strings, keeping order exact.
                        ranks[n] = [float(v) if np.isfinite(v) else str(v) for v in vals]
                    self.rankings.append(dict(call=self.calls, layer=int(module.layer_idx),head=h,step=int(step),
                        attention_type=kind,tiles=ids.tolist(),scores=ranks,
                        retained={s:m[h,ids].tolist() for s,m in masks.items()}))
        self.calls += 1
        result = _finish_eager_attention(query, values, masked, valid, dropout, module.training)
        if not torch.isfinite(result[0]).all(): raise FloatingPointError('nonfinite oracle output')
        return result
