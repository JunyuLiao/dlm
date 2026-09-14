"""DiffusionGemma adapter and shared dense-state pooling/routing screen."""
from dataclasses import asdict, replace
import math
import numpy as np
import torch
import torch.nn.functional as F
from dllm.attention.blasst.core import _prepare_attention_scores, _finish_eager_attention, _attention_type
from .operators import (Config, ValueCache, expand_summaries, block_state, diagnostics,
    screen_risks, streaming_mask, reference_output, rescue, sol_proxy, proxy_keep, select,
    Q_TILE, KV_TILE, POOLINGS)

SCREEN_STEPS = (0, 4, 12, 24)


def screen_configs():
    configs = {'blasst': Config(method='blasst'), 'mass': Config(method='mass'),
        'aligned': Config(method='aligned'), 'centered': Config(method='centered'),
        'compensate': Config(method='compensate'), 'zero_pv': Config(method='zero_pv')}
    for method in ('value', 'mass_value', 'risk'):
        for pool in POOLINGS: configs[f'{method}_{pool}'] = Config(method=method, pooling=pool)
    return configs


class Attention:
    def __init__(self, config=None, thresholds=None, screen=False):
        self.config = Config(**(config or {})); self.thresholds = thresholds or {}
        self.screen = screen; self.cache = ValueCache(); self.buckets = {}
        self.risk_arrays = {}; self.distributions = []; self.calls = 0; self.observed = set()

    @property
    def records(self): return list(self.buckets.values())

    def arrays(self): return {k: np.concatenate(v) for k, v in self.risk_arrays.items()}

    def record(self, state, mask, config, metadata, layer, step, kind, probe, prefix, rescued=None, measured=None):
        d = measured if measured is not None else diagnostics(state, mask, config, metadata)
        eligible = state['eligible']; count = eligible.sum(-1)
        changed = (mask & eligible).sum(-1)
        compensation = config.method in ('compensate', 'zero_pv')
        mass_best = d['mass_sum']; misplaced = torch.zeros_like(count)
        if 'mass' in state:
            if '_mass_ranks' not in state:
                order = state['mass'].sum(-2).masked_fill(~eligible,-torch.inf).argsort(dim=-1,descending=True,stable=True)
                state['_mass_ranks'] = torch.empty_like(order).scatter_(-1,order,
                    torch.arange(order.shape[-1],device=order.device).expand_as(order))
            # Unconstrained mass ranking at the identical per-head tile budget.
            # This is selection diagnosis, not a feasible/optimal task oracle.
            best = eligible & (state['_mass_ranks'] < (count-changed)[...,None])
            mass_best = (state['mass'] * best[...,None,:]).sum((-1,-2))
            misplaced = (best & mask & eligible).sum(-1)
        prefix_tiles = torch.arange(eligible.shape[-1], device=eligible.device) * KV_TILE < prefix
        # Boundary-spanning tiles are separately classified, not assigned twice.
        canvas_tiles = torch.arange(eligible.shape[-1], device=eligible.device) * KV_TILE >= prefix
        full_prefix = (torch.arange(eligible.shape[-1], device=eligible.device) + 1) * KV_TILE <= prefix
        boundary = prefix_tiles & ~full_prefix
        zeros = torch.zeros_like(count)
        fields = ('eligible', 'skipped', 'softmax_skipped', 'pv_omitted', 'compensated',
            'mass_sum', 'rows', 'error_sq', 'dense_sq', 'rescued_rows',
            'denominator_mass_sum','same_budget_best_mass_sum','wrongly_replaced_vs_mass','unnecessarily_retained_vs_mass',
            'prefix_eligible', 'prefix_skipped', 'canvas_eligible', 'canvas_skipped',
            'boundary_eligible', 'boundary_skipped', 'prefix_replaced','canvas_replaced','boundary_replaced')
        data = [count, zeros if compensation else changed, zeros if compensation else changed,
            changed, changed if config.method == 'compensate' else zeros,
            d['mass_sum'], d['rows'], d['error_sq'], d['dense_sq'],
            rescued.sum(-1) if rescued is not None else zeros,
            d['rows'] if compensation else d['mass_sum'],mass_best,misplaced,misplaced]
        for region in (full_prefix, canvas_tiles, boundary):
            data.extend(((eligible & region).sum(-1), zeros if compensation else (mask & eligible & region).sum(-1)))
        data.extend((mask & eligible & region).sum(-1) for region in (full_prefix,canvas_tiles,boundary))
        packed = torch.stack(data, -1).cpu().tolist()
        for batch in packed:
            for head, x in enumerate(batch):
                index = (layer, head, step, probe)
                r = self.buckets.setdefault(index, dict(layer=layer, head=head, step=step,
                    attention_type=kind, probe=probe, calls=0, **{f: 0. for f in fields}))
                for f, v in zip(fields, x): r[f] += v
                r['calls'] += 1

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        ek, ev, scores, valid = _prepare_attention_scores(module, q, k, v, mask,
            scaling=scaling, is_causal=is_causal, sliding_window=sliding_window)
        valid = torch.broadcast_to(valid, scores.shape)
        runtime = module._blasst_2d_runtime
        step = int(runtime.current_denoising_iteration); layer = int(module.layer_idx)
        kind = _attention_type(module, sliding_window); prefix = max(0, k.shape[-2]-q.shape[-2])
        length = int(valid.any((1, 2)).sum(-1).item())
        config = self.config
        if self.thresholds:
            policy = self.thresholds[kind]
            logt = policy['log_scale'] - math.log(length) if 'log_scale' in policy else policy['log_threshold']
            if policy.get('cap_one'): logt = min(0., logt)
            if policy.get('unattainable') and policy.get('cap_one'): logt = 0.
            config = replace(config, log_threshold=logt)
        observe = self.screen and step in SCREEN_STEPS and (layer, step) not in self.observed
        if observe: self.observed.add((layer, step))
        # Metadata is indexed by native KV heads and expanded only after cache.
        repeats = ev.shape[1] // v.shape[1]
        kvvalid = valid.any(-2)[:, ::repeats, :]
        meta = expand_summaries(self.cache.get(layer, v, kvvalid, prefix), repeats)
        position_masks = []
        for start in range(0, q.shape[-2], Q_TILE):
            ss = scores[..., start:start+Q_TILE, :]; vv = valid[..., start:start+Q_TILE, :]
            if config.method == 'dense' and not observe:
                eligible = F.pad(vv, (0, (-vv.shape[-1]) % KV_TILE)).unflatten(-1, (-1, KV_TILE)).any((-1,-3))
                has = vv.any(-1)
                probs = torch.softmax(torch.where(has[...,None],ss.float(),0.),-1).masked_fill(~vv,0.)
                dense = probs @ ev.float()
                rows = has.sum(-1)
                self.record({'eligible':eligible}, torch.zeros_like(eligible), config, meta, layer, step, kind,
                    'execution', prefix, measured=dict(rows=rows,mass_sum=rows,
                    error_sq=torch.zeros_like(rows),dense_sq=dense.square().sum((-1,-2))))
                continue
            state = block_state(ss, vv, ev)
            empty = None
            if config.method == 'dense': replaced = torch.zeros_like(state['eligible'])
            elif config.method == 'diagnostic':
                signal = dict(qk=state['b'].amax(-2), mass=state['mass'].sum(-2),
                    contribution=state['contrib'].square().sum((-1, -3)).sqrt())[config.pooling]
                keep, empty = rescue(select(signal, state['eligible'], config.mode, config.amount), state)
                replaced = state['eligible'] & ~keep
            elif config.method == 'sol':
                proxy = sol_proxy(q[..., start:start+Q_TILE, :], ek, vv, meta,
                    scaling if scaling is not None else q.shape[-1]**-.5, config.value_proxy)
                keep, z, degenerate = proxy_keep(proxy, state['eligible'], config)
                keep, empty = rescue(keep, state); replaced = state['eligible'] & ~keep
                if observe: self._distribution(z, state['eligible'], layer, step, kind, config.value_proxy)
            else: replaced = streaming_mask(state, meta, config)
            self.record(state, replaced, config, meta, layer, step, kind, 'execution', prefix, empty)
            if config.method != 'dense':
                positions = replaced.repeat_interleave(KV_TILE,-1)[...,:scores.shape[-1]][...,None,:]
                position_masks.append(positions.expand(*ss.shape))
            if observe: self._screen(state, meta, q[..., start:start+Q_TILE, :], ek, vv,
                scaling if scaling is not None else q.shape[-1]**-.5, layer, step, kind, prefix, length)
        self.calls += 1
        if config.method == 'dense':
            output = _finish_eager_attention(q, ev, scores, valid, dropout, module.training)
        else:
            # Preserve the original whole-query BF16 PV matmul shape. Splitting
            # it by physical Q tile changes CUDA rounding, even without skips.
            positions = torch.cat(position_masks,-2)
            output = (reference_output(scores,valid,ev,None,config,meta['vectors'],positions).transpose(1,2).contiguous(),None)
        if not torch.isfinite(output[0]).all(): raise FloatingPointError('nonfinite attention output')
        return output

    def _distribution(self, z, eligible, layer, step, kind, value_aware):
        a = z[eligible].float().cpu().numpy()
        self.distributions.append(dict(layer=layer, step=step, attention_type=kind, value_aware=value_aware,
            count=len(a), mean=float(a.mean()), std=float(a.std()),
            skew=float((a**3).mean()), fourth_moment=float((a**4).mean()),
            quantiles=np.quantile(a, [.01,.1,.25,.5,.75,.9,.99]).tolist()))

    def _screen(self, state, meta, q, k, valid, scale, layer, step, kind, prefix, length):
        for name, c in screen_configs().items():
            risks, row_risks = screen_risks(state, meta, c)
            adjusted = risks + math.log(length) if c.method == 'blasst' else risks
            self.risk_arrays.setdefault(f'{name}__{kind}', []).append(adjusted[state['eligible']].float().cpu().numpy())
            # Same-budget diagnostics isolate pooling quality from calibration.
            # Scores remain streaming-available estimates, not dense output.
            for target in (.25, .5, .75, .9):
                keep, empty = rescue(select(risks, state['eligible'], 'topk', target), state)
                self.record(state, state['eligible'] & ~keep, c, meta, layer, step, kind,
                    f'screen/{name}/s{int(100*target)}', prefix, empty)
        for signal in ('qk', 'mass', 'contribution'):
            values = dict(qk=state['b'].amax(-2), mass=state['mass'].sum(-2),
                contribution=state['contrib'].square().sum((-1, -3)).sqrt())[signal]
            for target in (.25, .5, .75, .9):
                keep, empty = rescue(select(values, state['eligible'], 'topk', target), state)
                self.record(state, state['eligible'] & ~keep, Config(), meta, layer, step, kind,
                    f'oracle/{signal}/s{int(100*target)}', prefix, empty)
        for value_aware in (False, True):
            proxy = sol_proxy(q, k, valid, meta, scale, value_aware)
            for mode in ('gaussian', 'topk'):
                for target in (.25, .5, .75, .9):
                    c = Config(method='sol', mode=mode, amount=target, value_proxy=value_aware)
                    keep, z, _ = proxy_keep(proxy, state['eligible'], c)
                    keep, empty = rescue(keep, state)
                    self.record(state, state['eligible'] & ~keep, c, meta, layer, step, kind,
                        f'sol/{"value" if value_aware else "plain"}/{mode}/s{int(100*target)}', prefix, empty)
                    if mode == 'gaussian' and target == .5:
                        self._distribution(z, state['eligible'], layer, step, kind, value_aware)
