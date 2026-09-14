"""Accelerated routing with native attention output and deferred GPU counters."""
from dataclasses import replace
import math

import torch
import torch.nn.functional as F
from dllm.attention.blasst.core import _prepare_attention_scores, _finish_eager_attention, _attention_type
from experiments.diffusion_gemma_value_aware.operators import (
    Config, ValueCache, expand_summaries, block_state, streaming_mask, reference_output,
    diagnostics,
)
from .kernels import block_statistics, route, query_state, GUARD

FIELDS = ('eligible', 'skipped', 'softmax_skipped', 'pv_omitted', 'compensated',
          'mass_sum', 'rows', 'error_sq', 'dense_sq', 'rescued_rows',
          'denominator_mass_sum', 'same_budget_best_mass_sum', 'wrongly_replaced_vs_mass',
          'unnecessarily_retained_vs_mass', 'prefix_eligible', 'prefix_skipped',
          'canvas_eligible', 'canvas_skipped', 'boundary_eligible', 'boundary_skipped',
          'prefix_replaced', 'canvas_replaced', 'boundary_replaced')


class Attention:
    def __init__(self, config=None, thresholds=None, validate=False):
        self.config = Config(**(config or {}))
        if self.config.method not in ('dense', 'blasst', 'value', 'mass', 'risk', 'aligned'):
            raise ValueError('Unrequested GPU method')
        self.thresholds = thresholds or {}
        self.cache = ValueCache()
        self.validate = validate
        self.buffers = {}
        self.calls = 0
        self.fallback_calls = 0
        self.validation = []
        self.finite = None
        self.distributions = []

    @property
    def records(self):
        if self.finite is not None and not bool(self.finite.item()):
            raise FloatingPointError('Nonfinite accelerated attention output')
        result = []
        for (layer, step, kind), (packed, calls) in sorted(self.buffers.items()):
            for batch in packed.cpu().tolist():
                for head, values in enumerate(batch):
                    result.append(dict(layer=layer, step=step, head=head, attention_type=kind,
                                       probe='execution', calls=calls, **dict(zip(FIELDS, values))))
        return result

    def record(self, state, skip, measured, layer, step, kind, prefix):
        eligible = state['eligible']
        count, changed = eligible.sum(-1), (skip & eligible).sum(-1)
        zeros = torch.zeros_like(count)
        best_mass, wrong = measured['mass_sum'], zeros
        if 'mass' in state:
            order = state['mass'].sum(-2).masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
            ranks = torch.empty_like(order).scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
            best = eligible & (ranks < (count-changed)[..., None])
            best_mass = (state['mass']*best[..., None, :]).sum((-1, -2))
            wrong = (best & skip & eligible).sum(-1)
        data = [count, changed, changed, changed, zeros,
                measured['mass_sum'], measured['rows'], measured['error_sq'], measured['dense_sq'], zeros,
                measured['mass_sum'], best_mass, wrong, wrong]
        starts = torch.arange(eligible.shape[-1], device=eligible.device)*64
        regions = (starts+64 <= prefix, starts >= prefix, (starts < prefix) & (starts+64 > prefix))
        for region in regions:
            data.extend(((eligible & region).sum(-1), (skip & eligible & region).sum(-1)))
        data.extend((skip & eligible & region).sum(-1) for region in regions)
        # Match old per-call FP32 packing followed by Python-double accumulation.
        packed = torch.stack(data, -1).double()
        key = (layer, step, kind)
        if key in self.buffers:
            old, calls = self.buffers[key]
            old.add_(packed)
            self.buffers[key] = (old, calls+1)
        else:
            self.buffers[key] = (packed, 1)

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        if dropout or module.training:
            raise ValueError('Only evaluation without attention dropout is supported')
        _, ev, scores, valid = _prepare_attention_scores(module, q, k, v, mask,
            scaling=scaling, is_causal=is_causal, sliding_window=sliding_window)
        valid = torch.broadcast_to(valid, scores.shape)
        batch, heads, nq, nk = scores.shape
        if batch != 1:
            raise ValueError('Experiment generation is fixed batch size1')
        layer, step = int(module.layer_idx), int(module._blasst_2d_runtime.current_denoising_iteration)
        kind = _attention_type(module, sliding_window)
        prefix = max(0, nk-nq)
        config = self.config
        if self.thresholds:
            policy = self.thresholds[kind]
            length = int(valid.any((1, 2)).sum(-1).item())
            logt = policy['log_scale']-math.log(length) if 'log_scale' in policy else policy['log_threshold']
            if policy.get('cap_one'):
                logt = min(0., logt)
            if policy.get('unattainable') and policy.get('cap_one'):
                logt = 0.
            config = replace(config, log_threshold=logt)
        meta = {}
        if config.method != 'dense':
            repeats = ev.shape[1]//v.shape[1]
            keyvalid = valid.any(-2)[:, ::repeats, :]
            # Identical original pooling/refresh/reference, including GQA.
            meta = expand_summaries(self.cache.get(layer, v, keyvalid, prefix), repeats)
        lognorm = None
        if config.method == 'aligned':
            lognorm = (meta['token_norms']/meta['ref'][..., None]).clamp_min(1e-30).log()
        normalized_risk = config.method in ('mass', 'risk')
        # Their original block log-sums are already needed for diagnostics.
        # Reuse them for the fused state loop rather than introducing a second
        # reduction-order difference in these cancellation-sensitive risks.
        stats = None if normalized_risk else block_statistics(scores, valid, lognorm)
        if config.method != 'dense' and not normalized_risk:
            masks, margin = route(stats, config, meta, config.log_threshold)
            # Only normalization-dependent risks accumulate log-sum rounding.
            # Gap/value/aligned risks use exact maxima of the same native
            # logits and identical PyTorch metadata; their exact ties need
            # no numerical guard and must still be retained by strict '<'.
        elif config.method == 'dense':
            masks = torch.zeros(stats[0].shape[:-1], dtype=torch.bool, device=q.device)
        fallback_any = False
        all_positions = []
        for qb, start in enumerate(range(0, nq, 128)):
            ss, vv = scores[..., start:start+128, :], valid[..., start:start+128, :]
            gpu_state = None if normalized_risk else query_state(stats, batch, heads, qb, ss.shape[-2])
            # Preserve the historical diagnostic reduction order. A trial
            # whole-PV reassociation failed long-context diagnostic parity,
            # and is deliberately NOT deployed in the final experiment.
            state = {'eligible': gpu_state['eligible']} if config.method == 'dense' else block_state(ss, vv, ev)
            fallback = False
            if normalized_risk:
                native_stats = []
                for field in ('b', 'logz', 'count', 'b'):
                    value = state[field].transpose(-1, -2).reshape(batch*heads, 1, -1, ss.shape[-2])
                    value = F.pad(value, (0, 128-ss.shape[-2]), value=0 if field == 'count' else -torch.inf).contiguous()
                    native_stats.append(value)
                selected, margin = route(native_stats, config, meta, config.log_threshold)
                skip = selected[:, 0, :].reshape(batch, heads, -1)
                fallback = bool((margin < GUARD).any().item())
                fallback_any |= fallback
                gpu_state = state
            else:
                skip = masks[:, qb, :].reshape(batch, heads, -1)
            if (fallback or self.validate) and config.method != 'dense':
                expected = streaming_mask(state, meta, config)
                if fallback:
                    skip = expected
                if self.validate:
                    mismatches = int((skip != expected).sum().item())
                    if mismatches:
                        raise AssertionError(f'GPU/reference mask mismatch: {layer}/{step}: {mismatches}')
                    if not torch.equal(state['count'], gpu_state['count']):
                        raise AssertionError('GPU valid-token counts differ')
                    torch.testing.assert_close(gpu_state['logz'], state['logz'], atol=2e-5, rtol=2e-6)
            positions = skip.repeat_interleave(64, -1)[..., :nk][..., None, :].expand_as(ss)
            all_positions.append(positions)
            if config.method == 'dense':
                has = vv.any(-1)
                probs = torch.softmax(torch.where(has[..., None], ss.float(), 0.), -1).masked_fill(~vv, 0.)
                dense = probs @ ev.float()
                measured = dict(mass_sum=has.sum(-1), rows=has.sum(-1),
                    error_sq=torch.zeros_like(has.sum(-1)), dense_sq=dense.square().sum((-1, -2)))
            else:
                measured = diagnostics(state, skip, config, meta)
            if self.validate:
                self.validation.append(dict(layer=layer, step=step, query_tile=qb, kind=kind,
                    eligible=int(state['eligible'].sum().item()), skipped=int(skip.sum().item()),
                    fallback=fallback, masks_equal=True, diagnostics_exact_reference=True))
            self.record(state, skip, measured, layer, step, kind, prefix)
        if config.method == 'dense':
            output = _finish_eager_attention(q, ev, scores, valid, dropout, False)
        else:
            positions = torch.cat(all_positions, -2)
            output = (reference_output(scores, valid, ev, None, config,
                                       position_mask=positions).transpose(1, 2).contiguous(), None)
        finite = torch.isfinite(output[0]).all()
        self.finite = finite if self.finite is None else self.finite & finite
        self.fallback_calls += int(fallback_any)
        self.calls += 1
        return output
