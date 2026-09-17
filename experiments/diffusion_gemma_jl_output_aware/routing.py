"""Native attention integration with sketch-only decisions and shared-state error."""
from dataclasses import replace

import torch
import torch.nn.functional as F

from dllm.attention.blasst.core import _prepare_attention_scores, _attention_type
from experiments.diffusion_gemma_value_aware.operators import Config as NativeConfig, reference_output
from experiments.diffusion_gemma_value_aware_gpu.routing import Attention as Counters, FIELDS
from .config import Config
from .projections import SketchCache
from . import reference, kernels


class Attention(Counters):
    def __init__(self, config=None, thresholds=None, validate=False, trusted=False, observer=None):
        super().__init__({}, None, False)
        self.config = Config(**(config or {}))
        self.cache = SketchCache(self.config)
        self.thresholds = thresholds or {}
        self.validate, self.trusted, self.observer = validate, trusted, observer
        self.work = dict(qk_valid_pairs=0, block_softmax_valid_pairs=0,
            projected_pv_madds=0, native_pv_valid_pairs=0, diagnostic_pv_valid_pairs=0,
            physical_pv_omittable_pairs=0, guard_reference_calls=0,
            native_pv_strategy='Whole-canvas native BF16 matmul with masked probabilities; skipped arithmetic is NOT claimed to be elided',
            logits_strategy='Bounded native canvas Q by K logits, preserving native QK/PV shape; no full-sequence square matrix')
        self.work_buffer = None

    def record(self, state, skip, measured, layer, step, kind, prefix):
        super().record(state, skip, measured, layer, step, kind, prefix)
        # Block softmax was required for every eligible candidate's sketch.
        # Unlike logical deletion, that work is never skipped by this router.
        self.buffers[layer, step, kind][0][..., FIELDS.index('softmax_skipped')] = 0

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        if dropout or module.training:
            raise ValueError('Evaluation without dropout is required')
        _, ev, scores, valid = _prepare_attention_scores(module, q, k, v, mask,
            scaling=scaling, is_causal=is_causal, sliding_window=sliding_window)
        valid = valid.expand_as(scores)
        b, heads, nq, nk = scores.shape
        if b != 1:
            raise ValueError('Batch-one experiment')
        layer = int(module.layer_idx)
        step = int(module._blasst_2d_runtime.current_denoising_iteration)
        kind = _attention_type(module, sliding_window)
        prefix = max(0, nk-nq)
        c = self.config
        if self.thresholds:
            policy = self.thresholds[kind]
            if 'log_scale' in policy:
                raise ValueError('Projected criteria use scalar local/global tau, not inverse-length lambda')
            c = replace(c, log_threshold=-float('inf') if policy.get('unpruned') else policy['log_threshold'])
        repeats = heads//v.shape[1]
        keyvalid = valid.reshape(b, v.shape[1], repeats, nq, nk).any((2, 3))
        meta = self.cache.get(layer, v, keyvalid, prefix)
        all_positions, states, skips = [], [], []
        for start in range(0, nq, 128):
            ss, vv = scores[..., start:start+128, :], valid[..., start:start+128, :]
            if self.trusted:
                st = reference.block_statistics(ss, vv, meta['z'].repeat_interleave(repeats, 1),
                                                 meta['projected_norm'].repeat_interleave(repeats, 1))
                skip = reference.route(st, meta['ref'].repeat_interleave(repeats, 1), c)
                fallback = False
            else:
                st = kernels.block_statistics(ss, vv, meta['z'], meta['projected_norm'])
                selected, margin, _ = kernels.route(st, meta['ref'], c)
                skip = selected[0, :, 0, :].unsqueeze(0)
                fallback = bool((margin < kernels.GUARD).any().item())
                if fallback or self.validate:
                    exact = reference.block_statistics(ss, vv, meta['z'].repeat_interleave(repeats, 1),
                                                        meta['projected_norm'].repeat_interleave(repeats, 1))
                    expected = reference.route(exact, meta['ref'].repeat_interleave(repeats, 1), c)
                    if fallback:
                        skip = expected
                        self.fallback_calls += 1
                    if self.validate:
                        if not torch.equal(skip, expected):
                            raise AssertionError(f'Projected GPU/reference mask mismatch at {layer}/{step}/{start}')
                        for key in ('mu', 'logz', 'mean_norm'):
                            torch.testing.assert_close(st[key], exact[key], atol=2e-5, rtol=5e-5)
                        if not torch.equal(st['count'], exact['count']):
                            raise AssertionError('Valid-token count mismatch')
            positions = skip.repeat_interleave(64, -1)[..., :nk][..., None, :].expand_as(ss)
            all_positions.append(positions)
            # Release projected means immediately; diagnostics do not route.
            states.append({'eligible': st['eligible']})
            skips.append(skip)
            if self.observer is not None:
                self.observer(layer, step, kind, start, prefix, ss, vv, ev, v, meta, st, skip, c)
            if self.validate:
                self.validation.append(dict(layer=layer, step=step, query_tile=start//128, kind=kind,
                    eligible=int(st['eligible'].sum()), skipped=int(skip.sum()),
                    fallback=fallback, masks_equal=True, diagnostics_shared_qkv=True))
            del st
        positions = torch.cat(all_positions, -2)
        # Preserve native whole-query BF16 final PV, including unpruned parity.
        output = reference_output(scores, valid, ev, None, NativeConfig(), position_mask=positions)
        has = valid.any(-1)
        dense_probs = torch.softmax(torch.where(has[..., None], scores.float(), 0.), -1).masked_fill(~valid, 0.)
        retained_mass = dense_probs.masked_fill(positions, 0.).sum(-1)
        # Full-dimensional data below are diagnostics ONLY, after mask selection.
        dense_out = dense_probs@ev.float()
        sparse_probs = dense_probs.masked_fill(positions, 0.)/retained_mass.clamp_min(1e-30)[..., None]
        sparse_out = sparse_probs@ev.float()
        for qb, start in enumerate(range(0, nq, 128)):
            pp = dense_probs[..., start:start+128, :]
            state = states[qb]
            state['mass'] = F.pad(pp, (0, (-nk)%64)).unflatten(-1, (-1, 64)).sum(-1)
            hh = has[..., start:start+128]
            dd, oo = dense_out[..., start:start+128, :], sparse_out[..., start:start+128, :]
            measured = dict(rows=hh.sum(-1), mass_sum=(retained_mass[..., start:start+128]*hh).sum(-1),
                error_sq=((oo-dd)*hh[..., None]).square().sum((-1, -2)),
                dense_sq=(dd*hh[..., None]).square().sum((-1, -2)))
            self.record(state, skips[qb], measured, layer, step, kind, prefix)
        valid_pairs = valid.sum()
        packed = torch.stack((valid_pairs, valid_pairs, valid_pairs*meta['z'].shape[-1],
                              valid_pairs, 2*valid_pairs, (valid & positions).sum())).double()
        self.work_buffer = packed if self.work_buffer is None else self.work_buffer+packed
        finite = torch.isfinite(output).all() & torch.isfinite(sparse_out).all() & torch.isfinite(dense_out).all()
        self.finite = finite if self.finite is None else self.finite & finite
        self.calls += 1
        return output.transpose(1, 2).contiguous(), None

    def work_accounting(self):
        result = dict(self.work)
        if self.work_buffer is not None:
            for name, value in zip(tuple(result)[:6], self.work_buffer.cpu().tolist()):
                result[name] = int(value)
        result['guard_reference_calls'] = self.fallback_calls
        result['cache'] = self.cache.work
        result['projection_manifest'] = self.cache.projections.manifest
        result['routing_uses_full_block_pv'] = self.config.family == 'identity'
        result['arithmetic'] = 'FP32 projection, block softmax, sketch means and retained state; native BF16 model output'
        return result
