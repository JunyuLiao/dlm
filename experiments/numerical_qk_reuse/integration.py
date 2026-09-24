"""Request-local native DG adapter for numerical_reuse_current_V.

Only decoder attention is overridden. The sampler, encoder/cache writes, model
V projection and output projection are untouched. Scores and decisions have
different clocks. Initial implementation uses the caller's CUDA stream.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch

from dllm.attention.blasst.core import _attention_type, _attention_validity
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from experiments.value_direction_hopper.integration import Sketches
from experiments.value_direction_hopper.query_adaptive import State
from .cache import Identity, ScoreCache
from .cached_executor import attention, route_only

SUPPORT_MODES = ('legacy_junyu_mask', 'native_mask')
# cached_scores: production M1/M3 -- the routing decision AND the final
# softmax/PV both consume the stale cached scores.
# routing_only_current_output: Phase D successor selected by the phaseAB
# diagnostic (results/numerical_qk_reuse_recovery_20260924/phaseAB_report.md).
# The routing decision (which KV64 tiles to retain) still comes from the
# stale cached scores -- that choice stays comparatively stable even at
# score_age 7 -- but the final softmax/PV always consumes freshly recomputed
# current QK restricted to that retained support. This does NOT skip QK
# compute; it is a support-reuse-only variant, not a faster relabeling of M1.
OUTPUT_MODES = ('cached_scores', 'routing_only_current_output')


@dataclass
class Decision:
    skipped: torch.Tensor
    eligible: torch.Tensor


class NativeReuseState(State):
    def begin(self, cur_step, canvas):
        super().begin(cur_step, canvas)
        self.router.begin_step(self.canvas, self.iteration - 1)


class Attention:
    def __init__(self, adapter, thresholds, *, score_period=8, decision_interval=1,
                 trace=False, max_cache_bytes=2 * 1024**3, support='legacy_junyu_mask',
                 output_mode='cached_scores'):
        if support not in SUPPORT_MODES:
            raise ValueError(support)
        if output_mode not in OUTPUT_MODES:
            raise ValueError(output_mode)
        self.support = support
        self.output_mode = output_mode
        self.thresholds = thresholds
        self.cache = ScoreCache(score_period, decision_interval, max_cache_bytes)
        self.projections = Projections()
        # Restores Junyu's producer-owned lease: unchanged encoder-cache V/norm
        # sketches are reprojected only when the source object/version/shape
        # actually changes, not on every decision-refresh call.
        self.sketches = Sketches(adapter, self.projections, fused=False)
        self.query_sensitivity = None
        self.policy_selector = None
        self.trace = trace
        self.canvas, self.step, self.epoch = -1, -1, 0
        self.sources, self.handles, self.valid_keys = {}, [], {}
        self.pending, self.call_metadata = [], []
        self.calls = self.score_calls = self.decision_calls = self.held_calls = 0
        self.current_qk_elements = self.reused_qk_elements = 0
        self.routing_only_extra_qk_elements = 0
        self.peak_score_bytes = 0
        self.unsupported_mask_refreshes = 0
        for name, module in adapter.model.named_modules():
            if type(module).__name__ == 'DiffusionGemmaEncoderModel':
                self.handles.append(module.register_forward_pre_hook(self.invalidate))
            if adapter.is_blasst_attention_module(name, module):
                self.handles.append(module.register_forward_pre_hook(self.identify, with_kwargs=True))
        if not self.handles:
            raise ValueError('No qualified native DG hooks')

    def invalidate(self, *_):
        # Invoked BEFORE prefill or causal commit can mutate the encoder cache.
        self.epoch += 1
        self.cache.clear()
        self.sources.clear()
        self.valid_keys.clear()

    def begin_step(self, canvas, step):
        if canvas != self.canvas:
            self.cache.clear()
            self.valid_keys.clear()
        self.canvas, self.step = canvas, step

    def identify(self, module, args, kwargs):
        cache = kwargs.get('past_key_values', args[3] if len(args) > 3 else None)
        if cache is None or getattr(cache, 'is_compileable', False):
            raise ValueError('First native prototype requires a populated dynamic encoder cache')
        layer = int(module.layer_idx)
        source = cache.layers[layer]
        # Native DynamicCache metadata supplies the absolute query position;
        # the layer tensor length supplies its stored (possibly windowed) prefix.
        absolute = cache.get_seq_length()
        if not isinstance(absolute, int):
            raise ValueError('Tensor-valued cache length unsupported without a device controller')
        prefix = source.keys.shape[-2]
        if prefix > absolute:
            raise ValueError('Stored prefix exceeds actual native position range')
        self.sources[layer] = (source.keys, source.values, absolute, prefix)

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        if dropout or module.training or self.step < 0:
            raise ValueError('Decoder inference inside a recorded native step required')
        b, h, nq, d = q.shape
        hk, source_nk = k.shape[1], k.shape[-2]
        layer = int(module.layer_idx)
        prefix_k, prefix_v, absolute, prefix = self.sources[layer]
        if source_nk != prefix + nq or v.shape != k.shape or b != 1:
            raise ValueError('Unsupported native KV concatenation or batched request')
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError('Frozen BF16 model geometry required')
        causal = bool(is_causal) if is_causal is not None else mask is None and nq > 1
        scale = float(scaling) if scaling is not None else d ** -.5
        # With mask=None this is the exact Junyu lower local bound. Align to64
        # to preserve original block boundaries and the sequential scan order.
        # native_mask never narrows beyond the already-supplied cache/canvas
        # keys: the installed native SDPA path ignores the sliding_window
        # keyword entirely (ALL_ATTENTION_FUNCTIONS sdpa forward ignores extra
        # kwargs) and DynamicSlidingWindowLayer already caps the stored
        # encoder prefix at the source, so no additional query-relative crop
        # is a correct model of what native dense actually attends.
        window = None if self.support == 'native_mask' else sliding_window
        crop = max(0, prefix - int(window) + 1) // 64 * 64 if window and mask is None else 0
        k, v = k[..., crop:, :], v[..., crop:, :].contiguous()
        nk = k.shape[-2]
        kind = _attention_type(module, sliding_window)
        signature = (causal, sliding_window, None if mask is None else
                     (tuple(mask.shape), str(mask.dtype)))
        identity = Identity(0, self.canvas, self.epoch, layer, b, h, hk, nq, nk, d,
                            absolute, absolute-prefix+crop, source_nk, scale,
                            str(q.dtype), str(q.device), signature, id(prefix_k))
        # Arbitrary mask contents are not inferred from pointers/shape.
        plan = self.cache.plan(identity, self.step, force_refresh=mask is not None)
        self.cache.reserve(identity)  # before allocation
        current_for_output = None
        if plan.score_refresh:
            score = self.observe_scores(q, k, mask, scale, causal, window, crop)
            self.cache.publish_scores(identity, self.step, score)
            valid = torch.isfinite(score).reshape(b, hk, h//hk, nq, nk).any((2, 3))
            self.valid_keys[layer] = valid
            self.score_calls += 1
            self.current_qk_elements += b*h*nq*nk
            self.unsupported_mask_refreshes += int(mask is not None)
            current_for_output = score
        else:
            self.reused_qk_elements += b*h*nq*nk
            if self.output_mode == 'routing_only_current_output':
                # The routing decision below still reads the stale cached
                # score; only the final softmax/PV gets a fresh recompute.
                # This is charged current-QK work, not reuse -- the whole
                # point of this mode is that it does NOT skip QK compute.
                current_for_output = self.observe_scores(q, k, mask, scale, causal, window, crop)
                self.current_qk_elements += b*h*nq*nk
                self.routing_only_extra_qk_elements += b*h*nq*nk
        entry = self.cache.get(identity)
        if plan.decision_refresh:
            valid = self.valid_keys[layer]
            # Leased sketch: reprojects only the aligned suffix (current canvas
            # plus the unaligned boundary) when the encoder-owned prefix source
            # object/version/shape are unchanged from the prior call. A commit
            # or new request invalidates the lease via Sketches' own encoder
            # pre-hook; query-only changes within a canvas never do.
            projected, ref = self.sketches.get(layer, v, valid, prefix - crop)
            threshold = float(self.thresholds[kind]['log_threshold'])
            # The fused route+PV call is only worth issuing when its output is
            # the one we actually return: in cached_scores mode always, and on
            # score anchors where the cached tensor IS the current one, so the
            # same QK is never observed twice.
            fused = (self.output_mode == 'cached_scores'
                     or current_for_output is entry.scores)
            if fused:
                result = attention(entry.scores, v, projected.contiguous(), ref.contiguous(),
                                   sensitivity=self.query_sensitivity,
                                   log_threshold=threshold, trace=self.trace)
                decision = Decision(result.skipped, result.eligible)
            else:
                # Selector only: no discarded PV, no discarded [B,H,Q,D]
                # output. Malformed cached scores stay detectable through the
                # route's own per-tile flag instead of through that PV pass.
                route = route_only(entry.scores, projected.contiguous(), ref.contiguous(),
                                   sensitivity=self.query_sensitivity,
                                   log_threshold=threshold)
                torch._assert_async(~route.invalid_tiles.any(),
                                    'Invalid cached scores in routing decision: request must fail')
                decision = Decision(route.skipped, route.eligible)
                # Same retained tiles as the stale-score routing decision;
                # the final softmax/PV consumes current, not cached, scores.
                result = attention(current_for_output, v, skipped=route.skipped,
                                   eligible=route.eligible, trace=self.trace)
            self.cache.publish_decision(identity, self.step, decision)
            self.decision_calls += 1
        else:
            scores_for_pv = current_for_output if self.output_mode == 'routing_only_current_output' else entry.scores
            result = attention(scores_for_pv, v, skipped=entry.decision.skipped,
                               eligible=entry.decision.eligible, trace=self.trace)
            self.held_calls += 1
        # Explicit asynchronous error, never consume zeroed invalid rows. This
        # is a measured device guard, not a steady-path host tensor read.
        torch._assert_async(~result.invalid_scores.any(), 'Invalid cached scores: request must fail')
        torch._assert_async(torch.isfinite(result.output).all(), 'Invalid cached-score attention output')
        self.calls += 1
        self.peak_score_bytes = max(self.peak_score_bytes, self.cache.storage_bytes)
        metadata = dict(canvas=self.canvas, decoder_call=self.step, layer=layer, kind=kind,
                        query_start=absolute, key_start=identity.key_start, q=nq, k=nk,
                        source_k=source_nk, score_anchor=entry.score_step,
                        score_age=self.step-entry.score_step, score_refresh=plan.score_refresh,
                        decision_refresh=plan.decision_refresh, reason=plan.reason,
                        current_qk_elements=b*h*nq*nk if current_for_output is not None else 0)
        self.call_metadata.append(metadata)
        # Tiny per-layer physical-work reductions stay on device until finish.
        self.pending.append(((result.skipped & result.eligible).sum(), result.eligible.sum()))
        return result.output.transpose(1, 2).contiguous(), None

    @staticmethod
    def observe_scores(q, k, mask, scale, causal, window, crop):
        """The ONLY current-QK producer in this adapter; called at real anchors."""
        repeated = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        # Both matmul and multiplication preserve Junyu BF16 score transforms.
        scores = torch.matmul(q, repeated.transpose(-1, -2)) * scale
        cropped_mask = None if mask is None else mask[..., crop:crop+k.shape[-2]]
        valid = _attention_validity(cropped_mask, q, k, is_causal=causal, sliding_window=window)
        if cropped_mask is not None and cropped_mask.dtype != torch.bool:
            if cropped_mask.dtype != torch.bfloat16:
                raise ValueError('Additive score bias requires the frozen BF16 specialization')
            scores = scores + cropped_mask
        return scores.float().masked_fill(~valid, -math.inf).contiguous()

    def records(self):
        values = torch.stack([torch.stack(x) for x in self.pending]).cpu().tolist() if self.pending else []
        return [dict(meta, physical_skipped=counts[0], physical_eligible=counts[1],
                     decision_geometry='per-head Q128 KV64')
                for meta, counts in zip(self.call_metadata, values)]

    def counters(self):
        return dict(attention_calls=self.calls, score_refresh_calls=self.score_calls,
                    decision_refresh_calls=self.decision_calls, held_decision_calls=self.held_calls,
                    current_qk_elements=self.current_qk_elements, reused_qk_elements=self.reused_qk_elements,
                    projected_current_v_tokens=self.sketches.projected_tokens,
                    reused_current_v_tokens=self.sketches.reused_tokens,
                    peak_score_bytes=self.peak_score_bytes,
                    unsupported_mask_refreshes=self.unsupported_mask_refreshes,
                    support=self.support, output_mode=self.output_mode,
                    routing_only_extra_qk_elements=self.routing_only_extra_qk_elements,
                    work_counter_scope='dispatch elements, not measured DRAM bytes',
                    cuda_graph_qualified=False, publication='same current CUDA stream')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()
        self.sources.clear()
        self.valid_keys.clear()
        self.sketches.close()


@contextmanager
def install(adapter, config, condition):
    if condition not in ('M1', 'M3'):
        raise ValueError(condition)
    if getattr(adapter.model, '_value_direction_lease', False):
        raise RuntimeError('Concurrent request binding unsupported')
    adapter.model._value_direction_lease = True
    binding = router = None
    try:
        binding = _install_dense(adapter)
        router = Attention(adapter, config['policy'], score_period=config['score_refresh_period'],
                           decision_interval=config['decision_interval'], trace=False,
                           support=config.get('support', 'legacy_junyu_mask'),
                           output_mode=config.get('output_mode', 'cached_scores'))
        binding.runtime.attention_override = router
        state = NativeReuseState('T', router, m_ref=config['m_ref'], beta=config['beta'],
                                 gamma=config['gamma'], diagnostics=config['diagnostic'])
        yield dict(binding=binding, router=router, state=state, counters=router.counters)
    finally:
        if router is not None:
            router.close()
        if binding is not None:
            binding.close()
        del adapter.model._value_direction_lease
