"""Request-local native DG adapter for numerical_reuse_current_V.

Only decoder attention is overridden. The sampler, encoder/cache writes, model
V projection and output projection are untouched. Scores and decisions have
different clocks. Initial implementation uses the caller's CUDA stream.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import math

import torch

from dllm.attention.blasst.core import _attention_type, _attention_validity
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from experiments.value_direction_hopper.integration import Sketches
from experiments.value_direction_hopper.query_adaptive import State
from .cache import Identity, ScoreCache
from .cached_executor import (allocate_summary, attention, preqk_attention,
                              route_only, summary_bytes as summary_nbytes, fused_guard)

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
# historical_route_preqk_current_output: same support and same current
# attention output as routing_only_current_output, but the current QK of a
# dropped tile is never computed. Only this mode physically avoids dropped
# current QK; routing_only_current_output remains the full-QK diagnostic
# reference that it is compared against.
OUTPUT_MODES = ('cached_scores', 'routing_only_current_output',
                'historical_route_preqk_current_output')
PREQK_MODE = 'historical_route_preqk_current_output'
# legacy_recompute: the selector recomputes every block's softmax/sketch
# product on every decision, as it always has.
# prefix_block_summary: for KV tiles lying WHOLLY inside the immutable prefix,
# the per-row block log mass and weighted projected value are computed once at
# the real score anchor and read back afterwards. Exact under the existing
# method -- those two quantities depend only on the frozen cached scores and
# the frozen prefix projected V. Decisions, risk, alpha and the retained scan
# state are still recomputed every step from live T and the live full-V RMS.
SELECTORS = ('legacy_recompute', 'prefix_block_summary')
# v23 Track P. native_bootstrap2_observe1: canvas call 0 returns true native
# attention with no observation; call 1 returns true native attention and ALSO
# observes current scores and runs the unchanged selector, publishing a map for
# later calls only; calls >= 2 follow the ordinary clocks with score origin 1.
BOOTSTRAP_POLICIES = ('native_bootstrap2_observe1',)
OBSERVATION_PRODUCERS = ('repeat_interleave', 'grouped_q')
# v25 named numerical-implementation variant. aligned16 keeps the producer at
# the real K, copies its FP32 bits into a 16-key-pitch buffer whose tail is
# -inf, and pads only the projected-V sketch passed to the route. Tile count,
# prefix boundary, reference RMS, legal pairs and the output consumer keep the
# real K. The route's load layout (hence reduction order) can differ, so stored
# summaries are NOT claimed bit-identical; qualification is by measurement.
ROUTE_STORAGES = ('logical', 'aligned16')


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
                 output_mode='cached_scores', selector='legacy_recompute',
                 selector_layers='local', max_summary_bytes=1024**3,
                 kernel_variant='static', telemetry='full', guard_mode='separate',
                 consumer='triton', support_build=None):
        if support not in SUPPORT_MODES:
            raise ValueError(support)
        if output_mode not in OUTPUT_MODES:
            raise ValueError(output_mode)
        if selector not in SELECTORS:
            raise ValueError(selector)
        if selector_layers not in ('local', 'all'):
            raise ValueError(selector_layers)
        if kernel_variant not in ('static', 'generic'):
            raise ValueError(kernel_variant)
        # full: per-call metadata dicts + two on-device tile-count reductions per
        # layer call (audit data, never used to choose support). minimal: host
        # counters, cache/phase ownership and every correctness guard only;
        # per-tile skip statistics are N/A (take them from a token-matched full
        # audit run), never reported as zeros.
        if telemetry not in ('full', 'minimal'):
            raise ValueError(telemetry)
        self.telemetry = telemetry
        # separate: the three original _assert_async chains (reference).
        # fused: one kernel + one assert over exactly the same conditions.
        if guard_mode not in ('separate', 'fused'):
            raise ValueError(guard_mode)
        self.guard_mode = guard_mode
        # v11: 'hopper' replaces ONLY the current-output consumer of ordinary
        # and held steps by the preselected-support Hopper kernel (vd_support_v1).
        # Anchors, selector, score clock, summaries, T and guards are unchanged.
        if consumer not in ('triton', 'hopper'):
            raise ValueError(consumer)
        self.consumer = consumer
        if consumer == 'hopper':
            # Aliased: a bare 'support' would shadow the mask-mode parameter.
            from experiments.value_direction_hopper import support as support_consumer
            if support_build is None:
                raise ValueError('hopper consumer requires a verified support_build identity')
            self.support_identity = support_consumer.load(support_build)
            self._support = support_consumer
        self.kernel_variant = kernel_variant
        self.selector = selector
        self.selector_layers = selector_layers
        assert isinstance(support, str) and support in SUPPORT_MODES, 'mask-mode parameter was shadowed'
        self.summaries = {}
        self.summary_hits = self.summary_builds = self.summary_misses = 0
        # Two SEPARATE, separately enforced budgets. max_cache_bytes bounds the
        # retained score cache only (ScoreCache.reserve); max_summary_bytes
        # bounds the prefix summaries only (checked before allocation). Their
        # sum is the history+summary bound; neither alone is a total bound.
        if max_summary_bytes < 0:
            raise ValueError(max_summary_bytes)
        self.max_summary_bytes = max_summary_bytes
        self.summary_budget_declines = 0
        self.peak_summary_bytes = self.peak_total_bytes = 0
        self.peak_score_transient_bytes = 0
        self.support = support
        self.output_mode = output_mode
        # v21 wrapper may set these after the parent v20 installer binds.
        self.output_score_precision = 'legacy_bf16_scores'
        self.output_layout = 'head_major'
        self.output_precision_extra_qk_elements_upper_bound = 0
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
        self.preqk_calls = 0
        # v23: None keeps the original M1/M3 behaviour at every call.
        self.bootstrap = None
        self.bootstrap_dense_calls = self.bootstrap_observation_calls = 0
        self.observation_producer = 'repeat_interleave'
        self.route_storage = 'logical'
        self.aligned_score_copies = self.aligned_pad_bytes = self.aligned_sketch_pads = 0
        self.summary_prefix_tiles = self.summary_recomputed_tiles = 0
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
        self.summaries.clear()

    def begin_step(self, canvas, step):
        if canvas != self.canvas:
            self.cache.clear()
            self.valid_keys.clear()
            self.summaries.clear()
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
        native_args = (module, q, k, v, mask)
        native_kwargs = dict(kwargs, dropout=dropout, scaling=scaling, is_causal=is_causal,
                             sliding_window=sliding_window)
        prefix_k, prefix_v, absolute, prefix = self.sources[layer]
        if source_nk != prefix + nq or v.shape != k.shape or b != 1:
            raise ValueError('Unsupported native KV concatenation or batched request')
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError('Frozen BF16 model geometry required')
        causal = bool(is_causal) if is_causal is not None else mask is None and nq > 1
        self._mask_present = mask is not None
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
        if self.bootstrap is not None and self.step <= 1:
            return self._bootstrap_call(native_args, native_kwargs, identity, layer, kind,
                                        q, k, v, mask, scale, causal, window, crop, prefix,
                                        b, h, nq, nk)
        # Arbitrary mask contents are not inferred from pointers/shape.
        plan = self.cache.plan(identity, self.step, force_refresh=mask is not None)
        self.cache.reserve(identity)  # before allocation
        current_for_output = None
        if plan.score_refresh:
            # The replaced entry stays referenced until publish, so the new
            # score tensor transiently coexists with it.
            self.peak_score_transient_bytes = max(
                self.peak_score_transient_bytes, self.cache.storage_bytes + identity.storage_bytes)
            score = self._observe(q, k, mask, scale, causal, window, crop)
            self.cache.publish_scores(identity, self.step, self._store_scores(score))
            valid = torch.isfinite(score).reshape(b, hk, h//hk, nq, nk).any((2, 3))
            self.valid_keys[layer] = valid
            self.score_calls += 1
            self.current_qk_elements += b*h*nq*nk
            self.unsupported_mask_refreshes += int(mask is not None)
            current_for_output = score
        else:
            # Score-cache reuse only. This is NOT avoided current-QK work:
            # routing_only recomputes the whole thing below, and only the
            # preqk mode actually leaves dropped tiles uncomputed.
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
        route_invalid = None
        if plan.decision_refresh:
            valid = self.valid_keys[layer]
            # Leased sketch: reprojects only the aligned suffix (current canvas
            # plus the unaligned boundary) when the encoder-owned prefix source
            # object/version/shape are unchanged from the prior call. A commit
            # or new request invalidates the lease via Sketches' own encoder
            # pre-hook; query-only changes within a canvas never do.
            projected, ref = self.sketches.get(layer, v, valid, prefix - crop)
            threshold = float(self.thresholds[kind]['log_threshold'])
            # Only KV tiles lying WHOLLY before the real prefix boundary are
            # immutable. Compute that from the boundary itself, never from the
            # tensor shape, and never round it up.
            prefix_tiles = max(0, (prefix - crop)) // 64
            summary, store_summary = self._summary_for(
                layer, kind, identity, prefix_tiles, b, h, nq, nk, plan, v.device)
            # The fused route+PV call is only worth issuing when its output is
            # the one we actually return: in cached_scores mode always, and on
            # score anchors where the cached tensor IS the current one, so the
            # same QK is never observed twice.
            fused = (self.output_mode == 'cached_scores'
                     or (current_for_output is entry.scores
                         and self.output_score_precision == 'legacy_bf16_scores'))
            if fused:
                result = attention(entry.scores, v, projected.contiguous(), ref.contiguous(),
                                   sensitivity=self.query_sensitivity,
                                   log_threshold=threshold, trace=self.trace,
                                   summary=summary, store_summary=store_summary,
                                   variant=self.kernel_variant, output_layout=self.output_layout)
                decision = Decision(result.skipped, result.eligible)
            else:
                # Selector only: no discarded PV, no discarded [B,H,Q,D]
                # output. Malformed cached scores stay detectable through the
                # route's own per-tile flag instead of through that PV pass.
                route = self._route(entry.scores, projected, ref,
                                    sensitivity=self.query_sensitivity,
                                    log_threshold=threshold,
                                    summary=summary, store_summary=store_summary,
                                    variant=self.kernel_variant)
                if self.guard_mode == 'fused':
                    route_invalid = route.invalid_tiles     # checked in the final fused guard
                else:
                    torch._assert_async(~route.invalid_tiles.any(),
                                        'Invalid cached scores in routing decision: request must fail')
                decision = Decision(route.skipped, route.eligible)
                # Same retained tiles as the stale-score routing decision;
                # the final softmax/PV consumes current, not cached, scores.
                if self.output_mode == PREQK_MODE:
                    # plan.score_refresh, not tensor identity: aligned16 stores a pitched copy.
                    if plan.score_refresh and self.output_score_precision != 'legacy_bf16_scores':
                        # The anchor's old arithmetic publishes the cache and
                        # chooses support. Current FP32 scores are formed in
                        # the retained-tile consumer as a second QK pass.
                        # Full geometry is a conservative dispatch charge;
                        # sparse retained-tile physical counts are separate.
                        duplicate = b*h*nq*nk
                        self.current_qk_elements += duplicate
                        self.output_precision_extra_qk_elements_upper_bound += duplicate
                    result = self._consume(q, k, v, route.skipped, route.eligible, scale, window, causal)
                    self.preqk_calls += 1
                else:
                    result = attention(current_for_output, v, skipped=route.skipped,
                                       eligible=route.eligible, trace=self.trace,
                                       variant=self.kernel_variant, output_layout=self.output_layout)
            self.cache.publish_decision(identity, self.step, decision)
            self.decision_calls += 1
        elif self.output_mode == PREQK_MODE:
            # Held support, current output, still no dropped-tile QK.
            result = self._consume(q, k, v, entry.decision.skipped, entry.decision.eligible,
                                   scale, window, causal)
            self.preqk_calls += 1
            self.held_calls += 1
        else:
            scores_for_pv = current_for_output if self.output_mode == 'routing_only_current_output' else entry.scores
            result = attention(scores_for_pv, v, skipped=entry.decision.skipped,
                               eligible=entry.decision.eligible, trace=self.trace,
                               variant=self.kernel_variant, output_layout=self.output_layout)
            self.held_calls += 1
        # Explicit asynchronous error, never consume zeroed invalid rows. This
        # is a measured device guard, not a steady-path host tensor read.
        returned = result.output.transpose(1, 2).contiguous()   # no-op for the Hopper model-layout output
        if self.guard_mode == 'fused':
            fused_guard(returned, result.invalid_scores, route_invalid)
        else:
            torch._assert_async(~result.invalid_scores.any(), 'Invalid cached scores: request must fail')
            torch._assert_async(torch.isfinite(result.output).all(), 'Invalid cached-score attention output')
        self.calls += 1
        score_bytes = self.cache.storage_bytes
        self.peak_score_bytes = max(self.peak_score_bytes, score_bytes)
        self.peak_total_bytes = max(self.peak_total_bytes, score_bytes + self.summary_bytes)
        metadata = None if self.telemetry != 'full' else dict(canvas=self.canvas, decoder_call=self.step, layer=layer, kind=kind,
                        query_start=absolute, key_start=identity.key_start, q=nq, k=nk,
                        source_k=source_nk, score_anchor=entry.score_step,
                        score_age=self.step-entry.score_step, score_refresh=plan.score_refresh,
                        decision_refresh=plan.decision_refresh, reason=plan.reason,
                        current_qk_elements=b*h*nq*nk if current_for_output is not None else 0,
                        # Dropped-tile QK is physically skipped only here; the
                        # exact executed/skipped element counts follow from the
                        # bitmap and (q, k) tails, computed offline.
                        qk_mode=('materialized_full' if current_for_output is not None
                                 else (PREQK_MODE if self.output_mode == PREQK_MODE
                                       else 'cached_scores_reuse')))
        if self.telemetry == 'full':
            self.call_metadata.append(metadata)
            # Tiny per-layer physical-work reductions stay on device until finish.
            self.pending.append(((result.skipped & result.eligible).sum(), result.eligible.sum()))
        return returned, None

    def _store_scores(self, score):
        """The tensor the score cache keeps (see ROUTE_STORAGES)."""
        if self.route_storage == 'logical':
            return score
        if self.route_storage != 'aligned16':
            raise ValueError(self.route_storage)
        if self.output_score_precision == 'legacy_bf16_scores' or self.output_mode != PREQK_MODE:
            raise ValueError('aligned16 storage is qualified only for the pre-QK FP32 current-output path')
        b, h, nq, nk = score.shape
        pitch = -(-nk // 16) * 16
        if pitch == nk:
            return score  # already aligned: the identical tensor is stored
        stored = torch.empty((b, h, nq, pitch), device=score.device, dtype=score.dtype)
        stored[..., :nk].copy_(score)
        stored[..., nk:].fill_(-math.inf)
        self.aligned_score_copies += 1
        self.aligned_pad_bytes += stored.numel() * stored.element_size()
        return stored

    def _route(self, scores, projected, ref, **kwargs):
        """route_only on the stored scores; pads only the sketch to the stored pitch."""
        nk, pitch = projected.shape[2], scores.shape[-1]
        if pitch != nk:
            if (self.route_storage != 'aligned16' or pitch % 16 or not 0 < pitch - nk < 16
                    or -(-pitch // 64) != -(-nk // 64)):
                raise ValueError('stored score pitch violates the aligned16 tile/extent invariant')
            projected = torch.nn.functional.pad(projected, (0, 0, 0, pitch - nk))
            self.aligned_sketch_pads += 1
        return route_only(scores, projected.contiguous(), ref.contiguous(), **kwargs)

    def _observe(self, q, k, mask, scale, causal, window, crop):
        if self.observation_producer == 'grouped_q':
            return self.observe_scores_grouped(q, k, mask, scale, causal, window, crop)
        if self.observation_producer != 'repeat_interleave':
            raise ValueError(self.observation_producer)
        return self.observe_scores(q, k, mask, scale, causal, window, crop)

    def _bootstrap_call(self, native_args, native_kwargs, identity, layer, kind,
                        q, k, v, mask, scale, causal, window, crop, prefix, b, h, nq, nk):
        """True native output at canvas calls 0 and 1; call 1 also observes.

        The observation uses the same current Q/K/V/mask as the native output
        and publishes scores and a decision stamped at call 1 for FUTURE calls.
        No sparse output is formed and no discarded PV is issued here.
        """
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        if self.bootstrap not in BOOTSTRAP_POLICIES:
            raise ValueError(self.bootstrap)
        output = sdpa_attention_forward(*native_args, **native_kwargs)
        if self.step == 0:
            self.bootstrap_dense_calls += 1
            return output
        from .cache import Plan
        self.cache.reserve(identity)
        self.peak_score_transient_bytes = max(
            self.peak_score_transient_bytes, self.cache.storage_bytes + identity.storage_bytes)
        score = self._observe(q, k, mask, scale, causal, window, crop)
        stored = self._store_scores(score)
        self.cache.publish_scores(identity, self.step, stored)
        valid = torch.isfinite(score).reshape(b, k.shape[1], h // k.shape[1], nq, nk).any((2, 3))
        self.valid_keys[layer] = valid
        projected, ref = self.sketches.get(layer, v, valid, prefix - crop)
        threshold = float(self.thresholds[kind]['log_threshold'])
        prefix_tiles = max(0, (prefix - crop)) // 64
        plan = Plan(True, True, 'bootstrap_observation', None, None)
        summary, store_summary = self._summary_for(
            layer, kind, identity, prefix_tiles, b, h, nq, nk, plan, v.device)
        route = self._route(stored, projected, ref,
                            sensitivity=self.query_sensitivity, log_threshold=threshold,
                            summary=summary, store_summary=store_summary,
                            variant=self.kernel_variant)
        torch._assert_async(~route.invalid_tiles.any(),
                            'Invalid scores in bootstrap observation: request must fail')
        self.cache.publish_decision(identity, self.step, Decision(route.skipped, route.eligible))
        self.bootstrap_observation_calls += 1
        self.current_qk_elements += b*h*nq*nk
        score_bytes = self.cache.storage_bytes
        self.peak_score_bytes = max(self.peak_score_bytes, score_bytes)
        self.peak_total_bytes = max(self.peak_total_bytes, score_bytes + self.summary_bytes)
        return output

    def _consume(self, q, k, v, skipped, eligible, scale, window, causal):
        if self.consumer == 'triton':
            return preqk_attention(q, k, v, skipped, eligible, scale=scale, window=window,
                                   is_causal=causal, trace=self.trace, variant=self.kernel_variant,
                                   output_score_precision=self.output_score_precision,
                                   output_layout=self.output_layout)
        if causal or self.trace:
            raise ValueError('hopper consumer is qualified for the bidirectional decoder with trace off')
        if self._mask_present:
            raise ValueError('hopper consumer does not consume explicit masks; refusing a silently wrong path')
        out, lse, invalid, _ = self._support.attention(q, k, v, skipped, eligible, scale=scale,
                                                       window=int(window or 0), layout=1)
        return SimpleNamespace(output=out, skipped=skipped, eligible=eligible, invalid_scores=invalid,
                               log_normalizer=lse)

    def _summary_for(self, layer, kind, identity, prefix_tiles, b, h, nq, nk, plan, device):
        """Return (summary, store) for this call's selector.

        Identity is compared by VALUE and carries the score-anchor generation,
        the projection lease's own identity tuple, the prefix boundary and the
        full structural identity of the call. A surviving buffer whose identity
        disagrees is discarded, never reused: a live pointer is not evidence
        that the values behind it are unchanged.
        """
        if self.selector != 'prefix_block_summary' or prefix_tiles <= 0:
            return None, False
        if self.selector_layers == 'local' and kind != 'local':
            return None, False
        lease = self.sketches.entries.get(layer)
        # The summary is bound to the anchor it was built from, so on reuse the
        # stored anchor step must match the entry actually being consumed.
        anchor = self.cache.get(identity).score_step
        wanted = (identity, anchor, None if lease is None else lease['identity'], prefix_tiles)
        held = self.summaries.get(layer)
        if plan.score_refresh:
            qb, kt = (nq + 127) // 128, (nk + 63) // 64
            # Release this layer's old buffers BEFORE allocating the new ones
            # (same-stream caching-allocator reuse is ordered after queued
            # readers), so a rebuild never transiently holds two summaries.
            self.summaries.pop(layer, None)
            needed = summary_nbytes(b, h, qb, prefix_tiles, 32)
            if self.summary_bytes + needed > self.max_summary_bytes:
                # Exact fallback: the legacy recompute path yields the same
                # decisions, so a declined summary costs time, not semantics.
                self.summary_budget_declines += 1
                self.summary_recomputed_tiles += prefix_tiles
                return None, False
            summary = allocate_summary(b, h, qb, kt, prefix_tiles, 32, device, wanted)
            if summary is None:
                return None, False
            if summary.bytes != needed:
                raise AssertionError('summary byte accounting disagrees with allocation')
            self.summaries[layer] = summary
            self.summary_builds += 1
            self.summary_prefix_tiles += prefix_tiles
            self.peak_summary_bytes = max(self.peak_summary_bytes, self.summary_bytes)
            return summary, True
        if held is not None and held.matches(wanted, prefix_tiles):
            self.summary_hits += 1
            self.summary_prefix_tiles += prefix_tiles
            return held, False
        if held is not None:
            self.summaries.pop(layer, None)
        self.summary_misses += 1
        self.summary_recomputed_tiles += prefix_tiles
        return None, False

    @property
    def summary_bytes(self):
        """LIVE resident summary bytes (always derived, never a stale tally)."""
        return sum(s.nbytes for s in self.summaries.values())

    @staticmethod
    def observe_scores(q, k, mask, scale, causal, window, crop):
        """The ONLY current-QK producer in this adapter; called at real anchors."""
        repeated = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        # Both matmul and multiplication preserve Junyu BF16 score transforms.
        scores = torch.matmul(q, repeated.transpose(-1, -2)) * scale
        return Attention._finish_scores(scores, q, k, mask, causal, window, crop)

    @staticmethod
    def observe_scores_grouped(q, k, mask, scale, causal, window, crop):
        """v23 Track E candidate: same BF16 matmul/scale convention without
        materializing K over the GQA group. Query head i = kv*G + j (the
        repeat_interleave order) becomes row j*Q+q of KV head kv; only the
        small Q tensor may be copied. Different GEMM shapes may round
        differently, so exactness is a measured property, never assumed."""
        b, h, nq, d = q.shape
        hk, nk = k.shape[1], k.shape[-2]
        grouped = q.reshape(b, hk, (h // hk) * nq, d)
        scores = torch.matmul(grouped, k.transpose(-1, -2)).view(b, h, nq, nk) * scale
        return Attention._finish_scores(scores, q, k, mask, causal, window, crop)

    @staticmethod
    def _finish_scores(scores, q, k, mask, causal, window, crop):
        cropped_mask = None if mask is None else mask[..., crop:crop+k.shape[-2]]
        valid = _attention_validity(cropped_mask, q, k, is_causal=causal, sliding_window=window)
        if cropped_mask is not None and cropped_mask.dtype != torch.bool:
            if cropped_mask.dtype != torch.bfloat16:
                raise ValueError('Additive score bias requires the frozen BF16 specialization')
            scores = scores + cropped_mask
        return scores.float().masked_fill(~valid, -math.inf).contiguous()

    def records(self):
        if self.telemetry != 'full':
            return None          # N/A in minimal mode: no fabricated per-tile counts
        values = torch.stack([torch.stack(x) for x in self.pending]).cpu().tolist() if self.pending else []
        return [dict(meta, physical_skipped=counts[0], physical_eligible=counts[1],
                     decision_geometry='per-head Q128 KV64')
                for meta, counts in zip(self.call_metadata, values)]

    def counters(self):
        return dict(attention_calls=self.calls, score_refresh_calls=self.score_calls,
                    bootstrap_policy=self.bootstrap,
                    bootstrap_dense_calls=self.bootstrap_dense_calls,
                    bootstrap_observation_calls=self.bootstrap_observation_calls,
                    observation_producer=self.observation_producer,
                    route_storage=self.route_storage,
                    aligned_score_copies=self.aligned_score_copies,
                    aligned_pad_bytes=self.aligned_pad_bytes,
                    aligned_sketch_pads=self.aligned_sketch_pads,
                    score_clock_origin=self.cache.origin,
                    decision_refresh_calls=self.decision_calls, held_decision_calls=self.held_calls,
                    current_qk_elements=self.current_qk_elements, reused_qk_elements=self.reused_qk_elements,
                    projected_current_v_tokens=self.sketches.projected_tokens,
                    reused_current_v_tokens=self.sketches.reused_tokens,
                    peak_score_bytes=self.peak_score_bytes,
                    unsupported_mask_refreshes=self.unsupported_mask_refreshes,
                    support=self.support, output_mode=self.output_mode,
                    selector=self.selector, selector_layers=self.selector_layers,
                    kernel_variant=self.kernel_variant, telemetry=self.telemetry, guard_mode=self.guard_mode,
                    consumer=self.consumer,
                    support_build=(self.support_identity['key'] if self.consumer == 'hopper' else None),
                    per_tile_statistics=('recorded' if self.telemetry == 'full' else
                                         'N/A (minimal telemetry; use a token-matched full audit run)'),
                    summary_builds=self.summary_builds, summary_hits=self.summary_hits,
                    summary_misses=self.summary_misses,
                    summary_resident_bytes=self.summary_bytes,
                    summary_live_bytes=self.summary_bytes,
                    summary_peak_bytes=self.peak_summary_bytes,
                    summary_budget_bytes=self.max_summary_bytes,
                    summary_budget_declines=self.summary_budget_declines,
                    score_live_bytes=self.cache.storage_bytes,
                    score_peak_bytes=self.peak_score_bytes,
                    score_peak_transient_bytes=self.peak_score_transient_bytes,
                    score_budget_bytes=self.cache.max_bytes,
                    history_summary_peak_bytes=self.peak_total_bytes,
                    memory_note=('score and summary budgets are separate; live bytes are '
                                 'at counters() time, peaks over the request'),
                    summary_prefix_tiles_served=self.summary_prefix_tiles,
                    summary_prefix_tiles_recomputed=self.summary_recomputed_tiles,
                    routing_only_extra_qk_elements=self.routing_only_extra_qk_elements,
                    output_precision_extra_qk_elements_upper_bound=self.output_precision_extra_qk_elements_upper_bound,
                    output_score_precision=self.output_score_precision,
                    output_layout=self.output_layout,
                    preqk_consumer_calls=self.preqk_calls,
                    work_counter_scope='dispatch elements, not measured DRAM bytes',
                    cuda_graph_qualified=False, publication='same current CUDA stream')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()
        self.sources.clear()
        self.valid_keys.clear()
        self.summaries.clear()
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
                           max_cache_bytes=config.get('max_cache_bytes', 2 * 1024**3),
                           max_summary_bytes=config.get('max_summary_bytes', 1024**3),
                           support=config.get('support', 'legacy_junyu_mask'),
                           output_mode=config.get('output_mode', 'cached_scores'),
                           selector=config.get('selector', 'legacy_recompute'),
                           selector_layers=config.get('selector_layers', 'local'),
                           kernel_variant=config.get('kernel_variant', 'static'),
                           telemetry=config.get('telemetry', 'full'),
                           guard_mode=config.get('guard_mode', 'separate'),
                           consumer=config.get('consumer', 'triton'),
                           support_build=config.get('support_build'))
        binding.runtime.attention_override = router
        state = NativeReuseState('T', router, m_ref=config['m_ref'], beta=config['beta'],
                                 gamma=config['gamma'], diagnostics=config['diagnostic'],
                                 fast_t=bool(config.get('fast_t', False)))
        yield dict(binding=binding, router=router, state=state, counters=router.counters)
    finally:
        if router is not None:
            router.close()
        if binding is not None:
            binding.close()
        del adapter.model._value_direction_lease
