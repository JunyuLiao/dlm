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
                              route_only, summary_bytes as summary_nbytes, fused_guard, tile_pool)

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
ROUTE_STORAGES = ('logical', 'aligned16', 'aligned16_odd')
# v26 M2: 'exact' is M1's weighted projected value; 'pooled' replaces only mu by
# the per-row mean of current projected V over real legal keys of the tile.
MU_MODES = ('exact', 'pooled', 'pooled_compact')


@dataclass
class Decision:
    skipped: torch.Tensor
    eligible: torch.Tensor


# v27 density gate (named variants): sampler-state-adaptive return to dense attention within a
# canvas. DiffusionGemma stops a canvas when the argmax is stable and the mean token entropy of the
# processed logits falls below 0.005; sparse attention error that keeps the entropy up inflates the
# denoising calls per canvas. The gate watches the SAME statistic (and the accepted-token count) and,
# once it fires, runs the rest of the canvas dense (hysteresis: it never switches back).
DENSITY_GATES = {
    'ent0.05': dict(entropy=0.05), 'ent0.02': dict(entropy=0.02), 'cap12': dict(cap=12),
    # v27: dense only for canvases still running after step 24 (the non-converging tail)
    'cap24': dict(cap=24),
    'stall2': dict(stall=2), 'ent0.05_stall2': dict(entropy=0.05, stall=2),
}


def carried_map(skipped, eligible, prefix, keys):
    """v27 cross-canvas carry: a decision taken when the prefix had ``prefix`` keys, extended to ``keys`` keys.
    Tiles lying wholly in that old prefix keep their decision; every newer tile (the old boundary tile, the
    canvases committed since, the current canvas) is kept."""
    import torch
    tiles, old = -(-keys // 64), prefix // 64
    if old > skipped.shape[-1] or tiles < skipped.shape[-1]:
        raise ValueError('carried map does not fit the current key extent')
    new_skipped = torch.zeros((*skipped.shape[:3], tiles), dtype=torch.bool, device=skipped.device)
    new_eligible = torch.ones_like(new_skipped)
    new_skipped[..., :old] = skipped[..., :old]
    new_eligible[..., :old] = eligible[..., :old]
    return new_skipped, new_eligible


def protect_generated(skipped, prompt_keys):
    """v27 output protection: clear the skip bit of every key tile at or after the tile holding the
    prompt's last key (tile prompt_keys // 64 onward; a tile mixing prompt and generated keys is kept)."""
    if prompt_keys is None or prompt_keys < 0:
        raise ValueError('output protection needs the request prompt extent')
    first = prompt_keys // 64
    if first < skipped.shape[-1]:
        skipped[..., first:] = False
    return skipped


def density_gate_fires(preset, step, entropy, accepted, previous_accepted, stalled_steps):
    """Pure decision for step ``step`` (0-based in the canvas) from the PREVIOUS step's sampler
    statistics. Returns (fire, stalled_steps)."""
    if accepted is not None and previous_accepted is not None and accepted <= previous_accepted:
        stalled_steps += 1
    else:
        stalled_steps = 0
    fire = ((preset.get('entropy') and entropy is not None and entropy < preset['entropy'])
            or (preset.get('cap') and step >= preset['cap'])
            or (preset.get('stall') and stalled_steps >= preset['stall']))
    return bool(fire), stalled_steps


class NativeReuseState(State):
    def begin(self, cur_step, canvas):
        super().begin(cur_step, canvas)
        self.router.begin_step(self.canvas, self.iteration - 1)

    def observe_logits(self, logits, accepted, cur_step):
        super().observe_logits(logits, accepted, cur_step)
        if getattr(self.router, 'density_gate', None) is not None:
            self.router.observe_sampler(logits, accepted)


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
        self.mu_mode = 'exact'
        self.min_route_keys = 0
        self.fused_observe = False
        self.fused_observations = 0
        # v27 fused fresh T (Junyu fresh-T information, not M1): selection inside the output kernel
        self.fresh_fused = False
        self.route_pipeline = False    # v27: pipelined summary-LOAD selector on decision calls
        self.pipelined_routes = 0
        self.risk_state = 'kept'       # v27 M1-DP: 'dense_prefix' (named variant)
        self.risk_budget_shift = None  # v27 risk budget on M1-DP: summed-risk bound (log shift over the threshold)
        self.risk_topk = None          # v27 target sparsity on M1-DP: kept fraction of prefix tiles by risk rank
        # v27 cross-canvas carry (named variant): a layer's last decision of an observed canvas is reused, extended
        # with every newer key tile kept, for the next carry_canvases - 1 canvases -- no dense bootstrap, no
        # observation and no decision there; the canvas after that observes again through the normal path
        self.carry_canvases = None
        self._carry, self._carried_layers, self.carried_calls, self.carry_snapshots = {}, set(), 0, 0
        # v27 observation step (named variant): canvas calls 0..observe_step-1 are true native and the fused
        # observation runs at call observe_step (1 = the original native_bootstrap2_observe1 schedule)
        self.observe_step = 1
        # v27 output protection (named variant): key tiles from the request's prompt end onward (the model's own
        # generated tokens) are never skipped; only wholly-prompt tiles may be dropped. prompt_keys is the GLOBAL
        # prefix length at the request's first canvas (the router is built per request).
        self.protect_output, self.prompt_keys, self.protected_routes = False, None, 0
        self.density_gate = None       # v27 density gate preset (DENSITY_GATES), named variant
        self._fa4_lists, self.fa4_list_builds = [], 0   # v27 FA4 consumer: block lists per keep map
        # v27 async observation route: the observation call's selector (whose decision serves only LATER calls)
        # runs on a side stream, overlapped with the rest of the forward; each layer waits for its own pending
        # route event at its next call's entry. Decisions are bit-identical to the synchronous route.
        self.async_route, self._route_stream, self._pending_routes, self.async_routes = False, None, {}, 0
        self.gate_dense = False
        self.gate_stalled, self.gate_previous_accepted = 0, None
        self.gate_entries, self.gate_dense_calls = [], 0
        self._sampler_entropy = self._sampler_accepted = None
        self.dp_states = {}
        self.dp_builds = self.dp_routes = 0
        self.fresh_fused_calls = 0
        self.fresh_tile_total = None   # device [pv_kept, visited] tiles; summed without host sync
        self.c64_splits = 2
        self.route_layers = None       # v27: routed-layer subset (others native)
        self.share_leader = {}         # v27: follower layer -> leader layer
        self.layer_native_calls = self.shared_calls = self.shared_native_calls = 0
        self.gated_native_calls = 0
        self.compact_pool_builds = 0
        # aligned_pad_bytes (historical name, kept for traceability) = total bytes of
        # newly ALLOCATED pitched buffers, not only the extra pad and not DRAM traffic.
        self.aligned_score_copies = self.aligned_pad_bytes = self.aligned_sketch_pads = 0
        self.aligned_extra_pad_bytes = self.aligned_copy_bytes = 0
        self.peak_score_physical_bytes = 0
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
        if self.carry_canvases:
            # snapshot the finished canvas's decisions (observed layers only) before the cache is cleared
            for layer, entry in list(self.cache.entries.items()):
                if layer in self._carried_layers or getattr(entry, 'decision', None) is None:
                    continue
                if entry.identity.canvas != self.canvas or entry.identity.encoder_epoch != self.epoch:
                    continue
                self._carry[layer] = dict(skipped=entry.decision.skipped, eligible=entry.decision.eligible,
                                          prefix=entry.identity.keys - entry.identity.queries,
                                          canvas=self.canvas, ext=None)
                self.carry_snapshots += 1
            self._carried_layers = set()
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
            self._fa4_lists = []
        if self.density_gate is not None:
            if canvas != self.canvas or step == 0:
                self.gate_dense, self.gate_stalled, self.gate_previous_accepted = False, 0, None
            elif not self.gate_dense and self._sampler_entropy is not None:
                # one host read per step (the native stop check already synchronizes every step)
                entropy, accepted = (float(x) for x in torch.stack([self._sampler_entropy,
                                                                    self._sampler_accepted.float()]).tolist())
                fire, self.gate_stalled = density_gate_fires(self.density_gate, step, entropy, accepted,
                                                             self.gate_previous_accepted, self.gate_stalled)
                self.gate_previous_accepted = accepted
                if fire:
                    self.gate_dense = True
                    self.gate_entries.append(step)
            self._sampler_entropy = self._sampler_accepted = None
        self.canvas, self.step = canvas, step

    def observe_sampler(self, logits, accepted):
        """Mean token entropy of the processed logits (the native stop statistic) and the accepted
        token count, kept on device until the next step's gate decision."""
        x = logits.float()
        logp = torch.log_softmax(x, -1)
        self._sampler_entropy = -(logp.exp() * logp).sum(-1).mean()
        self._sampler_accepted = accepted.sum()

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
        pending = self._pending_routes.pop(layer, None)
        if pending is not None:
            torch.cuda.current_stream().wait_event(pending)
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
        if self.protect_output and self.prompt_keys is None:
            if self.canvas != 0 or crop:
                raise ValueError('output protection must see the first canvas of the request on an uncropped layer')
            self.prompt_keys = prefix
        if self.min_route_keys and nk < self.min_route_keys:
            # v27 length gate: below the frozen key extent the routed path has no
            # measured saving, so the call is exactly native SDPA (no cache, no bitmap).
            from transformers.integrations.sdpa_attention import sdpa_attention_forward
            self.gated_native_calls += 1
            return self._dense_native(native_args, native_kwargs, q, k, v, scale, window, causal)
        if self.route_layers is not None and layer not in self.route_layers:
            # v27 layer subset: this routed-scope layer stays exact native SDPA.
            from transformers.integrations.sdpa_attention import sdpa_attention_forward
            self.layer_native_calls += 1
            return self._dense_native(native_args, native_kwargs, q, k, v, scale, window, causal)
        if self.gate_dense:
            # v27 density gate fired in this canvas: the rest of the canvas runs dense (64-row kernel)
            self.gate_dense_calls += 1
            return self._dense_native(native_args, native_kwargs, q, k, v, scale, window, causal)
        leader = self.share_leader.get(layer, layer)
        if leader != layer:
            # v27 cross-layer sharing: a follower consumes its leader's CURRENT-call
            # support on its own current Q/K/V (no score observation, no selector).
            if self.bootstrap is not None and self.step <= self.observe_step:
                from transformers.integrations.sdpa_attention import sdpa_attention_forward
                self.shared_native_calls += 1
                return self._dense_native(native_args, native_kwargs, q, k, v, scale, window, causal)
            lead = self.cache.entries.get(leader)
            if (lead is None or lead.decision is None or lead.identity.canvas != self.canvas or
                    lead.identity.encoder_epoch != self.epoch or lead.identity.keys != nk or
                    lead.identity.queries != nq or lead.identity.query_heads != h or
                    lead.decision_step is None or not 0 <= self.step - lead.decision_step or
                    lead.identity.mask_signature != signature):
                raise ValueError('shared-support follower has no compatible leader decision this call')
            result = self._consume(q, k, v, lead.decision.skipped, lead.decision.eligible, scale, window, causal)
            returned = result.output.transpose(1, 2).contiguous()
            torch._assert_async(~result.invalid_scores.any(), 'Invalid shared-support attention: request must fail')
            torch._assert_async(torch.isfinite(result.output).all(), 'Invalid shared-support attention output')
            self.shared_calls += 1
            self.preqk_calls += 1
            return returned, None
        if self.fresh_fused:
            return self._fresh_fused_call(layer, kind, q, k, v, scale, causal, window, prefix - crop, b, hk, nk)
        if self.carry_canvases and layer in self._carry:
            c = self._carry[layer]
            age = self.canvas - c['canvas']
            # valid only as a continuation of the same request: the prefix grew by exactly one canvas per step
            if crop or age < 1 or (nk - nq) != c['prefix'] + nq * age:
                del self._carry[layer]
            elif age < self.carry_canvases:
                ext = c['ext']
                if ext is None or ext[0] != (self.canvas, nk):
                    ext = c['ext'] = ((self.canvas, nk), *carried_map(c['skipped'], c['eligible'], c['prefix'], nk))
                self._carried_layers.add(layer)
                result = self._consume(q, k, v, ext[1], ext[2], scale, window, causal)
                returned = result.output.transpose(1, 2).contiguous()
                torch._assert_async(torch.isfinite(result.output).all(), 'Invalid carried-map attention output')
                self.carried_calls += 1
                self.preqk_calls += 1
                return returned, None
        if self.bootstrap is not None and self.step <= self.observe_step:
            return self._bootstrap_call(native_args, native_kwargs, identity, layer, kind,
                                        q, k, v, mask, scale, causal, window, crop, prefix,
                                        b, h, nq, nk)
        # Arbitrary mask contents are not inferred from pointers/shape.
        plan = self.cache.plan(identity, self.step, force_refresh=mask is not None)
        self.cache.reserve(identity)  # before allocation
        current_for_output = None
        if plan.score_refresh:
            # The replaced entry stays referenced until publish, so the new
            # score tensor transiently coexists with it (and, for aligned16,
            # with the logical producer output being copied from).
            self._reserve_observation(identity, b, h, nq, nk)
            score = self._observe(q, k, mask, scale, causal, window, crop)
            self.cache.publish_scores(identity, self.step, self._store_scores(score))
            # Physical residency changes only at publication; no per-call host work.
            self.peak_score_physical_bytes = max(self.peak_score_physical_bytes, self.cache.physical_bytes)
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
            if fused and self.mu_mode != 'exact':
                raise ValueError('pooled mu is qualified only on the route_only pre-QK path')
            if fused and self.protect_output:
                raise ValueError('output protection is qualified only on the route_only pre-QK path')
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
                route = self._route(entry.scores, projected, ref, valid=valid,
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

    def _pitch(self, nk):
        if self.route_storage == 'aligned16' or (self.route_storage == 'aligned16_odd' and nk % 2):
            return -(-nk // 16) * 16
        return nk

    def _reserve_observation(self, identity, b, h, nq, nk):
        """Physical budget check and transient-peak bound, before any allocation."""
        logical_new = b * h * nq * nk * 4
        pitch = self._pitch(nk)
        stored_new = b * h * nq * pitch * 4
        self.cache.reserve_physical(identity, stored_new)
        # Old entry + logical producer output + (if copied) the new pitched buffer.
        transient = self.cache.physical_bytes + logical_new + (stored_new if pitch != nk else 0)
        self.peak_score_transient_bytes = max(self.peak_score_transient_bytes, transient)

    def _store_scores(self, score):
        """The tensor the score cache keeps (see ROUTE_STORAGES)."""
        if self.route_storage == 'logical':
            return score
        if self.route_storage not in ('aligned16', 'aligned16_odd'):
            raise ValueError(self.route_storage)
        if self.route_storage == 'aligned16_odd' and score.shape[-1] % 2 == 0:
            return score  # v26 odd_only policy: KDIV>=2 keeps logical storage
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
        self.aligned_extra_pad_bytes += b * h * nq * (pitch - nk) * stored.element_size()
        self.aligned_copy_bytes += score.numel() * score.element_size()
        return stored

    def _route(self, scores, projected, ref, valid=None, key_offset=None, **kwargs):
        """route_only on the stored scores; pads only the sketch to the stored pitch."""
        nk, pitch = projected.shape[2], scores.shape[-1]
        if key_offset is None and pitch < nk and getattr(self, 'fused_observe', False):
            # a fused-observation tail: keys [key_offset, K) only; the prefix comes from the summary
            summary = kwargs.get('summary')
            if summary is None or kwargs.get('store_summary'):
                raise ValueError('fused-observation tail scores need the held prefix summary')
            key_offset = summary.prefix_tiles * 64
        if key_offset:
            kwargs['key_offset'] = key_offset
            kwargs.setdefault('num_stages', 3)   # tail/LOAD route: prefetch the next tile's summary
            pitch = nk
        if (self.route_pipeline and kwargs.get('summary') is not None and not kwargs.get('store_summary')
                and kwargs.get('variant') == 'generic'):
            # v27 pipelined summary-LOAD selector (bit-identical decisions; v27_route.py)
            kwargs.update(pipelined=True, num_stages=3)
            self.pipelined_routes += 1
        pool = dict(pool=self.mu_mode == 'pooled')
        if self.mu_mode == 'pooled_compact':
            # Built from the UNPADDED current projection over real legal keys;
            # alignment padding never enters the mean (tile extents are equal).
            pooled, count = tile_pool(projected, valid)
            pool = dict(pool='compact', pooled=pooled, pool_count=count)
            self.compact_pool_builds += 1
        if pitch != nk:
            if (self.route_storage not in ('aligned16', 'aligned16_odd') or pitch % 16 or not 0 < pitch - nk < 16
                    or -(-pitch // 64) != -(-nk // 64)):
                raise ValueError('stored score pitch violates the aligned16 tile/extent invariant')
            projected = torch.nn.functional.pad(projected, (0, 0, 0, pitch - nk))
            self.aligned_sketch_pads += 1
        if self.mu_mode not in MU_MODES:
            raise ValueError(self.mu_mode)
        if (self.risk_state == 'dense_prefix' and kwargs.get('summary') is not None
                and kwargs.get('variant') == 'generic'):
            route = self._route_dense_prefix(scores, projected.contiguous(), ref.contiguous(), pool, kwargs)
        else:
            route = route_only(scores, projected.contiguous(), ref.contiguous(), **pool, **kwargs)
        if getattr(route, 'pool_mismatch', None) is not None:
            torch._assert_async(~route.pool_mismatch.any(),
                                'compact pooled mu: row-varying legal key set; request must fail')
        if self.protect_output:
            protect_generated(route.skipped, self.prompt_keys)
            self.protected_routes += 1
        return route

    def _route_dense_prefix(self, scores, projected, ref, pool, kwargs):
        """v27 M1-DP (named variant): the M1 risk against the dense prefix state, precomputed
        once per prefix summary; decisions differ from M1's kept-state scan by construction."""
        from . import v27_dense_prefix as dp
        summary = kwargs['summary']
        if kwargs.get('store_summary'):
            # anchor: fill the summary with the unchanged STORE pass (its decisions are discarded)
            route_only(scores, projected, ref, **pool, **dict(kwargs, pipelined=False))
        live = {s.identity for s in self.summaries.values()}
        self.dp_states = {k: v for k, v in self.dp_states.items() if k in live}
        state = self.dp_states.get(summary.identity)
        if state is None:
            nq = scores.shape[2]
            kt = -(-(scores.shape[-1] + kwargs.get('key_offset', 0)) // 64)
            state = dp.build(summary, nq, kt, projected.shape[1],
                             pooled=pool.get('pooled') if pool.get('pool') == 'compact' else None)
            self.dp_states[summary.identity] = state
            self.dp_builds += 1
        self.dp_routes += 1
        budget = (None if self.risk_budget_shift is None
                  else float(kwargs['log_threshold']) + float(self.risk_budget_shift))
        return dp.route(scores, projected, ref, state, sensitivity=kwargs.get('sensitivity'),
                        log_threshold=kwargs['log_threshold'], key_offset=kwargs.get('key_offset', 0),
                        risk_budget=budget, risk_topk=self.risk_topk, **pool)

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
        if self.fused_observe and self.step == self.observe_step:
            return self._fused_bootstrap_observation(identity, layer, kind, q, k, v, mask, scale,
                                                     causal, window, crop, prefix, b, h, nq, nk)
        output = self._dense_native(native_args, native_kwargs, q, k, v, scale, window, causal)
        if self.step < self.observe_step:
            self.bootstrap_dense_calls += 1
            return output
        from .cache import Plan
        self.cache.reserve(identity)
        self._reserve_observation(identity, b, h, nq, nk)
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
        route = self._route(stored, projected, ref, valid=valid,
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
        self.peak_score_physical_bytes = max(self.peak_score_physical_bytes, self.cache.physical_bytes)
        self.peak_total_bytes = max(self.peak_total_bytes, score_bytes + self.summary_bytes)
        return output

    def _fused_bootstrap_observation(self, identity, layer, kind, q, k, v, mask, scale, causal,
                                     window, crop, prefix, b, h, nq, nk):
        """v27 fused observation (call 1): ONE dense pass returns the dense output and writes
        the M1 prefix summaries plus the compact observation-score tail. No full [B,H,Q,K]
        score tensor and no separate route STORE pass exist. The decision for FUTURE calls
        is formed by the route in LOAD mode on those summaries and the tail."""
        from .cache import Plan
        from .v27_consumer64 import fused_observe
        if (mask is not None or causal or window is not None or self.consumer not in ('triton64', 'fa4')
                or self.mu_mode not in ('exact', 'pooled_compact') or b != 1):
            raise ValueError('fused observation is qualified for bidirectional GLOBAL, triton64, exact/compact mu')
        self.cache.reserve(identity)
        hk = k.shape[1]
        valid = torch.ones((b, hk, nk), dtype=torch.bool, device=q.device)   # native legal: every real key
        self.valid_keys[layer] = valid
        projected, ref = self.sketches.get(layer, v, valid, prefix - crop)
        prefix_tiles = max(0, (prefix - crop)) // 64
        rank = 0 if self.mu_mode == 'pooled_compact' else 32
        qb, kt = (nq + 127) // 128, (nk + 63) // 64
        self.summaries.pop(layer, None)
        summary = allocate_summary(b, h, qb, kt, prefix_tiles, rank, q.device, None) if prefix_tiles else None
        if summary is None:
            raise ValueError('fused observation needs at least one wholly-prefix tile')
        # Named variant: BF16 inputs for the rank-32 mu estimate (1.9 vs 3.4 ms at 17.5K keys).
        output, tail = fused_observe(q, k, v, projected.contiguous(), scale, prefix_tiles, summary,
                                     splits=self.c64_splits, mu=rank > 0, mu_precision='bf16')
        self.cache.publish_scores(identity, self.step, tail)
        lease = self.sketches.entries.get(layer)
        summary.identity = (identity, self.step, None if lease is None else lease['identity'], prefix_tiles)
        self.summaries[layer] = summary
        self.summary_builds += 1
        self.summary_prefix_tiles += prefix_tiles
        self.peak_summary_bytes = max(self.peak_summary_bytes, self.summary_bytes)
        threshold = float(self.thresholds[kind]['log_threshold'])
        route_kwargs = dict(valid=valid, sensitivity=self.query_sensitivity, log_threshold=threshold,
                            summary=summary, store_summary=False, variant=self.kernel_variant,
                            key_offset=prefix_tiles * 64)
        if self.async_route:
            main = torch.cuda.current_stream()
            if self._route_stream is None:
                self._route_stream = torch.cuda.Stream(device=q.device)
            side = self._route_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                route = self._route(tail, projected, ref, **route_kwargs)
                torch._assert_async(~route.invalid_tiles.any(),
                                    'Invalid scores in fused observation: request must fail')
            done = torch.cuda.Event()
            done.record(side)
            # the side stream reads these main-stream tensors; the main stream later reads the decision
            for tensor in (tail, projected, ref, valid, summary.z, summary.mu, summary.active, summary.bad):
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(side)
            for tensor in (route.skipped, route.eligible):
                tensor.record_stream(main)
            self._pending_routes[layer] = done
            self.async_routes += 1
        else:
            route = self._route(tail, projected, ref, **route_kwargs)
            torch._assert_async(~route.invalid_tiles.any(), 'Invalid scores in fused observation: request must fail')
        torch._assert_async(torch.isfinite(output).all(), 'Invalid fused-observation output')
        self.cache.publish_decision(identity, self.step, Decision(route.skipped, route.eligible))
        self.bootstrap_observation_calls += 1
        self.fused_observations += 1
        self.current_qk_elements += b*h*nq*nk
        return output, None

    def _fresh_fused_call(self, layer, kind, q, k, v, scale, causal, window, prefix, b, hk, nk):
        """v27 fused fresh T. Every call (no bootstrap, no score cache, no bitmap): the 64-row
        output kernel forms the CURRENT scores, derives the block log-mass and weighted CURRENT
        projected V per KV64 tile, and issues V load + PV only for tiles the retained-state
        risk keeps. This is Junyu's fresh-T information executed inside the consumer; it is a
        named variant and never a substitute for Fan's M1 (historical QK)."""
        if causal or window is not None or self._mask_present or self.trace:
            raise ValueError('fused fresh T is qualified for bidirectional unmasked GLOBAL only')
        from .v27_consumer64 import consume64_fresh_t
        valid = torch.ones((b, hk, nk), dtype=torch.bool, device=q.device)
        projected, ref = self.sketches.get(layer, v, valid, prefix)
        threshold = float(self.thresholds[kind]['log_threshold'])
        out, counts = consume64_fresh_t(q, k, v, projected, ref, self.query_sensitivity, scale, threshold,
                                        splits=self.c64_splits, mu_precision='tf32')
        torch._assert_async(torch.isfinite(out).all(), 'Invalid fused fresh-T attention output')
        tiles = counts.sum((0, 1, 2))
        self.fresh_tile_total = tiles if self.fresh_tile_total is None else self.fresh_tile_total + tiles
        self.fresh_fused_calls += 1
        self.calls += 1
        return out, None

    def _dense_native(self, native_args, native_kwargs, q, k, v, scale, window, causal):
        """Native dense output; with the v27 64-row consumer, the same kernel all-kept."""
        if self.consumer == 'triton64' and not causal and window is None and self._mask_present is False:
            from .v27_consumer64 import dense64
            return dense64(q, k, v, scale, splits=self.c64_splits), None
        if self.consumer == 'fa4' and not causal and window is None and self._mask_present is False:
            from . import v27_fa4
            return v27_fa4.dense(q, k, v, scale), None
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(*native_args, **native_kwargs)

    def _consume(self, q, k, v, skipped, eligible, scale, window, causal):
        if self.consumer == 'fa4':
            # v27: the official FlashAttention-4 kernel through its block-sparse interface; the M1/M2/M3
            # keep map becomes FA4 full-block lists, built once per map object and reused while held
            if causal or window is not None or self.trace or self.output_layout != 'model_major':
                raise ValueError('fa4 consumer is qualified for bidirectional GLOBAL model-major only')
            from . import v27_fa4
            # keyed by the map OBJECTS (the entry keeps a reference, so a freed map's storage can never be
            # reused under a cached key); a held decision passes the same objects every call
            lists = next((l for s_ref, e_ref, l in self._fa4_lists if s_ref is skipped and e_ref is eligible), None)
            if lists is None:
                lists = v27_fa4.block_sparse_tensors(eligible & ~skipped)
                # >= 2 x the routed layers (G75 S15/S30 hold 30 maps); a smaller cache thrashes every held call
                self._fa4_lists = self._fa4_lists[-63:] + [(skipped, eligible, lists)]
                self.fa4_list_builds += 1
            out = v27_fa4.sparse_lists(q, k, v, lists, scale)
            return SimpleNamespace(output=out.transpose(1, 2), skipped=skipped, eligible=eligible,
                                   invalid_scores=torch.zeros(out.shape[:1] + (out.shape[2], out.shape[1]),
                                                              dtype=torch.bool, device=out.device))
        if self.consumer == 'triton64':
            # v27 64-row / split-KV consumer (named numerical variant). NaN/inf scores
            # propagate to the output and fail the finite-output guard.
            if causal or window is not None or self.trace or self.output_layout != 'model_major':
                raise ValueError('triton64 consumer is qualified for bidirectional GLOBAL model-major only')
            from .v27_consumer64 import consume64
            out = consume64(q, k, v, skipped, eligible, scale, splits=self.c64_splits)
            return SimpleNamespace(output=out.transpose(1, 2), skipped=skipped, eligible=eligible,
                                   invalid_scores=torch.zeros(out.shape[:1] + (out.shape[2], out.shape[1]),
                                                              dtype=torch.bool, device=out.device))
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
            rank = 0 if self.mu_mode == 'pooled_compact' else 32
            needed = summary_nbytes(b, h, qb, prefix_tiles, rank)
            if self.summary_bytes + needed > self.max_summary_bytes:
                # Exact fallback: the legacy recompute path yields the same
                # decisions, so a declined summary costs time, not semantics.
                self.summary_budget_declines += 1
                self.summary_recomputed_tiles += prefix_tiles
                return None, False
            summary = allocate_summary(b, h, qb, kt, prefix_tiles, rank, device, wanted)
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
                    gated_native_calls=self.gated_native_calls, min_route_keys=self.min_route_keys,
                    layer_native_calls=self.layer_native_calls, shared_calls=self.shared_calls,
                    carry_canvases=self.carry_canvases, carried_calls=self.carried_calls,
                    carry_snapshots=self.carry_snapshots, observe_step=self.observe_step,
                    protect_output=self.protect_output, protected_routes=self.protected_routes,
                    fused_observations=self.fused_observations,
                    fresh_fused_calls=self.fresh_fused_calls,
                    pipelined_routes=self.pipelined_routes,
                    risk_state=self.risk_state, dp_builds=self.dp_builds, dp_routes=self.dp_routes,
                    density_gate_dense_calls=self.gate_dense_calls, density_gate_entry_steps=list(self.gate_entries),
                    async_observation_routes=self.async_routes,
                    fa4_list_builds=self.fa4_list_builds,
                    fresh_fused_tiles=(dict(zip(('pv_kept', 'visited'), self.fresh_tile_total.tolist()))
                                       if self.fresh_tile_total is not None else None),
                    shared_native_calls=self.shared_native_calls,
                    bootstrap_observation_calls=self.bootstrap_observation_calls,
                    observation_producer=self.observation_producer,
                    route_storage=self.route_storage,
                    mu_mode=self.mu_mode, compact_pool_builds=self.compact_pool_builds,
                    aligned_score_copies=self.aligned_score_copies,
                    aligned_pad_bytes=self.aligned_pad_bytes,
                    aligned_buffer_bytes_allocated=self.aligned_pad_bytes,
                    aligned_extra_pad_bytes=self.aligned_extra_pad_bytes,
                    aligned_copy_bytes=self.aligned_copy_bytes,
                    score_physical_live_bytes=self.cache.physical_bytes,
                    score_physical_peak_bytes=self.peak_score_physical_bytes,
                    aligned_byte_fields_note=('aligned_pad_bytes == aligned_buffer_bytes_allocated: sum of '
                                              'allocated pitched buffers; extra_pad counts only pitch-K lanes; '
                                              'copy counts logical bytes copied; none is measured DRAM traffic'),
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
