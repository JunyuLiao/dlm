"""Bounded real-state geometry opportunity capture. No scored generations.

Input is the frozen, seven-arm v21 diagnostic config. Only native capture and
M3_R3_new replay run. Large Q/K/V/support tensors never enter the receipt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from scripts import v20_profile as base
from scripts import v21_numerical_diagnostic as diagnostic
from scripts.replay_harness import StepSnapshot, output_digest
from scripts.v9_step_replay_profile import decoder_call, native_registry


def _error(exc, stage):
    frame = traceback.extract_tb(exc.__traceback__)[-1:]
    return dict(stage=stage, error_type=type(exc).__name__,
                code_location=(None if not frame else dict(file=Path(frame[0].filename).name,
                                                           line=frame[0].lineno)))


def _selected(sequence, targets):
    reached = {int(s['call_index']) for s in sequence}
    requested = {int(t['call_index']) for t in targets}
    selected = requested & reached
    if sequence:
        selected.add(int(sequence[-1]['call_index']))
    return sorted(selected), sorted(requested - reached)


@contextmanager
def _spy_native(registry, native, selected, sink):
    if registry['sdpa'] is not native:
        raise RuntimeError('native SDPA registry already overridden')
    def spy(module, q, k, v, *args, **kwargs):
        result = native(module, q, k, v, *args, **kwargs)
        layer = int(getattr(module, 'layer_idx', -1))
        if layer in selected:
            sink[layer] = dict(module=module, q=q.detach().clone(), k=k.detach().clone(),
                               v=v.detach().clone(), args=args, kwargs=kwargs,
                               output=(result[0] if isinstance(result, (tuple, list)) else result).detach().clone())
        return result
    registry['sdpa'] = spy
    try:
        yield
    finally:
        registry['sdpa'] = native


@contextmanager
def _spy_dispatch(registry, selected, sink):
    """Observe untagged native LOCAL calls through the installed dispatcher."""
    dispatcher = registry['sdpa']
    def spy(module, q, k, v, *args, **kwargs):
        result = dispatcher(module, q, k, v, *args, **kwargs)
        layer = int(getattr(module, 'layer_idx', -1))
        if layer in selected:
            sink[layer] = dict(module=module, q=q.detach().clone(), k=k.detach().clone(),
                               v=v.detach().clone(), args=args, kwargs=kwargs,
                               output=(result[0] if isinstance(result, (tuple, list)) else result).detach().clone())
        return result
    registry['sdpa'] = spy
    try:
        yield
    finally:
        registry['sdpa'] = dispatcher


@contextmanager
def _spy_method(runtime, selected, sink):
    binding, router = runtime['binding'], runtime['router']
    owner = getattr(router, 'owner', router)
    original = binding.runtime.attention_override
    class Spy:
        def __call__(self, module, q, k, v, *args, **kwargs):
            result = original(module, q, k, v, *args, **kwargs)
            layer = int(getattr(module, 'layer_idx', -1))
            if layer in selected:
                entry = owner.cache.entries.get(layer) if layer == 5 else None
                if layer == 5 and (entry is None or entry.decision is None):
                    raise RuntimeError('M3 GLOBAL decision absent after original attention call')
                sink[layer] = dict(module=module, q=q.detach().clone(), k=k.detach().clone(),
                                   v=v.detach().clone(), args=args, kwargs=kwargs,
                                   output=(result[0] if isinstance(result, (tuple, list)) else result).detach().clone(),
                                   scores=(entry.scores.detach().clone() if entry else None),
                                   skipped=(entry.decision.skipped.detach().clone() if entry else None),
                                   eligible=(entry.decision.eligible.detach().clone() if entry else None),
                                   score_step=(entry.score_step if entry else None),
                                   decision_step=(entry.decision_step if entry else None),
                                   score_age=(owner.step-entry.score_step if entry else None),
                                   decision_age=(owner.step-entry.decision_step if entry else None),
                                   production_norm=(owner.sketches.entries[layer]['norm'].detach().clone()
                                                    if layer == 5 and layer in owner.sketches.entries else None),
                                   sensitivity=(owner.query_sensitivity.detach().clone()
                                                if owner.query_sensitivity is not None else None))
            return result
        def __getattr__(self, name):
            return getattr(original, name)
    binding.runtime.attention_override = Spy()
    try:
        yield owner
    finally:
        binding.runtime.attention_override = original


def _mask_contract(record):
    kwargs, args = record['kwargs'], record['args']
    mask = args[0] if args else kwargs.get('attention_mask')
    if mask is not None or kwargs.get('is_causal') is not False:
        raise ValueError('only explicit bidirectional native mask=None supports comparable attention timing')
    scale = kwargs.get('scaling')
    scale = float(scale) if scale is not None else record['q'].shape[-1] ** -.5
    if not 0 < scale < float('inf'):
        raise ValueError('invalid observed attention scale')
    return scale


def _event(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(end)), (time.perf_counter()-wall)*1000


def _timed_attention(record, owner, native):
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention
    scale = _mask_contract(record)
    q, k, v = (record[key] for key in ('q', 'k', 'v'))
    module, args, kwargs = (record[key] for key in ('module', 'args', 'kwargs'))
    funcs = {'native_attention': lambda: native(module, q, k, v, *args, **kwargs)}
    if record.get('skipped') is not None:
        if getattr(owner, 'consumer', None) != 'triton':
            raise ValueError('fixed-support diagnostic requires existing Triton consumer')
        funcs['coarse_preqk_fixed_support'] = lambda: preqk_attention(
            q, k, v, record['skipped'], record['eligible'], scale=scale,
            is_causal=False, window=None, variant=owner.kernel_variant,
            output_score_precision=owner.output_score_precision,
            output_layout=owner.output_layout)
    for fn in funcs.values():
        fn()  # specialization/warmup, outside accepted event samples
    rows = {name: [] for name in funcs}
    with base.no_compile_during_accepted() as misses:
        for repeat in range(3):
            order = list(funcs) if repeat % 2 == 0 else list(reversed(funcs))
            for name in order:
                result, event_ms, wall_ms = _event(funcs[name])
                value = result.output if name != 'native_attention' else (
                    result[0] if isinstance(result, (tuple, list)) else result)
                rows[name].append(dict(event_ms=event_ms, synchronized_wall_ms=wall_ms,
                                       output_digest=output_digest(value)))
    if misses:
        raise RuntimeError(f'JIT specialization during accepted attention timing: {misses}')
    if any(len({r['output_digest'] for r in samples}) != 1 for samples in rows.values()):
        raise RuntimeError('attention output drift during repeated fixed-state timing')
    return dict(status='qualified', boundary='complete_attention_call_only',
                input_path='M3_R3_new_same_history_QKV; native attention is called directly on these method QKV',
                full_model_forward_claim=False, warmup=1,
                repetitions=3, native_bracket_rotation=True, samples=rows,
                note='QKV/support prepared; excludes model, route, sampler and projection')


def _project_full(owner, record, legal):
    q, k, v = (record[key] for key in ('q', 'k', 'v'))
    b, h, nq, nk = q.shape[0], q.shape[1], q.shape[2], k.shape[-2]
    hk, width = v.shape[1], v.shape[-1]
    if v.shape[-2] != nk:
        raise ValueError('historical score/KV extent mismatch')
    valid = legal.expand(b, h, nq, nk).reshape(b, hk, h//hk, nq, nk).any((2, 3))
    matrix = owner.projections.get(int(record['module'].layer_idx), hk, width,
                                   'gaussian', 32, 1729, v.device)
    def project():
        values = v.float()
        projected = torch.matmul(values, matrix)
        # Match Sketches.get's actual rounding order, not sum(square(V)).
        norm2 = values.norm(dim=-1).square()
        reference = (norm2.masked_fill(~valid, 0.).sum(-1) /
                     valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
        return projected.contiguous(), reference.contiguous()
    (z, ref), event_ms, wall_ms = _event(project)
    parity = None
    production_norm = record.get('production_norm')
    if production_norm is not None:
        cached = (production_norm.masked_fill(~valid, 0.).sum(-1) /
                  valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
        parity = dict(bitwise_equal=bool(torch.equal(ref, cached)),
                      max_abs=float((ref-cached).abs().max()))
    return z, ref, dict(event_ms=event_ms, synchronized_wall_ms=wall_ms,
                         meaning='full current-V projection and valid-key RMS; diagnostic, not leased production cost',
                         valid_key_count=int(valid.sum()), cached_norm_reference_comparison=parity)


def _geometry(owner, record, layer, *, matched_screen=False):
    from dllm.attention.blasst.core import _attention_validity
    from experiments.numerical_qk_reuse.geometry_diagnostic import compare_geometries
    from experiments.numerical_qk_reuse.integration import Attention
    scale = _mask_contract(record)
    q, k, v = (record[key] for key in ('q', 'k', 'v'))
    if layer == 5:
        scores = record['scores']
        score_source = 'historical_M3_cache'
    else:
        scores = Attention.observe_scores(q, k, None, scale, False, None, 0)
        score_source = 'current_score_oracle_LOCAL_not_historical'
    sensitivity = record['sensitivity']
    if sensitivity is None:
        # The qualified production route maps None to neutral all-ones at
        # bootstrap (cached_executor.py route_only/attention). Preserve this
        # exact convention; do not invent a sensitivity history.
        sensitivity = torch.ones((q.shape[0], q.shape[2]), device=q.device,
                                 dtype=torch.float32)
        sensitivity_source = 'production_none_means_neutral_ones'
    else:
        sensitivity_source = 'live_T'
    legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    z, ref, projection = _project_full(owner, record, legal)
    kind = 'GLOBAL' if layer == 5 else 'LOCAL'
    threshold = float(owner.thresholds[kind.lower()]['log_threshold'])
    comparison = compare_geometries(scores, z, ref, sensitivity, q, k, v,
                                    scale=scale, threshold=threshold, kind=kind, legal=legal)
    if matched_screen:
        from experiments.numerical_qk_reuse.geometry_diagnostic import matched_q16_screen
        comparison['matched_q16_calibration'] = matched_q16_screen(
            scores, z, ref, sensitivity, q, k, v, scale=scale,
            threshold=threshold, kind='GLOBAL', legal=legal)
    return dict(score_source=score_source, threshold=threshold, scale=scale,
                query_sensitivity=sensitivity_source,
                score_age=record.get('score_age'),
                decision_age=record.get('decision_age'),
                source_qkv_digest=output_digest((q, k, v)),
                projection=projection, comparison=comparison)


def _layer_kinds(model):
    """GLOBAL/LOCAL identity from the loaded attention modules, not from a formula."""
    kinds = {}
    for module in model.modules():
        layer = getattr(module, 'layer_idx', None)
        if layer is None or not hasattr(module, 'is_sliding'):
            continue
        kind = 'LOCAL' if bool(module.is_sliding) else 'GLOBAL'
        # Encoder and decoder attention modules share an index; they must agree.
        if kinds.setdefault(int(layer), kind) != kind:
            raise RuntimeError(f'conflicting attention kind at layer {layer}')
    counts = {kind: sum(value == kind for value in kinds.values()) for kind in ('GLOBAL', 'LOCAL')}
    if counts != {'GLOBAL': 5, 'LOCAL': 25} or kinds.get(0) != 'LOCAL' or kinds.get(5) != 'GLOBAL':
        raise RuntimeError(f'unexpected attention layer identity {counts}')
    return kinds


def _attention_share(model, snapshot, registry, native_fn, kinds, reps=3):
    """Native complete-forward price with attention calls timed in place, plus
    timing-only oracles whose GLOBAL/LOCAL/all attention returns zeros.

    The oracles bound what ANY support policy could save on that layer class at
    this state (zero selection and zero consumer cost). Their logits are
    invalid and are never used for anything but elapsed time.
    """
    variants = ('native', 'no_global_attention', 'no_local_attention', 'no_attention')
    dropped = {'native': (), 'no_global_attention': ('GLOBAL',),
               'no_local_attention': ('LOCAL',), 'no_attention': ('GLOBAL', 'LOCAL')}
    templates, spans = {}, []

    def spy_for(name):
        def spy(module, q, k, v, *args, **kwargs):
            layer = int(module.layer_idx)
            if kinds[layer] in dropped[name]:
                return (torch.zeros_like(templates[layer]), None)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = native_fn(module, q, k, v, *args, **kwargs)
            end.record()
            if name == 'native':
                spans.append((kinds[layer], start, end))
                value = result[0] if isinstance(result, (tuple, list)) else result
                templates.setdefault(layer, torch.empty_like(value))
            return result
        return spy

    def forward(name):
        kw = base.prepare_step(snapshot, None)  # restoration outside the timer
        registry['sdpa'] = spy_for(name)
        try:
            spans.clear()
            logits, event_ms, wall_ms = _event(lambda: decoder_call(model, kw))
            per_kind = ({kind: sum(s.elapsed_time(e) for k, s, e in spans if k == kind)
                         for kind in ('GLOBAL', 'LOCAL')} if name == 'native' else None)
            return dict(event_ms=event_ms, synchronized_wall_ms=wall_ms,
                        output_digest=output_digest(logits), attention_event_ms=per_kind,
                        attention_calls=len(spans) if name == 'native' else None)
        finally:
            registry['sdpa'] = native_fn

    if registry['sdpa'] is not native_fn:
        raise RuntimeError('attention-share pass must start from the native registry')
    forward('native')  # warm native and build output templates for the oracles
    if set(templates) != set(kinds):
        raise RuntimeError('native pass did not visit every attention layer')
    for name in variants[1:]:
        forward(name)
    rows = {name: [] for name in variants}
    with base.no_compile_during_accepted() as misses:
        for repeat in range(reps):
            order = variants if repeat % 2 == 0 else tuple(reversed(variants))
            for name in order:
                rows[name].append(forward(name))
    templates.clear()
    if misses:
        raise RuntimeError(f'JIT specialization during accepted share timing: {misses}')
    if any(len({r['output_digest'] for r in samples}) != 1 for samples in rows.values()):
        raise RuntimeError('forward output drift across repeated same-state timing')
    median = lambda values: sorted(values)[len(values)//2]
    native = median([r['event_ms'] for r in rows['native']])
    summary = {name: dict(median_event_ms=median([r['event_ms'] for r in rows[name]]),
                          ratio_to_native=median([r['event_ms'] for r in rows[name]])/native)
               for name in variants}
    for kind in ('GLOBAL', 'LOCAL'):
        summary['native'][f'{kind}_attention_in_forward_median_ms'] = median(
            [r['attention_event_ms'][kind] for r in rows['native']])
    return dict(status='qualified', boundary='complete_model_forward_logits', reps=reps,
                rotation='alternating forward/reversed variant order', warmup=1,
                samples=rows, summary=summary,
                oracle_note='zero-attention variants are timing oracles with invalid logits')


def _validation_geometry(owner, record):
    """Frozen five-offset Q16 screen on a GLOBAL state; selection is not redone here."""
    from dllm.attention.blasst.core import _attention_validity
    from experiments.numerical_qk_reuse.geometry_diagnostic import matched_q16_screen
    scale = _mask_contract(record)
    q, k, v = (record[key] for key in ('q', 'k', 'v'))
    sensitivity, source = record['sensitivity'], 'live_T'
    if sensitivity is None:
        sensitivity = torch.ones((q.shape[0], q.shape[2]), device=q.device, dtype=torch.float32)
        source = 'production_none_means_neutral_ones'
    legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    z, ref, projection = _project_full(owner, record, legal)
    threshold = float(owner.thresholds['global']['log_threshold'])
    screen = matched_q16_screen(record['scores'], z, ref, sensitivity, q, k, v, scale=scale,
                                threshold=threshold, kind='GLOBAL', legal=legal)
    return dict(score_source='historical_M3_cache', threshold=threshold, scale=scale,
                query_sensitivity=source, score_age=record.get('score_age'),
                decision_age=record.get('decision_age'),
                source_qkv_digest=output_digest((q, k, v)), projection=projection,
                comparison=dict(matched_q16_calibration=screen))


def _observation_parity(owner, record, reps=5):
    """v23 Track E: legacy repeat_interleave producer versus grouped-Q producer
    on one real GLOBAL state. Bitwise score identity (including -inf and NaN
    payloads), route bitmap identity, event time and peak transient memory."""
    from dllm.attention.blasst.core import _attention_validity
    from experiments.numerical_qk_reuse.cached_executor import route_only
    from experiments.numerical_qk_reuse.integration import Attention
    scale = _mask_contract(record)
    q, k = record['q'], record['k']
    producers = {'repeat_interleave': Attention.observe_scores,
                 'grouped_q': Attention.observe_scores_grouped}
    call = {name: (lambda f=f: f(q, k, None, scale, False, None, 0)) for name, f in producers.items()}
    outputs = {name: fn() for name, fn in call.items()}   # warmup and values
    legacy, grouped = outputs['repeat_interleave'], outputs['grouped_q']
    bits_l, bits_g = legacy.view(torch.int32), grouped.view(torch.int32)
    differing = int((bits_l != bits_g).sum())
    finite = torch.isfinite(legacy) & torch.isfinite(grouped)
    max_abs = float((legacy-grouped).abs().masked_fill(~finite, 0).max()) if differing else 0.
    legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    z, ref, _ = _project_full(owner, record, legal)
    sensitivity = record['sensitivity']
    if sensitivity is None:
        sensitivity = torch.ones((q.shape[0], q.shape[2]), device=q.device, dtype=torch.float32)
    threshold = float(owner.thresholds['global']['log_threshold'])
    routes = {name: route_only(value, z, ref, sensitivity=sensitivity.contiguous(),
                               log_threshold=threshold, variant=owner.kernel_variant)
              for name, value in outputs.items()}
    bitmap_equal = bool(torch.equal(routes['repeat_interleave'].skipped, routes['grouped_q'].skipped)
                        and torch.equal(routes['repeat_interleave'].eligible, routes['grouped_q'].eligible))
    del outputs, legacy, grouped, bits_l, bits_g, routes
    samples = {name: [] for name in call}
    memory = {}
    with base.no_compile_during_accepted() as misses:
        for repeat in range(reps):
            order = list(call) if repeat % 2 == 0 else list(reversed(list(call)))
            for name in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                value, event_ms, wall_ms = _event(call[name])
                memory[name] = max(memory.get(name, 0), torch.cuda.max_memory_allocated()-before)
                samples[name].append(dict(event_ms=event_ms, synchronized_wall_ms=wall_ms))
                del value
    if misses:
        raise RuntimeError(f'JIT specialization during accepted producer timing: {misses}')
    median = lambda values: sorted(values)[len(values)//2]
    return dict(status='qualified', scores_bitwise_equal=(differing == 0),
                differing_score_elements=differing, max_abs_finite_difference=max_abs,
                route_bitmap_equal=bitmap_equal,
                shape=dict(query_heads=q.shape[1], kv_heads=k.shape[1], queries=q.shape[2],
                           keys=k.shape[-2], head_dim=q.shape[-1]),
                repeated_k_bytes=k.numel()*k.element_size()*(q.shape[1]//k.shape[1]),
                peak_transient_bytes=memory,
                median_event_ms={name: median([r['event_ms'] for r in rows]) for name, rows in samples.items()},
                samples=samples, reps=reps,
                boundary='score producer only (anchor observation), not a complete A call')


def _bootstrap_arm(selected_arm, parent_arm, arm_name, route_storage='logical', panel_selector=False):
    """Frozen v23 bootstrap arm rebuilt through the v20/v21 config builders
    from the bound M3 arm's own base (same model, policy, sources, numerics)."""
    from experiments.numerical_qk_reuse import v20, v21
    wrapped = selected_arm['config']
    base_config = dict(wrapped['parent_config'])
    for key in ('fingerprint', 'condition', 'plugin', 'v20_arm', 'v20_scope',
                'decision_interval', 'score_refresh_period', 'output_mode'):
        base_config.pop(key, None)
    if parent_arm == 'B_A8_matched':
        base_config['selector'] = 'legacy_recompute'
    elif panel_selector:
        # Same selector as the scored bootstrap6 panel binder (M1/M3 use stored prefix summaries).
        base_config.update(selector='prefix_block_summary', selector_layers='all')
    config = v21.effective_config(base_config, parent_arm, v20.GLOBAL_ONLY_NATIVE_LOCAL,
                                  output_score_precision=wrapped['output_score_precision'],
                                  output_layout=wrapped['output_layout'],
                                  bootstrap_policy='native_bootstrap2_observe1',
                                  observation_producer='grouped_q', route_storage=route_storage)
    return dict(selected_arm, name=arm_name, condition=config['condition'], config=config)


def _bootstrap_replay(model, sequence, runtime, upto):
    """Per-call phase and logits digest of the bootstrap arm on native states."""
    base.reset_arm(runtime)
    state = runtime['state']
    owner = getattr(runtime['router'], 'owner', runtime['router'])
    names = ('bootstrap_dense_calls', 'bootstrap_observation_calls', 'score_calls',
             'decision_calls', 'held_calls')
    rows = []
    for step in sequence[:upto]:
        idx = int(step['call_index'])
        kw = base.prepare_step(step['snapshot'], state)
        state.begin(int(kw['cur_step']), kw['current_canvas'])
        before = {n: getattr(owner, n) for n in names}
        result = decoder_call(model, kw)
        delta = {n: getattr(owner, n)-before[n] for n in names}
        phase = ('native' if delta['bootstrap_dense_calls'] else
                 'native+observe' if delta['bootstrap_observation_calls'] else
                 'A' if delta['score_calls'] else 'D' if delta['decision_calls'] else
                 'H' if delta['held_calls'] else '?')
        rows.append(dict(call_index=idx, phase=phase, per_layer_delta=delta,
                         output_digest=output_digest(result)))
    return rows


def _observation_components(owner, record, native_fn, reps=5):
    """v24: time each production piece of an observation (BO/A) on one real
    GLOBAL state, in isolation, on the same Q/K/V. Pieces are run exactly as
    the router runs them; the full-V projection is the diagnostic proxy for a
    fresh lease at canvas start. Sum of pieces is not a forward price."""
    from experiments.numerical_qk_reuse.cached_executor import (
        allocate_summary, preqk_attention, route_only)
    from experiments.numerical_qk_reuse.integration import Attention
    from dllm.attention.blasst.core import _attention_validity
    scale = _mask_contract(record)
    q, k, v = (record[key] for key in ('q', 'k', 'v'))
    module, args, kwargs = (record[key] for key in ('module', 'args', 'kwargs'))
    b, h, nq, nk, hk = q.shape[0], q.shape[1], q.shape[2], k.shape[-2], k.shape[1]
    prefix_tiles = max(0, nk - nq) // 64
    sensitivity = record['sensitivity']
    if sensitivity is None:
        sensitivity = torch.ones((b, nq), device=q.device, dtype=torch.float32)
    sensitivity = sensitivity.contiguous()
    threshold = float(owner.thresholds['global']['log_threshold'])
    score = Attention.observe_scores_grouped(q, k, None, scale, False, None, 0)
    legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    z, ref, _ = _project_full(owner, record, legal)
    z, ref = z.contiguous(), ref.contiguous()
    routed = route_only(score, z, ref, sensitivity=sensitivity, log_threshold=threshold,
                        variant=owner.kernel_variant)
    identity = ('v24_component_probe',)

    def with_summary():
        summary = allocate_summary(b, h, (nq + 127) // 128, (nk + 63) // 64, prefix_tiles, 32,
                                   q.device, identity)
        return route_only(score, z, ref, sensitivity=sensitivity, log_threshold=threshold,
                          summary=summary, store_summary=True, variant=owner.kernel_variant)

    pieces = {
        'native_attention': lambda: native_fn(module, q, k, v, *args, **kwargs),
        'score_producer_grouped_q': lambda: Attention.observe_scores_grouped(q, k, None, scale, False, None, 0),
        'valid_key_map': lambda: torch.isfinite(score).reshape(b, hk, h // hk, nq, nk).any((2, 3)),
        'full_current_v_projection_proxy': lambda: _project_full(owner, record, legal),
        'route_only_no_summary': lambda: route_only(score, z, ref, sensitivity=sensitivity,
                                                    log_threshold=threshold, variant=owner.kernel_variant),
        'route_with_summary_alloc_store': with_summary,
        'retained_consumer_fp32_model_major': lambda: preqk_attention(
            q, k, v, routed.skipped, routed.eligible, scale=scale, is_causal=False, window=None,
            variant=owner.kernel_variant, output_score_precision='fp32_scores_bf16_pv',
            output_layout='model_major'),
    }
    for fn in pieces.values():
        fn()
    samples = {name: [] for name in pieces}
    with base.no_compile_during_accepted() as misses:
        for repeat in range(reps):
            order = list(pieces) if repeat % 2 == 0 else list(reversed(list(pieces)))
            for name in order:
                _, event_ms, _ = _event(pieces[name])
                samples[name].append(event_ms)
    if misses:
        raise RuntimeError(f'JIT specialization during component timing: {misses}')
    median = lambda values: sorted(values)[len(values)//2]
    kept = float((routed.eligible & ~routed.skipped).sum()) / max(1., float(routed.eligible.sum()))
    return dict(status='qualified', keys=nk, queries=nq, prefix_tiles=prefix_tiles,
                kept_tile_fraction_of_eligible=kept, reps=reps,
                median_event_ms={name: median(v) for name, v in samples.items()}, samples=samples,
                note='isolated per-piece CUDA events on one GLOBAL layer; not additive to a forward price')


def _route_alignment(owner, record, reps=5):
    """v24 candidate probe: does a 16-aligned key extent reproduce the route
    exactly and faster? Padded keys carry -inf scores and zero sketches, which
    the unpadded kernel already loads for out-of-range lanes; tile count is
    unchanged. Bitwise identity of bitmaps, invalid flags and stored prefix
    summaries is measured, never assumed (alignment changes load layout)."""
    import torch.nn.functional as F
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only
    from experiments.numerical_qk_reuse.integration import Attention
    from dllm.attention.blasst.core import _attention_validity
    scale = _mask_contract(record)
    q, k = record['q'], record['k']
    b, h, nq, nk = q.shape[0], q.shape[1], q.shape[2], k.shape[-2]
    pad = (-nk) % 16
    if (nk + pad + 63) // 64 != (nk + 63) // 64:
        raise AssertionError('alignment pad changed tile count')
    sens = record['sensitivity']
    if sens is None:
        sens = torch.ones((b, nq), device=q.device, dtype=torch.float32)
    sens = sens.contiguous()
    threshold = float(owner.thresholds['global']['log_threshold'])
    score = Attention.observe_scores_grouped(q, k, None, scale, False, None, 0)
    legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    z, ref, _ = _project_full(owner, record, legal)
    z, ref = z.contiguous(), ref.contiguous()
    score_pad = F.pad(score, (0, pad), value=-float('inf')).contiguous()
    z_pad = F.pad(z, (0, 0, 0, pad)).contiguous()
    prefix_tiles = max(0, nk - nq) // 64
    qb, kt = (nq + 127) // 128, (nk + 63) // 64

    def run(sc, zz):
        summary = allocate_summary(b, h, qb, kt, prefix_tiles, 32, q.device, ('probe',))
        routed = route_only(sc, zz, ref, sensitivity=sens, log_threshold=threshold,
                            summary=summary, store_summary=True, variant=owner.kernel_variant)
        return routed, summary

    base_r, base_s = run(score, z)
    pad_r, pad_s = run(score_pad, z_pad)
    same = dict(skipped=bool(torch.equal(base_r.skipped, pad_r.skipped)),
                eligible=bool(torch.equal(base_r.eligible, pad_r.eligible)),
                invalid=bool(torch.equal(base_r.invalid_tiles, pad_r.invalid_tiles)),
                summary_z=bool(torch.equal(base_s.z.view(torch.int32), pad_s.z.view(torch.int32))),
                summary_mu=bool(torch.equal(base_s.mu.view(torch.int32), pad_s.mu.view(torch.int32))),
                summary_active=bool(torch.equal(base_s.active, pad_s.active)),
                summary_bad=bool(torch.equal(base_s.bad, pad_s.bad)))
    # Producer on zero-padded K, masked to -inf beyond K (candidate aligned producer).
    k_pad = F.pad(k, (0, 0, 0, pad))
    aligned = Attention.observe_scores_grouped(q, k_pad, None, scale, False, None, 0)
    aligned[..., nk:] = -float('inf')
    producer_equal = bool(torch.equal(aligned[..., :nk].contiguous().view(torch.int32), score.view(torch.int32)))
    del base_s, pad_s, aligned
    pieces = {'route_store_K': lambda: run(score, z), 'route_store_K16': lambda: run(score_pad, z_pad),
              'pad_scores_and_sketch_copy': lambda: (F.pad(score, (0, pad), value=-float('inf')).contiguous(),
                                                     F.pad(z, (0, 0, 0, pad)).contiguous()),
              'producer_K': lambda: Attention.observe_scores_grouped(q, k, None, scale, False, None, 0),
              'producer_zero_padded_K16': lambda: Attention.observe_scores_grouped(
                  q, F.pad(k, (0, 0, 0, pad)), None, scale, False, None, 0)}
    for fn in pieces.values():
        fn()
    samples = {n: [] for n in pieces}
    with base.no_compile_during_accepted() as misses:
        for repeat in range(reps):
            for n in (list(pieces) if repeat % 2 == 0 else list(reversed(list(pieces)))):
                samples[n].append(_event(pieces[n])[1])
    if misses:
        raise RuntimeError(f'JIT specialization during alignment timing: {misses}')
    median = lambda v: sorted(v)[len(v)//2]
    return dict(status='qualified', keys=nk, pad=pad, kdiv_before=next(d for d in (16, 8, 4, 2, 1) if nk % d == 0),
                route_outputs_bitwise_equal=same, producer_first_K_bitwise_equal=producer_equal,
                median_event_ms={n: median(v) for n, v in samples.items()}, samples=samples,
                note='diagnostic probe of an exact alignment candidate; JIT warm for both alignment classes before timing')


def _float_ulp_gap(a, b):
    """Max ULP distance between equal-shape FP32 tensors (finite, same-sign lanes)."""
    ai, bi = a.contiguous().view(torch.int32).long(), b.contiguous().view(torch.int32).long()
    ai = torch.where(ai < 0, -(ai & 0x7fffffff), ai)
    bi = torch.where(bi < 0, -(bi & 0x7fffffff), bi)
    both = torch.isfinite(a) & torch.isfinite(b)
    gap = (ai - bi).abs().masked_fill(~both, 0)
    return int(gap.max()) if gap.numel() else 0


def _tensor_delta(a, b):
    a, b = a.float(), b.float()
    same_bits = bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))
    finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs().masked_fill(~finite, 0)
    scale = a.abs().masked_fill(~finite, 0)
    return dict(bitwise_equal=same_bits,
                differing_elements=int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum()),
                elements=a.numel(), nonfinite_mismatch=int((torch.isfinite(a) != torch.isfinite(b)).sum()),
                max_abs=float(diff.max()) if diff.numel() else 0.,
                max_rel=float((diff / scale.clamp_min(1e-30)).max()) if diff.numel() else 0.,
                max_ulp=_float_ulp_gap(a, b))


GLOBAL_LAYERS = (5, 11, 17, 23, 29)


def _call_record(idx, phase, logits, owner):
    """CPU-side record of one replayed call (logits, decisions, clocks, anchor summaries)."""
    decisions = {int(layer): (entry.decision.skipped.detach().cpu(), entry.decision.eligible.detach().cpu())
                 for layer, entry in owner.cache.entries.items() if entry.decision is not None}
    clocks = {int(layer): (entry.score_step, entry.decision_step) for layer, entry in owner.cache.entries.items()}
    summaries = ({int(layer): (summary.z.detach().cpu(), summary.mu.detach().cpu())
                  for layer, summary in owner.summaries.items()} if phase in ('BO', 'A') else {})
    return dict(call_index=int(idx), phase=phase, logits=logits.detach().to('cpu'),
                argmax=logits.argmax(-1).to('cpu'), decisions=decisions, clocks=clocks, summaries=summaries)


def _compare_call(ref, cand, *, expect_summaries):
    """Functional comparison of one call. Missing or extra layers are failures,
    never skipped: the only legal empty decision set is bootstrap call 0 (B0)."""
    failures = []
    if ref['call_index'] != cand['call_index'] or ref['phase'] != cand['phase']:
        failures.append('call_index_or_phase')
    expected = set() if cand['phase'] == 'B0' else set(GLOBAL_LAYERS)
    for name in ('decisions', 'clocks'):
        if set(ref[name]) != expected or set(cand[name]) != expected:
            failures.append(f'{name}_layer_set')
    want_summary = set(GLOBAL_LAYERS) if (expect_summaries and cand['phase'] in ('BO', 'A')) else set()
    if set(ref['summaries']) != want_summary or set(cand['summaries']) != want_summary:
        failures.append('summary_layer_set')
    logits = _tensor_delta(ref['logits'], cand['logits'])
    argmax_equal = bool(torch.equal(ref['argmax'], cand['argmax']))
    if not logits['bitwise_equal']:
        failures.append('logits')
    common = set(ref['decisions']) & set(cand['decisions'])
    decisions_equal = {str(l): bool(torch.equal(ref['decisions'][l][0], cand['decisions'][l][0]) and
                                    torch.equal(ref['decisions'][l][1], cand['decisions'][l][1])) for l in common}
    tiles = {str(l): int((ref['decisions'][l][0] != cand['decisions'][l][0]).sum()) for l in common}
    if not all(decisions_equal.values()):
        failures.append('decisions')
    if any(ref['clocks'][l] != cand['clocks'][l] for l in set(ref['clocks']) & set(cand['clocks'])):
        failures.append('clocks')
    summaries = {str(l): dict(z=_tensor_delta(ref['summaries'][l][0], cand['summaries'][l][0]),
                              mu=_tensor_delta(ref['summaries'][l][1], cand['summaries'][l][1]))
                 for l in set(ref['summaries']) & set(cand['summaries'])}
    return dict(call_index=cand['call_index'], phase=cand['phase'], parent_phase=ref['phase'],
                logits={k: v for k, v in logits.items()}, argmax_equal=argmax_equal,
                decisions_equal=decisions_equal, decision_tiles_differing=tiles, summaries=summaries,
                level2_failures=failures)


def _sequence_verdict(parent_rows, rows, *, expected_calls):
    """Machine-readable gate for one arm. Execution, level 1 (intermediate bits),
    level 2 (functional identity) and pilot eligibility are separate fields."""
    structural = []
    if len(parent_rows) != expected_calls or len(rows) != expected_calls:
        structural.append('call_count')
    if [r['call_index'] for r in parent_rows] != [r['call_index'] for r in rows]:
        structural.append('call_indices')
    level2_failures = sorted({f for r in rows for f in r['level2_failures']} | set(structural))
    summary_bits = [d['bitwise_equal'] for r in rows for v in r['summaries'].values() for d in v.values()]
    return dict(executed=len(rows) == expected_calls,
                level1_summaries_bitwise_equal=(all(summary_bits) if summary_bits else None),
                level1_summary_records=len(summary_bits),
                level2_functional_identity=not level2_failures, level2_failures=level2_failures,
                pilot_eligible=not level2_failures)


def _aligned_replay(model, sequence, runtime, upto, parent=None, expect_summaries=False):
    """Replay native states through one runtime; with ``parent`` compare per call."""
    base.reset_arm(runtime)
    state = runtime['state']
    owner = getattr(runtime['router'], 'owner', runtime['router'])
    rows, keep = [], {}
    for step in sequence[:upto]:
        idx = int(step['call_index'])
        kw = base.prepare_step(step['snapshot'], state)
        state.begin(int(kw['cur_step']), kw['current_canvas'])
        before = (owner.bootstrap_observation_calls, owner.score_calls, owner.decision_calls,
                  owner.held_calls, owner.bootstrap_dense_calls)
        logits = decoder_call(model, kw)
        after = (owner.bootstrap_observation_calls, owner.score_calls, owner.decision_calls,
                 owner.held_calls, owner.bootstrap_dense_calls)
        delta = [x - y for x, y in zip(after, before)]
        phase = ('BO' if delta[0] else 'A' if delta[1] else 'D' if delta[2] else 'H' if delta[3]
                 else 'B0' if delta[4] else '?')
        record = _call_record(idx, phase, logits, owner)
        if parent is None:
            keep[idx] = record
            rows.append(dict(call_index=idx, phase=phase))
        else:
            ref = parent.get(idx)
            if ref is None:
                rows.append(dict(call_index=idx, phase=phase, level2_failures=['missing_parent_call'],
                                 summaries={}, logits={}, decisions_equal={}, decision_tiles_differing={},
                                 argmax_equal=False))
            else:
                rows.append(_compare_call(ref, record, expect_summaries=expect_summaries))
    return rows, keep


def _aligned_qualification(adapter, model, sequence, selected_arm):
    upto = min(len(sequence), 16)
    out = {}
    for arm_name, parent_arm in (('M3', 'M3_R3_A8_current_output'), ('M1', 'M1_R1_A8_current_output'),
                                 ('B', 'B_A8_matched')):
        parent_cfg = _bootstrap_arm(selected_arm, parent_arm, arm_name + '_boot', panel_selector=True)
        aligned_cfg = _bootstrap_arm(selected_arm, parent_arm, arm_name + '_boot_aligned16',
                                     route_storage='aligned16', panel_selector=True)
        if parent_cfg['config']['parent_config'].get('selector') != (
                'legacy_recompute' if parent_arm == 'B_A8_matched' else 'prefix_block_summary'):
            raise AssertionError('qualification selector differs from the scored panel')
        expect_summaries = parent_arm != 'B_A8_matched'
        with base.arm_context(adapter, parent_cfg) as runtime:
            parent_rows, kept = _aligned_replay(model, sequence, runtime, upto)
            parent_storage = getattr(runtime['router'], 'owner', runtime['router']).route_storage
        with base.arm_context(adapter, aligned_cfg) as runtime:
            rows, _ = _aligned_replay(model, sequence, runtime, upto, parent=kept,
                                      expect_summaries=expect_summaries)
            owner = getattr(runtime['router'], 'owner', runtime['router'])
            counters = dict(route_storage=owner.route_storage, copies=owner.aligned_score_copies,
                            buffer_bytes_allocated=owner.aligned_pad_bytes,
                            sketch_pads=owner.aligned_sketch_pads)
        kept.clear()
        verdict = _sequence_verdict(parent_rows, rows, expected_calls=upto)
        if parent_storage != 'logical' or counters['route_storage'] != 'aligned16':
            verdict['level2_failures'].append('route_storage_exposure')
            verdict.update(level2_functional_identity=False, pilot_eligible=False)
        out[arm_name] = dict(
            phases=''.join(r['phase'] for r in rows),
            parent_phases=''.join(r['phase'] for r in parent_rows),
            logits_bitwise_equal_calls=sum(bool(r['logits'].get('bitwise_equal')) for r in rows),
            argmax_equal_calls=sum(r['argmax_equal'] for r in rows), calls=len(rows),
            decision_mismatch_calls=sum(not all(r['decisions_equal'].values()) or not r['decisions_equal']
                                        for r in rows if r['phase'] != 'B0'),
            verdict=verdict, aligned_counters=counters, rows=rows)
    ok = all(a['verdict']['level2_functional_identity'] for a in out.values())
    return dict(status='qualified' if ok else 'level2_failed', sequence_calls=upto, arms=out,
                note='teacher-forced native states; complete A/D/H cycles with live causal T and current V')


def _native_stop(model, snapshot):
    kw = base.prepare_step(snapshot, None)
    result = model._denoising_step(**kw)
    return bool(result[3].all())


def _replay(model, sequence, selected, runtime, *, native):
    base.reset_arm(runtime)
    registry, native_fn = native_registry(model)
    state = runtime.get('state') if runtime else None
    records = {}
    outputs = {}
    inputs = {}
    for step in sequence[:max(selected)+1]:
        idx = int(step['call_index'])
        kw = base.prepare_step(step['snapshot'], state)
        if state is not None:
            state.begin(int(kw['cur_step']), kw['current_canvas'])
        if idx in selected:
            inputs[idx] = StepSnapshot.digest(kw, controller=state)
            sink = {}
            if native:
                with _spy_native(registry, native_fn, {0, 5}, sink):
                    result = decoder_call(model, kw)
            else:
                if registry['sdpa'] is native_fn:
                    raise RuntimeError('method replay lacks installed attention dispatcher')
                with _spy_dispatch(registry, {0}, sink), _spy_method(runtime, {5}, sink):
                    result = decoder_call(model, kw)
                if 0 in sink and 5 in sink:
                    sink[0]['sensitivity'] = sink[5]['sensitivity']
            if set(sink) != {0, 5}:
                raise RuntimeError(f'expected one LOCAL0 and GLOBAL5 attention, got {sorted(sink)}')
            records[idx] = sink
            outputs[idx] = output_digest(result)
        else:
            decoder_call(model, kw)
    return dict(records=records, input_digests=inputs, output_digests=outputs)


@torch.inference_mode()
def run(config, checkpoint=None, *, bootstrap_calibration=False, q16_validation=False,
        observation_parity=False, bootstrap_parity=False, observation_components=False,
        route_alignment=False, aligned_qualification=False):
    if sum(map(bool, (bootstrap_calibration, q16_validation, observation_parity, bootstrap_parity,
                      observation_components, route_alignment, aligned_qualification))) > 1:
        raise ValueError('calibration, validation and observation parity are separate stages')
    from dllm.models import create_adapter
    paths, proof = diagnostic.preflight(config)
    rows = {(dataset, id_): row for dataset, path in paths.items()
            for id_, row in base.load_rows(path).items()}
    selected_arm = next(a for a in config['arms'] if a['name'] == 'M3_R3_new')
    from experiments.numerical_qk_reuse import geometry_diagnostic
    adapter = create_adapter('diffusion_gemma', str(config['model']), device='cuda',
                             precision='bfloat16', revision=config['revision']).load()
    model = adapter.model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(schema='v21b_geometry_capture_v1', quality_eligible=False,
                  screen=('missing_bootstrap_and_two_LB_calibration_states' if bootstrap_calibration
                          else 'frozen_q16_offsets_validation_and_attention_share' if q16_validation
                          else 'v23_gqa_observation_producer_parity' if observation_parity
                          else 'v23_native_bootstrap_qualification' if bootstrap_parity
                          else 'v24_observation_components' if observation_components
                          else 'v24_route_alignment_probe' if route_alignment
                          else 'v25_aligned16_qualification' if aligned_qualification
                          else 'inherited_threshold_opportunity'),
                  source_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  source_sha256={
                      str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      str(Path(geometry_diagnostic.__file__).resolve()):
                          hashlib.sha256(Path(geometry_diagnostic.__file__).read_bytes()).hexdigest()},
                  preflight=proof, model=str(config['model']), revision=config['revision'],
                  source_commit=selected_arm['config']['parent_config'].get('source_commit'),
                  arm=dict(name=selected_arm['name'], condition=selected_arm['condition'],
                           fingerprint=selected_arm['config'].get('fingerprint'),
                           output_score_precision=selected_arm['config'].get('output_score_precision'),
                           output_layout=selected_arm['config'].get('output_layout')),
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0), gpu_uuid=proof['gpu_uuid']),
                  groups={}, errors=[])
    grouped = {}
    for target in config['targets']:
        grouped.setdefault((target['dataset'], target['id'], int(target['canvas'])), []).append(target)
    for (dataset, id_, canvas), targets in grouped.items():
        key = f'{dataset}|{id_}|canvas{canvas}'
        group = dict(dataset=dataset, id=id_, canvas=canvas, status='pending', targets=targets)
        captured = native = method = None
        report['groups'][key] = group
        if checkpoint:
            checkpoint(report)
        try:
            row = rows.get((dataset, id_)) or rows.get(('*', id_))
            if row is None:
                raise ValueError('target absent from gold-free manifest')
            captured, capture_proof = diagnostic.capture_checked(adapter, row, targets, 101)
            sequence = captured[canvas]
            selected, missing = _selected(sequence, targets)
            if bootstrap_calibration:
                wanted = {0, 3} if dataset == 'longbench_v2' else {0}
                reached = {int(s['call_index']) for s in sequence}
                selected, missing = sorted(wanted & reached), sorted(wanted - reached)
            elif q16_validation or observation_parity or observation_components or route_alignment:
                # Same states as the inherited-threshold capture plus bootstrap call 0.
                selected = sorted(set(selected) | {0})
            group.update(status='captured', reached_calls=len(sequence), missing_requested_calls=missing,
                         selected_calls=selected, native_path_proof=capture_proof)
            if not selected:
                group['status'] = 'missing_canvas'
                continue
            last = int(sequence[-1]['call_index'])
            try:
                stopped = _native_stop(model, sequence[-1]['snapshot'])
                group['last_capture'] = dict(call_index=last, native_stop_verified=stopped,
                    label=('same_history_native_stop' if stopped else 'last_captured_not_verified_near_stop'))
            except Exception as exc:
                group['last_capture'] = dict(call_index=last, native_stop_verified=None,
                    label='last_captured_native_stop_unknown', error=_error(exc, 'native_stop_probe'))
            registry, native_fn = native_registry(model)
            if aligned_qualification:
                try:
                    group['aligned16'] = _aligned_qualification(adapter, model, sequence, selected_arm)
                    group['status'] = 'complete'
                    if group['aligned16']['status'] != 'qualified':
                        # Keep the full report; the recorded error makes the stage exit nonzero.
                        group['status'] = 'failed'
                        report['errors'].append(dict(group=key, stage='aligned_qualification_gate',
                            failures={a: v['verdict']['level2_failures']
                                      for a, v in group['aligned16']['arms'].items()}))
                except Exception as exc:
                    group['status'] = 'failed'
                    group['error'] = _error(exc, 'aligned_qualification')
                    report['errors'].append(dict(group=key, **group['error']))
                group['calls'] = {}
                continue
            if bootstrap_parity:
                upto = min(len(sequence), 13)
                native = _replay(model, sequence, [0, 1], None, native=True)
                native['records'].clear()
                expected = {1: 'ADDDDDDDADDDD', 3: 'AHHDHHDHAHHDH', 8: 'AHHHHHHHAHHHH'}
                results = {}
                for arm_name, interval, parent_arm in (
                        ('M1_native_bootstrap2_observe1', 1, 'M1_R1_A8_current_output'),
                        ('M3_native_bootstrap2_observe1', 3, 'M3_R3_A8_current_output'),
                        ('B_native_bootstrap2_observe1', 8, 'B_A8_matched')):
                    arm = _bootstrap_arm(selected_arm, parent_arm, arm_name)
                    with base.arm_context(adapter, arm) as runtime:
                        call_rows = _bootstrap_replay(model, sequence, runtime, upto)
                    phases = ''.join('N' if r['phase'] == 'native' else 'A' if r['phase'] == 'native+observe'
                                     else r['phase'] for r in call_rows)
                    want = ('N' + expected[interval])[:upto]
                    logits_equal = {str(i): call_rows[i]['output_digest'] == native['output_digests'][i]
                                    for i in (0, 1) if i < len(call_rows)}
                    ok = phases == want and all(logits_equal.values())
                    results[arm_name] = dict(status='qualified' if ok else 'failed',
                                             phases=phases, expected=want,
                                             native_logits_equal=logits_equal, calls=call_rows)
                    if not ok:
                        report['errors'].append(dict(group=key, arm=arm_name, stage='bootstrap_parity',
                                                     phases=phases, expected=want,
                                                     native_logits_equal=logits_equal))
                group['bootstrap'] = results
                group['calls'] = {}
                group['status'] = 'complete' if all(r['status'] == 'qualified'
                                                    for r in results.values()) else 'failed'
                continue
            if observation_parity or observation_components or route_alignment:
                native = dict(records={}, input_digests={}, output_digests={})
                native_qkv_hashes = {}
            elif q16_validation:
                kinds = _layer_kinds(model)
                by_index = {int(s['call_index']): s for s in sequence}
                shares = {}
                for idx in selected:
                    try:
                        shares[str(idx)] = _attention_share(model, by_index[idx]['snapshot'],
                                                            registry, native_fn, kinds)
                    except Exception as exc:
                        shares[str(idx)] = dict(status='failed', error=_error(exc, 'attention_share'))
                        report['errors'].append(dict(group=key, call_index=idx,
                                                     **_error(exc, 'attention_share')))
                group['attention_share'] = shares
                native = dict(records={}, input_digests={}, output_digests={})
                native_qkv_hashes = {}
            else:
                native = _replay(model, sequence, selected, None, native=True)
                native_qkv_hashes = {idx: {layer: output_digest(tuple(record[name] for name in ('q', 'k', 'v')))
                                           for layer, record in layers.items()}
                                     for idx, layers in native['records'].items()}
                native['records'].clear()  # large native QKV clones are not needed for geometry/cost
            with base.arm_context(adapter, selected_arm) as runtime:
                method = _replay(model, sequence, selected, runtime, native=False)
                owner = getattr(runtime['router'], 'owner', runtime['router'])
                results = {}
                for idx in selected:
                    target_result = dict(call_index=idx, input_digest_native=native['input_digests'].get(idx),
                                         input_digest_method=method['input_digests'][idx],
                                         native_forward_digest=native['output_digests'].get(idx),
                                         method_forward_digest=method['output_digests'][idx], layers={})
                    for layer in ((5,) if (q16_validation or observation_parity or observation_components
                                           or route_alignment) else (0, 5)):
                        if route_alignment:
                            try:
                                target_result['layers']['5'] = dict(
                                    status='qualified', attention_kind='GLOBAL',
                                    alignment=_route_alignment(owner, method['records'][idx][5]))
                            except Exception as exc:
                                target_result['layers']['5'] = dict(status='failed', error=_error(exc, 'layer'))
                                report['errors'].append(dict(group=key, call_index=idx, layer=5,
                                                             **_error(exc, 'route_alignment')))
                            continue
                        if observation_components:
                            try:
                                target_result['layers']['5'] = dict(
                                    status='qualified', attention_kind='GLOBAL',
                                    components=_observation_components(owner, method['records'][idx][5], native_fn))
                            except Exception as exc:
                                target_result['layers']['5'] = dict(status='failed', error=_error(exc, 'layer'))
                                report['errors'].append(dict(group=key, call_index=idx, layer=5,
                                                             **_error(exc, 'observation_components')))
                            continue
                        if observation_parity:
                            try:
                                target_result['layers']['5'] = dict(
                                    status='qualified', attention_kind='GLOBAL',
                                    observation=_observation_parity(owner, method['records'][idx][5]))
                            except Exception as exc:
                                target_result['layers']['5'] = dict(status='failed', error=_error(exc, 'layer'))
                                report['errors'].append(dict(group=key, call_index=idx, layer=5,
                                                             **_error(exc, 'observation_parity')))
                            continue
                        if q16_validation:
                            try:
                                target_result['layers']['5'] = dict(
                                    status='qualified', attention_kind='GLOBAL',
                                    geometry=_validation_geometry(owner, method['records'][idx][5]))
                            except Exception as exc:
                                target_result['layers']['5'] = dict(status='failed', error=_error(exc, 'layer'))
                                report['errors'].append(dict(group=key, call_index=idx, layer=5,
                                                             **_error(exc, 'validation_geometry')))
                            continue
                        record = method['records'][idx][layer]
                        try:
                            geometry = _geometry(owner, record, layer,
                                matched_screen=bootstrap_calibration and dataset == 'longbench_v2'
                                               and idx == 3 and layer == 5)
                            layer_result = dict(status='geometry_qualified_timing_pending',
                                attention_kind=('LOCAL' if layer == 0 else 'GLOBAL'),
                                native_observed_qkv_digest=native_qkv_hashes[idx][layer],
                                geometry=geometry)
                            target_result['layers'][str(layer)] = layer_result
                            if checkpoint:
                                group['calls'] = results | {str(idx): target_result}
                                checkpoint(report)
                            timing = _timed_attention(record, owner, native_fn)
                            layer_result.update(status='qualified', timing=timing)
                        except Exception as exc:
                            layer_result = target_result['layers'].setdefault(str(layer), {})
                            layer_result.update(status='failed', error=_error(exc, 'layer'))
                            report['errors'].append(dict(group=key, call_index=idx, layer=layer,
                                                         **_error(exc, 'geometry_or_timing')))
                        if checkpoint:
                            group['calls'] = results | {str(idx): target_result}
                            checkpoint(report)
                    results[str(idx)] = target_result
                    method['records'].pop(idx, None)
                group['calls'] = results
            group['status'] = 'complete' if not any(
                layer['status'] != 'qualified' for call in group['calls'].values()
                for layer in call['layers'].values()) else 'failed'
        except Exception as exc:
            group['status'] = 'failed'
            group['error'] = _error(exc, 'capture_or_replay')
            report['errors'].append(dict(group=key, **group['error']))
        finally:
            if checkpoint:
                checkpoint(report)
            # Captured snapshots reference the live native cache; release them
            # before any later canvas can commit to that cache.
            if method is not None:
                method.clear()
            if native is not None:
                native.clear()
            if captured is not None:
                captured.clear()
    return report


def completeness(report, *, bootstrap_calibration, q16_validation=False):
    """Stage verdict from saved per-layer records; process exit 0 alone proves nothing.

    Natively unreachable requested calls are listed separately and are not
    implementation failures, but a calibration stage lacking either frozen
    LongBench call-3 five-point screen is still incomplete.
    """
    failed_layers, failed_groups, unreachable, screens = [], [], [], {}
    for key, group in report['groups'].items():
        unreachable += [f'{key}:{idx}' for idx in group.get('missing_requested_calls', [])]
        if group.get('status') not in ('complete', 'missing_canvas'):
            failed_groups.append(key)
        for idx, call in group.get('calls', {}).items():
            for layer, item in call.get('layers', {}).items():
                if item.get('status') != 'qualified':
                    failed_layers.append(f'{key}:{idx}:{layer}')
        if q16_validation:
            for idx in group.get('selected_calls', []):
                layer = group.get('calls', {}).get(str(idx), {}).get('layers', {}).get('5', {})
                screen = layer.get('geometry', {}).get('comparison', {}).get('matched_q16_calibration')
                screens[f'{key}:{idx}'] = len(screen['q16_candidates']) if screen else 0
                share = group.get('attention_share', {}).get(str(idx), {})
                if share.get('status') != 'qualified':
                    failed_layers.append(f'{key}:{idx}:attention_share')
        if bootstrap_calibration and group.get('dataset') == 'longbench_v2':
            layer = group.get('calls', {}).get('3', {}).get('layers', {}).get('5', {})
            screen = layer.get('geometry', {}).get('comparison', {}).get('matched_q16_calibration')
            screens[key] = len(screen['q16_candidates']) if screen else 0
    calibration_ok = (not (bootstrap_calibration or q16_validation)) or (
        screens and all(n == 5 for n in screens.values()))
    ok = not (failed_layers or failed_groups or report.get('errors')) and calibration_ok
    return dict(status='complete' if ok else 'incomplete', failed_layers=failed_layers,
                failed_groups=failed_groups, error_count=len(report.get('errors', [])),
                native_unreachable=unreachable, calibration_candidates=screens,
                rule='every selected layer qualified, no recorded error, and (calibration) five points per LB call-3 state')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qualify-tests', action='store_true',
                        help='run the geometry parity test under this billed stage before model capture')
    parser.add_argument('--bootstrap-calibration', action='store_true',
                        help='separate missing-bootstrap and frozen two-LB-state work/error screen')
    parser.add_argument('--aligned-qualification', action='store_true',
                        help='v25: parent vs aligned16 full A/D/H replay: logits, decisions, anchor summaries')
    parser.add_argument('--route-alignment', action='store_true',
                        help='v24: exactness/timing probe of a 16-aligned route key extent')
    parser.add_argument('--observation-components', action='store_true',
                        help='v24: isolated per-piece timing of observation work on real GLOBAL states')
    parser.add_argument('--bootstrap-parity', action='store_true',
                        help='v23 Track P: native logits at calls 0/1 and origin-1 A/D/H schedule')
    parser.add_argument('--observation-parity', action='store_true',
                        help='v23 Track E: grouped-Q versus repeat_interleave score producer on real GLOBAL states')
    parser.add_argument('--q16-validation', action='store_true',
                        help='frozen five-offset Q16 screen on every reached GLOBAL state plus native attention-share oracles')
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + '.partial.json')
    failure = args.out.with_name(args.out.name + '.failure.json')
    lock = args.out.with_name(args.out.name + '.lock')
    test_log = args.out.with_name(args.out.name + '.tests.log')
    occupied = [p for p in (args.out, partial, failure, lock, test_log,
                            args.out.with_name(args.out.name + '.writing')) if p.exists()]
    if occupied:
        raise FileExistsError(f'output already exists: {occupied}')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        if args.qualify_tests:
            with test_log.open('x', encoding='utf-8') as stream:
                result = subprocess.run([sys.executable, '-m', 'pytest',
                                         'tests/test_v21b_geometry.py', '-q'],
                                        cwd=Path(__file__).resolve().parents[1],
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=600, check=False)
            if result.returncode:
                raise RuntimeError(f'geometry qualification tests failed; exit={result.returncode}')
        report = run(config, checkpoint=lambda data: base.atomic_json(partial, data),
                     bootstrap_calibration=args.bootstrap_calibration,
                     q16_validation=args.q16_validation,
                     observation_parity=args.observation_parity,
                     bootstrap_parity=args.bootstrap_parity,
                     observation_components=args.observation_components,
                     route_alignment=args.route_alignment,
                     aligned_qualification=args.aligned_qualification)
        report['qualification_tests'] = dict(status=('passed' if args.qualify_tests else 'not_requested'),
                                             log=str(test_log) if args.qualify_tests else None)
        report['completeness'] = completeness(report, bootstrap_calibration=args.bootstrap_calibration,
                                              q16_validation=args.q16_validation)
        base.atomic_json(args.out, report)
        partial.unlink(missing_ok=True)
    except BaseException as exc:
        base.atomic_json(failure, dict(schema='v21b_geometry_capture_failure_v1',
                                       error=_error(exc, 'run'),
                                       config_sha256=(hashlib.sha256(config_bytes).hexdigest()
                                                      if 'config_bytes' in locals() else None),
                                       partial_path=str(partial) if partial.exists() else None))
        raise
    finally:
        lock.unlink(missing_ok=True)
    if report['completeness']['status'] != 'complete':
        # Results are already saved; a distinct code keeps partial evidence
        # while preventing a zero exit from reading as full qualification.
        raise SystemExit(3)


if __name__ == '__main__':
    main()
