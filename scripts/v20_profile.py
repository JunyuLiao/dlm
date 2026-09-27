"""Direct, teacher-forced v20 full-forward and native-step sequence profiler.

The JSON config names a gold-free manifest, model/revision, seed, at least two
predeclared inputs per dataset, and arms.  Each target has ``id``, ``dataset``,
``canvas``, ``call_index`` and optional ``fallback_call_index``.  Replay starts
at call zero of that *real* canvas and ends at the available call <= 16.  A
later state is reported missing when native generation ends before it.

Run in a dedicated timing process, separate from accepted request timings.
No profiler or component call instrumentation is used in accepted intervals.
CUDA event spans include host launch gaps; they are not GPU-active sums.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from types import MethodType

import torch

from scripts.replay_harness import (StepSnapshot, assert_native_path, dispatch_spy,
                                    execution_context, output_digest, reset_router)
from scripts.v9_step_replay_profile import decoder_call, native_registry


class CaptureDone(Exception):
    """Private interruption after all requested native calls are observed."""


def validate_config(config):
    if type(config.get('counter_twins', False)) is not bool or type(config.get('prepared_support_floor', False)) is not bool:
        raise ValueError('counter_twins and prepared_support_floor must be explicit booleans')
    if config.get('prepared_support_floor', False) and not config.get('counter_twins', False):
        raise ValueError('prepared support floor requires an untimed counter twin')
    if type(config.get('floor_reps', 3)) is not int or config.get('floor_reps', 3) < 3:
        raise ValueError('prepared support floor requires at least three repetitions')
    if int(config.get('reps', 0)) < 3 or int(config.get('blocks', 3)) < 3 or int(config['reps']) < int(config.get('blocks', 3)):
        raise ValueError('at least three repetitions and three timing blocks required')
    if int(config.get('warmup', 1)) < 1:
        raise ValueError('at least one cold/warmup observation required')
    boundaries = config.get('boundaries', ['model_forward', 'denoising_step'])
    lengths = config.get('sequence_lengths', [4, 16])
    if not boundaries or len(set(boundaries)) != len(boundaries) or any(
            b not in ('model_forward', 'denoising_step') for b in boundaries):
        raise ValueError('boundaries must select model_forward and/or denoising_step')
    if not lengths or len(set(lengths)) != len(lengths) or any(type(n) is not int or n not in (4, 16) for n in lengths):
        raise ValueError('sequence_lengths must select N4 and/or N16')
    if not config.get('model') or not config.get('revision') or not (config.get('manifest') or config.get('manifests')):
        raise ValueError('model, revision, and gold-free manifest(s) are required')
    targets = config.get('targets', [])
    if not targets:
        raise ValueError('predeclared targets required')
    seen = set()
    for target in targets:
        for key in ('id', 'dataset', 'canvas', 'call_index'):
            if key not in target:
                raise ValueError(f'target lacks {key}')
        key = (target['id'], int(target['canvas']), int(target['call_index']))
        if key in seen or min(key[1:]) < 0 or key[2] >= 16:
            raise ValueError(f'duplicate or negative target {key}')
        seen.add(key)
        fallback = target.get('fallback_call_index')
        if fallback is not None and not 0 <= int(fallback) < int(target['call_index']):
            raise ValueError('fallback must be an earlier, predeclared call')
    arms = config.get('arms', [])
    if not arms or len({a['name'] for a in arms}) != len(arms):
        raise ValueError('arms need unique names')
    if arms[0].get('condition') != 'native_dense':
        raise ValueError('first arm must be untouched native_dense for bracketing')
    for arm in arms:
        if not all(k in arm for k in ('name', 'condition', 'config')):
            raise ValueError('each arm needs name, condition, config')
        if arm['condition'] != 'native_dense' and not arm.get('plugin'):
            raise ValueError(f"arm {arm['name']} requires plugin module:install")


def load_rows(path):
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _rows
    rows = _rows(Path(path), allow_task_budgets=True, allow_thinking_off=True)
    if any(set(row) & set(GOLD_FIELDS) for row in rows):
        raise ValueError('profile manifest must be gold-free')
    return {str(row['id']): {k: v for k, v in row.items() if k not in GOLD_FIELDS}
            for row in rows}


def preflight_identity(config, paths):
    """Check frozen input/code/model bytes before loading a model onto GPU."""
    frozen_manifests = config.get('manifest_sha256', {})
    if type(frozen_manifests) is not dict or set(frozen_manifests) != set(paths):
        raise ValueError('profile manifest byte identities are incomplete')
    actual_manifests = {dataset: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        for dataset, path in paths.items()}
    if actual_manifests != frozen_manifests:
        raise ValueError('profile manifest byte identity drift')
    for arm in config['arms']:
        arm_config = arm['config']
        if arm_config.get('model') != str(Path(config['model']).resolve()) or arm_config.get('revision') != config['revision']:
            raise ValueError('profile arm model/revision identity drift')
        expected = arm_config.get('manifest_sha256_by_dataset')
        if expected != frozen_manifests:
            raise ValueError('profile arm manifest identity drift')
        source = arm_config.get('source_hashes')
        metadata = arm_config.get('model_metadata_hashes')
        if not isinstance(source, dict) or not source or not isinstance(metadata, dict) or not metadata:
            raise ValueError('profile arm lacks frozen source/model metadata bytes')
        for source_path, digest in source.items():
            if hashlib.sha256(Path(source_path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'profile source byte identity drift: {source_path}')
        for name, digest in metadata.items():
            if Path(name).name != name or hashlib.sha256((Path(config['model']) / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'profile model metadata byte identity drift: {name}')
    actual_uuid = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True,
        timeout=15).strip().splitlines()
    if len(actual_uuid) != 1 or not actual_uuid[0].strip():
        raise ValueError('profile requires one identifiable GPU')
    if config.get('gpu_uuid') and config['gpu_uuid'] != actual_uuid[0].strip():
        raise ValueError('profile GPU UUID differs from frozen host assignment')
    return dict(manifest_sha256=actual_manifests, gpu_uuid=actual_uuid[0].strip(),
                hostname=socket.gethostname())


def request_for(row, seed):
    from dllm.models import GenerationRequest
    budget = row.get('generation_budget', 8192)
    thinking = row.get('thinking', True)
    if type(budget) is not int or not 1 <= budget <= 8192 or type(thinking) is not bool:
        raise ValueError('invalid task generation contract')
    return GenerationRequest(prompt=row['prompt'], max_new_tokens=budget,
                             temperature=0.0, seed=seed, extra={'thinking': thinking})


def _prefix(cache):
    return dict(absolute=int(cache.get_seq_length()),
                stored_by_layer={str(i): int(layer.keys.shape[-2])
                                 for i, layer in enumerate(cache.layers)})


def capture(adapter, row, targets, seed):
    """Capture native calls outside observe's wrapper; never extend a canvas."""
    from experiments.value_direction_hopper.query_adaptive import State, observe
    model = adapter.model
    registry, native = native_registry(model)
    if registry['sdpa'] is not native or '_denoising_step' in vars(model):
        raise RuntimeError('native capture requires untouched SDPA and no step override')
    wanted = {int(t['canvas']) for t in targets}
    max_canvas = max(wanted)
    controller = State('T', None, m_ref=14.258454322814941, beta=3., gamma=.5,
                       diagnostics=False, fast_t=True)
    captured = {canvas: [] for canvas in wanted}
    count = {'canvas': -1}
    with observe(model, controller):
        inner = model._denoising_step

        def outer(this, **kwargs):
            cur = int(kwargs['cur_step'])
            if cur == 48:
                count['canvas'] += 1
            canvas = count['canvas']
            if canvas > max_canvas:
                raise CaptureDone()
            if canvas in wanted and len(captured[canvas]) < 16:
                cache = kwargs['past_key_values']
                captured[canvas].append(dict(snapshot=StepSnapshot(kwargs, controller=controller),
                                             canvas=canvas, call_index=48-cur,
                                             cur_step=cur, prefix=_prefix(cache),
                                             context=execution_context(model)))
            result = inner(**kwargs)
            if canvas == max_canvas and (len(captured.get(canvas, ())) >= 16 or bool(result[3].all())):
                raise CaptureDone()
            return result

        model._denoising_step = MethodType(outer, model)
        try:
            with dispatch_spy(model) as counts:
                try:
                    adapter.generate(request_for(row, seed))
                except CaptureDone:
                    pass
            assert_native_path(counts)
        finally:
            model._denoising_step = inner
    if '_denoising_step' in vars(model):
        raise RuntimeError('capture observer leaked')
    return captured, counts


def resolve_target(target, captured):
    canvas = int(target['canvas'])
    steps = captured.get(canvas, [])
    actual = [s['call_index'] for s in steps]
    desired = int(target['call_index'])
    fallback = target.get('fallback_call_index')
    selected = desired if desired in actual else (int(fallback) if fallback is not None and int(fallback) in actual else None)
    return dict(requested_call=desired, selected_call=selected,
                missing=selected is None, fallback_used=selected is not None and selected != desired,
                reached_calls=actual, reached_canvas=bool(steps),
                sequence_lengths={str(n): min(n, len(steps)) for n in (4, 16)})


def arm_context(adapter, arm):
    if arm['condition'] == 'native_dense':
        return nullcontext(None)
    module, function = arm['plugin'].split(':', 1)
    return getattr(importlib.import_module(module), function)(adapter, arm['config'], arm['condition'])


def reset_arm(runtime):
    if runtime is None:
        return
    router = runtime.get('router')
    if router is None:
        return
    if all(hasattr(router, key) for key in ('cache', 'valid_keys', 'summaries', 'sketches')):
        if hasattr(router, 'invalidate'):
            router.invalidate()
        reset_router(router)
    elif hasattr(router, 'invalidate'):
        router.invalidate()
        cache = getattr(router, 'cache', None)
        if hasattr(cache, 'invalidate'):
            cache.invalidate()
    elif hasattr(getattr(router, 'cache', None), 'invalidate'):
        router.cache.invalidate()
    else:
        raise RuntimeError('router has no qualified reset path')


def _measure(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end), (time.perf_counter()-wall_start)*1000


def scalar_stats(result):
    """Untimed compact numeric identity; no full vocabulary logits retained."""
    value = result if torch.is_tensor(result) else None
    if value is None:
        return None
    finite = torch.isfinite(value)
    if not bool(finite.all()):
        return dict(all_finite=False)
    x = value.float()
    return dict(all_finite=True, mean=float(x.mean()),
                rms=float(x.square().mean().sqrt()), max_abs=float(x.abs().max()))


def counter_snapshot(runtime):
    if runtime is None:
        return {}
    counter = runtime.get('counters')
    if callable(counter):
        values = counter()
    else:
        router = runtime.get('router')
        values = router.counters() if router is not None and hasattr(router, 'counters') else {}
    cumulative = {'attention_calls', 'score_refresh_calls', 'decision_refresh_calls',
                  'held_decision_calls', 'bootstrap_calls', 'bitmap_observation_calls',
                  'current_qk_elements', 'reused_qk_elements',
                  'projected_current_v_tokens', 'reused_current_v_tokens',
                  'unsupported_mask_refreshes', 'summary_builds', 'summary_hits',
                  'summary_misses', 'preqk_consumer_calls', 'calls'}
    return {key: value for key, value in values.items()
            if key in cumulative and type(value) in (int, float)}


@contextmanager
def no_compile_during_accepted():
    """Triton hook fires only on a specialization miss; accept no such miss."""
    from triton.runtime.jit import JITFunction
    before, after = JITFunction.cache_hook, JITFunction.compiled_hook
    if before is not None or after is not None:
        raise RuntimeError('accepted timing requires no preexisting Triton compile hooks')
    misses = []
    def miss(**kwargs):
        misses.append(getattr(kwargs.get('fn'), 'name', '<unknown>'))
        return False
    JITFunction.cache_hook = miss
    try:
        yield misses
    finally:
        JITFunction.cache_hook, JITFunction.compiled_hook = before, after


STATIC_STATE_FIELDS = ('method', 'm_ref', 'beta', 'gamma', 'allocation', 'bootstrap',
                       'seed', 'diagnostics', 'collect_margins', 'fast_t')


def prepare_step(snapshot, state):
    static = {key: getattr(state, key) for key in STATIC_STATE_FIELDS if state is not None and hasattr(state, key)}
    kwargs = snapshot.prepare(controller=state)
    for key, value in static.items():
        setattr(state, key, value)
    return kwargs


def replay(model, sequence, runtime, boundary):
    """One repetition; snapshot restoration is outside every accepted timer."""
    reset_arm(runtime)
    state = runtime.get('state') if runtime else None
    router = runtime.get('router') if runtime else None
    if state is not None and boundary == 'denoising_step':
        from experiments.value_direction_hopper.query_adaptive import observe
        context = observe(model, state)
    else:
        context = nullcontext()
    rows = []
    with context:
        for step in sequence:
            kw = prepare_step(step['snapshot'], state)
            if state is not None and boundary == 'model_forward':
                # Native observe does this inside the whole-step boundary.
                state.begin(int(kw['cur_step']), kw['current_canvas'])
            input_hash = StepSnapshot.digest(kw, controller=state)
            before = counter_snapshot(runtime)
            fn = (lambda: decoder_call(model, kw)) if boundary == 'model_forward' else (lambda: model._denoising_step(**kw))
            result, event_ms, wall_ms = _measure(fn)
            after = counter_snapshot(runtime)
            phase_delta = {key: after.get(key, 0)-before.get(key, 0)
                           for key in set(before) | set(after)}
            rows.append(dict(call_index=step['call_index'], cur_step=step['cur_step'],
                             input_digest=input_hash, output_digest=output_digest(result),
                             output_stats=scalar_stats(result), event_ms=event_ms,
                             wall_ms=wall_ms, phase_delta=phase_delta))
    return rows


def replay_epoch(model, sequence, runtime, boundary):
    """Direct outer sequence span; native fixture restoration is included."""
    reset_arm(runtime)
    state = runtime.get('state') if runtime else None
    if state is not None and boundary == 'denoising_step':
        from experiments.value_direction_hopper.query_adaptive import observe
        context = observe(model, state)
    else:
        context = nullcontext()
    with context:
        def run():
            last = None
            for step in sequence:
                kw = prepare_step(step['snapshot'], state)
                if state is not None and boundary == 'model_forward':
                    state.begin(int(kw['cur_step']), kw['current_canvas'])
                last = decoder_call(model, kw) if boundary == 'model_forward' else model._denoising_step(**kw)
            return last
        last, event_ms, wall_ms = _measure(run)
    return dict(event_ms=event_ms, wall_ms=wall_ms,
                last_output_digest=output_digest(last),
                definition='one outer span including per-step snapshot/RNG/controller restoration; no per-call synchronization or digest inside')


def replay_untimed(model, sequence, runtime, boundary, *, require_finite=False):
    """Separate diagnostic replay; device-to-host bitmaps never enter accepted timing."""
    reset_arm(runtime)
    state = runtime.get('state') if runtime else None
    if state is not None and boundary == 'denoising_step':
        from experiments.value_direction_hopper.query_adaptive import observe
        context = observe(model, state)
    else:
        context = nullcontext()
    rows = []
    with context:
        for step in sequence:
            kw = prepare_step(step['snapshot'], state)
            if state is not None and boundary == 'model_forward':
                state.begin(int(kw['cur_step']), kw['current_canvas'])
            input_hash = StepSnapshot.digest(kw, controller=state)
            result = decoder_call(model, kw) if boundary == 'model_forward' else model._denoising_step(**kw)
            if require_finite:
                stats = scalar_stats(result)
                if stats is None or not stats['all_finite']:
                    raise ValueError('prepared support floor produced nonfinite or unsupported full-forward output')
            rows.append((input_hash, output_digest(result)))
    return rows


def qkv_signature(q, k, v):
    """Untimed, bit-sensitive per-call QKV identity for the prepared floor."""
    return (output_digest(q), output_digest(k), output_digest(v))


def counter_replay(model, sequence, runtime, boundary, accepted_rows, *, retain_support=False):
    """Actual router decisions, but with an explicit untimed physical counter twin."""
    if runtime is None or not all(k in runtime for k in ('binding', 'router')):
        return dict(status='N/A', reason='no numerical native-legal router/bitmap'), []
    router = runtime['router']
    owner = getattr(router, 'owner', router)
    retain_support = retain_support and hasattr(router, 'score_calls')
    fresh_native = getattr(router, 'support_geometry', None) == 'native_legal'
    if not fresh_native and (getattr(owner, 'support', None) != 'native_mask' or
                             getattr(owner, 'output_mode', None) != 'historical_route_preqk_current_output' or
                             not (hasattr(router, 'score_calls') or hasattr(router, 'bootstrap_calls'))):
        return dict(status='N/A', reason='fresh or unsupported bitmap consumer'), []
    from experiments.numerical_qk_reuse.v20_counter import install_counter_twin
    with install_counter_twin(runtime['binding'], router, explicit_untimed=True,
                              retain_support=retain_support,
                              qkv_digest=qkv_signature if retain_support else None) as twin:
        observed = replay_untimed(model, sequence, runtime, boundary)
    expected = [(r['input_digest'], r['output_digest']) for r in accepted_rows]
    if observed != expected:
        raise AssertionError('untimed physical-counter replay input/output digest drift')
    if not twin.rows:
        raise AssertionError('numerical router counter twin saw no attention calls')
    return dict(status='qualified', input_output_digests=observed, physical=twin.summary()), twin.support_calls


def prepared_support_floor(model, sequence, runtime, boundary, accepted_rows, support_calls,
                           repetitions):
    """Free per-call support provision, timed same-consumer full-forward diagnostic."""
    from experiments.numerical_qk_reuse.v20_counter import install_prepared_support_floor
    with install_prepared_support_floor(runtime['binding'], runtime['router'], support_calls,
                                        explicit_untimed=True, qkv_digest=qkv_signature) as floor:
        floor.reset()
        observed = replay_untimed(model, sequence, runtime, boundary, require_finite=True)
        floor.assert_complete()
        expected_inputs = [r['input_digest'] for r in accepted_rows]
        if [x[0] for x in observed] != expected_inputs:
            raise AssertionError('prepared-support floor input snapshot drift')
        qkv_matches = list(floor.qkv_matches)
        if len(qkv_matches) != len(support_calls):
            raise AssertionError('prepared-support floor QKV verification incomplete')
        # Only the verification pass digests QKV. Turn that work off for timing.
        floor.qkv_digest = None
        spans = []
        for _ in range(repetitions):
            floor.reset()
            span = replay_epoch(model, sequence, runtime, boundary)
            floor.assert_complete()
            spans.append(span)
    return dict(status='same_qkv' if all(qkv_matches) else 'numerically_divergent_path',
                definition='separate full-forward same-consumer replay; each call receives '
                'its own frozen observed bitmap without selector, score cache, sketch or '
                'support-construction work; later-layer QKV may change when a prior '
                'attention output changes; not deployable',
                repetitions=repetitions, event_median_ms=statistics.median(x['event_ms'] for x in spans),
                wall_median_ms=statistics.median(x['wall_ms'] for x in spans),
                qkv_matches_reference_untimed=qkv_matches,
                final_outputs_finite_on_untimed_replay=True,
                output_digest_matches_reference=[a[1] == r['output_digest']
                                                 for a, r in zip(observed, accepted_rows)],
                last_output_digests=[x['last_output_digest'] for x in spans],
                spans=spans)


def summarize_repetitions(reps):
    if not reps:
        return None
    lengths = {len(rep) for rep in reps}
    if len(lengths) != 1:
        raise AssertionError('sequence length drift')
    signatures = [[(r['input_digest'], r['output_digest'], r['output_stats'], r['phase_delta'])
                   for r in rep] for rep in reps]
    if any(sig != signatures[0] for sig in signatures[1:]):
        raise AssertionError('input/output/phase drift across repetitions')
    totals = [sum(r['event_ms'] for r in rep) for rep in reps]
    walls = [sum(r['wall_ms'] for r in rep) for rep in reps]
    return dict(sequence_calls=len(reps[0]), event_sum_median_ms=statistics.median(totals),
                event_sum_min_ms=min(totals), event_sum_max_ms=max(totals),
                wall_sum_median_ms=statistics.median(walls),
                per_call_event_median_ms=[statistics.median(rep[i]['event_ms'] for rep in reps)
                                          for i in range(len(reps[0]))],
                phase_deltas=[r['phase_delta'] for r in reps[0]],
                input_output_digests=[(r['input_digest'], r['output_digest']) for r in reps[0]],
                output_stats=[r['output_stats'] for r in reps[0]],
                definition='sum of directly timed complete calls; snapshot preparation excluded between calls')


def profile(config, checkpoint=None):
    from dllm.models import create_adapter
    from scripts.v9_clean_request_timing import disk_cache_entries, triton_specializations
    validate_config(config)
    paths = (config.get('manifests') or {'*': config['manifest']})
    preflight = preflight_identity(config, paths)
    manifest_hash = preflight['manifest_sha256']
    rows = {(dataset, id_): row for dataset, path in paths.items()
            for id_, row in load_rows(path).items()}
    def selected_row(target):
        return rows.get((target['dataset'], target['id'])) or rows.get(('*', target['id']))
    missing = sorted(f"{t['dataset']}:{t['id']}" for t in config['targets'] if selected_row(t) is None)
    if missing:
        raise ValueError(f'target IDs absent from manifest: {missing}')
    adapter = create_adapter('diffusion_gemma', str(config['model']), device='cuda',
                             precision='bfloat16', revision=config['revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    report = dict(schema='v20_direct_full_forward_v1', model=str(config['model']),
                  revision=config['revision'], manifest_sha256=manifest_hash,
                  reps=int(config['reps']), blocks=int(config.get('blocks', 3)),
                  seed=int(config.get('seed', 101)),
                  selected_boundaries=config.get('boundaries', ['model_forward', 'denoising_step']),
                  selected_sequence_lengths=config.get('sequence_lengths', [4, 16]),
                  counter_twins=bool(config.get('counter_twins', False)),
                  prepared_support_floor=bool(config.get('prepared_support_floor', False)),
                  arms=config['arms'],
                  runtime_identity=dict(python=sys.version.split()[0], executable=sys.executable,
                                        torch=torch.__version__, cuda=torch.version.cuda,
                                        gpu=torch.cuda.get_device_name(0), gpu_uuid=preflight['gpu_uuid'],
                                        hostname=preflight['hostname']),
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__),
                                              Path(__file__).with_name('replay_harness.py'),
                                              Path(__file__).with_name('v9_step_replay_profile.py'),
                                              Path(__file__).resolve().parents[1] / 'experiments' /
                                              'numerical_qk_reuse' / 'v20_counter.py')},
                  boundaries={'model_forward': 'full model.forward decoder/logits; state.begin outside timer; sampler excluded',
                              'denoising_step': 'native _denoising_step through observe: model, T/controller, sampler, stop/control'},
                  timing_note='CUDA event is elapsed stream span with host launch gaps, not GPU-active time; wall includes synchronization',
                  targets={})
    grouped = {}
    for target in config['targets']:
        grouped.setdefault((str(target['dataset']), str(target['id']), int(target['canvas'])), []).append(target)
    native_arm = config['arms'][0]
    for (dataset, id_, target_canvas), targets in grouped.items():
        with torch.inference_mode():
            row = selected_row(targets[0])
            captured, capture_proof = capture(adapter, row, targets, report['seed'])
            shared_boundaries = None
            for target in targets:
                canvas = target_canvas
                sequence = captured[canvas]
                key = f"{dataset}|{id_}|canvas{canvas}|call{target['call_index']}"
                resolution = resolve_target(target, captured)
                target_report = dict(dataset=target['dataset'], id=id_, canvas=canvas,
                                     requested_call=int(target['call_index']), resolution=resolution,
                                     request_contract=dict(max_new_tokens=row_budget(row),
                                                           thinking=row.get('thinking', True)),
                                     native_capture_path_proof=capture_proof,
                                     captured=[{k: v for k, v in step.items() if k != 'snapshot'} for step in sequence],
                                     snapshot_inventory=sequence[0]['snapshot'].inventory() if sequence else None,
                                     boundaries={})
                report['targets'][key] = target_report
                if resolution['missing']:
                    if checkpoint:
                        checkpoint(report)
                    continue
                if shared_boundaries is not None:
                    target_report['boundaries'] = shared_boundaries
                    if checkpoint:
                        checkpoint(report)
                    continue
                for boundary in report['selected_boundaries']:
                    for length in report['selected_sequence_lengths']:
                        seq = sequence[:length]
                        if not seq or (length == 16 and len(seq) <= 4):
                            continue
                        label = f'N{length}'
                        timing = {arm['name']: [] for arm in config['arms']}
                        epoch_timing = {arm['name']: [] for arm in config['arms']}
                        proof = {}
                        warmup = {}
                        # Compile/warm every encountered specialization before accepted blocks.
                        for arm in config['arms']:
                            with arm_context(adapter, arm) as runtime:
                                observations = [replay(model, seq, runtime, boundary)
                                                for _ in range(int(config.get('warmup', 1)))]
                                warmup[arm['name']] = dict(
                                    first_event_sum_ms=sum(r['event_ms'] for r in observations[0]),
                                    first_wall_sum_ms=sum(r['wall_ms'] for r in observations[0]),
                                    observations=len(observations))
                        for block in range(report['blocks']):
                            candidates = config['arms'][1:]
                            rotation = candidates[block % len(candidates):] + candidates[:block % len(candidates)] if candidates else []
                            order = [native_arm] + rotation + [native_arm]
                            for position, arm in enumerate(order):
                                with arm_context(adapter, arm) as runtime:
                                    if arm['condition'] == 'native_dense' and arm['name'] not in proof:
                                        with dispatch_spy(model) as counts:
                                            replay(model, seq[:1], runtime, boundary)
                                        assert_native_path(counts)
                                        proof[arm['name']] = counts
                                    reps_in_block = report['reps'] // report['blocks'] + (block < report['reps'] % report['blocks'])
                                    for rep in range(reps_in_block):
                                        cache_before = triton_specializations()
                                        disk_before = disk_cache_entries()
                                        torch.cuda.reset_peak_memory_stats()
                                        with no_compile_during_accepted() as misses:
                                            step_rows = replay(model, seq, runtime, boundary)
                                            epoch_row = replay_epoch(model, seq, runtime, boundary)
                                        cache_after = triton_specializations()
                                        disk_after = disk_cache_entries()
                                        if misses or cache_after != cache_before or disk_after != disk_before:
                                            raise RuntimeError(f'new JIT/cache activity in accepted block: {misses}')
                                        peak_bytes = torch.cuda.max_memory_allocated()
                                        timing[arm['name']].append(dict(block=block, bracket=('open' if position == 0 else 'close' if position == len(order)-1 else None),
                                                                        repetition=rep, rows=step_rows, peak_allocated_bytes=peak_bytes,
                                                                        triton_misses=0, triton_specializations_before=cache_before,
                                                                        triton_specializations_after=cache_after,
                                                                        triton_disk_entries_before=disk_before,
                                                                        triton_disk_entries_after=disk_after))
                                        epoch_timing[arm['name']].append(dict(block=block, bracket=('open' if position == 0 else 'close' if position == len(order)-1 else None),
                                                                              repetition=rep, **epoch_row))
                        diagnostics = {}
                        if config.get('counter_twins', False):
                            for arm in config['arms']:
                                with arm_context(adapter, arm) as runtime:
                                    accepted = timing[arm['name']][0]['rows']
                                    measured, support_calls = counter_replay(
                                        model, seq, runtime, boundary, accepted,
                                        retain_support=bool(config.get('prepared_support_floor', False)))
                                    if config.get('prepared_support_floor', False):
                                        measured['prepared_support_floor'] = (
                                            prepared_support_floor(model, seq, runtime, boundary,
                                                                   accepted, support_calls,
                                                                   int(config.get('floor_reps', 3)))
                                            if measured['status'] == 'qualified' and support_calls and
                                            boundary == 'model_forward'
                                            else dict(status='N/A', reason='requires numerical bitmap and model_forward'))
                                    diagnostics[arm['name']] = measured
                        target_report['boundaries'].setdefault(boundary, {})[label] = dict(
                            reached_calls=len(seq), requested_calls=length,
                            diagnostic_replays=diagnostics,
                            native_path_proof=proof,
                            native_bracket_drift=[dict(block=block,
                                open_median_ms=statistics.median(sum(r['event_ms'] for r in x['rows'])
                                  for x in timing[native_arm['name']] if x['block'] == block and x['bracket'] == 'open'),
                                close_median_ms=statistics.median(sum(r['event_ms'] for r in x['rows'])
                                  for x in timing[native_arm['name']] if x['block'] == block and x['bracket'] == 'close'))
                                for block in range(report['blocks'])],
                            arms={name: dict(warmup=warmup[name],
                                             summary=summarize_repetitions([x['rows'] for x in samples]),
                                             direct_epoch=dict(event_median_ms=statistics.median(x['event_ms'] for x in epoch_timing[name]),
                                                               wall_median_ms=statistics.median(x['wall_ms'] for x in epoch_timing[name]),
                                                               observations=epoch_timing[name]),
                                             blocks=[dict(block=x['block'], bracket=x['bracket'], repetition=x['repetition'],
                                                          event_sum_ms=sum(r['event_ms'] for r in x['rows']),
                                                          wall_sum_ms=sum(r['wall_ms'] for r in x['rows']),
                                                          peak_allocated_bytes=x['peak_allocated_bytes'],
                                                          triton_misses=x['triton_misses'],
                                                          triton_specializations_before=x['triton_specializations_before'],
                                                          triton_specializations_after=x['triton_specializations_after'],
                                                          triton_disk_entries_before=x['triton_disk_entries_before'],
                                                          triton_disk_entries_after=x['triton_disk_entries_after']) for x in samples])
                                  for name, samples in timing.items()})
                        if checkpoint:
                            checkpoint(report)
                shared_boundaries = target_report['boundaries']
    return report


def row_budget(row):
    return row.get('generation_budget', 8192)


def atomic_json(path, payload):
    """Atomic same-directory checkpoint; caller owns the exclusive run lock."""
    path = Path(path)
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + '.partial.json')
    failure = args.out.with_name(args.out.name + '.failure.json')
    lock = args.out.with_name(args.out.name + '.lock')
    occupied = [path for path in (args.out, partial, failure, lock,
                                  args.out.with_name(args.out.name + '.writing')) if path.exists()]
    if occupied:
        raise FileExistsError(f'profile output/receipt already exists: {occupied}')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    latest = {'completed_targets': []}
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        def checkpoint(report):
            latest['completed_targets'] = list(report['targets'])
            atomic_json(partial, report)
        report = profile(config, checkpoint=checkpoint)
        atomic_json(args.out, report)
        if partial.exists():
            partial.unlink()
    except BaseException as exc:
        atomic_json(failure, dict(schema='v20_profile_failure_v1',
                                  error_type=type(exc).__name__,
                                  completed_targets=latest['completed_targets'],
                                  partial_path=str(partial) if partial.exists() else None,
                                  config_sha256=(hashlib.sha256(config_bytes).hexdigest()
                                                 if 'config_bytes' in locals() else None)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
