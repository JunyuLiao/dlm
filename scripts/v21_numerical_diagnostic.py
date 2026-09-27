"""Bounded v21 same-state numerical diagnostic; no generation-quality output.

Uses v20_profile's real native capture and snapshot replay. All probes are
untimed. QKV/logits/support live only within one request and never enter JSON.
The config has v20_profile's model, revision, manifests, manifest_sha256,
seed=101, and arms. Arms are native_dense then D_matched_legacy/new and
M3_R3_legacy/new/layout/combined using v21 nested effective configs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import torch

from scripts import v20_profile as parent
from scripts.replay_harness import StepSnapshot, output_digest
from scripts.v9_step_replay_profile import decoder_call, native_registry


PROTOCOL = (Path(__file__).resolve().parents[1] / 'results' /
            'fan_m1_m3_multidataset_20260927' / 'frozen_protocol.json')
ARM_MODES = {
    'D_matched_legacy': ('v20_control', 'legacy_bf16_scores', 'head_major'),
    'D_matched_new': ('v20_control', 'fp32_scores_bf16_pv', 'head_major'),
    'M3_R3_legacy': ('v20_method', 'legacy_bf16_scores', 'head_major'),
    'M3_R3_new': ('v20_method', 'fp32_scores_bf16_pv', 'head_major'),
    'M3_R3_layout': ('v20_method', 'legacy_bf16_scores', 'model_major'),
    'M3_R3_combined': ('v20_method', 'fp32_scores_bf16_pv', 'model_major'),
}


def frozen_targets(protocol):
    ids = protocol['ids']
    if any(len(ids.get(dataset, [])) < 2 for dataset in ('aime26', 'longbench_v2', 'ruler4k')):
        raise ValueError('v20 protocol lacks two IDs in each task')
    result = []
    for dataset in ('aime26', 'longbench_v2', 'ruler4k'):
        for i, id_ in enumerate(ids[dataset][:2]):
            canvas = 1 if i == 1 and dataset != 'ruler4k' else 0
            calls = (0, 3, 6) if canvas == 1 else (0, 1, 3)
            result.extend(dict(dataset=dataset, id=id_, canvas=canvas, call_index=call)
                          for call in calls)
    assert len(result) == 18
    return result


def validate_config(config):
    from experiments.numerical_qk_reuse import v21
    frozen = frozen_targets(json.loads(PROTOCOL.read_text()))
    targets = config.get('targets')
    if config.get('seed') != 101 or not isinstance(targets, list) or not targets or len(targets) > 18:
        raise ValueError('v21 targets must be an ordered frozen seed-101 subset of eighteen states')
    frozen_groups = [frozen[i:i+3] for i in range(0, len(frozen), 3)]
    selected_groups = [targets[i:i+3] for i in range(0, len(targets), 3)]
    if len(targets) % 3 or any(group not in frozen_groups for group in selected_groups) or (
            [frozen_groups.index(group) for group in selected_groups] !=
            sorted(set(frozen_groups.index(group) for group in selected_groups))):
        raise ValueError('v21 targets must keep each assigned frozen question triple in protocol order')
    arms = config.get('arms', [])
    if [a.get('name') for a in arms] != ['D_native', *ARM_MODES]:
        raise ValueError('v21 requires native-first fixed seven-arm diagnostic')
    if arms[0].get('condition') != 'native_dense':
        raise ValueError('first arm must be untouched native_dense')
    for arm in arms[1:]:
        if arm.get('plugin') != v21.PLUGIN:
            raise ValueError(f"{arm['name']} requires v21 wrapper plugin")
        kind, precision, layout = ARM_MODES[arm['name']]
        effective = arm['config']
        if (effective.get('parent_kind'), effective.get('output_score_precision'),
                effective.get('output_layout')) != (kind, precision, layout):
            raise ValueError(f"{arm['name']} output contract drift")
        v21.validate_effective(effective, arm['condition'])
        parent_config = effective['parent_config']
        if kind == 'v20_control' and arm['condition'] != v21.CONTROL_CONDITION:
            raise ValueError('D_matched must use v20_dense_consumer only')
        if kind == 'v20_control' and parent_config.get('v20_scope') != 'GLOBAL_ONLY_NATIVE_LOCAL':
            raise ValueError('D_matched must match the GLOBAL-only native-local scope')
        if kind == 'v20_method' and (parent_config.get('v20_arm') != 'M3_R3_A8_current_output' or
                                     parent_config.get('v20_scope') != 'GLOBAL_ONLY_NATIVE_LOCAL'):
            raise ValueError('M3_R3 must use frozen GLOBAL-only A8/R3 method')


def preflight(config):
    validate_config(config)
    paths = config.get('manifests') or {'*': config['manifest']}
    # v20 identity checker reads flat arm configs. The nested parent alone is
    # flattened for that existing source/model/manifest verification.
    flattened = {**config, 'arms': [config['arms'][0]] + [
        {**a, 'config': a['config']['parent_config']} for a in config['arms'][1:]]}
    proof = parent.preflight_identity(flattened, paths)
    from experiments.numerical_qk_reuse import v21
    for arm in config['arms'][1:]:
        v21.validate_effective(arm['config'], arm['condition'])
    return paths, proof


def target_key(target):
    return f"{target['dataset']}|{target['id']}|canvas{target['canvas']}|call{target['call_index']}"


def cache_structure(cache):
    """Structural/prefix owner check; inference tensors may lack versions.

    An ``inference_untracked`` marker is explicit: this does not claim a
    content proof when PyTorch does not expose a version counter. Native
    encoder prefix ownership is immutable within the captured canvas.
    """
    def one(tensor):
        if tensor is None:
            return None
        try:
            version = int(tensor._version)
        except RuntimeError:
            version = 'inference_untracked'
        return (id(tensor), tensor.data_ptr(), tuple(tensor.shape),
                tuple(tensor.stride()), str(tensor.dtype), version)
    return tuple((one(layer.keys), one(layer.values)) for layer in cache.layers)


def capture_checked(adapter, row, targets, seed):
    """Use v20 capture and its StepSnapshot, recording cache owner versions."""
    original = parent.StepSnapshot
    class IdentitySnapshot(original):
        def __init__(self, kwargs, *, controller=None):
            super().__init__(kwargs, controller=controller)
            self.cache_identity_at_capture = cache_structure(kwargs['past_key_values'])
    parent.StepSnapshot = IdentitySnapshot
    try:
        captured, proof = parent.capture(adapter, row, targets, seed)
    finally:
        parent.StepSnapshot = original
    for steps in captured.values():
        for step in steps:
            snapshot = step['snapshot']
            if cache_structure(snapshot.plain['past_key_values']) != snapshot.cache_identity_at_capture:
                raise RuntimeError('referenced native cache changed after pre-step capture')
            if parent._prefix(snapshot.plain['past_key_values']) != step['prefix']:
                raise RuntimeError('native cache position changed after pre-step capture')
    return captured, proof


@torch.inference_mode()
def error_metrics(native, other):
    if native.shape != other.shape:
        raise ValueError(f'comparison shape mismatch {tuple(native.shape)} vs {tuple(other.shape)}')
    a, b = native.float(), other.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    total = int(a.numel())
    if not bool(finite.all()):
        return dict(elements=total, nonfinite_native=int((~torch.isfinite(a)).sum()),
                    nonfinite_other=int((~torch.isfinite(b)).sum()), finite_comparison=False)
    delta = (a-b).abs()
    denominator = max(float(torch.linalg.vector_norm(a)), 1e-30)
    # Full max/L2 are exact. A fixed-stride sample caps the tail quantile's
    # memory/work on 256 x vocabulary logits without random selection.
    flat = delta.flatten()
    stride = max(1, math.ceil(flat.numel() / 1_048_576))
    sampled = flat[::stride]
    return dict(elements=total, finite_comparison=True, native_norm_denominator=denominator,
                relative_l2=float(torch.linalg.vector_norm(a-b))/denominator,
                max_abs=float(delta.max()), p99_abs=float(torch.quantile(sampled, .99)),
                p99_method='deterministic_stride_sample', p99_sample_stride=stride,
                p99_sample_elements=int(sampled.numel()),
                top1_mismatch_positions=int((a.argmax(-1) != b.argmax(-1)).sum()))


def tensor_metrics(observed, reference):
    values = error_metrics(reference, observed)
    if 'native_norm_denominator' in values:
        values['reference_norm_denominator'] = values.pop('native_norm_denominator')
    values.pop('top1_mismatch_positions', None)
    return values


@torch.inference_mode()
def signal_probe(snapshot, logits):
    """Run actual processor and stopper on the restored pre-step objects."""
    kw = parent.prepare_step(snapshot, None)
    prior_finished = kw['finished_denoising']
    if bool(prior_finished.any()):
        # Native applies a finished-row override before the criterion. The
        # frozen batch-one target is expected unfinished; fail this stage
        # rather than evaluate different logits on a completed row.
        raise ValueError('same-history signal probe requires unfinished native pre-step row')
    processor = kw.get('logits_processor')
    processed = (logits if processor is None else
                 processor(kw['input_ids'], logits, cur_step=kw['cur_step']))
    top = processed.argmax(-1)
    stopper = kw.get('diffusion_stopping_criteria')
    if stopper is None:
        raise ValueError('native diffusion stopping criterion absent')
    entropy = torch.distributions.Categorical(logits=processed).entropy().mean(-1)
    threshold = float(stopper.confidence_threshold)
    if abs(threshold - .005) > 1e-9:
        raise ValueError('native confidence threshold changed')
    history = stopper.argmax_canvas_history
    stable = (False if history is None else bool((history == top[None]).all()))
    stop = stopper(top, processed)
    return dict(processor_type=(type(processor).__name__ if processor is not None else None),
                stopper_type=type(stopper).__name__, confidence_threshold=threshold,
                processed_entropy=float(entropy.mean()), confident=bool((entropy < threshold).all()),
                stable=stable, native_criterion_stop=bool(stop.all()),
                prior_finished=False, effective_stop=bool(stop.all()),
                top_digest=output_digest(top))


def step_metrics(result):
    if not isinstance(result, (tuple, list)) or len(result) < 4:
        raise ValueError('unexpected native denoising step result')
    return dict(return_stop=bool(result[3].all()), current_canvas_digest=output_digest(result[0]),
                argmax_digest=output_digest(result[1]), self_condition_digest=output_digest(result[2]))


def qkv_meta(record):
    q, k, v = record['q'], record['k'], record['v']
    return dict(layer=5, q_shape=list(q.shape), k_shape=list(k.shape), v_shape=list(v.shape),
                q_stride=list(q.stride()), k_stride=list(k.stride()), v_stride=list(v.stride()),
                v_contiguous_would_copy=not v.is_contiguous(),
                v_contiguous_copy_bytes_if_taken=(v.numel()*v.element_size() if not v.is_contiguous() else 0),
                native_output_shape=list(record['output'].shape),
                native_output_stride=list(record['output'].stride()))


def output_storage(output):
    view = output.output
    exposed = view.transpose(1, 2)
    return dict(logical_shape=list(view.shape), logical_stride=list(view.stride()),
                storage_offset=int(view.storage_offset()),
                transpose_already_contiguous=bool(exposed.is_contiguous()),
                transpose_copy_bytes_if_taken=(0 if exposed.is_contiguous()
                                               else view.numel()*view.element_size()))


@contextmanager
def native_layer5(registry, native, sink):
    if registry['sdpa'] is not native:
        raise ValueError('native layer probe requires untouched SDPA registry')
    def observe(module, q, k, v, *args, **kwargs):
        result = native(module, q, k, v, *args, **kwargs)
        if int(getattr(module, 'layer_idx', -1)) == 5:
            out = result[0] if isinstance(result, (tuple, list)) else result
            sink.append(dict(module=module, q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
                             native_args=args, native_kwargs=kwargs,
                             mask=args[0] if args else kwargs.get('attention_mask'),
                             is_causal=kwargs.get('is_causal'), window=kwargs.get('sliding_window'),
                             scale=kwargs.get('scaling'), output=out.detach().clone()))
        return result
    registry['sdpa'] = observe
    try:
        yield
    finally:
        registry['sdpa'] = native


@contextmanager
def selected_layer5(runtime, sink, *, need_support):
    """Observe real post-call M3 decision without altering support or output."""
    binding, router = runtime['binding'], runtime['router']
    old = binding.runtime.attention_override
    owner = getattr(router, 'owner', router)
    class Observer:
        def __call__(self, module, q, k, v, *args, **kwargs):
            layer5 = int(getattr(module, 'layer_idx', -1)) == 5
            outputs = []
            if layer5:
                from experiments.numerical_qk_reuse import integration
                original_attention, original_consume = integration.attention, owner._consume
                def attention_spy(*a, **kw):
                    value = original_attention(*a, **kw)
                    outputs.append(output_storage(value))
                    return value
                def consume_spy(*a, **kw):
                    value = original_consume(*a, **kw)
                    outputs.append(output_storage(value))
                    return value
                integration.attention, owner._consume = attention_spy, consume_spy
            try:
                result = old(module, q, k, v, *args, **kwargs)
            finally:
                if layer5:
                    integration.attention, owner._consume = original_attention, original_consume
            if layer5:
                if len(outputs) != 1:
                    raise ValueError('layer-5 output storage hook expected one consumer')
                record = dict(q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
                              scale=float(kwargs.get('scaling') or q.shape[-1]**-.5),
                              output_storage=outputs[0])
                if need_support:
                    entry = owner.cache.entries.get(5)
                    if entry is None or entry.decision is None:
                        raise ValueError('layer-5 M3 decision not published')
                    record.update(skipped=entry.decision.skipped.detach().clone(),
                                  eligible=entry.decision.eligible.detach().clone(),
                                  score_anchor_call=int(entry.score_step),
                                  decision_anchor_call=int(entry.decision_step),
                                  phase=('A' if entry.score_step == owner.step else
                                         'D' if entry.decision_step == owner.step else 'H'))
                sink.append(record)
            return result
        def __getattr__(self, name):
            return getattr(old, name)
    binding.runtime.attention_override = Observer()
    try:
        yield
    finally:
        binding.runtime.attention_override = old


def normalize_native_output(record):
    out = record['output']
    q = record['q']
    if out.shape == q.shape:
        return out
    if out.shape == (q.shape[0], q.shape[2], q.shape[1], q.shape[3]):
        return out.transpose(1, 2)
    raise ValueError('unexpected native SDPA output layout')


@torch.inference_mode()
def full_fp32_attention(record, skipped=None, eligible=None):
    """Independent full FP32 QK/scale/softmax/PV mathematical reference."""
    from dllm.attention.blasst.core import _attention_validity
    q, k, v = (record[name] for name in ('q', 'k', 'v'))
    if record.get('mask') is not None or record.get('is_causal') or record.get('window'):
        raise ValueError('operator reference requires observed bidirectional mask=None geometry')
    b, h, nq, d = q.shape
    nk, hk = k.shape[2], k.shape[1]
    scale = float(record.get('scale') or d**-.5)
    keys = k.float().repeat_interleave(h//hk, dim=1)
    values = v.float().repeat_interleave(h//hk, dim=1)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        scores = torch.matmul(q.float(), keys.transpose(-1, -2)) * scale
        legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
        if skipped is not None:
            support = torch.ones_like(scores, dtype=torch.bool)
            for qb in range(skipped.shape[2]):
                for kt in range(skipped.shape[3]):
                    support[:, :, qb*128:min((qb+1)*128, nq), kt*64:min((kt+1)*64, nk)] &= (
                        (eligible[:, :, qb, kt] & ~skipped[:, :, qb, kt])[:, :, None, None])
            legal = legal & support
        scores = scores.masked_fill(~legal, -math.inf)
        probs = scores.softmax(-1)
        probs = torch.where(torch.isfinite(probs), probs, 0.)
        return probs @ values
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@torch.inference_mode()
def operator_probe(record):
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention
    q, k, v = (record[name] for name in ('q', 'k', 'v'))
    b, h, nq, d = q.shape
    nk = k.shape[2]
    shape = (b, h, (nq+127)//128, (nk+63)//64)
    skip = torch.zeros(shape, device=q.device, dtype=torch.bool)
    eligible = torch.ones_like(skip)
    scale = float(record.get('scale') or d**-.5)
    old = preqk_attention(q, k, v, skip, eligible, scale=scale, variant='generic')
    new = preqk_attention(q, k, v, skip, eligible, scale=scale, variant='generic',
                          output_score_precision='fp32_scores_bf16_pv')
    reference = full_fp32_attention(record)
    native = normalize_native_output(record)
    result = dict(qkv=qkv_meta(record), all_kept_tiles=int(skip.numel()),
                  native_vs_full_fp32=tensor_metrics(native, reference),
                  legacy_vs_full_fp32=tensor_metrics(old.output, reference),
                  fp32_bf16pv_vs_full_fp32=tensor_metrics(new.output, reference),
                  legacy_vs_native=tensor_metrics(old.output, native),
                  fp32_bf16pv_vs_native=tensor_metrics(new.output, native),
                  invalid_legacy=int(old.invalid_scores.sum()), invalid_new=int(new.invalid_scores.sum()))
    return result


@torch.inference_mode()
def same_support_probe(record):
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention
    q, k, v = (record[name] for name in ('q', 'k', 'v'))
    skip, eligible = record['skipped'], record['eligible']
    frozen = output_digest((skip, eligible))
    scale = record['scale']
    old = preqk_attention(q, k, v, skip, eligible, scale=scale, variant='generic')
    new = preqk_attention(q, k, v, skip, eligible, scale=scale, variant='generic',
                          output_score_precision='fp32_scores_bf16_pv')
    reference = full_fp32_attention(record, skip, eligible)
    if output_digest((skip, eligible)) != frozen:
        raise AssertionError('same-support operator probe mutated the decision bitmap')
    return dict(support_digest=frozen, eligible_tiles=int(eligible.sum()),
                skipped_tiles=int((eligible & skip).sum()),
                legacy_vs_full_fp32=tensor_metrics(old.output, reference),
                fp32_bf16pv_vs_full_fp32=tensor_metrics(new.output, reference),
                old_vs_new=tensor_metrics(old.output, new.output),
                invalid_legacy=int(old.invalid_scores.sum()), invalid_new=int(new.invalid_scores.sum()))


@torch.inference_mode()
def native_kernel_trace(model, record):
    from torch.profiler import ProfilerActivity, profile
    registry, native = native_registry(model)
    if registry['sdpa'] is not native:
        raise ValueError('native profiler requires untouched registry')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        native(record['module'], record['q'], record['k'], record['v'],
               *record['native_args'], **record['native_kwargs'])
        torch.cuda.synchronize()
    names = Counter(event.name for event in trace.events()
                    if str(event.device_type).endswith('CUDA'))
    return dict(schema='actual_native_cuda_profiler_events_v1', cuda_event_names=dict(names),
                assertion='recorded kernel names, no backend inferred from package version')


def _error_record(exc, group, stage):
    frames = traceback.extract_tb(exc.__traceback__)
    last = frames[-1] if frames else None
    return dict(group=group, stage=stage, error_type=type(exc).__name__,
                code_location=(None if last is None else
                               dict(file=Path(last.filename).name, function=last.name, line=last.lineno)))


def _safe_stage(report, group, stage, fn):
    try:
        return fn()
    except Exception as exc:
        report['stage_errors'].append(_error_record(exc, group, stage))
        return None


@torch.inference_mode()
def replay_forward_group(model, sequence, selected, runtime, *, native=False,
                         capture_support=False, need_support=False):
    """One canvas from call zero, restoring every native pre-step snapshot."""
    parent.reset_arm(runtime)
    state = runtime.get('state') if runtime else None
    registry, native_fn = native_registry(model)
    outputs, qkv, supports, inputs = {}, {}, {}, {}
    wanted = set(selected)
    for step in sequence[:max(wanted)+1]:
        idx = int(step['call_index'])
        kw = parent.prepare_step(step['snapshot'], state)
        if state is not None:
            state.begin(int(kw['cur_step']), kw['current_canvas'])
        if idx in wanted:
            inputs[idx] = StepSnapshot.digest(kw, controller=state)
        native_sink, support_sink = [], []
        if idx in wanted and native:
            with native_layer5(registry, native_fn, native_sink):
                result = decoder_call(model, kw)
            if len(native_sink) != 1:
                raise RuntimeError('native layer-5 QKV hook did not see exactly one call')
            qkv[idx] = native_sink[0]
        elif idx in wanted and capture_support:
            with selected_layer5(runtime, support_sink, need_support=need_support):
                result = decoder_call(model, kw)
            if len(support_sink) != 1:
                raise RuntimeError('M3 layer-5 support hook did not see exactly one call')
            supports[idx] = support_sink[0]
        else:
            result = decoder_call(model, kw)
        if idx in wanted:
            outputs[idx] = result.detach().clone()
    return dict(logits=outputs, qkv=qkv, support=supports, input_digests=inputs)


@torch.inference_mode()
def replay_step_group(model, sequence, selected, runtime):
    """Separate reset and observe; never reuse forward replay's router state."""
    from experiments.value_direction_hopper.query_adaptive import observe
    from contextlib import nullcontext
    parent.reset_arm(runtime)
    state = runtime.get('state') if runtime else None
    context = observe(model, state) if state is not None else nullcontext()
    rows = {}
    with context:
        for step in sequence[:max(selected)+1]:
            idx = int(step['call_index'])
            kw = parent.prepare_step(step['snapshot'], state)
            result = model._denoising_step(**kw)
            if idx in selected:
                rows[idx] = step_metrics(result)
    return rows


def _layout_report(group_report):
    arms = group_report['arms']
    comparisons = {}
    for left, right in (('M3_R3_legacy', 'M3_R3_layout'),
                        ('M3_R3_new', 'M3_R3_combined')):
        if left not in arms or right not in arms:
            continue
        comparisons[f'{left}|{right}'] = {}
        for idx in set(arms[left]['output_digests']) & set(arms[right]['output_digests']):
            same_output = arms[left]['output_digests'][idx] == arms[right]['output_digests'][idx]
            same_support = (arms[left]['support_digests'].get(idx) ==
                            arms[right]['support_digests'].get(idx))
            comparisons[f'{left}|{right}'][idx] = dict(exact_output_elements=same_output,
                                                       identical_support=same_support)
            if not same_output or not same_support:
                raise AssertionError('layout-only pair changed output elements or support')
    return comparisons


def _support_report(group_report):
    arms = group_report['arms']
    names = [name for name in ARM_MODES if name.startswith('M3_R3_') and name in arms]
    result = {}
    for idx in set.intersection(*(set(arms[name]['support_digests']) for name in names)) if names else ():
        digests = {name: arms[name]['support_digests'][idx] for name in names}
        phases = {name: arms[name]['support_phases'][idx] for name in names}
        result[idx] = dict(identical_bitmaps=len(set(digests.values())) == 1,
                           phases=phases, support_digest=next(iter(digests.values())))
        # Precision can change earlier-layer outputs and hence layer-5 QKV on
        # later calls. This is descriptive; exactness is asserted only for
        # layout-only pairs in _layout_report. The frozen-support operator
        # probe evaluates both precisions on legacy QKV and bitmap.
        if int(idx) == 0 and any(phase != 'A' for phase in phases.values()):
            raise AssertionError('call-zero M3 replay did not create an A anchor')
    return result


def run(config, checkpoint=None):
    from dllm.models import create_adapter
    paths, proof = preflight(config)
    rows = {(dataset, id_): row for dataset, path in paths.items()
            for id_, row in parent.load_rows(path).items()}
    def select(target):
        return rows.get((target['dataset'], target['id'])) or rows.get(('*', target['id']))
    if any(select(target) is None for target in config['targets']):
        raise ValueError('frozen diagnostic ID absent from gold-free manifest')
    adapter = create_adapter('diffusion_gemma', str(config['model']), device='cuda',
                             precision='bfloat16', revision=config['revision']).load()
    model = adapter.model
    # Match v20_profile's qualified replay setting; the independent reference
    # also scopes its own FP32 matmuls and restores this value afterward.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(schema='v21_numerical_diagnostic_v1',
                  quality_eligible=False, timing_eligible=False,
                  full_fp32_network_claim=False, model=str(config['model']),
                  revision=config['revision'], seed=101, preflight=proof,
                  protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                  source_sha256={str(Path(__file__).resolve()):
                                 hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  runtime_identity=dict(python=sys.version.split()[0], torch=torch.__version__,
                                        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                        hostname=proof['hostname'], gpu_uuid=proof['gpu_uuid'],
                                        tf32_matmul_at_native_capture=torch.backends.cuda.matmul.allow_tf32,
                                        tf32_cudnn_at_native_capture=torch.backends.cudnn.allow_tf32),
                  arms=[dict(name=a['name'], condition=a['condition'], plugin=a.get('plugin'),
                             fingerprint=a['config'].get('fingerprint'),
                             precision=a['config'].get('output_score_precision'),
                             layout=a['config'].get('output_layout'),
                             parent_kind=a['config'].get('parent_kind')) for a in config['arms']],
                  groups={}, stage_errors=[], trace_status='pending')
    grouped = {}
    for target in config['targets']:
        grouped.setdefault((target['dataset'], target['id'], target['canvas']), []).append(target)
    trace_attempted = False
    for (dataset, id_, canvas), targets in grouped.items():
        group_key = f'{dataset}|{id_}|canvas{canvas}'
        group = dict(dataset=dataset, id=id_, canvas=canvas, targets={}, arms={},
                     preparation_calls=[], capture_status='pending')
        report['groups'][group_key] = group
        if checkpoint:
            checkpoint(report)
        try:
            with torch.inference_mode():
                captured, proof_counts = capture_checked(adapter, select(targets[0]), targets, 101)
            sequence = captured[canvas]
            group['capture_status'] = 'ok'
            group['native_capture_path_proof'] = proof_counts
            group['captured_calls'] = [dict(call_index=s['call_index'], cur_step=s['cur_step'],
                                            prefix=s['prefix'], context=s['context']) for s in sequence]
        except Exception as exc:
            group['capture_status'] = 'failed'
            report['stage_errors'].append(_error_record(exc, group_key, 'capture'))
            if checkpoint:
                checkpoint(report)
            continue
        selected = []
        step_by_index = {int(step['call_index']): step for step in sequence}
        for target in targets:
            resolution = parent.resolve_target(target, captured)
            idx = int(target['call_index'])
            group['targets'][str(idx)] = dict(resolution=resolution,
                                               status=('missing' if resolution['missing'] else 'captured'))
            if not resolution['missing']:
                selected.append(idx)
        group['preparation_calls'] = [s['call_index'] for s in sequence[:max(selected)+1]
                                      if s['call_index'] not in selected] if selected else []
        if checkpoint:
            checkpoint(report)
        if not selected:
            del captured, sequence
            continue
        native = _safe_stage(report, group_key, 'native_forward', lambda:
            replay_forward_group(model, sequence, selected, None, native=True))
        if native is None:
            group['dependent_arms'] = 'skipped_after_native_forward_failure'
            if checkpoint:
                checkpoint(report)
            del captured, sequence
            continue
        group['native_input_digests'] = native['input_digests']
        group['native_output_digests'] = {str(i): output_digest(x) for i, x in native['logits'].items()}
        for i in selected:
            record = native['qkv'][i]
            group['targets'][str(i)]['operator'] = _safe_stage(
                report, group_key, f'operator_call{i}', lambda record=record: operator_probe(record))
            if not trace_attempted:
                trace_attempted = True
                report['native_kernel_trace'] = _safe_stage(
                    report, group_key, 'native_kernel_trace',
                    lambda record=record: native_kernel_trace(model, record))
                report['trace_status'] = 'ok' if report['native_kernel_trace'] is not None else 'failed'
        native_steps = _safe_stage(report, group_key, 'native_step', lambda:
            replay_step_group(model, sequence, selected, None))
        group['native_step'] = native_steps
        for i in selected:
            group['targets'][str(i)]['native_signal'] = _safe_stage(
                report, group_key, f'native_signal_call{i}',
                lambda i=i: signal_probe(step_by_index[i]['snapshot'], native['logits'][i]))
        if checkpoint:
            checkpoint(report)
        for arm in config['arms'][1:]:
            name = arm['name']
            try:
                with torch.inference_mode(), parent.arm_context(adapter, arm) as runtime:
                    capture_support = True
                    measured = replay_forward_group(model, sequence, selected, runtime,
                                                    capture_support=capture_support,
                                                    need_support=name.startswith('M3_R3_'))
                    steps = _safe_stage(report, group_key, f'{name}_step', lambda:
                        replay_step_group(model, sequence, selected, runtime))
                    counters = runtime['counters']()
                arm_report = dict(status='ok', output_digests={str(i): output_digest(x)
                                                               for i, x in measured['logits'].items()},
                                  input_digests=measured['input_digests'], step=steps,
                                  support_digests={str(i): output_digest((s['skipped'], s['eligible']))
                                                   for i, s in measured['support'].items() if 'skipped' in s},
                                  support_phases={str(i): s['phase'] for i, s in measured['support'].items()
                                                  if 'phase' in s},
                                  layer5_output_storage={str(i): s['output_storage']
                                                         for i, s in measured['support'].items()},
                                  output_score_precision=counters.get('output_score_precision'),
                                  output_layout=counters.get('output_layout'),
                                  output_precision_extra_qk_elements_upper_bound=
                                      counters.get('output_precision_extra_qk_elements_upper_bound'))
                for i in selected:
                    key = str(i)
                    arm_report.setdefault('targets', {})[key] = dict(
                        logits_vs_native=error_metrics(native['logits'][i], measured['logits'][i]),
                        same_history_signal=_safe_stage(report, group_key, f'{name}_signal_call{i}',
                            lambda i=i: signal_probe(step_by_index[i]['snapshot'], measured['logits'][i])))
                    if steps is not None and native_steps is not None:
                        arm_report['targets'][key]['step_vs_native'] = dict(
                            return_stop_changed=(steps[i]['return_stop'] != native_steps[i]['return_stop']),
                            current_canvas_changed=(steps[i]['current_canvas_digest'] !=
                                                    native_steps[i]['current_canvas_digest']),
                            argmax_changed=(steps[i]['argmax_digest'] != native_steps[i]['argmax_digest']),
                            self_condition_changed=(steps[i]['self_condition_digest'] !=
                                                    native_steps[i]['self_condition_digest']))
                    if name == 'M3_R3_legacy':
                        support = measured['support'][i]
                        arm_report['targets'][key]['same_support_operator'] = _safe_stage(
                            report, group_key, f'same_support_call{i}',
                            lambda support=support: same_support_probe(support))
                group['arms'][name] = arm_report
            except Exception as exc:
                group['arms'][name] = dict(status='failed', error_type=type(exc).__name__)
                report['stage_errors'].append(_error_record(exc, group_key, name))
            if checkpoint:
                checkpoint(report)
        group['layout_exactness'] = _safe_stage(report, group_key, 'layout_exactness',
                                                lambda: _layout_report(group))
        group['m3_support_identity'] = _safe_stage(report, group_key, 'm3_support_identity',
                                                   lambda: _support_report(group))
        for i in selected:
            group['targets'][str(i)]['status'] = 'diagnosed'
        if checkpoint:
            checkpoint(report)
        del native, captured, sequence
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + '.partial.json')
    failure = args.out.with_name(args.out.name + '.failure.json')
    lock = args.out.with_name(args.out.name + '.lock')
    occupied = [p for p in (args.out, partial, failure, lock,
                            args.out.with_name(args.out.name + '.writing')) if p.exists()]
    if occupied:
        raise FileExistsError(f'diagnostic output already exists: {occupied}')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        report = run(config, checkpoint=lambda result: parent.atomic_json(partial, result))
        parent.atomic_json(args.out, report)
        partial.unlink(missing_ok=True)
    except BaseException as exc:
        parent.atomic_json(failure, dict(schema='v21_numerical_diagnostic_failure_v1',
                                         error_type=type(exc).__name__,
                                         config_sha256=(hashlib.sha256(config_bytes).hexdigest()
                                                        if 'config_bytes' in locals() else None),
                                         partial_path=str(partial) if partial.exists() else None))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
