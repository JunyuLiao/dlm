"""Native request worker for the bounded refresh diagnostic.

No native forward, sampler invocation, or RNG draw is added. Factorial audit
hashes are deliberately outside the clean runtime-headroom workers.
"""
from __future__ import annotations
import functools
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import types


def digest(tensor):
    import torch
    value = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_schedule(schedule):
    from experiments.numerical_qk_reuse.v31_refresh_diagnostic import POLICIES
    if not schedule or schedule[0].get('policy') != 'control':
        raise ValueError('explicit original-control implicit warmup required')
    if any(not {'policy', 'label'} <= set(row) or row['policy'] not in POLICIES for row in schedule):
        raise ValueError('unsupported frozen request schedule')
    return schedule


def install_state_trace(adapter, runtime):
    """Hash small states every call; large self-conditioning only at the event.

    The previous sampler's mutable state references expose the state that native
    preparation uses for the next forward. Q/K/V at the first treated GLOBAL layer
    and all currently held support maps supplement this prefix fingerprint.
    """
    import torch
    from vllm.model_executor.models import diffusion_gemma as dg
    state = dict(args=None, history=[], post=[], event_seen=False, sampler_calls=0)
    original_begin, original_sample = adapter.begin_request, dg._compiled_sample_step

    def compact(args):
        slots = args[1]
        return {name: digest(args[index][slots]) for name, index in
                (('canvas', 5), ('argmax', 6), ('step', 7), ('encoder', 8), ('confident', 9))}

    def before_event(ad, layer, q, b, prefix, n):
        args = state['args']
        if args is None:
            raise RuntimeError('eligible refresh has no preceding native sampler state')
        value = dict(history_sha256=signature(state['history']),
            compact_state=compact(args), self_conditioning_sha256=digest(args[10][args[1]]),
            cpu_rng_sha256=digest(torch.get_rng_state()), cuda_rng_sha256=digest(torch.cuda.get_rng_state(q.device)),
            canvas=ad.canvas_id, prepared_canvas_step_one_based=ad._canvas_step,
            event_call_zero_based=runtime.episode[1], prefix=prefix, query_rows=n, first_layer=layer,
            current_q=digest(q), current_k=digest(b['k']), current_v=digest(b['v']),
            maps={str(k): digest(v['kept']) for k, v in sorted(ad.mage_state.items()) if v.get('kept') is not None},
            progress=dict(trig_at=ad._trig_at, counts=copy.deepcopy(ad._mage_count)))
        state['event_seen'] = True
        return dict(prefix_signature=signature(value), state_fingerprints=value,
                    sampler_calls_before_event=state['sampler_calls'],
                    qualification='measured prefix and first GLOBAL inputs; not a cloned full-engine checkpoint')

    @functools.wraps(original_sample)
    def sampler(*args, **kwargs):
        active = (adapter.bound and adapter.step_ctx is not None
                  and runtime.policy in ('control', 'dense_fresh', 'dense_hold', 'sparse_fresh', 'sparse_hold'))
        result = original_sample(*args, **kwargs)
        if active:
            state['args'] = args
            state['sampler_calls'] += 1
            item = dict(call=state['sampler_calls'], **compact(args),
                        cuda_rng=digest(torch.cuda.get_rng_state()), cpu_rng=digest(torch.get_rng_state()))
            if not state['event_seen']:
                state['history'].append(item)
            elif len(state['post']) < 8:
                # Full self-conditioning of the first next state detects an
                # output effect before a sampled token or RNG diverges.
                if not state['post']:
                    item['self_conditioning'] = digest(args[10][args[1]])
                state['post'].append(item)
        return result

    def begin(self, *args, **kwargs):
        state.update(args=None, history=[], post=[], event_seen=False, sampler_calls=0)
        return original_begin(*args, **kwargs)

    adapter.begin_request = types.MethodType(begin, adapter)
    dg._compiled_sample_step = sampler
    runtime.before_event_callback = before_event
    return state


def bind(adapter, runtime, schedule, trace=None):
    begin, receipt = adapter.begin_request, adapter._value_receipt
    state = dict(index=-1, entry=None)

    def start(self, *args, **kwargs):
        index = state['index'] + 1
        if index >= len(schedule):
            raise RuntimeError('native request exceeded the frozen schedule')
        entry = schedule[index]
        if runtime is not None:
            runtime.set_policy(entry['policy'])
        elif entry['policy'] != 'control':
            raise RuntimeError('non-MAGE controls cannot change refresh policy')
        result = begin(*args, **kwargs)
        state.update(index=index, entry=entry)
        return result

    def read(self):
        value = dict(receipt(), refresh_label=state['entry']['label'],
                     refresh_request_index=state['index'], refresh_policy=state['entry']['policy'])
        if trace is not None:
            value['refresh_post_event_states'] = list(trace['post'])
        return value

    adapter.begin_request = types.MethodType(start, adapter)
    adapter._value_receipt = types.MethodType(read, adapter)
    return state


def main():
    from experiments.numerical_qk_reuse import vllm_adapter
    from experiments.numerical_qk_reuse.v31_refresh_diagnostic import install
    schedule = validate_schedule(json.loads(Path(os.environ['REFRESH_SCHEDULE']).read_text()))
    audit = os.environ.get('REFRESH_AUDIT') == '1'
    profile = os.environ.get('HEADROOM_PROFILE') == '1'
    if os.environ.get('REPEATS', '1') != '1':
        raise RuntimeError('the frozen schedule owns repeats')
    if (audit or profile) and os.environ.get('VALUE_CLEAN_TIMING') == '1':
        raise RuntimeError('diagnostic hooks are forbidden in clean timing')
    original_install = vllm_adapter.install_vllm_patches
    collectors = []

    def install_adapter(adapter):
        result = original_install(adapter)
        runtime = install(adapter, audit=audit) if adapter.arm == 'mage' else None
        trace = install_state_trace(adapter, runtime) if audit and runtime is not None and not profile else None
        bind(adapter, runtime, schedule, trace)
        if profile:
            from scripts.v31_refresh_headroom_profile import install as profile_install
            collectors.append(profile_install(adapter, output_dir=Path(os.environ['HEADROOM_OUTPUT']),
                profile_request_indices=(int(os.environ.get('HEADROOM_PROFILE_INDEX', '1')),), warmup_requests=1))
        return result

    vllm_adapter.install_vllm_patches = install_adapter
    if profile:
        # Dense arms have no V31 adapter. Their geometry audit already receives
        # native phase transitions without changing the graph execution mode.
        from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
        original_audit_install = NativeGeometryAudit.install

        def install_dense(audit_object):
            result = original_audit_install(audit_object)
            from scripts.v31_refresh_headroom_profile import install as profile_install
            collector = profile_install(None, output_dir=Path(os.environ['HEADROOM_OUTPUT']),
                profile_request_indices=(int(os.environ.get('HEADROOM_PROFILE_INDEX', '1')),), warmup_requests=1)
            collectors.append(collector)
            begin, receipt = audit_object.begin_request, audit_object.receipt

            def start():
                collector.begin_request()
                return begin()

            def finish():
                value = dict(receipt())
                value['headroom'] = collector.end_request(boundary_completed=True)
                return value

            audit_object.begin_request, audit_object.receipt = start, finish
            return result

        NativeGeometryAudit.install = install_dense
    try:
        runpy.run_path(os.environ['REFRESH_BASE_WORKER'], run_name='__main__')
    finally:
        for collector in reversed(collectors):
            collector.close()


if __name__ == '__main__':
    main()
