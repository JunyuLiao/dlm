"""v14 CP2: complete-forward cost of CVM-T vs matched controls on REAL AIME states.

Capture (untouched native model): N consecutive real denoising steps of a chosen
canvas (StepSnapshot per step: kwargs, native sampler / stopping / processors,
CPU+CUDA RNG, pre-step T controller).
Replay per arm (teacher-forced, identical inputs across arms): before EVERY step the
snapshot is restored (RNG, native objects, controller); the arm's own router state
(anchors, keep masks, sketches) evolves naturally across the sequence, so the natural
phase sequence appears: step 0 = new-canvas anchor, 1..7 ordinary/held, 8 = A8 anchor.
Each step is a synchronized CUDA-event span of one complete denoising step
(decoder + native sampler + T bookkeeping + method work). Also: complete GLOBAL
attention calls for a GLOBAL layer, per phase, on captured raw inputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


class Stop(Exception):
    pass


def capture_sequence(adapter, row, canvas, n, seed):
    from dllm.models import GenerationRequest
    from experiments.value_direction_hopper.query_adaptive import State, observe
    from scripts.replay_harness import StepSnapshot
    model = adapter.model
    controller = State('T', None, m_ref=14.258454322814941, beta=3., gamma=.5, diagnostics=False, fast_t=True)
    steps, count = [], {'canvas': -1}
    with observe(model, controller):
        inner = model._denoising_step

        def outer(this, **kwargs):
            cur = int(kwargs['cur_step'])
            if cur == 48:
                count['canvas'] += 1
            if count['canvas'] == canvas:
                steps.append(dict(snapshot=StepSnapshot(kwargs, controller=controller), cur_step=cur,
                                  absolute=int(kwargs['past_key_values'].get_seq_length())))
                result = inner(**kwargs)
                if len(steps) == n or bool(result[3].all()):
                    raise Stop()
                return result
            if count['canvas'] > canvas:
                raise Stop()
            return inner(**kwargs)
        model._denoising_step = MethodType(outer, model)
        try:
            adapter.generate(GenerationRequest(prompt=row['prompt'], max_new_tokens=8192, temperature=0.0, seed=seed,
                                               extra={'thinking': True}))
        except Stop:
            pass
        finally:
            model._denoising_step = inner
    return steps


def arm_runtime(adapter, arm, ns):
    from contextlib import nullcontext
    from experiments.numerical_qk_reuse.global_scope import install
    from scripts.v10_request_runs import arm_config
    if arm['condition'] == 'native_dense':
        return nullcontext(None)
    if arm['condition'] in ('noop_binding', 'controller_only'):
        return overhead_control(adapter, binding=arm['condition'] == 'noop_binding', controller=bool(arm.get('controller')))
    return install(adapter, arm_config(ns, arm), arm['condition'])


@contextmanager
def overhead_control(adapter, *, binding, controller):
    """Overhead-isolation controls (profile only, never answer runs): the GLOBAL-only
    binding with a pass-through override (same _MaskGuard, native SDPA on every call)
    and/or the exact fast-T controller observing the native loop (no router)."""
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
    from experiments.numerical_qk_reuse.global_scope import GlobalScopeAdapter, _MaskGuard
    from experiments.value_direction_hopper.query_adaptive import State
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    bound = None
    if binding:
        bound = _install_dense(GlobalScopeAdapter(adapter))
        bound.runtime.attention_override = _MaskGuard(lambda module, q, k, v, mask, **kw:
                                                      sdpa_attention_forward(module, q, k, v, mask, **kw))
    try:
        state = State('T', None, m_ref=14.258454322814941, beta=3., gamma=.5, diagnostics=False, fast_t=True) if controller else None
        yield dict(binding=bound, router=None, state=state)
    finally:
        if bound is not None:
            bound.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--arms', required=True)
    parser.add_argument('--id', default='aime26/2')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--canvases', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', action='store_true', help='extra un-timed rep: per-step CUDA kernel launch inventory')
    args = parser.parse_args()
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _rows
    from experiments.value_direction_hopper.query_adaptive import observe
    from scripts.replay_harness import StepSnapshot, execution_context, output_digest
    row = next({k: v for k, v in r.items() if k not in GOLD_FIELDS} for r in _rows(args.manifest) if r['id'] == args.id)
    arms = json.loads(args.arms)
    ns = SimpleNamespace(phase='v14profile', ids=[args.id], manifest=args.manifest, policy=args.policy, model=args.model,
                         revision=args.revision, seeds=[args.seed])
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda', precision='bfloat16',
                             revision=args.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    report = dict(schema='v14_complete_forward_profile_v1', id=args.id, seed=args.seed, reps=args.reps, states={},
                  timing_note=('INVALID for every arm after the first: the --inventory torch.profiler pass leaves CUPTI tracing '
                               'overhead in the process (+~30 ms/step observed); use launch counts only') if args.inventory else None)
    for canvas in args.canvases:
        seq = capture_sequence(adapter, row, canvas, args.steps, args.seed)
        state_report = dict(canvas=canvas, steps=len(seq), absolute=[s['absolute'] for s in seq],
                            cur_steps=[s['cur_step'] for s in seq], arms={})
        for arm in arms:
            with torch.inference_mode(), arm_runtime(adapter, arm, ns) as runtime:
                state = runtime['state'] if runtime else None
                router = runtime['router'] if runtime else None
                ctx = observe(model, state) if state is not None else None
                if ctx:
                    ctx.__enter__()
                try:
                    step_fn = model._denoising_step
                    per_rep, phases, digests = [], [], []
                    import importlib
                    registry = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_')).ALL_ATTENTION_FUNCTIONS
                    call_events = []
                    for rep in range(args.reps + 2):                 # rep 0 warmup; last rep = in-situ call timing only
                        times, rep_phases, rep_digests = [], [], []
                        timing_calls = rep == args.reps + 1
                        # a real canvas starts after an encoder commit: invalidate leases/anchors per repetition
                        if router is not None and hasattr(router, 'invalidate'):
                            router.invalidate()
                        sketch = getattr(getattr(router, 'fresh', router), 'cache', None) if router is not None else None
                        if sketch is not None and hasattr(sketch, 'invalidate'):
                            sketch.invalidate()
                        entry = registry['sdpa']
                        if timing_calls:
                            def timed_entry(module, *a, _entry=entry, **k):
                                if type(module).__name__ != 'DiffusionGemmaDecoderTextAttention' or module.is_sliding:
                                    return _entry(module, *a, **k)
                                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                e0.record()
                                r = _entry(module, *a, **k)
                                e1.record()
                                call_events.append((current['i'], int(module.layer_idx), e0, e1))
                                return r
                            registry['sdpa'] = timed_entry
                        current = {'i': -1}
                        for i, s in enumerate(seq):
                            current['i'] = i
                            kw = s['snapshot'].prepare(controller=state)
                            before = dict(router.counts) if router is not None and hasattr(router, 'counts') else None
                            torch.cuda.synchronize()
                            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            a.record()
                            result = step_fn(**kw)
                            b.record()
                            torch.cuda.synchronize()
                            times.append(a.elapsed_time(b))
                            if before is not None:
                                after = router.counts
                                rep_phases.append({k: after[k] - before[k] for k in ('anchors', 'ordinary', 'fallback_native')})
                            rep_digests.append(output_digest(result))
                        registry['sdpa'] = entry
                        if timing_calls:
                            torch.cuda.synchronize()
                            continue
                        if rep:
                            per_rep.append(times)
                            phases.append(rep_phases)
                            digests.append(rep_digests)
                    inventory = None
                    if args.inventory:                               # un-timed: which kernels each real step launches
                        from torch.profiler import ProfilerActivity, profile
                        if router is not None and hasattr(router, 'invalidate'):
                            router.invalidate()
                        inventory = []
                        for i, s in enumerate(seq):
                            kw = s['snapshot'].prepare(controller=state)
                            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                                step_fn(**kw)
                                torch.cuda.synchronize()
                            counts = {}
                            for ev in prof.events():
                                if ev.device_type.name == 'CUDA':
                                    counts[ev.name] = counts.get(ev.name, 0) + 1
                            pick = lambda *keys: sum(c for n, c in counts.items() if any(k in n.lower() for k in keys))
                            inventory.append(dict(total_kernels=sum(counts.values()),
                                                  value_direction_v4_or_v5=pick('value_direction'),
                                                  support_consumer=pick('support'), planner=pick('_plan'),
                                                  native_sdpa_like=pick('flash', 'efficient_attention', 'fmha_cutlass', 'attention_kernel'),
                                                  top=sorted(counts.items(), key=lambda x: -x[1])[:12]))
                    step_medians = [statistics.median(r[i] for r in per_rep) for i in range(len(seq))]
                    per_step_calls = {}
                    for i, layer, e0, e1 in call_events:
                        per_step_calls.setdefault(i, {})[layer] = e0.elapsed_time(e1)
                    state_report['arms'][arm['name']] = dict(
                        global_call_ms_by_step={str(i): v for i, v in sorted(per_step_calls.items())},
                        step_median_ms=step_medians, sequence_total_median_ms=statistics.median(sum(r) for r in per_rep),
                        sequence_total_min_ms=min(sum(r) for r in per_rep), phases=phases[0] if phases else None,
                        outputs_identical_across_reps=all(d == digests[0] for d in digests), launch_inventory=inventory,
                        counters=router.counters() if router is not None and hasattr(router, 'counters') else None,
                        context=execution_context(model))
                finally:
                    if ctx:
                        ctx.__exit__(None, None, None)
            print(canvas, arm['name'], round(state_report['arms'][arm['name']]['sequence_total_median_ms'], 2),
                  [round(x, 1) for x in state_report['arms'][arm['name']]['step_median_ms']], flush=True)
        report['states'][str(canvas)] = state_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')


if __name__ == '__main__':
    main()
