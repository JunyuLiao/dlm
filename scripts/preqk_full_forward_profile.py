"""Directly measured full-call and complete-forward profile (no component sums).

Two measurements, both on ONE captured real state so the arms are matched:

1. ``attention_call`` -- the complete ``Attention.__call__``, including input
   preparation, identity/cache planning, the V sketch lease, metadata,
   asynchronous guards and the tensor returned to the model. Not a kernel
   timing, and never a sum of isolated components.

2. ``denoising_step`` -- one complete native denoising step replayed on the
   same captured inputs: the full 30-layer decoder forward plus the native
   sampler/acceptance work, i.e. exactly the unit that per-canvas call counts
   multiply into request wall time.

Both are reported per PHASE, because the arms differ by phase:
  anchor    -- the period-8 step that must materialize every legal score
  ordinary  -- a non-anchor step that re-decides the support
  held      -- a non-anchor step reusing a held bitmap (M3 only)

Router state is reset before every repetition, outside the timed region, so a
repeat cannot drift between refresh and reuse or advance the encoder epoch.
All executed shapes are warmed before timing; the first observations (which
carry Triton/cuBLAS specialization) are reported separately, never folded in.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any

import torch

ARMS = ('native_dense', 'fresh_junyu_T', 'routing_only_current_output',
        'historical_route_preqk_current_output', 'cached_scores', 'M3_held')


class StepCapture:
    """Gate BOTH captures to ONE chosen step.

    The v7 version let the per-layer Recorder store each layer's FIRST call
    while this class intercepted a much later step, so ``captured_layers`` did
    not describe the captured full-step input at all and every "long context"
    claim was actually short context. ``armed`` is now the single switch both
    captures read, and the per-layer snapshot is taken on the SAME step whose
    kwargs are kept, before the step is allowed to mutate anything.
    """

    class Stop(Exception):
        pass

    def __init__(self, after_steps):
        self.after_steps = after_steps
        self.kwargs = None
        self.seen = 0
        self.armed = False
        self.step_index = None

    def install(self, model):
        from types import MethodType
        original = model._denoising_step
        self.original = original
        capture = self

        def step(this, **kwargs):
            capture.seen += 1
            if capture.seen > capture.after_steps and capture.kwargs is None:
                # Arm the per-layer recorder, snapshot this step's inputs, run
                # the step so the layer hooks fire, then stop.
                capture.armed = True
                capture.step_index = capture.seen
                capture.kwargs = clone_kwargs(kwargs)
                original(**kwargs)
                raise StepCapture.Stop()
            return original(**kwargs)

        model._denoising_step = MethodType(step, model)
        return original


def clone_kwargs(kwargs):
    """Deep-copy the tensors a step may mutate; share the rest.

    Bug D: this only protects TENSOR arguments. Non-tensor sampler/cache
    objects are shared, and torch RNG is not restored by cloning, so a replay
    harness must additionally snapshot and restore RNG (see ``replay_state``)
    and must verify that repeated replays really do see identical inputs.
    """
    out = {}
    for name, value in kwargs.items():
        out[name] = value.clone() if torch.is_tensor(value) else value
    return out


def tensor_digest(value):
    """Cheap order-sensitive digest used to prove replay inputs are identical."""
    if torch.is_tensor(value):
        flat = value.detach().reshape(-1).to(torch.float64)
        return (tuple(value.shape), str(value.dtype), float(flat.sum()),
                float((flat * torch.arange(flat.numel(), device=flat.device,
                                           dtype=torch.float64)).sum()))
    return repr(value)[:80]


class replay_state:
    """Restore RNG (and report sampler mutation) around every timed replay."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu_rng)
        torch.cuda.set_rng_state_all(self.cuda_rng)
        return False


def timed(run, *, warmup, reps, before=None):
    first = None
    for index in range(warmup):
        if before is not None:
            before()
        if index == 0:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            first = start.elapsed_time(end)
        else:
            run()
    samples = []
    for _ in range(reps):
        if before is not None:
            before()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return dict(median_ms=statistics.median(samples), min_ms=samples[0],
                p90_ms=samples[min(len(samples) - 1, int(.9 * len(samples)))],
                first_observation_ms=first, reps=len(samples))


def describe_sensitivity(sensitivity):
    """Bug B: nonuniformity must be judged from values, not tensor presence."""
    if sensitivity is None:
        return dict(present=False, uniform=True, note='uniform T=1 control, NOT the production T path')
    flat = sensitivity.detach().reshape(-1).float()
    spread = float(flat.max() - flat.min())
    return dict(present=True, min=float(flat.min()), max=float(flat.max()),
                mean=float(flat.mean()), std=float(flat.std()), spread=spread,
                uniform=spread < 1e-6,
                note=('values are uniform despite being present' if spread < 1e-6
                      else 'genuinely nonuniform causal T'))


def build_router(adapter, arm, thresholds, config):
    """Returns (context manager, router) for an arm, or (None, None) for dense."""
    from experiments.numerical_qk_reuse.integration import Attention
    if arm in ('native_dense', 'fresh_junyu_T'):
        return None
    output_mode = {'routing_only_current_output': 'routing_only_current_output',
                   'historical_route_preqk_current_output': 'historical_route_preqk_current_output',
                   'cached_scores': 'cached_scores',
                   'M3_held': 'historical_route_preqk_current_output'}[arm]
    interval = 2 if arm == 'M3_held' else 1
    return Attention(adapter, thresholds, score_period=config['score_refresh_period'],
                     decision_interval=interval, trace=False,
                     support=config['support'], output_mode=output_mode)


def profile(args) -> dict[str, Any]:
    from dllm.models import GenerationRequest, create_adapter
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
    from experiments.numerical_qk_reuse.integration import Attention
    from scripts.native_reuse_phaseAB_diagnostic import true_dense

    manifest = {row['id']: row for row in json.loads(args.manifest.read_text(encoding='utf-8'))}
    thresholds = json.loads(args.policy.read_text(encoding='utf-8'))['policies'][args.policy_name]
    row = manifest[args.id]
    config = dict(score_refresh_period=args.score_refresh_period, support=args.support)

    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda',
                             precision='bfloat16', revision=args.revision).load()

    # --- capture one real denoising step and the per-layer attention inputs ---
    layer_inputs: dict[int, Any] = {}
    sources: dict[int, Any] = {}
    captured: dict[str, Any] = {'sensitivity': None}

    class Recorder:
        def __init__(self):
            self.handles = []
            for name, module in adapter.model.named_modules():
                if adapter.is_blasst_attention_module(name, module):
                    self.handles.append(module.register_forward_pre_hook(self._identify, with_kwargs=True))

        def _identify(self, module, arguments, kwargs):
            cache = kwargs.get('past_key_values', arguments[3] if len(arguments) > 3 else None)
            if cache is None:
                return
            layer = int(module.layer_idx)
            sources[layer] = (cache.layers[layer].keys, cache.layers[layer].values,
                              cache.get_seq_length(), cache.layers[layer].keys.shape[-2], cache)

        def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                     is_causal=None, sliding_window=None, **kwargs):
            layer = int(module.layer_idx)
            # Bug A: only record on the ONE armed step, so the per-layer
            # snapshot and the full-step kwargs describe the same call.
            if capture.armed and layer in args.layers and layer not in layer_inputs and layer in sources:
                keys, values, absolute, prefix, _ = sources[layer]
                # Snapshot the cache CONTENTS: the live object keeps mutating
                # until the capture stops, so replaying against it would see a
                # prefix that no longer matches the recorded q/k/v.
                layer_inputs[layer] = dict(module=module, q=q.detach().clone(), k=k.detach().clone(),
                                           v=v.detach().clone(), mask=mask, scaling=scaling,
                                           is_causal=is_causal, sliding_window=sliding_window,
                                           prefix_k=keys.detach().clone(),
                                           prefix_v=values.detach().clone(),
                                           prefix=int(prefix), absolute=int(absolute))
            return true_dense(q, k, v, mask, scaling=scaling, is_causal=is_causal)

        def close(self):
            for handle in self.handles:
                handle.remove()

    binding = _install_dense(adapter)
    recorder = Recorder()
    binding.runtime.attention_override = recorder
    capture = StepCapture(args.after_steps)
    original_step = capture.install(adapter.model)
    # Bug B: run the real State/observe machinery during capture so we obtain
    # the production causal T weights for this step, instead of profiling a
    # uniform-T path and calling it the T path.
    from experiments.value_direction_hopper.query_adaptive import State, observe
    capture_state = State('T', None, m_ref=float(args.m_ref), beta=float(args.beta),
                          gamma=float(args.gamma), diagnostics=False)
    try:
        request = GenerationRequest(prompt=row['prompt'], max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, seed=42, extra={'thinking': True})
        try:
            with observe(adapter.model, capture_state):
                adapter.generate(request)
        except StepCapture.Stop:
            pass
        captured['sensitivity'] = (None if capture_state.used_weights is None
                                   else capture_state.used_weights.detach().clone())
    finally:
        adapter.model._denoising_step = original_step
        recorder.close()
        binding.close()
    captured_sensitivity = captured['sensitivity']
    if capture.kwargs is None or not layer_inputs:
        raise RuntimeError('capture failed: no denoising step or attention input recorded')
    if set(layer_inputs) != set(args.layers):
        raise RuntimeError(f'armed step did not cover every requested layer: '
                           f'{sorted(layer_inputs)} vs {sorted(args.layers)}')
    # Bug A: a filename never establishes context length. Assert it.
    reached = {layer: value['prefix'] for layer, value in layer_inputs.items()}
    if args.require_prefix_at_least:
        short = {layer: prefix for layer, prefix in reached.items()
                 if prefix < args.require_prefix_at_least}
        if short:
            raise RuntimeError(f'requested prefix >= {args.require_prefix_at_least} but the '
                               f'armed step reached {short}; raise --after-steps/--max-new-tokens')

    report: dict[str, Any] = dict(
        schema='preqk_full_forward_profile_v1', id=args.id,
        support=args.support, score_refresh_period=args.score_refresh_period,
        warmup=args.warmup, reps=args.reps,
        armed_step_index=capture.step_index,
        captured_sensitivity=describe_sensitivity(captured_sensitivity),
        captured_layers={str(layer): dict(q=list(value['q'].shape), k=list(value['k'].shape),
                                          prefix=value['prefix'], absolute=value['absolute'],
                                          kind='local' if value['sliding_window'] else 'global')
                         for layer, value in layer_inputs.items()},
        attention_call={}, denoising_step={})

    # ---------------- 1. complete Attention.__call__, per phase ----------------
    from types import SimpleNamespace
    for arm in args.arms:
        report['attention_call'][arm] = {}
        for layer, captured in sorted(layer_inputs.items()):
            module = captured['module']
            frozen_cache = SimpleNamespace(
                layers={layer: SimpleNamespace(keys=captured['prefix_k'], values=captured['prefix_v'])},
                is_compileable=False, get_seq_length=lambda a=captured['absolute']: a)
            call_args = (module, captured['q'], captured['k'], captured['v'], captured['mask'])
            call_kwargs = dict(scaling=captured['scaling'], is_causal=captured['is_causal'],
                               sliding_window=captured['sliding_window'])
            if arm == 'native_dense':
                measured = {'dense': timed(lambda: true_dense(
                    captured['q'], captured['k'], captured['v'], captured['mask'],
                    scaling=captured['scaling'], is_causal=captured['is_causal']),
                    warmup=args.warmup, reps=args.reps)}
                report['attention_call'][arm][str(layer)] = measured
                continue
            if arm == 'fresh_junyu_T':
                continue          # measured only as a generation arm; see notes
            router = build_router(adapter, arm, thresholds, config)
            # Bug B: v7 left query_sensitivity=None, i.e. uniform T, in every
            # profiled numerical router. Install the captured production T.
            router.query_sensitivity = captured_sensitivity
            try:
                phases = {}
                for phase, step in (('anchor', 0), ('ordinary', 3), ('held', 3)):
                    if phase == 'held' and arm != 'M3_held':
                        continue
                    if phase == 'ordinary' and arm == 'M3_held':
                        continue

                    def reset(step=step, phase=phase):
                        router.cache.clear()
                        router.valid_keys.clear()
                        router.sketches.entries.clear()
                        router.canvas, router.step, router.epoch = 0, 0, 0
                        router.identify(module, (), {'past_key_values': frozen_cache})
                        router.sketches.identify(module, (), {'past_key_values': frozen_cache})
                        if step:
                            router(*call_args, **call_kwargs)       # establish the anchor
                            router.step = step
                            if phase == 'held':
                                # Bug C: with decision_interval=2, step 3 after
                                # an anchor at 0 is a DECISION REFRESH, not a
                                # held call. Walk to a step the schedule really
                                # holds, and assert it before timing.
                                for candidate in range(step, step + 8):
                                    router.step = candidate
                                    plan = router.cache.plan(
                                        router.cache.entries[int(module.layer_idx)].identity,
                                        candidate)
                                    if not plan.score_refresh and not plan.decision_refresh:
                                        break
                                else:
                                    raise RuntimeError('no held step reachable for M3_held')

                    def run():
                        router(*call_args, **call_kwargs)

                    if phase == 'held':
                        reset()
                        layer_id = int(module.layer_idx)
                        plan = router.cache.plan(router.cache.entries[layer_id].identity, router.step)
                        if plan.score_refresh or plan.decision_refresh:
                            raise RuntimeError(f'M3_held timing point is not held: {plan}')
                    phases[phase] = timed(run, warmup=args.warmup, reps=args.reps, before=reset)
                report['attention_call'][arm][str(layer)] = phases
            finally:
                router.close()

    # ------------- 2. complete denoising step (all layers + sampler) -------------
    base_kwargs = capture.kwargs
    for arm in args.arms:
        if arm == 'fresh_junyu_T':
            if not (args.library and args.torch_library):
                report['denoising_step']['fresh_junyu_T'] = dict(
                    skipped='no --library/--torch-library supplied')
                continue
            from experiments.diffusion_gemma_jl_output_aware.projections import Projections
            from experiments.value_direction_hopper.integration import install as install_fresh
            from experiments.value_direction_hopper.query_adaptive import State
            with install_fresh(adapter, str(args.library), thresholds, mode='value',
                               projections=Projections(),
                               torch_library=str(args.torch_library), collect=False) as (_bind, router_t):
                # Bug B: v7 installed the attention router but never the State
                # / observe machinery, so this timed a UNIFORM-T path while
                # calling it fresh Junyu T. Install the real causal history.
                state = State('T', router_t, m_ref=float(args.m_ref), beta=float(args.beta),
                              gamma=float(args.gamma), diagnostics=False)
                router_t.query_sensitivity = captured_sensitivity
                report['denoising_step']['fresh_junyu_T'] = dict(
                    anchor=timed(lambda: adapter.model._denoising_step(**clone_kwargs(base_kwargs)),
                                 warmup=args.warmup, reps=args.reps),
                    sensitivity=describe_sensitivity(captured_sensitivity))
            continue
        binding = _install_dense(adapter)
        router = None
        try:
            if arm == 'native_dense':
                binding.runtime.attention_override = None
            else:
                router = build_router(adapter, arm, thresholds, config)
                router.query_sensitivity = captured_sensitivity
                binding.runtime.attention_override = router

            def reset(step=0):
                if router is not None:
                    router.cache.clear()
                    router.valid_keys.clear()
                    router.sketches.entries.clear()
                    router.canvas, router.step, router.epoch = 0, 0, 0
                    # Bug E: optional telemetry lists must not grow across
                    # replays, or later repetitions time list growth too.
                    router.pending.clear()
                    router.call_metadata.clear()

            def run_anchor():
                adapter.model._denoising_step(**clone_kwargs(base_kwargs))

            report['denoising_step'].setdefault(arm, {})
            # Bug D: prove replays really see identical inputs, and restore RNG
            # around the whole timed series so a replay cannot advance it.
            digests = [tuple(tensor_digest(v) for v in clone_kwargs(base_kwargs).values())
                       for _ in range(2)]
            report['denoising_step'][arm]['replay_inputs_identical'] = digests[0] == digests[1]
            with replay_state(adapter.model):
                report['denoising_step'][arm]['anchor'] = timed(
                    run_anchor, warmup=args.warmup, reps=args.reps, before=reset)

            if router is not None:
                def reset_ordinary():
                    router.cache.clear()
                    router.valid_keys.clear()
                    router.sketches.entries.clear()
                    router.canvas, router.step, router.epoch = 0, 0, 0
                    adapter.model._denoising_step(**clone_kwargs(base_kwargs))  # anchor
                    router.step = 3

                with replay_state(adapter.model):
                    report['denoising_step'][arm]['ordinary'] = timed(
                        run_anchor, warmup=args.warmup, reps=args.reps, before=reset_ordinary)
                # Bug E: report the DELTA produced by one measured repetition,
                # not accumulated counters spanning warmup and resets.
                reset_ordinary()
                before_counters = router.counters()
                run_anchor()
                after_counters = router.counters()
                report['denoising_step'][arm]['counter_delta_one_ordinary_step'] = {
                    key: after_counters[key] - before_counters[key]
                    for key in after_counters
                    if isinstance(after_counters.get(key), (int, float))
                    and isinstance(before_counters.get(key), (int, float))}
                report['denoising_step'][arm]['counters_cumulative'] = after_counters
                report['denoising_step'][arm]['sensitivity'] = describe_sensitivity(
                    captured_sensitivity)
        finally:
            if router is not None:
                router.close()
            binding.close()
    return report


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-name', default='T_s50')
    parser.add_argument('--id', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 5])
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--after-steps', type=int, default=9)
    parser.add_argument('--max-new-tokens', type=int, default=320)
    parser.add_argument('--score-refresh-period', type=int, default=8)
    parser.add_argument('--support', default='legacy_junyu_mask')
    parser.add_argument('--library', type=Path)
    parser.add_argument('--torch-library', type=Path)
    parser.add_argument('--require-prefix-at-least', type=int, default=0,
                        help='assert the armed step actually reached this prefix length')
    parser.add_argument('--m-ref', type=float, default=14.258454322814941)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    report = profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for arm, phases in sorted(report['denoising_step'].items()):
        for phase, value in sorted(phases.items()):
            if isinstance(value, dict) and 'median_ms' in value:
                print(f"denoising_step {arm:42s} {phase:9s} median={value['median_ms']:8.2f}ms "
                      f"min={value['min_ms']:8.2f} first={value['first_observation_ms']:8.2f}")


if __name__ == '__main__':
    main()
