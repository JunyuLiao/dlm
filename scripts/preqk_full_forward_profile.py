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
    """Grabs one real _denoising_step invocation, then stops the generation."""

    class Stop(Exception):
        pass

    def __init__(self, after_steps):
        self.after_steps = after_steps
        self.kwargs = None
        self.seen = 0

    def install(self, model):
        from types import MethodType
        original = model._denoising_step
        self.original = original
        capture = self

        def step(this, **kwargs):
            capture.seen += 1
            if capture.seen > capture.after_steps and capture.kwargs is None:
                capture.kwargs = kwargs
                raise StepCapture.Stop()
            return original(**kwargs)

        model._denoising_step = MethodType(step, model)
        return original


def clone_kwargs(kwargs):
    """Deep-copy only the tensors a step may mutate; share the rest."""
    out = {}
    for name, value in kwargs.items():
        out[name] = value.clone() if torch.is_tensor(value) else value
    return out


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
            if layer in args.layers and layer not in layer_inputs and layer in sources:
                layer_inputs[layer] = dict(module=module, q=q.detach().clone(), k=k.detach().clone(),
                                           v=v.detach().clone(), mask=mask, scaling=scaling,
                                           is_causal=is_causal, sliding_window=sliding_window,
                                           cache=sources[layer][4],
                                           prefix=int(sources[layer][3]),
                                           absolute=int(sources[layer][2]))
            return true_dense(q, k, v, mask, scaling=scaling, is_causal=is_causal)

        def close(self):
            for handle in self.handles:
                handle.remove()

    binding = _install_dense(adapter)
    recorder = Recorder()
    binding.runtime.attention_override = recorder
    capture = StepCapture(args.after_steps)
    original_step = capture.install(adapter.model)
    try:
        request = GenerationRequest(prompt=row['prompt'], max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, seed=42, extra={'thinking': True})
        try:
            adapter.generate(request)
        except StepCapture.Stop:
            pass
    finally:
        adapter.model._denoising_step = original_step
        recorder.close()
        binding.close()
    if capture.kwargs is None or not layer_inputs:
        raise RuntimeError('capture failed: no denoising step or attention input recorded')

    report: dict[str, Any] = dict(
        schema='preqk_full_forward_profile_v1', id=args.id,
        support=args.support, score_refresh_period=args.score_refresh_period,
        warmup=args.warmup, reps=args.reps,
        captured_layers={str(layer): dict(q=list(value['q'].shape), k=list(value['k'].shape),
                                          prefix=value['prefix'],
                                          kind='local' if value['sliding_window'] else 'global')
                         for layer, value in layer_inputs.items()},
        attention_call={}, denoising_step={})

    # ---------------- 1. complete Attention.__call__, per phase ----------------
    for arm in args.arms:
        report['attention_call'][arm] = {}
        for layer, captured in sorted(layer_inputs.items()):
            module = captured['module']
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
            try:
                phases = {}
                for phase, step in (('anchor', 0), ('ordinary', 3), ('held', 3)):
                    if phase == 'held' and arm != 'M3_held':
                        continue
                    if phase == 'ordinary' and arm == 'M3_held':
                        continue

                    def reset(step=step):
                        router.cache.clear()
                        router.valid_keys.clear()
                        router.sketches.entries.clear()
                        router.canvas, router.step, router.epoch = 0, 0, 0
                        router.identify(module, (), {'past_key_values': captured['cache']})
                        router.sketches.identify(module, (), {'past_key_values': captured['cache']})
                        if step:
                            router(*call_args, **call_kwargs)       # establish the anchor
                            router.step = step

                    def run():
                        router(*call_args, **call_kwargs)

                    phases[phase] = timed(run, warmup=args.warmup, reps=args.reps, before=reset)
                report['attention_call'][arm][str(layer)] = phases
            finally:
                router.close()

    # ------------- 2. complete denoising step (all layers + sampler) -------------
    base_kwargs = capture.kwargs
    for arm in args.arms:
        if arm == 'fresh_junyu_T':
            continue
        binding = _install_dense(adapter)
        router = None
        try:
            if arm == 'native_dense':
                binding.runtime.attention_override = None
            else:
                router = build_router(adapter, arm, thresholds, config)
                binding.runtime.attention_override = router

            def reset(step=0):
                if router is not None:
                    router.cache.clear()
                    router.valid_keys.clear()
                    router.sketches.entries.clear()
                    router.canvas, router.step, router.epoch = 0, 0, 0

            def run_anchor():
                adapter.model._denoising_step(**clone_kwargs(base_kwargs))

            report['denoising_step'].setdefault(arm, {})
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

                report['denoising_step'][arm]['ordinary'] = timed(
                    run_anchor, warmup=args.warmup, reps=args.reps, before=reset_ordinary)
                report['denoising_step'][arm]['counters'] = router.counters()
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
