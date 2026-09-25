"""v9 corrected complete-step replay (supersedes v8's denoising_step section).

Capture: an UNBOUND native generation (registry ``sdpa`` asserted to be the
installed native function) runs to canvas ``--canvas``; the first two real
denoising steps of that canvas are snapshotted at the step boundary, OUTSIDE
the T observer, so the recorded sampler is the native one. Each snapshot holds
cloned tensors, the discovered mutable attributes of the native sampler /
stopping criterion / logits processors, CPU+CUDA RNG, and the capture-time T
controller's PRE-step state.

Rows (all under the production execution context, asserted equal to the one
read inside ``adapter.generate``):

  native_decoder_forward     decoder forward only, no binding
  native_denoising_step      decoder + native sampler/acceptance/stopping, no
                             binding, no T observer; dispatch spy must show
                             native SDPA only
  dense_eager_same_mask      ``_install_dense`` binding with no override: the
                             BLASST dispatcher's ``dense_eager_attention_forward``
                             (what v8 mislabeled native_dense); NOT native
  sparse_<selector>          the production M1 ``integration.install`` binding
                             + exactly ONE ``observe`` wrapper, T controller
                             restored to the captured pre-step state, real T
                             bookkeeping inside the timed step:
      anchor    canvas-local step 0, a genuine score anchor
      ordinary  canvas-local step 1 on age-1 history produced by actually
                replaying step 0 (untimed) from its own captured inputs

Before EVERY repetition (untimed): RNG, native objects, T controller and router
history are restored and the input+state digest recomputed; after it the
output digest and router telemetry deltas are recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from scripts.replay_harness import (StepSnapshot, assert_native_path, dispatch_spy,
                                    execution_context, output_digest, reset_router,
                                    telemetry, timed_rows)

STEP_PARAMS = ('decoder_forward', 'current_canvas', 'argmax_canvas', 'input_ids',
               'decoder_position_ids', 'self_conditioning_logits', 'mask_mapping',
               'past_key_values', 'finished_denoising', 'cur_step', 'sampler',
               'logits_processor', 'diffusion_stopping_criteria')


class Stop(Exception):
    pass


def native_registry(model):
    import importlib
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    modeling = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_'))
    return modeling.ALL_ATTENTION_FUNCTIONS, sdpa_attention_forward


def capture(adapter, row, args) -> dict[str, Any]:
    from dllm.models import GenerationRequest
    from experiments.value_direction_hopper.query_adaptive import State, observe
    model = adapter.model
    registry, native = native_registry(model)
    if registry['sdpa'] is not native or '_denoising_step' in vars(model):
        raise RuntimeError('capture must start from the untouched native model')
    controller = State('T', None, m_ref=args.m_ref, beta=args.beta, gamma=args.gamma, diagnostics=False)
    steps: list[dict[str, Any]] = []
    canvases = {'n': -1}
    with observe(model, controller):
        inner = model._denoising_step          # observe's single wrapper

        def outer(this, **kwargs):
            cur = int(kwargs['cur_step'])
            if cur == 48:
                canvases['n'] += 1
            local = 48 - cur
            if canvases['n'] == args.canvas and local < 2:
                cache = kwargs['past_key_values']
                steps.append(dict(snapshot=StepSnapshot(kwargs, controller=controller),
                                  context=execution_context(model), cur_step=cur,
                                  canvas=canvases['n'], local_index=local,
                                  absolute=int(cache.get_seq_length()),
                                  stored_prefix={str(i): int(layer.keys.shape[-2])
                                                 for i, layer in enumerate(cache.layers)}))
                result = inner(**kwargs)
                if len(steps) == 2:
                    raise Stop()
                return result
            return inner(**kwargs)

        model._denoising_step = MethodType(outer, model)
        try:
            adapter.generate(GenerationRequest(prompt=row['prompt'], max_new_tokens=(args.canvas + 2) * 256,
                                               temperature=0.0, seed=42, extra={'thinking': True}))
        except Stop:
            pass
        finally:
            model._denoising_step = inner
    if len(steps) != 2 or [s['local_index'] for s in steps] != [0, 1]:
        raise RuntimeError(f'did not capture canvas {args.canvas} steps 0 and 1: {len(steps)}')
    if '_denoising_step' in vars(model):
        raise RuntimeError('observe wrapper leaked past capture')
    return dict(steps=steps)


def decoder_call(model, kwargs):
    extra = {key: value for key, value in kwargs.items() if key not in STEP_PARAMS}
    return model.forward(decoder_input_ids=kwargs['current_canvas'],
                         self_conditioning_logits=kwargs['self_conditioning_logits'],
                         decoder_attention_mask=kwargs['mask_mapping'],
                         past_key_values=kwargs['past_key_values'],
                         decoder_position_ids=kwargs['decoder_position_ids'], **extra).logits


def check_rows(evidence_key='input_digest'):
    def check(fixture, result):
        return dict(output_digest=output_digest(result))
    return check


def summarize_identity(row: dict[str, Any]) -> dict[str, Any]:
    reps = row['per_repetition']
    inputs = {r.get('input_digest') for r in reps}
    outputs = {r.get('output_digest') for r in reps}
    row['inputs_identical_every_repetition'] = len(inputs) == 1
    row['outputs_identical_every_repetition'] = len(outputs) == 1
    if len(inputs) != 1 or len(outputs) != 1:
        raise AssertionError(f'replay drift: {len(inputs)} input / {len(outputs)} output digests')
    return row


def profile(args) -> dict[str, Any]:
    from dllm.models import create_adapter
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
    from experiments.numerical_qk_reuse import integration
    from experiments.value_direction_hopper.query_adaptive import observe

    manifest = {row['id']: row for row in json.loads(args.manifest.read_text(encoding='utf-8'))}
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda',
                             precision='bfloat16', revision=args.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False       # as the production runner
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    captured = capture(adapter, manifest[args.id], args)
    step0, step1 = captured['steps']
    production_context = step0['context']
    report: dict[str, Any] = dict(
        schema='v9_step_replay_profile_v1', id=args.id, canvas=args.canvas,
        warmup=args.warmup, reps=args.reps,
        captured=[{k: v for k, v in s.items() if k != 'snapshot'} for s in captured['steps']],
        snapshot_inventory=step0['snapshot'].inventory(),
        capture_trajectory='untouched native SDPA (unbound), native sampler, T observed by a '
                           'capture-time State outside which the step snapshot is taken',
        rows={}, dispatch={}, contexts={})

    def context_check(name):
        now = execution_context(model)
        mismatch = {k: (production_context[k], now[k]) for k in production_context
                    if k not in ('denoising_step_instance_override',) and production_context[k] != now[k]}
        report['contexts'][name] = dict(replay=now, mismatches_vs_production=mismatch)
        if mismatch:
            raise AssertionError(f'{name}: replay context differs from production: {mismatch}')

    def native_rows():
        registry, native = native_registry(model)
        assert registry['sdpa'] is native and '_denoising_step' not in vars(model)
        for name, snap, run in (
                ('native_decoder_forward', step0, lambda kw: decoder_call(model, kw)),
                ('native_denoising_step.anchor_inputs', step0, lambda kw: model._denoising_step(**kw)),
                ('native_denoising_step.step1_inputs', step1, lambda kw: model._denoising_step(**kw))):
            snapshot = snap['snapshot']

            def prepare(snapshot=snapshot):
                kw = snapshot.prepare()
                digests['input'] = StepSnapshot.digest(kw)
                return kw

            digests: dict[str, str] = {}
            with dispatch_spy(model) as counts:            # one bounded, untimed probe
                run(snapshot.prepare())
                torch.cuda.synchronize()
            assert_native_path(counts)
            report['dispatch'][name] = counts
            context_check(name)
            row = timed_rows(run, prepare, warmup=args.warmup, reps=args.reps,
                             check=lambda fx, res: dict(input_digest=digests['input'],
                                                        output_digest=output_digest(res)))
            report['rows'][name] = summarize_identity(row)

    def eager_rows():
        binding = _install_dense(adapter)
        try:
            snapshot = step0['snapshot']
            with dispatch_spy(model) as counts:
                model._denoising_step(**snapshot.prepare())
                torch.cuda.synchronize()
            if counts['dense_eager'] == 0 or counts['numerical_selector']:
                raise AssertionError(f'eager row did not enter dense_eager only: {counts}')
            report['dispatch']['dense_eager_same_mask'] = counts
            digests: dict[str, str] = {}

            def prepare():
                kw = snapshot.prepare()
                digests['input'] = StepSnapshot.digest(kw)
                return kw
            row = timed_rows(lambda kw: model._denoising_step(**kw), prepare, warmup=args.warmup,
                             reps=args.reps, check=lambda fx, res: dict(
                                 input_digest=digests['input'], output_digest=output_digest(res)))
            row['label'] = ('_install_dense + attention_override=None -> dense_eager_attention_forward; '
                            'NOT the untouched native SDPA path')
            report['rows']['dense_eager_same_mask.anchor_inputs'] = summarize_identity(row)
        finally:
            binding.close()

    def sparse_rows(selector, arm=None):
        arm = arm or {}
        config = dict(policy=policy['policies'][args.policy_name], score_refresh_period=8,
                      decision_interval=1, support='legacy_junyu_mask',
                      output_mode='historical_route_preqk_current_output', selector=selector,
                      selector_layers='local', m_ref=args.m_ref, beta=args.beta, gamma=args.gamma,
                      diagnostic=False, kernel_variant=arm.get('kernel_variant', 'static'),
                      telemetry=arm.get('telemetry', 'full'), guard_mode=arm.get('guard_mode', 'separate'),
                      consumer=arm.get('consumer', 'triton'), support_build=arm.get('support_build'))
        name = f"sparse_{arm['name']}" if 'name' in arm else f'sparse_{selector}'
        with integration.install(adapter, config, 'M1') as runtime:
            router, state = runtime['router'], runtime['state']
            with observe(model, state):                      # exactly ONE observer
                step = model._denoising_step
                with dispatch_spy(model) as counts:
                    reset_router(router)
                    step(**step0['snapshot'].prepare(controller=state))
                    torch.cuda.synchronize()
                if counts['numerical_selector'] == 0 or counts['dense_eager']:
                    raise AssertionError(f'{name} did not route through the selector: {counts}')
                report['dispatch'][name] = counts
                context_check(name)
                digests: dict[str, Any] = {}

                def prepare_anchor():
                    reset_router(router)
                    kw = step0['snapshot'].prepare(controller=state)
                    digests['input'] = StepSnapshot.digest(kw, controller=state)
                    digests['before'] = telemetry(router)
                    return kw

                def prepare_ordinary():
                    reset_router(router)
                    step(**step0['snapshot'].prepare(controller=state))   # real age-1 anchor
                    torch.cuda.synchronize()
                    kw = step1['snapshot'].prepare(controller=state)
                    digests['input'] = StepSnapshot.digest(kw, controller=state)
                    anchors = {entry.score_step for entry in router.cache.entries.values()}
                    if anchors != {0} or router.step != 0 or state.iteration != 1:
                        raise AssertionError(f'ordinary setup is not age-1 history: {anchors} '
                                             f'{router.step} {state.iteration}')
                    digests['before'] = telemetry(router)
                    return kw

                def check(expect_scores):
                    def inner(fixture, result):
                        after = telemetry(router)
                        delta = {k: after[k] - digests['before'][k] for k in after}
                        if (delta['score_calls'] > 0) != expect_scores or delta['calls'] != 30:
                            raise AssertionError(f'{name}: phase mismatch {delta}')
                        return dict(input_digest=digests['input'], output_digest=output_digest(result),
                                    telemetry_delta=delta, telemetry_after=after,
                                    router_step=router.step)
                    return inner

                def prepare_held_same_support():
                    # v10 consumer-only counterfactual: step 1's OWN decision is
                    # computed untimed, then step 1 is replayed with that exact
                    # bitmap held (no route, no sketch, no route guard); T
                    # bookkeeping and the current-output consumer remain.
                    router.cache.decision_interval = 1
                    reset_router(router)
                    step(**step0['snapshot'].prepare(controller=state))
                    step(**step1['snapshot'].prepare(controller=state))
                    torch.cuda.synchronize()
                    router.cache.decision_interval = 10 ** 6
                    kw = step1['snapshot'].prepare(controller=state)
                    digests['input'] = StepSnapshot.digest(kw, controller=state)
                    digests['before'] = telemetry(router)
                    return kw

                def check_held(fixture, result):
                    after = telemetry(router)
                    delta = {k: after[k] - digests['before'][k] for k in after}
                    if delta['decision_calls'] or delta['score_calls'] or delta['held_calls'] != 30:
                        raise AssertionError(f'{name}: held phase mismatch {delta}')
                    return dict(input_digest=digests['input'], output_digest=output_digest(result),
                                telemetry_delta=delta)

                phases = [('anchor', prepare_anchor, check(True)), ('ordinary', prepare_ordinary, check(False))]
                if args.held_counterfactual:
                    phases.append(('held_same_support', prepare_held_same_support, check_held))
                for phase, prepare, checker in phases:
                    row = timed_rows(lambda kw: step(**kw), prepare, warmup=args.warmup,
                                     reps=args.reps, check=checker)
                    row['label'] = ('production M1 binding + one observe wrapper; T bookkeeping '
                                    'inside the timed step' + ('' if phase == 'anchor' else
                                    '; age-1 history from an untimed real replay of step 0'))
                    report['rows'][f'{name}.{phase}'] = summarize_identity(row)
                router.cache.decision_interval = 1
        if '_denoising_step' in vars(model):
            raise RuntimeError('observer leaked')

    def fresh_t_rows(arm):
        """v11: genuine fresh Junyu T complete step: its own production routing,
        exactly ONE observe wrapper, T restored per repetition, collect off."""
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        from experiments.value_direction_hopper.integration import install as install_t
        from experiments.value_direction_hopper.query_adaptive import State
        name = f"fresh_{arm['name']}"
        with install_t(adapter, arm['library'], policy['policies'][args.policy_name], mode='value',
                       projections=Projections(), torch_library=arm['torch_library'], collect=False) as (_b, router_t):
            state = State('T', router_t, m_ref=args.m_ref, beta=args.beta, gamma=args.gamma, diagnostics=False)
            with observe(model, state):
                step = model._denoising_step
                with dispatch_spy(model) as counts:
                    step(**step1['snapshot'].prepare(controller=state))
                    torch.cuda.synchronize()
                if counts['dense_eager'] or counts['numerical_selector'] or counts['registry_sdpa'] != 30:
                    raise AssertionError(f'{name} did not route through fresh T: {counts}')
                report['dispatch'][name] = counts
                context_check(name)
                for label, snap in (('step0_inputs', step0), ('step1_inputs', step1)):
                    digests = {}

                    def prepare(snap=snap):
                        kw = snap['snapshot'].prepare(controller=state)
                        digests['input'] = StepSnapshot.digest(kw, controller=state)
                        return kw
                    row = timed_rows(lambda kw: step(**kw), prepare, warmup=args.warmup, reps=args.reps,
                                     check=lambda fx, res: dict(input_digest=digests['input'],
                                                                output_digest=output_digest(res)))
                    row['label'] = 'fresh Junyu T (TMA, fused projection, collect=False) + one observe wrapper'
                    report['rows'][f'{name}.{label}'] = summarize_identity(row)
        if '_denoising_step' in vars(model):
            raise RuntimeError('observer leaked')

    with torch.inference_mode():            # adapter.generate's own decorator
        order = args.order
        extra = {arm['name']: arm for arm in json.loads(args.sparse_arms)}
        for arm in order:
            if arm in extra and extra[arm].get('kind') == 'fresh_t':
                fresh_t_rows(extra[arm])
                continue
            if arm in extra:
                sparse_rows(extra[arm].get('selector', 'prefix_block_summary'), extra[arm])
                continue
            {'native': native_rows, 'eager': eager_rows,
             'legacy': lambda: sparse_rows('legacy_recompute'),
             'summary': lambda: sparse_rows('prefix_block_summary')}[arm]()
    report['order'] = order
    return report


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-name', default='T_s50')
    parser.add_argument('--id', required=True)
    parser.add_argument('--canvas', type=int, default=1)
    parser.add_argument('--order', nargs='+', default=['native', 'eager', 'legacy', 'summary'])
    parser.add_argument('--held-counterfactual', action='store_true',
                        help='v10: add the same-support consumer-only (held decision) phase')
    parser.add_argument('--sparse-arms', default='[]',
                        help='v10: JSON list of {name, selector, kernel_variant, telemetry, guard_mode}; '
                             'names usable in --order')
    parser.add_argument('--m-ref', type=float, default=14.258454322814941)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    report = profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    temp.replace(args.output)
    for name, row in report['rows'].items():
        print(f"{name:45s} median={row['median_ms']:8.2f} ms  min={row['min_ms']:8.2f}  "
              f"wall={row['wall_median_ms']:8.2f}  first={row['first_observation_ms']:8.2f}", flush=True)


if __name__ == '__main__':
    main()
