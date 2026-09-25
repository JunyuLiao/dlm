"""v11 CP2: direct COMPLETE attention calls on the SAME real inputs.

Raw attention-interface arguments (module, q, k, v, mask, kwargs) of chosen
LOCAL/GLOBAL layers are captured during an untouched native replay of a real
captured step (v9 harness), plus that step's production causal T. Every arm is
then timed on exactly those tensors:

  native            installed native SDPA function
  dense_same_mask   diagnostic dense SDPA with an explicit boolean legacy
                    legal mask (same legality as T/O/H1); NOT a speed baseline
  fresh_T           Junyu's complete Attention.__call__ (own routing, TMA,
                    fused projection, mask cache, returned tensor)
  O / H1            complete M1 Attention.__call__ (selector + consumer +
                    guards), ORDINARY phase: anchor at step 0 on the same
                    inputs (phase-cost simulation, labeled), decision reset
                    before every repetition so each call is a real decision
  consumer_triton / consumer_hopper
                    consumer-only on the identical bitmap/cropped K,V the O
                    call routed (supplemental; includes their Python adapters)

After warmup, arms are timed interleaved in rotating order; each sample is a
synchronized CUDA-event span around ONE call (includes host launch time).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--support-build', type=Path, required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--torch-library', required=True)
    parser.add_argument('--id', default='aime26/2')
    parser.add_argument('--canvas', type=int, default=6)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 5])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m-ref', type=float, default=14.258454322814941)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    args = parser.parse_args()
    from dllm.models import create_adapter
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    from experiments.numerical_qk_reuse import integration
    from experiments.value_direction_hopper import integration as junyu
    from experiments.value_direction_hopper.query_adaptive import State
    from scripts.replay_harness import execution_context
    from scripts.v9_step_replay_profile import capture, native_registry
    policy = json.loads(args.policy.read_text())['policies']['T_s50']
    manifest = {r['id']: r for r in json.loads(args.manifest.read_text())}
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda', precision='bfloat16',
                             revision=args.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    cap_args = SimpleNamespace(canvas=args.canvas, m_ref=args.m_ref, beta=args.beta, gamma=args.gamma)
    captured = capture(adapter, manifest[args.id], cap_args)
    step1 = captured['steps'][1]
    snapshot = step1['snapshot']
    registry, native = native_registry(model)
    raw = {}

    def recorder(module, q, k, v, mask, **kw):
        layer = int(module.layer_idx)
        if layer in args.layers and layer not in raw:
            raw[layer] = dict(module=module, q=q.clone(), k=k.clone(), v=v.clone(), mask=mask, kw=dict(kw))
        return native(module, q, k, v, mask, **kw)

    with torch.inference_mode():
        kwargs = snapshot.prepare()
        cache = kwargs['past_key_values']
        registry['sdpa'] = recorder
        try:
            model._denoising_step(**kwargs)
        finally:
            registry['sdpa'] = native
        torch.cuda.synchronize()
        controller = snapshot.controller.restore(State('T', None, m_ref=args.m_ref, beta=args.beta,
                                                       gamma=args.gamma, diagnostics=False))
        from experiments.value_direction_hopper.query_adaptive import weight
        sensitivity = weight('T', controller.margin, controller.confidence, controller.temporal,
                             beta=args.beta, m_ref=args.m_ref)
        report = dict(schema='v11_same_support_calls_v1', id=args.id, canvas=args.canvas,
                      absolute=step1['absolute'], context=execution_context(model),
                      sensitivity=dict(present=sensitivity is not None,
                                       spread=None if sensitivity is None else float(sensitivity.max() - sensitivity.min())),
                      layers={})
        m1_cfg = dict(policy=policy, score_refresh_period=8, decision_interval=1, support='legacy_junyu_mask',
                      output_mode='historical_route_preqk_current_output', selector='prefix_block_summary',
                      selector_layers='local', kernel_variant='generic', telemetry='minimal', guard_mode='fused')

        def m1_router(consumer):
            return integration.Attention(adapter, policy, score_period=8, decision_interval=1, trace=False,
                                         support='legacy_junyu_mask', output_mode=integration.PREQK_MODE,
                                         selector='prefix_block_summary', selector_layers='local',
                                         kernel_variant='generic', telemetry='minimal', guard_mode='fused',
                                         consumer=consumer, support_build=str(args.support_build))
        routers = dict(O=m1_router('triton'), H1=m1_router('hopper'))
        fresh = junyu.Attention(adapter, args.library, policy, mode='value', projections=Projections(),
                                torch_library=args.torch_library, collect=False)
        fresh.query_sensitivity = sensitivity
        try:
            for layer, x in sorted(raw.items()):
                module, q, k, v, mask, kw = x['module'], x['q'], x['k'], x['v'], x['mask'], x['kw']
                window = kw.get('sliding_window')
                for router in routers.values():
                    router.query_sensitivity = sensitivity
                    router.begin_step(0, 0)
                    router.identify(module, (), {'past_key_values': cache})
                    router.sketches.identify(module, (), {'past_key_values': cache})
                    router(module, q, k, v, mask, **kw)                      # anchor on the same inputs
                    router.begin_step(0, 1)
                fresh.cache.identify(module, (), {'past_key_values': cache})
                consumer_inputs = {}
                original = integration.preqk_attention

                def grab(qq, kk, vv, skipped, eligible, **kws):
                    consumer_inputs.update(q=qq, k=kk, v=vv, skipped=skipped.clone(), eligible=eligible.clone(),
                                           scale=kws['scale'], window=kws.get('window'))
                    return original(qq, kk, vv, skipped, eligible, **kws)
                integration.preqk_attention = grab
                try:
                    routers['O'].cache.entries[layer].decision_step = 0
                    routers['O'](module, q, k, v, mask, **kw)
                finally:
                    integration.preqk_attention = original
                ci = consumer_inputs
                legal = torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device)
                if window:
                    qi = torch.arange(q.shape[2], device=q.device)[:, None]
                    key = torch.arange(k.shape[2], device=q.device)[None, :]
                    legal = key >= qi + (k.shape[2] - q.shape[2]) - int(window) + 1
                rep = q.shape[1] // k.shape[1]

                def decision_reset(name):
                    def fn():
                        routers[name].cache.entries[layer].decision_step = 0
                    return fn
                from experiments.value_direction_hopper import support
                arms = dict(
                    native=(None, lambda: native(module, q, k, v, mask, **kw)),
                    dense_same_mask=(None, lambda: F.scaled_dot_product_attention(
                        q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=legal,
                        scale=kw.get('scaling')).transpose(1, 2).contiguous()),
                    fresh_T=(None, lambda: fresh(module, q, k, v, mask, **kw)),
                    O=(decision_reset('O'), lambda: routers['O'](module, q, k, v, mask, **kw)),
                    H1=(decision_reset('H1'), lambda: routers['H1'](module, q, k, v, mask, **kw)),
                    consumer_triton=(None, lambda: original(ci['q'], ci['k'], ci['v'], ci['skipped'], ci['eligible'],
                                                             scale=ci['scale'], window=ci['window'], variant='generic')),
                    consumer_hopper=(None, lambda: support.attention(ci['q'], ci['k'], ci['v'], ci['skipped'],
                                                                      ci['eligible'], scale=ci['scale'],
                                                                      window=int(ci['window'] or 0), layout=1)))
                names = list(arms)
                for name in names:                                      # warmup every arm
                    for _ in range(args.warmup):
                        if arms[name][0]:
                            arms[name][0]()
                        arms[name][1]()
                torch.cuda.synchronize()
                samples = {name: [] for name in names}
                for r in range(args.reps):
                    order = names[r % len(names):] + names[:r % len(names)]
                    if r % 2:
                        order = order[::-1]
                    for name in order:
                        if arms[name][0]:
                            arms[name][0]()
                        torch.cuda.synchronize()
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        arms[name][1]()
                        b.record()
                        torch.cuda.synchronize()
                        samples[name].append(a.elapsed_time(b))
                keep = (ci['eligible'] & ~ci['skipped'])
                routers['H1'].cache.entries[layer].decision_step = 0
                h1_out = routers['H1'](module, q, k, v, mask, **kw)[0]
                routers['O'].cache.entries[layer].decision_step = 0
                o_out = routers['O'](module, q, k, v, mask, **kw)[0]
                t_out = fresh(module, q, k, v, mask, **kw)[0]
                n_out = native(module, q, k, v, mask, **kw)[0]
                report['layers'][str(layer)] = dict(
                    kind='local' if window else 'global', q=q.shape[2], k_raw=k.shape[2], k_cropped=ci['k'].shape[2],
                    kept_fraction=keep.float().mean().item(), eligible_fraction=ci['eligible'].float().mean().item(),
                    timings_ms={name: dict(median=statistics.median(v), min=min(v), p90=sorted(v)[int(.9 * len(v))], n=len(v))
                                for name, v in samples.items()},
                    output_rel_diff=dict(
                        H1_vs_O=((h1_out.float() - o_out.float()).norm() / o_out.float().norm()).item(),
                        H1_vs_native=((h1_out.float() - n_out.float()).norm() / n_out.float().norm()).item(),
                        fresh_T_vs_native=((t_out.float() - n_out.float()).norm() / n_out.float().norm()).item()))
                print(layer, {k: round(v['median'], 4) for k, v in report['layers'][str(layer)]['timings_ms'].items()}, flush=True)
        finally:
            for router in routers.values():
                router.close()
            fresh.cache.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')


if __name__ == '__main__':
    main()
