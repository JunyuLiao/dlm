"""v12 CP1 bounded real-model probe (truncated requests; not complete answers).

1. Dispatch proof per arm (D_native, T_G, G1, G3, B8) over the first
   ``--steps`` denoising steps: every step must show 25 LOCAL registry calls
   that are UNTAGGED (so the BLASST dispatcher's first branch returns the
   captured original native SDPA with the original arguments) and 5 GLOBAL
   calls reaching exactly the arm's router (none for D); zero eager/BLASST-mask
   fallbacks; router history / sketch / summary state only on GLOBAL layers.
2. GLOBAL summary qualification: G1 with prefix_block_summary on GLOBAL layers
   vs G1 with legacy_recompute, same truncated trajectory: every routing
   decision (skipped, eligible, invalid tiles) must be identical, summaries
   must actually be built and hit, and the argmax canvases must agree.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


class Stop(Exception):
    pass


def digest(t):
    return hashlib.sha256(t.detach().to(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--arms', required=True, help='JSON list of v10 arm dicts')
    parser.add_argument('--id', default='aime26/2')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--summary-canvases', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import importlib
    import dllm.attention.blasst.integration as blasst
    from dllm.models import GenerationRequest, create_adapter
    from experiments.numerical_qk_reuse import integration
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _rows, _runtime
    from experiments.value_direction_hopper import integration as junyu
    from experiments.value_direction_hopper.query_adaptive import observe
    from scripts.v10_request_runs import arm_config
    row = next({k: v for k, v in r.items() if k not in GOLD_FIELDS} for r in _rows(args.manifest) if str(r['id']) == args.id)
    ns = SimpleNamespace(phase='v12probe', ids=[args.id], manifest=args.manifest, policy=args.policy,
                         model=args.model, revision=args.revision)
    arms = {a['name']: a for a in json.loads(args.arms)}
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda', precision='bfloat16',
                             revision=args.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    registry = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_')).ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as native
    report = dict(schema='v12_scope_probe_v1', id=args.id, arms={}, summary_qualification={})

    def run(config, max_steps=None, max_canvas=None, on_step=None):
        """Truncated production request through runner._runtime + one observe."""
        steps = {'n': 0, 'canvas': -1}
        with _runtime(adapter, config['condition'], config) as runtime:
            state = runtime.get('state') if isinstance(runtime, dict) else None
            with (observe(model, state) if state is not None else torch.no_grad()):
                inner = model._denoising_step

                def outer(this, **kwargs):
                    if int(kwargs['cur_step']) == 48:
                        steps['canvas'] += 1
                    if (max_steps is not None and steps['n'] >= max_steps) or \
                            (max_canvas is not None and steps['canvas'] > max_canvas):
                        raise Stop()
                    result = inner(**kwargs)
                    steps['n'] += 1
                    if on_step:
                        on_step(steps['n'], result)
                    return result
                model._denoising_step = MethodType(outer, model)
                try:
                    adapter.generate(GenerationRequest(prompt=row['prompt'], max_new_tokens=8192, temperature=0.0,
                                                       seed=42, extra={'thinking': True}))
                except Stop:
                    pass
                finally:
                    model._denoising_step = inner
            counters = runtime.get('counters') if isinstance(runtime, dict) else None
            tagged = {int(m.layer_idx): hasattr(m, '_blasst_2d_runtime') for n, m in model.named_modules()
                      if adapter.is_blasst_attention_module(n, m)}
            return steps, (counters() if callable(counters) else None), tagged

    # ---------------- 1. dispatch proof ----------------
    for name, arm in arms.items():
        config = arm_config(ns, arm)
        per_step = []
        counts = collections.Counter()
        saved = dict(eager=blasst.dense_eager_attention_forward, masked=blasst.blasst_2d_attention_forward,
                     m1=integration.Attention.__call__, t=junyu.Attention.__call__)

        def m1_spy(self, module, *a, **k):
            counts['m1_' + ('local' if module.is_sliding else 'global')] += 1
            return saved['m1'](self, module, *a, **k)

        def t_spy(self, module, *a, **k):
            counts['T_' + ('local' if module.is_sliding else 'global')] += 1
            return saved['t'](self, module, *a, **k)

        def eager_spy(*a, **k):
            counts['eager'] += 1
            return saved['eager'](*a, **k)

        def masked_spy(*a, **k):
            counts['blasst_masked'] += 1
            return saved['masked'](*a, **k)
        integration.Attention.__call__, junyu.Attention.__call__ = m1_spy, t_spy
        blasst.dense_eager_attention_forward, blasst.blasst_2d_attention_forward = eager_spy, masked_spy
        original_entry = {'fn': None}

        def registry_spy(module, *a, **k):
            if type(module).__name__ != 'DiffusionGemmaDecoderTextAttention':
                counts['encoder_calls'] += 1           # prefill/commit encoder; not a decoder step call
                return original_entry['fn'](module, *a, **k)
            counts['registry_' + ('local' if module.is_sliding else 'global')] += 1
            if module.is_sliding and hasattr(module, '_blasst_2d_runtime'):
                counts['local_tagged'] += 1
            return original_entry['fn'](module, *a, **k)

        def on_step(n, result):
            per_step.append(dict(counts))
            counts.clear()

        # Install the registry spy AFTER the binding exists (the binding patches registry['sdpa']).
        from contextlib import contextmanager

        @contextmanager
        def spied_runtime(adapter_, condition, cfg):
            with _runtime(adapter_, condition, cfg) as rt:
                original_entry['fn'] = registry['sdpa']
                registry['sdpa'] = registry_spy
                try:
                    yield rt
                finally:
                    registry['sdpa'] = original_entry['fn']
        started = time.perf_counter()
        # Simpler and explicit: re-implement the truncated run with the spied runtime.
        steps = {'n': 0}
        with spied_runtime(adapter, config['condition'], config) as runtime:
            st = runtime.get('state') if isinstance(runtime, dict) else None
            with (observe(model, st) if st is not None else torch.no_grad()):
                inner = model._denoising_step

                def outer(this, **kwargs):
                    if steps['n'] >= args.steps:
                        raise Stop()
                    result = inner(**kwargs)
                    steps['n'] += 1
                    on_step(steps['n'], result)
                    return result
                model._denoising_step = MethodType(outer, model)
                try:
                    adapter.generate(GenerationRequest(prompt=row['prompt'], max_new_tokens=8192, temperature=0.0,
                                                       seed=42, extra={'thinking': True}))
                except Stop:
                    pass
                finally:
                    model._denoising_step = inner
            counter_fn = runtime.get('counters') if isinstance(runtime, dict) else None
            effective = counter_fn() if callable(counter_fn) else None
            tagged = sorted(int(m.layer_idx) for n, m in model.named_modules()
                            if adapter.is_blasst_attention_module(n, m) and hasattr(m, '_blasst_2d_runtime'))
        integration.Attention.__call__, junyu.Attention.__call__ = saved['m1'], saved['t']
        blasst.dense_eager_attention_forward, blasst.blasst_2d_attention_forward = saved['eager'], saved['masked']
        routed_key = {'native_dense': None, 'global_T': 'T_global'}.get(config['condition'], 'm1_global')
        ok = all(s.get('registry_local', 0) == 25 and s.get('registry_global', 0) == 5 and not s.get('local_tagged')
                 and not s.get('eager') and not s.get('blasst_masked') and not s.get('m1_local') and not s.get('T_local')
                 and (routed_key is None or s.get(routed_key, 0) == 5)
                 and (routed_key is not None or not (s.get('m1_global') or s.get('T_global')))
                 for s in per_step)
        global_layers = [5, 11, 17, 23, 29]
        state_ok = effective is None or all(set(effective.get(k, [])) <= set(global_layers)
                                            for k in ('history_layers', 'sketch_layers', 'summary_layers'))
        report['arms'][name] = dict(condition=config['condition'], fingerprint=config['fingerprint'], steps=len(per_step),
                                    per_step=per_step, dispatch_ok=bool(ok), state_only_global=bool(state_ok),
                                    tagged_modules=tagged, effective=effective,
                                    config={k: config.get(k) for k in ('decision_interval', 'score_refresh_period', 'selector',
                                                                       'selector_layers', 'consumer', 'guard_mode', 'telemetry',
                                                                       'kernel_variant', 'collect', 'plugin', 'support_binary')},
                                    seconds=time.perf_counter() - started)
        print(name, 'dispatch_ok', ok, 'state_only_global', state_ok, 'tagged', tagged, flush=True)

    # ---------------- 2. GLOBAL summary qualification ----------------
    base = next(a for a in arms.values() if a['condition'] == 'global_M1')
    results = {}
    for selector in ('prefix_block_summary', 'legacy_recompute'):
        config = arm_config(ns, dict(base, name='G1_' + selector, selector=selector))
        decisions = []
        saved_route, saved_attn = integration.route_only, integration.attention

        def route_spy(*a, **k):
            r = saved_route(*a, **k)
            decisions.append(('route', r.skipped.clone(), r.eligible.clone(), r.invalid_tiles.clone()))
            return r

        def attn_spy(*a, **k):
            r = saved_attn(*a, **k)
            if k.get('skipped') is None and len(a) > 2:          # anchor route+PV (decision produced here)
                decisions.append(('anchor', r.skipped.clone(), r.eligible.clone(), r.invalid_scores.clone()))
            return r
        integration.route_only, integration.attention = route_spy, attn_spy
        canvases = []
        try:
            _, effective, _ = run(config, max_canvas=args.summary_canvases - 1,
                                  on_step=lambda n, res: canvases.append(res[1].clone()))
        finally:
            integration.route_only, integration.attention = saved_route, saved_attn
        results[selector] = dict(decisions=[(kind, digest(s), digest(e), digest(b)) for kind, s, e, b in decisions],
                                 skipped_fraction=float(torch.stack([d[1].float().mean() for d in decisions]).mean()) if decisions else None,
                                 canvases=[digest(c) for c in canvases], counters=effective)
    a, b = results['prefix_block_summary'], results['legacy_recompute']
    report['summary_qualification'] = dict(
        canvases_run=args.summary_canvases, decision_calls=len(a['decisions']),
        decisions_identical=a['decisions'] == b['decisions'], argmax_canvases_identical=a['canvases'] == b['canvases'],
        summary_builds=a['counters']['summary_builds'], summary_hits=a['counters']['summary_hits'],
        summary_misses=a['counters']['summary_misses'], summary_peak_bytes=a['counters']['summary_peak_bytes'],
        legacy_summary_builds=b['counters']['summary_builds'], skipped_fraction_mean=a['skipped_fraction'],
        qualified=bool(a['decisions'] == b['decisions'] and a['counters']['summary_hits'] > 0))
    print(json.dumps(report['summary_qualification']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')


if __name__ == '__main__':
    main()
