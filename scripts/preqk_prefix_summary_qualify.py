"""Real-state qualification for the exact prefix-block-summary selector.

Two levels, both on the real model:

1. ``states`` -- capture real attention inputs at an armed step on one LOCAL
   and one GLOBAL reference layer, with the production causal T actually in
   effect, and compare legacy-recompute against prefix-summary bitmaps,
   eligibility and malformed flags. A SATURATED local prefix is required (the
   sliding cache caps at ``sliding_window - 1``), and the script asserts it was
   reached rather than trusting a filename.

2. ``trajectory`` -- run the SAME bounded generation twice, once per selector,
   with identical seed/prompt/state, and compare the generated token sequence,
   per-canvas decoder calls and stop behaviour exactly. An exact selector must
   reproduce the trajectory token for token; anything else means a decision
   moved. This is the complete decoder-path check: the output consumer is
   unchanged between the two arms, so a divergence here cannot be blamed on
   consumer rounding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def compare_states(adapter, row, args, thresholds, frozen) -> dict[str, Any]:
    from dllm.models import GenerationRequest
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only
    from experiments.numerical_qk_reuse.integration import Attention
    from experiments.value_direction_hopper.query_adaptive import State, observe
    from scripts.native_reuse_phaseAB_diagnostic import true_dense

    captured: dict[int, Any] = {}
    sources: dict[int, Any] = {}
    armed = {'on': False, 'step': 0, 'seen': 0}

    class Recorder:
        def __init__(self):
            self.handles = []
            for name, module in adapter.model.named_modules():
                if adapter.is_blasst_attention_module(name, module):
                    self.handles.append(module.register_forward_pre_hook(
                        self._identify, with_kwargs=True))

        def _identify(self, module, arguments, kwargs):
            cache = kwargs.get('past_key_values', arguments[3] if len(arguments) > 3 else None)
            if cache is not None:
                layer = int(module.layer_idx)
                sources[layer] = (cache.layers[layer].keys.shape[-2], cache.get_seq_length())

        def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                     is_causal=None, sliding_window=None, **kwargs):
            layer = int(module.layer_idx)
            if armed['on'] and layer in args.layers and layer not in captured and layer in sources:
                prefix, absolute = sources[layer]
                captured[layer] = dict(q=q.detach().clone(), k=k.detach().clone(),
                                       v=v.detach().clone(), scaling=scaling,
                                       is_causal=is_causal, sliding_window=sliding_window,
                                       prefix=int(prefix), absolute=int(absolute))
            return true_dense(q, k, v, mask, scaling=scaling, is_causal=is_causal)

        def close(self):
            for handle in self.handles:
                handle.remove()

    from types import MethodType
    binding = _install_dense(adapter)
    recorder = Recorder()
    binding.runtime.attention_override = recorder
    state = State('T', None, m_ref=float(frozen['m_ref']), beta=float(frozen['beta']),
                  gamma=float(frozen['gamma']), diagnostics=False)
    original_step = adapter.model._denoising_step

    class Stop(Exception):
        pass

    def step(this, **kwargs):
        armed['seen'] += 1
        if armed['seen'] >= args.after_steps and not armed['on']:
            armed['on'] = True
            armed['step'] = armed['seen']
            original_step(**kwargs)
            raise Stop()
        return original_step(**kwargs)

    adapter.model._denoising_step = MethodType(step, adapter.model)
    try:
        request = GenerationRequest(prompt=row['prompt'], max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, seed=42, extra={'thinking': True})
        try:
            with observe(adapter.model, state):
                adapter.generate(request)
        except Stop:
            pass
        sensitivity = (None if state.used_weights is None
                       else state.used_weights.detach().clone().contiguous())
    finally:
        adapter.model._denoising_step = original_step
        recorder.close()
        binding.close()

    if set(captured) != set(args.layers):
        raise RuntimeError(f'armed step covered {sorted(captured)}, wanted {sorted(args.layers)}')
    rows = []
    for layer, item in sorted(captured.items()):
        kind = 'local' if item['sliding_window'] else 'global'
        if kind == 'local' and item['prefix'] < args.require_local_prefix:
            raise RuntimeError(f'local prefix {item["prefix"]} < required '
                               f'{args.require_local_prefix}; raise --after-steps/--max-new-tokens')
        q, k, v = item['q'], item['k'], item['v']
        b, h, nq, d = q.shape
        hk, nk = k.shape[1], k.shape[-2]
        scale = float(item['scaling']) if item['scaling'] is not None else d ** -.5
        scores = Attention.observe_scores(q, k, None, scale, bool(item['is_causal']),
                                          item['sliding_window'], 0)
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        matrix = Projections().get(layer, hk, d, 'gaussian', 32, 1729, v.device)
        current = v.float()
        projected = torch.matmul(current, matrix).contiguous()
        valid = torch.isfinite(scores).reshape(b, hk, h // hk, nq, nk).any((2, 3))
        reference = (current.square().sum(-1).masked_fill(~valid, 0.).sum(-1) /
                     valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12).contiguous()
        threshold = float(thresholds[kind]['log_threshold'])
        weights = sensitivity if (sensitivity is not None and sensitivity.shape == (b, nq)) else None
        prefix_tiles = item['prefix'] // 64
        qb, kt = (nq + 127) // 128, (nk + 63) // 64
        summary = allocate_summary(b, h, qb, kt, prefix_tiles, 32, v.device, ('real', layer, 0))
        route_only(scores, projected, reference, sensitivity=weights,
                   log_threshold=threshold, summary=summary, store_summary=True)
        legacy = route_only(scores, projected, reference, sensitivity=weights,
                            log_threshold=threshold)
        optimized = route_only(scores, projected, reference, sensitivity=weights,
                               log_threshold=threshold, summary=summary)
        spread = None if weights is None else float(weights.max() - weights.min())
        rows.append(dict(
            layer=layer, kind=kind, prefix=item['prefix'], absolute=item['absolute'],
            queries=nq, keys=nk, head_dim=d, heads=h, kv_heads=hk,
            prefix_tiles=prefix_tiles, key_tiles=kt,
            summarized_fraction=prefix_tiles / kt,
            summary_megabytes=summary.bytes / 1e6,
            nonuniform_T=bool(weights is not None and spread and spread > 1e-6),
            T_spread=spread,
            retained_tiles=int((legacy.eligible & ~legacy.skipped).sum()),
            eligible_tiles=int(legacy.eligible.sum()),
            skipped_bitmap_identical=bool(torch.equal(legacy.skipped, optimized.skipped)),
            eligible_identical=bool(torch.equal(legacy.eligible, optimized.eligible)),
            malformed_identical=bool(torch.equal(legacy.invalid_tiles, optimized.invalid_tiles)),
            disagreeing_tiles=int((legacy.skipped ^ optimized.skipped).sum()),
        ))
    return dict(armed_step=armed['step'], layers=rows)


def compare_trajectories(adapter, row, args, config) -> dict[str, Any]:
    """Exactness end to end: the same bounded generation under both selectors."""
    from dllm.models import GenerationRequest
    from experiments.numerical_qk_reuse.integration import install
    from experiments.value_direction_hopper.query_adaptive import observe

    outcomes = {}
    for selector in ('legacy_recompute', 'prefix_block_summary'):
        settings = dict(config, selector=selector)
        with install(adapter, settings, 'M1') as runtime:
            state = runtime['state']
            request = GenerationRequest(prompt=row['prompt'],
                                        max_new_tokens=args.trajectory_tokens,
                                        temperature=0.0, seed=42, extra={'thinking': True})
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with observe(adapter.model, state):
                output = adapter.generate(request)
            counters = runtime['counters']()
        outcomes[selector] = dict(
            tokens=list(output.completion_tokens),
            token_count=len(output.completion_tokens),
            termination=output.termination_reason,
            calls=counters['attention_calls'], counters=counters)
    left, right = outcomes['legacy_recompute'], outcomes['prefix_block_summary']
    return dict(
        tokens_identical=left['tokens'] == right['tokens'],
        first_divergence=next((i for i, (a, b) in enumerate(zip(left['tokens'], right['tokens']))
                               if a != b), None),
        token_counts=(left['token_count'], right['token_count']),
        terminations=(left['termination'], right['termination']),
        attention_calls=(left['calls'], right['calls']),
        legacy_counters=left['counters'], summary_counters=right['counters'])


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-name', default='T_s50')
    parser.add_argument('--id', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 5])
    parser.add_argument('--after-steps', type=int, default=90)
    parser.add_argument('--max-new-tokens', type=int, default=2200)
    parser.add_argument('--require-local-prefix', type=int, default=1023)
    parser.add_argument('--trajectory-tokens', type=int, default=512)
    parser.add_argument('--score-refresh-period', type=int, default=8)
    parser.add_argument('--support', default='legacy_junyu_mask')
    parser.add_argument('--output-mode', default='historical_route_preqk_current_output')
    parser.add_argument('--skip-trajectory', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    from dllm.models import create_adapter

    manifest = {r['id']: r for r in json.loads(args.manifest.read_text(encoding='utf-8'))}
    frozen = json.loads(args.policy.read_text(encoding='utf-8'))
    thresholds = frozen['policies'][args.policy_name]
    row = manifest[args.id]
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda',
                             precision='bfloat16', revision=args.revision).load()
    report: dict[str, Any] = dict(schema='preqk_prefix_summary_qualify_v1', id=args.id,
                                  support=args.support, output_mode=args.output_mode)
    report['states'] = compare_states(adapter, row, args, thresholds, frozen)
    if not args.skip_trajectory:
        config = dict(policy=thresholds, score_refresh_period=args.score_refresh_period,
                      decision_interval=1, support=args.support, output_mode=args.output_mode,
                      selector_layers='local', m_ref=float(frozen['m_ref']),
                      beta=float(frozen['beta']), gamma=float(frozen['gamma']),
                      diagnostic=False)
        report['trajectory'] = compare_trajectories(adapter, row, args, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for item in report['states']['layers']:
        print(f"L{item['layer']} {item['kind']:6s} prefix={item['prefix']:5d} "
              f"tiles {item['prefix_tiles']}/{item['key_tiles']} "
              f"({item['summarized_fraction']:.0%}) T_spread={item['T_spread']} "
              f"identical={item['skipped_bitmap_identical'] and item['eligible_identical'] and item['malformed_identical']} "
              f"disagreeing={item['disagreeing_tiles']} summary={item['summary_megabytes']:.2f}MB")
    if 'trajectory' in report:
        t = report['trajectory']
        print(f"trajectory tokens_identical={t['tokens_identical']} counts={t['token_counts']} "
              f"calls={t['attention_calls']} first_divergence={t['first_divergence']}")


if __name__ == '__main__':
    main()
