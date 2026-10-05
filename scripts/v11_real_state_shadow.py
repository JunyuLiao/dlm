"""v11 CP1: real-model shadow qualification of the preselected-support consumer.

Diagnostic only (truncated request). Production M1 (S selector, generic Triton
kernels, production causal T) runs normally and the MODEL CONSUMES THE TRITON
OUTPUT. At the chosen canvases, every ordinary-step ``preqk_attention`` call is
also executed by the new Hopper consumer on the SAME q/k/v views, support
bitmap, window and scale, and both are compared with an FP32 reference over
the same kept+legal entries (tolerances fixed in tests/test_v11_support_consumer.py).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


class Stop(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--support-build', type=Path, required=True)
    parser.add_argument('--id', default='aime26/2')
    parser.add_argument('--canvases', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from dllm.models import GenerationRequest, create_adapter
    from experiments.numerical_qk_reuse import integration
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _rows, _runtime
    from experiments.value_direction_hopper import support
    from experiments.value_direction_hopper.query_adaptive import observe
    from scripts.v10_request_runs import arm_config
    from tests.test_v11_support_consumer import reference
    identity = support.load(args.support_build)
    row = next({k: v for k, v in r.items() if k not in GOLD_FIELDS} for r in _rows(args.manifest) if str(r['id']) == args.id)
    ns = SimpleNamespace(phase='v11shadow', ids=[args.id], manifest=args.manifest, policy=args.policy,
                         model=args.model, revision=args.revision)
    config = arm_config(ns, dict(name='S', condition='M1', selector='prefix_block_summary', kernel_variant='generic',
                                 telemetry='minimal', guard_mode='fused'))
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda', precision='bfloat16',
                             revision=args.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, context = [], {'canvas': -1, 'router': None}
    original = integration.preqk_attention

    def shadow(q, k, v, skipped, eligible, *, scale, window=None, is_causal=False, trace=False, variant='static', **kw):
        result = original(q, k, v, skipped, eligible, scale=scale, window=window, is_causal=is_causal, trace=trace,
                          variant=variant, **kw)
        if context['canvas'] in args.canvases:
            out, lse, invalid, counters = support.attention(q, k, v, skipped, eligible, scale=scale,
                                                            window=int(window or 0), counters=True)
            ref = reference(q, k, v, skipped, eligible, int(window or 0), scale)
            new_err = (out.float() - ref).abs().max().item()
            tri_err = (result.output.float() - ref).abs().max().item()
            rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            keep = (eligible & ~skipped)
            records.append(dict(canvas=context['canvas'], step=context['router'].step, kind='local' if window else 'global',
                                q=q.shape[2], k=k.shape[2], heads=q.shape[1], kv_heads=k.shape[1],
                                kept_fraction=keep.float().mean().item(),
                                eligible_fraction=eligible.float().mean().item(),
                                new_vs_fp32_max=new_err, triton_vs_fp32_max=tri_err, new_rel_fro=rel,
                                new_vs_triton_max=(out.float() - result.output.float()).abs().max().item(),
                                within_envelope=bool(new_err <= 1.5 * tri_err + 1e-3 and rel <= 1e-2),
                                invalid_new=bool(invalid.any()), invalid_triton=bool(result.invalid_scores.any()),
                                counters_match_kept=bool(torch.equal(
                                    counters[..., 1], keep.sum(-1).repeat_interleave(2, -1)[..., :counters.shape[2]].to(torch.int32))),
                                sensitivity_spread=None))
        return result

    integration.preqk_attention = shadow
    try:
        with _runtime(adapter, config['condition'], config) as runtime:
            context['router'] = runtime['router']
            state = runtime['state']
            with observe(adapter.model, state):
                inner = adapter.model._denoising_step

                def outer(this, **kwargs):
                    if int(kwargs['cur_step']) == 48:
                        context['canvas'] += 1
                    if context['canvas'] > max(args.canvases):
                        raise Stop()
                    sens = runtime['router'].query_sensitivity
                    result = inner(**kwargs)
                    if records and records[-1]['sensitivity_spread'] is None:
                        spread = None if sens is None else float(sens.max() - sens.min())
                        for r in records:
                            if r['sensitivity_spread'] is None:
                                r['sensitivity_spread'] = spread
                    return result
                adapter.model._denoising_step = MethodType(outer, adapter.model)
                try:
                    adapter.generate(GenerationRequest(prompt=row['prompt'], max_new_tokens=8192, temperature=0.0,
                                                       seed=42, extra={'thinking': True}))
                except Stop:
                    pass
                finally:
                    adapter.model._denoising_step = inner
    finally:
        integration.preqk_attention = original
    summary = {}
    for kind in ('local', 'global'):
        for canvas in args.canvases:
            sel = [r for r in records if r['kind'] == kind and r['canvas'] == canvas]
            if sel:
                summary[f'{kind}@canvas{canvas}'] = dict(
                    calls=len(sel), keys=sorted({r['k'] for r in sel}), kept_fraction_mean=sum(r['kept_fraction'] for r in sel) / len(sel),
                    all_within_envelope=all(r['within_envelope'] for r in sel),
                    max_new_vs_fp32=max(r['new_vs_fp32_max'] for r in sel), max_triton_vs_fp32=max(r['triton_vs_fp32_max'] for r in sel),
                    max_new_rel_fro=max(r['new_rel_fro'] for r in sel), counters_exact=all(r['counters_match_kept'] for r in sel),
                    any_invalid=any(r['invalid_new'] or r['invalid_triton'] for r in sel),
                    T_nonuniform=any((r['sensitivity_spread'] or 0) > 0 for r in sel))
    report = dict(schema='v11_real_state_shadow_v1', id=args.id, support_build=identity['key'],
                  kernel_sha256=identity['kernel_sha256'], config_fingerprint=config['fingerprint'],
                  summary=summary, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
