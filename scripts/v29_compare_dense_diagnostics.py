"""Re-tabulate published diagnostic durations; never formal performance evidence."""
import argparse
import csv
import json
import math
from pathlib import Path


def collect(root):
    rows, scopes = [], []
    for campaign in ('cost32k_002', 'cost32k_events003'):
        for arm in ('dense', 'native', 'allkept', 'method'):
            folder = root / campaign / 'raw'
            records = [json.loads(line) for line in (folder / (arm+'.records.jsonl')).read_text().splitlines()]
            if len(records) != 1:
                raise ValueError('Exactly one original profiled request per arm required')
            r = records[0]
            p = json.loads((folder / (arm+'.profile.json')).read_text())
            if r['performance_claim_allowed'] is not False or r['diagnostic_only'] is not True:
                raise ValueError('This report only describes explicitly excluded diagnostic rows')
            n, c, prefills = r['denoise_forward_count'], r['commit_forward_count'], r['prefill_steps']
            if min(n, c, prefills) <= 0 or not math.isclose(r['wall_s'], r['prefill_s']+r['decode_span_s'], abs_tol=1e-8):
                raise ValueError('Invalid diagnostic boundary/counts')
            if any(r['compilation_deltas'].values()):
                raise ValueError('Original diagnostic reports timed compilation')
            row = dict(campaign=campaign, arm=arm, source_commit=r['deploy_commit'],
                requested_graph=r['cudagraph_mode'], engine_seed=r['engine_seed'],
                W_profiled_s=r['wall_s'], prefill_profiled_s=r['prefill_s'],
                S_profiled_s=r['decode_span_s'], N=n, C=c, prefill_forwards=prefills,
                output_tokens=r['output_tokens'], finish_reason=r['finish_reason'],
                S_over_N_profiled_ms=r['decode_span_s']/n*1000,
                N_over_C=n/c, output_tokens_over_C=r['output_tokens']/c,
                warm_N=None, warm_output_tokens=None, random_state_equal_verified=False,
                formal_performance_claim_allowed=False)
            rows.append(row)
            for name, event in p.get('cuda_event_spans', {}).get('spans', {}).items():
                denominator = n if name.startswith('denoise_') else None
                scopes.append(dict(campaign=campaign, arm=arm, scope=name,
                    calls=event['calls'], inclusive_event_sum_ms=event['sum_ms'],
                    per_actual_denoise_forward_ms=event['sum_ms']/denominator if denominator else None,
                    normalization='actual N' if denominator else 'none; mixed phase or nested leaf',
                    additive_cost=False))
    return rows, scopes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'results/v29_20261002'
    rows, scopes=collect(root)
    args.new_output.mkdir(exist_ok=False)
    for name, values in (('requests.csv',rows),('event_scopes.csv',scopes)):
        with (args.new_output/name).open('x',newline='',encoding='utf-8') as out:
            writer=csv.DictWriter(out,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    lines=['# Dense/native diagnostic accounting audit', '',
        'Recomputed from the unchanged public raw002/003 artifacts. These are profiled singleton observations,',
        '**not formal timing, a speed comparison, a causal overhead estimate or accuracy evidence.**',
        'W includes prefill; S/N includes commit/sampler/scheduling amortization. It is not pure model-forward time.',
        'Both campaigns used engine seed28001 and one preceding unprofiled warm request. Warm N/length and',
        'the actual sampling RNG states are absent from these public records; they are unknown, not equal.', '',
        '| Diagnostic | Arm | Requested graphs | W s | Prefill s | S s | N | C | Output tokens | S/N ms |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['campaign']} | {r['arm']} | {r['requested_graph']} | {r['W_profiled_s']:.3f} | {r['prefill_profiled_s']:.3f} | {r['S_profiled_s']:.3f} | {r['N']} | {r['C']} | {r['output_tokens']} | {r['S_over_N_profiled_ms']:.3f} |")
    lines.extend(['', '003 uses PIECEWISE for every arm. Its no-hook dense and native-hook differ in N and output length;',
        'therefore graph configuration alone does not explain that discrepancy. Native also differs between002/003,',
        'which have different source/profiler versions. Same seed and model math do not identify the first divergence.',
        'No claim is made that hooks are correct merely because they delegate the same attention function.', '',
        '## Observed attention scopes in003', '',
        '| Arm | GLOBAL inclusive ms/N | LOCAL inclusive ms/N |', '|---|---:|---:|'])
    for arm in ('dense','native','allkept','method'):
        lookup={s['scope']:s['per_actual_denoise_forward_ms'] for s in scopes if s['campaign']=='cost32k_events003' and s['arm']==arm}
        vals=[lookup.get(key) for key in ('denoise_global_attention','denoise_local_attention')]
        lines.append('| '+arm+' | '+' | '.join('unknown phase' if x is None else f'{x:.6f}' for x in vals)+' |')
    lines.extend(['', 'Each denoising forward contains5GLOBAL and25LOCAL layers. These event sums are inclusive,',
        'include dispatch gaps and instrumentation effects, and are from different generated trajectories.',
        'Dense no-hook has only mixed prefill/commit/denoise attention ranges: dividing them by N would be wrong.',
        'The complete event_scopes.csv preserves all recorded leaves; overlapping/nested/side-stream spans must',
        'not be added to manufacture a wall-time breakdown.002 has no qualified event-span decomposition.', '',
        'Next causal check: same-state native attention versus sparse/observer costs plus a fresh-engine',
        'default/no-hook PIECEWISE/hooked PIECEWISE diagnostic with actual graph-mode receipts and sampling',
        'state checks. Per-request RNG-reset/replay, if used, is separately labelled diagnostic and cannot',
        'retroactively change the existing official-sampler formal campaign. No published run is overwritten.', ''])
    (args.new_output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(requests=len(rows),event_scopes=len(scopes),new_GPU_runs=0)))


if __name__=='__main__':main()
