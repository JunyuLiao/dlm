"""Resumable native RULER4K adaptive end-to-end sweep.

Dense shards and frozen full-centered thresholds are reused from the audited
RULER4K bundle.  This is intentionally a focused adaptive extension: it does
not mutate the historical bundle or silently recalibrate on final questions.
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch

from dllm.models import create_adapter
from experiments import diffusion_gemma_ruler4k_gaussian_sweep as ruler
from experiments.diffusion_gemma_jl_output_aware import runner
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_value_aware.run import shard_path


SOURCE = Path('results/diffusion_gemma_ruler4k_value_direction_s70_v19')
THRESHOLD_SOURCE = Path('results/diffusion_gemma_ruler4k_gaussian_rank_sweep_v16')
ROOT = Path('results/diffusion_gemma_ruler4k_adaptive_final_v21')
TARGETS = (.5, .75)


def policies():
    out = {}
    for row in json.loads((THRESHOLD_SOURCE/'thresholds.json').read_text()):
        if row.get('name') != 'full_centered' or float(row.get('target', -1)) not in TARGETS:
            continue
        out.setdefault(float(row['target']), {})[row['attention_type']] = dict(
            log_threshold=float(row['log_threshold']),
            source=str(THRESHOLD_SOURCE/'thresholds.json'))
    if any(set(out.get(t, {})) != {'local', 'global'} for t in TARGETS):
        raise ValueError('missing frozen full-centered thresholds')
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, default=str)); temp.replace(path)


def aggregate(out):
    records = out.get('records', [])
    eligible = sum(r.get('eligible', 0) for r in records)
    skipped = sum(r.get('skipped', 0) for r in records)
    global_e = sum(r.get('eligible', 0) for r in records if r.get('attention_type') == 'global')
    global_s = sum(r.get('skipped', 0) for r in records if r.get('attention_type') == 'global')
    local_e = sum(r.get('eligible', 0) for r in records if r.get('attention_type') == 'local')
    local_s = sum(r.get('skipped', 0) for r in records if r.get('attention_type') == 'local')
    return dict(id=out['id'], score=out.get('score'), completion_tokens=out.get('completion_tokens', []),
                eligible=eligible, skipped=skipped, sparsity=skipped/max(1, eligible),
                global_eligible=global_e, global_skipped=global_s,
                local_eligible=local_e, local_skipped=local_s,
                global_sparsity=global_s/max(1, global_e), local_sparsity=local_s/max(1, local_e),
                finite_calls=out.get('finite_calls'), work=out.get('work_accounting', {}))


def run(root=ROOT, source=SOURCE, max_examples=None):
    root.mkdir(parents=True, exist_ok=True)
    setup = json.loads((source/'setup.json').read_text())
    rows = list(setup['final'])
    if max_examples is not None: rows = rows[:int(max_examples)]
    frozen = policies()
    config = dict(method='adaptive', family='gaussian', rank=64,
                  cascade_stages=(2, 8, 32, 64), projection_seed=1729,
                  interval_delta=1e-3, exact_fallback=True)
    manifest = dict(schema='ruler4k_adaptive_final_v21', source=str(source),
        source_thresholds=str(THRESHOLD_SOURCE/'thresholds.json'),
        targets=list(TARGETS), config=config, examples=len(rows),
        dense_reused=True, final_scores_not_used_for_thresholds=True,
        note='Native adaptive router keeps unresolved max-rank tiles; exact fallback is evaluated in frozen-history v20.')
    write_json(root/'setup.json', manifest); write_json(root/'thresholds.json', frozen)
    adapter = create_adapter('diffusion_gemma', setup['model'], device='cuda',
        precision='bfloat16', revision=setup['revision']).load()
    # RULER scoring is supplied by the audited RULER4K adapter, not the
    # AIME/LongBench protocol imported by the generic runner.
    old_score = runner.score; runner.score = ruler.base.score
    results = []
    try:
        total = len(rows)*len(TARGETS); done = 0
        for target in TARGETS:
            label = f'adaptive_analytic_s{int(target*100)}'
            policy = {kind: dict(v) for kind,v in frozen[target].items()}
            cond_dir = root/'final'/label/'shards'; cond_dir.mkdir(parents=True, exist_ok=True)
            for row in sorted(rows, key=lambda r:(len(r['prompt_tokens']), r['id'])):
                path = shard_path(root, 'final', label, row['id'])
                if path.exists():
                    out = runner.load_output(path)
                else:
                    out = runner.generate(adapter, row, 'adaptive', config, policy)
                    out['target'] = target; out['condition'] = label
                    runner.write_output(path, out)
                item = aggregate(out); item.update(target=target, condition=label)
                results.append(item)
                done += 1
                heartbeat = dict(stage='final', condition=label, completed=done,
                    total=total, target=target, id=row['id'], updated=time.time())
                write_json(root/'progress.json', heartbeat)
                (root/'progress.md').write_text(
                    f"# Adaptive RULER4K sweep heartbeat\n\n"
                    f"Completed {done}/{total}; condition `{label}`; latest `{row['id']}`.\n"
                    f"Updated Unix time {heartbeat['updated']:.3f}.\n")
                with (root/'progress.jsonl').open('a') as f: f.write(json.dumps(heartbeat)+'\n')
    finally:
        runner.score = old_score
        del adapter; torch.cuda.empty_cache()
    write_json(root/'results.json', results)
    rows_out=[]
    for target in TARGETS:
        group=[r for r in results if r['target']==target]
        def w(key): return sum(r[key]*r['eligible'] for r in group)/max(1,sum(r['eligible'] for r in group))
        global_s=sum(r['global_skipped'] for r in group); global_e=sum(r['global_eligible'] for r in group)
        local_s=sum(r['local_skipped'] for r in group); local_e=sum(r['local_eligible'] for r in group)
        rows_out.append(dict(condition=f'adaptive_analytic_s{int(target*100)}', examples=len(group),
            accuracy=sum(r['score'] or 0 for r in group)/max(1,len(group)),
            actual_sparsity=w('sparsity'), global_sparsity=global_s/max(1,global_e),
            local_sparsity=local_s/max(1,local_e), finite_calls=sum(r['finite_calls'] or 0 for r in group)))
    write_json(root/'summary.json', rows_out)
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0]));w.writeheader();w.writerows(rows_out)
    lines=['# RULER4K native uncertainty-adaptive extension','',
        f'Examples: {len(rows)}; dense outputs are reused from `{source}`.',
        'Thresholds are the previously audited full-centered v16 policies; no final score was used for selection.', '',
        '| condition | examples | accuracy | actual sparsity | global | local |', '|---|---:|---:|---:|---:|---:|']
    for x in rows_out: lines.append(f"| {x['condition']} | {x['examples']} | {100*x['accuracy']:.2f}% | {100*x['actual_sparsity']:.2f}% | {100*x['global_sparsity']:.2f}% | {100*x['local_sparsity']:.2f}% |")
    lines += ['', 'The native implementation computes QK/block softmax and rank-64 sketches, then gates physical tiles with the adaptive cascade. Unresolved tiles are retained at max rank; exact fallback is not silently claimed as deployed work.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    audit=dict(passed=len(results)==len(rows)*len(TARGETS), complete=len(results)==len(rows)*len(TARGETS),
        completed=len(results), expected=len(rows)*len(TARGETS), dense_reused=True,
        artifacts={p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in
                   (root/'setup.json',root/'thresholds.json',root/'results.json',root/'summary.json',root/'summary.csv',root/'report.md')})
    write_json(root/'audit.json',audit)
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--max-examples',type=int,default=None);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for native adaptive sweep')
    print(json.dumps(run(a.output,a.source,a.max_examples),indent=2))


if __name__=='__main__': main()
