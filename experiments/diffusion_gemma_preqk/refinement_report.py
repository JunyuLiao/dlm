"""Audit refinements and distinguish same-state allocation from rollout work."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .config import ROOT
from .online_report import report
from .work_volume import analyze
from .run import digest,sample_key
from .report import csv_write
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def paired_error_deltas(shards):
    rows=[]
    for data in shards:
        for kind in ('overall','local','global'):
            rs=[r for r in data['records'] if kind=='overall' or r['attention_type']==kind]
            total=lambda field:sum(r[field] for r in rs)
            allocated=(total('sparse_state_error_sq')/total('sparse_state_dense_sq'))**.5
            uniform=(total('shadow_uniform_error_sq')/total('shadow_uniform_dense_sq'))**.5
            n=total('sparse_state_rows');sn=total('shadow_uniform_rows');assert n==sn
            rows.append(dict(id=data['id'],attention_type=kind,allocated_error=allocated,uniform_shadow_error=uniform,
                error_delta=allocated-uniform,mass_delta=(total('sparse_state_mass_sum')-total('shadow_uniform_mass_sum'))/n,
                allocated_retained=total('eligible')-total('skipped'),shadow_retained=total('shadow_uniform_retained')))
            assert rows[-1]['allocated_retained']==rows[-1]['shadow_retained']
    return rows


def analyze_refinement(stage):
    report(stage);analyze(stage)
    setup=json.loads((stage/'freeze.json').read_text());audit=json.loads((stage/'audit.json').read_text())
    if not audit['complete']:raise RuntimeError('refinement conclusions require all frozen generations')
    parent=Path(setup['selection']['parent_stage']);regression=[];allocation=[]
    for name,entry in setup['conditions'].items():
        for row in (r for r in setup['samples'] if r['benchmark']==entry['benchmark']):
            path=stage/'conditions'/name/'shards'/f'{sample_key(row)}.json';data=json.loads(path.read_text())
            if entry['predictor']=='last_mass_control':
                oldpath=parent/'conditions'/'longbench__last_mass_s50__sparse'/'shards'/path.name
                old=json.loads(oldpath.read_text())
                regression.append(dict(id=row['id'],exact_tokens=data['completion_tokens']==old['completion_tokens'],
                    identical_config=data['config']==old['config'],same_attention_calls=data['attention_calls']==old['attention_calls'],
                    same_retained_tiles=data['costs']['retained_tiles']==old['costs']['retained_tiles'],
                    parent_sha256=digest(oldpath),current_sha256=digest(path)))
            if 'allocation_policy' in entry:allocation.append(data)
    if regression and not all(r[k] for r in regression for k in ('exact_tokens','identical_config','same_attention_calls','same_retained_tiles')):
        _write(stage/'regression_audit.json',dict(passed=False,comparisons=regression))
        raise RuntimeError('uniform control differs from completed parent; investigate before allocation conclusions')
    _write(stage/'regression_audit.json',dict(passed=True,comparisons=regression))
    paired=paired_error_deltas(allocation);csv_write(stage/'allocation_per_prompt.csv',paired)
    paired_summary=[]
    for kind in ('overall','local','global'):
        rs=[r for r in paired if r['attention_type']==kind]
        if not rs:continue
        x=np.asarray([r['error_delta'] for r in rs]);rng=np.random.default_rng(42)
        lo,hi=np.quantile(x[rng.integers(0,len(x),(4000,len(x)))].mean(-1),[.025,.975])
        paired_summary.append(dict(attention_type=kind,prompts=len(x),mean_error_delta=float(x.mean()),
            ci_low=float(lo),ci_high=float(hi),fraction_lower_error=float((x<0).mean()),
            mean_mass_delta=float(np.mean([r['mass_delta'] for r in rs]))))
    csv_write(stage/'allocation_paired_deltas.csv',paired_summary)
    summary=json.loads((stage/'summary.json').read_text())['conditions']
    work={r['condition']:r for r in csv.DictReader((stage/'trajectory_work.csv').open())}
    lines=['# Bounded refinement findings','',
        'All conditions use development examples only. These are not held-out noninferiority results. '
        'Physical sparsity includes initialization, refresh and exploration; retained-work ratios include changed denoising duration. '
        'No physical tile matmuls are avoided in the dense-mask execution backend.',
        '', '| Benchmark | Condition | Score | Dense | Δ dense (95% prompt CI) | Actual sparsity | Steps / dense | Retained tiles / dense |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        w=work[r['condition']]
        lines.append(f"| {r['benchmark']} | {r['predictor']} / {r['history_source']} | {r['score']:.2%} | {r['dense_score']:.2%} | {r['accuracy_delta']:+.2%} [{r['accuracy_delta_ci_low']:+.2%}, {r['accuracy_delta_ci_high']:+.2%}] | {r['overall_sparsity']:.2%} | {float(w['denoising_step_ratio']):.3f} | {float(w['retained_tile_volume_ratio']):.3f} |")
    lines+=['','## Fixed-budget, same-state allocation','',
        'The ranking is fixed to plain last-step mass. A uniform shadow mask is evaluated on exactly the allocated trajectory’s Q/K/V and causal history. '
        'Its total retention matches the allocated mask within each attention type, head, query block and forward. '
        'This isolates allocation on those states; the full uniform and allocated rollouts can have different duration and history. '
        'The fresh uniform control exactly reproduces the completed parent tokens and work counts.',
        '', '| Attention type | Prompts | Mean prompt error Δ vs uniform shadow | 95% CI | Fraction improved |',
        '|---|---:|---:|---:|---:|']
    for r in paired_summary:
        lines.append(f"| {r['attention_type']} | {r['prompts']} | {r['mean_error_delta']:+.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] | {r['fraction_lower_error']:.0%} |")
    lines+=['','## Interpretation boundaries','',
        'Single-call AIME experiments perturb only the first matching layer at step 1 in the entire request. '
        'They test a much smaller intervention than sustained sparsity, not a practical high-sparsity policy. '
        'A surviving isolated perturbation together with failing sustained execution is consistent with accumulated/distributed effects; '
        'it does not uniquely identify the mechanism. The lower-sparsity AIME history pair separately tests density and observation-source effects.',
        '', 'Layer groups and isolated-call locations were frozen from observational development error. '
        'A per-layer error ranking is not automatically a causal sensitivity ranking. '
        'Any allocation decision must consider both same-state preservation and TOTAL rollout work. '
        'Reference profiling and frozen held-out validation are separate gates; no net sparse speedup is established here.']
    (stage/'findings.md').write_text('\n'.join(lines)+'\n')
    _write(stage/'refinement_audit.json',dict(complete=True,generations=audit['completed_generations'],
        regression_passed=True,matched_allocation_prompts=len(allocation),heldout_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,default=ROOT/'refinement_v1')
    analyze_refinement(p.parse_args().stage)
