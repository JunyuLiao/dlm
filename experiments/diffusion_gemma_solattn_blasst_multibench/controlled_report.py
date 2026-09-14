"""Regenerate all scores and tables from completed immutable final shards."""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .config import condition_map
from .controlled_dataset import audit_manifest, score, digest
from .controlled_runner import load_completed
from .runner import _rows, _write
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import (
    aggregate_routing_stats, positional_token_id_disagreements, paired_bootstrap_ci,
)


def aggregate(calls):
    values=[]
    for call in calls:
        c=dict(call)
        if 'physical_total_tiles' in c: c['eligible_tiles']=c['physical_total_tiles']
        values.append(c)
    return aggregate_routing_stats(values)


def token_counts(dense,sparse):
    compared=max(len(dense),len(sparse))
    return compared-positional_token_id_disagreements(dense,sparse),compared


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def csv_write(path,rows):
    if not rows: return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def metrics(rows,dense,name,scores):
    calls=[c for r in rows for c in r['calls']]
    actual=aggregate(calls)
    dcall=[c for r in rows for c in dense[r['id']]['dense_counterfactual'][name]]
    counterfactual=aggregate(dcall)
    lamcalls=[c for r in rows for c in dense[r['id']]['dense_counterfactual']['blasst_lambda1']]
    maximum=aggregate(lamcalls)
    matched=compared=exact=0;deltas=[];sample_mass=[]
    task_scores=defaultdict(list);task_dense=defaultdict(list)
    for r in rows:
        d=dense[r['id']];m,n=token_counts(d['completion_tokens'],r['completion_tokens']);matched+=m;compared+=n
        exact+=int(d['completion_tokens']==r['completion_tokens'])
        s,ds=scores.get((name,r['id'])),scores.get(('dense',r['id']))
        task_scores[r['task']].append(s);task_dense[r['task']].append(ds)
        if s is not None and ds is not None: deltas.append(s-ds)
        sample_mass.append(aggregate(d['dense_counterfactual'][name])['overall']['retained_dense_attention_mass'])
    accuracy=mean([mean(v) for v in task_scores.values()]);dense_accuracy=mean([mean(v) for v in task_dense.values()])
    condition=condition_map()[name]
    thresholds={}
    for kind in ('global','local'):
        v=[c['effective_blasst_lambda'] for c in calls if c.get('attention_type')==kind and 'effective_blasst_lambda' in c]
        thresholds[kind]=dict(min=min(v),max=max(v),calls_at_one=sum(x==1 for x in v),calls=len(v)) if v else None
    return dict(benchmark=rows[0]['benchmark'],condition=name,method=condition.method,target_sparsity=condition.target_sparsity,
        beta=condition.beta,lambda_by_type=thresholds,n=len(rows),
        n_scored=sum(scores.get((name,r['id'])) is not None for r in rows),
        n_length_terminated=sum(r.get('termination_reason')=='length' for r in rows),
        actual_sparsity=actual['overall']['full_tile_sparsity'],global_sparsity=actual['global']['full_tile_sparsity'],local_sparsity=actual['local']['full_tile_sparsity'],
        retained_attention_mass=counterfactual['overall']['retained_dense_attention_mass'],
        sparse_trajectory_retained_mass=actual['overall']['retained_dense_attention_mass'],
        mass_sample_p10=float(np.quantile(sample_mass,.1)),mass_sample_p50=float(np.quantile(sample_mass,.5)),mass_sample_p90=float(np.quantile(sample_mass,.9)),
        matching_tokens=matched,compared_tokens=compared,token_agreement=matched/compared if compared else 1.,exact_sequence_match=exact/len(rows),
        accuracy=accuracy,dense_accuracy=dense_accuracy,accuracy_relative_to_dense=accuracy/dense_accuracy if dense_accuracy and accuracy is not None else None,
        accuracy_delta=accuracy-dense_accuracy if accuracy is not None and dense_accuracy is not None else None,
        paired_delta_95ci=list(paired_bootstrap_ci(deltas)) if deltas else None,
        actual=actual,dense_counterfactual=counterfactual,dense_lambda1_limit=maximum)


def build(output_dir):
    selection=audit_manifest(output_dir);manifest={r['id']:r for r in _rows(output_dir/'manifest.jsonl')}
    completed={name:load_completed(output_dir,name) for name in condition_map()}
    dense={r['id']:r for r in completed['dense']}
    runtime_path=output_dir/'model_runtime.json'
    layer_types=json.loads(runtime_path.read_text())['model_config']['text_config']['layer_types'] if runtime_path.exists() else []
    violations=[];scores={};scoring_errors=[];summary=[];per_task=[];per_layer=[];per_sample=[]
    for name,rows in completed.items():
        if len({r['id'] for r in rows})!=len(rows): violations.append(f'{name}: duplicate IDs')
        config_path=output_dir/'conditions'/name/'run_config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else {}
        if rows and config.get('condition')!=condition_map()[name].to_dict(): violations.append(f'{name}: threshold/condition mismatch')
        grading=output_dir/'conditions'/name/'livecodebench_grades.json'
        grades={g['id']:g for g in json.loads(grading.read_text())['grades']} if grading.exists() else {}
        for r in rows:
            original=manifest.get(r['id'])
            if original is None or any(original[k]!=r[k] for k in ('prompt_hash','prompt','seed','generation_budget','benchmark','task')):
                violations.append(f'{name}/{r["id"]}: manifest mismatch')
            if r['fingerprint']!=config.get('fingerprint'): violations.append(f'{name}/{r["id"]}: provenance mismatch')
            lcalls=r.get('per_layer',r['calls'])
            if layer_types:
                if {c['layer'] for c in lcalls}!=set(range(len(layer_types))): violations.append(f'{name}/{r["id"]}: missing layers')
                for c in lcalls:
                    kind='local' if layer_types[c['layer']]=='sliding_attention' else 'global'
                    if kind!=c['attention_type']: violations.append(f'{name}/{r["id"]}: layer type mismatch')
            if condition_map()[name].method=='blasst':
                for c in r['calls']:
                    p=config['blasst_length_aware_policy'][c['attention_type']]
                    expected=1. if p['unattainable'] else min(1.,max(float.fromhex('0x1.0p-1022'),p['alpha']*math.exp(p['gamma']*p['target_sparsity'])/c['valid_kv_length']))
                    if not math.isclose(c['effective_blasst_lambda'],expected,rel_tol=1e-12): violations.append(f'{name}/{r["id"]}: wrong effective lambda')
            stats=aggregate(r['calls'])
            for key in ('eligible_tiles','skipped_tiles'):
                if stats['overall'][key]!=stats['local'][key]+stats['global'][key]: violations.append(f'{name}/{r["id"]}: type count mismatch')
            if not 0<=stats['overall']['skipped_tiles']<=stats['overall']['eligible_tiles']: violations.append('invalid tile counts')
            if r['id'] not in dense: violations.append(f'{name}/{r["id"]}: no paired dense')
            try:
                if r['benchmark']=='livecodebench_v6':
                    g=grades.get(r['id'])
                    s=g['score'] if g and g['code_hash']==digest(r['code']) else None
                else: s=score(r,r['prediction'])
                scores[(name,r['id'])]=s
            except Exception as exc:
                scoring_errors.append(dict(condition=name,id=r['id'],error=repr(exc)))
                scores[(name,r['id'])]=None
    for name,all_rows in completed.items():
        by_benchmark=defaultdict(list)
        for r in all_rows:
            if r['id'] in dense: by_benchmark[r['benchmark']].append(r)
        for benchmark,rows in sorted(by_benchmark.items()):
            result=metrics(rows,dense,name,scores);summary.append(result)
            for task in sorted({r['task'] for r in rows}):
                item=metrics([r for r in rows if r['task']==task],dense,name,scores);item['task']=task;per_task.append(item)
            layer_calls=defaultdict(list)
            for r in rows:
                for c in r.get('per_layer',r['calls']): layer_calls[c['layer']].append(c)
            for layer,calls in sorted(layer_calls.items()):
                a=aggregate(calls)['overall']
                dc=[c for r in rows for c in dense[r['id']]['dense_counterfactual'][name] if c['layer']==layer]
                per_layer.append(dict(benchmark=benchmark,condition=name,layer=layer,attention_type=calls[0]['attention_type'],
                    eligible_tiles=a['eligible_tiles'],skipped_tiles=a['skipped_tiles'],actual_sparsity=a['full_tile_sparsity'],
                    dense_retained_mass=aggregate(dc)['overall']['retained_dense_attention_mass']))
            for r in rows:
                item=metrics([r],dense,name,scores);item['id']=r['id'];item['task']=r['task'];per_sample.append(item)
    complete=all(len(v)==122 for v in completed.values()) and not violations and not scoring_errors and all(v is not None for v in scores.values())
    audit=dict(complete=complete,expected=1098,completed={n:len(v) for n,v in completed.items()},violations=violations,scoring_errors=scoring_errors,
        ungraded=sum(v is None for v in scores.values()),selection=selection)
    _write(output_dir/'audit.json',audit)
    _write(output_dir/'summary.json',dict(audit=audit,results=summary,per_task=per_task))
    _write(output_dir/'per_sample_metrics.json',per_sample)
    csv_write(output_dir/'summary.csv',summary);csv_write(output_dir/'per_task.csv',per_task);csv_write(output_dir/'per_layer.csv',per_layer)
    _write(output_dir/'per_layer.json',per_layer)
    report(output_dir,summary,per_task,audit)
    return audit


def percent(x): return '—' if x is None else f'{100*x:.2f}%'


def report(output_dir,summary,per_task,audit):
    text=['# DiffusionGemma: controlled Sol-Attn / BLASST comparison','',
        '**Status: '+('complete' if audit['complete'] else 'INCOMPLETE — partial results, not a completed sweep')+'**','',
        '## Setup','',
        'Pinned DiffusionGemma revision f7f5b7f5fa82ffc52addd066915886d497f5517b; BF16, thinking off, native denoising, canvas 256. IMPORTANT: the adapter treats requested temperature=0 as a sentinel to preserve the native temperature schedule (t_min=0.4, t_max=0.8), NOT greedy decoding. Native maximum denoising steps=48, confidence threshold=0.005, stability threshold=1, entropy bound=0.1. The requested block size does not override the native canvas. All methods use the same cached tokenizer-formatted prompts, per-sample seeds reset before every generation, and token budgets. Decoder text attention only is instrumented; encoder/image attention is outside the reported decoder-attention denominator.',
        '',
        'RULER16K: 10 each of niah_multikey_1, niah_multivalue, niah_multiquery, vt, fwe (50). LongBench v1: 10 each of qasper, hotpotqa, gov_report, trec, passage_retrieval_en (50), selected by seed-42 hash of source ID. LongBench uses official prompts and generation budgets; prompts above 16,384 model tokens use head/tail truncation (2 samples). AIME2024: 10 questions, 5 from each 15-question half of the source file, stratified by position (2 early, 2 middle, 1 late); 2,048-token budget. LiveCodeBench release_v6: 12 questions (4 easy/4 medium/4 hard), 2,048-token budget, one Python solution each. These small subsets are iterative evaluations, not full leaderboard scores.',
        '',
        'RULER partial credit is preserved. LongBench uses the [official task scorers](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py): QA F1, ROUGE-L, classification and retrieval. AIME uses symbolic final-answer verification. LiveCodeBench uses the [official v6 test evaluator](https://github.com/LiveCodeBench/LiveCodeBench), pass@1, timeout 6s/test, in network-disabled read-only nonroot containers. Exact samples, source revisions, truncations and budgets are in manifest.jsonl and dataset_provenance.json.',
        '', '## Routing and calibration','',
        'Both methods use logical 64×64 physical tiles, post-normalization/post-RoPE Q/K, actual GQA expansion and structural masks. Prefix **and canvas** participate; there is no forced-dense prefix. A tile crossing the prefix/canvas boundary stays indivisible and is classified by its starting position. No correction term or optimized deployment kernel is used.',
        '',
        'Sol uses mean-pooled proxies, population row standardization across both regions, and skips z < β. Fixed Gaussian β values are −0.674490, 0, +0.674490, +1.281552. Degenerate populations keep all tiles; an otherwise empty row retains its maximum proxy. Thresholds are never refitted.',
        '',
        'BLASST reuses calibration_policy.json and the existing λL=α exp(γs) policy: λ=min(1, α exp(γ·target)/L), where L is the actual structurally valid KV length of each call. Local α=0.0823017123, γ=31.1098622; global α=0.418544834, γ=12.8301649. Previously unattainable local 50/75/90% targets use λ=1. The original fit R² is only 0.566 local / 0.690 global; transfer is evaluated, not assumed. These coefficients were inherited from prior calibration, without using final benchmark labels or refitting. The current existing BLASST implementation uses row-max/running-max votes and masks a physical tile only when all valid query rows vote to skip; it does NOT execute the older row-granular mask.',
        '', '## Metrics','',
        'Sparsity = Σ skipped eligible tiles / Σ eligible tiles, not a mean of layer/step/sample percentages. Structural impossibilities and padding are excluded. The whole-model column means all instrumented decoder attention, with global/local splits. Retained mass is the valid-query-row-weighted sum of dense probabilities on retained positions. The primary mass column uses cached counterfactual masks on the *dense trajectory*; sparse-trajectory mass is separately exported because post-divergence Q/K are different. A single dense generation per final sample produces all counterfactual sufficient statistics; no dense generations are rerun for reporting.',
        '',
        'Token agreement = matching token IDs / union of output positions; missing/extra positions disagree and comparisons continue after the first divergence. Scores are equal-task macro averages within each benchmark; relative accuracy is sparse score / paired dense score (undefined if dense=0). JSON includes paired prompt-bootstrap 95% intervals, exact-sequence agreement, raw denominators, regions, mass distributions, and λ ranges. No latency, throughput or speedup claims are made.',
        '', '## Main results','']
    for benchmark in sorted({r['benchmark'] for r in summary}):
        text+=['### '+benchmark,'', '| Method | Target | β / λ (global; local) | Actual | Global | Local | Dense-path mass | Token agreement | Score | Relative to dense | n |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in summary if r['benchmark']==benchmark]:
            if r['beta'] is not None: threshold=f"β={r['beta']:+.6f}"
            elif r['method']=='blasst':
                threshold='; '.join(f"{k}: {v['min']:.4g}–{v['max']:.4g}" for k,v in r['lambda_by_type'].items() if v)
            else: threshold='—'
            text.append('| '+' | '.join([r['method'],percent(r['target_sparsity']),threshold,percent(r['actual_sparsity']),percent(r['global_sparsity']),percent(r['local_sparsity']),percent(r['retained_attention_mass']),percent(r['token_agreement']),percent(r['accuracy']),percent(r['accuracy_relative_to_dense']),str(r['n'])])+' |')
        text.append('')
    text+=['## Per-task scores (before macro averaging)','', '| Benchmark/task | Condition | Score | Dense | n |','|---|---|---:|---:|---:|']
    for r in per_task:
        if r['benchmark'] in ('ruler16k','longbench'):
            text.append(f"| {r['benchmark']}/{r['task']} | {r['condition']} | {percent(r['accuracy'])} | {percent(r['dense_accuracy'])} | {r['n']} |")
    text+=['','## Empirical tradeoffs','']
    comparisons=[];correlations=[]
    for benchmark in sorted({r['benchmark'] for r in summary}):
        sol=[r for r in summary if r['benchmark']==benchmark and r['method']=='sol']
        blasst=[r for r in summary if r['benchmark']==benchmark and r['method']=='blasst']
        if sol:
            errors=[r['actual_sparsity']-r['target_sparsity'] for r in sol]
            gaps=[r['global_sparsity']-r['local_sparsity'] for r in sol]
            text.append(f"- {benchmark}: Sol's achieved-minus-target error ranges from {100*min(errors):+.2f} to {100*max(errors):+.2f} percentage points; global-minus-local sparsity ranges from {100*min(gaps):+.2f} to {100*max(gaps):+.2f} points.")
        for s in sol:
            if not blasst: continue
            b=min(blasst,key=lambda b:abs(b['actual_sparsity']-s['actual_sparsity']))
            gap=abs(s['actual_sparsity']-b['actual_sparsity'])
            comparisons.append(dict(benchmark=benchmark,sol=s['condition'],blasst=b['condition'],sparsity_gap=gap,
                sol_actual=s['actual_sparsity'],blasst_actual=b['actual_sparsity'],sol_score=s['accuracy'],blasst_score=b['accuracy'],
                comparable_within_5pp=gap<=.05))
        pairs=[r for r in sol+blasst if r['accuracy'] is not None]
        for x in ('retained_attention_mass','token_agreement'):
            xs=[r[x] for r in pairs];ys=[r['accuracy'] for r in pairs]
            corr=float(np.corrcoef(xs,ys)[0,1]) if len(pairs)>=3 and np.std(xs)>1e-12 and np.std(ys)>1e-12 else None
            correlations.append(dict(benchmark=benchmark,metric=x,pearson_r=corr,conditions=len(pairs)))
    text+=['','Nearest achieved-sparsity pairs (a gap above 5 percentage points is **not** treated as a matched comparison):','',
        '| Benchmark | Sol target / actual / score | BLASST target / actual / score | Actual gap | Within 5pp? |',
        '|---|---|---|---:|---|']
    for p in comparisons:
        text.append(f"| {p['benchmark']} | {p['sol']} / {percent(p['sol_actual'])} / {percent(p['sol_score'])} | {p['blasst']} / {percent(p['blasst_actual'])} / {percent(p['blasst_score'])} | {percent(p['sparsity_gap'])} | {p['comparable_within_5pp']} |")
    text+=['','Descriptive correlations across sparse configurations within each benchmark (small, non-independent condition sets; not evidence of causation):','',
        '| Benchmark | Diagnostic | Pearson r with score | Conditions |','|---|---|---:|---:|']
    for c in correlations:
        value='—' if c['pearson_r'] is None else f"{c['pearson_r']:.3f}"
        text.append(f"| {c['benchmark']} | {c['metric']} | {value} | {c['conditions']} |")
    _write(output_dir/'actual_sparsity_comparisons.json',comparisons)
    _write(output_dir/'correlations.json',correlations)
    text+=['','## Unattainability and anomalies','',
        'The following are dense-trajectory diagnostics at λ=1, not new calibration or proof about every possible sparse trajectory:','']
    for r in summary:
        if r['method']!='blasst': continue
        for kind in ('global','local'):
            maximum=r['dense_lambda1_limit'][kind]['full_tile_sparsity']
            if maximum+1e-8<r['target_sparsity']:
                text.append(f"- {r['benchmark']} {r['condition']} {kind}: dense-path λ=1 reaches {percent(maximum)}, below target {percent(r['target_sparsity'])}.")
    text+=['','Length-limited dense baselines (the official/cached budgets are kept unchanged across methods):','']
    for r in per_task:
        if r['method']=='dense' and r['n_length_terminated']:
            text.append(f"- {r['benchmark']}/{r['task']}: {r['n_length_terminated']}/{r['n']} outputs reached the token budget; score {percent(r['accuracy'])}.")
    if any(r['benchmark']=='ruler16k' and r['task']=='vt' and r['method']=='dense' and r['accuracy']==0 for r in per_task):
        text+=['','The RULER variable-tracking dense baseline scores zero on this subset under its official 30-token budget. Inspect the raw outputs: budget-limited answers/answer-prefix repetition can confound apparent sparse-vs-dense gains. Do not interpret such gains as proof of improved reasoning.']
    text+=['',f"Audit: {json.dumps(audit,sort_keys=True)}",'',
        'Failures are retained in failures.jsonl; missing or ungraded samples are never silently counted as incorrect. Inspect coverage and missing counts before interpreting partial tables. Per-layer counts and dense-path mass are in per_layer.csv/json; per-sample metrics retain raw numerators and denominators.', '']
    (output_dir/'report.md').write_text('\n'.join(text))
