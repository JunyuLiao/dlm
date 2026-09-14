"""Prompt-paired development quality and count-weighted sparse-state diagnostics."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_dataset import score
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import positional_token_id_disagreements
from .report import csv_write
from .run import digest


def quality_summary(rows,*,repeats=4000):
    """Task macro scores; paired seeds averaged WITHIN prompt before bootstrap."""
    prompts=defaultdict(list)
    for row in rows:prompts[row['task'],row.get('prompt_id',row['id'])].append((row['dense_score'],row['score']))
    tasks=defaultdict(list)
    for (task,ident),pairs in prompts.items():tasks[task].append(np.mean(pairs,axis=0))
    if not tasks:return {}
    rng=np.random.default_rng(42);boot=[];means=[]
    for task,pairs in sorted(tasks.items()):
        x=np.asarray(pairs);means.append(x.mean(0));deltas=x[:,1]-x[:,0]
        boot.append(deltas[rng.integers(0,len(x),(repeats,len(x)))].mean(-1))
    dense,sparse=np.mean(means,axis=0);lo,hi=np.quantile(np.mean(boot,axis=0),[.025,.975])
    compared=sum(r['compared_tokens'] for r in rows);matching=sum(r['matching_tokens'] for r in rows)
    return dict(prompts=len(prompts),generations=len(rows),tasks=len(tasks),dense_score=float(dense),score=float(sparse),
        accuracy_delta=float(sparse-dense),accuracy_delta_ci_low=float(lo),accuracy_delta_ci_high=float(hi),
        accuracy_relative_to_dense=float(sparse/dense) if dense else None,
        matching_tokens=matching,compared_tokens=compared,token_agreement=matching/compared if compared else 1.,
        sequence_exact_match=float(np.mean([r['sequence_exact_match'] for r in rows])),
        mean_output_tokens=float(np.mean([r['output_tokens'] for r in rows])),
        mean_dense_output_tokens=float(np.mean([r['dense_output_tokens'] for r in rows])),
        budget_hit_rate=float(np.mean([r['budget_hit'] for r in rows])),
        mean_denoising_steps=float(np.mean([r['denoising_steps'] for r in rows])))


def routing_summary(records):
    def total(name):return sum(float(r.get(name,0)) for r in records)
    eligible=total('eligible');skipped=total('skipped');rows=total('sparse_state_rows')
    out=dict(eligible=eligible,skipped=skipped,sparsity=skipped/eligible if eligible else None,
        retained_mass=total('sparse_state_mass_sum')/rows if rows else None,
        mass_sum=total('sparse_state_mass_sum'),diagnostic_query_rows=rows,
        error_sq=total('sparse_state_error_sq'),dense_sq=total('sparse_state_dense_sq'),
        relative_output_error=(total('sparse_state_error_sq')/total('sparse_state_dense_sq'))**.5 if total('sparse_state_dense_sq') else None,
        mean_query_error=total('sparse_state_query_error_sum')/rows if rows else None,
        max_query_error=max((r.get('sparse_state_query_error_max',0.) for r in records),default=0.),
        mean_head_call_query_p95=total('sparse_state_query_error_p95_sum')/total('sparse_state_diagnostic_calls') if total('sparse_state_diagnostic_calls') else None,
        rescued_rows=total('rescued_rows'),budget_excess=total('budget_excess'),
        mean_observation_age=total('age_sum')/eligible if eligible else None,
        unknown_tile_fraction=total('unknown_tiles')/eligible if eligible else None)
    for region in ('prefix','canvas','mixed'):
        n=total(region+'_eligible');d=total(region+'_skipped')
        out.update({region+'_eligible':n,region+'_skipped':d,region+'_sparsity':d/n if n else None})
    return out


def report(stage):
    setup=json.loads((stage/'freeze.json').read_text());sample_index={r['id']:r for r in setup['samples']}
    heldout=setup.get('schema')=='preqk_heldout_quality_v1'
    if 'dense_cache_manifest' in setup:
        setup['cached_dense']=json.loads(Path(setup['dense_cache_manifest']).read_text())
    cached_scores={};score_cache_path=stage/'score_cache.json'
    if score_cache_path.exists():cached_scores=json.loads(score_cache_path.read_text())
    scoring_hash=digest(Path('experiments/diffusion_gemma_solattn_blasst_multibench/controlled_dataset.py'))
    def get_score(row,prediction,identity):
        key=identity+':'+scoring_hash
        if key not in cached_scores:cached_scores[key]=score(row,prediction)
        return cached_scores[key]
    examples=[];routing={};missing={};fingerprints=set();sources=[];cost_rows=[]
    for name,condition in setup['conditions'].items():
        expected={r['id'] for r in setup['samples'] if r['benchmark']==condition['benchmark']};seen=set();records=[]
        costs=defaultdict(float)
        for path in sorted((stage/'conditions'/name/'shards').glob('*.json')):
            shard=json.loads(path.read_text());ident=shard['id'];assert ident in expected and ident not in seen
            row=sample_index[ident];assert all(shard[k]==row[k] for k in ('prompt_hash','seed','generation_budget'))
            assert hashlib.sha256(shard['prompt'].encode()).hexdigest()==row['prompt_hash']
            assert shard['condition']==name and shard['config']==condition['config']
            options={k:condition[k] for k in ('isolate','allocation_policy') if k in condition}
            if options:
                assert shard['router_options']==options
                if 'isolate' in options:assert shard['costs']['isolated_sparse_calls']==1
                if 'allocation_policy' in options:
                    assert shard['allocation_audit']
                    assert all(a['quota_delta']==0 and a['max_abs_row_delta']==0 for a in shard['allocation_audit'])
                    assert sum(r['eligible']-r['skipped'] for r in shard['records'])==sum(r['shadow_uniform_retained'] for r in shard['records'])
            seen.add(ident);fingerprints.add(shard['fingerprint']);sources.append(dict(path=str(path),sha256=digest(path)))
            cache=setup['cached_dense'][ident];assert digest(cache['path'])==cache['sha256']
            dense=json.loads(Path(cache['path']).read_text())
            assert all(dense[k]==row[k] for k in ('prompt','seed','generation_budget'))
            dt=dense['completion_tokens'];st=shard['completion_tokens']
            compared=max(len(dt),len(st));disagreed=positional_token_id_disagreements(dt,st)
            layer_ids={r['layer'] for r in shard['records']}
            # Each decoder forward calls each layer once. Check that assumption.
            calls_per_layer={layer:sum(r['calls'] for r in shard['records'] if r['layer']==layer and r['head']==0) for layer in layer_ids}
            assert len(set(calls_per_layer.values()))==1
            steps=next(iter(calls_per_layer.values()));assert steps*len(layer_ids)==shard['attention_calls']
            examples.append(dict(id=ident,prompt_id=row.get('prompt_id',ident),condition=name,benchmark=row['benchmark'],task=row['task'],seed=row['seed'],
                dense_score=get_score(row,dense['prediction'],cache['sha256']),score=get_score(row,shard['prediction'],digest(path)),
                matching_tokens=compared-disagreed,compared_tokens=compared,sequence_exact_match=dt==st,
                output_tokens=len(st),dense_output_tokens=len(dt),budget_hit=shard['termination_reason']=='length',
                denoising_steps=steps,adapter_reported_denoising_steps=shard['model_evaluations'],
                max_state_bytes=shard['max_state_bytes'],**routing_summary(shard['records'])))
            records.extend(shard['records'])
            for field,value in shard['costs'].items():costs[field]+=value
            for field in ('prefix_value_elements','canvas_value_elements','diagnostic_e2e_seconds'):costs[field]+=shard[field]
        missing[name]=sorted(expected-seen);routing[name]=records
        if seen:cost_rows.append(dict(condition=name,benchmark=condition['benchmark'],generations=len(seen),
            **dict(costs),executed_qk_saved=0,executed_pv_saved=0))
    assert len(fingerprints)<=1,'mixed online run fingerprints'
    if fingerprints:
        smoke=json.loads((stage/'smoke.json').read_text())
        assert smoke['passed'] and fingerprints=={smoke['fingerprint']},'shards lack matching passing CUDA smoke'
    _write(score_cache_path,cached_scores);csv_write(stage/'per_example.csv',examples)
    csv_write(stage/'cost_breakdown.csv',cost_rows)
    summary=[];breakdowns=[]
    for name,condition in setup['conditions'].items():
        subset=[r for r in examples if r['condition']==name]
        if not subset:continue
        measures={kind:routing_summary([r for r in routing[name] if kind=='overall' or r['attention_type']==kind]) for kind in ('overall','local','global')}
        row=dict(condition=name,benchmark=condition['benchmark'],predictor=condition['predictor'],history_source=condition['config']['history_source'],
            target_sparsity=condition['config']['sparsity'],refresh_interval=condition['config']['refresh_interval'],exploration_tiles=condition['config']['exploration_tiles'],
            complete=not missing[name],**quality_summary(subset),**{kind+'_'+k:v for kind,m in measures.items() for k,v in m.items()},
            max_state_bytes=max(r['max_state_bytes'] for r in subset),executed_qk_saved=0,executed_pv_saved=0)
        summary.append(row)
        for task in sorted({r['task'] for r in subset}):
            breakdowns.append(dict(condition=name,benchmark=condition['benchmark'],task=task,**quality_summary([r for r in subset if r['task']==task])))
    csv_write(stage/'summary.csv',summary);csv_write(stage/'per_task.csv',breakdowns)
    heterogeneity=[]
    for name,records in routing.items():
        for dimension in ('layer','head','step'):
            for label in sorted({r[dimension] for r in records}):
                rs=[r for r in records if r[dimension]==label]
                heterogeneity.append(dict(condition=name,dimension=dimension,group=label,head_calls=sum(r['calls'] for r in rs),**routing_summary(rs)))
    csv_write(stage/'routing_breakdowns.csv',heterogeneity)
    shadow_rows=[]
    for name,records in routing.items():
        if not any('shadow_uniform_rows' in r for r in records):continue
        for kind in ('overall','local','global'):
            rs=[r for r in records if kind=='overall' or r['attention_type']==kind]
            transformed=[]
            for r in rs:
                shadow={k.replace('shadow_uniform_','sparse_state_'):v for k,v in r.items() if k.startswith('shadow_uniform_')}
                shadow.update(eligible=r['shadow_uniform_eligible'],skipped=r['shadow_uniform_eligible']-r['shadow_uniform_retained'])
                transformed.append(shadow)
            allocated=routing_summary(rs);uniform=routing_summary(transformed)
            assert allocated['eligible']==uniform['eligible'] and allocated['skipped']==uniform['skipped']
            shadow_rows.append(dict(condition=name,attention_type=kind,**{'allocated_'+k:v for k,v in allocated.items()},
                **{'same_state_uniform_'+k:v for k,v in uniform.items()}))
    if shadow_rows:csv_write(stage/'allocation_same_state.csv',shadow_rows)
    audit=dict(complete=all(not m for m in missing.values()),missing=missing,fingerprints=sorted(fingerprints),
        completed_generations=len(examples),source_shards=sources,heldout_evaluated=heldout and bool(examples),executed_backend='dense-mask emulation',
        uncertainty='task-stratified paired prompt bootstrap; seeds averaged within prompt',
        token_agreement='all positional token IDs after divergence; missing/extra positions disagree; sum matches / sum union length',
        denoising_step_breakdown='descriptive call-weighted; unequal prompt survival is not a causal time comparison')
    _write(stage/'audit.json',audit);_write(stage/'summary.json',dict(audit=audit,conditions=summary,per_task=breakdowns))
    lines=['# Online-history '+('held-out quality validation' if heldout else 'development rollouts'),'',f"Completed sparse generations: {len(examples)}. Complete: {audit['complete']}. Held-out evaluation: {audit['heldout_evaluated']}.",
        '', 'Dense-reference history uses paid dense observations on each candidate’s sparse trajectory. '
        'Sparse history uses only observed sparse-softmax probabilities. Both choose masks before current QK. '
        'All attention-mass/error values below are same-state sparse-trajectory diagnostics, not measurements on the original dense trajectory. '
        'Executed attention is dense-mask emulation: no QK or PV speedup is established.',
        '', '| Benchmark | Predictor | History | Target | Actual | Global | Local | Mass | Token agreement | Score | Δ dense |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['benchmark']} | {r['predictor']} | {r['history_source']} | {r['target_sparsity']:.0%} | {r['overall_sparsity']:.2%} | {r['global_sparsity']:.2%} | {r['local_sparsity']:.2%} | {r['overall_retained_mass']:.4f} | {r['token_agreement']:.2%} | {r['score']:.2%} | {r['accuracy_delta']:+.2%} |")
    lines+=['','Per-task scores, prompt bootstrap intervals, raw counts, output lengths, budget-hit rates and denoising counts are in the CSV/JSON artifacts. '
        'Initialization, refresh and exploration are included in achieved physical sparsity. '
        'Step breakdowns are descriptive; matched-prompt step contrasts are needed before interpreting temporal trends. '
        + ('These are frozen held-out quality data. Interpret the prompt-level interval against the predeclared tolerance; no deployment speedup follows from quality preservation.' if heldout else
         'These are development data, not evidence of held-out noninferiority. Allocation and frozen validation decisions remain separate.')]
    (stage/'report.md').write_text('\n'.join(lines)+'\n')
