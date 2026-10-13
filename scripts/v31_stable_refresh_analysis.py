"""Official scoring and paired workload-separated, problem-clustered analysis."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import v31_official as official

ROOT=Path(__file__).resolve().parents[1]


def cell_key(row): return f"{row['dataset']}|{row['index']}|{row['panel_seed']}|{row['repeat']}"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonwrite(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def scoring_inputs(study):
    return dict(native_panel_cells=study/'private/cells_native_panel154.json',
        longbench_cells=study/'private/cells_longbench64_seed1.json',
        aime_cells=study/'private/cells_aime26_30x3_seed42_43_44.json',
        longbench_official_gold=Path('/home/exouser/dyh/ljy_value_20261007/pools/longbench_v2_ofc/longbench_v2_0shot_think_gold.json'),
        aime26_official_answers=ROOT/'results/v31_value_matrix_20261011/private/aime26/aime2026_answers.jsonl')
def distribution(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(values.max()))
def cluster_pair(candidates,control,score,refscore):
    clusters=defaultdict(list)
    for key,r in candidates.items(): clusters[r['index']].append(key)
    ids=sorted(clusters)
    numer=np.array([sum(candidates[k]['denoise_forwards'] for k in clusters[i]) for i in ids],dtype=float)
    denom=np.array([sum(control[k]['denoise_forwards'] for k in clusters[i]) for i in ids],dtype=float)
    adiff=np.array([np.mean([float(score[k])-float(refscore[k]) for k in clusters[i]]) for i in ids])
    gen=np.random.default_rng(51994); samples=gen.integers(len(ids),size=(10000,len(ids)))
    ratios=numer[samples].sum(1)/denom[samples].sum(1)
    delta=100*adiff[samples].mean(1)
    return dict(problem_clusters=len(ids),wins=sum(score[k] and not refscore[k] for k in candidates),
        losses=sum(refscore[k] and not score[k] for k in candidates),accuracy_delta_pp=100*adiff.mean(),
        accuracy_delta_cluster_ci95_pp=np.quantile(delta,[.025,.975]).tolist(),
        aggregate_N_ratio=numer.sum()/denom.sum(),aggregate_N_ratio_cluster_ci95=np.quantile(ratios,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);args=p.parse_args();study=args.study.resolve()
    completion=json.loads((study/'native_generation_complete.json').read_text())
    identity=json.loads((study/'specs/execution_identity.json').read_text())
    scoring_identity=json.loads((study/'specs/scoring_identity.json').read_text())
    for path,digest in scoring_identity['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('frozen scoring source mismatch')
    for label,digest in scoring_identity['input_sha256'].items():
        if sha(scoring_inputs(study)[label])!=digest:raise RuntimeError('frozen official scoring input mismatch')
    for path,digest in identity['source_sha256'].items():
        if sha(ROOT/path)!=digest: raise RuntimeError('source identity mismatch during final analysis')
    resultdir=study/'reports';resultdir.mkdir(exist_ok=True); rows={};scores={};receipts={};score_diagnostics={};configs={}
    expected={cell_key(r) for r in json.loads((study/'private/cells_native_panel154.json').read_text())}
    for arm in completion['arms']:
        entries=[e for e in completion['shards'] if e['arm']==arm]
        leaves=[study/e['path'] for e in entries]
        for entry,leaf in zip(entries,leaves):
            for name,digest in entry['artifacts_sha256'].items():
                if sha(leaf/name)!=digest:raise RuntimeError('completed native artifact identity mismatch')
        configs[arm]=[]
        rr=[];out=study/'scoring'/arm; (out/'private').mkdir(parents=True,exist_ok=True)
        private={ds:(out/'private'/f'run_{arm}_{ds}.private.jsonl').open('w') for ds in ('longbench_v2_0shot_think','aime26')}
        for leaf in leaves:
            receipt=json.loads((leaf/'receipt.json').read_text())
            if receipt['status']!='complete' or receipt['validation_errors']:raise RuntimeError('unclean native receipt')
            cfg=json.loads((leaf/'config.json').read_text())
            configs[arm].append(cfg)
            for path,digest in cfg['source_sha256'].items():
                if sha(ROOT/path)!=digest:raise RuntimeError('native leaf source mismatch')
            rr.extend(json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines())
            for line in (leaf/'private'/f'run_{arm}.private.jsonl').read_text().splitlines():
                r=json.loads(line);private[r['dataset']].write(line+'\n')
        for f in private.values():f.close()
        if len(rr)!=154 or {cell_key(r) for r in rr}!=expected:raise RuntimeError('native coverage error')
        rows[arm]={cell_key(r):r for r in rr};receipts[arm]=[str(leaf.relative_to(study)) for leaf in leaves]
        cmds=[
            [sys.executable,str(ROOT/'scripts/v31_score_longbench_official.py'),str(out/'longbench'),str(study/'private/manifests'),'/home/exouser/dyh/ljy_value_20261007/pools/longbench_v2_ofc','--cells',str(study/'private/cells_longbench64_seed1.json'),str(out/'private'/f'run_{arm}_longbench_v2_0shot_think.private.jsonl')],
            [sys.executable,str(ROOT/'scripts/v31_score_aime.py'),str(out/'aime'),str(study/'private/manifests'),str(ROOT/'results/v31_value_matrix_20261011/private/aime26/aime2026_answers.jsonl'),'--cells',str(study/'private/cells_aime26_30x3_seed42_43_44.json'),str(out/'private'/f'run_{arm}_aime26.private.jsonl')]]
        for cmd in cmds:subprocess.run(cmd,cwd=ROOT,check=True)
        scores[arm]={};score_diagnostics[arm]={}
        for ds,prefix in [('longbench_v2_0shot_think','longbench'),('aime26','aime')]:
            sc=json.loads((out/f'{prefix}.official.json').read_text())
            scores[arm][ds]=next(iter(sc.values()))
            planned={k for k in expected if k.startswith(ds+'|')}
            if set(scores[arm][ds])!=planned:raise RuntimeError('official score coverage mismatch')
            diag=next(iter(json.loads((out/f'{prefix}.summary.json').read_text())['arms'].values()))
            score_diagnostics[arm][ds]=diag[ds] if prefix=='longbench' else diag
    # Source/runtime identity and per-cell sampler bindings must match all controls.
    fairness=[]
    reference='v31_progress_dense_control'
    for arm in completion['arms']:
        for k,r in rows[arm].items():
            base=rows[reference][k]
            for field in official.COMPARE_FIELDS+('vllm','torch','gpu','seed_base','gpu_memory_utilization','fix_51994','fa4_local_fix'):
                if r.get(field)!=base.get(field):raise RuntimeError(f'fairness mismatch: {arm} {field}')
            a=r.get('receipts',{}).get('adapter',{})
            if r['cuda_graph_captures'] or a.get('native_local_skipped_tiles',0) or a.get('value_nonfinite_attention_calls',0) or a.get('order_errors',0) or a.get('stable_budget_errors',0):
                raise RuntimeError('native output/routing/accounting validation failure')
        for cfg in configs[arm]:
            for field in ('manifests_sha256','source_sha256','host_fingerprint','gpu_identity','model_source_inventory_sha256','runtime_settings','local_scope','model_revision'):
                if cfg[field]!=configs[reference][0][field]:raise RuntimeError(f'configuration fairness mismatch: {field}')
        fairness.append(dict(arm=arm,cells=len(rows[arm]),source_matched=True,native_sampler_bindings_matched=True,
            local_remained_dense=True,no_graph_captures=True,receipts=receipts[arm]))
    jsonwrite(resultdir/'fairness_audit.json',dict(status='complete',arms=fairness,scoring_identity_sha256=sha(study/'specs/scoring_identity.json')))
    reports=[]
    for arm in completion['arms']:
        for ds in ('longbench_v2_0shot_think','aime26'):
            rr={k:r for k,r in rows[arm].items() if r['dataset']==ds};vals=list(rr.values());sc=scores[arm][ds]
            n=np.array([r['denoise_forwards'] for r in vals]);nc=np.array([r['denoise_forwards']/max(1,r['canvases']) for r in vals]);t=np.array([r['output_tokens'] for r in vals])
            audits=[r['receipts']['adapter'] for r in vals]
            def total(field):return sum(a.get(field,0) or 0 for a in audits)
            reason=defaultdict(int)
            ages=defaultdict(int)
            for a in audits:
                for key,val in a.get('stable_refresh_reasons',{}).items(): reason[key]+=val
                for key,val in a.get('stable_age_histogram',{}).items():ages[key]+=val
            if arm in ('dense_full_fix51994','dense_piecewise'):
                reason['native_dense_global']=total('global_calls')
            elif arm=='allkept_fa4':
                reason['allkept_fa4_global']=total('global_calls')
            def max_audit(field):return max((a.get(field,0) or 0 for a in audits),default=0)
            units=total('stable_unit_updates')+total('stable_unit_noops')
            def unit_average(field):
                return sum((a.get(field) or 0)*(a.get('stable_unit_updates',0)+a.get('stable_unit_noops',0)) for a in audits)/units if units else None
            diag=score_diagnostics[arm][ds]
            stable_mode=(configs[arm][0]['stable_refresh_config'] or {}).get('selector',{}).get('mode')=='stable'
            report=dict(arm=arm,workload=ds,cells=len(vals),correct=sum(sc.values()),accuracy_pct=100*np.mean(list(sc.values())),
                N_total=int(n.sum()),N_mean=float(n.mean()),N_median=float(np.median(n)),N_p95=float(np.quantile(n,.95)),NC_mean=float(nc.mean()),
                NC_distribution=distribution(nc),output_tokens_distribution=distribution(t),
                output_tokens_mean=float(t.mean()),cap_hits=diag.get('capped',0),unfinished_responses=diag.get('no_final_response',0),
                unparsed_responses=diag.get('unparsed',0),finished_responses=diag.get('finished',0),
                global_kept_tiles=total('global_kept_tiles'),global_eligible_tiles=total('global_eligible_tiles'),
                global_calls=total('global_calls'),
                global_physical_sparsity=1-total('global_kept_tiles')/max(1,total('global_eligible_tiles')),
                local_calls=total('native_local_calls_audited'),local_skipped=total('native_local_skipped_tiles'),local_eligible_rectangles=total('native_local_eligible_rectangles'),
                overall_rectangle_denominator=total('overall_eligible_rectangles'),overall_skipped_rectangles=total('overall_skipped_rectangles'),
                dense_and_observation_reasons=dict(reason),swaps=total('stable_swaps'),unit_noops=total('stable_unit_noops'),
                guard_rejections=total('stable_removal_guard_rejected_units'),reentries=total('stable_reentries'),
                immediate_reversals=total('stable_immediate_reversals'),selector_audit_ms=total('stable_selector_audit_ms'),
                selector_audit_ms_per_observation=total('stable_selector_audit_ms')/max(1,reason['cheap_observations']),
                selector_state_bytes_max=max_audit('stable_state_bytes'),age_histogram=dict(ages),age_histogram_scope='at cache retirement/request end; not event-weighted',
                max_event_swaps=max_audit('stable_max_event_swaps'),max_cumulative_swaps=max_audit('stable_max_cumulative_swaps'),
                iou_previous=unit_average('stable_iou_previous'),iou_anchor=unit_average('stable_iou_anchor'),
                unit_updates=total('stable_unit_updates'),rejected_update_units=total('stable_rejected_update_units'),
                mean_swaps_per_observed_unit=total('stable_swaps')/max(1,units),
                mask_churn_per_observed_unit=total('stable_swaps')/max(1,128*units),
                peak_cuda_allocated_bytes_max=max(r.get('peak_cuda_allocated_bytes',0) for r in vals),
                persistence_rejections=total('stable_persistence_rejections') if stable_mode else None,
                cumulative_rejections=total('stable_cumulative_rejections') if stable_mode else None,
                stable_refresh_config=configs[arm][0]['stable_refresh_config'],
                graph_captures=sum(r['cuda_graph_captures'] for r in vals),paired={})
            for ref in ('v31_progress_dense_control','v31_no_progress_dense_hold'):
                base={k:r for k,r in rows[ref].items() if r['dataset']==ds}
                report['paired'][ref]=cluster_pair(rr,base,sc,scores[ref][ds])
            reports.append(report)
    jsonwrite(resultdir/'native_workload_results.json',reports)
    op_pointer=json.loads((study/'operator/real_replay_complete.json').read_text())
    operator=json.loads((study/op_pointer['leaf']/'summary.json').read_text())
    if sha(study/op_pointer['leaf']/'summary.json')!=op_pointer['summary_sha256']:raise RuntimeError('operator receipt identity mismatch')
    jsonwrite(resultdir/'operator_results.json',dict(coverage=operator['coverage'],snapshots=operator['snapshots'],
        max_retained_statistic_error=operator['max_retained_statistic_error'],selector_grid=operator['selector_grid'],
        parameters=operator['parameters'],metric_scope='pooled current-Q/K/V row errors; mask applies at next call; dense evaluation offline only'))
    op_records=[json.loads(line) for line in (study/op_pointer['leaf']/'records.jsonl').read_text().splitlines()]
    strata=[]
    for arm in operator['selector_grid']:
        for ds in ('longbench_v2_0shot_think','aime26'):
            for low,high in ((1,2),(3,5),(6,100000)):
                subset=[r for r in op_records if r['arm']==arm and r['dataset']==ds and low<=r['anchor_age']<=high]
                if subset:
                    strata.append(dict(arm=arm,workload=ds,anchor_age_reuse_calls=[low,high],frames=len(subset),
                        outside_logmass_mae=np.mean([r['outside_logmass_prediction_error'] for r in subset]),
                        true_retained_mass=np.mean([r['true_retained_mass_mean'] for r in subset]),
                        row_error_mean=np.mean([r['row_error_mean'] for r in subset]),
                        row_error_max=max(r['row_error_max'] for r in subset)))
    jsonwrite(resultdir/'operator_age_strata.json',strata)
    # Predeclared development recommendation; this is not a noninferiority claim.
    recommendations=[]
    for arm in completion['arms']:
        if arm in ('dense_full_fix51994','dense_piecewise','allkept_fa4','v31_progress_dense_control'):continue
        armrows=[r for r in reports if r['arm']==arm]
        criteria=[]
        for r in armrows:
            baseline=next(b for b in reports if b['arm']=='v31_progress_dense_control' and b['workload']==r['workload'])
            criteria.append(r['paired']['v31_progress_dense_control']['aggregate_N_ratio']<=1.02
                and r['NC_mean']<=1.02*baseline['NC_mean'] and r['N_p95']<=1.05*baseline['N_p95']
                and r['cap_hits']/r['cells']-baseline['cap_hits']/baseline['cells']<=.02)
        advantage=all(r['paired']['v31_progress_dense_control']['accuracy_delta_pp']>=0 for r in armrows) and any(r['paired']['v31_progress_dense_control']['accuracy_delta_pp']>0 for r in armrows)
        recommendations.append(dict(arm=arm,trajectory_gate_passed=all(criteria),development_accuracy_advantage=advantage,
            recommendation='gather larger audit evidence' if all(criteria) and advantage else 'retain as diagnostic; no expansion yet'))
    jsonwrite(resultdir/'recommendations.json',recommendations)
    lines=['# Stable cheap refresh development panel','', 'Previously examined development data; fresh controls, fixed native settings, no clean timing.','',
        '## Operator diagnostics','',
        '| Selector | True retained mass | Row error mean / p95 / max | Next-call change p95 / max | Swaps | Max cumulative |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('v31_progress_hold','raw_s2',operator['parameters']['selected_grid_arm'],'hold_no_progress','stable_probe4_s2'):
        x=operator['selector_grid'][name]
        lines.append(f"| {name} | {x['true_retained_mass_mean']:.6f} | {x['row_error_mean']:.4f} / {x['pooled_row_error_p95']:.4f} / {x['row_error_max']:.4f} | {x['row_output_change_p95']:.4f} / {x['row_output_change_max']:.4f} | {x['swaps']} | {x['max_cumulative']} |")
    lines+=['','## Native accuracy and denoising trajectory','',
        '| Arm | Workload | Correct / cells | Accuracy | Mean N | p95 N | Mean N/C | N ratio vs progress V31 | Paired wins/losses |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        pair=r['paired']['v31_progress_dense_control']
        lines.append(f"| {r['arm']} | {r['workload']} | {r['correct']}/{r['cells']} | {r['accuracy_pct']:.2f}% | {r['N_mean']:.2f} | {r['N_p95']:.1f} | {r['NC_mean']:.2f} | {pair['aggregate_N_ratio']:.4f} | {pair['wins']}/{pair['losses']} |")
    lines+=['','## Selector overhead and work','',
        'Audit measurements include observation and ranking instrumentation. QK is duplicated for selected tiles and mandatory tail; no selector PV or KV-head union is used. These measurements are not clean latency.', '',
        '| Arm | Workload | Observations / updates | QK rectangles | Probe rectangles | Audit ms/observation | Max final state MiB | GLOBAL sparsity |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        why=r['dense_and_observation_reasons']
        lines.append(f"| {r['arm']} | {r['workload']} | {why.get('cheap_observations',0)} / {why.get('actual_updates',0)} | {why.get('selector_qk_rectangles',0)} | {why.get('probe_qk_rectangles',0)} | {r['selector_audit_ms_per_observation']:.3f} | {r['selector_state_bytes_max']/2**20:.2f} | {100*r['global_physical_sparsity']:.2f}% |")
    lines += ['', 'Problem-clustered confidence intervals, receipt counters, and physical accounting are in native_workload_results.json. Lower N and higher sparsity are not end-to-end speed claims. The 503-cell expansion is a later gate.', '', '## Development recommendation', '']
    for r in recommendations:lines.append(f"- `{r['arm']}`: {r['recommendation']}; trajectory gate passed={r['trajectory_gate_passed']}.")
    (study/'summary.md').write_text('\n'.join(lines)+'\n')
    jsonwrite(study/'final_complete.json',dict(status='complete',arms=completion['arms'],native_cells=154*len(completion['arms']),
        source_identity_sha256=sha(study/'specs/execution_identity.json'),operator_complete=bool((study/'operator/real_replay_complete.json').exists()),
        scoring_identity_sha256=sha(study/'specs/scoring_identity.json'),fairness_audit_sha256=sha(resultdir/'fairness_audit.json'),
        official_scoring_complete=True,clean_timing='paused',data_status='previously examined development panel'))


if __name__=='__main__':main()
