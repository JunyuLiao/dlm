"""Official scoring and paired workload-separated, problem-clustered analysis."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def cell_key(row): return f"{row['dataset']}|{row['index']}|{row['panel_seed']}|{row['repeat']}"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonwrite(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
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
    for path,digest in identity['source_sha256'].items():
        if sha(ROOT/path)!=digest: raise RuntimeError('source identity mismatch during final analysis')
    resultdir=study/'reports';resultdir.mkdir(exist_ok=True); rows={};scores={};receipts={}
    for arm in completion['arms']:
        leaves=[study/e['path'] for e in completion['shards'] if e['arm']==arm]
        rr=[];out=study/'scoring'/arm; (out/'private').mkdir(parents=True,exist_ok=True)
        private={ds:(out/'private'/f'run_{arm}_{ds}.private.jsonl').open('w') for ds in ('longbench_v2_0shot_think','aime26')}
        for leaf in leaves:
            receipt=json.loads((leaf/'receipt.json').read_text())
            if receipt['status']!='complete' or receipt['validation_errors']:raise RuntimeError('unclean native receipt')
            cfg=json.loads((leaf/'config.json').read_text())
            for path,digest in cfg['source_sha256'].items():
                if sha(ROOT/path)!=digest:raise RuntimeError('native leaf source mismatch')
            rr.extend(json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines())
            for line in (leaf/'private'/f'run_{arm}.private.jsonl').read_text().splitlines():
                r=json.loads(line);private[r['dataset']].write(line+'\n')
        for f in private.values():f.close()
        if len(rr)!=154 or len({cell_key(r) for r in rr})!=154:raise RuntimeError('native coverage error')
        rows[arm]={cell_key(r):r for r in rr};receipts[arm]=[str(leaf.relative_to(study)) for leaf in leaves]
        cmds=[
            [sys.executable,str(ROOT/'scripts/v31_score_longbench_official.py'),str(out/'longbench'),str(study/'private/manifests'),'/home/exouser/dyh/ljy_value_20261007/pools/longbench_v2_ofc','--cells',str(study/'private/cells_longbench64_seed1.json'),str(out/'private'/f'run_{arm}_longbench_v2_0shot_think.private.jsonl')],
            [sys.executable,str(ROOT/'scripts/v31_score_aime.py'),str(out/'aime'),str(study/'private/manifests'),str(ROOT/'results/v31_value_matrix_20261011/private/aime26/aime2026_answers.jsonl'),'--cells',str(study/'private/cells_aime26_30x3_seed42_43_44.json'),str(out/'private'/f'run_{arm}_aime26.private.jsonl')]]
        for cmd in cmds:subprocess.run(cmd,cwd=ROOT,check=True)
        scores[arm]={}
        for ds,prefix in [('longbench_v2_0shot_think','longbench'),('aime26','aime')]:
            sc=json.loads((out/f'{prefix}.official.json').read_text())
            scores[arm][ds]=next(iter(sc.values()))
    reports=[]
    for arm in completion['arms']:
        for ds in ('longbench_v2_0shot_think','aime26'):
            rr={k:r for k,r in rows[arm].items() if r['dataset']==ds};vals=list(rr.values());sc=scores[arm][ds]
            n=np.array([r['denoise_forwards'] for r in vals]);nc=np.array([r['denoise_forwards']/max(1,r['canvases']) for r in vals]);t=np.array([r['output_tokens'] for r in vals])
            audits=[r['receipts']['adapter'] for r in vals]
            def total(field):return sum(a.get(field,0) or 0 for a in audits)
            reason=defaultdict(int)
            for a in audits:
                for key,val in a.get('stable_refresh_reasons',{}).items(): reason[key]+=val
            report=dict(arm=arm,workload=ds,cells=len(vals),correct=sum(sc.values()),accuracy_pct=100*np.mean(list(sc.values())),
                N_total=int(n.sum()),N_mean=float(n.mean()),N_median=float(np.median(n)),N_p95=float(np.quantile(n,.95)),NC_mean=float(nc.mean()),
                output_tokens_mean=float(t.mean()),cap_hits=sum(r['finish_reason']=='length' for r in vals),
                global_kept_tiles=total('global_kept_tiles'),global_eligible_tiles=total('global_eligible_tiles'),
                global_physical_sparsity=1-total('global_kept_tiles')/max(1,total('global_eligible_tiles')),
                local_calls=total('native_local_calls_audited'),local_skipped=total('native_local_skipped_tiles'),
                overall_rectangle_denominator=total('overall_eligible_rectangles'),overall_skipped_rectangles=total('overall_skipped_rectangles'),
                dense_and_observation_reasons=dict(reason),swaps=total('stable_swaps'),unit_noops=total('stable_unit_noops'),
                guard_rejections=total('stable_removal_guard_rejected_units'),reentries=total('stable_reentries'),
                immediate_reversals=total('stable_immediate_reversals'),selector_audit_ms=total('stable_selector_audit_ms'),
                graph_captures=sum(r['cuda_graph_captures'] for r in vals),paired={})
            for ref in ('v31_progress_dense_control','v31_no_progress_dense_hold'):
                base={k:r for k,r in rows[ref].items() if r['dataset']==ds}
                report['paired'][ref]=cluster_pair(rr,base,sc,scores[ref][ds])
            reports.append(report)
    jsonwrite(resultdir/'native_workload_results.json',reports)
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
        '| Arm | Workload | Correct / cells | Accuracy | Mean N | p95 N | Mean N/C | N ratio vs progress V31 | Paired wins/losses |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        pair=r['paired']['v31_progress_dense_control']
        lines.append(f"| {r['arm']} | {r['workload']} | {r['correct']}/{r['cells']} | {r['accuracy_pct']:.2f}% | {r['N_mean']:.2f} | {r['N_p95']:.1f} | {r['NC_mean']:.2f} | {pair['aggregate_N_ratio']:.4f} | {pair['wins']}/{pair['losses']} |")
    lines += ['', 'Problem-clustered confidence intervals, receipt counters, and physical accounting are in native_workload_results.json. Lower N and higher sparsity are not end-to-end speed claims. The 503-cell expansion is a later gate.', '', '## Development recommendation', '']
    for r in recommendations:lines.append(f"- `{r['arm']}`: {r['recommendation']}; trajectory gate passed={r['trajectory_gate_passed']}.")
    (study/'summary.md').write_text('\n'.join(lines)+'\n')
    jsonwrite(study/'final_complete.json',dict(status='complete',arms=completion['arms'],native_cells=154*len(completion['arms']),
        source_identity_sha256=sha(study/'specs/execution_identity.json'),operator_complete=bool((study/'operator/real_replay_complete.json').exists()),
        official_scoring_complete=True,clean_timing='paused',data_status='previously examined development panel'))


if __name__=='__main__':main()
