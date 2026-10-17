"""Paired full503 official quality, native work, and clean latency estimates."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_official as official

FIELDS=['score_points','N','C','T','S_seconds','W_seconds','prefill_seconds']
ROW_FIELDS=['denoise_forwards','canvases','output_tokens','decode_s','wall_s','prefill_s']
CONTROL='current_v31_control'


def bootstrap(a,b,seed=1729,draws=20000):
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,draws,500):
        ix=rng.integers(0,len(a),(min(500,draws-start),len(a)))
        x,y=a[ix],b[ix]
        delta=(y-x).mean(1)
        ratios=x[:,:,[4,5]].sum(1)/y[:,:,[4,5]].sum(1)
        geom=np.exp(np.log(x[:,:,[4,5]]/y[:,:,[4,5]]).mean(1))
        samples.append(np.concatenate((delta,ratios,geom),axis=1))
    values=np.concatenate(samples)
    point=np.concatenate(((b-a).mean(0),a[:,[4,5]].sum(0)/b[:,[4,5]].sum(0),
                          np.exp(np.log(a[:,[4,5]]/b[:,[4,5]]).mean(0))))
    return dict(metrics=FIELDS+['S_total_speedup','W_total_speedup','S_geomean_speedup','W_geomean_speedup'],
                estimate=point.tolist(),ci95=np.quantile(values,[.025,.975],axis=0).T.tolist(),draws=draws,seed=seed)


def arrays(result):
    keys=sorted(next(iter(result.values()))['official'])
    if len(keys)!=503:raise RuntimeError('require all503 questions')
    values={};rows={}
    for label,data in result.items():
        rr={base.key(r):r for r in data['records']}
        if set(rr)!=set(keys) or set(data['official'])!=set(keys) or len(rr)!=len(data['records']):raise RuntimeError('official/public coverage mismatch')
        rows[label]=rr
        values[label]=np.array([[100*int(data['official'][k])]+[rr[k][f] for f in ROW_FIELDS] for k in keys],dtype=float)
    values[CONTROL]=(values['control_before']+values['control_after'])/2
    return keys,values,rows


def summary(values,rr,pass_name):
    n=len(values);means=values.mean(0);totals=values.sum(0)
    result=dict(questions=n,score_points=float(means[0]),mean_N=float(means[1]),mean_C=float(means[2]),
        mean_T=float(means[3]),N_per_C=float(totals[1]/totals[2]),
        p50_N=float(np.median(values[:,1])),p95_N=float(np.quantile(values[:,1],.95)),
        p50_T=float(np.median(values[:,3])),p95_T=float(np.quantile(values[:,3],.95)),
        cap_rate=float(np.mean([official.finish_class(r['finish_reason'])=='capped' for r in rr])),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rr),
        actual_trials=len(rr),cuda_graph_captures=sum(r['cuda_graph_captures'] for r in rr))
    if pass_name=='clean':
        result.update(S_total_s=float(totals[4]),W_total_s=float(totals[5]),prefill_total_s=float(totals[6]),
            mean_S=float(means[4]),mean_W=float(means[5]),S_per_N=float(totals[4]/totals[1]),
            p50_W=float(np.median(values[:,5])),p95_W=float(np.quantile(values[:,5],.95)),
            total_policy='sum of503 per-question values; V31 averages the two actual control bookends')
    else:
        work=base.aggregate(rr)
        local=sum(r['receipts']['adapter']['native_local_eligible_rectangles'] for r in rr)
        if any(r['receipts']['adapter']['native_local_skipped_tiles']!=0 for r in rr):raise RuntimeError('LOCAL sparsity')
        extra=defaultdict(float);module=0
        for r in rr:
            a=r['receipts']['adapter']
            for name,value in a.get('value_refinement_work',{}).items():extra[name]+=value
            module+=a.get('value_anchor_module_cuda_elapsed_ms',0)
        work.update(local_eligible_tiles=local,local_kept_tiles=local,extra_refinement_work=dict(extra),
            value_anchor_module_cuda_elapsed_ms=module,
            mean_global_kept_tiles=work['global_kept_tiles']/len(rr),
            mean_extra_qk_rectangles=extra.get('anchor_full_qk_rectangles',0)/len(rr),
            mean_extra_pv_rectangles=extra.get('anchor_full_pv_rectangles',0)/len(rr),
            overall_denominator='decode GLOBAL full support plus LOCAL window-intersecting H x Q128 x KV64 rectangles',
            timing_interpretation='instrumented audit times are not clean latency')
        result['work']=work
    return result


def report(study,pass_name,result):
    keys,values,rows=arrays(result);summaries={}
    for arm,value in values.items():
        rr=list(rows['control_before'].values())+list(rows['control_after'].values()) if arm==CONTROL else list(rows[arm].values())
        summaries[arm]=summary(value,rr,pass_name)
    comparisons={}
    for reference in (CONTROL,'dense_full_fix51994'):
        for arm in ('full_swap8','residual_direction_guard',CONTROL):
            if arm==reference:continue
            r=bootstrap(values[reference],values[arm])
            if pass_name=='audit':
                # Audit wall time is deliberately excluded from performance inference.
                r['metrics']=r['metrics'][:4];r['estimate']=r['estimate'][:4];r['ci95']=r['ci95'][:4]
            comparisons[arm+'_vs_'+reference]=r
    # Frozen-index shard clusters are a sensitivity analysis, separate from the question interval.
    sensitivity={}
    for arm in ('full_swap8','residual_direction_guard'):
        grouped=[];control=[]
        for shard in range(16):
            ix=[i for i,k in enumerate(keys) if int(k.split('|')[1])%16==shard]
            grouped.append(values[arm][ix].sum(0));control.append(values[CONTROL][ix].sum(0))
        # Normalize group sums to the average shard size so score/N differences have per-question units.
        estimate=bootstrap(np.array(control)/(503/16),np.array(grouped)/(503/16))
        if pass_name=='audit':
            for k in ('metrics','estimate','ci95'):estimate[k]=estimate[k][:4]
        sensitivity[arm]=estimate
    noise=dict(bookend_hash_mismatches=sum(rows['control_before'][k]['output_hash']!=rows['control_after'][k]['output_hash'] for k in keys),
        score_disagreements=int(np.count_nonzero(values['control_before'][:,0]!=values['control_after'][:,0])),
        mean_after_minus_before=dict(zip(FIELDS,(values['control_after']-values['control_before']).mean(0).tolist())))
    if pass_name=='audit':noise['mean_after_minus_before']={k:v for k,v in noise['mean_after_minus_before'].items() if k in FIELDS[:4]}
    data=dict(status='complete',pass_name=pass_name,protocol_sha256=base.sha(study/'protocol.json'),summaries=summaries,
        paired_question_bootstrap=comparisons,engine_shard_sensitivity=sensitivity,control_noise=noise,
        inference='seed42 previously examined full cohort; no held-out or noninferiority claim; control averages both bookends')
    info=base.read(study/'private/pools/longbench_v2/longbench_v2_0shot_think_rowinfo.json')['rows']
    metadata={r['index']:r for r in info}
    data['input_policy']=dict(truncated_questions=sum(bool(r['truncated']) for r in info),
        prompt_token_range=[min(r['prompt_token_count'] for r in info),max(r['prompt_token_count'] for r in info)],
        output_budget=16384,additional_truncation=False)
    data['subgroups']={}
    for flag in (False,True):
        ix=[i for i,k in enumerate(keys) if bool(metadata[int(k.split('|')[1])]['truncated'])==flag]
        data['subgroups']['truncated' if flag else 'untruncated']=dict(questions=len(ix),
            means={arm:dict(zip(FIELDS if pass_name=='clean' else FIELDS[:4],v[ix].mean(0).tolist())) for arm,v in values.items()})
    data['trajectory_diagnostics']={}
    for arm in ('full_swap8','residual_direction_guard'):
        c,v=values[CONTROL],values[arm]
        saved=c[:,1]-v[:,1];largest=np.argsort(-saved)[:5]
        conditions={'all':np.ones(len(c),dtype=bool),'both_correct':(c[:,0]==100)&(v[:,0]==100),
                    'both_wrong':(c[:,0]==0)&(v[:,0]==0)}
        data['trajectory_diagnostics'][arm]=dict(fewer_calls=int((saved>0).sum()),more_calls=int((saved<0).sum()),
            total_calls_saved=float(saved.sum()),top5_calls_saved=float(saved[largest].sum()),
            top5_indices=[int(keys[i].split('|')[1]) for i in largest],
            conditioned={name:dict(questions=int(mask.sum()),mean_N_delta=float((v[mask,1]-c[mask,1]).mean()) if mask.any() else None)
                         for name,mask in conditions.items()},
            interpretation='descriptive outcome-conditioned subsets; no exclusions from primary comparison')
    path=study/'reports'/pass_name
    base.write(path/'summary.json',data,True)
    cells=[]
    for i,key in enumerate(keys):
        cells.append(dict(key=key,arms={arm:dict(zip(FIELDS,v[i].tolist())) for arm,v in values.items()},
            output_hashes={arm:rr[key]['output_hash'] for arm,rr in rows.items()}))
    base.write(path/'cells.json',cells,True)
    lines=[f'# LongBench-v2 full503, seed42: {pass_name}','',data['inference'],
        '', '| Method | Accuracy % | N/problem | N/C | Tokens/problem | S/problem (s) | W/problem (s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in [CONTROL,'full_swap8','residual_direction_guard','dense_full_fix51994','dense_piecewise','allkept_fa4']:
        r=summaries[arm];s=f"{r['mean_S']:.3f}" if pass_name=='clean' else 'audit only';w=f"{r['mean_W']:.3f}" if pass_name=='clean' else 'audit only'
        lines.append(f"| {arm} | {r['score_points']:.3f} | {r['mean_N']:.3f} | {r['N_per_C']:.3f} | {r['mean_T']:.1f} | {s} | {w} |")
    text='\n'.join(lines)+'\n';p=path/'summary.md'
    if p.exists() and p.read_text()!=text:raise RuntimeError('immutable report changed')
    if not p.exists():p.write_text(text)
    base.write(path/'final_complete.json',dict(status='complete',protocol_sha256=base.sha(study/'protocol.json'),
        files_sha256={name:base.sha(path/name) for name in ('summary.json','cells.json','summary.md')}),True)


def finish(study):
    a={r['key']:r for r in base.read(study/'reports/clean/cells.json')}
    b={r['key']:r for r in base.read(study/'reports/audit/cells.json')}
    if set(a)!=set(b) or len(a)!=503:raise RuntimeError('cross-pass coverage')
    result={}
    for arm in a[next(iter(a))]['output_hashes']:
        result[arm]=dict(hash_mismatches=sum(a[k]['output_hashes'][arm]!=b[k]['output_hashes'][arm] for k in a),
            differing_questions={metric:sum(a[k]['arms'][arm][metric]!=b[k]['arms'][arm][metric] for k in a)
                for metric in ('score_points','N','C','T')})
    base.write(study/'reports/cross_pass.json',dict(status='complete',arms=result,
        policy='clean quality and latency remain paired within the clean pass; audit statistics are separate'),True)
    base.write(study/'final_complete.json',dict(status='complete',protocol_sha256=base.sha(study/'protocol.json'),
        reports={name:base.sha(study/name) for name in ['reports/clean/final_complete.json','reports/audit/final_complete.json','reports/cross_pass.json']}),True)
