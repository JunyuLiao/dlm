"""Sanitized, protocol-bound reporting for completed independent attempts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(attempt):
    p = Path(attempt)
    receipt = json.loads((p/'attempt_complete.json').read_text())
    config = json.loads((p/'config.json').read_text())
    records = [json.loads(x) for x in (p/'records.jsonl').read_text().splitlines()]
    if receipt['status'] != 'complete' or len(records) != config['cell_count']:
        raise ValueError('incomplete attempt')
    if any(r['cuda_graph_captures'] for r in records):
        raise ValueError('timed CUDA capture detected')
    return config, records


def aggregate(config, records):
    n,c,t,s,w = [sum(r[k] for r in records) for k in
        ('denoise_forwards','canvases','output_tokens','decode_s','wall_s')]
    a = [r['receipts']['adapter'] for r in records if r.get('receipts')]
    eligible = sum(r.get('global_eligible_tiles',0) for r in a)
    kept = sum(r.get('global_kept_tiles',0) for r in a)
    return dict(arm=config['arm'],purpose=config['purpose'],items=len(records),N=n,C=c,T=t,
        N_per_C=n/c,S_total_s=s,W_total_s=w,S_per_N_ms=1000*s/n,
        W_mean_s=w/len(records),S_mean_s=s/len(records),
        global_eligible_tiles=eligible or None,global_kept_tiles=kept if eligible else None,
        global_skipped_tiles=eligible-kept if eligible else None,
        global_sparsity=1-kept/eligible if eligible else None,
        local_sparsity=0,
        overall_rectangle_sparsity=(sum(r.get('overall_skipped_rectangles',0) for r in a)/sum(r.get('overall_eligible_rectangles',0) for r in a)) if sum(r.get('overall_eligible_rectangles',0) for r in a) else None,
        overall_denominator_status='audit-only actual decode calls; window-intersecting H x Q128 x KV64 rectangles, geometry-derived; not CUDA CTA count',
        native_local_attention_ms=sum(r.get('native_local_attention_ms',0) for r in a) or None,
        phases={phase:dict(calls=sum(r.get('value_phase_tiles',{}).get(phase,{}).get('calls',0) for r in a),
                          eligible_tiles=sum(r.get('value_phase_tiles',{}).get(phase,{}).get('eligible_tiles',0) for r in a),
                          kept_tiles=sum(r.get('value_phase_tiles',{}).get(phase,{}).get('kept_tiles',0) for r in a))
                for phase in ('initial','refresh','held','carried','dense')},
        selections=sum(r.get('mage_selections',0) for r in a),
        refreshes=sum(r.get('mage_reselections',0) for r in a),
        held=sum(r.get('mage_reused_calls',0) for r in a),
        carried=sum(r.get('mage_carried_calls',0) for r in a),
        invalid_rows=sum(r.get('value_invalid_rows',0) or 0 for r in a),
        discovery_ms=sum((r.get('value_audit_timings') or {}).get('discovery_ms',0) for r in a) or None,
        selection_ms=sum((r.get('value_audit_timings') or {}).get('selection_ms',0) for r in a) or None,
        global_attention_ms=sum((r['receipts'].get('timing') or {}).get('global_ms_total',0) for r in records if r.get('receipts')) or None,
        candidate_evaluations=sum(r.get('value_candidate_evaluations',0) for r in a) or None,
        peak_summary_bytes=max((r.get('value_peak_summary_bytes',0) for r in a),default=0) or None,
        peak_cuda_bytes=max(r['peak_cuda_allocated_bytes'] for r in records),
        threshold=config.get('threshold'),capped=sum(r['finish_reason']=='length' for r in records),
        timed_cuda_captures=0)


def paired(reference, other):
    key = lambda r:(r['dataset'],r['index'],r['panel_seed'],r['repeat'])
    r,o = {key(x):x for x in reference},{key(x):x for x in other}
    if r.keys()!=o.keys():
        raise ValueError('unmatched cells')
    bind = ('manifest_sha256','prompt_sha256','budget','rng_seed','max_model_len','chunk','block_size','gpu','torch','vllm')
    for k in r:
        if any(r[k][field]!=o[k][field] for field in bind):
            raise ValueError('mismatched binding or runtime')
    # Resample whole question clusters, including every seed/repeat within them.
    groups = sorted({k[:2] for k in r})
    sums = np.array([[sum(r[k]['wall_s'] for k in r if k[:2]==g),
                     sum(o[k]['wall_s'] for k in o if k[:2]==g),
                     sum(r[k]['decode_s'] for k in r if k[:2]==g),
                     sum(o[k]['decode_s'] for k in o if k[:2]==g)] for g in groups])
    rng = np.random.default_rng(1729)
    draws = sums[rng.integers(len(groups),size=(10000,len(groups)))].sum(1)
    ratio = sums.sum(0)
    return dict(question_clusters=len(groups),W_speedup=float(ratio[0]/ratio[1]),
        W_speedup_cluster_ci95=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist(),
        S_speedup=float(ratio[2]/ratio[3]),
        S_speedup_cluster_ci95=np.quantile(draws[:,2]/draws[:,3],[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',action='append',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise ValueError('new report output required')
    data={}
    for attempt in args.attempt:
        config,records=load(attempt)
        if config['arm'] in data:
            raise ValueError('one completed attempt per arm per report')
        data[config['arm']]=(config,records)
    summary={name:aggregate(*items) for name,items in data.items()}
    dense=data.get('dense_full_fix51994')
    if dense:
        for name,(config,records) in data.items():
            if config['purpose']=='clean' and dense[0]['purpose']=='clean':
                summary[name]['paired_dense']=paired(dense[1],records)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='completed attempts only; study completion separate',
        aggregates=summary),indent=2)+'\n')


if __name__=='__main__':
    main()
