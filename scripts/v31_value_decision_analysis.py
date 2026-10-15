"""Item-clustered analysis of the completed same-state decision batch.

The suffix frames are correlated observations within a native trajectory.
This report keeps workloads separate and resamples requests, never rows or
frames as if they were independent task-accuracy trials. It issues no native
promotion and has no GPU or timing work.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts import v31_value_decision_replay as io

FIELDS=('top1_disagreement','acceptance_disagreement','common_noise_disagreement',
        'kl_mean','kl_unsettled_mean','top1_disagreement_unsettled',
        'acceptance_disagreement_unsettled','stop_flip')


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        if not row['sparse_benefit_eligible']:
            continue
        for arm,values in row['metrics'].items():
            groups[arm,row['request']].append((row,values))
    items=[]
    for (arm,request),frames in sorted(groups.items()):
        result=dict(arm=arm,request=request,frames=len(frames),rows=sum(r['rows'] for r,_ in frames),
                    max_age=max(r['age'] for r,_ in frames))
        for field in FIELDS:
            weights=np.array([v.get('previously_unsettled_rows',0) if 'unsettled' in field else r['rows'] for r,v in frames])
            if not weights.sum():
                continue
            candidate=np.array([float(v[field]) for _,v in frames])
            anchor=np.array([float(r['metrics']['mass_anchor'][field]) for r,_ in frames])
            result[field]=float(np.average(candidate,weights=weights))
            result[field+'_minus_mass']=float(np.average(candidate-anchor,weights=weights))
            if field in ('top1_disagreement','acceptance_disagreement','common_noise_disagreement'):
                result[field+'_rows']=round(float((candidate*weights).sum()))
            if field=='stop_flip':
                result['stop_flip_frames']=sum(bool(v[field]) for _,v in frames)
        items.append(result)
    aggregates=[]
    for arm in sorted({r['arm'] for r in items}):
        same=[r for r in items if r['arm']==arm]
        result=dict(arm=arm,requests=len(same),frames=sum(r['frames'] for r in same),rows=sum(r['rows'] for r in same))
        for field in FIELDS:
            available=[r for r in same if field in r]
            if not available:
                continue
            delta=np.array([r[field+'_minus_mass'] for r in available])
            rng=np.random.default_rng(1729)
            boot=delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1)
            result[field]=dict(item_mean=float(np.mean([r[field] for r in available])),
                delta_item_mean=float(delta.mean()),ci95=list(map(float,np.quantile(boot,[.025,.975]))),
                better_items=int((delta < -1e-10).sum()),worse_items=int((delta > 1e-10).sum()),
                tied_items=int((abs(delta) <= 1e-10).sum()))
            count_field=field+'_rows'
            if count_field in available[0]:result[field]['row_count']=sum(r[count_field] for r in available)
        result['stop_flip_frames']=sum(r['stop_flip_frames'] for r in same)
        aggregates.append(result)
    return dict(items=items,aggregates=aggregates)


def run(root):
    dest=root/'decision_analysis'
    if dest.exists():
        marker=io.read(dest/'analysis_complete.json')
        if marker['summary_sha256']!=io.sha(dest/'summary.json'):
            raise RuntimeError('completed analysis changed')
        return
    done=io.read(root/'decision_batch_complete.json');protocol=io.read(root/'decision_batch_protocol.json')
    if done['protocol_sha256']!=io.sha(root/'decision_batch_protocol.json'):
        raise RuntimeError('decision batch identity changed')
    result=dict(status='complete',workloads={},input_sha256={},native_accuracy=False,timing='paused',
        scope='eager final-GLOBAL-layer same-state diagnostics on previously examined prompts',
        uncertainty='20,000 paired request-cluster bootstrap draws; exploratory small samples, not noninferiority',
        numerical_floor='same_mask_numerical_floor_vs_native uses native as its reference; do not interpret its minus_mass as a dense-reference treatment delta',
        promotion='requires separate evidence review and exact-source H100 tests')
    lines=['# Paired item-level decision analysis','',result['scope']+'.', '',
           'Only sparse frames enter this analysis. Repetitions do not increase sample size. ',
           'The numerical-floor row measures a different reference and is excluded from the treatment table.', '',
           '| Workload | Variant | Items | Frames | Top1 disagree rows | Acceptance disagree rows | KL delta, item mean | Top1 better / worse items |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for suite,spec in protocol['suites'].items():
        study=root/spec['study'];report=io.read(study/'report/summary.json')
        if io.sha(study/'report/summary.json')!=done['reports_sha256'][suite]:
            raise RuntimeError('completed report changed')
        rows=[]
        for rel,digest in report['input_sha256'].items():
            if io.sha(study/rel)!=digest:raise RuntimeError('decision records changed')
            result['input_sha256'][str((study/rel).relative_to(root))]=digest
            if rel.startswith('repeat01/'):
                rows.extend(json.loads(x) for x in (study/rel).read_text().splitlines())
        result['workloads'][suite]=summarize(rows)
        for r in result['workloads'][suite]['aggregates']:
            if r['arm']=='same_mask_numerical_floor_vs_native':continue
            t=r['top1_disagreement'];a=r['acceptance_disagreement'];k=r['kl_mean']
            lines.append(f"| {suite} | {r['arm']} | {r['requests']} | {r['frames']} | {t['row_count']} | {a['row_count']} | {k['delta_item_mean']:+.6g} | {t['better_items']} / {t['worse_items']} |")
    io.write(dest/'summary.json',result,True)
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    io.write(dest/'analysis_complete.json',dict(status='complete',summary_sha256=io.sha(dest/'summary.json'),
        batch_sha256=io.sha(root/'decision_batch_complete.json')),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=io.DEFAULT)
    run(p.parse_args().study.resolve())
