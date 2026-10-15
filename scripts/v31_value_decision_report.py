"""Summarize complete same-state suffix diagnostics; never manufacture promotion."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from scripts import v31_value_decision_replay as io


def report(study):
    if (study/'report/report_complete.json').exists():
        done=io.read(study/'report/report_complete.json')
        assert done['summary_sha256']==io.sha(study/'report/summary.json')
        assert done['qualification_sha256']==io.sha(study/'qualification_complete.json')
        return
    marker=io.read(study/'qualification_complete.json')
    assert marker['protocol_sha256']==io.sha(study/'protocol.json')
    assert marker['reproducibility_sha256']==io.sha(study/'reproducibility.json')
    assert io.read(study/'reproducibility.json')['status']=='passed'
    protocol=io.read(study/'protocol.json');rows=[];inputs={}
    for repeat in range(1,protocol['repeats']+1):
        attempts=sorted((study/f'repeat{repeat:02d}/diagnostics_active').glob('attempt*/complete.json'))
        if not attempts:raise RuntimeError('missing completed active diagnostic')
        done=attempts[-1];m=io.read(done)
        assert m['protocol_sha256']==io.sha(study/'protocol.json')
        assert m['records_sha256']==io.sha(done.parent/'records.jsonl')
        p=done.parent/'decisions/decisions.jsonl'
        if m.get('decisions_sha256') is not None and m['decisions_sha256']!=io.sha(p):
            raise RuntimeError('completed decision records changed')
        inputs[str(p.relative_to(study))]=io.sha(p)
        r=[json.loads(x) for x in p.read_text().splitlines()]
        if not r:raise RuntimeError('no same-state decision measurements')
        rows.extend(dict(x,repeat=repeat) for x in r)
    # Repetitions qualify the measurement; do not count them as more prompts.
    first=[r for r in rows if r['repeat']==1]
    comparisons=defaultdict(list)
    fields=['top1_disagreement','acceptance_disagreement','common_noise_disagreement',
            'kl_mean','kl_max','entropy_mean','stop_flip','top1_disagreement_unsettled',
            'acceptance_disagreement_unsettled','kl_unsettled_mean']
    for r in first:
        if r.get('sparse_benefit_eligible') is False:continue
        if 'mass_anchor' not in r['metrics']:raise RuntimeError('missing matched mask reference')
        for arm,v in r['metrics'].items():
            comparisons[r['dataset'],arm,'all'].append((r,v))
            comparisons[r['dataset'],arm,str(r['age'])].append((r,v))
    metrics=[]
    for (dataset,arm,age),items in sorted(comparisons.items()):
        out=dict(dataset=dataset,arm=arm,age=age,frames=len(items),requests=len({x['request'] for x,_ in items}))
        for field in fields:
            eligible=[(r,v) for r,v in items if field in v]
            if not eligible:continue
            out[field]=sum(float(v[field]) for _,v in eligible)/len(eligible)
            if all(field in r['metrics']['mass_anchor'] for r,_ in eligible):
                delta=[float(v[field])-float(r['metrics']['mass_anchor'][field]) for r,v in eligible]
                out[field+'_minus_mass']=sum(delta)/len(delta)
                out[field+'_worse_frames']=sum(d>1e-8 for d in delta)
        metrics.append(out)
    out=study/'report';out.mkdir(exist_ok=True)
    result=dict(status='complete',scope=protocol['scope'],metrics=metrics,input_sha256=inputs,
        diagnostic_frames=len(first),native_accuracy=False,timing='paused',
        promotion='not issued by this report: interpret all workloads together before native development; no task-accuracy claim')
    io.write(out/'summary.json',result,True)
    lines=['# Same-state final-layer decision diagnostic','',result['promotion'], '',
        '| Workload | Arm | Frames | Top-1 disagreement | Acceptance disagreement | Mean KL |',
        '|---|---|---:|---:|---:|---:|']
    for r in metrics:
        if r['age']=='all':lines.append(f"| {r['dataset']} | {r['arm']} | {r['frames']} | {r['top1_disagreement']:.6f} | {r['acceptance_disagreement']:.6f} | {r['kl_mean']:.6f} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    io.write(out/'report_complete.json',dict(status='complete',summary_sha256=io.sha(out/'summary.json'),
        qualification_sha256=io.sha(study/'qualification_complete.json')),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    report(p.parse_args().study.resolve())
