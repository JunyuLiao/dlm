"""Official scoring of a complete workload while other native workloads run.

Uses the frozen official scorers, isolated scoring outputs and unchanged
generation records. It cannot launch generation or promote an arm to full503.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import numpy as np
from scripts import v31_value_decision_replay as io
from scripts import v31_value_native_study as native


def run(study,suite,attempt):
    protocol=native.verify(study)
    for rel in ('scripts/v31_value_native_study.py','scripts/v31_v2_local_threshold_sweep.py'):
        if io.sha(io.ROOT/rel)!=protocol['execution_fingerprints'][rel]:
            raise RuntimeError('use the matching native aggregation implementation')
    dest=study/f'workload_checkpoints/{suite}'/attempt
    if dest.exists():raise RuntimeError('preserve the existing checkpoint attempt; use a new attempt')
    cfg=protocol['suites'][suite];planned=io.read(study/f'private/schedules/{suite}/cells.json')
    arms=protocol['arms'];rows_by={};paths={};inputs={}
    for arm in arms:
        rows=[];leaves=[]
        for shard in range(1,cfg['shards']+1):
            markers=list((study/f'runs/{suite}/{arm}/shard{shard:03d}').glob('attempt*/attempt_complete.json'))
            if len(markers)!=1:raise RuntimeError('a complete unique attempt is required for every paired shard')
            marker=markers[0];leaf=marker.parent;proof=io.read(marker)
            if proof['status']!='complete' or proof['protocol_sha256']!=io.sha(study/'protocol.json'):
                raise RuntimeError('invalid generation completion')
            if proof['records_sha256']!=io.sha(leaf/'records.jsonl'):raise RuntimeError('completed records changed')
            for rel,digest in proof['private_sha256'].items():
                if io.sha(leaf/'private'/rel)!=digest:raise RuntimeError('completed private generation changed')
            inputs[str(marker.relative_to(study))]=io.sha(marker)
            rows+=native.base.jsonl(leaf/'records.jsonl');leaves.append(leaf)
        if native.validate(rows,suite,arm,planned):raise RuntimeError('native workload receipt validation failed')
        rows_by[arm]=rows;paths[arm]=leaves
    dest.mkdir(parents=True)
    execution=study/'private/execution';pool=study/f'private/pools/{suite}'
    scores={};bindings={};summaries=[];score_hashes={}
    for arm in arms:
        score_dir=dest/f'private/scoring/{arm}';score_dir.mkdir(parents=True)
        prefix=score_dir/'official'
        if suite=='aime26':
            command=[native.base.PYTHON,str(execution/'scripts/v31_score_aime.py'),str(prefix),str(pool),str(pool/'aime2026_answers.jsonl')]
        elif suite=='longbench_v2':
            command=[native.base.PYTHON,str(execution/'scripts/v31_score_longbench_official.py'),str(prefix),str(pool),str(pool)]
        elif suite=='math500':
            command=[native.base.PYTHON,str(execution/'scripts/v31_score_math500_bound.py'),str(prefix),str(pool)]
        else:
            scorer=io.read(study/'private/scorer_location.json')['ruler']
            command=[native.base.PYTHON,str(execution/'scripts/v31_score_ruler.py'),str(prefix),scorer,str(pool),str(pool)]
        command+=['--cells',str(study/f'private/schedules/{suite}/cells.json')]
        command+=[str(f) for leaf in paths[arm] for f in sorted((leaf/'private').glob('*.private.jsonl'))]
        env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
        with (score_dir/'scorer.log').open('w') as log:
            subprocess.run(command,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        official=io.read(str(prefix)+'.summary.json');coverage=official['coverage'][arm]
        if coverage['scored']!=cfg['count'] or coverage['missing'] or coverage['dropped']:
            raise RuntimeError('official coverage is incomplete')
        scores[arm]=io.read(str(prefix)+'.official.json')[arm]
        bindings.update(io.read(str(prefix)+'.binding.json'))
        score_hashes[arm]=dict(official_sha256=io.sha(str(prefix)+'.official.json'),binding_sha256=io.sha(str(prefix)+'.binding.json'))
        rows=rows_by[arm];work=native.base.aggregate(rows);extra=defaultdict(int)
        local=sum(r['receipts']['adapter']['native_local_eligible_rectangles'] for r in rows)
        work.update(local_eligible_tiles=local,local_kept_tiles=local,
            overall_kept_rectangles=work['overall_eligible_rectangles']-work['overall_skipped_rectangles'],
            overall_denominator='actual decode GLOBAL full-support and LOCAL window-intersecting H x Q128 x KV64 rectangles')
        for r in rows:
            for k,v in r['receipts']['adapter'].get('value_refinement_work',{}).items():extra[k]+=v
        summaries.append(dict(arm=arm,score_points=float(np.mean(list(scores[arm].values())))*(1 if suite=='ruler128k' else 100),
            work=work,extra_refinement_work=dict(extra),p95_N=float(np.quantile([r['denoise_forwards'] for r in rows],.95)),
            cap_rate=sum(native.official_rules.finish_class(r['finish_reason'])=='capped' for r in rows)/len(rows)))
    native.official_rules.check_comparable(bindings,arms,set(scores[arms[0]]))
    task_of={}
    if suite=='ruler128k':task_of={r['index']:r['task'] for r in io.read(pool/'ruler128k_v33ofc_subset_rowinfo.json')['rows']}
    comparisons=[]
    for arm in arms:
        if arm=='current_v31_control':continue
        comparisons.append(dict(arm=arm,**native.paired(suite,scores['current_v31_control'],scores[arm],task_of),
            trajectory=native.paired_trajectory(suite,rows_by['current_v31_control'],rows_by[arm],task_of)))
    result=dict(status='complete_workload_checkpoint',suite=suite,items_per_arm=cfg['count'],
        protocol_sha256=io.sha(study/'protocol.json'),analysis_source_sha256=io.sha(Path(__file__)),
        generation_source_commit=protocol['source_commit'],input_markers_sha256=inputs,official_scoring_sha256=score_hashes,
        summaries=summaries,comparisons=comparisons,timing='paused; wall fields are instrumented audit measurements',
        interpretation='complete paired workload only; remaining workloads and full503 promotion remain pending; no adaptive arm changes')
    io.write(dest/'summary.json',result,True)
    lines=[f'# Complete {suite} native checkpoint','',result['interpretation']+'.', '',
        '| Arm | Official score | N/problem-trial | N/canvas | Canvases/trial | Output tokens/trial | Cap rate |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        w=r['work'];lines.append(f"| {r['arm']} | {r['score_points']:.3f} | {w['mean_N']:.3f} | {w['N_per_C']:.3f} | {w['mean_C']:.3f} | {w['mean_T']:.1f} | {r['cap_rate']:.3f} |")
    lines+=['', 'Paired differences versus V31; 95% cluster-bootstrap intervals are exploratory.', '',
        '| Arm | Score difference (points) | 95% interval | Calls/trial difference | 95% interval |',
        '|---|---:|---|---:|---|']
    for r in comparisons:
        n=r['trajectory']['denoise_forwards']
        lines.append(f"| {r['arm']} | {r['delta_score_points']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {n['delta']:+.3f} | [{n['ci95'][0]:+.3f}, {n['ci95'][1]:+.3f}] |")
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    native.verify(study)
    io.write(dest/'checkpoint_complete.json',dict(status='complete',summary_sha256=io.sha(dest/'summary.json'),
        protocol_sha256=io.sha(study/'protocol.json')),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--suite',choices=['aime26','longbench_v2','math500','ruler128k'],required=True)
    p.add_argument('--attempt',required=True);a=p.parse_args()
    if not (a.attempt.startswith('attempt') and len(a.attempt)==10 and a.attempt[7:].isdigit()):p.error('use attempt001 style')
    run(a.study.resolve(),a.suite,a.attempt)
