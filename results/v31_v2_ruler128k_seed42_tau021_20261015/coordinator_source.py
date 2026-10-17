"""Frozen paired RULER128K seed-42 V31 versus exact online V2 tau=.021 audit."""
from __future__ import annotations
import argparse
import collections
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import numpy as np

NAME='v31_v2_ruler128k_seed42_tau021_20261015'
PARENT='v31_v2_tau021_longbench_20261015'
ARMS=['current_v31_control','v2_tau_0p021']
DATASET='ruler128k_v33ofc_subset'
SETTINGS=dict(dataset=DATASET,count=104,shards=4,max_len=136192,block=32,chunk=16384,memory=.85,mage_k=8192)


def read(path):return json.loads(Path(path).read_text())


def load(study):
    path=study/'private/execution/scripts/v31_v2_local_threshold_sweep.py'
    spec=importlib.util.spec_from_file_location('frozen_v2',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.SUITES['ruler128k']=SETTINGS.copy()
    return m


def prepare(study,ruler):
    if (study/'protocol.json').exists():return
    if study.exists():raise RuntimeError('Retain partial preparation; choose a new attempt directory')
    parent=study.parent/PARENT;m=load(parent);old=read(parent/'protocol.json');m.verify(parent,old)
    if m.runtime_identity()!=old['runtime']:raise RuntimeError('Pinned runtime changed')
    sys.path.insert(0,str(parent/'private/execution/scripts'));import v31_official as op
    scorer_identity=op.ruler_verify_checkout(ruler)
    pool=study.parent/'v31_value_performance_20261010/private/ruler128k_pool'
    historical=read(study.parent/'ruler128k_historical_report_20261015/manifest.json')
    manifest=pool/f'{DATASET}_generation_manifest.json'
    if m.sha(manifest)!=historical['manifest_sha256']:raise RuntimeError('Historical prompt identity changed')
    original=read(pool/f'cells_{DATASET}.json')
    if m.sha(pool/f'cells_{DATASET}.json')!=historical['cells_sha256']:raise RuntimeError('Original schedule changed')
    cells=[dict(c,seed=42) for c in original]
    assert len(cells)==104 and all(c['seed']==1 for c in original)
    assert [dict(c,seed=1) for c in cells]==original
    info=read(pool/f'{DATASET}_rowinfo.json')
    assert collections.Counter(r['task'] for r in info['rows'])=={t:8 for t in op.RULER_TASKS}
    study.mkdir();(study/'private').mkdir();(study/'receipts').mkdir()
    shutil.copytree(parent/'private/execution',study/'private/execution',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    dst=study/'private/pools/ruler128k';dst.parent.mkdir(parents=True);shutil.copytree(pool,dst)
    shutil.copyfile(parent/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
    schedule=study/'private/schedules/ruler128k';schedule.mkdir(parents=True)
    m.write(schedule/'cells.json',cells,True)
    for i in range(4):m.write(schedule/f'shard{i+1:03d}.json',cells[i::4],True)
    m.write(schedule/'smoke.json',cells[:2],True);m.check_inputs(dst,schedule/'cells.json')
    native=dict(old['frozen_native'],thinking=False,panel_seed=42)
    proto=dict(schema='ruler128k_seed42_v31_v2_v1',date='2026-10-15',
        source_commit=old['source_commit'],execution_fingerprints=old['execution_fingerprints'],
        input_fingerprints={str(f.relative_to(study)):m.sha(f) for folder in [dst,schedule] for f in folder.iterdir() if f.is_file()},
        model_revision=old['model_revision'],model_inventory_sha256=m.sha(study/'model_source_fingerprints.json'),
        runtime=old['runtime'],ruler=scorer_identity,suites={'ruler128k':SETTINGS},arms=ARMS,
        selector='value_v2_online_preserve_mass',threshold=.021,projection_rank=32,projection_seed=1729,
        frozen_native=native,source_campaign=PARENT,scope='GLOBAL only; LOCAL dense; unchanged native FA4 and sampler',
        seed_change='panel seed 1 -> 42; seed base 31 and subset generator seed 6464 fixed',
        timing='instrumented audit only; clean timing paused',uncertainty='100000 task-cluster bootstrap draws at seed1729; full-credit McNemar secondary')
    shutil.copyfile(Path(__file__),study/'coordinator_source.py')
    proto['coordinator_sha256']=m.sha(study/'coordinator_source.py')
    proto['coordinator_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    m.write(study/'protocol.json',proto,True)
    m.write(study/'receipts/source_fingerprints.json',dict(source_commit=proto['source_commit'],execution_fingerprints=proto['execution_fingerprints'],protocol_sha256=m.sha(study/'protocol.json')),True)
    m.write(study/'private/scorer_location.json',{'ruler':str(ruler.resolve())},True)
    m.write(study/'receipts/preflight.json',dict(status='passed',same_104_prompts=True,panel_seed=42,subset_generator_seed=6464,
        runtime_matched=True,ruler_source_verified=True,prior_native_qualification=read(parent.parent/'v31_v2_local_threshold_sweep_20261014_attempt002/qualification/tests_receipt.json')),True)
    m.verify(study,proto)
    (study/'README.md').write_text('# RULER128K seed42: V31 versus V2 tau=0.021\n\n'
        'The existing 104 prompts (8 x 13 tasks) are unchanged. Only native panel seed42 and V2 threshold0.021 are requested. '
        'Both arms execute the same frozen source snapshot, native sampler and FA4 consumer; GLOBAL-only, LOCAL dense. '
        'No historical generation is reused. Four interleaved26-item shards preserve the historical order, with alternating arm order. '
        'Other campaigns and clean timing remain paused. Resume the frozen coordinator with --stage run --study <this-directory>; '
        'complete leaf attempts are validated and skipped, while partial attempts receive a new immutable directory.\n')


def monitor(study,stop):
    while not stop.is_set():
        state=read(study/'state.json') if (study/'state.json').exists() else {}
        p=study/state.get('attempt','unused')/'records.jsonl'
        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True)
        event=dict(event='monitor',time_unix=time.time(),state=state,gpu=gpu.stdout.strip(),
            current_records=sum(1 for _ in p.open()) if p.exists() else 0,
            completed_shards=len(list((study/'runs').glob('*/*/shard*/attempt*/attempt_complete.json'))))
        with (study/'monitor.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True);stop.wait(1800)


def score(m,study,arm,attempts,cells,qual=False):
    dest=study/('qualification/scoring' if qual else 'scoring')/arm
    if (dest/'official.summary.json').exists():return read(dest/'official.summary.json')
    dest.mkdir(parents=True,exist_ok=True);execution=study/'private/execution';pool=study/'private/pools/ruler128k'
    ruler=read(study/'private/scorer_location.json')['ruler']
    files=[str(f) for a in attempts for f in (a/'private').glob('*.private.jsonl')]
    env=m.env_for(study,'ruler128k','value_v2_online_preserve_mass',.021 if arm.startswith('v2') else None)
    command=[m.PYTHON,str(execution/'scripts/v31_score_ruler.py'),str(dest/'official'),ruler,str(pool),str(pool),'--cells',str(cells),*files]
    with (dest/'scorer.log').open('w') as f:subprocess.run(command,cwd=execution,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    s=read(dest/'official.summary.json');c=s['coverage'][arm]
    assert c['scored']==len(read(cells)) and not c['missing'] and not c['dropped']
    return s


def paired(m,control,candidate,crows,vrows,task_of):
    assert control.keys()==candidate.keys();tasks=collections.defaultdict(list)
    for key in sorted(control):tasks[task_of[int(key.split('|')[1])]].append(candidate[key]-control[key])
    means=np.asarray([np.mean(v) for v in tasks.values()]);rng=np.random.default_rng(1729)
    boot=means[rng.integers(len(means),size=(100000,len(means)))].mean(1)
    a=sum(control[k]>=100-1e-9 and candidate[k]<100-1e-9 for k in control)
    b=sum(candidate[k]>=100-1e-9 and control[k]<100-1e-9 for k in control);n=a+b
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(a,b)+1))/2**n) if n else 1.
    c={m.key(r):r for r in crows};v={m.key(r):r for r in vrows}
    return dict(delta_score_points=float(means.mean()),task_clusters=len(tasks),bootstrap_draws=100000,
        task_cluster_ci95=list(map(float,np.quantile(boot,[.025,.975]))),
        v31_only_full_credit=a,v2_only_full_credit=b,full_credit_mcnemar_exact_p=p,
        output_hash_mismatches=sum(c[k]['output_hash']!=v[k]['output_hash'] for k in c),
        output_token_count_mismatches=sum(c[k]['output_tokens']!=v[k]['output_tokens'] for k in c),
        denoising_call_mismatches=sum(c[k]['denoise_forwards']!=v[k]['denoise_forwards'] for k in c),
        calls_fewer=sum(v[k]['denoise_forwards']<c[k]['denoise_forwards'] for k in c),
        calls_more=sum(v[k]['denoise_forwards']>c[k]['denoise_forwards'] for k in c),
        per_item_global_kept_delta=m.quantiles([v[k]['receipts']['adapter']['global_kept_tiles']-c[k]['receipts']['adapter']['global_kept_tiles'] for k in c]))


def finish(m,study,protocol,paths):
    sys.path.insert(0,str(study/'private/execution/scripts'));import v31_official as op
    pool=study/'private/pools/ruler128k';task_of={r['index']:r['task'] for r in read(pool/f'{DATASET}_rowinfo.json')['rows']}
    summaries={};rows_by={};scores={};bindings={};metric_rows=[];phase_rows=[];task_rows=[];cost=[]
    for arm in ARMS:
        rows=[r for a in paths[arm] for r in m.jsonl(a/'records.jsonl')];assert len(rows)==104
        s=score(m,study,arm,paths[arm],study/'private/schedules/ruler128k/cells.json')
        official=read(study/'scoring'/arm/'official.official.json')[arm]
        a=m.aggregate(rows);d=s['arms'][arm]
        result=dict(arm=arm,official_score=d['official']['_overall'],full_credit_items=sum(v>=100-1e-9 for v in official.values()),
            items=104,work=a,official_scorer=s,attempts=[str(p.relative_to(study)) for p in paths[arm]])
        dest=study/'reports'/arm;m.write(dest/'summary.json',result,True)
        m.write(dest/'final_complete.json',dict(status='complete',records=104,scored=104,protocol_sha256=m.sha(study/'protocol.json')),True)
        summaries[arm]=result;rows_by[arm]=rows;scores[arm]=official
        bindings.update(read(study/'scoring'/arm/'official.binding.json'))
        for task,value in d['official'][DATASET]['tasks'].items():
            task_rows.append(dict(arm=arm,task=task,items=8,official_score=value))
        metric_rows.append(dict(arm=arm,official_score=result['official_score'],full_credit_items=result['full_credit_items'],**{k:a[k] for k in ['N','C','T','mean_N','median_N','mean_C','N_per_C','mean_T','global_eligible_tiles','global_kept_tiles','global_skipped_tiles','global_sparsity','overall_eligible_rectangles','overall_skipped_rectangles','overall_sparsity','audit_wall_s','audit_decode_s']},
            overall_kept_rectangles=a['overall_eligible_rectangles']-a['overall_skipped_rectangles'],audit_S_per_N=a['audit_decode_s']/a['N'],capped=d.get('capped',0),nulls=d.get('nulls',0)))
        for phase,v in a['phases'].items():phase_rows.append(dict(arm=arm,phase=phase,**v))
        cost.append(dict(arm=arm,physical_fa4_global_kept_tiles=a['global_kept_tiles'],**{k:a[k] for k in ['selector_discovery_qk_rectangles','selector_candidate_evaluations','selector_discovery_ms','selector_selection_ms','N','audit_wall_s']},
            adapter_split_fa4_calls=sum(r['receipts']['adapter'].get('split_fa4_calls',0) for r in rows),
            worker_gpu_reservation_s=sum(read(p/'receipt.json')['worker_reservation_seconds'] for p in paths[arm]),timing='instrumented audit; not clean latency'))
    assert op.check_comparable(bindings,ARMS,set(scores[ARMS[0]]))==104
    comparison=paired(m,scores[ARMS[0]],scores[ARMS[1]],rows_by[ARMS[0]],rows_by[ARMS[1]],task_of)
    m.write(study/'paired_comparison.json',comparison,True);m.write(study/'metrics.json',metric_rows,True);m.write(study/'cost_separation.json',cost,True)
    for name,data in [('metrics',metric_rows),('phase_accounting',phase_rows),('task_scores',task_rows)]:
        with (study/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    lines=['# RULER128K seed42: V31 versus V2 tau=0.021','',
        'Same104 prompts, 8 per13 tasks. Seed42 is the native panel seed; base31 and Gaussian32 projection seed1729 stay fixed. Official RULER gives partial credit.',
        '', '| Method | Official score /100 | Full-credit items | N/problem | Median N | C/problem | N/C | Tokens/problem | GLOBAL kept tiles | GLOBAL sparsity | Overall sparsity |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metric_rows:lines.append(f"| {r['arm']} | {r['official_score']:.3f} | {r['full_credit_items']}/104 | {r['mean_N']:.3f} | {r['median_N']:.1f} | {r['mean_C']:.2f} | {r['N_per_C']:.3f} | {r['mean_T']:.2f} | {r['global_kept_tiles']:,} | {r['global_sparsity']:.2%} | {r['overall_sparsity']:.2%} |")
    c,v=metric_rows;low,high=comparison['task_cluster_ci95']
    lines+=['',f"V2 minus V31: {comparison['delta_score_points']:+.3f} score points; 13-task-cluster bootstrap95% interval [{low:+.3f}, {high:+.3f}]. Full-credit binary McNemar p={comparison['full_credit_mcnemar_exact_p']:.4f}.",
        f"Native call ratio {v['N']/c['N']:.5f}; absolute retained GLOBAL work ratio {v['global_kept_tiles']/c['global_kept_tiles']:.5f}.",
        '', '| Task (8 items each) | V31 score | V2 score | Difference |', '|---|---:|---:|---:|']
    task_scores={arm:summaries[arm]['official_scorer']['arms'][arm]['official'][DATASET]['tasks'] for arm in ARMS}
    for task in sorted(task_scores[ARMS[0]]):
        cs=task_scores[ARMS[0]][task];vs=task_scores[ARMS[1]][task]
        lines.append(f'| {task} | {cs:.3f} | {vs:.3f} | {vs-cs:+.3f} |')
    lines += [
        '', 'Both arms share identical source/protocol/runtime/model/input pins and use fresh generation. LOCAL stays dense; native FA4 executes attention; projected V is selector-only.',
        'GLOBAL/overall work counts audited denoising attention tiles, not literal CUDA kernel launches. Additional selector discovery/selection and instrumented wall time are separate; clean timing stays paused.',
        'These previously examined104 prompts are not a fresh held-out confirmation. Other campaigns remain paused.','']
    (study/'summary.md').write_text('\n'.join(lines))
    m.verify(study,protocol)
    for arm in ARMS:
        for a in paths[arm]:
            rr=m.jsonl(a/'records.jsonl');planned=read(study/'private/schedules/ruler128k'/(a.parent.name+'.json'))
            assert not m.validate_records(rr,'ruler128k',arm,.021 if arm.startswith('v2') else None,planned)
    m.write(study/'receipts/fairness_audit.json',dict(status='passed',same_source_protocol_model_runtime_seed=True,
        source_files_verified=len(protocol['execution_fingerprints']),official_cells_per_arm=104,paired_binding_cells=104,
        invalid_rows=0,missing_scorer_cells=0,unexpected_request_cuda_captures=0,LOCAL_skipped_tiles=0,
        selector='value_v2_online_preserve_mass',threshold=.021,panel_seed=42,other_campaigns_resumed=False,clean_timing=False),True)
    m.write(study/'state.json',dict(status='complete',updated_unix=time.time()))
    m.write(study/'final_complete.json',dict(status='complete',arms=ARMS,records_per_arm=104,source_commit=protocol['source_commit'],
        protocol_sha256=m.sha(study/'protocol.json'),panel_seed=42,threshold=.021,timing='paused'),True)
    print(json.dumps(dict(event='complete',metrics=metric_rows,paired=comparison)),flush=True)


def run(study):
    m=load(study);protocol=read(study/'protocol.json')
    if m.sha(Path(__file__))!=protocol['coordinator_sha256']:raise RuntimeError('Coordinator source drift')
    if (study/'final_complete.json').exists():return
    m.verify(study,protocol)
    if m.runtime_identity()!=protocol['runtime']:raise RuntimeError('Runtime drift')
    stop=threading.Event();thread=threading.Thread(target=monitor,args=(study,stop),daemon=True);thread.start()
    try:
        smoke=study/'private/schedules/ruler128k/smoke.json'
        for arm in ARMS:
            a=m.run_shard(study,protocol,'ruler128k',arm,.021 if arm.startswith('v2') else None,smoke,True)
            score(m,study,arm,[a],smoke,True)
        m.write(study/'qualification/final_complete.json',dict(status='complete',arms=ARMS,cells_per_arm=2,scored_per_arm=2),True)
        paths={arm:[] for arm in ARMS}
        for index,shard in enumerate(sorted((study/'private/schedules/ruler128k').glob('shard*.json'))):
            for arm in ARMS if index%2==0 else ARMS[::-1]:
                paths[arm].append(m.run_shard(study,protocol,'ruler128k',arm,.021 if arm.startswith('v2') else None,shard))
        finish(m,study,protocol,paths)
    finally:stop.set();thread.join(timeout=5)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','run'],default='run')
    ap.add_argument('--study',type=Path,default=Path.cwd()/'results'/NAME)
    ap.add_argument('--ruler',type=Path,default=Path('/home/exouser/dyh/RULER_c3f5e3b_clean'))
    args=ap.parse_args();study=args.study.resolve()
    if args.stage=='prepare':prepare(study,args.ruler)
    else:run(study)


if __name__=='__main__':main()
