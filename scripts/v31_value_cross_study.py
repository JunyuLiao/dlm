"""At most four diagnostic engines/96 measured requests. Never resumes full503."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_official as official
from scripts.v31_vllm_paired_bench import request_seed

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/v31_value_e2e_longbench503_20261017/native_attempt001'
MODES=['full_swap8','residual_direction_guard']


def entry(backend,mode,label=None):
    return dict(backend=backend,mode=mode,label=label or backend+'_'+mode)


def schedule(pool,engine):
    control=entry('control','current_v31_control','warmup_control')
    candidates=[entry(b,m) for m in MODES for b in ('reference','device_exact')]
    arms=[control];cells=[]
    warm=dict(pool[0],seed=42)
    for e in candidates:cells.append(warm);arms.append(e)
    for i in (0,49):
        for seed in (1,42):
            c=dict(pool[i],seed=seed);shift=(i+seed+engine)%4
            inner=candidates[shift:]+candidates[:shift]
            if engine%2:inner=inner[::-1]
            group=[entry('control','current_v31_control','control_before')]+inner+[entry('control','current_v31_control','control_after')]
            cells.extend([c]*6);arms.extend(group)
    assert len(cells)==28 and len(arms)==29
    return cells,arms


def prepare(study):
    if study.exists():raise RuntimeError('preserve attempt')
    if base.read(PARENT/'PAUSED_USER_REQUEST.json')['status']!='paused_user_request':raise RuntimeError('parent not paused')
    study.mkdir(parents=True);execution=study/'private/execution'
    shutil.copytree(PARENT/'private/execution',execution)
    for name in ('v31_value_cross_worker.py','v31_value_cross_study.py'):
        shutil.copyfile(ROOT/'scripts'/name,execution/'scripts'/name)
    # Read-only input link. The diagnostic owns its own caches and generations.
    (study/'private/pools').symlink_to(PARENT/'private/pools',target_is_directory=True)
    cells=base.read(PARENT/'private/schedules/longbench_v2/cells.json')
    plans=[]
    for engine,pass_name in enumerate(('audit','clean','clean','audit')):
        rr,arms=schedule(cells,engine)
        folder=study/f'engine{engine+1:03d}_{pass_name}/attempt001';(folder/'private').mkdir(parents=True)
        base.write(folder/'private/cells.json',rr,True);base.write(folder/'private/arms.json',arms,True)
        plans.append(dict(engine=engine+1,pass_name=pass_name,folder=str(folder.relative_to(study)),
                          cells_sha256=base.sha(folder/'private/cells.json'),arms_sha256=base.sha(folder/'private/arms.json')))
    parent=base.read(PARENT/'protocol.json')
    protocol=dict(status='frozen_bounded_diagnostic',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_protocol_sha256=base.sha(PARENT/'protocol.json'),plans=plans,suites=parent['suites'],runtime=parent['runtime'],
        execution_fingerprints={str(p.relative_to(execution)):base.sha(p) for p in execution.rglob('*') if p.is_file()},
        question_indices=[0,49],panel_seeds=[1,42],measured_requests=96,explicit_warmup_requests=16,implicit_warmup_requests=4,
        native=parent['frozen_native'],qualification='existing 256-unit and native same-state backend checks; dispatch/schedule CPU tests; each engine must pass structural receipts',
        budget='no new engine after cumulative diagnostic reservation reaches 3000 seconds; full503 never launched',
        inference='two predeclared questions, two repeats per pass; diagnostic only; all valid outcomes retained')
    base.write(study/'protocol.json',protocol,True)
    base.write(study/'receipts/source_fingerprints.json',dict(source_commit=protocol['source_commit'],execution_fingerprints=protocol['execution_fingerprints']),True)


def run(study,prior_seconds):
    protocol=base.read(study/'protocol.json');execution=study/'private/execution'
    for rel,digest in protocol['execution_fingerprints'].items():
        if base.sha(execution/rel)!=digest:raise RuntimeError('source drift')
    base.SUITES.update(protocol['suites']);reservation=prior_seconds
    pool=study/'private/pools/longbench_v2';pools=official.Pools(pool,fields=('thinking',))
    gold=base.read(pool/'longbench_v2_0shot_think_gold.json');final,_=official.load_final_response()
    for plan in protocol['plans']:
        if reservation>=3000:break
        if base.read(PARENT/'PAUSED_USER_REQUEST.json')['status']!='paused_user_request':raise RuntimeError('parent pause changed')
        leaf=study/plan['folder']
        if (leaf/'receipt.json').exists():raise RuntimeError('preserve previous engine attempt')
        for name in ('cells','arms'):
            if base.sha(leaf/f'private/{name}.json')!=plan[name+'_sha256']:raise RuntimeError('schedule drift')
        cells=base.read(leaf/'private/cells.json');arms=base.read(leaf/'private/arms.json')
        base.wait_idle()
        if base.runtime_identity()!=protocol['runtime']:raise RuntimeError('runtime drift')
        env=base.env_for(study,'longbench_v2','current_v31_control');clean=plan['pass_name']=='clean'
        env.update(STUDY_ARM='bounded_cross',PYTHONHASHSEED='0',FROZEN_SOURCE_SHA256=base.sha(study/'receipts/source_fingerprints.json'),
            VALUE_AUDIT=str(int(not clean)),VALUE_CLEAN_TIMING=str(int(clean)),VALUE_BOUNDARY_TIMING=str(int(clean)),
            VALUE_CROSS_SCHEDULE=str(leaf/'private/arms.json'),VALUE_CROSS_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
        cmd=[base.PYTHON,str(execution/'scripts/v31_value_cross_worker.py'),str(base.MODEL),str(pool),str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
        base.write(leaf/'config.json',dict(plan,protocol_sha256=base.sha(study/'protocol.json')),True)
        start=time.monotonic()
        with (leaf/'private/worker.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
            base.write(study/'state.json',dict(status='running',engine=plan['engine'],pass_name=plan['pass_name'],pid=proc.pid,cumulative_gpu_seconds=reservation))
            print(dict(event='launch',engine=plan['engine'],pass_name=plan['pass_name'],measured=24),flush=True)
            proc.wait()
        elapsed=time.monotonic()-start;reservation+=elapsed
        rows=base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
        private=base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
        errors=[];sanitized=[]
        if proc.returncode or len(rows)!=28 or len(private)!=28:errors.append('worker/coverage failure')
        if not errors:
            for ordinal,(r,t,c,e) in enumerate(zip(rows,private,cells,arms[1:])):
                a=(r.get('receipts') or {}).get('adapter') or {}
                if (r['index']!=c['index'] or r['panel_seed']!=c['seed'] or r['rng_seed']!=request_seed(31,c,0)
                    or r['cuda_graph_captures'] or r['value_clean_timing']!=clean
                    or a.get('cross_label')!=e['label'] or a.get('cross_request_index')!=ordinal+1
                    or r['sampler_calls']!=r['denoise_forwards']+r['canvases']):errors.append('identity/capture/count mismatch')
                if e['backend']!='control':
                    if a.get('value_refinement_layers')!=[29] or a.get('value_refinement_prefix_budget')!=128:errors.append('scope/budget')
                    if not clean and a.get('value_refinement_work',{}).get('budget_errors')!=0:errors.append('budget audit')
                if not clean and a.get('native_local_skipped_tiles')!=0:errors.append('LOCAL scope')
                meta,_,verified=official.bind(t,pools)
                if not verified:errors.append('unbound private record')
                score=int(official.lb_extract_answer(final(t['completion'],bool(meta['thinking'])))==gold[t['id']])
                if ordinal>=4:
                    sanitized.append(dict(index=r['index'],panel_seed=r['panel_seed'],engine=plan['engine'],pass_name=plan['pass_name'],
                        backend=e['backend'],mode=e['mode'],label=e['label'],score=score,
                        **{k:r[k] for k in ('denoise_forwards','canvases','output_tokens','finish_reason','output_hash','decode_s','wall_s','rng_seed')}))
        receipt=dict(status='failed' if errors else 'complete',returncode=proc.returncode,errors=sorted(set(errors)),gpu_reservation_s=elapsed,
                     protocol_sha256=base.sha(study/'protocol.json'),records=len(rows),measured_records=len(sanitized))
        base.write(leaf/'receipt.json',receipt,True)
        if errors:raise RuntimeError('preserved diagnostic failure: '+str(errors))
        base.write(leaf/'scored.json',sanitized,True)
        base.write(leaf/'attempt_complete.json',dict(receipt,files_sha256={n:base.sha(leaf/n) for n in ('records.jsonl','private/generation.private.jsonl','scored.json','config.json','receipt.json')}),True)
        print(dict(event='complete',engine=plan['engine'],gpu_seconds=elapsed,cumulative_gpu_seconds=reservation),flush=True)
    completed=list(study.glob('engine*/attempt001/attempt_complete.json'))
    base.write(study/'state.json',dict(status='complete' if len(completed)==4 else 'budget_limited_partial',engines=len(completed),cumulative_gpu_seconds=reservation,parent_full503='paused'))
    if len(completed)==4:base.write(study/'final_complete.json',dict(status='complete_diagnostic',protocol_sha256=base.sha(study/'protocol.json'),engines={str(p.relative_to(study)):base.sha(p) for p in completed},gpu_reservation_s=reservation-prior_seconds),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--study',type=Path,required=True);p.add_argument('--prior-gpu-seconds',type=float,default=0)
    a=p.parse_args();study=a.study.resolve()
    if a.action=='prepare':prepare(study)
    else:run(study,a.prior_gpu_seconds)
