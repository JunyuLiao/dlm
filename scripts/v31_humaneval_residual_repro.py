"""Bounded residual-specific diagnosis; never releases the accuracy matrix."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_humaneval_cfgdet_study as campaign

ROOT=campaign.ROOT
PARENT=ROOT/'results/humaneval_value_cfgdet_k256_20261019/attempt001'
DEFAULT=ROOT/'results/humaneval_residual_repro_20261019/attempt001'
SPEC=ROOT/'docs/HUMANEVAL_RESIDUAL_REPRO_PLAN_20261019.md'
read,write,sha=campaign.read,campaign.write,campaign.sha
FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','native_interval_steps')


def prepare(root):
    parent=campaign.verify(PARENT)
    failure=read(PARENT/'receipts/early_residual_qualification_failure.json')
    assert failure['status']=='failed_method_qualification'
    assert read(PARENT/'receipts/controller_stopped_after_active_work.json')['gpu_reservations_closed']
    assert not root.exists()
    (root/'private').mkdir(parents=True)
    for name in ('v31_humaneval_residual_repro_worker.py','v31_humaneval_residual_repro.py'):
        shutil.copyfile(ROOT/'scripts'/name,root/'private'/name)
    trajectory=campaign.INVESTIGATION/'private/source_addenda/attempt031/worker.py'
    shutil.copyfile(trajectory,root/'private/trajectory_worker.py')
    shutil.copyfile(SPEC,root/'spec.md')
    jobs={}
    for part in ('audit','clean'):
        original=PARENT/f'qualification/thinking_on/{part}/shared/qualification/attempt001'
        campaign.verify_leaf(PARENT,original)
        rows=read(original/'private/cells.json');arms=read(original/'private/arms.json')
        indices=[i for i,r in enumerate(rows) if r['index']==129 and r['seed']==43]
        assert len(indices)==6
        target=next(i for i in indices if arms[i+1]['arm']=='residual_direction_guard')
        prefix=max(indices)+1
        selected=rows[:prefix]+[rows[target]]*2
        entries=arms[:prefix+1]+[arms[target+1]]*2
        live=['clean']*len(entries);live[prefix+1]='observe'
        folder=root/'private'/part;folder.mkdir()
        write(folder/'cells.json',selected,True);write(folder/'arms.json',entries,True)
        write(folder/'live.json',live,True)
        jobs[part]=dict(original=str(original.relative_to(ROOT)),
            original_marker_sha256=sha(original/'attempt_complete.json'),
            original_target_row=target,observed_row=prefix,final_bookend_row=prefix+1,
            requests=len(selected),warmup_requests=8)
    p=dict(schema='humaneval_residual_repro_v1',source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),failure_receipt_sha256=sha(PARENT/'receipts/early_residual_qualification_failure.json'),
        runtime=parent['runtime'],jobs=jobs,phase='k256',maximum_job_reservation_s=750,
        phase_cap_s=campaign.phase_caps()['k256'],maximum_new_reservation_s=1500,
        seed=43,index=129,prefix_budget_tokens=256,full_accuracy_released=False,
        files_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    write(root/'protocol.json',p,True)
    return p


def verify(root):
    p=read(root/'protocol.json');campaign.verify(PARENT)
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    assert campaign.base.runtime_identity()==p['runtime']
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    return p


def run_job(root,part):
    p=verify(root);config=p['jobs'][part];leaf=root/'runs'/part/'attempt001'
    if (leaf/'attempt_complete.json').exists():return read(leaf/'attempt_complete.json')
    if leaf.exists():raise RuntimeError('new immutable diagnostic attempt required')
    (leaf/'private').mkdir(parents=True)
    original=ROOT/config['original'];launch=read(original/'private/launch.json')
    rows=read(root/f'private/{part}/cells.json');arms=read(root/f'private/{part}/arms.json')
    env=campaign.base.env_for(PARENT,'thinking_on',campaign.CONTROL);env.update(launch['environment'])
    policy=PARENT/'private/execution/scripts/v31_humaneval_cfgdet_policy_worker.py'
    env.update(HUMANEVAL_REQUEST_ARMS=str(root/f'private/{part}/arms.json'),
        HUMANEVAL_REPRO_LIVE_SCHEDULE=str(root/f'private/{part}/live.json'),
        HUMANEVAL_POLICY_SOURCE=str(policy),HUMANEVAL_POLICY_SOURCE_SHA256=sha(policy),
        HUMANEVAL_KERNEL_INVENTORY=str(leaf/'private/kernel_inventory.json'),
        HUMANEVAL_TRAJECTORY_SOURCE=str(root/'private/trajectory_worker.py'),
        HUMANEVAL_TRAJECTORY_DIR=str(leaf/'private/trajectory'),
        HUMANEVAL_RESIDUAL_TRACE_DIR=str(leaf/'private/residual'),
        FROZEN_SOURCE_SHA256=sha(root/'protocol.json'))
    command=list(launch['command']);command[1]=str(root/'private/v31_humaneval_residual_repro_worker.py')
    command[4:7]=[str(root/f'private/{part}/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl')]
    write(leaf/'private/launch.json',dict(command=command,environment=env),True)
    campaign.base.wait_idle()
    budget=read(campaign.INVESTIGATION/'gpu_budget.json')
    assert all('elapsed_s' in r for r in budget['reservations'])
    spent=sum(r['elapsed_s'] for r in budget['reservations'] if r['phase']=='k256')
    reserve=min(750,campaign.phase_caps()['k256']-spent-30)
    if reserve<180:raise RuntimeError('existing phase budget exhausted')
    reservation=dict(name=str(leaf.relative_to(ROOT)),phase='k256',reserved_s=reserve,
                     protocol_sha256=sha(root/'protocol.json'))
    budget['reservations'].append(reservation);write(campaign.INVESTIGATION/'gpu_budget.json',budget)
    start=time.monotonic();errors=[];checked=0
    print(dict(event='residual_diagnostic_launch',part=part,requests=len(rows),reserved_s=reserve),flush=True)
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=PARENT/'private/execution',env=env,stdout=log,stderr=subprocess.STDOUT)
        write(leaf/'private/process.json',dict(pid=proc.pid),True)
        while proc.poll() is None:
            public=campaign.read_complete_jsonl(leaf/'records.jsonl')
            for i in range(checked,len(public)):
                errors.extend(campaign.validate(public[i],rows[i],'shared',arms[i+1],part,
                    oracle=part=='audit',prefix_budget=256))
            checked=len(public)
            if errors or time.monotonic()-start>=reserve:
                if not errors:errors.append('GPU reservation expired')
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            time.sleep(5)
    reservation.update(elapsed_s=time.monotonic()-start,returncode=proc.returncode)
    write(campaign.INVESTIGATION/'gpu_budget.json',budget)
    public=campaign.base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    for i in range(checked,len(public)):
        errors.extend(campaign.validate(public[i],rows[i],'shared',arms[i+1],part,part=='audit',256))
    if proc.returncode or len(public)!=len(rows):errors.append('worker/coverage failure')
    errors.extend(campaign.policy_errors(leaf/'private/kernel_inventory.json','PIECEWISE'))
    bookends=[]
    if len(public)==len(rows):
        a=public[config['original_target_row']]
        for name,index in [('observed',config['observed_row']),('final_unobserved',config['final_bookend_row'])]:
            diff={key:a[key]!=public[index][key] for key in FIELDS}
            bookends.append(dict(variant=name,differences=diff,passed=not any(diff.values())))
    result=dict(status='failed' if errors else 'complete',errors=sorted(set(errors)),records=len(public),
        observer_qualified=bool(bookends) and all(r['passed'] for r in bookends),bookends=bookends,
        gpu_reservation_s=reservation['elapsed_s'],protocol_sha256=sha(root/'protocol.json'),
        files_sha256={str(f.relative_to(leaf)):sha(f) for f in leaf.rglob('*') if f.is_file()})
    write(leaf/('failed_attempt.json' if errors else 'attempt_complete.json'),result,True)
    print(dict(event='residual_diagnostic_end',part=part,status=result['status'],
        observer_qualified=result['observer_qualified'],gpu_reservation_s=result['gpu_reservation_s']),flush=True)
    if errors:raise RuntimeError('preserved failed residual diagnostic: '+str(result['errors']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--study',type=Path,default=DEFAULT);args=parser.parse_args();root=args.study.absolute()
    if args.action=='prepare':prepare(root)
    else:
        for part in ('audit','clean'):run_job(root,part)
