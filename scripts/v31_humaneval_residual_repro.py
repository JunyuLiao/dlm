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


def prepare(root,mixed=False,all_cases=False,attention=False,crossover=False,cublas=False,torch_guard=False):
    cublas=cublas or torch_guard
    all_cases=all_cases or cublas
    attention=attention or crossover
    parent=campaign.verify(PARENT)
    if torch_guard:
        gate=ROOT/'results/humaneval_selector_definedness_20261019/attempt003'
        receipt=read(gate/'receipt.json');assert receipt['status']=='complete'
        for name,digest in receipt['files_sha256'].items():assert sha(gate/name)==digest
        flags=read(gate/'private/runtime_flags.json')
        assert flags['guard'] is True and flags['fill'] is False and flags['cublas_bytes']==33554432
    failure=read(PARENT/'receipts/early_residual_qualification_failure.json')
    assert failure['status']=='failed_method_qualification'
    assert read(PARENT/'receipts/controller_stopped_after_active_work.json')['gpu_reservations_closed']
    assert not root.exists()
    (root/'private').mkdir(parents=True)
    for name in ('v31_humaneval_residual_repro_worker.py','v31_humaneval_residual_repro.py'):
        shutil.copyfile(ROOT/'scripts'/name,root/'private'/name)
    trajectory=campaign.INVESTIGATION/'private/source_addenda/attempt031/worker.py'
    shutil.copyfile(trajectory,root/'private/trajectory_worker.py')
    if cublas:
        policy=(PARENT/'private/execution/scripts/v31_humaneval_cfgdet_policy_worker.py').read_text()
        assert policy.count('policy=policy,')==1
        extra=("blas_workspace=dict(environment=os.environ.get('CUBLAS_WORKSPACE_CONFIG'), "
               "cublas_bytes=torch._C._cuda_getCublasWorkspaceSize(), "
               "cublaslt_bytes=torch._C._cuda_getCublasLtWorkspaceSize(), "
               "preferred_library=str(torch.backends.cuda.preferred_blas_library())),\n            policy=policy,")
        policy=policy.replace('policy=policy,',extra,1)
        if torch_guard:
            marker='def install_compilation_policy():\n'
            assert policy.count(marker)==1
            policy=policy.replace(marker,marker+'    import torch\n    torch.use_deterministic_algorithms(True, warn_only=False)\n    torch.utils.deterministic.fill_uninitialized_memory = False\n',1)
            marker='torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),'
            assert policy.count(marker)==1
            policy=policy.replace(marker,marker+'\n            torch_fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,',1)
        (root/'private/cublas_policy.py').write_text(policy)
    if mixed or all_cases or attention:
        shutil.copyfile(ROOT/'scripts/v31_humaneval_gpu_fingerprint.py',root/'private/humaneval_gpu_fingerprint.py')
        gate=read(ROOT/'results/humaneval_mixed_repro_preflight_20261019/attempt001/receipt.json')
        assert gate['status']=='complete'
        tested=read(ROOT/'results/humaneval_mixed_repro_preflight_20261019/attempt001/sources.json')
        assert tested['scripts/v31_humaneval_gpu_fingerprint.py']==sha(root/'private/humaneval_gpu_fingerprint.py')
        path=root/'private/trajectory_worker.py';source=path.read_text()
        assert source.count('point = dict(canvas=')==1
        source=source.replace('import functools','from humaneval_gpu_fingerprint import fingerprint_cuda\nimport functools',1)
        source=source.replace('point = dict(canvas=','point = dict(logits_gpu128=fingerprint_cuda(args[0]),canvas=',1)
        path.write_text(source)
    if attention:
        shutil.copyfile(ROOT/'scripts/v31_humaneval_attention_fingerprint.py',root/'private/attention_fingerprint.py')
    spec=(ROOT/'docs/HUMANEVAL_TORCH_GUARD_PLAN_20261019.md' if torch_guard else
          ROOT/'docs/HUMANEVAL_BLAS_WORKSPACE_PLAN_20261019.md' if cublas else
          ROOT/'docs/HUMANEVAL_TRACE_CROSSOVER_PLAN_20261019.md' if crossover else
          ROOT/'docs/HUMANEVAL_ATTENTION_TRACE_PLAN_20261019.md' if attention else
          ROOT/'docs/HUMANEVAL_FULL_TRACE_PLAN_20261019.md' if all_cases else
          ROOT/'docs/HUMANEVAL_MIXED_REPRO_PLAN_20261019.md' if mixed else SPEC)
    shutil.copyfile(spec,root/'spec.md')
    jobs={}
    job_names=('crossover001',) if crossover else ('clean001','clean002') if mixed or all_cases or attention else ('audit','clean')
    for part in job_names:
        native_part='clean' if mixed or all_cases or attention else part
        original=PARENT/f'qualification/thinking_on/{native_part}/shared/qualification/attempt001'
        campaign.verify_leaf(PARENT,original)
        rows=read(original/'private/cells.json');arms=read(original/'private/arms.json')
        indices=[i for i,r in enumerate(rows) if r['index']==129 and r['seed']==43]
        assert len(indices)==6
        target=next(i for i in indices if arms[i+1]['arm']=='residual_direction_guard')
        prefix=max(indices)+1
        conditions=None;attention_modes={}
        if crossover:
            base_indices=[i for i,r in enumerate(rows) if r['index']==116 and r['seed']==42]
            assert len(base_indices)==6
            start,end=min(base_indices),max(base_indices)+1
            target=next(i for i in base_indices if arms[i+1]['arm']=='value_v3a_singleton_delete')
            offset=target-start;prefix=end
            conditions=('unobserved','sampler','post_only','pre_post','sampler','unobserved')
            selected=rows[:start]+[rows[i] for _ in conditions for i in base_indices]
            entries=arms[:start+1]+[arms[i+1] for _ in conditions for i in base_indices]
            live=['clean']*len(entries);bookends=[];condition_rows=[]
            for number,condition in enumerate(conditions):
                row=start+6*number+offset;condition_rows.append(dict(condition=condition,row=row))
                if condition!='unobserved':
                    live[row+1]='observe'
                    bookends.append(dict(original=start+offset,observed=row,final=start+30+offset,
                        arm='value_v3a_singleton_delete',label=f'cycle{number:03d}_{condition}'))
                if condition in ('post_only','pre_post'):attention_modes[str(row+1)]=condition
        elif attention:
            targets=[i for i,r in enumerate(rows) if (r['index'],r['seed'],arms[i+1]['label']) in (
                (116,42,'value_v3a_singleton_delete'),(116,43,'control_after'))]
            assert len(targets)==2
            prefix=len(rows);selected=rows+[rows[i] for i in targets]
            entries=arms+[arms[i+1] for i in targets];live=['clean']*len(entries)
            for i in targets:live[i+1]='observe'
            bookends=[dict(original=i,observed=i,final=prefix+j,arm=arms[i+1]['arm'],
                label=f"seed{rows[i]['seed']:02d}_q{rows[i]['index']:03d}_"+arms[i+1]['label']) for j,i in enumerate(targets)]
        elif all_cases:
            prefix=len(rows)
            selected=rows+rows[8:]
            entries=arms+arms[9:]
            live=['clean']+['observe']*len(rows)+['clean']*(len(rows)-8)
            bookends=[dict(original=i,observed=i,final=prefix+i-8,arm=arms[i+1]['arm'],
                label=f"seed{rows[i]['seed']:02d}_q{rows[i]['index']:03d}_"+arms[i+1]['label'])
                for i in range(8,len(rows))]
        elif mixed:
            selected=rows[:prefix]+[rows[i] for i in indices]*2
            entries=arms[:prefix+1]+[arms[i+1] for i in indices]*2
            live=['clean']*len(entries)
            for i in range(prefix+1,prefix+7):live[i]='observe'
            bookends=[dict(original=i,observed=prefix+j,final=prefix+6+j,
                           arm=arms[i+1]['arm'],label=arms[i+1]['label']) for j,i in enumerate(indices)]
        else:
            selected=rows[:prefix]+[rows[target]]*2
            entries=arms[:prefix+1]+[arms[target+1]]*2
            live=['clean']*len(entries);live[prefix+1]='observe'
            bookends=[dict(original=target,observed=prefix,final=prefix+1,arm='residual_direction_guard')]
        assert len(entries)==len(selected)+1 and len(live)==len(entries)
        if all_cases:assert len(selected)==152 and len(bookends)==72
        if attention and not crossover:assert len(selected)==82 and len(bookends)==2
        if crossover:assert len(selected)==74 and len(bookends)==4
        for group in bookends:
            positions=[group[key] for key in ('original','observed','final')]
            assert all(selected[i]==selected[positions[0]] for i in positions)
            assert all(entries[i+1]==entries[positions[0]+1] for i in positions)
            assert live[group['observed']+1]=='observe' and live[group['final']+1]=='clean'
        folder=root/'private'/part;folder.mkdir()
        write(folder/'cells.json',selected,True);write(folder/'arms.json',entries,True)
        write(folder/'live.json',live,True)
        if crossover:write(folder/'attention_modes.json',attention_modes,True)
        jobs[part]=dict(original=str(original.relative_to(ROOT)),
            original_marker_sha256=sha(original/'attempt_complete.json'),
            original_target_row=target,observed_row=prefix,final_bookend_row=prefix+1,
            requests=len(selected),warmup_requests=8,native_pass=native_part,bookend_groups=bookends)
        if crossover:jobs[part]['conditions']=condition_rows
    p=dict(schema='humaneval_residual_repro_v1',source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),failure_receipt_sha256=sha(PARENT/'receipts/early_residual_qualification_failure.json'),
        runtime=parent['runtime'],jobs=jobs,phase='k256',maximum_job_reservation_s=900 if crossover else 1000 if all_cases else 750,
        phase_cap_s=campaign.phase_caps()['k256'],maximum_new_reservation_s=900 if crossover else 2000 if all_cases else 1500,
        seed=None if all_cases or attention else 43,index=None if all_cases or attention else 129,
        seeds=[42] if crossover else [42,43] if all_cases or attention else [43],cohort=parent['cohort'] if all_cases else [116] if attention else [129],
        prefix_budget_tokens=256,full_accuracy_released=False,mixed_group=mixed or all_cases or attention,
        full_cohort_trace=all_cases,two_way_observer=all_cases or (attention and not crossover),attention_fingerprints=attention,
        trace_crossover=crossover,
        cublas_workspace_config=':4096:8' if cublas else None,
        torch_deterministic_guard=torch_guard,
        guard_preflight_receipt_sha256=sha(ROOT/'results/humaneval_selector_definedness_20261019/attempt003/receipt.json') if torch_guard else None,
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
    native_part=config.get('native_pass',part)
    if (leaf/'attempt_complete.json').exists():return read(leaf/'attempt_complete.json')
    if leaf.exists():raise RuntimeError('new immutable diagnostic attempt required')
    (leaf/'private').mkdir(parents=True)
    original=ROOT/config['original'];launch=read(original/'private/launch.json')
    rows=read(root/f'private/{part}/cells.json');arms=read(root/f'private/{part}/arms.json')
    env=campaign.base.env_for(PARENT,'thinking_on',campaign.CONTROL);env.update(launch['environment'])
    policy=root/'private/cublas_policy.py' if p.get('cublas_workspace_config') else PARENT/'private/execution/scripts/v31_humaneval_cfgdet_policy_worker.py'
    env.update(HUMANEVAL_REQUEST_ARMS=str(root/f'private/{part}/arms.json'),
        HUMANEVAL_REPRO_LIVE_SCHEDULE=str(root/f'private/{part}/live.json'),
        HUMANEVAL_POLICY_SOURCE=str(policy),HUMANEVAL_POLICY_SOURCE_SHA256=sha(policy),
        HUMANEVAL_KERNEL_INVENTORY=str(leaf/'private/kernel_inventory.json'),
        HUMANEVAL_TRAJECTORY_SOURCE=str(root/'private/trajectory_worker.py'),
        HUMANEVAL_TRAJECTORY_DIR=str(leaf/'private/trajectory'),
        HUMANEVAL_RESIDUAL_TRACE_DIR=str(leaf/'private/residual'),
        FROZEN_SOURCE_SHA256=sha(root/'protocol.json'))
    if p.get('cublas_workspace_config'):env['CUBLAS_WORKSPACE_CONFIG']=p['cublas_workspace_config']
    if p.get('attention_fingerprints'):
        env.update(HUMANEVAL_ATTENTION_TRACE_SOURCE=str(root/'private/attention_fingerprint.py'),
                   HUMANEVAL_ATTENTION_TRACE_DIR=str(leaf/'private/attention'))
        if p.get('trace_crossover'):env['HUMANEVAL_ATTENTION_MODES']=str(root/f'private/{part}/attention_modes.json')
    command=list(launch['command']);command[1]=str(root/'private/v31_humaneval_residual_repro_worker.py')
    command[4:7]=[str(root/f'private/{part}/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl')]
    write(leaf/'private/launch.json',dict(command=command,environment=env),True)
    campaign.base.wait_idle()
    budget=read(campaign.INVESTIGATION/'gpu_budget.json')
    assert all('elapsed_s' in r for r in budget['reservations'])
    spent=sum(r['elapsed_s'] for r in budget['reservations'] if r['phase']=='k256')
    reserve=min(p['maximum_job_reservation_s'],campaign.phase_caps()['k256']-spent-30)
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
                errors.extend(campaign.validate(public[i],rows[i],'shared',arms[i+1],native_part,
                    oracle=native_part=='audit',prefix_budget=256))
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
        errors.extend(campaign.validate(public[i],rows[i],'shared',arms[i+1],native_part,native_part=='audit',256))
    if proc.returncode or len(public)!=len(rows):errors.append('worker/coverage failure')
    errors.extend(campaign.policy_errors(leaf/'private/kernel_inventory.json','PIECEWISE'))
    if p.get('cublas_workspace_config') and (leaf/'private/kernel_inventory.json').exists():
        actual=read(leaf/'private/kernel_inventory.json').get('blas_workspace',{})
        if actual.get('environment')!=p['cublas_workspace_config'] or actual.get('cublas_bytes')!=33554432:
            errors.append('ineffective BLAS workspace configuration')
    if p.get('torch_deterministic_guard') and (leaf/'private/kernel_inventory.json').exists():
        actual=read(leaf/'private/kernel_inventory.json')
        if actual.get('torch_deterministic_algorithms') is not True or actual.get('torch_fill_uninitialized_memory') is not False:
            errors.append('ineffective framework guard or unexpected memory fill')
    bookends=[]
    if len(public)==len(rows):
        for group in config['bookend_groups']:
            a=public[group['original']]
            variants=([('within_engine_unobserved',group['final'])] if p.get('two_way_observer',p.get('full_cohort_trace')) else
                      [('observed',group['observed']),('final_unobserved',group['final'])])
            for name,index in variants:
                diff={key:a[key]!=public[index][key] for key in FIELDS}
                bookends.append(dict(arm=group['arm'],variant=name,differences=diff,passed=not any(diff.values())))
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
    parser.add_argument('--study',type=Path,default=DEFAULT);parser.add_argument('--mixed',action='store_true')
    parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--attention',action='store_true')
    parser.add_argument('--crossover',action='store_true')
    parser.add_argument('--cublas-workspace',action='store_true')
    parser.add_argument('--torch-guard',action='store_true')
    args=parser.parse_args();root=args.study.absolute()
    if args.action=='prepare':prepare(root,args.mixed,args.all_cases,args.attention,args.crossover,args.cublas_workspace,args.torch_guard)
    else:
        for part in read(root/'protocol.json')['jobs']:run_job(root,part)
