"""Freeze and charge the bounded complete-contiguous-operand FA4 replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_humaneval_cfgdet_study as campaign

ROOT=campaign.ROOT
read,write,sha=campaign.read,campaign.write,campaign.sha
NATIVE=ROOT/'results/humaneval_mixed_repro_20261019/attempt013'
PARENT=ROOT/'results/humaneval_value_cfgdet_k256_20261019/attempt001'


def prepare(root):
    parent=campaign.verify(PARENT);assert not root.exists()
    for name in ('clean001','clean002'):
        leaf=NATIVE/'runs'/name/'attempt001';marker=read(leaf/'attempt_complete.json')
        assert marker['status']=='complete'
        for path,digest in marker['files_sha256'].items():assert sha(leaf/path)==digest
    gates=[]
    for directory,source,key,tests in (
        ('humaneval_operand_restore_preflight_20261019/attempt001','v31_humaneval_operand_restore.py','restore_source_sha256',4),
        ('humaneval_last_global_compare_preflight_20261019/attempt003','v31_humaneval_last_global_compare.py','analyzer_source_sha256',5)):
        gate=ROOT/'results'/directory;r=read(gate/'receipt.json');assert r['status']=='complete' and r['tests']==tests
        for path,digest in r['files_sha256'].items():assert sha(gate/path)==digest
        assert r[key]==sha(ROOT/'scripts'/source)
        gates.append(dict(receipt=str((gate/'receipt.json').relative_to(ROOT)),sha256=sha(gate/'receipt.json')))
    execution=root/'private/execution';(execution/'scripts').mkdir(parents=True);(execution/'scripts/__init__.py').write_text('')
    for name in ('v31_humaneval_operand_restore.py','v31_humaneval_last_global_compare.py','v31_humaneval_observe_operand_replay.py'):
        shutil.copyfile(ROOT/'scripts'/name,execution/'scripts'/name)
    shutil.copyfile(Path(__file__),root/'private/controller.py')
    shutil.copyfile(ROOT/'docs/HUMANEVAL_OBSERVE_OPERAND_REPLAY_PLAN_20261019.md',root/'spec.md')
    cases=[]
    for request,sample,label in ((23,100,'full_swap8_q129_seed42'),(24,154,'residual_q129_seed42')):
        folder=NATIVE/'runs/clean001/attempt001/private/last_global'
        path=folder/f'request{request:03d}.private.pt';trace=read(folder/f'request{request:03d}.json')
        assert path.exists()
        cases.append(dict(label=label,sampler_index=sample,raw_path=str(path),raw_sha256=sha(path)))
    p=dict(status='frozen_before_gpu',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime=parent['runtime'],cases=cases,repeats=16,phase='k256',maximum_job_reservation_s=240,
        phase_cap_s=campaign.phase_caps()['k256'],phase_allocation_file=str(campaign.allocation_file().relative_to(ROOT)),
        phase_allocation_sha256=sha(campaign.allocation_file()),native_model_forwards=0,native_sampler_calls=0,
        native_protocol_sha256=sha(NATIVE/'protocol.json'),native_base_worker=str(PARENT/'private/execution/scripts/v31_humaneval_cfgdet_base_worker.py'),
        native_base_worker_sha256=sha(PARENT/'private/execution/scripts/v31_humaneval_cfgdet_base_worker.py'),
        interface_sha256=trace['bindings']['observing_dense_fwd']['source_sha256'],preflight_receipts=gates,
        files_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    write(root/'protocol.json',p,True)


def run(root):
    campaign.verify(PARENT)  # Binds the frozen HumanEval suites before env_for.
    p=read(root/'protocol.json');assert not (root/'receipt.json').exists() and not (root/'failed_attempt.json').exists()
    for name,digest in p['files_sha256'].items():assert sha(root/name)==digest
    assert sha(Path(p['native_base_worker']))==p['native_base_worker_sha256']
    assert sha(NATIVE/'protocol.json')==p['native_protocol_sha256']
    assert sha(ROOT/p['phase_allocation_file'])==p['phase_allocation_sha256']
    assert campaign.base.runtime_identity()==p['runtime']
    launch=read(NATIVE/'runs/clean001/attempt001/private/launch.json')
    env=campaign.base.env_for(PARENT,'thinking_on',campaign.CONTROL);env.update(launch['environment'])
    e=root/'private/execution';parent=PARENT/'private/execution'
    env.update(PYTHONPATH=f'{e}:{parent}/src:{parent}')
    command=['/home/exouser/dyh/v31_env_20261005/bin/python',str(e/'scripts/v31_humaneval_observe_operand_replay.py'),str(root)]
    write(root/'private/launch.json',dict(command=command,environment={k:env[k] for k in ('PYTHONPATH','FA4_LOCAL_FIX_DIR')},
        cublas_workspace_config=env.get('CUBLAS_WORKSPACE_CONFIG')),True)
    # Finish fallible CPU launch preparation before opening a GPU reservation.
    campaign.base.wait_idle()
    budget,reservation=campaign.reserve(root,p,root)
    start=time.monotonic();failure=None;code=1
    try:
        with (root/'private/worker.log').open('w') as log:
            result=subprocess.run(command,cwd=e,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=reservation['reserved_s'])
            code=result.returncode
    except Exception as error:failure=repr(error)
    finally:
        reservation.update(elapsed_s=time.monotonic()-start,returncode=code)
        write(campaign.INVESTIGATION/'gpu_budget.json',budget)
    summary=root/'private/operator_summary.json';complete=code==0 and summary.exists()
    if complete:
        data=read(summary);complete=(data['native_model_forwards']==data['native_sampler_calls']==data['unexpected_cuda_captures']==0 and
            len(data['cases'])==len(p['cases']) and all(x['qkv_preserved'] and len(x['variants'])==7 for x in data['cases']))
    receipt=dict(status='complete_operator_diagnostic' if complete else 'failed',returncode=code,failure=failure,
        gpu_reservation_s=reservation['elapsed_s'],protocol_sha256=sha(root/'protocol.json'),
        files_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    write(root/('receipt.json' if complete else 'failed_attempt.json'),receipt,True)
    print({k:v for k,v in receipt.items() if k!='files_sha256'},flush=True)
    if not complete:raise RuntimeError('preserved failed observation replay')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['prepare','run']);parser.add_argument('study',type=Path)
    args=parser.parse_args();root=args.study.absolute()
    if args.action=='prepare':prepare(root)
    else:run(root)
