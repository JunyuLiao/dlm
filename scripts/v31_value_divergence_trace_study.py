"""One optional four-arm first-divergence diagnostic; full503 stays paused."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_value_e2e_study as e2e

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/v31_value_e2e_longbench503_20261017/native_attempt001'


def run(study,prior_seconds):
    if study.exists():raise RuntimeError('preserve attempt')
    if prior_seconds>=3000:raise RuntimeError('diagnostic GPU launch budget exhausted')
    if base.read(PARENT/'PAUSED_USER_REQUEST.json')['status']!='paused_user_request':raise RuntimeError('parent not paused')
    (study/'private').mkdir(parents=True);execution=study/'private/execution'
    shutil.copytree(PARENT/'private/execution',execution)
    for f in ('v31_value_divergence_trace_worker.py','v31_value_divergence_trace_study.py'):
        shutil.copyfile(ROOT/'scripts'/f,execution/'scripts'/f)
    (study/'private/pools').symlink_to(PARENT/'private/pools',target_is_directory=True)
    cell=base.read(PARENT/'private/schedules/longbench_v2/cells.json')[0]
    if cell['index']!=0 or cell['seed']!=42:raise RuntimeError('predeclared trace case')
    arms=['current_v31_control','full_swap8','residual_direction_guard','current_v31_control','full_swap8','residual_direction_guard','current_v31_control']
    cells=[cell]*6
    base.write(study/'private/cells.json',cells,True);base.write(study/'private/arms.json',arms,True)
    parent=base.read(PARENT/'protocol.json')
    protocol=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parent_protocol_sha256=base.sha(PARENT/'protocol.json'),
        runtime=parent['runtime'],scope='one question0 seed42 four-arm trace; 6 early denoise calls; no native sampler changes, added forwards or random draws',
        arms=arms,measured_requests=4,explicit_warmups=2,implicit_warmups=1,prior_diagnostic_gpu_s=prior_seconds,
        execution_fingerprints={str(p.relative_to(execution)):base.sha(p) for p in execution.rglob('*') if p.is_file()},
        cells_sha256=base.sha(study/'private/cells.json'),suites=parent['suites'])
    base.write(study/'protocol.json',protocol,True)
    base.write(study/'receipts/source_fingerprints.json',dict(source_commit=protocol['source_commit'],execution_fingerprints=protocol['execution_fingerprints']),True)
    base.SUITES.update(parent['suites']);base.wait_idle()
    if base.runtime_identity()!=parent['runtime']:raise RuntimeError('runtime changed')
    env=base.env_for(study,'longbench_v2','current_v31_control')
    env.update(STUDY_ARM='bounded_first_divergence',PYTHONHASHSEED='0',FROZEN_SOURCE_SHA256=base.sha(study/'receipts/source_fingerprints.json'),
        VALUE_AUDIT='1',VALUE_CLEAN_TIMING='0',VALUE_BOUNDARY_TIMING='0',VALUE_TRACE_ROOT=str(study/'private/trace'),VALUE_TRACE_FIRST_REQUEST='3',
        NATIVE_INTERLEAVED_ARMS=str(study/'private/arms.json'),NATIVE_INTERLEAVED_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
    cmd=[base.PYTHON,str(execution/'scripts/v31_value_divergence_trace_worker.py'),str(base.MODEL),str(study/'private/pools/longbench_v2'),
         str(study/'private/cells.json'),str(study/'records.jsonl'),str(study/'private/generation.private.jsonl'),'mage','PIECEWISE']
    start=time.monotonic()
    with (study/'private/worker.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
        base.write(study/'state.json',dict(status='running',pid=proc.pid,full503='paused'))
        proc.wait()
    rows=base.jsonl(study/'records.jsonl') if (study/'records.jsonl').exists() else []
    errors=[]
    if proc.returncode or len(rows)!=6:errors.append('worker/coverage')
    if not errors:
        for r,arm in zip(rows,arms[1:]):errors+=e2e.validate(r,cell,arm,'audit')
        for index in range(3,7):
            if len(list((study/f'private/trace/request{index:03d}').glob('call*.pt')))!=6:errors.append('trace coverage')
    receipt=dict(status='failed' if errors else 'complete',returncode=proc.returncode,errors=sorted(set(errors)),gpu_reservation_s=time.monotonic()-start,
                 protocol_sha256=base.sha(study/'protocol.json'),records=len(rows),control_bookend_hash_equal=rows[2]['output_hash']==rows[5]['output_hash'] if len(rows)==6 else None)
    base.write(study/'receipt.json',receipt,True)
    if errors:raise RuntimeError('preserved trace failure '+str(errors))
    base.write(study/'final_complete.json',dict(receipt,files_sha256={str(p.relative_to(study)):base.sha(p) for p in
        [study/'records.jsonl',study/'private/generation.private.jsonl',*sorted((study/'private/trace').glob('request*/call*.pt'))]}),True)
    base.write(study/'state.json',dict(status='complete_diagnostic',full503='paused'))
    print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--prior-gpu-seconds',type=float,required=True)
    a=p.parse_args();run(a.study.resolve(),a.prior_gpu_seconds)
