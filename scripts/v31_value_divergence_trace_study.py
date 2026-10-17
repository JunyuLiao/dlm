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


def run(study,prior_seconds,backend_localization=False):
    if study.exists():raise RuntimeError('preserve attempt')
    if prior_seconds>=3000:raise RuntimeError('diagnostic GPU launch budget exhausted')
    if base.read(PARENT/'PAUSED_USER_REQUEST.json')['status']!='paused_user_request':raise RuntimeError('parent not paused')
    (study/'private').mkdir(parents=True);execution=study/'private/execution'
    shutil.copytree(PARENT/'private/execution',execution)
    for f in ('v31_value_divergence_trace_worker.py','v31_value_divergence_trace_study.py','v31_value_cross_worker.py'):
        shutil.copyfile(ROOT/'scripts'/f,execution/'scripts'/f)
    (study/'private/pools').symlink_to(PARENT/'private/pools',target_is_directory=True)
    cell=base.read(PARENT/'private/schedules/longbench_v2/cells.json')[0]
    if cell['index']!=0 or cell['seed']!=42:raise RuntimeError('predeclared trace case')
    arms=['current_v31_control','full_swap8','residual_direction_guard','current_v31_control','full_swap8','residual_direction_guard','current_v31_control']
    labels=['control_before','full_swap8','residual_direction_guard','control_after']
    if backend_localization:
        from scripts.v31_value_cross_study import entry
        cell=dict(cell,seed=1)
        c=entry('control','current_v31_control','warmup_control')
        ref=entry('reference','residual_direction_guard');dev=entry('device_exact','residual_direction_guard')
        warm_arms=[entry(b,m) for m in ('full_swap8','residual_direction_guard') for b in ('reference','device_exact')]
        arms=[c,*warm_arms,dict(c,label='control_before'),ref,dev,dict(c,label='control_after')]
        labels=[e['label'] for e in arms[5:]]
    warm_count=4 if backend_localization else 2
    first=warm_count+1
    cells=[dict(cell,seed=42)]*warm_count+[cell]*4
    base.write(study/'private/cells.json',cells,True);base.write(study/'private/arms.json',arms,True)
    parent=base.read(PARENT/'protocol.json')
    protocol=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parent_protocol_sha256=base.sha(PARENT/'protocol.json'),
        runtime=parent['runtime'],scope=f"one question0 seed{cell['seed']} four-arm trace; exact full-logit fingerprints first6 calls; attention/state/RNG up to512 calls for backend localization; no added forwards or random draws",
        backend_localization=backend_localization,trace_labels=labels,panel_seed=cell['seed'],
        arms=arms,measured_requests=4,explicit_warmups=warm_count,implicit_warmups=1,trace_first_request=first,prior_diagnostic_gpu_s=prior_seconds,
        execution_fingerprints={str(p.relative_to(execution)):base.sha(p) for p in execution.rglob('*') if p.is_file()},
        cells_sha256=base.sha(study/'private/cells.json'),suites=parent['suites'])
    base.write(study/'protocol.json',protocol,True)
    base.write(study/'receipts/source_fingerprints.json',dict(source_commit=protocol['source_commit'],execution_fingerprints=protocol['execution_fingerprints']),True)
    base.SUITES.update(parent['suites']);base.wait_idle()
    if base.runtime_identity()!=parent['runtime']:raise RuntimeError('runtime changed')
    env=base.env_for(study,'longbench_v2','current_v31_control')
    env.update(STUDY_ARM='bounded_first_divergence',PYTHONHASHSEED='0',FROZEN_SOURCE_SHA256=base.sha(study/'receipts/source_fingerprints.json'),
        VALUE_AUDIT='1',VALUE_CLEAN_TIMING='0',VALUE_BOUNDARY_TIMING='0',VALUE_TRACE_ROOT=str(study/'private/trace'),VALUE_TRACE_FIRST_REQUEST=str(first),VALUE_TRACE_CALL_LIMIT='512' if backend_localization else '6',
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
    if proc.returncode or len(rows)!=len(cells):errors.append('worker/coverage')
    if not errors:
        for r,arm,planned in zip(rows,arms[1:],cells):
            if not backend_localization:errors+=e2e.validate(r,planned,arm,'audit')
            else:
                a=r['receipts']['adapter']
                if (r['index']!=0 or r['panel_seed']!=planned['seed'] or r['cuda_graph_captures'] or a.get('cross_label')!=arm['label']
                    or a['native_local_skipped_tiles']!=0 or r['sampler_calls']!=r['denoise_forwards']+r['canvases']):errors.append('trace identity/accounting')
                if arm['backend']!='control' and (a['value_refinement_work']['budget_errors']!=0 or a['value_refinement_layers']!=[29]):errors.append('trace budget/scope')
        for index in range(first,first+4):
            count=len(list((study/f'private/trace/request{index:03d}').glob('call*.pt')))
            if (count<6 if backend_localization else count!=6):errors.append('trace coverage')
    receipt=dict(status='failed' if errors else 'complete',returncode=proc.returncode,errors=sorted(set(errors)),gpu_reservation_s=time.monotonic()-start,
                 protocol_sha256=base.sha(study/'protocol.json'),records=len(rows),control_bookend_hash_equal=rows[warm_count]['output_hash']==rows[-1]['output_hash'] if len(rows)==len(cells) else None)
    base.write(study/'receipt.json',receipt,True)
    if errors:raise RuntimeError('preserved trace failure '+str(errors))
    base.write(study/'final_complete.json',dict(receipt,files_sha256={str(p.relative_to(study)):base.sha(p) for p in
        [study/'records.jsonl',study/'private/generation.private.jsonl',*sorted((study/'private/trace').glob('request*/call*.pt'))]}),True)
    base.write(study/'state.json',dict(status='complete_diagnostic',full503='paused'))
    print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--prior-gpu-seconds',type=float,required=True);p.add_argument('--backend-localization',action='store_true')
    a=p.parse_args();run(a.study.resolve(),a.prior_gpu_seconds,a.backend_localization)
