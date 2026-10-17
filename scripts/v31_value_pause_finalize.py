"""Finalize the already-running reference worker while its coordinator is stopped.

Never resumes the queue. Validates with the frozen coordinator's rules, records
the OS exit witness, preserves raw records, and then kills only the stopped
coordinator with a matching PID start identity.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time


def proc(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    s=p.read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=s[0],start_ticks=int(s[19]),exit_code=int(s[49]))


def run(study):
    sys.path[:0]=[str(study/'private/execution/src'),str(study/'private/execution')]
    from scripts import v31_value_e2e_study as frozen
    from scripts import v31_v2_local_threshold_sweep as base
    read,write,sha=base.read,base.write,base.sha
    pause=read(study/'private/pause_20261017.json')
    parent=proc(pause['coordinator_pid'])
    if not parent or parent['state']!='T' or parent['start_ticks']!=pause['coordinator_start_ticks']:
        raise RuntimeError('exact coordinator must remain stopped')
    if len(pause['children'])!=1:raise RuntimeError('one active worker required')
    worker=pause['children'][0]
    while True:
        status=proc(worker['pid'])
        if not status or status['start_ticks']!=worker['start_ticks']:raise RuntimeError('worker identity/exit witness lost')
        if status['state']=='Z':break
        time.sleep(5)
    witness=dict(status,observed_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    evidence=study/'pause_records/pause001';evidence.mkdir(parents=True,exist_ok=True)
    write(evidence/'request.json',read(study/'PAUSED_USER_REQUEST.json'),True)
    write(evidence/'worker_exit.json',witness,True)
    p=frozen.verify(study)
    leaf=study/pause['active_attempt'];config=read(leaf/'config.json')
    if config['family'] not in frozen.REFERENCES:raise RuntimeError('this finalizer is restricted to the observed reference worker')
    if (leaf/'attempt_complete.json').exists():raise RuntimeError('attempt already finalized')
    rows=base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    private=base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
    cells=read(leaf/'private/cells.json');errors=[]
    if witness['exit_code'] or len(rows)!=len(cells) or len(private)!=len(cells):errors.append('worker/coverage failure')
    if not errors:
        for row,cell in zip(rows,cells):errors+=frozen.validate(row,cell,config['family'],config['pass_name'])
    receipt=dict(status='failed' if errors else 'complete',returncode=witness['exit_code']//256,
        errors=sorted(set(errors)),protocol_sha256=sha(study/'protocol.json'),
        gpu_reservation_s=time.monotonic()-worker['start_ticks']/os.sysconf('SC_CLK_TCK'),records=len(rows),
        warmup_work={m:{} for m in frozen.MODES},control_drift=[],
        pause_finalization=dict(verifier_sha256=sha(__file__),frozen_coordinator_sha256=sha(frozen.__file__),
            worker_exit_sha256=sha(evidence/'worker_exit.json'),
            reservation_interpretation='upper bound through observed process exit; coordinator was stopped by user request'))
    write(leaf/'receipt.json',receipt,True)
    if not errors:
        exports={}
        for group in config['groups']:
            for i,label in zip(range(group['start'],group['end']),group['labels']):exports.setdefault(label,[]).append(i)
        files=['records.jsonl','private/generation.private.jsonl','receipt.json','config.json']
        for label,indices in exports.items():
            for rel,data in [(f'exports/{label}/records.jsonl',[dict(rows[i],study_arm=label) for i in indices]),
                (f'exports/{label}/private/run_{label}.private.jsonl',[private[i] for i in indices])]:
                dest=leaf/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('x') as f:
                    for row in data:f.write(json.dumps(row)+'\n')
                files.append(rel)
        receipt['files_sha256']={name:sha(leaf/name) for name in files}
        write(leaf/'attempt_complete.json',receipt,True)
    parent=proc(pause['coordinator_pid'])
    if not parent or parent['state']!='T' or parent['start_ticks']!=pause['coordinator_start_ticks']:
        raise RuntimeError('coordinator identity changed before termination')
    os.kill(parent['pid'],signal.SIGKILL)
    final=dict(status='paused_user_request',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        coordinator_pid=parent['pid'],coordinator_terminated_without_resuming_queue=True,
        active_worker_exit=witness,active_attempt=pause['active_attempt'],
        active_attempt_status=receipt['status'],completed_engine_jobs=len(list((study/'runs').glob('**/attempt_complete.json'))),
        protocol_sha256=sha(study/'protocol.json'),resume_requires_explicit_user_authorization=True,
        resume_wrapper='scripts/v31_value_e2e_resume.py',private_command_record_sha256=sha(study/'private/pause_20261017.json'))
    write(evidence/'final_pause_receipt.json',final,True)
    write(study/'PAUSED_USER_REQUEST.json',final)
    write(study/'state.json',dict(stage='paused_user_request',resumable=True,active_worker=None,
        last_attempt=pause['active_attempt'],completed_engine_jobs=final['completed_engine_jobs'],
        resume_requires_explicit_user_authorization=True,timing='full503 paused; targeted diagnostics separately authorized'))
    print(json.dumps(final,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    run(p.parse_args().study.resolve())
