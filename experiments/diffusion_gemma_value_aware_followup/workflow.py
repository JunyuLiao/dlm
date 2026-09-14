"""Queue follow-up stages behind the live worker without competing for H100.

Reuses the existing exclusive-lock/900-second stage supervisor. The dependency
is an actual process identity (PID plus start time), not a stale status file.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from . import jobs
from .protocol import ROOT,sha

STAGES=('development','secondary','final')


def process_identity(pid):
    try:
        directory=Path('/proc')/str(pid)
        stat=(directory/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
        return dict(pid=int(pid),start_ticks=parts[19],state=parts[0],
            command=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    except FileNotFoundError:
        return None


def same_live_process(expected,current):
    return bool(expected and current and current['state'] not in ('Z','X')
        and current['pid']==expected['pid'] and current['start_ticks']==expected['start_ticks'])


def stage_command(stage,root,extra):
    if stage not in STAGES:
        raise ValueError('unsupported queued stage')
    command=[sys.executable,'-u','-m',__package__+'.'+stage]
    if stage in ('development','final'):
        command.append('run')
    return [*command,'--output',str(root),*extra]


def supervise(root,state_path):
    state=json.loads(state_path.read_text())
    if state['source_sha256']!=sha(Path(__file__).read_bytes()):
        raise ValueError('queued workflow source changed')
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state['pid']=os.getpid();state['status']='waiting';_write(state_path,state)
        expected=state['after_identity']
        while expected:
            current=process_identity(expected['pid'])
            if not same_live_process(expected,current):
                break
            _append(root/'queue_monitor.jsonl',dict(time=time.time(),queue_pid=os.getpid(),
                after=current,status='verified_live_dependency'))
            time.sleep(900)
        failures=[]
        for stage in state['stages']:
            state.update(status='running',stage=stage,stage_started=time.time());_write(state_path,state)
            extra=['--phase',state['final_phase']] if stage=='final' else []
            try:
                # Same proven monitor/lock code; only the module command changes.
                with patch.object(jobs,'worker_args',stage_command):
                    jobs.supervise(root,stage,extra)
            except (Exception,SystemExit):
                error=dict(stage='queued_stage',queued_stage=stage,traceback=traceback.format_exc())
                failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
                # An individual stage failure must not suppress independent
                # secondary checks or discard completed development outputs.
        state.update(status='complete' if not failures else 'finished_with_failures',
            failures=failures,finished=time.time());_write(state_path,state)
        if failures:
            raise SystemExit(1)


def launch(root,after_pid,stages,final_phase):
    if not stages or len(stages)!=len(set(stages)) or not set(stages)<=set(STAGES):
        raise ValueError('invalid queued stage selection')
    root.mkdir(parents=True,exist_ok=True)
    state_path=root/'queued_workflow.json'
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.exists():
            prior=json.loads(state_path.read_text());live=process_identity(prior.get('pid',0))
            if live and live['state'] not in ('Z','X'):
                raise RuntimeError('a queued workflow is already live')
        identity=process_identity(after_pid) if after_pid else None
        if identity and identity['state'] not in ('Z','X'):
            job=json.loads((root/'job.json').read_text())
            if job.get('supervisor_pid')!=after_pid or 'diffusion_gemma_value_aware_followup' not in identity['command']:
                raise ValueError('dependency is not this experiment’s current supervisor')
        state=dict(stages=stages,final_phase=final_phase,after_identity=identity,status='launching',
            source_sha256=sha(Path(__file__).read_bytes()),created=time.time())
        if state_path.exists():
            old=state_path.read_bytes();archive=root/'queue_history'/f'{sha(old)}.json'
            _write(archive,json.loads(old))
        _write(state_path,state)
    with (root/'queued_workflow.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.workflow','supervise',
            '--output',str(root),'--state-file',str(state_path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(queue_pid=child.pid,after_identity=identity,stages=stages)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','supervise'))
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--after-supervisor',type=int,default=0)
    p.add_argument('--stages',nargs='+',choices=STAGES,default=['development','secondary'])
    p.add_argument('--final-phase',choices=('broad50','full'),default='broad50');p.add_argument('--state-file',type=Path)
    args=p.parse_args()
    if args.command=='launch':launch(args.output,args.after_supervisor,args.stages,args.final_phase)
    else:supervise(args.output,args.state_file or args.output/'queued_workflow.json')


if __name__=='__main__':
    main()
