"""Queue one authorized next stage after a specific live Linux process exits.

pidfd is used when the Python build provides it. Otherwise kernel /proc process
identity/state is checked every900s, never an experiment lock/state file.
GPU availability, when needed, is also checked every900s.
Never terminates another process or launches beside an occupied H100.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from .protocol import ROOT
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append


def process_state(pid):
    """Kernel process state and start ticks guard against PID reuse."""
    try: fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return dict(state=fields[0],start_ticks=fields[19])


def same_live_process(pid,start_ticks):
    state=process_state(pid)
    return state is not None and state['start_ticks']==start_ticks and state['state'] not in ('Z','X')


def stage_command(stage,methods=None,targets=None,max_rounds=None):
    """A queued development pass must not resurrect a default Cartesian sweep."""
    if stage not in ('refinement-screen','development','calibrate','finalize'):
        raise ValueError('unsupported queued stage')
    if stage=='finalize':
        if not methods or targets or max_rounds is not None:
            raise ValueError('final handoff requires selected methods only; all frozen targets are mandatory')
        return [sys.executable,'-u','-m','experiments.diffusion_gemma_value_aware.finalize',
            '--methods',*methods]
    if stage in ('development','calibrate'):
        if not methods or not targets:
            raise ValueError('calibration/development queue requires explicit methods and targets')
        if not set(targets)<={.25,.5,.75,.9}:
            raise ValueError('unsupported queued target')
        if stage=='calibrate' and (max_rounds is None or max_rounds<1):
            raise ValueError('calibration queue requires an explicit positive round limit')
        if stage=='development' and max_rounds is not None:
            raise ValueError('development does not use calibration rounds')
    elif methods or targets or max_rounds is not None:
        raise ValueError('refinement-screen does not accept development selections')
    command=[sys.executable,'-u','-m','experiments.diffusion_gemma_value_aware.launch',stage]
    if methods:command+=['--methods',*methods]
    if targets:command+=['--targets',*[str(v) for v in targets]]
    if max_rounds is not None:command+=['--max-rounds',str(max_rounds)]
    return command


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--stage',choices=['refinement-screen','development','calibrate','finalize'],required=True)
    p.add_argument('--methods',nargs='+');p.add_argument('--targets',nargs='+',type=float)
    p.add_argument('--max-rounds',type=int)
    args=p.parse_args();proc=Path('/proc')/str(args.after_pid)
    next_command=stage_command(args.stage,args.methods,args.targets,args.max_rounds)
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'experiments.diffusion_gemma_value_aware.evaluate calibrate' not in command:
        raise RuntimeError('the requested predecessor is not our calibration worker')
    start=process_state(args.after_pid)['start_ticks']
    fd=os.pidfd_open(args.after_pid) if hasattr(os,'pidfd_open') else None
    data=dict(pid=os.getpid(),after_pid=args.after_pid,after_process_start_ticks=start,
        after_command=command,stage=args.stage,next_command=next_command,
        queued_at=time.time(),wait_backend='pidfd' if fd is not None else 'kernel_proc_900s')
    _write(ROOT/'queued_stage.json',data)
    try:
        while True:
            if fd is not None:
                ready,_,_=select.select([fd],[],[],900)
                if ready:break
            else:
                if not same_live_process(args.after_pid,start):break
            _append(ROOT/'queue_monitor.jsonl',dict(time=time.time(),after_pid=args.after_pid,process_handle_live=True))
            if fd is None:time.sleep(900)
    finally:
        if fd is not None:os.close(fd)
    # The predecessor supervisor writes terminal state immediately after wait.
    terminal=None
    for _ in range(10):
        path=ROOT/'calibrate_terminal.json'
        if path.exists():
            candidate=json.loads(path.read_text())
            if candidate['pid']==args.after_pid:terminal=candidate;break
        time.sleep(1)
    if terminal is None or terminal['exit_code']!=0:
        _write(ROOT/'queued_stage_terminal.json',dict(status='not_launched',reason='predecessor terminal success not verified',predecessor=terminal))
        return
    while True:
        status=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
        if status.returncode==0 and not status.stdout.strip():break
        _append(ROOT/'queue_monitor.jsonl',dict(time=time.time(),waiting_for_free_gpu=True,
            compute_processes=status.stdout.strip(),error=status.stderr.strip()))
        time.sleep(900)
    _write(ROOT/'queued_stage_launch.json',dict(stage=args.stage,started=time.time(),predecessor=terminal))
    result=subprocess.run(next_command)
    _write(ROOT/'queued_stage_terminal.json',dict(stage=args.stage,exit_code=result.returncode,finished=time.time()))
    sys.exit(result.returncode)


if __name__=='__main__':main()
