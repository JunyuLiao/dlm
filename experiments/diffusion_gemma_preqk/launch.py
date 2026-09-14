"""Polite single-H100 queue and 15-minute durable monitoring.

Never terminates another GPU process. Launches only after nvidia-smi reports no
compute processes. Run this supervisor outside the GPU-hiding sandbox.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from .config import ROOT


def gpu_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    return [r for r in output.splitlines() if r.strip()]


def pipeline_command(pipeline,root,*,stage=None,conditions=None):
    if pipeline=='screen':return [sys.executable,'-m','experiments.diffusion_gemma_preqk.run','screen-pipeline','--output',str(root)]
    if pipeline=='online':return [sys.executable,'-m','experiments.diffusion_gemma_preqk.online_run','pipeline','--stage',str(root)]
    if pipeline=='validation':return [sys.executable,'-m','experiments.diffusion_gemma_preqk.validation','pipeline','--stage',str(root)]
    if pipeline=='performance':
        if stage is None or not conditions:raise ValueError('performance requires a completed source stage and explicit conditions')
        return [sys.executable,'-m','experiments.diffusion_gemma_preqk.performance','--stage',str(stage),'--output',str(root),'--conditions',*conditions]
    raise ValueError(pipeline)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    parser.add_argument('--pipeline',choices=('screen','online','performance','validation'),default='screen')
    parser.add_argument('--stage',type=Path);parser.add_argument('--conditions',nargs='+')
    parser.add_argument('--interval',type=int,default=900);args=parser.parse_args()
    root=args.output or (ROOT if args.pipeline=='screen' else ROOT/'online_v1')
    if args.pipeline=='performance' and (args.output is None or args.stage is None or not args.conditions):
        parser.error('performance requires --output, --stage and --conditions')
    if args.interval<60:raise ValueError('do not busy-poll the shared GPU')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            processes=gpu_processes()
            _write(root/'job.json',dict(supervisor_pid=os.getpid(),status='waiting_for_idle_gpu' if processes else 'launching',
                updated=time.time(),other_compute_processes=processes,poll_seconds=args.interval))
            _append(root/'monitor.jsonl',dict(time=time.time(),event='queue_check',other_compute_processes=processes))
            if not processes:break
            time.sleep(args.interval)
        command=pipeline_command(args.pipeline,root,stage=args.stage,conditions=args.conditions)
        with (root/'gpu.log').open('a') as log:
            worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(supervisor_pid=os.getpid(),pid=worker.pid,status='running',started=time.time(),command=command,poll_seconds=args.interval))
            while True:
                try:code=worker.wait(timeout=args.interval)
                except subprocess.TimeoutExpired:
                    progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else None
                    _append(root/'monitor.jsonl',dict(time=time.time(),event='running',pid=worker.pid,progress=progress,
                        completed_shards=len(list(root.glob({'screen':'screen/shards/*.json','online':'conditions/*/shards/*.json','performance':'timings/*/*.json','validation':'**/shards/*.json'}[args.pipeline]))),gpu_processes=gpu_processes()))
                    continue
                _write(root/'job.json',dict(supervisor_pid=os.getpid(),pid=worker.pid,status='finished' if code==0 else 'failed',exit_code=code,finished=time.time()))
                _append(root/'monitor.jsonl',dict(time=time.time(),event='worker_exit',exit_code=code));break


if __name__=='__main__':main()
