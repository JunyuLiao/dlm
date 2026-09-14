"""One worker, process-verified lightweight health check every 15 minutes."""
import argparse
import json
import os
import subprocess
import shutil
import sys
import time
from .protocol import ROOT
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','screen','calibrate','development','run-final','refinement-screen','confirm'])
    p.add_argument('--methods',nargs='+');p.add_argument('--targets',nargs='+',type=float);p.add_argument('--max-rounds',type=int)
    p.add_argument('--seed-offset',type=int)
    args=p.parse_args()
    from .efficient_iteration import guard_legacy_run
    guard_legacy_run(ROOT,args.command)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/f'{args.command}.log').open('a',buffering=1) as log:
        module='run' if args.command in ('smoke','screen') else ('refine_run' if args.command=='refinement-screen' else ('confirmation' if args.command=='confirm' else 'evaluate'))
        subcommand='screen' if args.command=='refinement-screen' else ('run' if args.command=='confirm' else args.command)
        command=[sys.executable,'-u','-m',f'experiments.diffusion_gemma_value_aware.{module}',subcommand]
        if args.methods:command+=['--methods',*args.methods]
        if args.targets:command+=['--targets',*[str(v) for v in args.targets]]
        if args.max_rounds is not None:command+=['--max-rounds',str(args.max_rounds)]
        if args.seed_offset is not None:command+=['--seed-offset',str(args.seed_offset)]
        worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        _write(ROOT/'job.json',dict(pid=worker.pid,supervisor_pid=os.getpid(),stage=args.command,started=time.time()))
        while True:
            try: code=worker.wait(timeout=900);break
            except subprocess.TimeoutExpired:
                progress=json.loads((ROOT/'progress.json').read_text()) if (ROOT/'progress.json').exists() else None
                gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                record=dict(time=time.time(),pid=worker.pid,alive=worker.poll() is None,stage=args.command,
                    completed=len(list(ROOT.glob('**/shards/*.json'))),progress=progress,
                    gpu=gpu.stdout.strip(),gpu_check_error=gpu.stderr.strip(),
                    disk_free_bytes=shutil.disk_usage(ROOT).free,
                    failure_lines=sum(1 for _ in (ROOT/'failures.jsonl').open()) if (ROOT/'failures.jsonl').exists() else 0)
                _append(ROOT/'monitor.jsonl',record);print(json.dumps(record),flush=True)
        terminal=dict(exit_code=code,finished=time.time(),stage=args.command,pid=worker.pid)
        _write(ROOT/f'{args.command}_terminal.json',terminal);print(json.dumps(terminal),flush=True)
        if code: sys.exit(code)


if __name__=='__main__': main()
