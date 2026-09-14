"""Single-worker stage launcher with persistent 900-second status checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append
from .protocol import ROOT
from .run import gpu_idle


def worker_args(stage, root, extra):
    if stage not in ('supplement', 'calibrate'):
        raise ValueError('unsupported follow-up stage')
    return [sys.executable, '-u', '-m', __package__+'.'+stage, '--output', str(root), *extra]


def supervise(root, stage, extra):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle()
        command = worker_args(stage, root, extra)
        with (root/f'{stage}.log').open('a', buffering=1) as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            _write(root/'job.json', dict(pid=worker.pid, supervisor_pid=os.getpid(),
                stage=stage, command=command, started=time.time()))
            while True:
                try:
                    code = worker.wait(timeout=900); break
                except subprocess.TimeoutExpired:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                        '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
                    progress = json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else None
                    _append(root/'monitor.jsonl', dict(time=time.time(), stage=stage, pid=worker.pid,
                        alive=worker.poll() is None, gpu=gpu.stdout.strip(), gpu_error=gpu.stderr.strip(),
                        progress=progress, disk_free_bytes=shutil.disk_usage(root).free))
            _write(root/f'{stage}_terminal.json', dict(exit_code=code, pid=worker.pid,
                finished=time.time(), stage=stage, command=command,
                scope='This stage alone is not completion of the full study'))
            if code:
                raise SystemExit(code)


def launch(root, stage, extra):
    root.mkdir(parents=True, exist_ok=True)
    worker_args(stage, root, extra)
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); gpu_idle()
    with (root/'supervisor.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', __package__+'.jobs',
            'supervise', '--stage', stage, '--output', str(root), '--', *extra],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid, stage=stage)), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=('launch', 'supervise'))
    p.add_argument('--stage', choices=('supplement', 'calibrate'), required=True)
    p.add_argument('--output', type=Path, default=ROOT)
    args, extra = p.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    (launch if args.command == 'launch' else supervise)(args.output, args.stage, extra)


if __name__ == '__main__':
    main()
