"""Detach one authorized diagnostic worker; retain its log and launch record."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .experiment import atomic


def launch(root):
    root=root.resolve()
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log=root/f'worker_{stamp}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
             PYTHONPATH='src:.',OMP_NUM_THREADS='4')
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.trajectory_experiment','run','--root',str(root)]
    with log.open('xb') as f:
        child=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=child.pid,log=str(log),command=cmd,started=stamp)
    atomic(root/f'launch_{stamp}.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    launch(p.parse_args().root)
