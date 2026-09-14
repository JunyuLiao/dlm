"""Run the resumable sweep with process-verified 30-minute status snapshots."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .run import ROOT, fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _append, _write


def main():
    audit=json.loads((ROOT/'smoke.json').read_text())
    assert audit['passed'] and audit['fingerprint']==fingerprint(ROOT)
    log=(ROOT/'execution.log').open('a',buffering=1)
    worker=subprocess.Popen([sys.executable,'-u','-m','experiments.diffusion_gemma_oracle.run','run'],stdout=log,stderr=subprocess.STDOUT)
    _write(ROOT/'job.json',dict(pid=worker.pid,supervisor_pid=os.getpid(),started=time.time()))
    while True:
        try:
            code=worker.wait(timeout=1800)
            _append(ROOT/'monitor.jsonl',dict(time=time.time(),pid=worker.pid,alive=False,exit_code=code))
            break
        except subprocess.TimeoutExpired:
            progress=json.loads((ROOT/'progress.json').read_text()) if (ROOT/'progress.json').exists() else None
            _append(ROOT/'monitor.jsonl',dict(time=time.time(),pid=worker.pid,alive=worker.poll() is None,
                completed=len(list(ROOT.glob('conditions/*/shards/*.json'))),progress=progress))
    log.close()
    subprocess.run([sys.executable,'-m','experiments.diffusion_gemma_oracle.run','report'],check=True)
    _write(ROOT/'job_terminal.json',dict(exit_code=code,finished=time.time()))


if __name__=='__main__':main()
