"""Frozen v1 phase policy on final prompts: isolates the v2 late-global change.

The v1 phase policy was selected before any v2 final answers were observed.
This is a diagnostic ablation, not a threshold candidate selected on final
accuracy. It uses the unchanged core router/decoder and separate shards.
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

from experiments.diffusion_gemma_jl_output_aware.projections import Projections

from .experiment import atomic, fingerprint, sha
from . import query_adaptive_guardrail as core


ROOT=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v2'
SOURCE=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_guardrail_v1'/'configs'/'thresholds'/'T_s70.json'


def run(root):
    root=Path(root)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,manifests=core.prepare(root)
        policy=json.loads(SOURCE.read_text())['policy']
        spec=dict(core_fingerprint=cfg['fingerprint'],source_policy_sha256=sha(SOURCE),
            runner_sha256=sha(Path(__file__)),stage='phase_ablation_v1',
            method='T',target=70,policy=policy,
            predeclared_before_v2_final=True)
        spec['fingerprint']=fingerprint(spec)
        path=root/'configs/phase_ablation_v1.json'
        if path.exists() and json.loads(path.read_text())!=spec:
            raise ValueError('Ablation provenance changed')
        atomic(path,spec)
        adapter=core._adapter(cfg);projections=Projections()
        errors=[]
        for row in manifests['final']:
            try:core.cached(adapter,root,'phase_ablation_v1','T',70,row,
                            policy,cfg,projections)
            except Exception as error:
                errors.append(dict(id=row['id'],error=repr(error),
                                   traceback=traceback.format_exc()))
                atomic(root/'phase_ablation_failures.json',errors)
                if 'illegal memory' in str(error) or 'device-side assert' in str(error):raise


def launch(root):
    root=Path(root).resolve()
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    command=[sys.executable,'-m',
        'experiments.value_direction_hopper.query_adaptive_guardrail_ablation',
        'run','--root',str(root)]
    logfile=root/f'phase_ablation_{int(time.time())}.log'
    with logfile.open('xb') as output:
        worker=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],
            env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    atomic(root/'phase_ablation_launch.json',dict(pid=worker.pid,
        log=str(logfile),command=command,started=time.time()))
    print(json.dumps(dict(pid=worker.pid,log=str(logfile))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('run','launch'))
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.stage=='launch':launch(args.root)
    else:run(args.root)
