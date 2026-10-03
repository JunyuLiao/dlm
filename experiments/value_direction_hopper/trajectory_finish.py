"""Finish collection, run the queued schedule control, then audit fresh reports.

The collector caches its report module. After every generation shard is verified,
replace that obsolete CPU reporting stage with a fresh optimized analysis process.
Never signal the collector while any required generation shard is missing.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .experiment import atomic, sha, shard_path


def expected(main):
    config=json.loads((main/'configuration.json').read_text())
    rows=json.loads((main/'manifest.json').read_text())
    return config,[(shard_path(main/regime,m,r),regime,m,r) for regime,methods in
        (('adaptive',config['methods']),('fixed512',config['fixed_methods'])) for r in rows for m in methods]


def verify_collection(main):
    config,items=expected(main)
    missing=[str(p) for p,*_ in items if not p.exists()]
    if missing:return False,len(items)-len(missing)
    for p,regime,method,row in items:
        r=json.loads(p.read_text())
        for key,value in [('status','complete'),('fingerprint',config['fingerprint']),('id',row['id']),
                          ('method',method),('regime',regime),('prompt_hash',row['prompt_hash']),
                          ('seed',row['seed']),('generation_budget',row['generation_budget'])]:
            if r[key]!=value:raise ValueError(f'Invalid completed shard {p}: {key}')
        if regime=='fixed512' and r['metadata']['actual_denoising_step_count']!=512:
            raise ValueError('Non512 fixed result')
        if sha(r['trace_path'])!=r['trace_sha256']:raise ValueError('Trace hash mismatch')
    return True,len(items)


def active(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return False
    return path.read_text().split(') ',1)[1].split()[0]!='Z'


def finish(main,supplement,pid):
    while True:
        ready,count=verify_collection(main)
        atomic(main/'finalizer_status.json',dict(stage='waiting_for_collection',completed=count,pid=os.getpid(),updated=time.time()))
        if ready:break
        if not active(pid):raise RuntimeError('Collector exited with missing shards; inspect failures before resuming')
        for _ in range(15):time.sleep(60)
    if active(pid):
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        required=(b'experiments.value_direction_hopper.trajectory_experiment',b'run',str(main).encode())
        if not all(x in cmd for x in required):raise RuntimeError('Collector PID no longer identifies the expected job')
        atomic(main/'analysis_handoff.json',dict(verified_generation_shards=count,collector_pid=pid,
            reason='All generation complete; replace cached CPU report with current vectorized raw-shard reporter',time=time.time()))
        os.kill(pid,signal.SIGTERM)
        for _ in range(60):
            if not active(pid):break
            time.sleep(1)
        if active(pid):raise RuntimeError('Collector did not exit; no escalation or GPU overlap attempted')
    atomic(main/'status.json',dict(status='analysis_and_supplement',completed=count,expected=count,pid=os.getpid(),updated=time.time()))
    atomic(main/'finalizer_status.json',dict(stage='fixed48_control',pid=os.getpid(),updated=time.time()))
    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='4')
    subprocess.run([sys.executable,'-m','experiments.value_direction_hopper.trajectory_fixed48','run',
                    '--root',str(supplement),'--main',str(main)],env=env,check=True)
    atomic(main/'finalizer_status.json',dict(stage='fresh_report',pid=os.getpid(),updated=time.time()))
    archive=main/'analysis_sources';archive.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('trajectory*.py'):(archive/p.name).write_bytes(p.read_bytes())
    subprocess.run([sys.executable,'-m','experiments.value_direction_hopper.trajectory_experiment','report',
                    '--root',str(main)],env=env,check=True)
    audit=json.loads((main/'audit.json').read_text())
    control=json.loads((supplement/'summary.json').read_text())
    if not audit['complete'] or audit['violations'] or not control['complete']:
        raise RuntimeError('Incomplete final audit')
    atomic(main/'status.json',dict(status='complete',completed=count,expected=count,supplement_samples=26,
        supplement_methods=3,updated=time.time()))
    atomic(main/'finalizer_status.json',dict(stage='complete',pid=os.getpid(),updated=time.time()))
    print(json.dumps(dict(status='complete',main=str(main),supplement=str(supplement))),flush=True)


def launch(main,supplement,pid):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log=main/f'finalizer_{stamp}.log'
    cmd=[sys.executable,'-m','experiments.value_direction_hopper.trajectory_finish','--main',str(main),
         '--supplement',str(supplement),'--collection-pid',str(pid)]
    with log.open('xb') as stream:
        process=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=process.pid,command=cmd,log=str(log),started=stamp)
    atomic(main/f'finalizer_launch_{stamp}.json',record)
    atomic(main/'finalizer_launch.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True,type=Path)
    p.add_argument('--supplement',required=True,type=Path);p.add_argument('--collection-pid',required=True,type=int)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    (launch if a.detach else finish)(a.main.resolve(),a.supplement.resolve(),a.collection_pid)
