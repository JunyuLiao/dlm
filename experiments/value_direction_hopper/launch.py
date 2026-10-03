"""Detach the frozen, lock-protected evaluator from an interactive tool session."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def launch(root):
    root=root.resolve();config=json.loads((root/'configuration.json').read_text())
    with (root/'worker.lock').open('a') as lock:
        # Do not terminate, replace or compete with an existing live worker.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        changed=[p for p,h in config['source_hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
        if changed:raise ValueError('Frozen executable sources changed: '+repr(changed))
    command=[sys.executable,'-m','experiments.value_direction_hopper.experiment','run',
             '--root',str(root),'--library',config['library'],'--split',config['split'],
             '--methods',*config['methods']]
    if config.get('torch_library'):command+=['--torch-library',config['torch_library']]
    if config.get('profile'):command+=['--profile']
    source_rows=json.loads((Path(config['source'])/(config['split']+'_manifest.json')).read_text())
    if config['samples']!=len(source_rows):command+=['--limit',str(config['samples'])]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log=root/f'worker_{stamp}.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONPATH='src:.',OMP_NUM_THREADS='4')
    with log.open('xb') as stream:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=process.pid,log=str(log),command=command,started=stamp)
    (root/f'launch_{stamp}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);launch(p.parse_args().root)
