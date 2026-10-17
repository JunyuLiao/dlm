"""Explicit user-authorized resume guard; never call while authorization is pending."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--user-authorized-resume',action='store_true')
    p.add_argument('--authorization-note')
    a=p.parse_args()
    if not a.user_authorized_resume or not a.authorization_note:
        p.error('full503 remains paused: a new explicit user authorization and recorded note are required')
    study=a.study.resolve();pause=study/'PAUSED_USER_REQUEST.json'
    record=json.loads(pause.read_text())
    if record['status']!='paused_user_request':raise RuntimeError('expected a completed user-requested pause')
    lock=(study/'private/resume.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(),True)
    folder=study/'resume_records';folder.mkdir(exist_ok=True)
    i=max([int(x.stem.removeprefix('resume')) for x in folder.glob('resume*.json')],default=0)+1
    authorization=dict(status='explicit_user_authorized',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        note=a.authorization_note,pause_sha256=hashlib.sha256(pause.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest())
    dest=folder/f'resume{i:03d}.json'
    with dest.open('x') as f:json.dump(authorization,f,indent=2);f.write('\n')
    execution=study/'private/execution'
    command=['/home/exouser/dyh/v31_env_20261005/bin/python',str(execution/'scripts/v31_value_e2e_study.py'),
             '--stage','run','--study',str(study)]
    env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}')
    os.execvpe(command[0],command,env)

if __name__=='__main__':main()
