"""Start K128 only after corrected K256 independently verifies and seals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT=next(p for p in (Path.cwd(),*Path(__file__).absolute().parents) if (p/'.git').exists())
PYTHON='/home/exouser/dyh/v31_env_20261005/bin/python'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root):
    path=root/'final_complete.json'
    if not path.exists():return False
    p=read(path)
    if p['status']!='complete' or p['official_rescore_mismatches']!=0:
        raise RuntimeError('unqualified parent completion')
    if sha(root/'protocol.json')!=p['protocol_sha256'] or sha(root/'native_complete.json')!=p['native_complete_sha256']:
        raise RuntimeError('parent protocol/native marker drift')
    analyses=[f for f in (root/'analysis').glob('attempt*/analysis_complete.json') if sha(f)==p['analysis_complete_sha256']]
    if len(analyses)!=1:raise RuntimeError('parent independent analysis seal missing')
    for name,digest in read(analyses[0])['files_sha256'].items():
        if sha(analyses[0].parent/name)!=digest:raise RuntimeError('parent analysis drift')
    return True


def run(first,second,hold):
    print(json.dumps(dict(event='waiting_for_verified_k256',root=str(first))),flush=True)
    last=time.monotonic();deadline=last+24*3600
    while not seal(first):
        if hold.exists():
            print(json.dumps(dict(event='followup_held',marker=str(hold))),flush=True);return
        if time.monotonic()>deadline:raise RuntimeError('parent wait deadline reached; no dependent launch')
        if time.monotonic()-last>=1800:
            try:state=read(first/'state.json')
            except (FileNotFoundError,ValueError):state={}
            print(json.dumps(dict(event='parent_progress',state=state)),flush=True);last=time.monotonic()
        time.sleep(30)
    if hold.exists():return
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT))
    if not second.exists():
        subprocess.run([PYTHON,str(ROOT/'scripts/v31_humaneval_cfgdet_study.py'),'prepare',
            '--study',str(second),'--budget','128'],cwd=ROOT,env=env,check=True)
    a,b=read(first/'protocol.json'),read(second/'protocol.json')
    if b['prefix_budget_tokens']!=128 or b['numerical_policy_sha256']!=a['numerical_policy_sha256']:
        raise RuntimeError('followup budget/numerical policy mismatch')
    if seal(second):return
    execution=second/'private/execution'
    env['PYTHONPATH']=str(execution/'src')+':'+str(execution)
    print(json.dumps(dict(event='starting_corrected_k128',protocol_sha256=sha(second/'protocol.json'))),flush=True)
    subprocess.run([PYTHON,str(execution/'scripts/v31_humaneval_cfgdet_study.py'),'run',
                    '--study',str(second)],cwd=ROOT,env=env,check=True)
    if not seal(second):raise RuntimeError('followup exited without independent completion seal')
    print(json.dumps(dict(event='both_corrected_panels_verified',k256=str(first),k128=str(second))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--k256',type=Path,required=True)
    p.add_argument('--k128',type=Path,required=True);p.add_argument('--hold',type=Path,required=True)
    a=p.parse_args();run(a.k256.absolute(),a.k128.absolute(),a.hold.absolute())
