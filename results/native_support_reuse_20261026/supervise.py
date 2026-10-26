"""Finish already-authorized serial qualification and CPU analysis; no confirmation."""
from pathlib import Path
import json,hashlib,time,os,subprocess
S=Path(__file__).resolve().parent;REPO=S.parents[1]
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
paths=json.loads((S/'private/runtime_paths.json').read_text());runs=S/'postprocessing_launches';runs.mkdir(exist_ok=False)
while not (S/'attempt002/launches/treatments/exit.json').exists():
    time.sleep(10)
assert json.loads((S/'attempt002/launches/treatments/exit.json').read_text())['returncode']==0
jobs=[('full_off_qualification','attempt003/launch_qualification.py',True),('physical_alias_audit','physical_alias_audit.py',False),('applied_numerical_audit','applied_numerical_audit.py',False),('analysis','analysis_source/attempt002/analyze.py',False)]
for name,relative,gpu in jobs:
    args=[str(S/relative)];env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'src')+':'+str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if not gpu:env['CUDA_VISIBLE_DEVICES']=''
    start=time.time();save(runs/(name+'_launch.json'),dict(command=['<pinned_python>',*args],GPU_launcher=gpu,source_sha256=hashlib.sha256((S/relative).read_bytes()).hexdigest(),started_unix=start))
    with (S/'private'/(name+'_postprocessing.log')).open('x') as log:p=subprocess.run([paths['python'],*args],env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    save(runs/(name+'_exit.json'),dict(returncode=p.returncode,elapsed_seconds=time.time()-start))
    if p.returncode:raise RuntimeError('Preserved failed postprocessing job: '+name)
    print(name+' complete',flush=True)
save(S/'postprocessing_complete.json',dict(passed=True,conditional_confirmation_launched=False,scope='qualification and complete frozen development analysis only'))
