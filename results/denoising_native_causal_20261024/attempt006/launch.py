import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--group',required=True,choices=['smoke','short','long']);a=p.parse_args()
r=Path(__file__).resolve().parent
checks=[]
for cmd in [['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']]:
 x=subprocess.run(cmd,text=True,capture_output=True);checks.append(dict(command=cmd,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr))
 if x.returncode:raise RuntimeError('GPU driver unavailable')
if checks[1]['stdout'].strip():raise RuntimeError('GPU occupied; do not preempt')
env=os.environ.copy();overrides=dict(PYTHONPATH=str(r.parents[2]/'src')+':'+str(r.parents[2]),HF_HUB_OFFLINE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_V2_MODEL_RUNNER='1')
for name,leaf in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:overrides[name]=str(r/'cache'/leaf)
env.update(overrides)
command=['timeout','--signal=TERM','--kill-after=30s','1800','/home/exouser/dyh/v31_env_20261005/bin/python',str(r/'code/worker.py'),'--root',str(r),'--group',a.group]
started=time.time()
with (r/(a.group+'_launch.json')).open('x') as f:json.dump(dict(command=command,environment_overrides=overrides,started_unix=started,gpu_checks=checks,spec_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f,indent=2)
with (r/'private'/(a.group+'.log')).open('x') as f:ret=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
with (r/(a.group+'_exit.json')).open('x') as f:json.dump(dict(returncode=ret.returncode,seconds=time.time()-started),f,indent=2)
print(json.dumps(dict(group=a.group,returncode=ret.returncode,seconds=time.time()-started)),flush=True)
raise SystemExit(ret.returncode)
