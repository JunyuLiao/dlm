"""Idle-device guarded launch; strict shared allocation and forward ceilings."""
import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
if (r/'launch_receipt.json').exists():raise RuntimeError('immutable attempt already launched')
checks=[]
for cmd in [['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']]:
 x=subprocess.run(cmd,text=True,capture_output=True);checks.append(dict(command=cmd,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr))
 if x.returncode:raise RuntimeError('GPU driver unavailable; no worker launched')
if checks[1]['stdout'].strip():raise RuntimeError('GPU occupied; preemption prohibited')
protocol=json.loads((r/'protocol.json').read_text())
started=time.time();budget=protocol['resource']['seconds']
previous=[]
for p in r.parent.glob('attempt*/exit_receipt.json'):
 previous.append(json.loads(p.read_text()))
charged=sum(x['allocated_worker_seconds'] for x in previous)
forward_charge=sum(x.get('forwards',0) for x in previous)
remaining=budget-charged
if remaining<300 or forward_charge>=10000:raise RuntimeError('shared resource ceiling')
env=os.environ.copy();overrides=dict(PYTHONPATH=str(r.parents[2]/'src')+':'+str(r.parents[2]),HF_HUB_OFFLINE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_V2_MODEL_RUNNER='1',PULSEHOLD_DEADLINE_UNIX=str(started+remaining-30))
for var,leaf in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('MPLCONFIGDIR','matplotlib')]:overrides[var]=str(r/'cache'/leaf)
env.update(overrides)
env['PULSEHOLD_PRIOR_FORWARDS']=str(forward_charge)
overrides['PULSEHOLD_PRIOR_FORWARDS']=str(forward_charge)
cmd=['timeout','--signal=TERM','--kill-after=15s',str(max(1,int(remaining-15))),'/home/exouser/dyh/v31_env_20261005/bin/python',str(r/'code/worker.py'),'--root',str(r)]
with (r/'launch_receipt.json').open('x') as f:json.dump(dict(command=cmd,environment_overrides=overrides,started_unix=started,gpu_checks=checks,prior_allocated_seconds=charged,prior_forwards=forward_charge,source_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f,indent=2)
with (r/'private/worker.log').open('x') as log:ret=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
resource=json.loads((r/'private/live_resource.json').read_text()) if (r/'private/live_resource.json').exists() else dict(forwards=0)
receipt=dict(returncode=ret.returncode,allocated_worker_seconds=time.time()-started,forwards=resource['forwards'],resource=resource,previous_allocated_seconds=charged,previous_forwards=forward_charge)
with (r/'exit_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
if ret.returncode:
 with (r/'partial_failure.json').open('x') as f:json.dump(dict(status='partial_or_failed_preserved',reason='See private worker log; no completion claim',**receipt),f,indent=2)
print(json.dumps(receipt),flush=True)
raise SystemExit(ret.returncode)
