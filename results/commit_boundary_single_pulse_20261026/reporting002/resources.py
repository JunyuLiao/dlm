"""Allocated worker/call accounting from every launched generation worker."""
from pathlib import Path
from collections import Counter
import json,hashlib
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gather():
 inputs={};total=Counter();jobs=[];requests=[]
 def read(p):inputs[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
 for track in ['boundary','panel']:
  for root in sorted((R/track).glob('attempt*')):
   for leaf in sorted((root/'launches').glob('*')):
    launch=read(leaf/'launch_receipt.json');exit=read(leaf/'exit_receipt.json')
    job=dict(track=track,attempt=root.name,job=leaf.name,started_unix=launch['started_unix'],**exit);jobs.append(job)
    if not launch['GPU']:total['CPU_orchestration_seconds']+=exit['allocated_seconds'];continue
    total['GPU_workers']+=1;total['allocated_GPU_worker_seconds']+=exit['allocated_seconds'];total['GPU_worker_failures']+=exit['returncode']!=0
    cmd=launch['command']
    if track=='boundary':work=root/'executions'/cmd[cmd.index('--job')+1]
    else:work=root/'runs'/cmd[cmd.index('--phase')+1]/cmd[cmd.index('--arm')+1]
    records=[]
    for p in sorted(work.glob('*/receipt.json')) if track=='boundary' else sorted(work.glob('*.json')):
     value=read(p)
     if 'model_forwards' in value and 'output_sha256' in value:
      records.append(value);requests.append({k:value.get(k) for k in ['label','id','seed','arm','audit','N','C','T','model_forwards','correct']}|dict(track=track,attempt=root.name,job=leaf.name))
    marker=work/'final_complete.json'
    if exit['returncode']==0:
     m=read(marker);assert m['passed'];assert sum(x['model_forwards'] for x in records)+m['initialization_engine_forwards']==m['model_executions']
     total['initialization_engine_forwards']+=m['initialization_engine_forwards']
    else:
     if (work/'failure.json').exists():m=read(work/'failure.json')
     else:
      proof=read(root/'pre_model_import_failure.json');assert proof['engine_execute_forwards']==0 and proof['startup_model_forwards']==0
      m=dict(model_executions=0,startup_forwards=0)
    total['engine_execute_forwards']+=m['model_executions'];total['startup_model_calls_outside_execute']+=m['startup_forwards'];total['complete_native_requests']+=len(records);total['warm_requests']+=sum(x['label']=='warm' for x in records)
 total['explicit_total_model_calls']=total['engine_execute_forwards']+total['startup_model_calls_outside_execute']
 gpu=sorted([j for j in jobs if j['GPU']],key=lambda j:j['started_unix']);overlaps=[]
 for a,b in zip(gpu,gpu[1:]):
  if a['started_unix']+a['allocated_seconds']>b['started_unix']:overlaps.append([a['job'],b['job']])
 assert not overlaps,'GPU allocations overlap'
 return dict(**total,allocated_GPU_worker_hours=total['allocated_GPU_worker_seconds']/3600,jobs=jobs,requests=requests,inputs=inputs,source_receipts_complete=True,no_resource_ceilings=True,
  notes=['Parent allocation includes startup, warmup, qualification, scoring and cleanup with10-second polling resolution. CPU cache hashing within the worker remains allocated time.','Allocation is not active GPU utilization or clean latency. Operator recomputation is not a model forward and is accounted separately.','Superseded panel attempts001/002 launched zero GPU workers. Attempt003 launched one worker that failed before model initialization with zero model calls. Their own waiting controllers were identified before SIGTERM. No native worker or unrelated workload was preempted.'])
if __name__=='__main__':
 out=R/'closeout001';out.mkdir(exist_ok=False);d=gather()
 with (out/'resources.json').open('x') as f:json.dump(d,f,indent=2,sort_keys=True)
 print(json.dumps({k:v for k,v in d.items() if k not in ['jobs','requests','inputs']}))
