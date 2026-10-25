"""CPU-only readout relay; GPU scheduling remains in the frozen controllers."""
from pathlib import Path
import os,subprocess,time,json,hashlib
R=Path(__file__).resolve().parents[1];B=R/'boundary/attempt001';C=R/'panel/attempt004';OUT=R/'reporting_execution002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True)
def main():
 OUT.mkdir(exist_ok=False);paths=json.loads((R/'private/runtime_paths.json').read_text());logs=R/'private/reporting_logs';logs.mkdir(exist_ok=True)
 def wait_for(marker,root):
  while not marker.exists():
   if (root/'controller_stopped.json').exists():raise RuntimeError(str(root.relative_to(R))+' controller stopped; inspect preserved failure')
   (OUT/'progress.json').write_text(json.dumps(dict(waiting=str(marker.relative_to(R)),unix=time.time())))
   time.sleep(10)
 def run(name,script):
  start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R.parents[1]/'src')+':'+str(R.parents[1]));leaf=OUT/name;leaf.mkdir(exist_ok=False)
  save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',str(script.relative_to(R.parents[1]))],source_sha256=sha(script),started_unix=start,CPU_only=True,model_forwards=0))
  with (logs/(name+'.log')).open('x') as log:proc=subprocess.run([paths['python'],str(script)],env=env,stdout=log,stderr=subprocess.STDOUT)
  save(leaf/'exit_receipt.json',dict(returncode=proc.returncode,elapsed_seconds=time.time()-start,CPU_only=True,model_forwards=0))
  if proc.returncode:raise RuntimeError('CPU reporting failed: '+name)
 wait_for(B/'execution_complete.json',B)
 assert json.loads((B/'final_complete.json').read_text())['passed'],'Existing boundary analysis must already be complete'
 save(OUT/'reused_boundary_analysis.json',dict(passed=True,summary_sha256=sha(B/'analysis001/summary.json'),no_model_forwards=True))
 wait_for(C/'qualification_complete.json',C);run('online_legacy',R/'reporting002/qualify_online_legacy.py')
 wait_for(C/'execution_complete.json',C)
 for name,script in [('fairness','fairness.py'),('positions_content','position_content.py'),('resources','resources.py'),('plots','plots.py')]:run(name,R/'reporting002'/script)
 save(OUT/'final_complete.json',dict(passed=True,CPU_only=True,model_forwards=0,finished_unix=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(OUT/'failure.json',dict(passed=False,error_class=type(e).__name__,reason=str(e),CPU_only=True,model_forwards=0));raise
