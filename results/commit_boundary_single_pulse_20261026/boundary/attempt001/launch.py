"""Serial workers with source/binding/device checks, no campaign limits."""
import json,hashlib,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 protocol=json.loads((ROOT/'protocol.json').read_text());paths=json.loads((ROOT/'private/runtime_paths.json').read_text());binding=json.loads((ROOT/'binding_receipt.json').read_text())
 assert sha(ROOT/'protocol.json')==binding['protocol_sha256']
 for job in ['references','development']:
  for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
  for p,h in json.loads((ROOT/'private/external_sources.json').read_text()).items():assert sha(Path(p))==h
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
  leaf=ROOT/'launches'/job;leaf.mkdir(parents=True,exist_ok=False)
  env=os.environ.copy()
  for n in list(env):
   if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
  env.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
  for n,v in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:env[n]=str(ROOT/'cache'/v)
  args=[str(ROOT/'code/worker.py'),'--root',str(ROOT),'--job',job];start=time.time()
  save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',*args],source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(ROOT/'protocol.json'),started_unix=start,GPU=True,no_resource_ceilings=True))
  logs=ROOT/'private/logs';logs.mkdir(exist_ok=True)
  with (logs/(job+'.log')).open('x') as log:
   p=subprocess.Popen([paths['python'],*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(leaf/'process_identity.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
   while p.poll() is None:
    (ROOT/'private/controller_heartbeat.json').write_text(json.dumps(dict(job=job,pid=p.pid,unix=time.time())));time.sleep(10)
  save(leaf/'exit_receipt.json',dict(returncode=p.returncode,allocated_seconds=time.time()-start,GPU=True))
  if p.returncode:
   save(ROOT/'controller_stopped.json',dict(passed=False,failed_job=job));raise RuntimeError('Boundary worker failed: '+job)
 save(ROOT/'execution_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':main()
