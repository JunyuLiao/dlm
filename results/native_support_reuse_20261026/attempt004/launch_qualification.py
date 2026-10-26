from pathlib import Path
import json,hashlib,os,subprocess,time
R=Path(__file__).resolve().parent;REPO=R.parents[2]
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=json.loads((R.parent/'private/runtime_paths.json').read_text());binding=json.loads((R/'binding.json').read_text());protocol=json.loads((R/'protocol.json').read_text())
for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; no preemption'
assert json.loads((R.parent/'attempt002/runs/treatments/final_complete.json').read_text())['passed']
env=os.environ.copy()
for n in list(env):
    if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
env.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
for n,v in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions')]:env[n]=str(R/'private/cache'/v)
leaf=R/'launches/qualification';leaf.mkdir(parents=True,exist_ok=False);(R/'private/logs').mkdir(parents=True,exist_ok=True);start=time.time();args=[str(R/'code/qualify_worker.py')]
save(leaf/'launch.json',dict(command=['<pinned_python>',*args],GPU=True,source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(R/'protocol.json'),started_unix=start,no_campaign_timeout=True))
with (R/'private/logs/qualification.log').open('x') as log:
    p=subprocess.Popen([paths['python'],*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(leaf/'process.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
    while p.poll() is None:time.sleep(10)
save(leaf/'exit.json',dict(returncode=p.returncode,allocated_GPU_worker_seconds=time.time()-start))
if p.returncode:raise RuntimeError('Preserved failed full-off qualification')
