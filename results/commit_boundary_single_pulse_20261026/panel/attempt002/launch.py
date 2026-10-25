"""Serial profile -> prospective manifest checkpoint -> engineering -> panel."""
import hashlib,json,subprocess,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[3];STUDY=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 assert json.loads((ROOT/'model_runtime_verification001.json').read_text())['passed']
 protocol=json.loads((ROOT/'protocol.json').read_text());paths=json.loads((ROOT/'private/runtime_paths.json').read_text());binding=json.loads((ROOT/'binding_receipt.json').read_text());logs=ROOT/'private/logs';logs.mkdir(exist_ok=True)
 boundary=STUDY/'boundary/attempt001'
 while not (boundary/'execution_complete.json').exists():
  assert not (boundary/'controller_stopped.json').exists(),'boundary engineering failure requires review/repair'
  time.sleep(10)
 assert json.loads((boundary/'execution_complete.json').read_text())['passed']
 def run(name,args,gpu):
  for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
  for p,h in json.loads((ROOT/'private/external_sources.json').read_text()).items():assert sha(Path(p))==h
  if gpu:assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
  leaf=ROOT/'launches'/name;leaf.mkdir(parents=True,exist_ok=False);env=os.environ.copy()
  for n in list(env):
   if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
  env.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
  if not gpu:env['CUDA_VISIBLE_DEVICES']=''
  for n,v in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:env[n]=str(ROOT/'cache'/v)
  start=time.time();save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',*args],started_unix=start,GPU=gpu,source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(ROOT/'protocol.json'),no_resource_ceilings=True))
  with (logs/(name+'.log')).open('x') as log:
   p=subprocess.Popen([paths['python'],*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(leaf/'process_identity.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
   while p.poll() is None:
    (ROOT/'private/controller_heartbeat.json').write_text(json.dumps(dict(job=name,pid=p.pid,unix=time.time())));time.sleep(10)
  save(leaf/'exit_receipt.json',dict(returncode=p.returncode,allocated_seconds=time.time()-start,GPU=gpu))
  if p.returncode:raise RuntimeError('Panel job failed: '+name)
 def worker(phase,arm):run(phase+'_'+arm,[str(ROOT/'code/worker.py'),'--root',str(ROOT),'--phase',phase,'--arm',arm],True)
 try:
  for arm in ['corrected_dense_full','matched_allkept_layer29']:worker('profile',arm)
  run('select',[str(ROOT/'code/select.py')],False)
  assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip(),'unrelated staged files'
  subprocess.run(['git','add','--',str(ROOT/'manifest_identity.json')],cwd=REPO,check=True)
  subprocess.run(['git','commit','-m','Freeze baseline-profiled twelve-question single-pulse cohort before treatments'],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
  save(ROOT/'manifest_binding.json',dict(source_checkpoint=binding['source_checkpoint'],cohort_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),manifest_sha256=sha(ROOT/'manifest_identity.json'),protocol_sha256=sha(ROOT/'protocol.json')))
  for arm in protocol['arms']:worker('qualify',arm)
  pairpaths=list((ROOT/'runs/qualify').glob('*/pair*.json'));assert len(pairpaths)==21 and all(json.loads(p.read_text())['passed'] for p in pairpaths)
  # Native FULL/PW must match; pulse branches must share the active common state.
  equality_keys=['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards','anchor']
  control_checks=[]
  for i in range(3):
   records={arm:json.loads((ROOT/f'runs/qualify/{arm}/audit{i:03d}.json').read_text()) for arm in protocol['arms']}
   checks={k:records['corrected_dense_full'][k]==records['dense_piecewise'][k] for k in equality_keys}
   aa=records['matched_allkept_layer29']
   for arm in protocol['arms'][3:]:
    v=records[arm];checks[arm+'_prestate']=v['pre_intervention_native_state']==aa['pre_intervention_native_state']
    checks[arm+'_current_QKV']=v['runtime']['current_operand_route']==aa['runtime']['current_operand_route']
   control_checks.append(dict(cell=i,checks=checks,passed=all(checks.values())))
  save(ROOT/'qualification_controls.json',dict(passed=all(x['passed'] for x in control_checks),cells=control_checks))
  assert all(x['passed'] for x in control_checks),'native/consumer common-state qualification failed'
  save(ROOT/'qualification_complete.json',dict(passed=True,pairs=len(pairpaths),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in pairpaths},manifest_binding_sha256=sha(ROOT/'manifest_binding.json')))
  for arm in protocol['arms']:worker('panel',arm)
  run('analyze',[str(ROOT/'code/analyze.py')],False)
  save(ROOT/'execution_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'manifest_identity.json')))
 except BaseException as e:
  save(ROOT/'controller_stopped.json',dict(passed=False,error_class=type(e).__name__,reason=str(e)));raise
if __name__=='__main__':main()
