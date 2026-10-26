"""One worker/GPU: qualify all consumers, profile, freeze selected cohort."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;S=R.parents[1];REPO=R.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    assert json.loads((R/'model_runtime_verification001.json').read_text())['passed']
    paths=json.loads((S/'private/runtime_paths.json').read_text());protocol=json.loads((R/'protocol.json').read_text());binding=json.loads((R/'binding.json').read_text())
    def run(name,args,gpu):
        for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
        if gpu:assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; never preempt unrelated work'
        env=os.environ.copy()
        for n in list(env):
            if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
        env.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        for n,v in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions'),('MPLCONFIGDIR','matplotlib')]:env[n]=str(R/'cache'/v)
        leaf=R/'launches'/name;leaf.mkdir(parents=True,exist_ok=False);logs=R/'private/logs';logs.mkdir(parents=True,exist_ok=True);start=time.time()
        save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',*args],GPU=gpu,started_unix=start,source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(R/'protocol.json'),no_campaign_timeout=True))
        with (logs/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([paths['python'],*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(leaf/'process_identity.json',dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
            while proc.poll() is None:
                (R/'private/controller_heartbeat.json').write_text(json.dumps(dict(job=name,pid=proc.pid,unix=time.time())));time.sleep(10)
        save(leaf/'exit_receipt.json',dict(GPU=gpu,returncode=proc.returncode,allocated_seconds=time.time()-start))
        if proc.returncode:raise RuntimeError('Preserved failed job '+name)
    for arm in protocol['arms']:run('qualify_'+arm,[str(R/'code/native_worker.py'),'--job','qualify','--arm',arm],True)
    checks=[]
    for i in range(3):
        a=json.loads((R/f'runs/qualify/corrected_dense_full/observed{i:03d}.json').read_text());b=json.loads((R/f'runs/qualify/dense_piecewise/observed{i:03d}.json').read_text())
        eq={k:a[k]==b[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards']};assert all(eq.values());checks.append(eq)
    qualifications=list((R/'runs/qualify').glob('*/qualification*.json'));assert len(qualifications)==9 and all(json.loads(p.read_text())['passed'] for p in qualifications)
    save(R/'qualification_complete.json',dict(passed=True,observer_control_cells=9,FULL_PW_checks=checks,input_hashes={str(p.relative_to(R)):sha(p) for p in qualifications}))
    for arm in protocol['arms']:run('profile_'+arm,[str(R/'code/native_worker.py'),'--job','profile','--arm',arm],True)
    run('cohort_selection',[str(R/'code/cohort_select.py')],False)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip(),'Unrelated staged files'
    subprocess.run(['git','add','--',str(R/'manifest_identity.json')],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m','Freeze baseline-only development and confirmation support-discovery cohort'],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
    save(R/'manifest_binding.json',dict(source_checkpoint=binding['source_checkpoint'],cohort_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),manifest_sha256=sha(R/'manifest_identity.json'),protocol_sha256=sha(R/'protocol.json')))
    save(R/'execution_complete.json',dict(passed=True,profile_requests=216,qualification_control_cells=9,protocol_sha256=sha(R/'protocol.json'),manifest_sha256=sha(R/'manifest_identity.json')))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(R/'controller_stopped.json',dict(passed=False,error_class=type(e).__name__,reason=str(e)));raise
