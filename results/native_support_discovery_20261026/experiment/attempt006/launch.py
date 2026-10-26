"""Serial qualified discovery; no campaign ceilings, timeout, or unrelated resume."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;S=R.parents[1];P=S/'profiling/attempt002';REPO=R.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def checkpoint(paths,message):
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip(),'Unrelated staged files; preserve them'
    subprocess.run(['git','add','--',*[str(p) for p in paths]],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m',message],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
def main():
    paths=json.loads((S/'private/runtime_paths.json').read_text());protocol=json.loads((R/'protocol.json').read_text());binding=json.loads((R/'binding.json').read_text())
    while not (P/'execution_complete.json').exists():
        if (P/'controller_stopped.json').exists():raise RuntimeError('Profiling did not qualify; preserve blocked discovery')
        (R/'private/controller_heartbeat.json').parent.mkdir(parents=True,exist_ok=True);(R/'private/controller_heartbeat.json').write_text(json.dumps(dict(stage='waiting_for_frozen_baseline_cohort',unix=time.time())));time.sleep(10)
    assert json.loads((P/'execution_complete.json').read_text())['passed']
    save(R/'cohort_binding.json',dict(profiling_manifest_sha256=sha(P/'manifest_identity.json'),profiling_manifest_binding_sha256=sha(P/'manifest_binding.json'),private_manifest_sha256=sha(P/'private/manifest.json'),profiling_completion_sha256=sha(P/'execution_complete.json'),protocol_sha256=sha(R/'protocol.json')))
    def run(name,args,gpu):
        for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
        if gpu:assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; no preemption'
        env=os.environ.copy()
        for n in list(env):
            if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
        env.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        for n,v in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions'),('MPLCONFIGDIR','matplotlib')]:env[n]=str(R/'cache'/v)
        leaf=R/'launches'/name;leaf.mkdir(parents=True,exist_ok=False);logs=R/'private/logs';logs.mkdir(parents=True,exist_ok=True);start=time.time()
        save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',*args],GPU=gpu,started_unix=start,source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(R/'protocol.json'),no_campaign_timeout=True))
        with (logs/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([paths['python'],*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(leaf/'process_identity.json',dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
            while proc.poll() is None:
                (R/'private/controller_heartbeat.json').write_text(json.dumps(dict(stage=name,pid=proc.pid,unix=time.time())));time.sleep(10)
        save(leaf/'exit_receipt.json',dict(GPU=gpu,returncode=proc.returncode,allocated_seconds=time.time()-start))
        if proc.returncode:raise RuntimeError('Preserved failed job '+name)
    run('environment_verification',[str(R/'verify_environment.py')],False)
    reuse=json.loads((R/'reference_binding.json').read_text())
    for name,h in reuse['input_hashes'].items():assert sha(R/name)==h,name
    for name,h in reuse['unchanged_GPU_source_hashes'].items():assert sha(R/'code'/name)==h,name
    assert json.loads((R/'runs/references/final_complete.json').read_text())['passed']
    run('independent_numerical',[str(R/'code/numerical_verify.py')],False)
    assert json.loads((R/'numerical_qualification.json').read_text())['passed']
    assert json.loads((R/'runs/early/final_complete.json').read_text())['passed']
    assert sha(R/'branch_manifest.json')==reuse['original_branch_manifest_sha256']
    point=checkpoint([R/'banks',R/'early_maps.json',R/'branch_manifest.json',R/'numerical_qualification.json',R/'cohort_binding.json'],'Freeze current-only support banks and exact discovery branches before sparse outcomes')
    save(R/'branch_binding.json',dict(checkpoint=point,branch_manifest_sha256=sha(R/'branch_manifest.json'),protocol_sha256=sha(R/'protocol.json'),private_data_in_git=False))
    run('treatments',[str(R/'code/study_worker.py'),'--job','treatments'],True)
    run('analysis',[str(R/'code/analyze.py')],False)
    a=json.loads((R/'analysis001/summary.json').read_text())
    save(R/('development_complete.json' if not a['any_rule_advances'] else 'confirmation_required.json'),dict(passed=True,protocol_sha256=sha(R/'protocol.json'),analysis_sha256=sha(R/'analysis001/summary.json'),scope='Complete finite development diagnostics; conditional confirmation pending only if a gate passes',any_rule_advances=a['any_rule_advances']))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(R/'controller_stopped.json',dict(passed=False,error_class=type(e).__name__,reason=str(e)));raise
