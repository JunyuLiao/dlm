"""One-H100 H controller; refuses to overlap the unfrozen R study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)


def main():
    root=Path(__file__).resolve().parent;repo=root.parents[3];study=root.parents[1]
    assert (study/'track_r/attempt001/readout_frozen.json').exists()
    assert json.loads((study/'track_r/attempt001/execution_complete.json').read_text())['passed']
    protocol=json.loads((root/'protocol.json').read_text());paths=json.loads((root/'private/runtime_paths.json').read_text());binding=json.loads((root/'binding_receipt.json').read_text())
    assert sha(study/'track_r/attempt001/readout_frozen.json')==protocol['R_readout_sha256']
    assert sha(root/'protocol.json')==binding['protocol_sha256']
    (root/'private/logs').mkdir(exist_ok=True)
    def run(job,args,gpu):
        for p,h in protocol['sources'].items():assert sha(repo/p)==h,p
        for p,h in json.loads((root/'private/external_sources.json').read_text()).items():assert sha(Path(p))==h
        if gpu:assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        leaf=root/'launches'/job;leaf.mkdir(parents=True,exist_ok=False)
        env=os.environ.copy()
        for n in list(env):
            if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
        env.update(PYTHONPATH=str(repo/'src')+':'+str(repo),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        for n,v in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:env[n]=str(root/'cache'/v)
        start=time.time();save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',*args],source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(root/'protocol.json'),started_unix=start,GPU=gpu,no_resource_ceilings=True))
        with (root/'private/logs'/(job+'.log')).open('x') as log:
            p=subprocess.Popen([paths['python'],*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            save(leaf/'process_identity.json',dict(pid=p.pid,start_ticks=ticks))
            while p.poll() is None:
                (root/'private/controller_heartbeat.json').write_text(json.dumps(dict(job=job,pid=p.pid,unix=time.time())))
                time.sleep(15)
        save(leaf/'exit_receipt.json',dict(returncode=p.returncode,allocated_seconds=time.time()-start,GPU=gpu))
        print(json.dumps(dict(job=job,returncode=p.returncode)),flush=True)
        if p.returncode:raise RuntimeError('H job failed: '+job)
    try:
        for job in ['full','references','development']:run(job,[str(root/'code/worker.py'),'--root',str(root),'--job',job],True)
        run('decide',[str(root/'code/decide.py')],False)
        if json.loads((root/'development_decision.json').read_text())['confirmation_authorized']:
            run('confirmation',[str(root/'code/worker.py'),'--root',str(root),'--job','confirmation'],True)
        run('analyze',[str(root/'code/analyze.py')],False)
        save(root/'execution_complete.json',dict(passed=True,H_gated_leaf_complete=True,Stage_BC=False,production_optimization=False))
    except BaseException as error:
        save(root/'controller_stopped.json',dict(passed=False,error_class=type(error).__name__,reason=str(error),no_budget_stop=True));raise


if __name__=='__main__':main()
