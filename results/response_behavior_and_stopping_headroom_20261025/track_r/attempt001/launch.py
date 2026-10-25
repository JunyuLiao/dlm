"""Persistent one-H100 controller; scientific gates, no resource/watchdog ceiling."""
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
    root=Path(__file__).resolve().parent;repo=root.parents[3]
    protocol=json.loads((root/'protocol.json').read_text());binding=json.loads((root/'binding_receipt.json').read_text());paths=json.loads((root/'private/runtime_paths.json').read_text())
    private=root/'private/logs';private.mkdir(exist_ok=True)
    def run(name,args,gpu=False):
        for p,h in protocol['sources'].items():assert sha(repo/p)==h,p
        for p,h in json.loads((root/'private/external_sources.json').read_text()).items():assert sha(Path(p))==h
        for p,h in json.loads((root/'private/scorer_sources.json').read_text()).items():assert sha(Path(p))==h
        if gpu:assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied; no preemption'
        leaf=root/'launches'/name;leaf.mkdir(parents=True,exist_ok=False)
        env=os.environ.copy()
        for n in list(env):
            if n.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(n)
        env.update(PYTHONPATH=str(repo/'src')+':'+str(repo),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='12288')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        for key,val in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:env[key]=str(root/'cache'/val)
        start=time.time();command=[paths['python'],*args]
        save(leaf/'launch_receipt.json',dict(command=['<pinned_python>',*args],started_unix=start,protocol_sha256=sha(root/'protocol.json'),
            source_checkpoint=binding['source_checkpoint'],checkout_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),GPU=gpu,GPU_idle_before=gpu,no_resource_ceilings=True))
        with (private/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            save(leaf/'process_identity.json',dict(pid=process.pid,start_ticks=ticks,dedicated_worker=gpu))
            while process.poll() is None:
                (root/'private/controller_heartbeat.json').write_text(json.dumps(dict(job=name,child_pid=process.pid,unix=time.time())))
                time.sleep(15)
        save(leaf/'exit_receipt.json',dict(returncode=process.returncode,allocated_seconds=time.time()-start,GPU=gpu))
        print(json.dumps(dict(job=name,returncode=process.returncode,allocated_seconds=time.time()-start)),flush=True)
        if process.returncode:raise RuntimeError('stopped after failed job '+name)
    try:
        for arm in protocol['legacy_arms']:
            run('legacy_'+arm,[str(root/'code/worker.py'),'--root',str(root),'--phase','legacy','--benchmark','all','--arm',arm],True)
            run('score_legacy_'+arm,[str(root/'code/scoring.py'),'--root',str(root),'--phase','legacy','--benchmark','all','--arm',arm])
        run('legacy_gate',[str(root/'code/legacy_gate.py')])
        for task in ['aime26','humaneval']:
            for arm in protocol['arms']:
                name='panel_'+task+'_'+arm
                run(name,[str(root/'code/worker.py'),'--root',str(root),'--phase','panel','--benchmark',task,'--arm',arm],True)
                run('score_'+name,[str(root/'code/scoring.py'),'--root',str(root),'--phase','panel','--benchmark',task,'--arm',arm])
        run('analyze',[str(root/'code/analyze.py')])
        save(root/'execution_complete.json',dict(passed=True,Track_R_complete=True,Track_H_launched=False,Stage_BC_launched=False))
    except BaseException as error:
        save(root/'controller_stopped.json',dict(passed=False,error_class=type(error).__name__,reason=str(error),Track_H_launched=False,no_budget_stop=True))
        raise


if __name__=='__main__':main()
