"""One worker, fixed scientific gate order, no campaign resource ceiling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root=Path(__file__).resolve().parent;repo=root.parents[2]
    protocol=json.loads((root/'protocol.json').read_text())
    paths=json.loads((root/'private/runtime_paths.json').read_text());py=paths['python']
    for path,h in protocol['sources'].items():assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==h,path
    private=root/'private/logs';private.mkdir(exist_ok=True)
    for job in ['smoke','full','a1','analysis']:
        directory=root/'launches'/job;directory.mkdir(parents=True,exist_ok=False)
        if job!='analysis':
            clients=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
            assert not clients.strip(),'GPU occupied; no preemption'
        env=os.environ.copy()
        for name in list(env):
            if name.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(name)
        env.update(PYTHONPATH=str(repo/'src')+':'+str(repo),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
            TORCHINDUCTOR_DETERMINISTIC='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_USE_V2_MODEL_RUNNER='1',MAX_MODEL_LEN='262144')
        for key,leaf in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
            env[key]=str(root/'cache'/leaf)
        command=[py,str(root/'code/worker.py'),'--root',str(root),'--job',job]
        if job=='analysis':
            env['CUDA_VISIBLE_DEVICES']='';command=[py,str(root/'code/analyze.py')]
        started=time.time()
        with (directory/'launch_receipt.json').open('x') as f:json.dump(dict(job=job,started_unix=started,
            protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),source_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            command=['<pinned_python>',*command[1:]],no_campaign_ceiling=True,GPU_idle_before=job!='analysis'),f,indent=2)
        with (private/(job+'.log')).open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            with (directory/'process_identity.json').open('x') as f:json.dump(dict(pid=process.pid,start_ticks=identity,job=job,dedicated_process=True),f,indent=2)
            while process.poll() is None:time.sleep(15)
        receipt=dict(job=job,returncode=process.returncode,allocated_worker_seconds=time.time()-started,no_campaign_ceiling=True)
        with (directory/'exit_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(receipt),flush=True)
        if process.returncode:raise SystemExit(process.returncode)
    result=json.loads((root/'analysis001/summary.json').read_text())
    with (root/'execution_complete.json').open('x') as f:json.dump(dict(passed=True,A1_complete=True,A2_qualified=result['A2_qualified'],
        A2_launched=False,next_action='A2 needs new implementation/source binding if qualified, otherwise close study',stage_BC_launched=False),f,indent=2)


if __name__=='__main__':main()
