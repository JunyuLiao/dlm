"""Recover only the unstarted frozen jobs after the launcher's parent exited.

Never resumes or relaunches a worker. The already running FULL worker is observed
through its pinned PID/start-time identity and its completion marker.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent / 'attempt002'
REPO = ROOT.parents[2]


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(identity):
    path = Path(f'/proc/{identity["pid"]}/stat')
    if not path.exists():
        return False
    fields = path.read_text().rsplit(')', 1)[1].split()
    return fields[0] != 'Z' and fields[19] == identity['start_ticks']


def main():
    recovery = ROOT / 'controller_recovery001'
    recovery.mkdir(exist_ok=False)
    save(recovery / 'receipt.json', dict(
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(ROOT / 'protocol.json'),
        observed_unix=time.time(), reason='interactive-turn interruption ended launcher parent; FULL worker survived',
        worker_relaunched=False, scientific_schedule_changed=False,
        remaining_jobs=['a1', 'analysis'], no_campaign_ceiling=True))
    identity = json.loads((ROOT / 'launches/full/process_identity.json').read_text())
    while alive(identity):
        time.sleep(15)
    marker = ROOT / 'executions/full/final_complete.json'
    complete = json.loads(marker.read_text()) if marker.exists() else {}
    launch = json.loads((ROOT / 'launches/full/launch_receipt.json').read_text())
    save(recovery / 'full_observed_exit.json', dict(
        observed_unix=time.time(), passed=complete.get('passed', False),
        exit_code_available=False, completion_marker_sha256=sha(marker) if marker.exists() else None,
        launch_to_observed_exit_upper_bound_seconds=time.time()-launch['started_unix'],
        worker_reported_allocation_seconds=complete.get('allocation_seconds'),
        observer_poll_seconds=15, original_identity=identity))
    if not complete.get('passed', False):
        raise SystemExit('FULL worker did not complete; no subsequent GPU job launched')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    paths=json.loads((ROOT/'private/runtime_paths.json').read_text())
    for path, value in protocol['sources'].items():
        assert sha(REPO/path)==value, path
    for job in ['a1', 'analysis']:
        directory=ROOT/'launches'/job
        directory.mkdir(parents=True,exist_ok=False)
        if job!='analysis':
            clients=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
            assert not clients.strip(), 'GPU occupied; no preemption'
        env=os.environ.copy()
        for name in list(env):
            if name.startswith(('MAGE_', 'VALUE_', 'FORCE_', 'RESIDENT_', 'OPTIMIZED_')):
                env.pop(name)
        env.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO), HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
            TORCHINDUCTOR_DETERMINISTIC='1', VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_V2_MODEL_RUNNER='1', MAX_MODEL_LEN='262144')
        for key, leaf in [('VLLM_CACHE_ROOT','vllm'), ('TRITON_CACHE_DIR','triton'), ('TORCHINDUCTOR_CACHE_DIR','inductor'), ('CUDA_CACHE_PATH','cuda')]:
            env[key]=str(ROOT/'cache'/leaf)
        command=[paths['python'],str(ROOT/'code/worker.py'),'--root',str(ROOT),'--job',job]
        if job=='analysis':
            env['CUDA_VISIBLE_DEVICES']=''
            command=[paths['python'],str(ROOT/'code/analyze.py')]
        started=time.time()
        save(directory/'launch_receipt.json', dict(job=job,started_unix=started,protocol_sha256=sha(ROOT/'protocol.json'),
            source_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            command=['<pinned_python>',*command[1:]], no_campaign_ceiling=True,GPU_idle_before=job!='analysis',
            controller_source_sha256=sha(Path(__file__)),recovery_receipt='controller_recovery001/receipt.json'))
        with (ROOT/'private/logs'/(job+'.log')).open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            save(directory/'process_identity.json',dict(pid=process.pid,start_ticks=ticks,job=job,dedicated_process=True))
            while process.poll() is None:
                time.sleep(15)
        receipt=dict(job=job,returncode=process.returncode,allocated_worker_seconds=time.time()-started,no_campaign_ceiling=True)
        save(directory/'exit_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    result=json.loads((ROOT/'analysis001/summary.json').read_text())
    save(ROOT/'execution_complete.json',dict(passed=True,A1_complete=True,A2_qualified=result['A2_qualified'],A2_launched=False,
        next_action='A2 needs new implementation/source binding if qualified, otherwise close study',stage_BC_launched=False))


if __name__ == '__main__':
    main()
