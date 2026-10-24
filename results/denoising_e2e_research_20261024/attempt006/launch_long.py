"""Same finite long panel, with an explicit four-hour no-progress watchdog."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    here=Path(__file__).resolve().parent
    root=here.parent/'attempt003'
    protocol=json.loads((root/'e2e_protocol.json').read_text())
    binding=json.loads((here/'execution_binding.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=binding['launcher_sha256']:
        raise RuntimeError('long launcher source drift')
    if hashlib.sha256((root/'e2e_protocol.json').read_bytes()).hexdigest()!=binding['parent_protocol_sha256']:
        raise RuntimeError('parent protocol drift')
    for p,h in binding['native_method_sources'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('selector source drift')
    for arm in [a for a in protocol['arms'] if a!='v31_optimized']:
        parent=root/'launches/long'/arm
        parent.mkdir(parents=True,exist_ok=False)
        for f,h in protocol['sources'].items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise RuntimeError('source drift '+f)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        if apps.strip():raise RuntimeError('GPU occupied; no preemption')
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('MAGE_','VALUE_','FORCE_','RESIDENT_','OPTIMIZED_')):env.pop(key)
        overrides=dict(PYTHONPATH='/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm',
            HF_HUB_OFFLINE='1',TORCHINDUCTOR_DETERMINISTIC='1',VLLM_USE_V2_MODEL_RUNNER='1',
            VLLM_ENABLE_V1_MULTIPROCESSING='0',MAX_MODEL_LEN='262144',
            RESIDENT_SPARSE_EXECUTION='1',RESIDENT_V31_PAGED_REFRESH='1')
        for variable,leaf in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
            overrides[variable]=str(root/'cache/long'/leaf)
        env.update(overrides)
        command=['/home/exouser/dyh/v31_env_20261005/bin/python',str(here/'code/long_worker.py'),
            '--root',str(root),'--arm',arm,'--suite','long','--stage','panel']
        started=time.time();binding_hash=hashlib.sha256((here/'execution_binding.json').read_bytes()).hexdigest()
        with (parent/'launch_receipt.json').open('x') as f:json.dump(dict(command=command,
            environment_overrides=overrides,source_checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            protocol_sha256=binding['parent_protocol_sha256'],execution_binding_sha256=binding_hash,
            gpu_idle_before=True,started_unix=started,campaign_budget=None,
            watchdog='14400seconds without native forward progress; active inference has no wall allocation cap'),f,indent=2)
        log=root/'private/logs/long'/(arm+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        heartbeat=root/'private'/('long_heartbeat_'+arm+'.json')
        signals=[]
        with log.open('x') as stream:
            process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            identity=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            with (parent/'process_identity.json').open('x') as f:json.dump(dict(pid=process.pid,
                process_group=os.getpgid(process.pid),start_ticks=identity,dedicated_session=True,
                command=command),f,indent=2)
            while process.poll() is None:
                last=heartbeat.stat().st_mtime if heartbeat.exists() else started
                if time.time()-last>binding['watchdog']['no_progress_seconds']:
                    # Target only this verified dedicated process group.
                    if Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]!=identity:
                        raise RuntimeError('worker identity changed')
                    os.killpg(process.pid,signal.SIGTERM);signals.append('SIGTERM_no_forward_progress_4h')
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);signals.append('SIGKILL_after_15s')
                    break
                time.sleep(15)
            process.wait()
        receipt=dict(returncode=process.returncode,allocated_worker_seconds=time.time()-started,
            status='worker_completed' if process.returncode==0 else 'failed_preserved',
            arm=arm,suite='long',watchdog_signals=signals,execution_binding_sha256=binding_hash)
        with (parent/'exit_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(receipt),flush=True)
        if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
