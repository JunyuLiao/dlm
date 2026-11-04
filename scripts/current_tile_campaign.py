"""Serial nine-panel coordinator with physical-work-only recalibration."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1]
ORDER=[('llada22','humaneval'),('trado','humaneval'),('trado','ruler32k'),
       ('llada22','ruler128k'),('trado','aime26'),('llada22','aime26'),
       ('diffusiongemma','humaneval'),('diffusiongemma','aime26'),('diffusiongemma','ruler128k')]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def state(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(v)+'\n');temp.replace(p)


def prepare(root):
    write(root/'protocol.json',{'status':'frozen','date':'2026-11-03','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'panels':[{'model':m,'suite':s,'requests':90 if s=='aime26' else 164 if s=='humaneval' else 130} for m,s in ORDER],
      'spec_sha256':sha(REPO/'docs/CURRENT_TILE_MULTIMODEL_20261103.md'),
      'hf_worker_sha256':sha(REPO/'scripts/current_tile_multimodel_worker.py'),
      'single_tile_selector_sha256':sha(REPO/'experiments/numerical_qk_reuse/v2_current_tile_init.py'),
      'multiple_tile_selector_sha256':sha(REPO/'experiments/numerical_qk_reuse/v2_current_canvas_init.py'),
      'qualification_source_sha256':sha(REPO/'scripts/current_tile_campaign_qualify.py'),
      'per_panel_source_freeze':'committed source hashes and input/model identity in each attempt before generation',
      'calibration_target':[.48,.52],'full_benchmark_acceptance':[.4,.6],'recalibrate_outside':True,
      'small_calibration_only':True,'accuracy_blinded':True,'monitor_seconds':900,'gpu_hour_limit':None,
      'native_samplers_unchanged':True,'other_methods':'historical evidence read-only',
      'ruler_lengths':{'llada22':131072,'trado':32768,'diffusiongemma':131072}})


def run(root):
    if (root/'USER_SCOPE_CHANGE_20261104.json').exists():
        raise RuntimeError('Original nine-panel coordinator stopped by user; see USER_SCOPE_CHANGE_20261104.json')
    p=read(root/'protocol.json')
    assert p['hf_worker_sha256']==sha(REPO/'scripts/current_tile_multimodel_worker.py')
    assert p['single_tile_selector_sha256']==sha(REPO/'experiments/numerical_qk_reuse/v2_current_tile_init.py')
    assert p['multiple_tile_selector_sha256']==sha(REPO/'experiments/numerical_qk_reuse/v2_current_canvas_init.py')
    amendment=root/'execution_amendment001.json'
    binding=read(amendment) if amendment.exists() else None
    if binding:
        assert binding['worker_sha256']==sha(REPO/'scripts/current_tile_multimodel_worker_qualified.py')
    recovery_file=root/'maintenance_recovery_20261104.json'
    recovery=read(recovery_file) if recovery_file.exists() else None
    if recovery:
        assert recovery['worker_sha256']==sha(REPO/'scripts/current_tile_multimodel_recovery.py')
    stop=threading.Event();active={'phase':'qualification','pid':os.getpid()}
    def monitor():
        while True:
            g=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
            v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**active,'returncode':g.returncode,
               'nvidia_smi':g.stdout,'stderr':g.stderr,'benchmark_record_files':len(list(root.glob('panels/**/records.jsonl')))}
            folder=root/'private/health';folder.mkdir(parents=True,exist_ok=True)
            with (folder/'gpu_health.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
            if stop.wait(900):return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def child(command,log):
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if gpu.stdout.strip():raise RuntimeError('GPU occupied: '+gpu.stdout.strip())
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as f:
            proc=subprocess.Popen(command,cwd=REPO,env=dict(os.environ,PYTHONPATH='src:.'),stdout=f,stderr=subprocess.STDOUT)
            active.update(worker_pid=proc.pid)
            state(root/'private/active_worker.json',active)
            code=proc.wait()
        if code:raise RuntimeError(f'worker exit {code}; inspect {log.relative_to(root)}')
    try:
        gate=root/('qualification/cuda_receipt_20261104.json' if recovery else 'qualification/cuda_receipt.json')
        if not gate.exists():child([sys.executable,str(REPO/'scripts/current_tile_campaign_qualify.py'),str(gate)],root/('private/logs/qualification_20261104.log' if recovery else 'private/logs/qualification.log'))
        assert read(gate)['status']=='pass'
        for model,suite in ORDER:
            if model=='trado' and suite=='ruler32k' or model=='diffusiongemma':
                active.update(model=model,suite=suite,phase='waiting_data')
                while not (root/'data_preparation_complete.json').exists():time.sleep(30)
            parent=root/'panels'/model/suite
            done=[f for f in parent.glob('attempt*/generation_complete.json') if read(f).get('accepted')]
            if done:continue
            attempt=1+max([int(d.name[-3:]) for d in parent.glob('attempt*')],default=0)
            while True:
                active.update(model=model,suite=suite,attempt=attempt,phase='calibration_and_generation')
                script='current_tile_gemma_panel.py' if model=='diffusiongemma' else ('current_tile_multimodel_worker_qualified.py' if binding else 'current_tile_multimodel_worker.py')
                if recovery and model!='diffusiongemma':script='current_tile_multimodel_recovery.py'
                command=[sys.executable,str(REPO/'scripts'/script),str(root),model,suite,'--attempt',str(attempt)]
                log=root/'private/logs'/model/suite/f'attempt{attempt:03}.log'
                child(command,log)
                result=read(parent/f'attempt{attempt:03}'/'generation_complete.json')
                print(json.dumps({'event':'panel_finished',**result}),flush=True)
                if result['accepted']:break
                attempt+=1
        active.update(phase='official_scoring')
        child([sys.executable,str(REPO/'scripts/current_tile_campaign_report.py'),str(root)],root/'private/logs/scoring.log')
        assert read(root/'final_complete.json')['status']=='complete'
    finally:
        stop.set();thread.join(timeout=35)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('root',type=Path)
    a=p.parse_args();(prepare if a.mode=='prepare' else run)(a.root.resolve())
