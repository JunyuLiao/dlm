"""Queue engineering probes after the exclusive final-evaluation GPU worker.

No status busy-polling. A filesystem lock waits for the preceding worker, then
independent probes run serially and preserve individual failures and logs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch(bundle,final):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');bundle=bundle.resolve();final=final.resolve()
    log=bundle/f'postrun_{stamp}.log'
    with log.open('xb') as stream:
        p=subprocess.Popen([sys.executable,'-m','experiments.value_direction_hopper.postrun','--bundle',str(bundle),'--final',str(final)],
            cwd=Path(__file__).resolve().parents[2],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH='src:.',OMP_NUM_THREADS='4'),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print(json.dumps(dict(pid=p.pid,log=str(log))),flush=True)


def run(bundle,final):
    from .experiment import atomic
    bundle=bundle.resolve();final=final.resolve();config=json.loads((final/'configuration.json').read_text())
    kernel=config['library'];bridge=config['torch_library'];states=str(bundle/'native_qkv_v2')
    prefix=[sys.executable,'-m','experiments.value_direction_hopper.']
    def module(name,*args):return [sys.executable,'-m','experiments.value_direction_hopper.'+name,*map(str,args)]
    tasks=[('native_microbenchmark',bundle/'microbench_native_v1.json',module('microbench','--library',kernel,'--bridge',bridge,'--states',states,'--output',bundle/'microbench_native_v1.json')),
        ('resources',bundle/'resource_limits_v1.json',module('resources','--library',bundle/'build/resources_7ecb74680a0b5f7b.so','--output',bundle/'resource_limits_v1.json')),
        ('length_sweep',bundle/'length_sweep_v1.json',module('length_sweep','--library',kernel,'--bridge',bridge,'--output',bundle/'length_sweep_v1.json')),
        ('nsight_systems',bundle/'nsys_native_v1.nsys-rep',['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','--output',str(bundle/'nsys_native_v1'),
            *module('nsys_probe','--library',kernel,'--bridge',bridge,'--states',states)])]
    results=[]
    with (bundle/'postrun.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(bundle/'postrun_status.json',dict(status='waiting_for_final_worker',pid=os.getpid(),updated=time.time()))
        with (final/'worker.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            state=json.loads((final/'status.json').read_text())
            if state.get('status')!='complete':raise RuntimeError('Final worker is not complete; preserve pending work and resume it first')
            for name,target,command in tasks:
                if target.exists():results.append(dict(name=name,status='cached',output=str(target)));continue
                atomic(bundle/'postrun_status.json',dict(status='running',pid=os.getpid(),task=name,completed=results,updated=time.time()))
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');log=bundle/f'{name}_{stamp}.log'
                with log.open('xb') as stream:
                    proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                record=dict(name=name,returncode=proc.returncode,output=str(target),exists=target.exists(),log=str(log));results.append(record)
                print(json.dumps(record),flush=True)
            atomic(bundle/'postrun_status.json',dict(status='complete' if all(r.get('returncode',0)==0 and r.get('exists',True) for r in results) else 'complete_with_failures',results=results,updated=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--detach',action='store_true')
    a=p.parse_args();(launch if a.detach else run)(a.bundle,a.final)
