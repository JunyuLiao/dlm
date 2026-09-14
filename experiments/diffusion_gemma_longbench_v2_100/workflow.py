"""Isolated resumable900-result experiment; one H100,15-minute supervision."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from experiments.diffusion_gemma_value_aware.protocol import frozen_write,sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware_gpu.workflow import equivalent
from .protocol import ROOT,MODEL,REVISION,METHODS,TARGETS,prepare,execution
from .runner import cached
from .validation import smoke
from .calibration import prepare_policies,condition
from . import nemo


def failure(root,stage,**details):
    error=dict(stage=stage,**details,traceback=traceback.format_exc())
    _append(root/'failures.jsonl',error);print(error,flush=True);return error


def dense_gate(outputs):
    graded=nemo.evaluate([dict(generation=o['prediction'],expected_answer='A') for o in outputs])
    # This is a completion/format sanity check, NOT accuracy-based selection.
    limited=sum(o['termination_reason']=='length' for o in outputs)
    parsed=sum(g['predicted_answer'] in tuple('ABCD') for g in graded)
    return dict(passed=len(outputs)==100 and limited<=5 and parsed>=90,count=len(outputs),
        length_limited=limited,parsed=parsed,criterion='100 cached dense outputs; at most5 length limits; at least90 parsable; no correctness criterion')


def freeze_conditions(root,setup,contract):
    proof=json.loads((root/'validation/smoke.json').read_text())
    if not proof['passed'] or proof['fingerprint']!=contract['fingerprint'] or len(proof['cases'])!=18:
        raise ValueError('Complete matching18-case CUDA smoke required')
    if not json.loads((root/'dense_gate.json').read_text())['passed']:raise ValueError('Dense completion gate failed')
    pilot=json.loads((root/'pilot/decision.json').read_text())
    paths=[Path(__file__),Path(__file__).with_name('calibration.py'),Path(__file__).with_name('validation.py'),
           root/'execution_contract.json',root/'setup.json',root/'validation/smoke.json',root/'dense_gate.json',root/'calibration_protocol.json']
    cal=json.loads((root/'calibration_protocol.json').read_text())
    common={**contract['sources'],**setup['sources'],**proof['sources'],**pilot['sources'],**cal['sources'],
            **{str(p):sha(p.read_bytes()) for p in paths}}
    check_sources(common)
    result={}
    for name,target in [('dense',0.)]+[(n,t) for t in TARGETS for n in METHODS]:
        label='dense' if name=='dense' else f'{name}_s{int(target*100)}'
        try:
            c=condition(root,setup,contract,name,target)
            sources=dict(common)
            for source in c['policy_sources'].values():sources[source['path']]=source['sha256']
            c.update(fingerprint=contract['fingerprint'],sources=sources,expected_per_benchmark=dict(longbench_v2=100))
            frozen_write(root/'final_configs'/f'{label}.json',c);result[label]=c
        except Exception:failure(root,'freeze_condition',condition=label)
    return result


def work(root):
    import torch
    from dllm.models import create_adapter
    setup=prepare(root);contract=execution(root,freeze=True)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'runtime.json',dict(model=adapter.model.config.to_dict(),generation=adapter.model.generation_config.to_dict(),
        runtime=contract['runtime'],fingerprint=contract['fingerprint']))
    _write(root/'phase.json',dict(stage='validation',started=time.time()))
    smoke(adapter,root,setup,contract)
    errors=[];dense=[]
    _write(root/'phase.json',dict(stage='dense',expected=100,started=time.time()))
    ordered=sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id']))
    for row in ordered:
        try:dense.append(cached(adapter,root,row,'dense','dense',{},None,contract))
        except Exception:
            errors.append(failure(root,'dense',id=row['id']));torch.cuda.empty_cache()
    gate=dense_gate(dense);_write(root/'dense_gate.json',gate);del dense
    if not gate['passed']:
        _write(root/'run_terminal.json',dict(complete=False,stage='dense_completion_gate',gate=gate,finished=time.time()))
        raise RuntimeError('Dense completion gate failed; preserve all outputs and investigate before a misleading full sweep')
    _write(root/'phase.json',dict(stage='calibration',started=time.time()))
    errors.extend(prepare_policies(adapter,root,setup,contract))
    available=freeze_conditions(root,setup,contract)
    for label,c in available.items():
        if label=='dense':continue
        _write(root/'phase.json',dict(stage='final',condition=label,expected=100,started=time.time()))
        for row in ordered:
            try:
                path=shard_path(root,'final',label,row['id'])
                if path.exists():cached(None,root,row,'final',label,c['config'],c['thresholds']['longbench_v2'],contract)
                elif equivalent(root,row,label,c,available,contract) is None:
                    cached(adapter,root,row,'final',label,c['config'],c['thresholds']['longbench_v2'],contract)
            except Exception:
                errors.append(failure(root,'final',condition=label,id=row['id']));torch.cuda.empty_cache()
    del adapter;torch.cuda.empty_cache()
    from .report import regenerate
    _write(root/'phase.json',dict(stage='report',started=time.time()))
    audit=regenerate(root)
    _write(root/'run_terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=900,
        failed_attempts=len(errors),finished=time.time()))
    if not audit['complete']:raise RuntimeError('Some configurations failed; successful independent results remain cached')


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
        command=[sys.executable,'-u','-m',__package__+'.workflow','work','--output',str(root)]
        with (root/'run.log').open('a',buffering=1) as log:
            worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=worker.pid,supervisor_pid=os.getpid(),started=time.time(),command=command))
            while True:
                try:code=worker.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    snapshot={}
                    for key in ('phase','progress'):
                        p=root/f'{key}.json';snapshot[key]=json.loads(p.read_text()) if p.exists() else None
                    _append(root/'monitor.jsonl',dict(time=time.time(),pid=worker.pid,alive=True,gpu=gpu.stdout.strip(),
                        gpu_error=gpu.stderr.strip(),disk_free_bytes=shutil.disk_usage(root).free,**snapshot))
            _write(root/'supervisor_terminal.json',dict(exit_code=code,pid=worker.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    prepare(root)
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        p=subprocess.Popen([sys.executable,'-u','-m',__package__+'.workflow','supervise','--output',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=p.pid,status='launched; verify GPU process and job.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','launch','supervise','work','report'))
    p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args()
    if a.command=='report':
        from .report import regenerate
        print(json.dumps(regenerate(a.output)))
    elif a.command=='prepare':print(json.dumps({k:v for k,v in prepare(a.output).items() if k not in ('final','calibration','development','sources')}))
    else:globals()[a.command](a.output)


if __name__=='__main__':main()
