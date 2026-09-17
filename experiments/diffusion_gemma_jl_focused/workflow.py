"""One H100, resumable focused sweep and automatic 15-minute health records."""
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

import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_jl_output_aware.runner import cached
from experiments.diffusion_gemma_jl_output_aware.calibration import fit_one
from .protocol import ROOT,MODEL,REVISION,BASELINES,PROJECTED,CONFIGS,TARGETS,prepare,execution,read,sha,frozen_write
from .reuse import dispatch,import_policies,import_baselines,import_shared,contribution_screen,smoke,diagnostics,audit_policy
from .gate import require as require_tests


def phase(root,stage,**details):
    _write(root/'phase.json',dict(stage=stage,started=time.time(),**details))
    print('phase',stage,details,flush=True)


def failure(root,stage,**details):
    record=dict(stage=stage,**details,traceback=traceback.format_exc())
    _append(root/'failures.jsonl',record); print(record,flush=True)
    torch.cuda.empty_cache()
    return record


def freeze_conditions(root,setup,contract):
    proof=root/'validation'/contract['fingerprint']/'smoke.json'
    validated=read(proof)
    if not validated['passed'] or len(validated['cases'])!=12:
        raise ValueError('All three routers require two-example CUDA smoke')
    common={**contract['sources'],**setup['source_hashes'],**require_tests(root,contract)['sources'],
        str(root/'test_gate.json'):sha((root/'test_gate.json').read_bytes()),str(proof):sha(proof.read_bytes()),
        str(root/'inherited_tensor_validation.json'):sha((root/'inherited_tensor_validation.json').read_bytes())}
    check_sources(common)
    conditions={}
    for name,target in [('dense',0.)]+[(n,t) for t in TARGETS for n in CONFIGS]:
        label='dense' if name=='dense' else f'{name}_s{int(100*target)}'
        config={} if name=='dense' else CONFIGS[name]
        thresholds={};sources=dict(common);missing=[]
        for benchmark in ('aime26','longbench_v2'):
            if name=='dense': thresholds[benchmark]=None;continue
            p=root/'policies'/benchmark/f'{label}.json'
            if not p.exists():missing.append(benchmark);continue
            policy=read(p);audit_policy(root,policy,setup,contract)
            if policy['config']!=config or policy['target']!=target or policy['name']!=name:
                raise ValueError('Policy does not match final condition')
            thresholds[benchmark]=policy['policy'];sources[str(p)]=sha(p.read_bytes())
        c=dict(fingerprint=contract['fingerprint'],name=name,target=target,config=config,
            thresholds=thresholds,sources=sources,unavailable_benchmarks=missing,
            target_metric='physical_sparsity',expected_per_benchmark=dict(aime26=30,longbench_v2=50))
        if not missing:
            frozen_write(root/'final_configs'/f'{label}.json',c)
        else:
            _write(root/'pending_configs'/f'{label}.json',c)
        conditions[label]=c
    return conditions


def work(root):
    torch.backends.cuda.matmul.allow_tf32=False
    setup=prepare(root);contract=execution(root,freeze=True)
    require_tests(root,contract)
    failures=[]
    phase(root,'reuse_verified_calibration')
    import_policies(root,setup,contract)
    phase(root,'reuse_baselines',expected=560)
    import_baselines(root,setup,contract)
    sources=import_shared(root,setup,contract)
    phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'runtime.json',dict(model=adapter.model.config.to_dict(),generation=adapter.model.generation_config.to_dict(),
        runtime=contract['runtime'],fingerprint=contract['fingerprint']))
    phase(root,'two_example_cuda_smoke')
    smoke(adapter,root,setup,contract)
    with dispatch():
        phase(root,'contribution32_shared_proposals',expected=len(sources))
        contribution_screen(root,sources,contract)
        phase(root,'contribution32_calibration')
        for target in TARGETS:
            for benchmark in ('aime26','longbench_v2'):
                try:
                    fit_one(adapter,root,setup,contract,benchmark,'contribution_gaussian_r32',PROJECTED['contribution_gaussian_r32'],target)
                except Exception:
                    failures.append(failure(root,'calibration',benchmark=benchmark,target=target))
        conditions=freeze_conditions(root,setup,contract)
        phase(root,'shared_operator_diagnostics')
        try:diagnostics(root,sources,contract)
        except Exception:failures.append(failure(root,'shared_operator_diagnostics'))
        # Shorter prompts and lower target first. No screening/elimination by
        # final or tiny-development accuracy; all three requested methods run.
        rows=sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id']))
        for target in TARGETS:
            for name in PROJECTED:
                label=f'{name}_s{int(100*target)}';c=conditions[label]
                phase(root,'final',condition=label,expected=80)
                for row in rows:
                    if row['benchmark'] not in c['thresholds']:continue
                    try:
                        cached(adapter,root,row,'final',label,name,c['config'],c['thresholds'][row['benchmark']],contract)
                    except Exception:
                        failures.append(failure(root,'final',condition=label,id=row['id']))
    del adapter;torch.cuda.empty_cache()
    phase(root,'report')
    from .report import regenerate,verify
    audited=regenerate(root)
    if audited['complete']:verify(root)
    _write(root/'run_terminal.json',dict(complete=audited['complete'],completed=audited['completed'],
        expected=1040,failed_attempts=len(failures),finished=time.time()))


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
        with (root/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.workflow','work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:code=child.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    status={k:read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase','progress')}
                    _append(root/'monitor.jsonl',dict(pid=child.pid,time=time.time(),alive=True,gpu=gpu.stdout.strip(),
                        gpu_error=gpu.stderr.strip(),disk_free_bytes=shutil.disk_usage(root).free,**status))
            _write(root/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    prepare(root);gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.workflow','supervise','--output',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,health_interval_seconds=900,output=str(root))))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','launch','supervise','work','report','verify'))
    p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command in ('report','verify'):
        from .report import regenerate,verify
        print(json.dumps((regenerate if a.command=='report' else verify)(a.output)))
    else:globals()[a.command](a.output)


if __name__=='__main__':main()
