"""One-H100 dense baseline / shared-screen stage with a 15-minute supervisor.

This initial stage is not completion of the follow-up: sparse calibration,
all six value-aware families, controls, final curves and reporting follow.
"""
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

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append,_install_dense,_request,_set_context
from experiments.diffusion_gemma_value_aware import run as original
from .engine import cached,contract,check_result
from .protocol import ROOT,MODEL,REVISION,PREVIOUS,prepare,sha


def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=20)
    if result.stdout.strip():raise RuntimeError('GPU is occupied; refusing a competing worker')


def smoke(adapter,root,setup,execution):
    path=root/'smoke.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['fingerprint']!=execution['fingerprint'] or not prior['passed']:
            raise ValueError('incompatible smoke')
        return prior
    # Include a near-limit truncated code context, not only easy short QA.
    examples=[next(r for r in setup['calibration'] if r['benchmark']=='aime26'),
        max((r for r in setup['calibration'] if r['benchmark']=='longbench_v2'),key=lambda r:len(r['prompt_tokens']))]
    tests=[]
    for example in examples:
        row=dict(example,generation_budget=32)
        binding=_install_dense(adapter)
        try:
            _set_context(binding,row);native=adapter.generate(_request(row))
        finally:binding.close()
        if row['benchmark']=='aime26':
            previous_path=original.shard_path(PREVIOUS,'refinement_smoke','mass_exact_unpruned',row['id'])
            previous=json.loads(previous_path.read_text())
            if any(previous[k]!=row[k] for k in ('id','prompt_hash','seed','generation_budget')):
                raise ValueError('previous dense replay manifest mismatch')
            if previous['completion_tokens']!=native.completion_tokens:
                raise ValueError('previous expected AIME dense tokens did not reproduce')
            tests.append(dict(id=row['id'],case='previous_expected_dense_replay',dense_parity=True,
                source=str(previous_path),source_sha256=sha(previous_path.read_bytes())))
        for config in ({},dict(method='value',pooling='vector_mean',log_threshold=-100.)):
            result,_=original.generate(adapter,row,config)
            if result['completion_tokens']!=native.completion_tokens:
                raise ValueError('unpruned native dense parity failed')
            result['fingerprint']=execution['fingerprint']
            check_result(result,row,execution['fingerprint'],config,None)
            if any(r['skipped'] for r in result['records']):raise ValueError('unpruned smoke skipped tiles')
            tests.append(dict(id=row['id'],config=config,dense_parity=True,finite_calls=result['finite_calls'],
                prefix_eligible=sum(r['prefix_eligible'] for r in result['records']),
                canvas_eligible=sum(r['canvas_eligible'] for r in result['records'])))
        # An actual sparse value rule exercises both KV regions and validates
        # finite generation separately from the unpruned parity checks.
        config=dict(method='value',pooling='vector_mean',log_threshold=0.)
        result,_=original.generate(adapter,row,config);result['fingerprint']=execution['fingerprint']
        check_result(result,row,execution['fingerprint'],config,None)
        tests.append(dict(id=row['id'],config=config,dense_parity=None,finite_calls=result['finite_calls'],
            skipped=sum(r['skipped'] for r in result['records']),
            prefix_skipped=sum(r['prefix_skipped'] for r in result['records']),
            canvas_skipped=sum(r['canvas_skipped'] for r in result['records'])))
        print('smoke passed',row['id'],flush=True)
    if not all(any(t.get(k,0)>0 for t in tests) for k in ('prefix_skipped','canvas_skipped')):
        raise ValueError('missing prefix/canvas sparse smoke coverage')
    result=dict(passed=True,fingerprint=execution['fingerprint'],tests=tests)
    _write(path,result);return result


def initial(root):
    import torch
    from dllm.models import create_adapter
    setup=prepare(root);execution=contract(root)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'runtime.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,
        model=adapter.model.config.to_dict(),generation=adapter.model.generation_config.to_dict()))
    smoke(adapter,root,setup,execution)
    failures=[]
    stages=[('dense',setup['final'],False),('screen',setup['calibration']+setup['development'],True)]
    for stage,rows,screen in stages:
        _write(root/'phase.json',dict(stage=stage,expected=len(rows),started=time.time()))
        for row in sorted(rows,key=lambda r:(r['generation_budget'],r['id'])):
            try:
                cached(adapter,root,row,stage,'dense',screen=screen,execution=execution)
                _append(root/'completed.jsonl',dict(stage=stage,id=row['id'],finished=time.time()))
            except Exception:
                error=dict(stage=stage,id=row['id'],traceback=traceback.format_exc())
                _append(root/'failures.jsonl',error);failures.append(error);print(error,flush=True)
            finally:torch.cuda.empty_cache()
    # Successful process exit alone never proves complete baseline/screen data.
    missing=[]
    for stage,rows,screen in stages:
        for row in rows:
            try:cached(None,root,row,stage,'dense',screen=screen,execution=execution)
            except Exception as error:missing.append(dict(stage=stage,id=row['id'],error=str(error)))
    _write(root/'initial_audit.json',dict(complete=not missing,dense_expected=60,screen_expected=18,
        missing=missing,failed_attempts=len(failures),fingerprint=execution['fingerprint']))
    if missing:raise RuntimeError('initial stage incomplete; preserve failures and resume missing results')


def supervise(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle()
        with (root/'initial.log').open('a',buffering=1) as log:
            worker=subprocess.Popen([sys.executable,'-u','-m',__package__+'.run','initial','--output',str(root)],
                stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=worker.pid,supervisor_pid=os.getpid(),stage='initial',started=time.time()))
            while True:
                try:code=worker.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],
                        capture_output=True,text=True,timeout=20)
                    progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else None
                    _append(root/'monitor.jsonl',dict(time=time.time(),pid=worker.pid,alive=worker.poll() is None,
                        gpu=gpu.stdout.strip(),gpu_error=gpu.stderr.strip(),progress=progress,
                        disk_free_bytes=shutil.disk_usage(root).free))
            _write(root/'initial_terminal.json',dict(exit_code=code,pid=worker.pid,finished=time.time(),
                scope='Initial dense/screen stage only; sparse sweeps and full goal remain incomplete'))
            if code:raise SystemExit(code)


def launch(root):
    root.mkdir(parents=True,exist_ok=True)
    # Preflight the lock too; supervisor takes it again for its whole lifetime.
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-u','-m',__package__+'.run','supervise','--output',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=process.pid,scope='initial dense/screen only')),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initial','supervise','launch'])
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    {'initial':initial,'supervise':supervise,'launch':launch}[args.command](args.output)


if __name__=='__main__':main()
