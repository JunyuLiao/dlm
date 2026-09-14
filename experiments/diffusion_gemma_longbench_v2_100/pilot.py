"""Disjoint dense-completion gate and conservative H100 capacity checks."""
import argparse
import json
from pathlib import Path
import traceback

import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_fingerprint
from experiments.diffusion_gemma_value_aware.protocol import frozen_write,sha
from .protocol import ROOT,PARENT,MODEL,REVISION,selection,items,render,rows_for,INPUT_CANDIDATES,OUTPUT_CANDIDATES
from .runner import generate
from . import nemo


def identity(root):
    paths=[Path(__file__),Path(__file__).with_name('protocol.py'),Path(__file__).with_name('runner.py'),Path(__file__).with_name('nemo.py')]
    paths += [PARENT/'execution_contract.json',PARENT/'final_configs/mass_s75.json']
    parent=json.loads((PARENT/'execution_contract.json').read_text())
    for path,digest in parent['sources'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('Validated parent source changed')
    return _fingerprint(dict(selection=selection(root),sources={str(p):sha(p.read_bytes()) for p in paths},nemo_revision=nemo.REVISION))


def run(root=ROOT):
    from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
    gpu_idle()
    fp=identity(root)
    decision=root/'pilot'/'decision.json'
    if decision.exists():
        previous=json.loads(decision.read_text())
        if previous['fingerprint']!=fp:
            from .protocol import audit_pilot_reuse
            audit_pilot_reuse(root)
        return previous
    selected=selection(root)
    byid={r['_id']:r for r in items()}
    longest=max((byid[r['_id']] for r in selected['groups']['development']),key=lambda r:len(r['context']))
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    capacity=[];input_budget=None;sources={}
    parent=Path('results/diffusion_gemma_value_aware_gpu_50_75')
    policy=json.loads((parent/'final_configs/mass_s75.json').read_text())['thresholds']['longbench_v2']
    for cap in INPUT_CANDIDATES:
        proof=root/'pilot'/f'capacity_{cap}.json'
        try:
            if proof.exists():
                trial=json.loads(proof.read_text())
                if trial['fingerprint']!=fp:raise ValueError('Capacity pilot source changed')
            else:
                row=render(adapter,longest,cap,16,'development')
                if row['original_prompt_tokens']<cap:
                    break # Do not claim an untested longer context fits.
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                # Mass75 covers the normalization-dependent diagnostics and
                # fallback; dense covers the untouched whole-query output.
                for name,config,threshold in (('dense',{},None),('mass75',{'method':'mass'},policy)):
                    out=generate(adapter,row,config,threshold,validate=True)
                    out['fingerprint']=fp
                    _write(root/'pilot'/f'capacity_{cap}_{name}_raw.json',out)
                    del out
                peak=torch.cuda.max_memory_allocated()
                total=torch.cuda.get_device_properties(0).total_memory
                trial=dict(fingerprint=fp,cap=cap,passed=peak<total-3*1024**3,
                    peak_allocated_bytes=peak,total_bytes=total,
                    criterion='at least3GiB allocation headroom; no timing/speedup measurement',row_id=row['id'])
                _write(proof,trial)
        except torch.cuda.OutOfMemoryError:
            trial=dict(fingerprint=fp,cap=cap,passed=False,error='CUDA out of memory',traceback=traceback.format_exc())
            _write(proof,trial);torch.cuda.empty_cache()
        for name in ('dense','mass75'):
            raw=root/'pilot'/f'capacity_{cap}_{name}_raw.json'
            if raw.exists():sources[str(raw)]=sha(raw.read_bytes())
        sources[str(proof)]=sha(proof.read_bytes())
        capacity.append(trial);print('capacity',trial,flush=True)
        if not trial['passed']:break
        input_budget=cap
    if input_budget is None:raise RuntimeError('No tested input cap fits safely; preserve pilot and investigate')
    attempts=[];budget=None
    for output_budget in OUTPUT_CANDIDATES:
        rows=rows_for(adapter,'development',input_budget,output_budget,root)
        outputs=[]
        for row in rows:
            path=root/'pilot'/f'dense_{input_budget}_{output_budget}'/(row['source_id']+'.json')
            if path.exists():
                out=json.loads(path.read_text())
                if out['fingerprint']!=fp or out['prompt_hash']!=row['prompt_hash']:raise ValueError('Pilot cache changed')
            else:
                out=generate(adapter,row)
                out['fingerprint']=fp;_write(path,out)
            outputs.append(out);sources[str(path)]=sha(path.read_bytes())
            print('dense development',input_budget,output_budget,row['id'],out['termination_reason'],len(out['completion_tokens']),flush=True)
        graded=nemo.evaluate([dict(index=r['id'],generation=o['prediction'],expected_answer=r['expected']) for r,o in zip(rows,outputs)])
        limited=sum(o['termination_reason']=='length' for o in outputs)
        parsable=sum(g['predicted_answer'] in tuple('ABCD') for g in graded)
        trial=dict(output_budget=output_budget,length_limited=limited,parsable=parsable,count=6)
        attempts.append(trial);_write(root/'pilot'/'completion_trials.json',attempts)
        if limited==0 and parsable>=5:budget=output_budget;break
        if limited==0:break # A nontruncated formatting failure is not a cap problem.
    passed=budget is not None
    result=dict(fingerprint=fp,passed=passed,input_budget=input_budget,output_budget=budget,
        capacity=capacity,completion_trials=attempts,sources=sources,
        selection_uses_correctness=False,final_examples_used=False)
    frozen_write(decision,result)
    if not passed:raise RuntimeError('Dense development completion/format gate failed; inspect before final evaluation')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT)
    print(json.dumps(run(parser.parse_args().output),indent=2))
