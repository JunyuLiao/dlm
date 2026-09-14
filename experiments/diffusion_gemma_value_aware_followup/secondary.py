"""Fresh native-parity/finite-output smoke for guarded Sol and rankings."""
import argparse
import json
from pathlib import Path
import traceback
import time

from experiments.diffusion_gemma_value_aware.ranking_guards import GUARD
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from .engine import contract
from .execution import cached,provenance
from .evidence import read_result,raw_source,DECODING_FIELDS
from .calibrate import smoke_dense
from .protocol import ROOT,MODEL,REVISION,prepare,sha


def cases():
    result={}
    for value in (False,True):
        label='value' if value else 'plain'
        result[f'sol_{label}_gaussian90']=dict(method='sol',value_proxy=value,mode='gaussian',amount=.9)
        result[f'sol_{label}_unpruned']=dict(method='sol',value_proxy=value,mode='topk',amount=0.)
    for signal in ('qk','mass','contribution'):
        result[f'{signal}_unpruned']=dict(method='diagnostic',pooling=signal,mode='topk',amount=0.)
        result[f'{signal}_topk90']=dict(method='diagnostic',pooling=signal,mode='topk',amount=.9)
        if signal!='qk':
            result[f'{signal}_topp95']=dict(method='diagnostic',pooling=signal,mode='topp',amount=.95)
    return {n:dict(c,aggregation=GUARD) for n,c in result.items()}


def examples(setup):
    return [dict(next(r for r in setup['calibration'] if r['benchmark']=='aime26'),generation_budget=16),
        dict(max((r for r in setup['calibration'] if r['benchmark']=='longbench_v2'),
            key=lambda r:len(r['prompt_tokens'])),generation_budget=16)]


def check_case(result,native,name):
    if name.endswith('unpruned'):
        if result['completion_tokens']!=native['completion_tokens'] or any(r['pv_omitted'] for r in result['records']):
            raise ValueError('secondary unpruned native parity/retention failed')
    for key in DECODING_FIELDS:
        if result['generation_metadata'].get(key)!=native['generation_metadata'].get(key):
            raise ValueError('secondary smoke unrelated decoding change')


def require_smoke(root,setup,execution):
    from experiments.diffusion_gemma_value_aware.run import shard_path
    path=root/'secondary_smoke.json';data,source=raw_source(path)
    if not data['passed'] or data['fingerprint']!=execution['fingerprint'] or data['cases']!=cases():
        raise ValueError('matching complete guarded Sol/ranking CUDA smoke required')
    if data['source_sha256']!=sha(Path(__file__).read_bytes()):
        raise ValueError('secondary smoke source changed')
    sources={source['path']:source['sha256']}
    for row in examples(setup):
        native_path=shard_path(root,'sparse_smoke','native_dense16',row['id'])
        native,source=raw_source(native_path);sources[source['path']]=source['sha256']
        identity=dict(fingerprint=execution['fingerprint'],
            **{k:row[k] for k in ('id','prompt_hash','seed','generation_budget')})
        if native['identity']!=identity:
            raise ValueError('secondary native reference identity mismatch')
        for name,config in cases().items():
            result,source=read_result(root,row,'secondary_smoke',name,config,None,execution)
            check_case(result,native,name);sources[source['path']]=source['sha256']
    return sources


def execute(root):
    setup=prepare(root);execution=contract(root);adapter=None;failures=[]
    import torch
    from dllm.models import create_adapter
    for row in examples(setup):
        for name,config in cases().items():
            try:
                try:
                    result,_=read_result(root,row,'secondary_smoke',name,config,None,execution)
                except FileNotFoundError:
                    if adapter is None:
                        adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',
                            precision='bfloat16',revision=REVISION).load()
                    result=cached(adapter,root,row,'secondary_smoke',name,config,None,execution)
                native=smoke_dense(adapter,root,row,execution)
                check_case(result,native,name)
            except Exception:
                error=dict(stage='secondary_smoke',id=row['id'],case=name,traceback=traceback.format_exc())
                failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
            finally:
                torch.cuda.empty_cache()
    _write(root/'secondary_smoke.json',dict(passed=not failures,failures=failures,
        fingerprint=execution['fingerprint'],cases=cases(),source_sha256=sha(Path(__file__).read_bytes()),finished=time.time()))
    if failures:
        raise RuntimeError('some secondary smoke cases failed; independent cases continued')
    require_smoke(root,setup,execution)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();execute(args.output)


if __name__=='__main__':
    main()
