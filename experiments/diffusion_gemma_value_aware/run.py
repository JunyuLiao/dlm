"""Resumable prepare/smoke/screen stages; final policies are frozen separately."""
import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense,_set_context,_request,_write,_append
from .protocol import ROOT,MODEL,REVISION,AIME,prepare,fingerprint,score,sha
from .routing import Attention


def shard_path(root, stage, condition, identity):
    return root/stage/condition/'shards'/f'{sha(identity)}.json'


def generate(adapter,row,config=None,thresholds=None,screen=False):
    binding=_install_dense(adapter); router=Attention(config,thresholds,screen)
    binding.runtime.attention_override=router
    try:
        _set_context(binding,row); output=adapter.generate(_request(row))
    finally: binding.close()
    expected=row['prompt_tokens']
    if isinstance(expected,list): assert output.prompt_tokens==expected,'prompt ID mismatch'
    else: assert len(output.prompt_tokens)==expected,'prompt length mismatch'
    records=router.records
    assert {r['layer'] for r in records}==set(range(30)),'incomplete layer coverage'
    assert {r['head'] for r in records}==set(range(16)),'incomplete head coverage'
    assert {r['attention_type'] for r in records}=={'local','global'},'layer type coverage'
    result=dict(id=row['id'],prompt_hash=row['prompt_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
        config=config or {},thresholds=thresholds,screen=screen,prediction=output.text,completion_tokens=output.completion_tokens,
        generation_metadata=output.metadata,termination_reason=output.termination_reason,records=records,
        distributions=router.distributions,finite_calls=router.calls,
        metadata_cache=dict(reused_blocks=router.cache.reused_blocks,refreshed_blocks=router.cache.refreshed_blocks))
    return result,router.arrays() if screen else None


def cache(adapter,row,path,fp,config=None,thresholds=None,screen=False,root=ROOT):
    if path.exists():
        result=json.loads(path.read_text())
        assert result['fingerprint']==fp and result['config']==(config or {}) and result['thresholds']==thresholds and result['screen']==screen,'incompatible cache'
        if screen: assert path.with_suffix('.npz').exists(),'missing calibration risks'
        return result
    _write(root/'progress.json',dict(pid=os.getpid(),path=str(path),id=row['id'],started=time.time()))
    result,arrays=generate(adapter,row,config,thresholds,screen)
    # Cached AIME26 dense used only as a parity check here; never copy old
    # sparse results with different operators/threshold provenance.
    if not config and row['benchmark']=='aime26':
        previous=AIME/'dense'/f'{row["source_id"]}.json'
        old=json.loads(previous.read_text())
        assert all(old[k]==row[k] for k in ('prompt_hash','seed','generation_budget'))
        assert old['completion_tokens']==result['completion_tokens'],'expected cached dense parity failed'
        result['dense_cache_parity']=dict(passed=True,path=str(previous),sha256=sha(previous.read_bytes()))
    result.update(fingerprint=fp,score=score(row,result['prediction']))
    if arrays is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays)
        temporary.replace(path.with_suffix('.npz'))
    _write(path,result)
    print(time.strftime('%FT%TZ',time.gmtime()),row['id'],str(path),'completed',flush=True)
    return result


def smoke(adapter,setup,root):
    fp=fingerprint(root); path=root/'smoke.json'
    if path.exists():
        prior=json.loads(path.read_text());assert prior['fingerprint']==fp and prior['passed'];return
    records=[]
    for benchmark in ('aime26','longbench'):
        row=dict(next(r for r in setup['calibration'] if r['benchmark']==benchmark),generation_budget=32)
        binding=_install_dense(adapter)
        try: _set_context(binding,row); dense=adapter.generate(_request(row))
        finally: binding.close()
        for config in ({},{'method':'mass','log_threshold':-100.},
                       {'method':'value','log_threshold':0.}, {'method':'centered','log_threshold':-2.},
                       {'method':'compensate','log_threshold':-3.}, {'method':'zero_pv','log_threshold':-3.},
                       {'method':'sol','mode':'gaussian','amount':.5,'value_proxy':True}):
            result,_=generate(adapter,row,config)
            unpruned=not config or config.get('log_threshold')==-100.
            if unpruned: assert result['completion_tokens']==dense.completion_tokens,'dense smoke parity'
            rr=result['records']
            records.append(dict(benchmark=benchmark,config=config,passed=True,dense_parity=True if unpruned else None,
                finite_calls=result['finite_calls'],eligible=sum(r['eligible'] for r in rr),skipped=sum(r['skipped'] for r in rr),
                pv_omitted=sum(r['pv_omitted'] for r in rr),prefix_eligible=sum(r['prefix_eligible'] for r in rr),
                canvas_eligible=sum(r['canvas_eligible'] for r in rr),metadata_cache=result['metadata_cache']))
            print('smoke',benchmark,config,'passed',flush=True)
    _write(path,dict(passed=True,fingerprint=fp,tests=records))


def execute(root,command):
    setup=prepare(root);fp=fingerprint(root)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'model_runtime.json',dict(model=adapter.model.config.to_dict(),generation=adapter.model.generation_config.to_dict(),
        torch=torch.__version__,transformers=__import__('transformers').__version__,fingerprint=fp))
    smoke(adapter,setup,root)
    if command=='smoke': return
    failures=[]
    # No held-out examples are read for the initial screen/selection stage.
    rows=setup['calibration']+setup['development']
    for row in sorted(rows,key=lambda r:(r['generation_budget'],r['id'])):
        try: cache(adapter,row,shard_path(root,'screen','dense',row['id']),fp,screen=True,root=root)
        except Exception:
            error=dict(id=row['id'],stage='screen',traceback=traceback.format_exc())
            failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
        finally: torch.cuda.empty_cache()
    _write(root/'screen_status.json',dict(finished=True,failures=failures,expected=len(rows),fingerprint=fp))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','screen'])
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    if args.command=='prepare': prepare(args.output)
    else: execute(args.output,args.command)


if __name__=='__main__': main()
