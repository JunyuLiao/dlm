"""Single-GPU resumable calibration, smoke, and paired AIME sweep."""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch

from dllm.models import create_adapter
from dllm.attention.blasst.core import Blasst2DConfig, Blasst2DStats, blasst_2d_attention_forward
from experiments.diffusion_attention_threshold_modeling.routing import FreshRoutingAttention
from experiments.diffusion_gemma_solattn_blasst_multibench.diagnostics import CheckedAttention, sol_config
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense, _set_context, _request, _write, _append, _fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_runner import code_hashes
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate
from .protocol import ROOT, MODEL, REVISION, MODES, TARGETS, conditions, prepare, sha, numeric_score
from .calibration import MarginCollector, calibrate


def experiment_config(root):
    paths=list(Path(__file__).parent.glob('*.py'))
    return dict(setup=json.loads((root/'setup.json').read_text()),
        final_sha256=sha((root/'final.json').read_bytes()),calibration_sha256=sha((root/'calibration.json').read_bytes()),
        code_hashes={**code_hashes(),**{str(p):sha(p.read_bytes()) for p in paths}},
        torch=torch.__version__,transformers=__import__('transformers').__version__)


def install(adapter,method,target=0.,lambdas=None):
    binding=_install_dense(adapter)
    if method=='blasst':
        stats=Blasst2DStats(record_layers=True,record_heads=False)
        binding.runtime.config=Blasst2DConfig(enable_blasst_2d=True,q_tile_size=64,kv_tile_size=64,
            collect_blasst_stats=True,apply_blasst_mask=True,local_blasst_lambda=lambdas['local'],global_blasst_lambda=lambdas['global'])
        binding.runtime.stats=stats
        fn=blasst_2d_attention_forward
    elif method=='sol': stats=FreshRoutingAttention(sol_config(target));fn=stats
    else: stats=MarginCollector();fn=stats
    checked=CheckedAttention(fn);binding.runtime.attention_override=checked
    return binding,stats,checked


def generate(adapter,row,method,target=0.,lambdas=None):
    binding,stats,checked=install(adapter,method,target,lambdas)
    try:
        _set_context(binding,row);out=adapter.generate(_request(row))
        if out.prompt_tokens!=row['prompt_tokens']: raise RuntimeError('actual input token IDs differ from frozen prompt')
        if not out.completion_tokens or not checked.calls: raise RuntimeError('empty output/attention coverage')
        if method=='sol': calls=stats.stats.per_call;layers=calls
        elif method=='blasst': calls=list(stats.per_step.values());layers=list(stats.per_layer.values())
        else: calls=stats.calls;layers=calls
        types={c['attention_type'] for c in calls}
        if types!={'global','local'}: raise RuntimeError('missing attention type')
        if {c['layer'] for c in layers}!=set(range(adapter.model.config.text_config.num_hidden_layers)): raise RuntimeError('missing layer')
        result=dict(id=row['id'],source_id=row['source_id'],mode=row['mode'],prompt_hash=row['prompt_hash'],
            prompt_token_hash=row['prompt_token_hash'],seed=row['seed'],generation_budget=row['generation_budget'],
            method=method,target_sparsity=target,lambdas=lambdas,prediction=out.text,completion_tokens=out.completion_tokens,
            termination_reason=out.termination_reason,generation_metadata=out.metadata,
            score=numeric_score(out.text,row['expected']),calls=calls,per_layer=layers,finite_attention_calls=checked.calls)
        arrays=stats.arrays() if method=='dense' else None
        return result,arrays
    finally: binding.close()


def smoke(adapter,root,fingerprint,rows):
    path=root/'smoke.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['fingerprint']!=fingerprint or not old['passed']: raise RuntimeError('incompatible smoke')
        return
    records=[]
    for mode in MODES:
        row=dict(next(r for r in rows if r['mode']==mode),generation_budget=32)
        binding=_install_dense(adapter)
        try: _set_context(binding,row);reference=adapter.generate(_request(row))
        finally: binding.close()
        dense,_=generate(adapter,row,'dense');zero,_=generate(adapter,row,'sol',0.)
        sol,_=generate(adapter,row,'sol',.9);blasst,_=generate(adapter,row,'blasst',.9,{'local':1.,'global':1.})
        a=aggregate(sol['calls']);b=aggregate(blasst['calls'])
        record=dict(mode=mode,dense_parity=dense['completion_tokens']==reference.completion_tokens,
            zero_prune_parity=zero['completion_tokens']==reference.completion_tokens,
            zero_skipped=aggregate(zero['calls'])['overall']['skipped_tiles']==0,
            prefix_skipped=a['regions']['prefix']['skipped_tiles'],canvas_skipped=a['regions']['canvas']['skipped_tiles'],
            blasst_eligible=b['overall']['eligible_tiles'],layer_ids=sorted({c['layer'] for c in sol['calls']}),
            head_ids=sorted({h for c in sol['calls'] for h in c['head_ids']}))
        records.append(record);print('SMOKE',record,flush=True)
    passed=all(r['dense_parity'] and r['zero_prune_parity'] and r['zero_skipped'] and r['prefix_skipped']>0 and r['canvas_skipped']>0 and r['blasst_eligible']>0 for r in records)
    _write(path,dict(passed=passed,fingerprint=fingerprint,records=records))
    if not passed: raise RuntimeError('smoke failed')


def calibrate_all(adapter,root,fingerprint,rows):
    path=root/'policy.json'
    if path.exists():
        policy=json.loads(path.read_text())
        if policy['fingerprint']!=fingerprint: raise RuntimeError('incompatible calibration')
        return policy
    policy=dict(fingerprint=fingerprint,modes={})
    for mode in MODES:
        parts={k:[] for k in ('local','global')};lengths={k:[] for k in parts}
        for i,row in enumerate(r for r in rows if r['mode']==mode):
            folder=root/'calibration_dense'/mode
            path_json=folder/(sha(row['id'])+'.json');path_npz=path_json.with_suffix('.npz')
            _write(root/'progress.json',dict(condition='calibration_dense_'+mode,id=row['id'],index=i+1,total=6,started=time.time()))
            if path_json.exists() and path_npz.exists():
                result=json.loads(path_json.read_text())
                if result['fingerprint']!=fingerprint: raise RuntimeError('incompatible calibration shard')
                with np.load(path_npz) as data: arrays={k:data[k] for k in parts}
            else:
                result,arrays=generate(adapter,row,'dense');result['fingerprint']=fingerprint
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path_npz,**arrays);_write(path_json,result)
            for kind in parts:
                parts[kind].append(arrays[kind]);lengths[kind].extend(c['valid_kv_length'] for c in result['calls'] if c['attention_type']==kind)
            print('CALIBRATION',mode,i+1,'/6',flush=True)
        policy['modes'][mode]={kind:calibrate(np.concatenate(parts[kind]),float(np.mean(lengths[kind]))) for kind in parts}
    _write(root/'policy.json',policy)
    return policy


def selected_lambdas(policy,mode,target):
    return {kind:policy['modes'][mode][kind]['targets'][str(target)]['lambda_value'] for kind in ('local','global')}


def validate_calibration(adapter,root,fingerprint,rows,policy):
    """Verify transfer to sparse calibration trajectories, without final labels."""
    checks=[]
    for mode in MODES:
        for target in TARGETS:
            lambdas=selected_lambdas(policy,mode,target);results=[]
            for row in (r for r in rows if r['mode']==mode):
                path=root/'calibration_verification'/mode/f's{int(target*100)}'/(sha(row['id'])+'.json')
                if path.exists():
                    result=json.loads(path.read_text())
                    if result['fingerprint']!=fingerprint or result['lambdas']!=lambdas: raise RuntimeError('verification provenance mismatch')
                else:
                    _write(root/'progress.json',dict(condition='calibration_verify_'+mode,id=row['id'],index=len(results)+1,total=6,started=time.time()))
                    result,_=generate(adapter,row,'blasst',target,lambdas);result['fingerprint']=fingerprint;_write(path,result)
                results.append(result)
            stats=aggregate(c for r in results for c in r['calls'])
            checks.append(dict(mode=mode,target=target,lambdas=lambdas,stats=stats,
                within_two_points={k:abs(stats[k]['full_tile_sparsity']-target)<=.02 for k in ('overall','global','local')}))
            print('VERIFY',mode,target,{k:stats[k]['full_tile_sparsity'] for k in ('overall','global','local')},flush=True)
    _write(root/'calibration_verification.json',checks)


def run_final(adapter,root,fingerprint,rows,policy):
    failures=[]
    # BLASST is inexpensive in this reference implementation; run it before Sol.
    grid=conditions();grid=grid[:1]+[c for c in grid if c['method']=='blasst']+[c for c in grid if c['method']=='sol']
    for condition in grid:
        for i,row in enumerate(rows):
            directory=root/'conditions'/condition['name'];path=directory/'shards'/(sha(row['id'])+'.json')
            lam=selected_lambdas(policy,row['mode'],condition['target']) if condition['method']=='blasst' else None
            run_fingerprint=_fingerprint(dict(experiment=fingerprint,condition=condition,policy=policy))
            _write(directory/'config.json',dict(fingerprint=run_fingerprint,condition=condition,policy_sha256=_fingerprint(policy)))
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['fingerprint']!=run_fingerprint: raise RuntimeError('incompatible final shard')
                continue
            _write(root/'progress.json',dict(condition=condition['name'],id=row['id'],index=i+1,total=len(rows),started=time.time()))
            try:
                result,arrays=generate(adapter,row,condition['method'],condition['target'],lam)
                result.update(fingerprint=run_fingerprint,beta=condition['beta'])
                _write(path,result)
                if arrays is not None: np.savez_compressed(path.with_suffix('.npz'),**arrays)
                print(time.strftime('%FT%TZ',time.gmtime()),condition['name'],f'{i+1}/{len(rows)}',row['id'],flush=True)
            except Exception as exc:
                error=dict(condition=condition['name'],id=row['id'],error=repr(exc),traceback=traceback.format_exc())
                _append(root/'failures.jsonl',error);failures.append(error);print('FAILED',error,flush=True)
            finally: torch.cuda.empty_cache()
    _write(root/'run_status.json',dict(finished=True,failures=failures))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report']);parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args();root=args.output_dir
    if args.command=='report':
        from .report import report
        print(report(root));return
    prepare(root)
    if args.command=='prepare': return
    config=experiment_config(root);fingerprint=_fingerprint(config)
    path=root/'experiment.json'
    if path.exists() and json.loads(path.read_text())!=config: raise RuntimeError('experiment changed: use fresh directory')
    _write(path,config)
    rows=json.loads((root/'final.json').read_text());calrows=json.loads((root/'calibration.json').read_text())
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'model_runtime.json',dict(adapter=adapter.runtime_metadata(),generation_config=adapter.model.generation_config.to_dict(),model_config=adapter.model.config.to_dict()))
    smoke(adapter,root,fingerprint,calrows)
    policy=calibrate_all(adapter,root,fingerprint,calrows)
    validate_calibration(adapter,root,fingerprint,calrows,policy)
    run_final(adapter,root,fingerprint,rows,policy)
    from .report import report
    print(report(root),flush=True)


if __name__=='__main__': main()
