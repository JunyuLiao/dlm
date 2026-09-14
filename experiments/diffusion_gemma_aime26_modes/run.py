"""Single-H100 runner; all final rows and calibration are immutable/resumable."""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dllm.models import create_adapter
from dllm.models.base import GenerationRequest
from dllm.attention.blasst.core import Blasst2DConfig, Blasst2DStats, blasst_2d_attention_forward
from dllm.attention.blasst.core import _prepare_attention_scores, _finish_eager_attention, _attention_type
from experiments.diffusion_gemma_aime30.calibration import MarginCollector, calibrate
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense, _set_context, _write, _append, _fingerprint
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_runner import code_hashes
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import aggregate
from experiments.diffusion_gemma_solattn_blasst_multibench.diagnostics import CheckedAttention
from .protocol import ROOT, MODEL, REVISION, MODES, CONDITIONS, TARGETS, prepare, sha, final_response, numeric_score


class DenseCounter:
    """Actual dense counts, not counterfactual BLASST decisions with masking off."""
    def __init__(self): self.calls=[]

    @torch.no_grad()
    def __call__(self,module,q,k,v,mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kw):
        _,ev,s,valid=_prepare_attention_scores(module,q,k,v,mask,scaling=scaling,is_causal=is_causal,sliding_window=sliding_window)
        valid=torch.broadcast_to(valid,s.shape)
        b,h,nq,nk=s.shape
        blocks=F.pad(valid,(0,(-nk)%64,0,(-nq)%64)).reshape(b,h,(nq+63)//64,64,(nk+63)//64,64).any(-1).any(-2)
        count=int(blocks.sum());prefix=max(0,nk-nq)
        by_k=blocks.sum((0,1,2)).cpu().tolist()
        regions={name:dict(eligible_tiles=0,skipped_tiles=0,retained_tiles=0) for name in ('prefix','canvas')}
        for j,n in enumerate(by_k):
            start,end=j*64,min(nk,(j+1)*64)
            # Match the existing BLASST convention: boundary-straddling tiles
            # belong to prefix when their starting KV position is in prefix.
            region='prefix' if start<prefix else 'canvas'
            regions[region]['eligible_tiles']+=n;regions[region]['retained_tiles']+=n
        self.calls.append(dict(layer=int(module.layer_idx),attention_type=_attention_type(module,sliding_window),
            head_ids=list(range(h)),denoising_step=module._blasst_2d_runtime.current_denoising_iteration,
            eligible_tiles=count,skipped_tiles=0,retained_tiles=count,valid_rows=int(valid.any(-1).sum()),
            valid_qk_elements=int(valid.sum()),skipped_valid_qk_elements=0,retained_dense_attention_mass=1.,region_counts=regions))
        return _finish_eager_attention(q,ev,s,valid,dropout,module.training)


def request(row):
    return GenerationRequest(prompt=row['prompt'], max_new_tokens=row['generation_budget'], block_size=256,
        temperature=0., seed=row['seed'], extra={'thinking': row['thinking']})


def generate(adapter, row, lambdas=None, collect=False):
    binding = _install_dense(adapter)
    if collect:
        stats = MarginCollector(); fn = stats
    elif lambdas is None:
        stats = DenseCounter(); fn = stats
    else:
        stats = Blasst2DStats(record_layers=True, record_heads=False)
        binding.runtime.config = Blasst2DConfig(enable_blasst_2d=True, q_tile_size=64, kv_tile_size=64,
            collect_blasst_stats=True, apply_blasst_mask=lambdas is not None,
            local_blasst_lambda=lambdas['local'] if lambdas else 1., global_blasst_lambda=lambdas['global'] if lambdas else 1.)
        binding.runtime.stats = stats; fn = blasst_2d_attention_forward
    checked = CheckedAttention(fn); binding.runtime.attention_override = checked
    try:
        _set_context(binding, row); out = adapter.generate(request(row))
        assert out.prompt_tokens == row['prompt_tokens'], 'prompt mismatch'
        assert out.metadata['thinking'] == row['thinking']
        assert checked.calls and out.completion_tokens
        calls = stats.calls if collect or lambdas is None else list(stats.per_step.values())
        layers = calls if collect or lambdas is None else list(stats.per_layer.values())
        types = adapter.model.config.text_config.layer_types
        assert {c['layer'] for c in layers} == set(range(len(types)))
        assert all(c['attention_type'] == ('local' if types[c['layer']]=='sliding_attention' else 'global') for c in layers)
        raw = adapter.tokenizer.decode(out.completion_tokens, skip_special_tokens=False)
        answer = final_response(raw, row['thinking'])
        result = {k:row[k] for k in ('id','source_id','mode','thinking','prompt_hash','prompt_token_hash','generation_budget','seed')}
        result.update(prediction=out.text, raw_completion=raw, final_response=answer, completion_tokens=out.completion_tokens,
            score=numeric_score(answer,row['expected']), termination_reason=out.termination_reason,
            generation_metadata=out.metadata, elapsed_seconds=out.elapsed_seconds,
            calls=calls, per_layer=layers, lambdas=lambdas, finite_attention_calls=checked.calls)
        return result, stats.arrays() if collect else None
    finally:
        binding.close()


def cached(adapter, row, path, fp, root, condition, lambdas=None, collect=False):
    npz = path.with_suffix('.npz')
    if path.exists() and (not collect or npz.exists()):
        result = json.loads(path.read_text())
        assert result['fingerprint']==fp and result['lambdas']==lambdas, 'cache mismatch'
        return result, dict(np.load(npz)) if collect else None
    _write(root/'progress.json',dict(condition=condition,id=row['id'],started=time.time()))
    result, arrays = generate(adapter,row,lambdas,collect)
    result['fingerprint'] = fp
    path.parent.mkdir(parents=True,exist_ok=True)
    if collect: np.savez_compressed(npz,**arrays)
    _write(path,result)
    print(time.strftime('%FT%TZ',time.gmtime()),condition,row['id'],'score',result['score']['correct'],flush=True)
    torch.cuda.empty_cache()
    return result, arrays


def selected(policy, mode, target):
    return {k:policy['modes'][mode][k]['targets'][str(target)]['lambda_value'] for k in ('local','global')}


def smoke(adapter, rows, root, fp):
    path=root/'smoke.json'
    if path.exists():
        r=json.loads(path.read_text()); assert r['fingerprint']==fp and r['passed']; return
    records=[]
    for mode in MODES:
        row=dict(next(r for r in rows if r['mode']==mode),generation_budget=32)
        binding=_install_dense(adapter)
        try:
            _set_context(binding,row); reference=adapter.generate(request(row))
        finally: binding.close()
        dense,_=generate(adapter,row)
        collector,_=generate(adapter,row,collect=True)
        sparse,_=generate(adapter,row,{'local':1.,'global':1.})
        stat=aggregate(sparse['calls'])
        assert dense['completion_tokens']==reference.completion_tokens==collector['completion_tokens'], 'dense parity failed'
        assert aggregate(dense['calls'])['overall']['skipped_tiles']==0
        assert all(stat['regions'][k]['eligible_tiles']>0 for k in ('prefix','canvas'))
        assert all(stat[k]['eligible_tiles']>0 for k in ('local','global'))
        assert stat['overall']['skipped_tiles']>0
        records.append(dict(mode=mode,dense_parity=True,stats=stat,finite_calls=sparse['finite_attention_calls']))
    _write(path,dict(passed=True,fingerprint=fp,records=records))
    print('SMOKE PASSED all four modes',flush=True)


def execute(root):
    manifests=prepare(root)
    files=list(Path(__file__).parent.glob('*.py'))+[Path('experiments/diffusion_gemma_aime30/calibration.py'),Path('experiments/diffusion_gemma_aime30/protocol.py')]
    config=dict(setup=json.loads((root/'setup.json').read_text()),manifests={k:_fingerprint(v) for k,v in manifests.items()},
        code={**code_hashes(),**{str(p):sha(p.read_bytes()) for p in files}},torch=torch.__version__,transformers=__import__('transformers').__version__)
    fp=_fingerprint(config)
    if (root/'experiment.json').exists(): assert json.loads((root/'experiment.json').read_text())==config, 'protocol/code changed'
    _write(root/'experiment.json',config)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    _write(root/'model_runtime.json',dict(model_config=adapter.model.config.to_dict(),generation_config=adapter.model.generation_config.to_dict()))
    smoke(adapter,manifests['calibration'],root,fp)
    # Fresh dense baselines first; never import the earlier capped results.
    failures=[]
    for row in manifests['final']:
        try:
            cached(adapter,row,root/'conditions/dense/shards'/f'{sha(row["id"])}.json',fp,root,'dense')
        except Exception as exc:
            error=dict(condition='dense',id=row['id'],error=repr(exc),traceback=traceback.format_exc())
            _append(root/'failures.jsonl',error);failures.append(error);print('FAILED',error,flush=True)
            torch.cuda.empty_cache()
    policy_path=root/'policy.json'
    if policy_path.exists():
        policy=json.loads(policy_path.read_text()); assert policy['fingerprint']==fp
    else:
        policy=dict(fingerprint=fp,modes={})
        for mode in MODES:
            parts={k:[] for k in ('local','global')}; lengths={k:[] for k in parts}
            for row in (r for r in manifests['calibration'] if r['mode']==mode):
                result,arr=cached(adapter,row,root/'calibration_dense'/mode/f'{sha(row["id"])}.json',fp,root,'calibration_dense',collect=True)
                for k in parts:
                    parts[k].append(arr[k]);lengths[k].extend(c['valid_kv_length'] for c in result['calls'] if c['attention_type']==k)
            policy['modes'][mode]={k:calibrate(np.concatenate(parts[k]),float(np.mean(lengths[k]))) for k in parts}
        _write(policy_path,policy)
    checks=[]
    for mode in MODES:
        for target in TARGETS:
            lam=selected(policy,mode,target); results=[]
            for row in (r for r in manifests['calibration'] if r['mode']==mode):
                r,_=cached(adapter,row,root/'verification'/mode/f's{int(target*100)}'/f'{sha(row["id"])}.json',fp,root,'verify',lam)
                results.append(r)
            stats=aggregate(c for r in results for c in r['calls'])
            checks.append(dict(mode=mode,target=target,lambdas=lam,stats=stats,
                within_two_points={k:abs(stats[k]['full_tile_sparsity']-target)<=.02 for k in ('overall','global','local')}))
    _write(root/'calibration_verification.json',checks)
    for name,target in CONDITIONS[1:]:
        for row in manifests['final']:
            try:
                cached(adapter,row,root/'conditions'/name/'shards'/f'{sha(row["id"])}.json',fp,root,name,selected(policy,row['mode'],target))
            except Exception as exc:
                error=dict(condition=name,id=row['id'],error=repr(exc),traceback=traceback.format_exc())
                _append(root/'failures.jsonl',error);failures.append(error);print('FAILED',error,flush=True)
                torch.cuda.empty_cache()
        from .report import report
        report(root)
    _write(root/'run_status.json',dict(finished=True,failures=failures))
    from .report import report
    print(report(root),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--output-dir',type=Path,default=ROOT)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.output_dir)
    elif a.command=='run': execute(a.output_dir)
    else:
        from .report import report
        print(report(a.output_dir))

if __name__=='__main__': main()
