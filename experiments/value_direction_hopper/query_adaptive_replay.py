"""Same-input decoder replays on a small task-balanced diagnostic subset.

The returned forward is matched-kernel dense, so sparse replays cannot alter
the evolving diagnostic trajectory. Every alternate pass is extra diagnostic
work and is excluded from production timing and generation-call counts.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request,_set_context
from .experiment import atomic,shard_path
from .integration import install
from .query_adaptive import State,observe
from .query_adaptive_granularity import Probe as GranularityProbe
from .trajectory import AttentionProbe


def accepted_from_logits(processed,entropy_bound):
    entropy=torch.distributions.Categorical(logits=processed).entropy()
    ordered,indices=entropy.sort(-1)
    chosen=ordered.cumsum(-1)-ordered<=entropy_bound
    return torch.zeros_like(chosen).scatter(-1,indices,chosen),entropy


def per_position(processed,reference,accepted,reference_accepted,sensitivity):
    x=processed.float();d=reference.float()
    top=x.argmax(-1);dense_top=d.argmax(-1)
    first=x.topk(2,-1).values;dense_first=d.topk(2,-1).values
    p=(first[...,0]-torch.logsumexp(x,-1)).exp()
    dense_p=(dense_first[...,0]-torch.logsumexp(d,-1)).exp()
    if sensitivity is None:sensitivity=torch.ones_like(p)
    rows=[]
    for i in range(top.shape[-1]):
        rows.append(dict(position=i,sensitivity=float(sensitivity[0,i]),
            top1_disagree=bool(top[0,i]!=dense_top[0,i]),
            confidence_loss=float(dense_p[0,i]-p[0,i]),
            margin_loss=float((dense_first[0,i,0]-dense_first[0,i,1])-(first[0,i,0]-first[0,i,1])),
            accepted=bool(accepted[0,i]),dense_accepted=bool(reference_accepted[0,i])))
    return rows


@contextmanager
def replay(model,router,probe,granularity,state,policies,records,steps=(2,3)):
    previous=model._denoising_step;had='_denoising_step' in model.__dict__;saved=model.__dict__.get('_denoising_step')
    def step(this,**kwargs):
        cur_step=int(kwargs['cur_step']);iteration=49-cur_step
        original_forward=kwargs['decoder_forward']
        if iteration not in steps:return previous(**kwargs)
        def replay_forward(*args,**forward_kwargs):
            original_policy=router.thresholds;original_sensitivity=router.query_sensitivity
            original_selector=router.policy_selector
            output={};masks={};tile_flags={};errors={};granularity_records={}
            definitions=[('dense',None,None)]
            for target in (50,70):
                definitions.extend([(f'unweighted_s{target}',policies[f'unweighted_s{target}'],None),
                                    (f'CT_s{target}',policies[f'CT_s{target}'],original_sensitivity)])
            definitions.append(('dense_repeat',None,None))
            try:
                for name,policy,sensitivity in definitions:
                    router.thresholds=original_policy if policy is None else policy
                    router.policy_selector=(lambda iteration,kind:None) if policy is None else None
                    router.query_sensitivity=sensitivity
                    granularity.current=name
                    begin=len(probe.errors)
                    begin_granularity=len(granularity.records)
                    result=original_forward(*args,**forward_kwargs)
                    output[name]=result
                    tile_flags[name]=[(entry[0],entry[-2].clone(),entry[-1].clone()) for entry in router.pending]
                    masks[name]=router.records()
                    router.pending.clear()
                    errors[name]=probe.errors[begin:]
                    granularity_records[name]=granularity.records[begin_granularity:]
                if not torch.equal(output['dense'].logits,output['dense_repeat'].logits):
                    raise AssertionError('Replay changes same-state dense output; cache or runtime mutation suspected')
            finally:
                router.thresholds=original_policy;router.query_sensitivity=original_sensitivity
                router.policy_selector=original_selector
                granularity.current=None
            processor=kwargs['logits_processor'];input_ids=kwargs['input_ids']
            native_step=torch.tensor(cur_step,device=output['dense'].logits.device,dtype=torch.int32)
            processed={name:processor(input_ids,result.logits,cur_step=native_step)
                       for name,result in output.items() if name!='dense_repeat'}
            sampler=kwargs['sampler'];stopper=kwargs['diffusion_stopping_criteria']
            selections={name:accepted_from_logits(value,sampler.entropy_bound)
                        for name,value in processed.items()}
            dense=processed['dense'];dense_accept=selections['dense'][0]
            snapshot=dict(iteration=iteration,remaining_schedule_step=cur_step,
                same_input=True,extra_forward_passes=len(definitions)-1,
                dense_repeat_exact=True,temperature=.4+.4*cur_step/48,
                methods={},position_records=[])
            if original_sensitivity is not None:
                tile=original_sensitivity.reshape(1,2,128)
                snapshot['query_tile_sensitivity']=dict(mean=[float(x) for x in tile.mean(-1).flatten()],
                    std=[float(x) for x in tile.std(-1).flatten()],
                    minimum=[float(x) for x in tile.amin(-1).flatten()],
                    maximum=[float(x) for x in tile.amax(-1).flatten()])
            for name,value in processed.items():
                selected,entropy=selections[name];top=value.argmax(-1)
                history=stopper.argmax_canvas_history
                stable=bool((history==top[None]).all()) if history is not None else False
                confident=bool(entropy.mean()<stopper.confidence_threshold)
                routing=masks[name]
                eligible=sum(x['eligible'] for x in routing);skipped=sum(x['skipped'] for x in routing)
                attention=errors[name]
                error_sq=sum(x['error_sq'] for x in attention);dense_sq=sum(x['dense_sq'] for x in attention)
                snapshot['methods'][name]=dict(top1_disagreement=float((top!=dense.argmax(-1)).float().mean()),
                    acceptance_disagreement=float((selected!=dense_accept).float().mean()),
                    mean_entropy=float(entropy.mean()),stable=stable,confident=confident,
                    would_stop=stable and confident,eligible_tiles=eligible,skipped_tiles=skipped,
                    physical_sparsity=skipped/max(1,eligible),
                    sampled_attention_relative_error=(error_sq/max(dense_sq,1.e-12))**.5,
                    attention_samples=attention,granularity=granularity_records[name],
                    fixed_temperature_confidence=float((output[name].logits.float().topk(1,-1).values.squeeze(-1)/.8-
                        torch.logsumexp(output[name].logits.float()/.8,-1)).exp().mean()))
                if name!='dense':
                    for row in per_position(value,dense,selected,dense_accept,original_sensitivity):
                        index=row['position']
                        row['previous_margin']=None if state.margin is None else float(state.margin[0,index])
                        row['previous_confidence']=None if state.confidence is None else float(state.confidence[0,index])
                        row['temporal_instability']=None if state.temporal is None else float(state.temporal[0,index])
                        row['method']=name;snapshot['position_records'].append(row)
            for target in (50,70):
                left=tile_flags[f'unweighted_s{target}'];right=tile_flags[f'CT_s{target}']
                if len(left)!=len(right):raise AssertionError('Replay layer coverage differs')
                gained=lost=0
                for (layer,a,eligible),(other,b,other_eligible) in zip(left,right):
                    if layer!=other or not torch.equal(eligible,other_eligible):raise AssertionError('Replay geometry differs')
                    gained+=int((a&~b).sum());lost+=int((~a&b).sum())
                snapshot['methods'][f'CT_s{target}']['physical_tiles_newly_retained_vs_unweighted']=gained
                snapshot['methods'][f'CT_s{target}']['physical_tiles_newly_skipped_vs_unweighted']=lost
            records.append(snapshot)
            return output['dense']
        kwargs['decoder_forward']=replay_forward
        return previous(**kwargs)
    model._denoising_step=MethodType(step,model)
    try:yield
    finally:
        if had:model._denoising_step=saved
        else:del model._denoising_step


def run(root,limit=None):
    root=Path(root);cfg=json.loads((root/'configs/configuration.json').read_text())
    frozen=json.loads((root/'configs/frozen_policies.json').read_text())
    rows=json.loads((root/'configs/final_manifest.json').read_text())
    selected=list({r['task']:r for r in reversed(rows)}.values())
    if len(selected)!=13:raise ValueError('Expected one diagnostic sample per task')
    if limit is not None:selected=selected[:limit]
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter=create_adapter('diffusion_gemma',cfg['model'],device='cuda',precision='bfloat16',revision=cfg['revision']).load()
    projections=Projections();summaries=[]
    for row in selected:
        dest=shard_path(root/'diagnostic_replays','matched_dense',row)
        if dest.exists():summaries.append(json.loads(dest.read_text()));continue
        policies=frozen['policies'];records=[]
        with install(adapter,cfg['library'],policies['CT_s70'],mode='value',projections=projections,
                     torch_library=cfg['torch_library'],collect=True) as (binding,router):
            _set_context(binding,row)
            state=State('CT',router,m_ref=frozen['m_ref'],beta=frozen['beta'],gamma=frozen['gamma'],
                        seed=row['seed'],diagnostics=True)
            probe=AttentionProbe(router,[]);granularity=GranularityProbe(probe,router)
            binding.runtime.attention_override=granularity
            with observe(adapter.model,state):
                with replay(adapter.model,router,probe,granularity,state,policies,records):
                    output=adapter.generate(_request(row))
        result=dict(id=row['id'],task=row['task'],prompt_hash=row['prompt_hash'],seed=row['seed'],
                    reference='matched-kernel all-retained dense evolving trajectory',
                    actual_denoising_step_count=output.metadata['actual_denoising_step_count'],
                    snapshots=records,projection_manifest=projections.manifest)
        returned=len(output.completion_tokens)
        for snapshot in records:
            snapshot['returned_answer_tokens']=returned
            for position in snapshot['position_records']:
                position['inside_returned_answer']=position['position']<returned
        atomic(dest,result);summaries.append(result)
        print(json.dumps(dict(event='replay',id=row['id'],snapshots=len(records))),flush=True)
    atomic(root/'diagnostic_replays/summary.json',dict(complete=len(summaries)==13,
        samples=len(summaries),snapshots=sum(len(x['snapshots']) for x in summaries),
        sample_ids=[x['id'] for x in summaries],extra_forwards_excluded_from_timing=True))
    from .query_adaptive_replay_report import report
    report(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3')
    parser.add_argument('--limit',type=int);args=parser.parse_args();run(args.root,args.limit)
