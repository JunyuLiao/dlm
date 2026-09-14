"""Pre-QK selection with strictly post-selection dense diagnostic labels.

Stage-one collector: first eight CONSECUTIVE denoising calls of first canvas,
all layers/heads/query tiles. Later calls execute native dense attention without
diagnostic overhead. This deliberately bounded early-step screen is not a claim
about all-step temporal predictability; online candidates must test full rollouts.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dllm.attention.blasst.core import _prepare_attention_inputs, _finish_eager_attention, _attention_type
from experiments.diffusion_gemma_oracle.routing import signals, select
from .config import TILE, screening_configs
from .state import RequestState, fixed_budget, protected_set, cover_rows

FIELDS=('eligible','skipped','mass_sum','rows','error_sq','dense_sq','query_error_sum',
        'query_error_p95','query_error_max','rescued_rows','budget_excess','important','important_kept',
        'new_important','new_important_missed','prefix_eligible','prefix_skipped','canvas_eligible','canvas_skipped',
        'mixed_eligible','mixed_skipped')


def geometry(valid,width=TILE):
    q,k=valid.shape[-2:]
    tiled=F.pad(valid,(0,(-k)%width,0,(-q)%width)).unflatten(-2,(-1,width)).unflatten(-1,(-1,width))
    return tiled.any((-1,-3)),tiled


def pooled_proxy(query,key,scale):
    def mean(x):
        n=x.shape[-2];pad=(-n)%TILE
        sums=F.pad(x.float(),(0,0,0,pad)).unflatten(-2,(-1,TILE)).sum(-2)
        counts=F.pad(torch.ones(n,device=x.device),(0,pad)).unflatten(-1,(-1,TILE)).sum(-1)
        return sums/counts[:,None]
    return mean(query) @ mean(key).transpose(-2,-1)*scale


def evaluate_masks(masks,eligible,mass,contrib,has,oracle,previous,prefix_length,rescues,excess):
    """FP32 reference softmax/PV, exact joint deletion and renormalization.

    Error percentiles are per head/query-tile, not pooled percentiles. Raw sums
    and counts allow micro-weighting; prompt summaries remain independent units.
    """
    dense=contrib.sum(-2);norm=dense.square().sum(-1).sqrt().clamp_min(1e-12)
    rows=has.sum(-1);dense_sq=(dense.square().sum(-1)*has).sum(-1)
    newly=oracle if previous is None else oracle & ~previous
    starts=torch.arange(eligible.shape[-1],device=eligible.device)*TILE
    prefix_tiles=starts+TILE<=prefix_length;canvas_tiles=starts>=prefix_length
    mixed_tiles=~prefix_tiles&~canvas_tiles
    data=[]
    for i,keep in enumerate(masks):
        retained=(mass*keep[...,None,:]).sum(-1)
        out=torch.einsum('bht,bhqtd->bhqd',keep.float(),contrib)/retained.clamp_min(1e-30)[...,None]
        delta=(out-dense)*has[...,None];error=delta.square().sum(-1).sqrt()/norm
        # Query padding absent in native canvas256; support explicit empty rows.
        masked_error=error.masked_fill(~has,torch.nan)
        p95=torch.nanquantile(masked_error,.95,dim=-1).nan_to_num()
        pmax=error.masked_fill(~has,0).amax(-1)
        values=(eligible.sum(-1),(eligible&~keep).sum(-1),(retained*has).sum(-1),rows,
            delta.square().sum((-1,-2)),dense_sq,(error*has).sum(-1),p95,pmax,rescues[i],excess[i],
            oracle.sum(-1),(oracle&keep).sum(-1),newly.sum(-1),(newly&~keep).sum(-1),
            (eligible&prefix_tiles).sum(-1),(eligible&prefix_tiles&~keep).sum(-1),
            (eligible&canvas_tiles).sum(-1),(eligible&canvas_tiles&~keep).sum(-1),
            (eligible&mixed_tiles).sum(-1),(eligible&mixed_tiles&~keep).sum(-1))
        data.append(torch.stack(values,-1))
    return torch.stack(data)


class ScreeningAttention:
    def __init__(self,request_id,benchmark,output_dir=None,producer=None,steps=8):
        self.state=RequestState(request_id);self.benchmark=benchmark;self.output_dir=Path(output_dir) if output_dir else None
        self.producer=producer;self.steps=steps;self.first_canvas={};self.calls=0;self.files=[]
        self.previous_oracle={};self.last_masks={};self.records=[];self.coverage=set();self.max_state_bytes=0
        self.configs=screening_configs();self.targets=(.4,.5) if benchmark=='aime24' else (.5,.75)

    @torch.no_grad()
    def __call__(self,module,query,key,value,attention_mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kwargs):
        keys,values,valid,scale=_prepare_attention_inputs(module,query,key,value,attention_mask,
            scaling=scaling,is_causal=is_causal,sliding_window=sliding_window)
        runtime=module._blasst_2d_runtime;step=int(runtime.current_denoising_iteration);layer=int(module.layer_idx)
        prefix=key.shape[-2]-query.shape[-2];kind=_attention_type(module,sliding_window)
        # Shape/step gating never depends on current attention scores.
        info=self.first_canvas.setdefault(layer,{'last':-1,'done':False})
        if step<=info['last']:info['done']=True
        observe=not info['done'] and step<self.steps
        info['last']=step;self.calls+=1
        candidates={};meta={};history=None;history_available=False
        if observe:
            eligible,blocked_valid=geometry(valid)
            if self.producer is None:
                raise RuntimeError('producer cache is required, including explicit prefix cache identity')
            norms,identity=self.producer.get(module,query.shape[1])
            history=self.state.get(layer,eligible,step=step,prefix_length=prefix,query_length=query.shape[-2],cache_identity=identity)
            history_available=history.last_step>=0
            proxy=pooled_proxy(query,keys,scale)  # Explicitly paid full K summary scan in this reference screen.
            scores_for_ranking={}
            for name,config in self.configs.items():
                score=history.predict(config,eligible,value_summary=norms.get(config.value_weight),proxy=proxy)
                scores_for_ranking[name]=score
                for target in self.targets:
                    cfg=replace(config,sparsity=target)
                    target_score=self.last_masks[layer,target].float() if name=='last_mask' and (layer,target) in self.last_masks else score
                    if name=='last_mask':scores_for_ranking[f'last_mask_s{int(target*100)}']=target_score
                    protection=protected_set(eligible,prefix,key.shape[-2],cfg.protection)
                    keep,extra=fixed_budget(target_score,eligible,protection,target,salt=42+step*997+layer*31,random_protection=cfg.random_protection)
                    rescues=[]
                    for qi in range(keep.shape[-2]):
                        keep[...,qi,:],rescue=cover_rows(keep[...,qi,:],blocked_valid[...,qi,:,:,:]);rescues.append(rescue)
                    label=f'{name}_s{int(target*100)}';candidates[label]=keep
                    meta[label]=(torch.stack(rescues,-1),extra['budget_excess'])
            # All cheap candidate masks above are final BEFORE the next full-QK line.
        scores=torch.matmul(query,keys.transpose(-2,-1))*scale
        if attention_mask is not None and attention_mask.dtype!=torch.bool:scores=scores+attention_mask[...,:keys.shape[-2]]
        scores=scores.masked_fill(~valid,-torch.inf)
        if observe:
            means=[];peaks=[];tensor_rows=[];rank_rows={};oracle_current={s:[] for s in self.targets}
            names=list(candidates)+[f'oracle_{sig}_s{int(s*100)}' for sig in ('qk','blasst','mass','contribution') for s in self.targets]
            for qi,start in enumerate(range(0,query.shape[-2],TILE)):
                imp,e,mass,contrib,has,vb=signals(scores[...,start:start+TILE,:],valid[...,start:start+TILE,:],values)
                means.append(mass.sum(-2)/has.sum(-1).clamp_min(1)[...,None]);peaks.append(mass.amax(-2))
                masks=[candidates[n][...,qi,:] for n in candidates]
                rescues=[meta[n][0][...,qi] for n in candidates];excess=[meta[n][1][...,qi] for n in candidates]
                for sig in ('qk','blasst','mass','contribution'):
                    for target in self.targets:
                        mask=select(imp[sig],e,'topk',target);mask,rescue=cover_rows(mask,vb)
                        masks.append(mask);rescues.append(rescue);excess.append(torch.zeros_like(rescue))
                per_target={}
                for target in self.targets:
                    oracle,_=cover_rows(select(imp['mass'],e,'topk',target),vb);oracle_current[target].append(oracle)
                    old=self.previous_oracle.get((layer,target))
                    per_target[target]=(oracle,None if old is None or history.last_step<0 else old[...,qi,:])
                # Last-mask baseline reuses previous exact mass mask (dense-history upper reference).
                # Online last_mask must instead reuse its own selected mask and be labeled separately.
                result=[]
                for i,name in enumerate(names):
                    target=int(name.rsplit('_s',1)[1])/100
                    oracle,previous=per_target[target]
                    result.append(evaluate_masks([masks[i]],e,mass,contrib,has,oracle,previous,prefix,[rescues[i]],[excess[i]])[0])
                tensor_rows.append(torch.stack(result))
                if qi==0:
                    # Raw scores for rank diagnostics, all heads, with exact tile IDs.
                    rank_rows['selected_q0']=torch.stack(masks).cpu().numpy()
                    rank_rows.update({n:score[...,qi,:].cpu().numpy() for n,score in scores_for_ranking.items()})
                    rank_rows.update({'oracle_'+n:x.cpu().numpy() for n,x in imp.items()})
                    rank_rows['eligible']=e.cpu().numpy()
                    alpha=mass;dense=contrib.sum(-2)
                    risk=(contrib-alpha[...,None]*dense[...,None,:]).norm(dim=-1)/(1-alpha).clamp_min(1e-9)
                    defined=(alpha<1-1e-6)&vb.any(-1)
                    rank_rows['deletion_risk_mean']=(risk*defined).sum(-2).div(defined.sum(-2).clamp_min(1)).cpu().numpy()
                    rank_rows['deletion_risk_count']=defined.sum(-2).cpu().numpy()
                    rank_rows['value_rms']=norms['rms'].cpu().numpy();rank_rows['value_max']=norms['max'].cpu().numpy()
            history.update(torch.stack(means,-2),torch.stack(peaks,-2),eligible,eligible,step,eligible=eligible)
            for target in self.targets:
                current=torch.stack(oracle_current[target],-2)
                self.previous_oracle[layer,target]=current;self.last_masks[layer,target]=current
            metrics=torch.stack(tensor_rows,dim=3).cpu().numpy() # candidate,B,H,Qtile,field
            record=dict(layer=layer,step=step,attention_type=kind,prefix_length=prefix,kv_length=key.shape[-2],names=names,fields=FIELDS,
                history_available=history_available,source='dense_trajectory',query_tiles=metrics.shape[3])
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True,exist_ok=True);path=self.output_dir/f'layer{layer:02d}_step{step:02d}.npz'
                np.savez_compressed(path,metrics=metrics,**rank_rows);record['path']=str(path);self.files.append(str(path))
            else:record['metrics']=metrics
            self.records.append(record);self.coverage.add((layer,kind,step))
            self.max_state_bytes=max(self.max_state_bytes,self.state.bytes+self.producer.bytes)
        result=_finish_eager_attention(query,values,scores,valid,dropout,module.training)
        if observe and not torch.isfinite(result[0]).all():raise FloatingPointError('nonfinite diagnostic attention output')
        return result
