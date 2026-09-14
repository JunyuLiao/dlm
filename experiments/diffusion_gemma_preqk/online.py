"""Single-candidate own-history rollouts, deliberately dense-mask emulation.

No current QK or dense diagnostic enters selection. After selection the reference
backend computes full QK; only masked sparse-softmax observations enter sparse
history. Dense-reference history is explicitly paid counterfactual observation
on the candidate's current state, NOT replay of the original dense trajectory.
"""
from dataclasses import replace
import torch
import torch.nn.functional as F
from dllm.attention.blasst.core import _prepare_attention_inputs,_attention_type,_finish_eager_attention
from .config import TILE
from .state import RequestState,fixed_budget,protected_set,cover_rows
from .routing import geometry,pooled_proxy


class CacheIdentity:
    """Identity-only producer hook: history-only routers never scan V for norms."""
    def __init__(self,adapter):
        self.layers={};self.handles=[]
        for name,module in adapter.model.named_modules():
            if adapter.is_blasst_attention_module(name,module):
                self.handles.append(module.register_forward_pre_hook(self._hook,with_kwargs=True))

    def _hook(self,module,args,kwargs):
        cache=kwargs.get('past_key_values',args[3] if len(args)>3 else None)
        if cache is None:identity=(None,)
        else:
            c=cache.layers[int(module.layer_idx)]
            identity=(id(cache),c.keys.data_ptr(),c.values.data_ptr(),tuple(c.keys.shape))
        self.layers[int(module.layer_idx)]=identity

    def get(self,module,heads):return {},self.layers[int(module.layer_idx)]

    @property
    def bytes(self):return 0

    def close(self):
        for h in self.handles:h.remove()


def observations(scores,valid):
    """Row-normalized tile mean/peak from observed scores, no dropped labels."""
    has=valid.any(-1);p=torch.softmax(torch.where(has[...,None],scores.float(),0),-1)*valid
    q,k=p.shape[-2:];pad=(-k)%TILE
    block=F.pad(p,(0,pad)).unflatten(-1,(-1,TILE)).sum(-1)
    means=[];peaks=[]
    for start in range(0,q,TILE):
        has_q=has[...,start:start+TILE];mass=block[...,start:start+TILE,:]
        means.append(mass.sum(-2)/has_q.sum(-1).clamp_min(1)[...,None]);peaks.append(mass.amax(-2))
    return torch.stack(means,-2),torch.stack(peaks,-2)


def direct_diagnostics(scores,valid,token_keep,values):
    """Exact jointly masked FP32 output, without materializing block vectors.

    Same softmax probabilities and renormalization as the oracle's contribution
    sum; this changes only diagnostic arithmetic, never executed generation.
    Returns additive per-head statistics and separately labeled tail statistics.
    """
    has=valid.any(-1)
    p=torch.softmax(torch.where(has[...,None],scores.float(),0),-1)*valid
    retained=(p*token_keep).sum(-1)
    dense=p @ values.float()
    sparse=(p*token_keep/retained.clamp_min(1e-30)[...,None]) @ values.float()
    delta=(sparse-dense)*has[...,None]
    error=delta.norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-12)
    return dict(mass_sum=(retained*has).sum(-1),rows=has.sum(-1),error_sq=delta.square().sum((-1,-2)),
        dense_sq=(dense.square().sum(-1)*has).sum(-1),query_error_sum=(error*has).sum(-1),
        query_error_p95=torch.nanquantile(error.masked_fill(~has,torch.nan),.95,dim=-1).nan_to_num(),
        query_error_max=error.masked_fill(~has,0).amax(-1))


class OnlineAttention:
    def __init__(self,request_id,config,producer,*,diagnostics=False,isolate=None,allocation_policy=None):
        if config.execution!='reference':raise NotImplementedError('no compatible measured sparse backend integrated')
        if config.allocation not in ('uniform','paired_layer_shift'):raise NotImplementedError(config.allocation)
        if (config.allocation=='paired_layer_shift')!=(allocation_policy is not None):raise ValueError('explicit frozen allocation policy required')
        from .allocation import PairedLayerAllocation
        self.allocator=PairedLayerAllocation(allocation_policy) if allocation_policy is not None else None
        self.isolate_fired=False
        self.config=config;self.producer=producer;self.diagnostics=diagnostics;self.isolate=isolate
        self.state=RequestState(request_id);self.buckets={};self.calls=0;self.max_state_bytes=0
        self.last_decisions=None;self.costs=dict(full_qk_elements=0,full_pv_elements=0,eligible_tiles=0,retained_tiles=0,
            initialized_tiles=0,refresh_tiles=0,exploration_tiles=0,proxy_key_elements=0,
            generation_qk_flops=0,generation_pv_flops=0,diagnostic_pv_elements=0,diagnostic_pv_flops=0,history_mask_selections=0,
            isolated_sparse_calls=0)

    @property
    def records(self):return list(self.buckets.values())

    @torch.no_grad()
    def __call__(self,module,query,key,value,attention_mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kwargs):
        if dropout or module.training:raise ValueError('evaluation-only router requires dropout0/eval')
        keys,values,valid,scale=_prepare_attention_inputs(module,query,key,value,attention_mask,
            scaling=scaling,is_causal=is_causal,sliding_window=sliding_window)
        layer=int(module.layer_idx);step=int(module._blasst_2d_runtime.current_denoising_iteration)
        prefix=keys.shape[-2]-query.shape[-2];kind=_attention_type(module,sliding_window)
        eligible,blocks=geometry(valid);norms,identity=self.producer.get(module,query.shape[1]);cfg=self.config
        history=self.state.get(layer,eligible,step=step,prefix_length=prefix,query_length=query.shape[-2],cache_identity=identity)
        initialize=history.last_step<0
        refresh=not initialize and bool(cfg.refresh_interval) and step%cfg.refresh_interval==0
        isolate_target=self.isolate is not None and not self.isolate_fired and (layer,step)==tuple(self.isolate)
        isolated_dense=self.isolate is not None and not isolate_target
        if isolate_target and (initialize or refresh):raise RuntimeError('isolated perturbation lacks causal history or coincides with refresh')
        if isolate_target:self.isolate_fired=True;self.costs['isolated_sparse_calls']+=1
        active=not (initialize or refresh or isolated_dense)
        delta,base_budget=(self.allocator.shift(layer,kind,eligible,cfg.sparsity,active) if self.allocator else (None,None))
        shadow_keep=None
        if initialize or refresh or isolated_dense:
            keep=eligible.clone();extra={'budget_excess':torch.zeros_like(eligible.sum(-1)),
                'forced':torch.zeros_like(eligible.sum(-1)),'protected':torch.zeros_like(eligible.sum(-1))}
        else:
            proxy=pooled_proxy(query,keys,scale) if cfg.estimator=='proxy' else None
            if proxy is not None:self.costs['proxy_key_elements']+=keys.numel()
            score=history.predict(cfg,eligible,value_summary=norms.get(cfg.value_weight),proxy=proxy)
            protected=protected_set(eligible,prefix,keys.shape[-2],cfg.protection)
            keep,extra=fixed_budget(score,eligible,protected,cfg.sparsity,
                salt=42+step*997+layer*31,random_protection=cfg.random_protection,
                explore=cfg.exploration_tiles,age=history.age,budget_delta=delta)
            if self.allocator and self.diagnostics:
                shadow_keep,_=fixed_budget(score,eligible,protected,cfg.sparsity,
                    salt=42+step*997+layer*31,random_protection=cfg.random_protection,
                    explore=cfg.exploration_tiles,age=history.age)
        rescues=[]
        for qi in range(keep.shape[-2]):
            keep[...,qi,:],rescue=cover_rows(keep[...,qi,:],blocks[...,qi,:,:,:]);rescues.append(rescue)
            if shadow_keep is not None:shadow_keep[...,qi,:],_=cover_rows(shadow_keep[...,qi,:],blocks[...,qi,:,:,:])
        if self.allocator:self.allocator.observe(layer,kind,keep,base_budget)
        if self.allocator and self.diagnostics and shadow_keep is None:shadow_keep=keep.clone()
        # This immutable causal mask is chosen before any current full-QK operation.
        self.last_decisions=keep.clone()
        token_keep=keep.repeat_interleave(TILE,-2).repeat_interleave(TILE,-1)[...,:query.shape[-2],:keys.shape[-2]]
        scores=query @ keys.transpose(-2,-1)*scale
        if attention_mask is not None and attention_mask.dtype!=torch.bool:scores=scores+attention_mask[...,:keys.shape[-2]]
        scores=scores.masked_fill(~valid,-torch.inf);masked=scores.masked_fill(~token_keep,-torch.inf)
        observation_valid=valid if cfg.history_source=='dense_reference' else valid&token_keep
        mean,peak=observations(scores if cfg.history_source=='dense_reference' else masked,observation_valid)
        observed=eligible if cfg.history_source=='dense_reference' else keep
        age=history.age.clone();observation_counts=history.observations.clone()
        history.update(mean,peak,observed,keep,step,eligible=eligible,rho=cfg.rho,
            conditional=cfg.history_source=='sparse' and not(initialize or refresh or isolated_dense),
            correct=cfg.renormalization=='estimated_coverage')
        if cfg.estimator=='last_mask':
            # Match the screen's previous IMPORTANT mask semantics. An executed
            # mask alone is not evidence of importance, and dense initialization
            # would otherwise make the next prediction an uninformative all-ones
            # mask. Acquire the next binary estimate from permitted observations
            # only, after this step's immutable decision. Charge the extra sort.
            mean_config=replace(cfg,estimator='last_mass',kv_neighborhood='none',query_neighborhood='none',value_weight='none')
            estimate=history.predict(mean_config,eligible)
            next_mask,_=fixed_budget(estimate,eligible,torch.zeros_like(eligible),cfg.sparsity)
            for qi in range(next_mask.shape[-2]):
                next_mask[...,qi,:],_=cover_rows(next_mask[...,qi,:],blocks[...,qi,:,:,:])
            history.last_mask=next_mask;self.costs['history_mask_selections']+=1
        start_tiles=torch.arange(eligible.shape[-1],device=query.device)*TILE
        prefix_tiles=start_tiles+TILE<=prefix;canvas_tiles=start_tiles>=prefix;mixed=~prefix_tiles&~canvas_tiles
        packed=torch.stack([eligible.sum(-1),(eligible&~keep).sum(-1),
            (eligible&prefix_tiles).sum(-1),(eligible&prefix_tiles&~keep).sum(-1),
            (eligible&canvas_tiles).sum(-1),(eligible&canvas_tiles&~keep).sum(-1),
            (eligible&mixed).sum(-1),(eligible&mixed&~keep).sum(-1),
            torch.stack(rescues,-1),extra['budget_excess'],(age*eligible).sum(-1),
            (observation_counts*eligible).sum(-1),((observation_counts==0)&eligible).sum(-1)],-1)
        packed=packed.sum(-2).cpu().tolist() # B,H,field, aggregated across query tiles
        fields=('eligible','skipped','prefix_eligible','prefix_skipped','canvas_eligible','canvas_skipped',
            'mixed_eligible','mixed_skipped','rescued_rows','budget_excess','age_sum','observation_count_sum','unknown_tiles')
        for b,heads in enumerate(packed):
            for h,data in enumerate(heads):
                index=(layer,step,h)
                record=self.buckets.setdefault(index,dict(layer=layer,step=step,head=h,attention_type=kind,calls=0,
                    initialization_calls=0,refresh_calls=0,**{f:0. for f in fields}))
                for field,x in zip(fields,data):record[field]+=x
                record['calls']+=1;record['initialization_calls']+=int(initialize);record['refresh_calls']+=int(refresh)
        if self.diagnostics:
            d=direct_diagnostics(scores,valid,token_keep,values)
            fields=tuple(d);packed=torch.stack(list(d.values()),-1).cpu().tolist()
            for b,heads in enumerate(packed):
                for h,data in enumerate(heads):
                    record=self.buckets[layer,step,h]
                    for field,x in zip(fields,data):
                        name='sparse_state_'+field
                        if field=='query_error_max':record[name]=max(record.get(name,0),x)
                        elif field=='query_error_p95':record[name+'_sum']=record.get(name+'_sum',0)+x
                        else:record[name]=record.get(name,0)+x
                    record['sparse_state_diagnostic_calls']=record.get('sparse_state_diagnostic_calls',0)+1
            if shadow_keep is not None:
                shadow_tokens=shadow_keep.repeat_interleave(TILE,-2).repeat_interleave(TILE,-1)[...,:query.shape[-2],:keys.shape[-2]]
                shadow=direct_diagnostics(scores,valid,shadow_tokens,values)
                shadow.update(eligible=eligible.sum((-1,-2)),retained=shadow_keep.sum((-1,-2)))
                fields=tuple(shadow)
                for b,heads in enumerate(torch.stack(list(shadow.values()),-1).cpu().tolist()):
                    for h,data in enumerate(heads):
                        record=self.buckets[layer,step,h]
                        for field,x in zip(fields,data):
                            name='shadow_uniform_'+field
                            if field=='query_error_p95':name+='_sum'
                            record[name]=max(record.get(name,0),x) if field=='query_error_max' else record.get(name,0)+x
                        record['shadow_uniform_diagnostic_calls']=record.get('shadow_uniform_diagnostic_calls',0)+1
        elements=query.shape[0]*query.shape[1]*query.shape[-2]*keys.shape[-2]
        self.costs['full_qk_elements']+=elements;self.costs['full_pv_elements']+=elements
        self.costs['generation_qk_flops']+=2*elements*query.shape[-1]
        self.costs['generation_pv_flops']+=2*elements*values.shape[-1]
        if self.diagnostics:
            factor=2 if shadow_keep is None else 4
            self.costs['diagnostic_pv_elements']+=factor*elements
            self.costs['diagnostic_pv_flops']+=2*factor*elements*values.shape[-1]
        self.costs['eligible_tiles']+=int(eligible.sum());self.costs['retained_tiles']+=int(keep.sum())
        self.costs['initialized_tiles']+=int(keep.sum()) if initialize else 0
        self.costs['refresh_tiles']+=int(keep.sum()) if refresh else 0
        self.costs['exploration_tiles']+=int((extra['forced']-extra['protected']).sum())
        self.calls+=1;self.max_state_bytes=max(self.max_state_bytes,self.state.bytes+self.producer.bytes)
        result=_finish_eager_attention(query,values,masked,valid,0.,False)
        if not torch.isfinite(result[0]).all():raise FloatingPointError('nonfinite online attention output')
        return result
