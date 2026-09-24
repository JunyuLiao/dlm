"""Small-state reference diagnostics for 128-row physical decisions.

Not the production router. Used only on sampled same-QKV replays to count
row-level hypothetical skips and rows dominating physical tile maxima.
"""
import torch


def route_rows(state,reference,sensitivity,threshold):
    z=state['logz'];mu=state['mu'];active=state['active'];eligible=state['eligible']
    previous=torch.full_like(z[...,0],-torch.inf)
    output=torch.zeros_like(mu[...,0,:])
    physical=[];row_skip=[];dominators=[]
    for j in range(z.shape[-1]):
        a=active[...,j];support=torch.isfinite(previous)
        combined=torch.logaddexp(previous,z[...,j]);safe=torch.where(torch.isfinite(combined),combined,0.)
        alpha=torch.exp(z[...,j]-safe).masked_fill(~a,0.)
        risk=(alpha[...,None]*(mu[...,j,:]-output)).norm(dim=-1)/reference[...,None].clamp_min(1.e-12)
        weighted=risk*sensitivity[:,None,:]
        log_risk=torch.where(a,torch.where(support,weighted.log(),torch.inf),-torch.inf)
        skip=eligible[...,j]&(log_risk.amax(-1)<threshold)
        physical.append(skip)
        row_skip.append(a&support&(log_risk<threshold))
        dominators.append(log_risk.argmax(-1))
        include=a&~skip[...,None]
        newz=torch.logaddexp(previous,z[...,j].masked_fill(~include,-torch.inf))
        safe_new=torch.where(torch.isfinite(newz),newz,0.)
        oldscale=torch.exp(previous-safe_new)
        newscale=torch.exp(z[...,j].masked_fill(~include,-torch.inf)-safe_new)
        output=oldscale[...,None]*output+newscale[...,None]*mu[...,j,:]
        previous=newz
    return dict(physical=torch.stack(physical,-1),row_skip=torch.stack(row_skip,-1),
                dominator=torch.stack(dominators,-1))


class Probe:
    def __init__(self,inner,router):self.inner,self.router=inner,router;self.current=None;self.records=[]
    def __call__(self,module,q,k,v,mask,**kwargs):
        result=self.inner(module,q,k,v,mask,**kwargs)
        if self.current not in ('unweighted_s70','CT_s70') or int(module.layer_idx) not in (0,5,29):return result
        from dllm.attention.blasst.core import _attention_validity,_attention_type
        from experiments.diffusion_gemma_jl_output_aware.reference import block_statistics
        if q.shape[-2]<128:return result
        valid=_attention_validity(mask,q,k,is_causal=bool(kwargs.get('is_causal',False)),
                                  sliding_window=kwargs.get('sliding_window'))
        hk=k.shape[1];h=q.shape[1]
        validkv=valid.reshape(q.shape[0],hk,h//hk,q.shape[-2],k.shape[-2]).any((2,3))
        z,ref=self.router.cache.get(int(module.layer_idx),v,validkv,max(0,k.shape[-2]-q.shape[-2]))
        qq=q[:,:1,:128];kk=k[:,:1];vv=valid[:,:1,:128]
        scores=(qq@kk.transpose(-1,-2))*kwargs.get('scaling',q.shape[-1]**-.5)
        if mask is not None and mask.dtype!=torch.bool:
            scores+=mask[:,:1,:128]
        state=block_statistics(scores,vv,z[:,:1])
        sensitivity=self.router.query_sensitivity
        if sensitivity is None:sensitivity=torch.ones((q.shape[0],q.shape[-2]),device=q.device)
        kind=_attention_type(module,kwargs.get('sliding_window'))
        threshold=self.router.thresholds[kind]['log_threshold']
        routed=route_rows(state,ref[:,:1],sensitivity[:,:128],threshold)
        actual=self.router.pending[-1][-2][:,0,0]
        estimate=routed['physical'][:,0]
        votes=routed['row_skip'][:,0]
        active=state['active'][:,0]
        eligible=state['eligible'][:,0]
        mixed=(votes.sum(-2)>0)&(~estimate)&eligible
        dominant=routed['dominator'][:,0]
        leader=sensitivity[:,:128].gather(-1,dominant).flatten()
        self.records.append(dict(method=self.current,layer=int(module.layer_idx),kind=kind,
            eligible=int(eligible.sum()),actual_skipped=int(actual.sum()),reference_skipped=int(estimate.sum()),
            reference_disagreement=int((actual^estimate).sum()),
            hypothetical_row_skip_votes=int(votes.sum()),valid_row_tile_pairs=int(active.sum()),
            mixed_query_tiles=int(mixed.sum()),
            dominating_row_sensitivity_mean=float(leader.mean()),
            all_row_sensitivity_mean=float(sensitivity[:,:128].mean())))
        return result
