"""Exact dense-trajectory physical-tile margin calibration, including lambda=1."""
import math
import numpy as np
import torch
import torch.nn.functional as F

from dllm.attention.blasst.core import _prepare_attention_scores, _finish_eager_attention, _attention_type
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.calibration import candidate_lambdas, fit_lambda_relation
from .protocol import TARGETS


def physical_margins(scores, valid, width=64):
    """A tile skips iff max(valid-row online margins) < log(lambda)."""
    valid=torch.broadcast_to(valid,scores.shape)
    b,h,q,k=scores.shape
    vb=F.pad(valid,(0,(-k)%width,0,(-q)%width)).reshape(b,h,(q+width-1)//width,width,(k+width-1)//width,width)
    sb=F.pad(scores.float().masked_fill(~valid,-torch.inf),(0,(-k)%width,0,(-q)%width),value=-torch.inf).reshape_as(vb)
    maxima=sb.amax(-1)
    previous=F.pad(maxima.cummax(-1).values[...,:-1],(1,0),value=-torch.inf)
    row_valid=vb.any(-1)
    margins=(maxima-previous).masked_fill(~row_valid,-torch.inf).amax(-2)
    return margins,row_valid.any(-2)


class MarginCollector:
    def __init__(self): self.values={'local':[],'global':[]};self.calls=[]
    @torch.no_grad()
    def __call__(self,module,q,k,v,mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kw):
        ek,ev,s,valid=_prepare_attention_scores(module,q,k,v,mask,scaling=scaling,is_causal=is_causal,sliding_window=sliding_window)
        margin,eligible=physical_margins(s,valid)
        kind=_attention_type(module,sliding_window)
        values=margin[eligible].cpu().numpy();self.values[kind].append(values)
        self.calls.append(dict(attention_type=kind,layer=int(module.layer_idx),head_ids=list(range(q.shape[1])),
            denoising_step=module._blasst_2d_runtime.current_denoising_iteration,
            eligible_tiles=int(eligible.sum()),skipped_tiles=0,retained_tiles=int(eligible.sum()),
            valid_kv_length=int(valid.any((1,2)).sum()),valid_rows=int(valid.any(-1).sum()),retained_dense_attention_mass=1.))
        return _finish_eager_attention(q,ev,s,valid,dropout,module.training)

    def arrays(self):
        return {kind:np.concatenate(v) for kind,v in self.values.items()}


def calibrate(values,length=1.):
    """Use scalar empirical thresholds for this prompt mode, no final-set fitting.

    Existing grid/relation are recorded for continuity. Direct monotonic search
    verifies each selected lambda exactly and avoids assuming fit quality.
    """
    a=np.sort(np.asarray(values,dtype=np.float32))
    if not a.size or np.isnan(a).any(): raise ValueError('missing or NaN physical margins')
    def evaluate(lam):
        n=int(np.searchsorted(a,np.float32(math.log(lam)),side='left'))
        return dict(lambda_value=lam,skipped_tiles=n,eligible_tiles=int(a.size),sparsity=n/a.size)
    grid=[evaluate(x) for x in sorted(set(candidate_lambdas())|{1.})]
    relation=fit_lambda_relation([dict(r,**{'lambda':r['lambda_value']},valid_kv_length=length) for r in grid])
    ceiling=grid[-1]['sparsity']; policies={}
    for target in TARGETS:
        trace=[]
        if ceiling<target:
            chosen=grid[-1];unattainable=True
        else:
            unattainable=False
            low,high=math.log(1e-8),0.
            candidates=list(grid)
            for _ in range(32):
                mid=(low+high)/2;r=evaluate(math.exp(mid));trace.append(r);candidates.append(r)
                if r['sparsity']<target: low=mid
                else: high=mid
            chosen=min(candidates,key=lambda r:(abs(r['sparsity']-target),r['lambda_value']))
        policies[str(target)]=dict(chosen,target_sparsity=target,unattainable=unattainable,
            lambda1_ceiling=ceiling,absolute_error=abs(chosen['sparsity']-target),search_trace=trace)
    return dict(grid=grid,relation_diagnostic=relation,targets=policies,lambda1_ceiling=ceiling)
