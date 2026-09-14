"""Paper Algorithm 2 fit and exact physical-margin verification."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from dllm.attention.blasst.core import _prepare_attention_scores, _finish_eager_attention, _attention_type

def margins(scores,valid,q_width=128,kv_width=64):
    valid=torch.broadcast_to(valid,scores.shape)
    b,h,q,k=scores.shape
    shape=(b,h,(q+q_width-1)//q_width,q_width,(k+kv_width-1)//kv_width,kv_width)
    vb=F.pad(valid,(0,(-k)%kv_width,0,(-q)%q_width)).reshape(shape)
    sb=F.pad(scores.float().masked_fill(~valid,-torch.inf),(0,(-k)%kv_width,0,(-q)%q_width),value=-torch.inf).reshape(shape)
    maxima=sb.amax(-1)
    previous=F.pad(maxima.cummax(-1).values[...,:-1],(1,0),value=-torch.inf)
    row_valid=vb.any(-1)
    row_margin=(maxima-previous).masked_fill(~row_valid,-torch.inf)
    return row_margin.amax(-2),row_valid.any(-2)

class Collector:
    def __init__(self): self.values={'local':[],'global':[]};self.calls=[]
    @torch.no_grad()
    def __call__(self,module,q,k,v,mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kw):
        _,ev,s,valid=_prepare_attention_scores(module,q,k,v,mask,scaling=scaling,is_causal=is_causal,sliding_window=sliding_window)
        m,e=margins(s,valid);kind=_attention_type(module,sliding_window)
        length=int(valid.any((1,2)).sum())
        a=m[e].cpu().numpy()
        # log(lambda L) threshold = log(lambda)+log(L); infinity denotes
        # each query's first valid tile, which always remains retained.
        self.values[kind].append(a.astype(np.float64)+math.log(length))
        self.calls.append(dict(attention_type=kind,layer=int(module.layer_idx),head_ids=list(range(q.shape[1])),
            denoising_step=module._blasst_2d_runtime.current_denoising_iteration,valid_kv_length=length,
            eligible_tiles=int(e.sum()),skipped_tiles=0,retained_tiles=int(e.sum()),valid_rows=int(valid.any(-1).sum()),retained_dense_attention_mass=1.))
        return _finish_eager_attention(q,ev,s,valid,dropout,module.training)
    def arrays(self): return {k:np.concatenate(v) for k,v in self.values.items()}

def fit(parts,targets=(.25,.5,.75,.9)):
    """Fit log(lambda L)=log(alpha)+gamma*s, then verify exact counts.

    Each part is one calibration sample's margins adjusted by actual call L.
    A calibration-only monotonic correction records where the paper fit misses
    by >2pp, while preserving the paper's inverse-length deployment rule.
    """
    grid=np.linspace(-20,80,201);points=[]
    arrays=[np.sort(np.asarray(a,dtype=np.float64)) for a in parts]
    for i,a in enumerate(arrays):
        for x in grid:
            s=float(np.searchsorted(a,x,side='left')/len(a))
            if .01<=s<=.97: points.append(dict(sample=i,log_scale=float(x),sparsity=s))
    if len(points)<3: raise ValueError('insufficient calibration range')
    gamma,intercept=np.polyfit([p['sparsity'] for p in points],[p['log_scale'] for p in points],1)
    a=np.sort(np.concatenate(arrays));finite=a[np.isfinite(a)]
    def measured(x): return float(np.searchsorted(a,x,side='left')/len(a))
    policies={}
    for target in targets:
        predicted=float(intercept+gamma*target);selected=predicted;trace=[]
        ceiling=len(finite)/len(a)
        if abs(measured(predicted)-target)>.02:
            lo=float(finite.min()-1);hi=float(finite.max()+1)
            for _ in range(48):
                mid=(lo+hi)/2;s=measured(mid);trace.append(dict(log_scale=mid,sparsity=s))
                if s<target: lo=mid
                else: hi=mid
            selected=min((lo,hi),key=lambda x:abs(measured(x)-target))
        policies[str(target)]=dict(target=target,paper_log_scale=predicted,paper_sparsity=measured(predicted),
            log_scale=selected,achieved_dense_sparsity=measured(selected),unattainable=ceiling<target,
            asymptotic_ceiling=ceiling,correction_trace=trace)
    return dict(alpha=math.exp(float(intercept)),gamma=float(gamma),points=points,targets=policies,
        relation='lambda * valid_KV_length = alpha * exp(gamma * target)',
        correction='calibration-only bisection of scale if paper prediction misses target by >2pp')
