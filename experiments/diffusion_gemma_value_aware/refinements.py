"""Evidence-driven ablations, separate from the frozen initial operators.

The screen found loose normalized-mass/running-output estimates harmful.
These ablations isolate those two components. Exact block mass still requires
block exponentials/summation: deleted blocks omit PV, NOT block softmax.
Previous-output risk uses only a cached output from the preceding denoising
step at the same layer/canvas/prefix, never the current dense output.

The adapter uses scoped function injection because the initial reference code
and its already completed caches are frozen. It is deliberately single-worker,
single-thread instrumentation, like the existing model attention overrides.
No custom CUDA/Triton code is used.
"""
from dataclasses import replace
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from . import routing as base_routing
from .operators import (Config,KV_TILE,Q_TILE,aggregate_risk,dense_previous,
    log_risk,rescue,select,streaming_mask)
from .protocol import sha

MODES=('exact_mass','previous_output','previous_output_exact_mass')
CONFIGS={'mass_exact':dict(method='mass',mode='exact_mass'),
    'centered_previous':dict(method='centered',mode='previous_output'),
    'centered_previous_exact':dict(method='centered',mode='previous_output_exact_mass')}


def source_sha():return sha(Path(__file__).read_bytes())


def refined_risk(config,state,meta,m,z,out,j=None,previous_output=None):
    """Exact online block mass, optional previous-step output surrogate."""
    b=state['b'] if j is None else state['b'][...,j]
    n=state['count'] if j is None else state['count'][...,j]
    blockz=state['logz'] if j is None else state['logz'][...,j]
    if 'previous_output' in config.mode and previous_output is not None:
        # Broadcast one past per-query output across present candidate blocks.
        out=previous_output[...,None,:] if j is None else previous_output
    risk=log_risk(config,b,n,m,z,out,meta,j)
    if 'exact_mass' in config.mode:
        # Replace only alpha_hat's log factor; all other factors unchanged.
        boundz=b+n.clamp_min(1).float().log()
        log_bound=boundz-torch.logaddexp(z,boundz)
        # -softplus(z-blockz) avoids subtracting nearly equal large logs when
        # the current block's online probability is extremely close to1.
        log_exact=-F.softplus(z-blockz)
        if config.method=='mass':risk=log_exact
        else:risk=risk-log_bound+log_exact
    return torch.where(n>0,torch.where(torch.isfinite(z),risk,torch.inf),-torch.inf)


def refined_screen_risks(state,meta,config,previous_output=None):
    m,z,out=dense_previous(state)
    r=refined_risk(config,state,meta,m,z,out,previous_output=previous_output)
    return aggregate_risk(r,state['active'],config),r


def refined_mask(state,meta,config,previous_output=None):
    if config.mode not in MODES:return streaming_mask(state,meta,config)
    b,z=state['b'],state['logz']
    m=torch.full_like(b[...,0],-torch.inf);running_z=m.clone()
    out=torch.zeros_like(state['block_mean'][...,0,:]);masks=[]
    for j in range(b.shape[-1]):
        r=refined_risk(config,state,meta,m,running_z,out,j,previous_output)
        score=aggregate_risk(r[...,None],state['active'][...,j,None],config).squeeze(-1)
        skip=state['eligible'][...,j]&(score<config.log_threshold);masks.append(skip)
        include=state['active'][...,j]&~skip[...,None]
        addz=z[...,j].masked_fill(~include,-torch.inf)
        newz=torch.logaddexp(running_z,addz);safez=torch.where(torch.isfinite(newz),newz,0.)
        a,c=torch.exp(running_z-safez),torch.exp(addz-safez)
        # Retained-prefix output supports the warm-up fallback only. New
        # exact-mass mass-only routing never needs the current output at all.
        if config.method=='centered':out=a[...,None]*out+c[...,None]*state['block_mean'][...,j,:]
        running_z=newz;m=torch.maximum(m,b[...,j].masked_fill(~include,-torch.inf))
    return torch.stack(masks,-1)


class RefinedAttention(base_routing.Attention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.past_outputs={};self.previous=None;self.query_start=0
        self.previous_calls=0;self.warmup_calls=0;self.refinement_diagnostics=[]

    def route(self,state,meta,config):
        n=state['b'].shape[-2]
        previous=None if self.previous is None else self.previous[...,self.query_start:self.query_start+n,:]
        self.query_start+=n
        return refined_mask(state,meta,config,previous)

    @torch.no_grad()
    def __call__(self,module,q,k,v,mask,**kwargs):
        step=int(module._blasst_2d_runtime.current_denoising_iteration);layer=int(module.layer_idx)
        prefix=max(0,k.shape[-2]-q.shape[-2]);cached=self.past_outputs.get(layer)
        # A new canvas/prefix, nonconsecutive step or shape change invalidates
        # the estimate. Warm-up uses the original retained running output.
        self.previous=(cached['output'] if cached is not None and cached['step']==step-1
            and cached['prefix']==prefix and cached['shape']==tuple(q.shape) and step>0 else None)
        self.query_start=0
        if self.previous is None:self.warmup_calls+=1
        else:self.previous_calls+=1
        with patch.object(base_routing,'streaming_mask',self.route):
            output=super().__call__(module,q,k,v,mask,**kwargs)
        self.past_outputs[layer]=dict(step=step,prefix=prefix,shape=tuple(q.shape),
            output=output[0].transpose(1,2).float().detach().clone())
        if 'exact_mass' in self.config.mode:
            for head in range(q.shape[1]):self.buckets[layer,head,step,'execution']['softmax_skipped']=0.
        return output

    def _screen(self,state,meta,q,k,valid,scale,layer,step,kind,prefix,length):
        n=q.shape[-2]
        previous=None if self.previous is None else self.previous[...,self.query_start:self.query_start+n,:]
        self.query_start+=n
        for name,config in CONFIGS.items():
            c=Config(**config)
            risks,rows=refined_screen_risks(state,meta,c,previous)
            self.risk_arrays.setdefault(f'{name}__{kind}',[]).append(risks[state['eligible']].float().cpu().numpy())
            for target in (.25,.5,.75,.9):
                keep,empty=rescue(select(risks,state['eligible'],'topk',target),state)
                probe=f'refine/{name}/s{int(target*100)}'
                self.record(state,state['eligible']&~keep,c,meta,layer,step,kind,probe,prefix,empty)
                if 'exact_mass' in c.mode:
                    for head in range(q.shape[1]):self.buckets[layer,head,step,probe]['softmax_skipped']=0.
        # Diagnose estimator looseness and query-tile risk inflation directly.
        m,z,out=dense_previous(state);b=state['b'];count=state['count']
        boundz=b+count.clamp_min(1).float().log()
        bound=boundz-torch.logaddexp(z,boundz)
        exact=state['logz']-torch.logaddexp(z,state['logz'])
        active=state['active']&torch.isfinite(z)
        error=(bound-exact)[active].cpu().numpy()
        ratios=[]
        for c in (Config(method='mass'),Config(method='centered')):
            risk=log_risk(c,b,count,m,z,out,meta)
            maximum=risk.masked_fill(~active,-torch.inf).amax(-2)
            mean=torch.logsumexp(risk.masked_fill(~active,-torch.inf),-2)-active.sum(-2).clamp_min(1).log()
            x=(maximum-mean)[active.any(-2)].cpu().numpy()
            ratios.append(dict(method=c.method,count=len(x),log_worst_over_mean_quantiles=np.quantile(x,[.1,.5,.9,.99]).tolist() if len(x) else []))
        self.refinement_diagnostics.append(dict(layer=layer,step=step,attention_type=kind,previous_output_available=previous is not None,
            mass_bound_log_overestimate_count=len(error),mass_bound_log_overestimate_quantiles=np.quantile(error,[.1,.5,.9,.99]).tolist() if len(error) else [],
            row_inflation=ratios))


def cache_refined(adapter,row,path,fp,config=None,thresholds=None,screen=False,root=None):
    """Reuse the frozen generator/cache with explicit additional provenance."""
    from . import run
    import json
    code_sha=source_sha()
    if path.exists():
        d=json.loads(path.read_text())
        if d.get('refinement_sha256')!=code_sha:raise RuntimeError('refinement source changed; use a new phase directory')
    original=run.generate
    def generate(*args,**kwargs):
        routers=[]
        def factory(*a,**kw):
            router=RefinedAttention(*a,**kw);routers.append(router);return router
        with patch.object(run,'Attention',factory):
            # Reuse prompt/coverage/generation checks; the scoped class injection
            # is restored even on failure and cannot affect another condition.
            result,arrays=original(*args,**kwargs)
        result['refinement_sha256']=code_sha
        result['refinement_diagnostics']=routers[0].refinement_diagnostics
        result['previous_output_coverage']=dict(available_calls=routers[0].previous_calls,warmup_calls=routers[0].warmup_calls)
        return result,arrays
    kwargs=dict(config=config,thresholds=thresholds,screen=screen)
    if root is not None:kwargs['root']=root
    with patch.object(run,'generate',generate):return run.cache(adapter,row,path,fp,**kwargs)
