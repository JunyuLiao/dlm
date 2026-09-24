"""Causally available, device-resident per-query weights for Gaussian-32.

Only wraps the native denoising step and sampler. It never replaces proposals,
acceptance, renoising, self-conditioning, or native stopping.
"""
from contextlib import contextmanager
from types import MethodType

import torch


METHODS=('M','C','T','MT','CT')


def weight(method, margin, confidence, temporal, *, beta=3., m_ref=1.):
    """All inputs are B,Q FP32 and come from completed prior iterations."""
    if method not in METHODS:raise ValueError(method)
    if beta<0 or m_ref<=0:raise ValueError('Positive m_ref and nonnegative beta required')
    if margin is None:
        if confidence is not None or temporal is not None:raise ValueError('Partial history')
        return None
    sm=1+beta*m_ref/(margin.clamp_min(0)+m_ref)
    sc=1+beta*(1-confidence).clamp_min(0).sqrt()
    st=1+beta*temporal
    result={'M':sm,'C':sc,'T':st,'MT':(sm*st).sqrt(),'CT':(sc*st).sqrt()}[method]
    return result.clamp(1,1+beta).contiguous()


def shuffle_within_tiles(values, generator, tile=128):
    """Permute association with rows while preserving each physical tile's values."""
    chunks=[]
    for begin in range(0,values.shape[-1],tile):
        part=values[:,begin:begin+tile]
        permutation=torch.randperm(part.shape[-1],device=part.device,generator=generator)
        chunks.append(part[:,permutation])
    return torch.cat(chunks,dim=-1).contiguous()


class State:
    def __init__(self,method,router,*,m_ref,beta=3.,gamma=.5,allocation='normal',bootstrap=False,seed=42,diagnostics=True,collect_margins=False):
        if method not in METHODS+('unweighted','kernel_dense','native_dense'):raise ValueError(method)
        if allocation not in ('normal','shuffle','uniform'):raise ValueError(allocation)
        if not 0<=gamma<1:raise ValueError('Invalid gamma')
        self.method,self.router,self.m_ref,self.beta,self.gamma=method,router,m_ref,beta,gamma
        self.allocation,self.bootstrap,self.diagnostics=allocation,bootstrap,diagnostics
        self.collect_margins=collect_margins
        self.seed=seed;self.generator=None;self.canvas=-1;self.iteration=0
        self.margin=None;self.confidence=None;self.temporal=None;self.previous_top=None
        self.current=None;self.steps=[];self.canvases=[];self.used_weights=None
        self.margin_samples=[];self.profile_events=[]

    def begin(self,cur_step,canvas):
        if cur_step==48 or self.canvas<0:
            if self.canvas>=0:self.finish_canvas()
            self.canvas+=1;self.iteration=0;self.margin=self.confidence=self.temporal=self.previous_top=None
            self.generator=torch.Generator(device=canvas.device).manual_seed(self.seed+7919*self.canvas+104729)
        self.iteration+=1
        chosen=None
        if self.method in METHODS and self.margin is not None:
            chosen=weight(self.method,self.margin,self.confidence,self.temporal,beta=self.beta,m_ref=self.m_ref)
            if self.allocation=='uniform':chosen=chosen.mean(-1,keepdim=True).expand_as(chosen).contiguous()
            elif self.allocation=='shuffle':chosen=shuffle_within_tiles(chosen,self.generator)
        self.used_weights=chosen
        if self.router is not None:
            self.router.query_sensitivity=chosen
            if self.bootstrap:self.router.policy_selector=lambda iteration,kind:None if self.iteration==1 else self.router.thresholds[kind]
        self.current=dict(canvas_index=self.canvas,iteration=self.iteration,
            remaining_schedule_step=cur_step,weight_active=chosen is not None,
            bootstrap_dense=bool(self.bootstrap and self.iteration==1))
        if self.diagnostics and chosen is not None:
            x=chosen.detach().float().flatten();q=torch.quantile(x,torch.tensor([.1,.5,.9],device=x.device))
            self.current.update(sensitivity_mean=float(x.mean()),sensitivity_p10=float(q[0]),
                sensitivity_p50=float(q[1]),sensitivity_p90=float(q[2]))

    def observe_logits(self,logits,accepted,cur_step):
        # Native logits_processor has applied temperature by this point. The
        # frozen configuration contains no other prediction processor. Undo
        # ONLY that scalar for the raw-logit margin, without changing logits.
        x=logits.float();two=x.topk(2,dim=-1).values
        logz=torch.logsumexp(x,dim=-1)
        probability=(two[...,0]-logz).exp()
        temperature=.4+.4*(cur_step/48)
        margin=((two[...,0]-two[...,1])*temperature).clamp_min(0)
        top=logits.argmax(-1)
        flip=None if self.previous_top is None else top!=self.previous_top
        if self.temporal is None:self.temporal=torch.zeros_like(probability)
        if flip is not None:self.temporal=self.gamma*self.temporal+(1-self.gamma)*flip.float()
        self.previous_top=top.detach();self.margin=margin.detach();self.confidence=probability.detach()
        if self.diagnostics:
            entropy=torch.distributions.Categorical(logits=logits).entropy()
            p=torch.quantile(probability.flatten(),torch.tensor([.1,.5,.9],device=x.device))
            m=torch.quantile(margin.flatten(),torch.tensor([.1,.5,.9],device=x.device))
            self.current.update(temperature=temperature,confidence_mean=float(probability.mean()),
                confidence_p10=float(p[0]),confidence_p50=float(p[1]),confidence_p90=float(p[2]),
                margin_mean=float(margin.mean()),margin_p10=float(m[0]),margin_p50=float(m[1]),margin_p90=float(m[2]),
                processed_entropy_mean=float(entropy.mean()),accepted=int(accepted.sum()),
                renoised=int((~accepted).sum()),argmax_flips=None if flip is None else int(flip.sum()))
            if self.collect_margins:self.margin_samples.extend(margin[margin>0].flatten().detach().cpu().tolist())

    def finish_step(self,result,cur_step):
        if self.diagnostics:
            self.current.update(native_stop=bool(result[3].item()),cap_stop=cur_step==1,
                completed_by_native_or_cap=bool(result[3].item()) or cur_step==1)
            self.steps.append(self.current)

    def finish_canvas(self):
        if self.canvas<0:return
        if self.diagnostics:
            group=[s for s in self.steps if s['canvas_index']==self.canvas]
            if group:self.canvases.append(dict(canvas_index=self.canvas,iterations=len(group),
                stopped_natively=group[-1]['native_stop'],hit_cap=group[-1]['cap_stop'],
                final_entropy=group[-1].get('processed_entropy_mean'),
                final_flips=group[-1].get('argmax_flips'),dense_bootstrap_steps=sum(s['bootstrap_dense'] for s in group)))


class Sampler:
    def __init__(self,inner,state):self.inner,self.state=inner,state
    def __getattr__(self,name):return getattr(self.inner,name)
    def accept_canvas(self,current,proposed,logits,cur_step):
        result=self.inner.accept_canvas(current,proposed,logits,cur_step)
        if self.state.method in METHODS or self.state.diagnostics or self.state.collect_margins:
            self.state.observe_logits(logits,self.inner.accepted_token_mask,self.state.current['remaining_schedule_step'])
        return result


@contextmanager
def observe(model,state):
    original=model._denoising_step;had='_denoising_step' in model.__dict__;saved=model.__dict__.get('_denoising_step')
    def step(this,**kwargs):
        cur_step=int(kwargs['cur_step']);state.begin(cur_step,kwargs['current_canvas'])
        kwargs['sampler']=Sampler(kwargs['sampler'],state)
        result=original(**kwargs)
        state.finish_step(result,cur_step)
        return result
    model._denoising_step=MethodType(step,model)
    try:yield state
    finally:
        state.finish_canvas()
        if state.router is not None:
            state.router.query_sensitivity=None;state.router.policy_selector=None
        if had:model._denoising_step=saved
        else:del model._denoising_step
