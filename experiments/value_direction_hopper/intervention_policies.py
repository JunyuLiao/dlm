"""Decode-time interventions; no changes to attention kernels or stopper."""
from contextlib import contextmanager
from types import MethodType
import torch


def choose_dense(schedule, step, previous_dense=False, previous=None, limits=None):
    """One-based step. Signals from a completed transition affect the next call."""
    if schedule=='dense':return True
    if schedule=='first':return step==1
    if schedule=='late':return step>=4
    if schedule!='reactive' or previous is None or previous_dense:return False
    if previous['churn'] is None:return False
    return (previous['churn']>limits['churn'] or
            (previous['entropy']>limits['entropy_floor'] and previous['entropy_progress']<=limits['progress']) or
            (previous['accepted_fraction']<.99 and previous['acceptance_progress']<=limits['acceptance_progress']))


def protection(previous, stored, top, confidence, margin, old_top, strong=.999, weak=.98):
    if previous is None:previous=torch.zeros_like(top,dtype=torch.bool);stored=top.clone()
    sustain=previous & (top==stored) & (confidence>=weak)
    establish=(confidence>=strong)&(margin>=.98)
    if old_top is None:establish.zero_()
    else:establish &= top==old_top
    marked=sustain|establish
    return marked,torch.where(establish,top,stored)


def additional(mask,entropy,confidence,k,random_generator=None):
    """Shared candidate rule and exact per-step k; independent random generator."""
    rejected=(~mask).flatten().nonzero().flatten()
    extra=torch.zeros_like(mask)
    if not rejected.numel():return extra,[]
    order=rejected[torch.argsort(entropy.flatten()[rejected],stable=True)]
    pool=order[:min(32,len(order))]
    pool=pool[confidence.flatten()[pool]>=.5]
    n=min(k,len(pool))
    if random_generator is not None:
        indices=torch.randperm(len(pool),generator=random_generator,device=mask.device)[:n]
    else:indices=torch.argsort(confidence.flatten()[pool],descending=True,stable=True)[:n]
    extra.flatten()[pool[indices]]=True
    return extra,pool.detach().cpu().tolist()


class Sampler:
    def __init__(self,inner,state):self.inner,self.state=inner,state
    def __getattr__(self,name):return getattr(self.inner,name)
    def accept_canvas(self,current,proposed,logits,cur_step):
        state=self.state;policy=state.policy
        if not state.diagnostics and policy.get('acceptance','none')=='none' and policy['schedule']!='reactive':
            return self.inner.accept_canvas(current,proposed,logits,cur_step)
        entropy=torch.distributions.Categorical(logits=logits).entropy()
        x=logits.float();values=x.topk(2,-1).values
        probs=(values-torch.logsumexp(x,-1)[...,None]).exp()
        confidence=probs[...,0];margin=probs[...,0]-probs[...,1];top=logits.argmax(-1)
        result=self.inner.accept_canvas(current,proposed,logits,cur_step)
        native=self.inner.accepted_token_mask.clone();accepted=native.clone()
        change=policy.get('acceptance','none');effective_bound=float(self.inner.entropy_bound)
        extra=torch.zeros_like(native);pool=[]
        if change=='adaptive':
            progress=float(native.float().mean())
            effective_bound*=1+policy['strength']*progress
            values_,indices=entropy.sort(-1)
            keep=values_.cumsum(-1)-values_<=effective_bound
            accepted=torch.zeros_like(keep).scatter(-1,indices,keep)
            result=torch.where(accepted,proposed,current)
        elif change in ('random','ranked'):
            extra,pool=additional(native,entropy,confidence,policy['k'],state.generator if change=='random' else None)
            accepted|=extra;result=torch.where(accepted,proposed,current)
        elif change=='protect':
            state.protected,state.protected_tokens=protection(state.protected,state.protected_tokens,top,confidence,margin,state.previous_top,
                                                            strong=policy['strong'],weak=policy['weak'])
            accepted|=state.protected
            result=torch.where(state.protected,state.protected_tokens,result)
        self.inner.accepted_token_mask=accepted
        previous=state.previous_stats
        h=float(entropy.float().mean());fraction=float(accepted.float().mean())
        churn=None if state.previous_top is None else float((top!=state.previous_top).float().mean())
        record=dict(step=state.step,dense=state.dense,entropy=h,confidence=float(confidence.mean()),
            confidence_median=float(confidence.median()),accepted_fraction=fraction,accepted=int(accepted.sum()),
            native_accepted=int(native.sum()),extra_accepted=int((accepted&~native).sum()),churn=churn,
            entropy_progress=None if previous is None else previous['entropy']-h,
            acceptance_progress=None if previous is None else fraction-previous['accepted_fraction'],
            entropy_bound=effective_bound,protected=int(state.protected.sum()) if state.protected is not None else 0)
        if state.diagnostics:
            record.update(top1=top.flatten().cpu().tolist(),accepted_positions=accepted.flatten().cpu().tolist(),
                protected_positions=[] if state.protected is None else state.protected.flatten().cpu().tolist(),
                protected_tokens=[] if state.protected_tokens is None else state.protected_tokens.flatten().cpu().tolist(),
                added_positions=extra.flatten().cpu().tolist(),candidate_positions=pool,
                confidence_per_position=confidence.flatten().cpu().tolist())
        state.current=record;state.previous_stats=record;state.previous_top=top.detach().clone()
        return result


class Controller:
    def __init__(self,policy,thresholds,seed,device,limits,diagnostics=True):
        self.policy,self.thresholds,self.limits=policy,thresholds,limits
        self.generator=torch.Generator(device=device).manual_seed(seed+104729)
        self.step=0;self.dense=False;self.previous_stats=None;self.previous_top=None
        self.protected=None;self.protected_tokens=None;self.current=None;self.records=[];self.diagnostics=diagnostics
    def selector(self,iteration,kind):
        return None if self.dense else self.thresholds[kind]


@contextmanager
def control(model,state,draft_score=None):
    original=model._denoising_step;had='_denoising_step' in model.__dict__;saved=model.__dict__.get('_denoising_step')
    def step(this,**kwargs):
        state.step+=1
        state.dense=choose_dense(state.policy['schedule'],state.step,state.dense,state.previous_stats,state.limits)
        kwargs['sampler']=Sampler(kwargs['sampler'],state)
        result=original(**kwargs)
        if state.diagnostics:
            record=state.current
            record['terminated']=bool(result[3].item()) or int(kwargs['cur_step'])==1
            record['input_length']=int(kwargs['input_ids'].shape[-1])
            if draft_score is not None:record['draft_score']=draft_score(result[1][0])
            state.records.append(record)
        return result
    model._denoising_step=MethodType(step,model)
    try:yield state
    finally:
        if had:model._denoising_step=saved
        else:del model._denoising_step
