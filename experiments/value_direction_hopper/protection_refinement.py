"""Protection-only decoder interventions and physical-query-tile rescue.

Never changes logits, predicted argmax, or the native stopping criterion.
Rescue is a reference two-pass implementation; both passes are accounted for.
"""
from contextlib import contextmanager
from types import MethodType
import torch
from .intervention_policies import protection


def variants():
    return {
        'existing':dict(mode='existing'),
        'hysteresis':dict(mode='refined',consistency=True,margin=0.,hysteresis=True),
        'confidence_only':dict(mode='refined',consistency=True,margin=0.,hysteresis=False),
        'strict_margin':dict(mode='refined',consistency=True,margin=.999,hysteresis=False),
        'immediate':dict(mode='refined',consistency=False,margin=0.,hysteresis=False),
        'very_confident':dict(mode='refined',consistency=True,immediate_p=.9999,margin=0.,hysteresis=False),
        'rescue_conservative':dict(mode='existing',rescue='conservative'),
        'rescue_dense':dict(mode='existing',rescue='dense'),
    }


def update_protection(spec,was,stored,top,prob,margin,old_top,stored_prob):
    if spec['mode'] in ('static','dense','native'):return torch.zeros_like(top,dtype=torch.bool),top
    if spec['mode']=='existing':return protection(was,stored,top,prob,margin,old_top)
    if was is None:was=torch.zeros_like(top,dtype=torch.bool);stored=top.clone()
    if spec['hysteresis']:
        # Margin is the best fresh alternative minus the protected token's
        # probability, not the fresh maximum on its own.
        reopen=(stored_prob<.90)&((prob-stored_prob)>.05)
        sustain=was&~reopen
    else:sustain=was&(top==stored)&(prob>=.98)
    enter=(prob>=.999)&(margin>=spec['margin'])
    if spec['consistency']:
        enter &= (torch.zeros_like(top,dtype=torch.bool) if old_top is None else top==old_top) | (prob>=spec.get('immediate_p',2.))
    enter &= ~sustain
    return sustain|enter,torch.where(enter,top,stored)


class State:
    def __init__(self,spec,thresholds,diagnostics=True):
        self.spec,self.thresholds,self.diagnostics=spec,thresholds,diagnostics
        self.step=0;self.top=None;self.protected=None;self.stored=None;self.entropy=None;self.margin=None
        self.churn=None;self.rescue_rows=None;self.records=[];self.current={};self.decoder_events=[];self.policy_events=[]
        self.work=[];self.rescue_calls=0
    def begin(self):
        self.step+=1;self.rescue_rows=None
        if self.spec.get('rescue') and self.top is not None and self.step<=5:
            difficult=(self.entropy>.005)|(self.margin<.98)|self.churn
            # Protected queries remain eligible if their fresh prediction is
            # unstable. Bound logical rescue to8 rows before physical closure.
            risk=self.entropy.float()+self.churn.float()
            available=difficult.flatten().nonzero().flatten()
            if available.numel():
                chosen=available[risk.flatten()[available].topk(min(8,len(available))).indices]
                self.rescue_rows=torch.zeros_like(self.top,dtype=torch.bool)
                self.rescue_rows.flatten()[chosen]=True


class Sampler:
    def __init__(self,inner,state):self.inner,self.state=inner,state
    def __getattr__(self,name):return getattr(self.inner,name)
    def accept_canvas(self,current,proposed,logits,cur_step):
        s=self.state
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if s.spec.get('profile'):start.record()
        answer=self.inner.accept_canvas(current,proposed,logits,cur_step)
        if s.spec['mode'] in ('static','dense','native') and not s.diagnostics:
            # Profiling dense/static must not add protection work that normal
            # inference does not require.
            return answer
        native=self.inner.accepted_token_mask.clone()
        x=logits.float();z=torch.logsumexp(x,-1);v=x.topk(2,-1).values
        p=(v[...,0]-z).exp();margin=p-(v[...,1]-z).exp();top=logits.argmax(-1)
        entropy=torch.distributions.Categorical(logits=logits).entropy()
        was=torch.zeros_like(top,dtype=torch.bool) if s.protected is None else s.protected
        oldstored=top if s.stored is None else s.stored
        stored_prob=(x.gather(-1,oldstored[...,None]).squeeze(-1)-z).exp()
        churn=torch.zeros_like(top,dtype=torch.bool) if s.top is None else s.top!=top
        marked,stored=update_protection(s.spec,s.protected,s.stored,top,p,margin,s.top,stored_prob)
        accepted=native|marked
        answer=torch.where(marked,stored,answer)
        self.inner.accepted_token_mask=accepted
        if s.spec.get('profile'):end.record();s.policy_events.append((start,end))
        if s.diagnostics:
            def cpu(t):return t.flatten().detach().cpu().tolist()
            s.current=dict(step=s.step,protected=int(marked.sum()),newly_protected=int((marked&~was).sum()),
                reopened=int((was&~marked).sum()),accepted=int(accepted.sum()),renoised=int((~accepted).sum()),
                churn=int(churn.sum()),protected_churn=int((was&churn).sum()),unprotected_churn=int((~was&churn).sum()),
                entropy=float(entropy.mean()),confidence=float(p.mean()),margin=float(margin.mean()),
                top1=cpu(top),entropy_positions=cpu(entropy),confidence_positions=cpu(p),margin_positions=cpu(margin),
                previous_protected=cpu(was),protected_positions=cpu(marked),protected_tokens=cpu(stored),
                reopened_positions=cpu(was&~marked),churn_positions=cpu(churn),stored_probability=cpu(stored_prob),
                logical_rescued=0 if s.rescue_rows is None else int(s.rescue_rows.sum()))
        s.protected,s.stored,s.top=marked,stored,top.detach().clone()
        s.entropy,s.margin,s.churn=entropy,margin,churn
        return answer


class Stopper:
    def __init__(self,inner,state):self.inner,self.state=inner,state
    def __call__(self,top,logits):
        history=self.inner.argmax_canvas_history
        unstable=torch.ones_like(top,dtype=torch.bool) if history is None else (history!=top[None]).any(0)
        stable=torch.ones(top.shape[0],dtype=torch.bool,device=top.device) if self.inner.stability_threshold==0 else ~unstable.any(-1)
        entropy=torch.distributions.Categorical(logits=logits).entropy()
        confident=entropy.mean(-1)<self.inner.confidence_threshold
        result=self.inner(top,logits)
        if not torch.equal(result,stable&confident):raise AssertionError('Native stopper reconstruction mismatch')
        self.state.current.update(stable=bool(stable.item()),confident=bool(confident.item()),
            stopper_unstable=unstable.flatten().cpu().tolist(),would_stop=bool(result.item()))
        return result


@contextmanager
def control(model,state,score_draft=None):
    original=model._denoising_step;had='_denoising_step' in model.__dict__;saved=model.__dict__.get('_denoising_step')
    def step(this,**kwargs):
        state.begin();kwargs['sampler']=Sampler(kwargs['sampler'],state)
        if state.diagnostics:kwargs['diffusion_stopping_criteria']=Stopper(kwargs['diffusion_stopping_criteria'],state)
        forward=kwargs['decoder_forward']
        if state.spec.get('profile'):
            def measured(*args,**kw):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();out=forward(*args,**kw);b.record();state.decoder_events.append((a,b));return out
            kwargs['decoder_forward']=measured
        result=original(**kwargs)
        if state.diagnostics:
            r=state.current;r['input_length']=int(kwargs['input_ids'].shape[-1]);r['terminal']=bool(result[3].item()) or int(kwargs['cur_step'])==1
            if score_draft:r.update(score_draft(result[1][0]))
            state.records.append(r)
        return result
    model._denoising_step=MethodType(step,model)
    try:yield state
    finally:
        if had:model._denoising_step=saved
        else:del model._denoising_step


class Rescue:
    """Two full passes with physical128-query closure, clearly cost-accounted."""
    def __init__(self,router,state):self.router,self.state=router,state
    def __call__(self,module,q,k,v,mask,**kwargs):
        s=self.state;base=self.router(module,q,k,v,mask,**kwargs)
        if s.rescue_rows is None or not bool(s.rescue_rows.any()):return base
        n=q.shape[-2]
        rowmask=s.rescue_rows.reshape(-1)
        if n!=len(rowmask):raise ValueError('Rescue query geometry mismatch')
        tiles=rowmask.reshape(-1,128).any(-1)
        first=self.router.pending.pop() if self.router.collect else None
        original=self.router.thresholds
        self.router.thresholds={kind:dict(log_threshold=(-float('inf') if s.spec['rescue']=='dense' else entry['log_threshold']-1.)) for kind,entry in original.items()}
        try:safer=self.router(module,q,k,v,mask,**kwargs)
        finally:self.router.thresholds=original
        if self.router.collect:
            second=self.router.pending.pop();masktiles=tiles[None,None,:,None]
            merged=torch.where(masktiles,second[-2],first[-2]);eligible=first[-1]
            self.router.pending.append((*first[:-2],merged,eligible))
            s.work.append(dict(step=s.step,layer=int(module.layer_idx),base_retained=int((first[-1]&~first[-2]).sum()),
                executed_retained=int((first[-1]&~first[-2]).sum()+(second[-1]&~second[-2]).sum()),
                extra_output_retained=int((eligible&~merged).sum()-(first[-1]&~first[-2]).sum()),
                extra_qk_tiles=int(second[-1].sum()),physical_rescued_query_tiles=int(tiles.sum())))
            s.work[-1].update(base_eligible=int(first[-1].sum()),base_skipped=int(first[-2].sum()),
                             safer_eligible=int(second[-1].sum()),safer_skipped=int(second[-2].sum()),
                             output_skipped=int(merged.sum()),attention_type=first[2])
        s.rescue_calls+=1
        select=tiles.repeat_interleave(128)[None,:,None,None]
        return torch.where(select,safer[0],base[0]),None
