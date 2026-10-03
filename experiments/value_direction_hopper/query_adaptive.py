"""Causally available, device-resident per-query weights for Gaussian-32.

Only wraps the native denoising step and sampler. It never replaces proposals,
acceptance, renoising, self-conditioning, or native stopping.
"""
from contextlib import contextmanager
import math
from types import MethodType

import torch


METHODS=('M','C','T','MT','CT')
PROPOSAL_METHODS=('T_prior','T_hybrid','T_run')
CAUSAL_METHODS=('M_prior','C_prior','T_smooth')
ANCHOR_METHODS=('H_anchor','C_soft')
GATED_METHODS=('M_gate','M_gate_norm','C_gate','T_gate')
TAIL_METHODS=('C_tail',)
SOFT_GATE_METHODS=('C_gate_soft',)
UNIFORM_METHODS=(PROPOSAL_METHODS+CAUSAL_METHODS+ANCHOR_METHODS+GATED_METHODS+
                 TAIL_METHODS+SOFT_GATE_METHODS)
ALL_METHODS=(METHODS+PROPOSAL_METHODS+CAUSAL_METHODS+ANCHOR_METHODS+GATED_METHODS+
             TAIL_METHODS+SOFT_GATE_METHODS)


def percentile_rank(values):
    """Per-canvas average ranks in [0,1]; equal values get equal ranks.

    Runs on the input device without copying query history to the host.
    A singleton has rank zero because it has no relative tail information.
    """
    width = values.shape[-1]
    if width <= 1:
        return torch.zeros_like(values)
    sorted_values = values.sort(dim=-1).values.contiguous()
    query = values.contiguous()
    left = torch.searchsorted(sorted_values, query, right=False)
    right = torch.searchsorted(sorted_values, query, right=True)
    return (left + right - 1).to(values.dtype) / (2 * (width - 1))


def weight(method, margin, confidence, temporal, *, beta=3., m_ref=1.,
           trajectory=None, drift=None, run_uncertainty=None,
           stable_run=None, gate_tau=2.5, margin_scale=1., entropy=None,
           anchor_lambda=.9, soft_prior_alpha=.25, tail_tau=.75,
           tail_lambda=.5, soft_gate_lambda=.2):
    """Return a bounded per-query value-direction risk coefficient.

    ``T`` is the historical flip-EMA coefficient.  ``T_prior`` completes the
    missing early flip history with a causal per-position trajectory prior:
    ``trajectory`` is the EMA of the previous sampler's renoised mask.  Its
    hazard is

    ``trajectory + (1 - trajectory) * temporal``.

    ``T_hybrid`` adds top-1 confidence drift with a noisy-OR hazard, while
    ``T_run`` replaces the renoising EMA with exponentially decaying
    uncertainty from consecutive accepted calls.  ``M_prior`` and ``C_prior``
    apply the same causal trajectory completion to the historical margin and
    confidence signals.  ``T_smooth`` is the temporal completion with a
    separately configurable, slower trajectory EMA (the state supplies that
    EMA rate).  The ``*_gate`` variants delay relaxation using a causal
    stable-run gate. ``M_gate_norm`` uses a robust log-margin scale fitted
    once on dense calibration trajectories and then frozen. ``H_anchor`` uses
    a convex blend of the causal trajectory prior and processed entropy,
    weighted by ``anchor_lambda``. ``C_soft`` uses the confidence prior with a
    tempered trajectory contribution: ``alpha*q + (1-alpha*q)*u_C``.
    ``C_tail`` keeps absolute confidence uncertainty and adds a causal lift
    only to the high-uncertainty percentile tail.
    ``C_gate_soft`` keeps the gate trajectory but replaces its hard floor with
    a small confidence-dependent floor controlled by ``soft_gate_lambda``.

    Thus a position that is still being revised remains protected even when it
    has not flipped yet, while an accepted, non-flipping position relaxes back
    toward coefficient one.  All tensors are B,Q FP32 and come from completed
    prior iterations.  The first call is initialized by ``State.begin`` with
    coefficient ``1 + beta`` because no history exists yet.
    """
    if method not in ALL_METHODS:raise ValueError(method)
    if beta<0 or m_ref<=0:raise ValueError('Positive m_ref and nonnegative beta required')
    # The historical M/C/T methods need a complete first-step history.  The
    # causal uniform methods deliberately accept a missing first-step history;
    # State.begin supplies their maximum-sensitivity prior for that call.
    if method not in UNIFORM_METHODS and margin is None:
        if confidence is not None or temporal is not None:raise ValueError('Partial history')
        return None
    if method in UNIFORM_METHODS:
        if temporal is None:
            raise ValueError(f'{method} requires temporal history')
        if method == 'T_run':
            if run_uncertainty is None:
                run_uncertainty = torch.ones_like(temporal)
            if run_uncertainty.shape != temporal.shape:
                raise ValueError('run uncertainty and temporal history must have the same shape')
            run_uncertainty = run_uncertainty.clamp(0, 1)
            hazard = run_uncertainty + (1 - run_uncertainty) * temporal.clamp(0, 1)
        else:
            if trajectory is None:
                trajectory = torch.ones_like(temporal)
            if trajectory.shape != temporal.shape:
                raise ValueError('trajectory and temporal history must have the same shape')
            trajectory = trajectory.clamp(0, 1)
            gated = method in GATED_METHODS
            if method in ('T_prior','T_smooth','T_gate'):
                base_uncertainty = temporal.clamp(0, 1)
            elif method in ('M_prior','M_gate'):
                if margin is None:
                    raise ValueError(f'{method} requires margin history')
                base_uncertainty = m_ref / (margin.clamp_min(0) + m_ref)
            elif method == 'M_gate_norm':
                if margin is None:
                    raise ValueError('M_gate_norm requires margin history')
                if not math.isfinite(margin_scale) or margin_scale <= 0:
                    raise ValueError('Invalid margin_scale')
                base_uncertainty = ((math.log1p(m_ref) -
                    torch.log1p(margin.clamp_min(0))) / margin_scale).clamp(0, 1)
            elif method in ('C_prior','C_gate','C_tail','C_gate_soft'):
                if confidence is None:
                    raise ValueError(f'{method} requires confidence history')
                if method == 'C_tail':
                    if not math.isfinite(tail_tau) or not 0 <= tail_tau < 1:
                        raise ValueError('Invalid tail_tau')
                    if not math.isfinite(tail_lambda) or not 0 <= tail_lambda <= 1:
                        raise ValueError('Invalid tail_lambda')
                if method == 'C_gate_soft':
                    if not math.isfinite(soft_gate_lambda) or not 0 <= soft_gate_lambda <= 1:
                        raise ValueError('Invalid soft_gate_lambda')
                base_uncertainty = (1 - confidence).clamp_min(0).sqrt()
            elif method == 'H_anchor':
                if entropy is None:
                    raise ValueError('H_anchor requires entropy history')
                if not math.isfinite(anchor_lambda) or not 0 <= anchor_lambda <= 1:
                    raise ValueError('Invalid anchor_lambda')
                base_uncertainty = (entropy / (1 + entropy.clamp_min(0))).clamp(0, 1)
            elif method == 'C_soft':
                if confidence is None:
                    raise ValueError('C_soft requires confidence history')
                if not math.isfinite(soft_prior_alpha) or not 0 <= soft_prior_alpha <= 1:
                    raise ValueError('Invalid soft_prior_alpha')
                base_uncertainty = (1 - confidence).clamp_min(0).sqrt()
            else:
                if drift is None:
                    drift = torch.zeros_like(temporal)
                if drift.shape != temporal.shape:
                    raise ValueError('drift and temporal history must have the same shape')
                drift = drift.clamp(0, 1)
                base_uncertainty = 1 - (1 - temporal.clamp(0, 1)) * (1 - drift)
            base_uncertainty = base_uncertainty.clamp(0, 1)
            if method == 'H_anchor':
                hazard = ((1 - anchor_lambda) * trajectory +
                          anchor_lambda * base_uncertainty).clamp(0, 1)
            elif method == 'C_soft':
                hazard = (soft_prior_alpha * trajectory +
                          (1 - soft_prior_alpha * trajectory) * base_uncertainty).clamp(0, 1)
            elif method == 'C_tail':
                # Keep the absolute uncertainty level, then lift only the
                # high-uncertainty tail using completed trajectory history.
                ranks = percentile_rank(base_uncertainty)
                tail = ((ranks - tail_tau) / (1 - tail_tau)).clamp(0, 1)
                hazard = (base_uncertainty + tail_lambda * trajectory * tail *
                          (1 - base_uncertainty)).clamp(0, 1)
            elif method == 'C_gate_soft':
                if stable_run is None:
                    stable_run = torch.zeros_like(temporal)
                if stable_run.shape != temporal.shape:
                    raise ValueError('stable run and temporal history must have the same shape')
                if not math.isfinite(gate_tau) or gate_tau <= 0:
                    raise ValueError('Invalid gate_tau')
                gate = -torch.expm1(-stable_run.clamp_min(0) / gate_tau)
                a = gate * (1 - trajectory)
                a = soft_gate_lambda + (1 - soft_gate_lambda) * a
                hazard = 1 - a * (1 - base_uncertainty)
            elif gated:
                if stable_run is None:
                    stable_run = torch.zeros_like(temporal)
                if stable_run.shape != temporal.shape:
                    raise ValueError('stable run and temporal history must have the same shape')
                if not math.isfinite(gate_tau) or gate_tau <= 0:
                    raise ValueError('Invalid gate_tau')
                gate = -torch.expm1(-stable_run.clamp_min(0) / gate_tau)
                # At stable_run=0, h=1. After stable accepted calls, this
                # converges to the ordinary trajectory-prior hazard.
                hazard = 1 - gate * (1 - trajectory) * (1 - base_uncertainty)
            else:
                hazard = trajectory + (1 - trajectory) * base_uncertainty
        return (1 + beta * hazard).clamp(1, 1 + beta).contiguous()
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
    def __init__(self,method,router,*,m_ref,beta=3.,gamma=.5,trajectory_gamma=None,
                 gate_tau=2.5,margin_scale=1.,anchor_lambda=.9,soft_prior_alpha=.25,
                 tail_tau=.75,tail_lambda=.5,
                 soft_gate_lambda=.2,
                 allocation='normal',bootstrap=False,
                 seed=42,diagnostics=True,collect_margins=False,
                 collect_all_stats=None):
        if method not in ALL_METHODS+('unweighted','kernel_dense','native_dense'):raise ValueError(method)
        if allocation not in ('normal','shuffle','uniform'):raise ValueError(allocation)
        if not 0<=gamma<1:raise ValueError('Invalid gamma')
        self.method,self.router,self.m_ref,self.beta,self.gamma=method,router,m_ref,beta,gamma
        self.trajectory_gamma = gamma if trajectory_gamma is None else float(trajectory_gamma)
        if not 0 <= self.trajectory_gamma < 1: raise ValueError('Invalid trajectory_gamma')
        self.gate_tau=float(gate_tau); self.margin_scale=float(margin_scale)
        self.anchor_lambda=float(anchor_lambda)
        self.soft_prior_alpha=float(soft_prior_alpha)
        self.tail_tau=float(tail_tau)
        self.tail_lambda=float(tail_lambda)
        self.soft_gate_lambda=float(soft_gate_lambda)
        if any(not math.isfinite(v) or v <= 0 for v in (self.gate_tau, self.margin_scale)):
            raise ValueError('Invalid gate parameters')
        if not math.isfinite(self.anchor_lambda) or not 0 <= self.anchor_lambda <= 1:
            raise ValueError('Invalid anchor_lambda')
        if not math.isfinite(self.soft_prior_alpha) or not 0 <= self.soft_prior_alpha <= 1:
            raise ValueError('Invalid soft_prior_alpha')
        if not math.isfinite(self.tail_tau) or not 0 <= self.tail_tau < 1:
            raise ValueError('Invalid tail_tau')
        if not math.isfinite(self.tail_lambda) or not 0 <= self.tail_lambda <= 1:
            raise ValueError('Invalid tail_lambda')
        if not math.isfinite(self.soft_gate_lambda) or not 0 <= self.soft_gate_lambda <= 1:
            raise ValueError('Invalid soft_gate_lambda')
        self.allocation,self.bootstrap,self.diagnostics=allocation,bootstrap,diagnostics
        self.collect_margins=collect_margins
        # Keep complete traces for diagnostics, margin collection, and the
        # router-less offline audit API. Routed production states with
        # diagnostics disabled use method-specific statistics instead.
        if collect_all_stats is None:
            self.collect_all_stats = diagnostics or collect_margins or router is None
        else:
            self.collect_all_stats = bool(collect_all_stats)
        self.seed=seed;self.generator=None;self.canvas=-1;self.iteration=0
        self.margin=None;self.confidence=None;self.temporal=None;self.trajectory=None
        self.entropy=None
        self.drift=None;self.stable_run=None;self.previous_top=None
        self.observed=False
        self.current=None;self.steps=[];self.canvases=[];self.used_weights=None
        self.margin_samples=[];self.profile_events=[]

    def _stat_requirements(self):
        """Return only the completed-call statistics this method consumes."""
        if self.collect_all_stats:
            return dict(top=True, confidence=True, margin=True, entropy=True,
                        temporal=True, drift=True, stable_run=True)
        method=self.method
        # Keep the historical composite implementation byte-for-byte
        # compatible: ``weight`` evaluates all three component tensors before
        # selecting M/C/T/MT/CT.
        if method in METHODS:
            return dict(top=True, confidence=True, margin=True, entropy=False,
                        temporal=True, drift=False, stable_run=False)
        needs_confidence=method in (
            'C','CT','C_prior','C_gate','C_tail','C_gate_soft','C_soft','T_hybrid')
        needs_margin=method in ('M','MT','M_prior','M_gate','M_gate_norm')
        needs_entropy=method=='H_anchor'
        needs_temporal=method in (
            'T','MT','CT','T_prior','T_smooth','T_hybrid','T_gate')
        needs_stable_run=method in (
            'T_run','M_gate','M_gate_norm','C_gate','T_gate','C_gate_soft')
        # Temporal EMAs and stable-run gates need the completed top-1 winner.
        needs_top=needs_temporal or needs_stable_run
        return dict(top=needs_top, confidence=needs_confidence,
                    margin=needs_margin, entropy=needs_entropy,
                    temporal=needs_temporal, drift=method=='T_hybrid',
                    stable_run=needs_stable_run)

    def begin(self,cur_step,canvas):
        if cur_step==48 or self.canvas<0:
            if self.canvas>=0:self.finish_canvas()
            self.canvas+=1;self.iteration=0
            self.margin=self.confidence=self.temporal=self.trajectory=None
            self.entropy=None
            self.drift=self.stable_run=self.previous_top=None
            self.observed=False
            self.generator=torch.Generator(device=canvas.device).manual_seed(self.seed+7919*self.canvas+104729)
        self.iteration+=1
        chosen=None
        # T_prior deliberately starts with a full uncertainty prior.  This is
        # the only non-history initialization; later calls use only completed
        # sampler outcomes and previous logits.
        if self.method in UNIFORM_METHODS and not self.observed:
            chosen=torch.full(canvas.shape,1+self.beta,device=canvas.device,
                              dtype=torch.float32).contiguous()
        elif self.method in ALL_METHODS and self.observed:
            chosen=weight(self.method,self.margin,self.confidence,self.temporal,
                          beta=self.beta,m_ref=self.m_ref,
                          trajectory=self.trajectory,drift=self.drift,
                          run_uncertainty=None if self.stable_run is None else
                          torch.exp(-self.stable_run/2.),
                          stable_run=self.stable_run, gate_tau=self.gate_tau,
                          margin_scale=self.margin_scale, entropy=self.entropy,
                          anchor_lambda=self.anchor_lambda,
                          soft_prior_alpha=self.soft_prior_alpha,
                          tail_tau=self.tail_tau, tail_lambda=self.tail_lambda,
                          soft_gate_lambda=self.soft_gate_lambda)
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
        requirements=self._stat_requirements()
        # Native logits_processor has applied temperature by this point. The
        # frozen configuration contains no other prediction processor. Undo
        # ONLY that scalar for the raw-logit margin, without changing logits.
        temperature=.4+.4*(cur_step/48)
        x=None;top=None;top_value=None;probability=None;margin=None;entropy=None
        if requirements['confidence'] or requirements['margin'] or requirements['entropy']:
            x=logits.float()
        if requirements['confidence']:
            # C_gate needs top-1 confidence, not top-2.  Use the indexed max
            # only when the method also needs the winner identity for flips.
            if requirements['top']:
                top_value,top=x.max(dim=-1)
            else:
                top_value=x.amax(dim=-1)
            probability=(top_value-torch.logsumexp(x,dim=-1)).exp()
        elif requirements['top']:
            top=logits.argmax(-1)
        if requirements['margin']:
            two=x.topk(2,dim=-1).values
            margin=((two[...,0]-two[...,1])*temperature).clamp_min(0)
        if requirements['entropy']:
            entropy=torch.distributions.Categorical(logits=logits).entropy()

        flip=None if top is None or self.previous_top is None else top!=self.previous_top
        if requirements['temporal']:
            if self.temporal is None:self.temporal=torch.zeros_like(top,dtype=torch.float32)
            if flip is not None:self.temporal=self.gamma*self.temporal+(1-self.gamma)*flip.float()
        elif self.temporal is None:
            # ``weight`` validates this argument for all uniform methods even
            # when the selected formula does not use temporal flips.
            self.temporal=torch.zeros(accepted.shape,device=accepted.device,dtype=torch.float32)

        if self.method in UNIFORM_METHODS:
            # The sampler's acceptance mask is the causal signal that a query
            # position still needs trajectory work. It is available only
            # after the current attention call.
            renoised=(~accepted).float()
            if self.trajectory is None:self.trajectory=torch.ones_like(renoised)
            tg=self.trajectory_gamma
            self.trajectory=tg*self.trajectory+(1-tg)*renoised
            if requirements['drift']:
                if self.drift is None:self.drift=torch.zeros_like(renoised)
                if self.confidence is not None:
                    self.drift=self.gamma*self.drift+(1-self.gamma)*(
                        probability-self.confidence).abs().clamp(0,1)
            if requirements['stable_run']:
                if self.stable_run is None:self.stable_run=torch.zeros_like(renoised)
                self.stable_run=(self.stable_run+1.)*accepted.float()
                if flip is not None:self.stable_run=self.stable_run.masked_fill(flip,0.)

        self.previous_top=None if top is None else top.detach()
        self.margin=None if margin is None else margin.detach()
        self.confidence=None if probability is None else probability.detach()
        # Stored after the completed call; H_anchor can only consume it on the
        # next State.begin, so current-call logits never affect current routing.
        self.entropy=None if entropy is None else entropy.detach()
        self.observed=True
        if self.diagnostics:
            # diagnostics implies collect_all_stats, so these values are
            # present. Keep the historical record schema unchanged.
            entropy=self.entropy
            p=torch.quantile(probability.flatten(),torch.tensor([.1,.5,.9],device=x.device))
            m=torch.quantile(margin.flatten(),torch.tensor([.1,.5,.9],device=x.device))
            self.current.update(temperature=temperature,confidence_mean=float(probability.mean()),
                confidence_p10=float(p[0]),confidence_p50=float(p[1]),confidence_p90=float(p[2]),
                margin_mean=float(margin.mean()),margin_p10=float(m[0]),margin_p50=float(m[1]),margin_p90=float(m[2]),
                processed_entropy_mean=float(entropy.mean()),accepted=int(accepted.sum()),
                renoised=int((~accepted).sum()),argmax_flips=None if flip is None else int(flip.sum()))
            if self.method in UNIFORM_METHODS:
                q=torch.quantile(self.trajectory.flatten(),torch.tensor([.1,.5,.9],device=x.device))
                self.current.update(trajectory_mean=float(self.trajectory.mean()),
                    trajectory_p10=float(q[0]),trajectory_p50=float(q[1]),
                    trajectory_p90=float(q[2]),drift_mean=float(self.drift.mean()),
                    stable_run_mean=float(self.stable_run.mean()))
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
        if self.state.method in ALL_METHODS or self.state.diagnostics or self.state.collect_margins:
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
