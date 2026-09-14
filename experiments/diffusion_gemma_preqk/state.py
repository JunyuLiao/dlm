"""Compact causal state. No method receives current dense logits/labels."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .config import TILE


def smooth(values, eligible, axis, reduction):
    if reduction=='none': return values
    shifted=[]; masks=[]
    for offset in (-1,0,1):
        value=torch.roll(values,offset,axis); mask=torch.roll(eligible,offset,axis).clone()
        if offset:
            index=[slice(None)]*values.ndim;index[axis]=0 if offset>0 else -1;mask[tuple(index)]=False
        shifted.append(value);masks.append(mask)
    values=torch.stack(shifted);masks=torch.stack(masks)
    if reduction=='max':return values.masked_fill(~masks,-torch.inf).amax(0).masked_fill(~eligible,0)
    return (values*masks).sum(0)/masks.sum(0).clamp_min(1)


def protected_set(eligible, prefix_length, kv_length, category):
    blocks=torch.arange(eligible.shape[-1],device=eligible.device)
    out=torch.zeros_like(eligible)
    if category=='none': return out
    if category=='beginning':out |= blocks==0
    if category=='prefix_end' and prefix_length:out |= blocks==(prefix_length-1)//TILE
    if category=='canvas_end':out |= blocks==(kv_length-1)//TILE
    if category=='diagonal':
        queries=(prefix_length+torch.arange(eligible.shape[-2],device=eligible.device)*TILE)//TILE
        out |= (blocks[None,:]-queries[:,None]).abs()<=1
    return out & eligible


def fixed_budget(score, eligible, protected, sparsity, *, salt=0, random_protection=False, explore=0, age=None, budget_delta=None):
    """Protection/exploration consumes the same budget; infeasibility is explicit.

    A private seeded generator never consumes generation RNG state.
    """
    n=eligible.sum(-1);budget=(n-torch.floor(n*sparsity).long()).clamp_min(1).minimum(n)
    if budget_delta is not None:budget=(budget+budget_delta).clamp_min(1).minimum(n)
    if random_protection:
        generator=torch.Generator(device=score.device).manual_seed(int(salt))
        hashed=torch.rand(score.shape,device=score.device,generator=generator)
        order=hashed.masked_fill(~eligible,-torch.inf).argsort(dim=-1,descending=True,stable=True)
        ranks=torch.empty_like(order).scatter_(-1,order,torch.arange(score.shape[-1],device=score.device).expand_as(order))
        protected=eligible & (ranks<protected.sum(-1,keepdim=True))
    forced=protected.clone()
    if explore and age is not None:
        order=age.masked_fill(~eligible | protected,-1).argsort(dim=-1,descending=True,stable=True)
        ranks=torch.empty_like(order).scatter_(-1,order,torch.arange(score.shape[-1],device=score.device).expand_as(order))
        available=(budget-protected.sum(-1)).clamp_min(0).clamp_max(explore)
        forced |= eligible & ~protected & (ranks<available[...,None])
    needed=forced.sum(-1);infeasible=(needed-budget).clamp_min(0)
    # Required protection wins when it exceeds capacity; all excess is reported.
    priority=score.masked_fill(~eligible,-torch.inf).masked_fill(forced,torch.inf)
    order=priority.argsort(dim=-1,descending=True,stable=True)
    ranks=torch.empty_like(order).scatter_(-1,order,torch.arange(score.shape[-1],device=score.device).expand_as(order))
    keep=eligible & (ranks<torch.maximum(budget,needed)[...,None])
    return keep,dict(budget=budget,protected=protected.sum(-1),forced=needed,budget_excess=infeasible)


def cover_rows(keep, valid_blocks):
    """Geometry-only rescue: choose each uncovered row's first eligible tile."""
    row_tiles=valid_blocks.any(-1)
    missing=row_tiles.any(-1) & ~(row_tiles & keep[...,None,:]).any(-1)
    first=row_tiles.long().argmax(-1)
    rescue=F.one_hot(first,keep.shape[-1]).bool() & missing[...,None]
    return keep | rescue.any(-2),missing.sum(-1)


@dataclass
class History:
    mean: torch.Tensor
    peak: torch.Tensor
    ema: torch.Tensor
    frequency: torch.Tensor
    observations: torch.Tensor
    age: torch.Tensor
    last_mask: torch.Tensor
    last_step: int
    signature: tuple

    @classmethod
    def empty(cls,eligible,signature):
        z=torch.zeros_like(eligible,dtype=torch.float32)
        return cls(z.clone(),z.clone(),z.clone(),z.clone(),torch.zeros_like(eligible,dtype=torch.int32),
            torch.zeros_like(eligible,dtype=torch.int32),torch.zeros_like(eligible),-1,signature)

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.mean,self.peak,self.ema,self.frequency,self.observations,self.age,self.last_mask))

    def predict(self,config,eligible,*,value_summary=None,proxy=None):
        fields=dict(last_mass=self.mean,last_mask=self.last_mask.float(),peak=self.peak,
            frequency=self.frequency/self.observations.clamp_min(1),ema_mass=self.ema)
        if config.estimator=='proxy':
            if proxy is None:raise ValueError('cheap current proxy required')
            score=proxy.clone()
        else:score=fields[config.estimator].clone()
        # Unobserved means unknown, not zero: use an observed-row mean prior.
        observed=eligible if config.estimator=='proxy' else ((self.observations>0) & eligible)
        prior=(score*observed).sum(-1,keepdim=True)/observed.sum(-1,keepdim=True).clamp_min(1)
        score=torch.where(observed,score,prior.clamp_min(1e-12))
        score=smooth(score,eligible,-1,config.kv_neighborhood)
        score=smooth(score,eligible,-2,config.query_neighborhood)
        if config.value_weight!='none':
            if value_summary is None:raise ValueError('value summaries must be supplied by producer cache')
            score=score*value_summary[...,None,:]
        return score

    def update(self,mean,peak,observed,keep,step,*,eligible,rho=.8,conditional=False,correct=True):
        if step<=self.last_step:raise ValueError('nonconsecutive/replayed history update')
        if self.last_step>=0 and step!=self.last_step+1:raise ValueError('history gap must reset, not masquerade as adjacent')
        if conditional and correct:
            total=self.mean.sum(-1,keepdim=True)
            coverage=(self.mean*observed).sum(-1,keepdim=True)/total.clamp_min(1e-12)
            coverage=torch.where(total>0,coverage,torch.ones_like(coverage)).clamp(1e-3,1)
            mean=mean*coverage;peak=peak*coverage
        first=self.observations==0
        self.mean=torch.where(observed,mean,self.mean)
        self.peak=torch.where(observed,peak,self.peak)
        self.ema=torch.where(observed,torch.where(first,mean,rho*self.ema+(1-rho)*mean),self.ema)
        # Binary importance, NOT frequency of being selected by this router.
        # Structurally impossible blocks must not lower the uniform-mass cutoff.
        access=mean>=eligible.sum(-1,keepdim=True).clamp_min(1).reciprocal()
        self.frequency+=access.float()*observed
        self.observations+=observed.to(torch.int32)
        self.age=torch.where(observed,0,self.age+1)
        self.last_mask=keep.clone()
        self.last_step=step


class RequestState:
    def __init__(self,request_id):
        self.request_id=request_id;self.layers={};self.resets=0

    def get(self,layer,eligible,*,step,prefix_length,query_length,cache_identity):
        signature=(self.request_id,prefix_length,query_length,cache_identity,tuple(eligible.shape))
        old=self.layers.get(layer)
        if old is None or old.signature!=signature or step!=old.last_step+1 or step==0:
            old=History.empty(eligible,signature);self.layers[layer]=old;self.resets+=1
        return old

    @property
    def bytes(self):return sum(s.bytes for s in self.layers.values())
