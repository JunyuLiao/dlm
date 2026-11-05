"""Aggressive preceding-seen-max BLASST port, with inherited four-call reuse.

No value projection or V2 threshold enters this selector. Current full-D128
attention output uses the pinned FA4 consumer, and discovery is counted.
"""
import math

import torch
import triton as tr
import triton.language as tl

ARM='blasst_aggressive_inverse_length_reuse_r4'


@tr.jit
def _maxima(Q,K,M,NQ:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,
            QT:tl.constexpr,JT:tl.constexpr,SQH:tl.constexpr,SQQ:tl.constexpr,
            SKH:tl.constexpr,SKK:tl.constexpr,SCALE:tl.constexpr):
    h=tl.program_id(0);qb=tl.program_id(1);j=tl.program_id(2)
    qi=qb*128+tl.arange(0,128);ki=j*64+tl.arange(0,64);d=tl.arange(0,128)
    q=tl.load(Q+h*SQH+qi[:,None]*SQQ+d[None],qi[:,None]<NQ,0.)
    k=tl.load(K+(h//(H//HK))*SKH+ki[:,None]*SKK+d[None],ki[:,None]<NK,0.)
    score=tl.dot(q,tl.trans(k))*SCALE
    score=tl.where((qi[:,None]<NQ)&(ki[None]<NK),score,-float('inf'))
    maximum=tl.max(score,1)
    tl.store(M+((h*QT+qb)*JT+j)*128+tl.arange(0,128),maximum)


def block_maxima(q,k,scale):
    assert q.shape[0]==k.shape[0]==1 and q.shape[-1]==k.shape[-1]==128
    _,h,nq,_=q.shape;nk=k.shape[2];qt=tr.cdiv(nq,128);jt=tr.cdiv(nk,64)
    result=torch.empty((h*qt,jt,128),device=q.device,dtype=torch.float32)
    _maxima[(h,qt,jt)](q,k,result,nq,nk,h,k.shape[1],qt,jt,q.stride(1),q.stride(2),
        k.stride(1),k.stride(2),scale,num_warps=4)
    return result


def select_maxima(maxima,log_lambda,mandatory):
    """Exact vectorized online preceding-seen rule; skipped tiles update seen."""
    if not math.isfinite(log_lambda) and log_lambda!=-math.inf:raise ValueError('finite lambda or dense -inf required')
    active=torch.isfinite(maxima)
    seen=torch.cummax(maxima,dim=1).values
    previous=torch.cat((torch.full_like(seen[:,:1],-math.inf),seen[:,:-1]),dim=1)
    risk=torch.where(active,torch.where(torch.isfinite(previous),maxima-previous,math.inf),-math.inf)
    eligible=active.any(-1)
    keep=eligible&(mandatory|(risk.amax(-1)>=log_lambda))
    bad=(torch.isnan(maxima)|torch.isposinf(maxima)).any((1,2))
    keep[bad]=True
    return keep


def reference(maxima,log_lambda,mandatory):
    """Independent sequential CPU oracle, strict skipping and all-seen updates."""
    kept=torch.zeros_like(mandatory);seen=torch.full_like(maxima[:,0],-math.inf)
    for j in range(maxima.shape[1]):
        b=maxima[:,j];active=torch.isfinite(b)
        risk=torch.where(active,torch.where(torch.isfinite(seen),b-seen,math.inf),-math.inf)
        kept[:,j]=active.any(-1)&(mandatory[:,j]|(risk.amax(-1)>=log_lambda))
        seen=torch.maximum(seen,b)
    kept[(torch.isnan(maxima)|torch.isposinf(maxima)).any((1,2))]=True
    return kept


def install(harness,base_runtime):
    """Model adapters keep calling their original runtime API and sampler."""
    class BlasstRuntime(base_runtime):
        def __init__(self):
            super().__init__('dense_native');self.arm=ARM;self.log_kappa=0.;self.lambda_min=math.inf;self.lambda_max=0.
            self.map_kept_counts={}

        def reset(self):
            super().reset();self.map_kept_counts.clear();self.lambda_min=math.inf;self.lambda_max=0.

        def block(self,start,end):
            super().block(start,end);self.map_kept_counts.clear()

        def invalidate(self):
            super().invalidate();self.map_kept_counts.clear()

        def attention(self,q,k,v,layer,scale,*,offset=0,block_size=0):
            if self.phase!='denoise':return super().attention(q,k,v,layer,scale,offset=offset,block_size=block_size)
            nq,nk,heads=q.shape[2],k.shape[2],q.shape[1]
            if block_size and nq>block_size:raise ValueError('only active denoising block may be sparse')
            qt,jt=tr.cdiv(nq,128),tr.cdiv(nk,64);self.counts['eligible_tiles']+=heads*qt*jt
            key=(layer,nq,nk);log_lambda=self.log_kappa-math.log(nk)
            lam=math.exp(log_lambda);self.lambda_min=min(self.lambda_min,lam);self.lambda_max=max(self.lambda_max,lam)
            refresh=key not in self.maps or (self.call_in_block-1)%4==0
            if refresh:
                self.counts['selection_layer_calls']+=1;self.counts['blasst_selection_layer_calls']+=1
                maximum=block_maxima(q,k,scale)
                mandatory=torch.zeros((heads*qt,jt),device=q.device,dtype=torch.bool)
                mandatory[:,self.block_start//64:tr.cdiv(self.block_end,64)]=True
                keep=select_maxima(maximum,log_lambda,mandatory)
                torch._assert_async((keep|~mandatory).all()&keep[:,0].all(),'BLASST first/current support missing')
                keep=keep.reshape(1,heads,qt,jt)
                self.maps[key]=(keep,self.consumer.block_sparse_tensors(keep))
                self.map_kept_counts[key]=int(keep.sum().item())
                self.counts['discovery_qk_pairs']+=heads*nq*nk
                self.counts['blasst_maximum_elements']+=heads*qt*jt*128
            else:self.counts['held_layer_calls']+=1
            self.counts['kept_tiles']+=self.map_kept_counts[key]
            return self.consumer.sparse_lists(q,k,v,self.maps[key][1],scale)
    runtime=BlasstRuntime();runtime.consumer=__import__(base_runtime.__module__,fromlist=['v27_fa4']).v27_fa4
    heartbeat=getattr(harness,'heartbeat',getattr(harness.runtime,'heartbeat',None))
    if heartbeat is not None:runtime.heartbeat=heartbeat
    harness.runtime=runtime
    return runtime
