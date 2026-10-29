"""Experimental distributed single-QK online traversal using Triton clusters.

One logical program owns a physical Q128 selection unit and all output
channels. The compiler distributes its tensors across a static CTA cluster;
the threshold vote remains over the complete Q128 unit. No dense PV discovery.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _online(Q,K,V,Z,NU,HELD,O,KEEP,N:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,
            SCALE:tl.constexpr,TAU:tl.constexpr,STICKY:tl.constexpr,PT:tl.constexpr):
    qb,head=tl.program_id(0),tl.program_id(1);kh=head//(H//HK)
    row=qb*128+tl.arange(0,128);dd=tl.arange(0,512);jj=tl.arange(0,64);rr=tl.arange(0,32)
    q=tl.load(Q+(head*N+row[:,None])*512+dd[None,:],row[:,None]<N,0.)
    state=tl.full((128,32),0.,tl.float32);acc=tl.full((128,512),0.,tl.float32)
    rm=tl.full((128,),-float('inf'),tl.float32);rs=tl.full((128,),0.,tl.float32)
    om=tl.full((128,),-float('inf'),tl.float32);os=tl.full((128,),0.,tl.float32)
    reference=tl.load(NU+kh);tiles=tl.cdiv(NK,64)
    for tile in range(tiles):
        key=tile*64+jj
        k=tl.load(K+(kh*NK+key[None,:])*512+dd[:,None],key[None,:]<NK,0.)
        score=tl.dot(q,k)*SCALE
        score=tl.where((row[:,None]<N)&(key[None,:]<NK),score,-float('inf'))
        mx=tl.max(score,1);safe=tl.where(mx>-float('inf'),mx,0.)
        p=tl.exp(score-safe[:,None]);den=tl.sum(p,1);prob=p/tl.maximum(den[:,None],1.e-30)
        next_m=tl.maximum(rm,mx);next_safe=tl.where(next_m>-float('inf'),next_m,0.)
        mass=den*tl.exp(mx-next_safe);rs=rs*tl.exp(rm-next_safe)+mass;rm=next_m
        eta=mass/tl.maximum(rs,1.e-30)
        z=tl.load(Z+(kh*NK+key[:,None])*32+rr[None,:],key[:,None]<NK,0.)
        mu=tl.dot(prob,z,input_precision='tf32x3')
        delta=mu-state;risk=tl.sqrt(tl.sum(delta*delta,1))*eta/tl.maximum(reference,1.e-12)
        risk=tl.max(tl.where(row<N,risk,0.),0)
        old_keep=tl.load(HELD+(head*tl.cdiv(N,128)+qb)*tiles+tile)
        threshold=TAU*tl.exp(-STICKY*old_keep)
        take=(tile==0)|(tile>=PT)|~(risk<threshold)
        tl.store(KEEP+(head*tl.cdiv(N,128)+qb)*tiles+tile,take)
        if take:
            state=(1.-eta[:,None])*state+eta[:,None]*mu
            next_o=tl.maximum(om,mx);safe_o=tl.where(next_o>-float('inf'),next_o,0.)
            alpha=tl.exp(om-safe_o);beta=tl.exp(mx-safe_o)
            os=os*alpha+den*beta;om=next_o
            v=tl.load(V+(kh*NK+key[:,None])*512+dd[None,:],key[:,None]<NK,0.)
            acc=acc*alpha[:,None]+tl.dot((p*beta[:,None]).to(tl.bfloat16),v)
    tl.store(O+(head*N+row[:,None])*512+dd[None,:],acc/tl.maximum(os[:,None],1.e-30),row[:,None]<N)

def online(q,k,v,z,nu,scale,threshold,prefix_tiles,*,held=None,sticky=0.,ctas=4,warps=4,stages=1):
    b,h,n,d=q.shape;hk,nk=k.shape[1:3]
    if b!=1 or d!=512 or any(not x.is_contiguous() for x in [q,k,v,z]):raise ValueError('contiguous native inputs required')
    keep=torch.empty((1,h,tr.cdiv(n,128),tr.cdiv(nk,64)),device=q.device,dtype=torch.bool)
    if held is None:held=torch.zeros_like(keep)
    out=torch.empty_like(q)
    _online[(tr.cdiv(n,128),h)](q,k,v,z,nu,held,out,keep,n,nk,h,hk,float(scale),float(threshold),float(sticky),prefix_tiles,num_warps=warps,num_ctas=ctas,num_stages=stages,enable_fp_fusion=False)
    return out.transpose(1,2),keep
