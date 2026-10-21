"""Single-QK factored refresh: parallel QK/PZ producer, ordered conditional PV.

Unlike discovery plus sparse FA4, the consumer never recomputes QK. A buffered
probability tile links stages. This is an explicit scheduling control; it does
not install native hooks. Every buffer/launch is charged by refresh().
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _produce(Q,K,Z,P,MU,MX,DEN,N:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,JT:tl.constexpr,SCALE:tl.constexpr):
    block,h,j=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    row=block*32+tl.arange(0,32);key=j*64+tl.arange(0,64);dd=tl.arange(0,512);rr=tl.arange(0,32)
    kh=h//(H//HK)
    q=tl.load(Q+(h*N+row[:,None])*512+dd[None,:],row[:,None]<N,0.)
    k=tl.load(K+(kh*NK+key[None,:])*512+dd[:,None],key[None,:]<NK,0.)
    score=tl.dot(q,k)*SCALE;score=tl.where((row[:,None]<N)&(key[None,:]<NK),score,-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx>-float('inf'),mx,0.)
    p=tl.exp2((score-safe[:,None])*1.4426950408889634);den=tl.sum(p,1)
    z=tl.load(Z+(kh*NK+key[:,None])*32+rr[None,:],key[:,None]<NK,0.)
    mu=tl.dot(p/tl.maximum(den,1.e-30)[:,None],z,input_precision='tf32x3')
    base=((h*tr.cdiv(N,128)+row//128)*JT+j)*128+row%128
    tl.store(P+base[:,None]*64+tl.arange(0,64)[None,:],p.to(tl.bfloat16),row[:,None]<N)
    tl.store(MU+base[:,None]*32+rr[None,:],mu,row[:,None]<N)
    tl.store(MX+base,mx,row<N);tl.store(DEN+base,den,row<N)

@tr.jit
def _consume(P,MU,MX,DEN,V,NU,HELD,KEEP,OUT,STATE,LM,N:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,JT:tl.constexpr,PT:tl.constexpr,TAU,STICKY:tl.constexpr,TRACE:tl.constexpr,CHANNELS:tl.constexpr):
    unit,ch=tl.program_id(0),tl.program_id(1);h=unit//tr.cdiv(N,128);kh=h//(H//HK)
    row=tl.arange(0,128);rr=tl.arange(0,32);key=tl.arange(0,64);dim=ch*CHANNELS+tl.arange(0,CHANNELS)
    state=tl.full((128,32),0.,tl.float32);acc=tl.full((128,CHANNELS),0.,tl.float32)
    rm=tl.full((128,),-float('inf'),tl.float32);rs=tl.full((128,),0.,tl.float32);om=rm;os=rs
    nu=tl.load(NU+kh)
    for j in range(JT):
        base=(unit*JT+j)*128+row
        mx=tl.load(MX+base);den=tl.load(DEN+base);mu=tl.load(MU+base[:,None]*32+rr[None,:])
        nxt=tl.maximum(rm,mx);safe=tl.where(nxt>-float('inf'),nxt,0.);mass=den*tl.exp2((mx-safe)*1.4426950408889634)
        rs=rs*tl.exp2((rm-safe)*1.4426950408889634)+mass;rm=nxt;eta=mass/tl.maximum(rs,1.e-30)
        delta=mu-state;risk=tl.max(eta*eta*tl.sum(delta*delta,1)/(nu*nu),0)
        held=tl.load(HELD+unit*JT+j)
        take=(j==0)|(j>=PT)|(risk>=TAU*TAU*tl.where(held,tl.exp(-2*STICKY),1.))
        if ch==0:tl.store(KEEP+unit*JT+j,take)
        if take:
            state=(1-eta[:,None])*state+eta[:,None]*mu
            nxt=tl.maximum(om,mx);safe=tl.where(nxt>-float('inf'),nxt,0.)
            alpha=tl.exp2((om-safe)*1.4426950408889634);beta=tl.exp2((mx-safe)*1.4426950408889634)
            p=tl.load(P+base[:,None]*64+key[None,:]).to(tl.float32)
            p=(p*beta[:,None]).to(tl.bfloat16)
            v=tl.load(V+(kh*NK+j*64+key[:,None])*512+dim[None,:],j*64+key[:,None]<NK,0.)
            acc=acc*alpha[:,None]+tl.dot(p,v)
            os=os*alpha+den*beta;om=nxt
    qi=(unit%tr.cdiv(N,128))*128+row
    tl.store(OUT+(h*N+qi[:,None])*512+dim[None,:],acc/tl.maximum(os,1.e-30)[:,None],qi[:,None]<N)
    if TRACE and ch==0:
        tl.store(STATE+(h*N+qi[:,None])*32+rr[None,:],state,qi[:,None]<N)
        tl.store(LM+h*N+qi,rm+tl.log(tl.maximum(rs,1.e-30)),qi<N)

class FactoredKernel:
    production=False
    def __init__(self,channels=128):self.channels=channels
    def __call__(self,q,k,v,z,nu,scale,threshold,*,prefix_tiles,held=None,sticky=0.,trace=False,**kw):
        _,h,n,d=q.shape;hk,nk=k.shape[1:3];qb=tr.cdiv(n,128);jt=tr.cdiv(nk,64)
        if n%128:raise ValueError('factored initial control requires full Q128 tiles')
        p=torch.empty((h*qb,jt,128,64),device=q.device,dtype=torch.bfloat16)
        mu=torch.empty((h*qb,jt,128,32),device=q.device);mx=torch.empty((h*qb,jt,128),device=q.device);den=torch.empty_like(mx)
        keep=torch.empty((1,h,qb,jt),device=q.device,dtype=torch.bool);held=torch.zeros_like(keep) if held is None else held
        out=torch.empty_like(q);state=torch.empty((h,n,32) if trace else (1,),device=q.device);lm=torch.empty((h,n) if trace else (1,),device=q.device)
        _produce[(tr.cdiv(n,32),h,jt)](q,k,z,p,mu,mx,den,n,nk,h,hk,jt,float(scale),num_warps=4)
        _consume[(h*qb,512//self.channels)](p,mu,mx,den,v,nu,held,keep,out,state,lm,n,nk,h,hk,jt,prefix_tiles,threshold,sticky,trace,self.channels,num_warps=8,num_stages=1)
        return out.transpose(1,2),keep,state if trace else None,lm if trace else None,None
