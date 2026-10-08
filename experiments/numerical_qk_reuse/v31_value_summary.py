"""Bounded-memory full-support sketch reductions for deletion selectors."""
import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['JT'])
def _output(ALPHA, MU, OUT, JT):
    unit, chunk = tl.program_id(0), tl.program_id(1)
    row = chunk*4+tl.arange(0,4)
    rr = tl.arange(0,32)
    jj = tl.arange(0,32)
    output = tl.full((4,32),0.,tl.float32)
    for start in range(tl.cdiv(JT,32)):
        j = start*32+jj
        ix = (unit*JT+j[:,None])*128+row[None,:]
        alpha = tl.load(ALPHA+ix,j[:,None]<JT,0.)
        mean = tl.load(MU+ix[:,:,None]*32+rr[None,None,:],j[:,None,None]<JT,0.)
        output += tl.sum(alpha[:,:,None]*mean,0)
    tl.store(OUT+unit*128*32+row[:,None]*32+rr[None,:],output)


@tr.jit(do_not_specialize=['JT'])
def _residual(ALPHA, MU, OUT, G, JT):
    unit,j,chunk = tl.program_id(0),tl.program_id(1),tl.program_id(2)
    row = chunk*32+tl.arange(0,32)
    rr = tl.arange(0,32)
    ix = (unit*JT+j)*128+row
    alpha = tl.load(ALPHA+ix)
    mean = tl.load(MU+ix[:,None]*32+rr[None,:])
    output = tl.load(OUT+unit*128*32+row[:,None]*32+rr[None,:])
    tl.store(G+ix[:,None]*32+rr[None,:],alpha[:,None]*(mean-output))


def full_support_cuda(stats):
    z=stats.log_mass
    normalizer=torch.logsumexp(z,1,keepdim=True)
    alpha=torch.where(torch.isfinite(z),torch.exp(z-torch.where(torch.isfinite(normalizer),normalizer,0.)),0.)
    u,jt,_=z.shape
    output=torch.empty((u,128,32),device=z.device)
    workspace = getattr(stats,'workspace',None)
    g=torch.empty_like(stats.mean) if workspace is None else workspace.take('g',tuple(stats.mean.shape),torch.float32,z.device)
    _output[(u,32)](alpha,stats.mean,output,jt,num_warps=4)
    _residual[(u,jt,4)](alpha,stats.mean,output,g,jt,num_warps=4)
    return alpha,output,g
