"""Sparse-output refresh variants; inherited selectors and held FA4 are untouched.

QK/PZ discovery is required on every key tile. Only selected tiles reach the
full-dimensional FA4 output. The fused selector keeps its normalized projected
state in registers and never materializes per-tile rank32 summaries.
"""
import math
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def _discover(Q, K, Z, NU, HELD, KEEP, STATE, LMOUT, RISK,
              N: tl.constexpr, NK, H: tl.constexpr, HK: tl.constexpr,
              D: tl.constexpr, JT, PT, QS0: tl.constexpr, QS1: tl.constexpr,
              KS0, KS1: tl.constexpr, SCALE: tl.constexpr,
              LOG_TAU, STICKY: tl.constexpr,
              SINK: tl.constexpr, RECENT: tl.constexpr, TRACE: tl.constexpr,
              RATIO: tl.constexpr, SPLIT_D: tl.constexpr):
    unit = tl.program_id(0)
    qb = tr.cdiv(N, 128)
    h = unit // qb
    kh = h // (H // HK)
    row = tl.arange(0, 128)
    qi = (unit % qb) * 128 + row
    dd = tl.arange(0, D)
    rr = tl.arange(0, 32)
    kk = tl.arange(0, 64)
    if not SPLIT_D:
        q = tl.load(Q + h*QS0 + qi[:, None]*QS1 + dd[None, :], qi[:, None] < N, 0.)
    previous = tl.full((128,), -float('inf'), tl.float32)
    running_max = tl.full((128,), -float('inf'), tl.float32)
    running_sum = tl.full((128,), 0., tl.float32)
    output = tl.full((128, 32), 0., tl.float32)
    support = tl.full((128,), False, tl.int1)
    bad_unit = tl.full((), False, tl.int1)
    nu = tl.maximum(tl.load(NU+kh), 1.e-12)
    for j in tl.range(JT,disable_licm=SPLIT_D):
        key = j*64+kk
        if SPLIT_D:
            scores = tl.full((128,64),0.,tl.float32)
            for part in range(D//64):
                dim = part*64+tl.arange(0,64)
                q_part = tl.load(Q+h*QS0+qi[:,None]*QS1+dim[None,:],qi[:,None]<N,0.)
                k_part = tl.load(K+kh*KS0+key[None,:]*KS1+dim[:,None],key[None,:]<NK,0.)
                scores = tl.dot(q_part,k_part,scores)
            scores *= SCALE
        else:
            k = tl.load(K+kh*KS0+key[None, :]*KS1+dd[:, None], key[None, :] < NK, 0.)
            scores = tl.dot(q, k)*SCALE
        legal = (qi[:, None] < N) & (key[None, :] < NK)
        bad = tl.sum((legal & ((scores != scores) | (scores == float('inf')))).to(tl.int32), 1) > 0
        scores = tl.where(legal, scores, -float('inf'))
        mx = tl.max(scores, 1)
        active = mx > -float('inf')
        safe = tl.where(active, mx, 0.)
        if RATIO:
            p = tl.exp2((scores-safe[:, None])*1.4426950408889634)
        else:
            p = lib.exp(scores-safe[:, None])
        den = tl.sum(p, 1)
        p = p/tl.maximum(den, 1.e-30)[:, None]
        z = tl.load(Z+(kh*NK+key[:, None])*32+rr[None, :], key[:, None] < NK, 0.)
        mu = tl.dot(p, z, input_precision='tf32x3')
        bad |= tl.sum(((mu != mu) | (tl.abs(mu) == float('inf'))).to(tl.int32), 1) > 0
        lm = tl.where(active & ~bad, safe+lib.log(tl.maximum(den, 1.e-30)), -float('inf'))
        mu = tl.where(bad[:, None], 0., mu)
        if RATIO:
            new_max = tl.maximum(running_max, mx)
            new_max = tl.where(new_max > -float('inf'), new_max, 0.)
            old_scale = tl.exp2((running_max-new_max)*1.4426950408889634)
            new_mass = den*tl.exp2((mx-new_max)*1.4426950408889634)
            running_sum = running_sum*old_scale+new_mass
            eta = new_mass/tl.maximum(running_sum, 1.e-30)
            running_max = new_max
            combined = tl.where(running_sum > 0, running_max+lib.log(tl.maximum(running_sum, 1.e-30)), -float('inf'))
        else:
            mx = tl.maximum(previous, lm)
            safe = tl.where(mx > -float('inf'), mx, 0.)
            combined = safe+lib.log(lib.exp(previous-safe)+lib.exp(lm-safe))
            eta = tl.where(active, lib.exp(lm-tl.where(combined > -float('inf'), combined, 0.)), 0.)
        delta = mu-output
        if RATIO:
            squared = tl.max(tl.where(active, eta*eta*tl.sum(delta*delta, 1), 0.), 0)
            if TRACE:
                risk = 0.5*lib.log(squared)-lib.log(nu)
        else:
            rho = eta*tl.sqrt(tl.sum(delta*delta, 1))/nu
            risk = tl.max(tl.where(active, lib.log(rho), -float('inf')), 0)
        first = tl.sum((active & ~support).to(tl.int32), 0) > 0
        bad_unit |= tl.sum(bad.to(tl.int32), 0) > 0
        held = tl.load(HELD+unit*JT+j)
        mandatory = (j >= PT) | (j < SINK) | (j >= tl.maximum(0, PT-RECENT))
        if RATIO:
            limit = nu*nu*tl.exp(2*LOG_TAU)*tl.where(held, tl.exp(-2*STICKY), 1.)
            accept = squared >= limit
        else:
            accept = risk+STICKY*held >= LOG_TAU
        take = (tl.sum(active.to(tl.int32), 0) > 0) & (first | mandatory | bad_unit | accept)
        tl.store(KEEP+unit*JT+j, take)
        output = tl.where(take, (1-eta[:, None])*output+eta[:, None]*mu, output)
        previous = combined
        support |= active & take
        if TRACE:
            tl.store(RISK+unit*JT+j, risk)
    # A malformed later tile makes the whole unit dense, including prior tiles.
    if bad_unit:
        for j in range(JT):
            tl.store(KEEP+unit*JT+j, True)
    if TRACE:
        tl.store(STATE+unit*128*32+row[:, None]*32+rr[None, :], output)
        tl.store(LMOUT+unit*128+row, previous)


def discover(q, k, sketch, nu, scale, prefix_tiles, threshold, *, held=None,
             sticky=0., sink=0, recent=0, warps=8, trace=False, ratio=False,split_d=False):
    if (q.ndim != 4 or q.shape[0] != 1 or q.dtype != torch.bfloat16 or
        k.dtype != q.dtype or q.shape[-1] != k.shape[-1] or q.stride(-1) != 1 or k.stride(-1) != 1):
        raise ValueError('native BF16 batch-one Q/K required')
    _, h, n, d = q.shape
    hk, nk = k.shape[1:3]
    if (not q.is_cuda or h % hk or d not in (256, 512) or
        sketch.shape != (1, hk, nk, 32) or sketch.dtype != torch.float32 or
        not sketch.is_contiguous() or nu.shape != (1, hk) or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('invalid current Gaussian32 geometry or threshold')
    qb, jt = tr.cdiv(n, 128), tr.cdiv(nk, 64)
    if not 0 <= prefix_tiles <= jt or min(sink, recent) < 0:
        raise ValueError('invalid protected prefix')
    keep = torch.empty((h*qb, jt), device=q.device, dtype=torch.bool)
    if held is None:
        held = torch.zeros_like(keep)
    if held.numel() != keep.numel() or held.dtype != torch.bool or held.device != q.device:
        raise ValueError('invalid current held map')
    held = held.reshape_as(keep).contiguous()
    state = torch.empty((h*qb, 128, 32), device=q.device) if trace else keep
    lm = torch.empty((h*qb, 128), device=q.device) if trace else keep
    risk = torch.empty((h*qb, jt), device=q.device) if trace else keep
    _discover[(h*qb,)](q, k, sketch, nu, held, keep, state, lm, risk,
        n, nk, h, hk, d, jt, prefix_tiles, q.stride(1), q.stride(2),
        k.stride(1), k.stride(2), float(scale), math.log(threshold) if threshold else -math.inf,
        float(sticky), int(sink), int(recent), trace, ratio, split_d,num_warps=warps, num_stages=1)
    return keep, state, lm, risk


def parallel_discover(q, k, sketch, nu, scale, prefix_tiles, threshold, *,
                      held=None, sticky=0., sink=0, recent=0,fast_scan=False):
    from .v31_value_kernels import statistics_cuda, online_cuda
    from .v31_value_selectors import mandatory_map
    stats = statistics_cuda(q, k, sketch, nu.reshape(-1), scale, prefix_tiles)
    mandatory = mandatory_map(stats, sink, recent)
    held = torch.zeros_like(mandatory) if held is None else held.reshape_as(mandatory)
    if fast_scan:
        keep=torch.empty_like(mandatory)
        _summary_scan[(keep.shape[0],)](stats.log_mass,stats.mean,stats.nu,held,mandatory,stats.invalid,keep,
            keep.shape[1],float(threshold),float(sticky),num_warps=4)
        return keep,None,None,None
    result = online_cuda(stats, threshold, True, mandatory, held, sticky)
    return result


@tr.jit
def _summary_scan(LM,MU,NU,HELD,PROT,BAD,KEEP,JT,TAU,STICKY:tl.constexpr):
    unit=tl.program_id(0)
    row,rr=tl.arange(0,128),tl.arange(0,32)
    state=tl.full((128,32),0.,tl.float32)
    maximum=tl.full((128,),-float('inf'),tl.float32)
    denominator=tl.full((128,),0.,tl.float32)
    nu=tl.maximum(tl.load(NU+unit*128+row),1.e-12)
    bad_unit=tl.full((),False,tl.int1)
    for j in range(JT):
        ix=(unit*JT+j)*128+row
        lm=tl.load(LM+ix)
        active=lm > -float('inf')
        mx=tl.maximum(maximum,lm)
        safe=tl.where(mx > -float('inf'),mx,0.)
        old=tl.exp2((maximum-safe)*1.4426950408889634)
        mass=tl.exp2((lm-safe)*1.4426950408889634)
        first=tl.sum((active & (denominator==0)).to(tl.int32),0)>0
        denominator=denominator*old+mass
        eta=mass/tl.maximum(denominator,1.e-30)
        maximum=mx
        mu=tl.load(MU+ix[:,None]*32+rr[None,:])
        delta=mu-state
        risk2=tl.max(eta*eta*tl.sum(delta*delta,1)/(nu*nu),0)
        held=tl.load(HELD+unit*JT+j)
        protected=tl.load(PROT+unit*JT+j)
        bad_unit|=tl.sum(tl.load(BAD+ix).to(tl.int32),0)>0
        take=(tl.sum(active.to(tl.int32),0)>0)&(first|protected|bad_unit|
             (risk2>=TAU*TAU*tl.where(held,tl.exp(-2*STICKY),1.)))
        tl.store(KEEP+unit*JT+j,take)
        state=tl.where(take,(1-eta[:,None])*state+eta[:,None]*mu,state)
    if bad_unit:
        for j in range(JT):tl.store(KEEP+unit*JT+j,True)


@tr.jit
def _attention(Q, K, V, Z, NU, HELD, KEEP, OUT,
               N: tl.constexpr, NK, H: tl.constexpr, HK: tl.constexpr,
               D: tl.constexpr, JT, PT, QS0: tl.constexpr, QS1: tl.constexpr,
               KS0, KS1: tl.constexpr, VS0, VS1: tl.constexpr,
               SCALE: tl.constexpr, TAU, STICKY: tl.constexpr):
    unit = tl.program_id(0)//2
    half = tl.program_id(0)%2
    qb = tr.cdiv(N, 128)
    h, kh = unit//qb, (unit//qb)//(H//HK)
    row, col = tl.arange(0, 128), tl.arange(0, D)
    qi = (unit % qb)*128+row
    kk, rr = tl.arange(0, 64), tl.arange(0, 32)
    q = tl.load(Q+h*QS0+qi[:, None]*QS1+col[None, :],qi[:, None]<N,0.)
    state = tl.full((128, 32),0.,tl.float32)
    routing_max = tl.full((128,),-float('inf'),tl.float32)
    routing_sum = tl.full((128,),0.,tl.float32)
    output_max = tl.full((128,),-float('inf'),tl.float32)
    output_sum = tl.full((128,),0.,tl.float32)
    output_rows = tl.arange(0,64)+half*64
    output_qi = (unit % qb)*128+output_rows
    acc = tl.full((64,D),0.,tl.float32)
    nu = tl.maximum(tl.load(NU+kh),1.e-12)
    for j in range(JT):
        keys = j*64+kk
        k = tl.load(K+kh*KS0+keys[None, :]*KS1+col[:, None],keys[None, :]<NK,0.)
        score = tl.dot(q,k)*SCALE
        score = tl.where((qi[:, None]<N)&(keys[None, :]<NK),score,-float('inf'))
        mx = tl.max(score,1)
        safe = tl.where(mx > -float('inf'),mx,0.)
        p = tl.exp2((score-safe[:, None])*1.4426950408889634)
        den = tl.sum(p,1)
        z = tl.load(Z+(kh*NK+keys[:, None])*32+rr[None,:],keys[:,None]<NK,0.)
        mean = tl.dot(p/tl.maximum(den,1.e-30)[:,None],z,input_precision='tf32x3')
        next_max = tl.maximum(routing_max,mx)
        safe_next = tl.where(next_max > -float('inf'),next_max,0.)
        mass = den*tl.exp2((mx-safe_next)*1.4426950408889634)
        routing_sum = routing_sum*tl.exp2((routing_max-safe_next)*1.4426950408889634)+mass
        eta = mass/tl.maximum(routing_sum,1.e-30)
        routing_max = next_max
        delta = mean-state
        risk2 = tl.max(eta*eta*tl.sum(delta*delta,1),0)
        held = tl.load(HELD+unit*JT+j)
        take = (j==0)|(j>=PT)|(risk2>=TAU*TAU*nu*nu*tl.where(held,tl.exp(-2*STICKY),1.))
        if half == 0:
            tl.store(KEEP+unit*JT+j,take)
        if take:
            state = (1-eta[:,None])*state+eta[:,None]*mean
            next_output_max = tl.maximum(output_max,mx)
            safe_output_max = tl.where(next_output_max > -float('inf'),next_output_max,0.)
            alpha = tl.exp2((output_max-safe_output_max)*1.4426950408889634)
            beta = tl.exp2((mx-safe_output_max)*1.4426950408889634)
            weights = tl.gather(p*beta[:,None],tl.broadcast_to(output_rows[:,None],(64,64)),0).to(tl.bfloat16)
            v = tl.load(V+kh*VS0+keys[:,None]*VS1+col[None,:],keys[:,None]<NK,0.)
            alpha_half = tl.gather(alpha,output_rows,0)
            acc = acc*alpha_half[:,None]+tl.dot(weights,v)
            output_sum = output_sum*alpha+den*beta
            output_max = next_output_max
    den_half = tl.gather(output_sum,output_rows,0)
    result = acc/tl.maximum(den_half,1.e-30)[:,None]
    tl.store(OUT+(h*N+output_qi[:,None])*D+col[None,:],result,output_qi[:,None]<N)


def attention(q,k,v,sketch,nu,scale,prefix_tiles,threshold,*,held=None,sticky=0.,ctas=1,warps=8):
    """Duplicate Q128 routing across Q64 full-PV consumers, without cross-CTA waits.

    Unlike the FA4-output variants this has its own BF16 P rounding and requires
    independent operator and native qualification. Finite QKV is required.
    """
    _,h,n,d=q.shape
    hk,nk=k.shape[1:3]
    jt=tr.cdiv(nk,64)
    keep=torch.empty((h*tr.cdiv(n,128),jt),device=q.device,dtype=torch.bool)
    held=torch.zeros_like(keep) if held is None else held.reshape_as(keep).contiguous()
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    if ctas != 1: raise ValueError('multi-CTA Triton planner unsupported for mixed PZ/PV shapes')
    _attention[(2*h*tr.cdiv(n,128),)](q,k,v,sketch,nu,held,keep,out,
        n,nk,h,hk,d,jt,prefix_tiles,q.stride(1),q.stride(2),k.stride(1),k.stride(2),
        v.stride(1),v.stride(2),float(scale),float(threshold),float(sticky),
        num_ctas=ctas,num_warps=warps,num_stages=1)
    return out.transpose(1,2),keep.reshape(1,h,tr.cdiv(n,128),jt)


def refresh(q, k, v, matrix, scale, prefix_tiles, threshold, *, backend='fused',
            held=None, sticky=0., sink=0, recent=0, warps=8, ratio=False,ctas=1,split_d=False,fast_scan=False):
    from .v31_value_projection import refresh as project
    from . import v27_fa4
    _, hk, nk, _ = v.shape
    sketch = torch.empty((1, hk, nk, 32), device=v.device, dtype=torch.float32)
    norms = torch.empty((1, hk, nk), device=v.device, dtype=torch.float32)
    valid = torch.ones_like(norms, dtype=torch.bool)
    nu = project(v, matrix, sketch, norms, valid, 0)
    if backend == 'attention':
        if sink or recent: raise ValueError('cluster prototype supports mandatory canvas only')
        out,keep=attention(q,k,v,sketch,nu,scale,prefix_tiles,threshold,held=held,sticky=sticky,ctas=ctas,warps=warps)
        return out,keep,v27_fa4.block_sparse_tensors(keep)
    if backend not in ('fused', 'parallel'):
        raise ValueError('unknown sparse refresh backend')
    fn = discover if backend == 'fused' else parallel_discover
    extra = dict(warps=warps,ratio=ratio,split_d=split_d) if backend == 'fused' else dict(fast_scan=fast_scan)
    keep = fn(q, k, sketch, nu, scale, prefix_tiles, threshold,
              held=held, sticky=sticky, sink=sink, recent=recent, **extra)[0]
    keep = keep.reshape(1, q.shape[1], tr.cdiv(q.shape[2], 128), -1)
    lists = v27_fa4.block_sparse_tensors(keep)
    out = v27_fa4.sparse_lists(q, k, v, lists, scale)
    return out, keep, lists
