"""Independent, explicit builds of the V2 router/output-WG pipeline."""
import hashlib
import json
from pathlib import Path
import subprocess
import os
import torch
from .v2_online_sm90 import Kernel
from .v2_online_pipeline import OnlineRefresh,pack_lists


def build(directory,*,production=False,unroll=32,queue=2,fp16pz=False,lookahead=False,batch2=False,register_p=False,register_p256=False,tma_single=False,register_tma=False,fp16x2=False,leader_load=False,tma_interleave=False,q128=False,q128_two=False,three=False,packed=False,unscaled=False,packed_early=False,packed_local=False,packed_overlap=False,packed_chunk=False,packed_vote=False,packed256=False,pzbound=False,broadcast=False,dynamic_registers=False,widepz=False,wgbar=False,prefetchv=False,packed_exp2=False,pv128=False,exp2_prefetch=False,packedk=False,q32=False,earlyv=False,fourwg=False,q128_batch2=False,mmasync=False,deferred_scale=False,double_score=False,risk_unscaled=False,stable_operand=False,sfu=False,scale_uniform=False,fp16diag=False,cluster_producer=False,cluster_vector=False,staged_qk=False,fast_math=False,fold=False,fold256=False):
    if unroll not in (1,2,4,8,16,32):raise ValueError('supported QK unrolling required')
    if queue!=2:raise ValueError('two-slot queue required')
    source=Path(__file__).with_name('csrc')/'v2_online_warpspec_sm90.cu'
    if fp16pz:source=source.with_name('v2_online_fp16pz_sm90.cu')
    if lookahead:source=source.with_name('v2_online_warpspec_lookahead_sm90.cu')
    if batch2:source=source.with_name('v2_online_warpspec_batch2_sm90.cu')
    if register_p:source=source.with_name('v2_online_register_p_sm90.cu')
    if register_p256:source=source.with_name('v2_online_register_p256_sm90.cu')
    if tma_single:source=source.with_name('v2_online_tma_single_sm90.cu')
    if register_tma:source=source.with_name('v2_online_register_tma_sm90.cu')
    if fp16x2:source=source.with_name('v2_online_fp16x2_sm90.cu')
    if leader_load:source=source.with_name('v2_online_leader_load_sm90.cu')
    if tma_interleave:source=source.with_name('v2_online_tma_interleave_sm90.cu')
    if q128:source=source.with_name('v2_online_q128_register_sm90.cu')
    if q128_two:source=source.with_name('v2_online_q128_twochannel_sm90.cu')
    if three:source=source.with_name('v2_online_warpspec_three_sm90.cu')
    if packed:source=source.with_name('v2_online_q128_packed_sm90.cu')
    if unscaled:source=source.with_name('v2_online_q128_unscaled_sm90.cu')
    if packed_early:source=source.with_name('v2_online_q128_packed_early_sm90.cu')
    if packed_local:source=source.with_name('v2_online_q128_packed_local_sm90.cu')
    if packed_overlap:source=source.with_name('v2_online_q128_packed_overlap_sm90.cu')
    if packed_chunk:source=source.with_name('v2_online_q128_packed_chunk_sm90.cu')
    if packed_vote:source=source.with_name('v2_online_q128_vote_sm90.cu')
    if packed256:source=source.with_name('v2_online_q128_packed256_sm90.cu')
    if pzbound:source=source.with_name('v2_online_q128_pzbound_sm90.cu')
    if broadcast:source=source.with_name('v2_online_q128_vote_broadcast_sm90.cu')
    if widepz:source=source.with_name('v2_online_q128_widepz_sm90.cu')
    if wgbar:source=source.with_name('v2_online_q128_wgbar_sm90.cu')
    if prefetchv:source=source.with_name('v2_online_q128_prefetchv_sm90.cu')
    if packed_exp2:source=source.with_name('v2_online_q128_exp2_sm90.cu')
    if pv128:source=source.with_name('v2_online_warpspec_pv128_sm90.cu')
    if exp2_prefetch:source=source.with_name('v2_online_q128_exp2_prefetch_sm90.cu')
    if packedk:source=source.with_name('v2_online_q128_packedk_sm90.cu')
    if q32:source=source.with_name('v2_online_q32_cluster_sm90.cu')
    if earlyv:source=source.with_name('v2_online_q128_earlyv_sm90.cu')
    if fourwg:source=source.with_name('v2_online_q128_fourwg_sm90.cu')
    if q128_batch2:source=source.with_name('v2_online_q128_batch2_sm90.cu')
    if mmasync:source=source.with_name('v2_online_q128_mmasync_sm90.cu')
    if deferred_scale:source=source.with_name('v2_online_q128_deferred_scale_sm90.cu')
    if double_score:source=source.with_name('v2_online_q128_double_score_sm90.cu')
    if risk_unscaled:source=source.with_name('v2_online_q128_risk_unscaled_sm90.cu')
    if stable_operand:source=source.with_name('v2_online_q128_stable_operand_sm90.cu')
    if sfu:source=source.with_name('v2_online_q128_sfu_sm90.cu')
    if scale_uniform:source=source.with_name('v2_online_q128_scale_uniform_sm90.cu')
    if fp16diag:source=source.with_name('v2_online_q128_fp16diag_sm90.cu')
    if cluster_producer:source=source.with_name('v2_online_cluster_producer_sm90.cu')
    if cluster_vector:source=source.with_name('v2_online_cluster_vector_sm90.cu')
    if staged_qk:source=source.with_name('v2_online_staged_qk_sm90.cu')
    if fold:source=source.with_name('v2_online_q128_fold_sm90.cu')
    if fold256:source=source.with_name('v2_online_q128_fold256_sm90.cu')
    header=source.with_name('v2_online_sm90.h')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,header,Path(__file__))}
    options=dict(fold256=fold256,fold=fold,fast_math=fast_math,staged_qk=staged_qk,cluster_vector=cluster_vector,cluster_producer=cluster_producer,fp16diag=fp16diag,scale_uniform=scale_uniform,sfu=sfu,stable_operand=stable_operand,risk_unscaled=risk_unscaled,double_score=double_score,deferred_scale=deferred_scale,mmasync=mmasync,q128_batch2=q128_batch2,fourwg=fourwg,earlyv=earlyv,q32=q32,packedk=packedk,exp2_prefetch=exp2_prefetch,pv128=pv128,production=production,warpspec=any((lookahead,batch2,three,pv128,cluster_producer,cluster_vector)) or not any((fp16pz,register_p,register_p256,tma_single,register_tma,fp16x2,leader_load,tma_interleave,q128,q128_two,packed,unscaled,packed_early,packed_local,packed_overlap,packed_chunk,packed_vote,packed256,pzbound,broadcast,widepz,wgbar,prefetchv,packed_exp2,exp2_prefetch,packedk,q32,earlyv,fourwg,q128_batch2,mmasync,deferred_scale,double_score,risk_unscaled,stable_operand,sfu,scale_uniform,fp16diag,staged_qk,fold,fold256)),tma_all=True,async_vote=True,unroll=unroll,queue=queue,
                 fp16pz=fp16pz,lookahead=lookahead,batch2=batch2,register_p=register_p,register_p256=register_p256,tma_single=tma_single,register_tma=register_tma,fp16x2=fp16x2,leader_load=leader_load,tma_interleave=tma_interleave,q128=q128,q128_two=q128_two,three=three,packed=packed or packed_early or packed_local or packed_overlap or packed_chunk or packed_vote or packed256 or pzbound or broadcast or widepz or wgbar or prefetchv or packed_exp2 or exp2_prefetch or packedk or earlyv or fourwg or q128_batch2 or mmasync or deferred_scale or double_score or risk_unscaled or stable_operand or sfu or scale_uniform or fp16diag or cluster_producer or cluster_vector or staged_qk or fold or fold256,packed_exp2=packed_exp2,prefetchv=prefetchv,wgbar=wgbar,widepz=widepz,dynamic_registers=dynamic_registers,broadcast=broadcast,pzbound=pzbound,packed256=packed256,packed_vote=packed_vote,packed_chunk=packed_chunk,packed_overlap=packed_overlap,packed_local=packed_local,packed_early=packed_early,unscaled=unscaled,pz='fp16_diagnostic' if fp16diag else ('fp16' if fp16pz else ('fp16x2' if fp16x2 else ('bf16single_diagnostic' if pzbound else 'bf16x3'))),output='retained_max_scaled_sum',fast_div=True,schedules={0:'conditional_v',1:'early_v',2:'early_v_next_k',3:'conditional_v_next_k'})
    key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16]
    dest=Path(directory)/key;dest.mkdir(parents=True,exist_ok=True);library=dest/'v2_online_sm90.so'
    if not library.exists():
        src=source.read_text()
        needle='#pragma unroll\n        for(int kk=0;kk<512;' if fp16pz or register_p or register_p256 or tma_single or register_tma or fp16x2 or leader_load or tma_interleave or q128 or q128_two or packed or unscaled or packed_early or packed_local or packed_overlap or packed_vote or packed256 or pzbound or broadcast or widepz or wgbar or prefetchv or packed_exp2 or exp2_prefetch or packedk or q32 or earlyv or fourwg or mmasync or deferred_scale or double_score or risk_unscaled or stable_operand or sfu or scale_uniform or fp16diag or fold or fold256 else '#pragma unroll\n            for(int kk=0;kk<512;'
        if cluster_producer or cluster_vector:needle='#pragma unroll\n            for(int kk=0;kk<512;'
        if q128_batch2:needle='#pragma unroll\n            for(int kk=0;kk<128;'
        if packed_chunk:needle='#pragma unroll\n            for(int kk=0;kk<256;'
        if batch2:needle='#pragma unroll\n                for(int kk=0;kk<256;'
        if not staged_qk:
            if src.count(needle)!=1:raise RuntimeError('QK unroll source drift')
            src=src.replace(needle,needle.replace('#pragma unroll',f'#pragma unroll {unroll}'))
        if dynamic_registers:
            anchor='    if(wg==0){'
            if src.count(anchor)!=1:raise RuntimeError('dynamic register role patch drift')
            src=src.replace(anchor,'''    if(wg==0){
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
#endif''',1)
            anchor='    }else{\n        float output[128]={};'
            if src.count(anchor)!=1:raise RuntimeError('dynamic register consumer patch drift')
            src=src.replace(anchor,'''    }else{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
#endif
        float output[128]={};''',1)
        if production:src=src.replace('if(a.trace)','if constexpr(false)').replace('if(a.trace&&','if(false&&')
        (dest/source.name).write_text(src);(dest/header.name).write_bytes(header.read_bytes())
        cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
        cmd=['/usr/local/cuda/bin/nvcc','--prec-div=false','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC',
             '-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/source.name),
             '-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
        if fast_math:cmd.insert(1,'--use_fast_math')
        tmp=dest/'tmp';tmp.mkdir(exist_ok=True)
        env=os.environ.copy();env['TMPDIR']=str(tmp.resolve())
        r=subprocess.run(cmd,text=True,capture_output=True,env=env)
        (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
        if r.returncode:raise RuntimeError(r.stderr[-9000:])
        exported=subprocess.run(['/usr/bin/nm','-D',str(library)],text=True,capture_output=True)
        if exported.returncode or 'v2_online_sm90_params_size' not in exported.stdout:
            raise RuntimeError('compiled library missing required ABI; preserve failed attempt')
    return library

import triton as tr
import triton.language as tl

@tr.jit
def _pack_z(Z,P,N:tl.constexpr,JT:tl.constexpr,WIDE:tl.constexpr,HALF:tl.constexpr):
    tile,head=tl.program_id(0),tl.program_id(1);x=tl.arange(0,2048)
    key=tile*64+x//32;dim=x%32
    z=tl.load(Z+(head*N+key)*32+dim,key<N,0.)
    if HALF:hi=z.to(tl.float16);lo=tl.full((2048,),0.,tl.float16)
    else:hi=z.to(tl.bfloat16);lo=(z-hi.to(tl.float32)).to(tl.bfloat16)
    if WIDE:
        key=x//32;dim=x%32
        base=dim+(key%8)*32+(key//8)*512
        hi_at=base^(((base>>6)&3)<<3)
        low=base+256;lo_at=low^(((low>>6)&3)<<3)
        tl.store(P+(head*JT+tile)*4096+hi_at,hi)
        tl.store(P+(head*JT+tile)*4096+lo_at,lo)
    else:
        physical=x^(((x>>6)&3)<<3)
        tl.store(P+(head*JT+tile)*4096+physical,hi)
        tl.store(P+(head*JT+tile)*4096+2048+physical,lo)

def pack_z(z,wide=False,half=False):
    b,h,n,r=z.shape
    if b!=1 or r!=32 or not z.is_contiguous():raise ValueError('contiguous B1 Gaussian32 required')
    packed=torch.empty((h,tr.cdiv(n,64),4096),device=z.device,dtype=torch.float16 if half else torch.bfloat16)
    _pack_z[(tr.cdiv(n,64),h)](z,packed,n,tr.cdiv(n,64),wide,half,num_warps=4)
    return packed

class PackedKernel(Kernel):
    def __call__(self,q,k,v,z,nu,scale,threshold,*,prefix_tiles,held=None,sticky=0.,schedule=3,trace=False,tma=False,prepacked=None,phase_profile=False):
        import ctypes as ct
        import math
        from .v2_online_sm90 import Params
        b,h,n,d=q.shape;hk,nk=k.shape[1:3];qb=tr.cdiv(n,128);jt=tr.cdiv(nk,64)
        if (b!=1 or d!=512 or h%hk or k.shape!=v.shape or
            any(x.dtype!=torch.bfloat16 or x.device!=q.device or not x.is_contiguous() for x in (q,k,v)) or
            z.shape!=(b,hk,nk,32) or z.dtype!=torch.float32 or not z.is_contiguous() or z.device!=q.device or
            nu.shape!=(b,hk) or nu.dtype!=torch.float32 or not nu.is_contiguous() or nu.device!=q.device or
            not math.isfinite(threshold) or threshold<0 or not math.isfinite(sticky) or not 0<=prefix_tiles<=jt):
            raise ValueError('finite native QKV/sketch geometry and fixed threshold required')
        if held is not None and (held.numel()!=h*qb*jt or held.dtype!=torch.bool or held.device!=q.device):
            raise ValueError('matching current held map required')
        if trace and self.production:raise ValueError('diagnostic binary required')
        if phase_profile and not self.options.get('phase_profile'):raise ValueError('explicit stage-clock diagnostic binary required')
        packed=pack_z(z,self.options.get('widepz',False),self.options.get('fp16diag',False)) if prepacked is None else prepacked
        if packed.shape!=(hk,jt,4096) or packed.dtype!=(torch.float16 if self.options.get('fp16diag',False) else torch.bfloat16) or not packed.is_contiguous() or packed.device!=q.device:raise ValueError('packed sketch geometry')
        pk=stage_qk(q,k) if self.options.get('staged_qk') else (pack_k(k) if self.options.get('packedk') else None)
        if self.options.get('global_queue'):
            # One private resident ring per captured invocation. Queue counters
            # are initialized inside the cooperative kernel before grid sync.
            pk=torch.empty((h*qb*self.options.get('queue_units_per_qblock',2)*self.options['queue_bytes'],),device=q.device,dtype=torch.uint8)
        out=torch.empty_like(q);skipped=torch.empty((1,h,qb,jt),device=q.device,dtype=torch.bool);eligible=torch.empty_like(skipped)
        normalizer=torch.empty((1,h,n),device=q.device);state=torch.empty((1,h,n,32) if trace else (1,),device=q.device);risks=torch.empty(skipped.shape if trace else (1,),device=q.device)
        if phase_profile:risks=torch.empty((h,qb,4,jt,5),device=q.device)
        if held is not None:held=held.reshape_as(skipped).contiguous()
        p=Params(*(x.data_ptr() if x is not None else 0 for x in (q,k,v,z,nu,pk,out,skipped,eligible,normalizer,state,risks)),b,h,hk,n,nk,d,0,1,1,int(trace or phase_profile),scale,math.log(threshold) if threshold else -math.inf,0,2,packed.data_ptr(),0,0,0 if held is None else held.data_ptr(),sticky,prefix_tiles)
        status=self.fn(ct.byref(p),torch.cuda.current_stream(q.device).cuda_stream,schedule)
        if status:raise RuntimeError(f'CUDA launch error {status}')
        return out.transpose(1,2),~skipped,state if trace else None,normalizer if trace else None,risks if trace or phase_profile else None


class PackedRefresh(OnlineRefresh):
    """Complete refresh charging fresh projection, packed sketch, output and lists."""
    def __init__(self,library,schedule=3):
        super().__init__(library)
        self.kernel=PackedKernel(library)
        self.schedule=schedule

    def __call__(self,q,k,v,matrix,scale,prefix_tiles,threshold,*,held=None,sticky=0.):
        from .v31_value_projection import refresh as project
        b,h,n,d=q.shape;hk,nk=k.shape[1:3]
        if b!=1 or d!=512 or k.shape!=v.shape or h%hk or not 0<=prefix_tiles<=tr.cdiv(nk,64):raise ValueError('native QKV geometry required')
        z=torch.empty((b,hk,nk,32),device=q.device);norm=torch.empty((b,hk,nk),device=q.device)
        key=(b,hk,nk,str(q.device))
        if key not in self.valid:self.valid[key]=torch.ones_like(norm,dtype=torch.bool)
        nu=project(v,matrix,z,norm,self.valid[key],0)
        out,keep,*_=self.kernel(q,k,v,z,nu,scale,threshold,prefix_tiles=prefix_tiles,held=held,sticky=sticky,schedule=self.schedule)
        return out,keep,pack_lists(keep)


@tr.jit
def _pack_k(K,PK,N:tl.constexpr,JT:tl.constexpr):
    tile,head,chunk=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    x=chunk*4096+tl.arange(0,4096);key=tile*64+x//512;dim=x%512
    value=tl.load(K+(head*N+key)*512+dim,key<N,0.)
    dest=(head*JT+tile)*32768+(dim//64)*4096+(x//512)*64+dim%64
    tl.store(PK+dest,value)

def pack_k(k):
    _,h,n,d=k.shape
    pk=torch.empty((h,tr.cdiv(n,64),32768),device=k.device,dtype=k.dtype)
    _pack_k[(tr.cdiv(n,64),h,8)](k,pk,n,tr.cdiv(n,64),num_warps=4)
    return pk

# Explicit alternative projection: BF16 V times high/residual BF16 matrix.
# Original Gaussian FP32 matrix is untouched; this is a qualified numerical arm.
@tr.jit
def _project_split(V,R,Z,NORM,N:tl.constexpr,H:tl.constexpr):
    block,head=tl.program_id(0),tl.program_id(1)
    row=block*32+tl.arange(0,32);rr=tl.arange(0,32);dd=tl.arange(0,64)
    acc=tl.full((32,32),0.,tl.float32);norm=tl.full((32,),0.,tl.float32)
    for chunk in range(8):
        dim=chunk*64+dd
        v=tl.load(V+(head*N+row[:,None])*512+dim[None,:],row[:,None]<N,0.)
        r=tl.load(R+(head*512+dim[:,None])*32+rr[None,:])
        hi=r.to(tl.bfloat16);lo=(r-hi.to(tl.float32)).to(tl.bfloat16)
        acc=tl.dot(v,lo,acc);acc=tl.dot(v,hi,acc)
        vf=v.to(tl.float32);norm+=tl.sum(vf*vf,1)
    norm=tl.sqrt(norm);norm=norm*norm
    tl.store(Z+(head*N+row[:,None])*32+rr[None,:],acc,row[:,None]<N)
    tl.store(NORM+head*N+row,norm,row<N)

def project_split(v,matrix,z,norm,valid,start=0):
    from .v31_value_projection import _parts,_finish
    b,h,n,d=v.shape
    if b!=1 or d!=512 or start!=0:raise ValueError('full native-value refresh only')
    _project_split[(tr.cdiv(n,32),h)](v,matrix,z,norm,n,h,num_warps=4,num_stages=2,enable_fp_fusion=False)
    chunks=tr.cdiv(n,4096);parts=torch.empty((b,h,chunks,2),device=v.device);nu=torch.empty((b,h),device=v.device)
    _parts[(chunks,h,b)](norm,valid,parts,n,h,chunks,*valid.stride(),num_warps=4,enable_fp_fusion=False)
    _finish[(b*h,)](parts,nu,chunks,num_warps=4,enable_fp_fusion=False)
    return nu


@tr.jit
def _stage_qk(Q,K,S,N:tl.constexpr,NK:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,JT:tl.constexpr):
    block,h,j=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    row=block*32+tl.arange(0,32);key=j*64+tl.arange(0,64);dd=tl.arange(0,512);kh=h//(H//HK)
    q=tl.load(Q+(h*N+row[:,None])*512+dd[None,:],row[:,None]<N,0.)
    k=tl.load(K+(kh*NK+key[None,:])*512+dd[:,None],key[None,:]<NK,0.)
    score=tl.dot(q,k)
    base=((h*tr.cdiv(N,128)+row//128)*JT+j)*128+row%128
    tl.store(S+base[:,None]*64+tl.arange(0,64)[None,:],score,row[:,None]<N)

def stage_qk(q,k):
    _,h,n,d=q.shape;hk,nk=k.shape[1:3];jt=tr.cdiv(nk,64)
    s=torch.zeros((h,tr.cdiv(n,128),jt,128,64),device=q.device)
    _stage_qk[(tr.cdiv(n,32),h,jt)](q,k,s,n,nk,h,hk,jt,num_warps=4)
    return s


class FusedPackedRefresh(PackedRefresh):
    """Qualified optional projection+packing fusion; all current V charged."""
    def __call__(self,q,k,v,matrix,scale,prefix_tiles,threshold,*,held=None,sticky=0.):
        from .fused_value_pack import project
        b,h,n,d=q.shape;hk,nk=k.shape[1:3]
        z=torch.empty((b,hk,nk,32),device=q.device);norm=torch.empty((b,hk,nk),device=q.device)
        key=(b,hk,nk,str(q.device))
        if key not in self.valid:self.valid[key]=torch.ones_like(norm,dtype=torch.bool)
        nu,packed=project(v,matrix,z,norm,self.valid[key])
        out,keep,*_=self.kernel(q,k,v,z,nu,scale,threshold,prefix_tiles=prefix_tiles,held=held,sticky=sticky,schedule=self.schedule,prepacked=packed)
        return out,keep,pack_lists(keep)

class CanvasPackedRefresh(PackedRefresh):
    """Reuse only unchanged-prefix projection within a single native canvas.

    The caller supplies the layer/canvas identity. Every new canvas recomputes
    all current V; refresh within that canvas projects only current canvas V.
    QK/softmax/online decisions/full PV remain current and uncached.
    """
    def __init__(self,library,schedule=2):
        super().__init__(library,schedule);self.canvas_state={}
    def clear(self):self.canvas_state.clear()
    def __call__(self,q,k,v,matrix,scale,prefix_tiles,threshold,*,held=None,sticky=0.,cache_key,canvas,prefix):
        from .fused_value_pack import project
        b,h,n,d=q.shape;hk,nk=k.shape[1:3];tag=(canvas,nk,matrix.data_ptr(),str(v.device))
        state=self.canvas_state.get(cache_key)
        if state is None or state['tag']!=tag:
            state=dict(tag=tag,z=torch.empty((b,hk,nk,32),device=v.device),norm=torch.empty((b,hk,nk),device=v.device),valid=torch.ones((b,hk,nk),device=v.device,dtype=torch.bool),packed=None)
            self.canvas_state[cache_key]=state;start=0
        else:start=prefix
        nu,packed=project(v,matrix,state['z'],state['norm'],state['valid'],start=start,packed=state['packed']);state['packed']=packed
        out,keep,*_=self.kernel(q,k,v,state['z'],nu,scale,threshold,prefix_tiles=prefix_tiles,held=held,sticky=sticky,schedule=self.schedule,prepacked=packed)
        return out,keep,pack_lists(keep)
