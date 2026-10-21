"""Build the independently named key-partitioned online V2 attention kernel."""
import hashlib
import json
from pathlib import Path
import subprocess
import os
import torch
from .v2_online_sm90 import Kernel


def build(directory,*,production=False,single_qk=False,stage_logits=False,profile=False,prefetch_staging=False,leader_vote=False,fused_norm=False,async_vote=False,tma3d=False,wavefront=False,wavefront_async=False,wavefront_tma=False,tma_all=False,overlap=False,stash=False,exp2=False,pzfull=False,duplicate_qk=False,fast_div=False,channel_split=False,channel_stage=False,channel_pair=False,channel_serial=False,channel_router=False,channel_duplicate=False,overlap_router=False,qk_unroll=None):
    root=Path(__file__).resolve().parents[2]
    source=Path(__file__).with_name('csrc')/'v2_online_partitioned_sm90.cu'
    if single_qk:source=source.with_name('v2_online_singleqk_sm90.cu')
    if prefetch_staging:source=source.with_name('v2_online_prefetch_sm90.cu')
    if fused_norm:source=source.with_name('v2_online_fusednorm_sm90.cu')
    if tma3d:source=source.with_name('v2_online_tma3d_sm90.cu')
    if wavefront:source=source.with_name('v2_online_wavefront_sm90.cu')
    if wavefront_async:source=source.with_name('v2_online_wavefront_async_sm90.cu')
    if wavefront_tma:source=source.with_name('v2_online_wavefront_tma_sm90.cu')
    if tma_all:source=source.with_name('v2_online_tma_all_sm90.cu')
    if overlap:source=source.with_name('v2_online_overlap_sm90.cu')
    if stash:source=source.with_name('v2_online_overlap_stash_sm90.cu')
    if exp2:source=source.with_name('v2_online_exp2_sm90.cu')
    if pzfull:source=source.with_name('v2_online_pzfull_sm90.cu')
    if duplicate_qk:source=source.with_name('v2_online_duplicateqk_sm90.cu')
    if channel_split:source=source.with_name('v2_online_channel_split_sm90.cu')
    if channel_stage:source=source.with_name('v2_online_channel_stage_sm90.cu')
    if channel_pair:source=source.with_name('v2_online_channel_pair_sm90.cu')
    if channel_serial:source=source.with_name('v2_online_channel_pair_serial_sm90.cu')
    if channel_router:source=source.with_name('v2_online_channel_router_sm90.cu')
    if channel_duplicate:source=source.with_name('v2_online_channel_duplicate_sm90.cu')
    if overlap_router:source=source.with_name('v2_online_overlap_router_sm90.cu')
    header=source.with_name('v2_online_sm90.h')
    cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,header,Path(__file__))}
    options=dict(production=production,partitioned=True,single_qk=single_qk,stage_logits=stage_logits,profile=profile,prefetch_staging=prefetch_staging,leader_vote=leader_vote,fused_norm=fused_norm,async_vote=async_vote,tma3d=tma3d,wavefront=wavefront or wavefront_async or wavefront_tma,wavefront_async=wavefront_async or wavefront_tma,wavefront_tma=wavefront_tma,tma_all=tma_all,overlap=overlap,stash=stash,exp2=exp2,pzfull=pzfull,duplicate_qk=duplicate_qk,fast_div=fast_div,channel_split=channel_split,channel_stage=channel_stage,channel_pair=channel_pair,channel_serial=channel_serial,channel_router=channel_router,channel_duplicate=channel_duplicate,overlap_router=overlap_router,qk_unroll=qk_unroll,pz='bf16x3',output='retained_max_scaled_sum',
                 schedules={0:'conditional_v',1:'early_v',2:'early_v_next_k',3:'conditional_v_next_k'})
    key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16]
    path=Path(directory)/key;path.mkdir(parents=True,exist_ok=True)
    library=path/'v2_online_sm90.so'
    if not library.exists():
        src=source.read_text()
        if qk_unroll is not None:
            if qk_unroll not in (1,2,4,8,16):raise ValueError('QK unroll must be 1/2/4/8/16')
            src=src.replace('#pragma unroll 4\n        for(int kk=0;kk<512;', '#pragma unroll\n        for(int kk=0;kk<512;')
            needle='#pragma unroll\n        for(int kk=0;kk<512;'
            if src.count(needle)!=1:raise RuntimeError('QK unroll patch drift')
            src=src.replace(needle,f'#pragma unroll {qk_unroll}\n        for(int kk=0;kk<512;')
        if leader_vote or async_vote:
            src=src.replace('warp_risk[4],vote[2];','warp_risk[4],vote[2];\n    int take,epoch;')
            begin=src.index('        auto cluster=cg::this_cluster();auto* first=')
            end=src.index('        if(take){',begin)
            vote=src[begin:end]
            decision='        bool take=j==0||j>=a.protect_tile||!(risk<limit);'
            if vote.count(decision)!=1:raise RuntimeError('leader vote decision patch drift')
            vote=vote.replace(decision,decision+'\n        s.take=take;')
            src=src[:begin]+'''        if(t==0){
'''+vote+'''        }
        __syncthreads();bool take=s.take;
'''+src[end:]
        if async_vote:
            src=src.replace('namespace v2_pipeline {','''namespace v2_pipeline {
__device__ __forceinline__ void publish_epoch(int* ptr,int epoch){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;"::"r"(addr),"r"(epoch):"memory");
}
__device__ __forceinline__ int partner_epoch(int* ptr,int rank){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(ptr)),remote;int epoch;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(remote):"r"(addr),"r"(rank));
    asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];":"=r"(epoch):"r"(remote):"memory");
    return epoch;
}
''')
            src=src.replace('    load_k(a,s,0,kh);copy_wait();__syncthreads();proxy();',
                '    if(t==0)s.epoch=0;\n    load_k(a,s,0,kh);copy_wait();__syncthreads();proxy();cluster_sync();')
            if 'if(t==0)s.epoch=0;' not in src:
                anchor='    float output['
                if src.count(anchor)!=1:raise RuntimeError('epoch initialization patch drift')
                src=src.replace(anchor,'    if(t==0)s.epoch=0;__syncthreads();cluster_sync();\n'+anchor)
            if src.count('if(t==0)s.epoch=0;')!=1:raise RuntimeError('epoch initialization missing')
            needle='''        cluster_sync();
        if(t==0){'''
            if src.count(needle)!=1:raise RuntimeError('async vote patch drift')
            src=src.replace(needle,'''        if(t==0){
        publish_epoch(&s.epoch,j+1);
        while(partner_epoch(&s.epoch,1-half)<j+1){}
''')
        if stage_logits:
            if not single_qk:raise ValueError('staged logits require the single-QK source')
            needle='''            warpgroup_commit();warpgroup_wait<0>();fence(score);
            #pragma unroll
            for(int r=0;r<2;++r){'''
            replacement='''            warpgroup_commit();warpgroup_wait<0>();fence(score);
            #pragma unroll
            for(int r=0;r<2;++r)for(int n=0;n<8;++n)for(int c=0;c<2;++c){
                int row=warp*16+lane/4+r*8,key=n*8+lane%4*2+c;
                probability[row*64+key]=score[n*4+r*2+c];
            }
            asm volatile("bar.sync 2, 128;":::"memory");
            #pragma unroll
            for(int r=0;r<2;++r){
                float local_score[16];
                #pragma unroll
                for(int n=0;n<8;++n)for(int c=0;c<2;++c){
                    int row=warp*16+lane/4+r*8,key=n*8+lane%4*2+c;
                    local_score[n*2+c]=probability[row*64+key];
                }'''
            if src.count(needle)!=1:raise RuntimeError('staged logits source drift')
            src=src.replace(needle,replacement)
            start=src.index('                float local_score[16];')
            end=src.index('        }else{',start)
            body=src[start:end].replace('score[reg]','local_score[n*2+c]')
            src=src[:start]+body+src[end:]
        if profile:
            src=src.replace('namespace v2_pipeline {','''namespace v2_pipeline {
__device__ __forceinline__ unsigned long long stamp_clock(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
#define STAMP(F) if(t==0&&a.timings){a.timings[((blockIdx.y*gridDim.x+blockIdx.x)*tiles+j)*6+F]=stamp_clock()-phase_start;phase_start=stamp_clock();}
''')
            src=src.replace('    for(int j=0;j<tiles;++j){','    for(int j=0;j<tiles;++j){\n        unsigned long long phase_start=0;if(t==0&&a.timings)phase_start=stamp_clock();')
            src=src.replace('        float mu[8]={};','        STAMP(0);\n        float mu[8]={};')
            src=src.replace('        float worst=0.f;','        STAMP(1);\n        float worst=0.f;')
            src=src.replace('        if(take){','        STAMP(2);\n        if(take){')
            src=src.replace('        if constexpr(!PREFETCH_K)if(j+1<tiles)', '        STAMP(3);\n        if constexpr(!PREFETCH_K)if(j+1<tiles)')
            src=src.replace('        if constexpr(PREFETCH_K){if(!take', '        STAMP(3);\n        if constexpr(PREFETCH_K){if(!take')
            src=src.replace('        copy_wait();__syncthreads();proxy();\n    }','        copy_wait();__syncthreads();proxy();\n        STAMP(4);\n    }')
        if production:src=src.replace('if(a.trace)','if constexpr(false)').replace('if(a.trace&&channel==0)','if constexpr(false)')
        (path/source.name).write_text(src);(path/header.name).write_bytes(header.read_bytes())
        cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC',
             '-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',
             str(path/source.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart',
             '-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
        if fast_div:cmd.insert(1,'--prec-div=false')
        tmp=path/'tmp';tmp.mkdir(exist_ok=True)
        env=os.environ.copy();env['TMPDIR']=str(tmp.resolve())
        result=subprocess.run(cmd,text=True,capture_output=True,env=env)
        (path/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
        if result.returncode:raise RuntimeError(result.stderr[-9000:])
    return library


# One pass compacts an existing keep map; no top-k, sort or host readback.
import triton as tr
import triton.language as tl


@tr.jit
def _pack_lists(KEEP,CNT,IDX,ZCNT,ZIDX,JT:tl.constexpr,BLOCK:tl.constexpr):
    unit=tl.program_id(0);j=tl.arange(0,BLOCK)
    chosen=tl.load(KEEP+unit*JT+j,j<JT,False)
    pos=tl.cumsum(chosen.to(tl.int32),0)-1;count=tl.sum(chosen.to(tl.int32),0)
    tl.store(IDX+unit*JT+pos,j,chosen&(j<JT))
    tl.store(IDX+unit*JT+j,0,(j>=count)&(j<JT))
    tl.store(CNT+unit,count);tl.store(ZCNT+unit,0);tl.store(ZIDX+unit,0)


def pack_lists(keep):
    """Construct FA4 lists with one GPU kernel, including deterministic padding."""
    from . import v27_fa4
    v27_fa4.load()
    if keep.dtype!=torch.bool or keep.ndim!=4 or not keep.is_contiguous():raise ValueError('contiguous B,H,QB,KV bool map required')
    b,h,qb,jt=keep.shape
    count=torch.empty((b,h,qb),device=keep.device,dtype=torch.int32)
    indices=torch.empty((b,h,qb,jt),device=keep.device,dtype=torch.int32)
    zero_count=torch.empty_like(count);zero_indices=torch.empty((b,h,qb,1),device=keep.device,dtype=torch.int32)
    _pack_lists[(b*h*qb,)](keep,count,indices,zero_count,zero_indices,jt,tr.next_power_of_2(jt),num_warps=4)
    return v27_fa4._BST(mask_block_cnt=zero_count,mask_block_idx=zero_indices,
        full_block_cnt=count,full_block_idx=indices,block_size=(128,64))


class OnlineRefresh:
    """Qualified all-TMA refresh: current-V projection, online attention and lists.

    This explicit experimental entry point does not install native model hooks.
    Inputs use B,H,N,D contiguous BF16 QKV and a current FP32 Gaussian matrix.
    Returns model-major B,N,H,D output, the reusable Q128/KV64 map and FA4 lists.
    Finite inputs and positive current-value RMS are required.
    """
    def __init__(self,library):
        self.kernel=Kernel(library)
        if not self.kernel.options.get('tma_all') or not self.kernel.options.get('async_vote'):
            raise ValueError('qualified all-TMA acquire/release binary required')
        self.valid={}

    def __call__(self,q,k,v,matrix,scale,prefix_tiles,threshold,*,held=None,sticky=0.):
        from .v31_value_projection import refresh as project
        b,h,n,d=q.shape;hk,nk=k.shape[1:3]
        if (b!=1 or d!=512 or k.shape!=v.shape or h%hk or
            any(x.dtype!=torch.bfloat16 or x.device!=q.device or not x.is_contiguous() for x in (q,k,v)) or
            not 0<=prefix_tiles<=nk//64 or threshold<0):
            raise ValueError('finite contiguous native QKV and valid prefix/threshold required')
        z=torch.empty((b,hk,nk,32),device=q.device,dtype=torch.float32)
        norm=torch.empty((b,hk,nk),device=q.device,dtype=torch.float32)
        key=(b,hk,nk,str(q.device))
        if key not in self.valid:self.valid[key]=torch.ones_like(norm,dtype=torch.bool)
        valid=self.valid[key]
        nu=project(v,matrix,z,norm,valid,0)
        out,keep,*_=self.kernel(q,k,v,z,nu,scale,threshold,prefix_tiles=prefix_tiles,
            held=held,sticky=sticky,schedule=3)
        return out,keep,pack_lists(keep)


class WavefrontKernel:
    """Online four-block cluster plus reduction of full-PV partials; no QK replay."""
    def __init__(self,path):
        import ctypes as ct
        self.path=Path(path);self.options=json.loads((self.path.parent/'build_receipt.json').read_text())['options']
        if not self.options.get('wavefront'):raise ValueError('wavefront binary required')
        self.production=self.options.get('production',False);self.lib=ct.CDLL(str(path))
        self.fn=self.lib.v2_online_wavefront_launch;self.fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int];self.fn.restype=ct.c_int
        self.combine=self.lib.v2_online_wavefront_combine
        self.combine.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p];self.combine.restype=ct.c_int

    def __call__(self,q,k,v,z,nu,scale,threshold,*,prefix_tiles,held=None,sticky=0.,schedule=0,trace=False,**kwargs):
        import ctypes as ct
        import math
        from .v2_online_sm90 import Params
        b,h,n,d=q.shape;nk=k.shape[2];hk=k.shape[1];qb=(n+127)//128;jt=(nk+63)//64
        if b!=1 or d!=512 or any(not x.is_contiguous() for x in (q,k,v,z,nu)):raise ValueError('native contiguous geometry required')
        if trace and self.production:raise ValueError('production binary has no trace')
        output=torch.empty_like(q);partial=torch.empty((4,h,n,512),device=q.device);stats=torch.empty((4,h,n,2),device=q.device)
        skipped=torch.empty((1,h,qb,jt),device=q.device,dtype=torch.bool);eligible=torch.empty_like(skipped)
        lm=torch.empty((1,h,n),device=q.device);state=torch.empty((1,h,n,32) if trace else (1,),device=q.device);risk=torch.empty_like(skipped,dtype=torch.float32) if trace else torch.empty((1,),device=q.device)
        if held is not None:held=held.reshape_as(skipped).contiguous()
        params=Params(*(x.data_ptr() if x is not None else 0 for x in (q,k,v,z,nu,None,partial,skipped,eligible,lm,state,risk)),
            b,h,hk,n,nk,d,0,1,1,int(trace),scale,math.log(threshold) if threshold else -math.inf,0,2,stats.data_ptr(),0,0,
            0 if held is None else held.data_ptr(),sticky,prefix_tiles)
        stream=torch.cuda.current_stream().cuda_stream
        status=self.fn(ct.byref(params),stream,schedule)
        if status:raise RuntimeError(f'wavefront launch error {status}')
        status=self.combine(partial.data_ptr(),stats.data_ptr(),output.data_ptr(),n,h,stream)
        if status:raise RuntimeError(f'wavefront combine error {status}')
        return output.transpose(1,2),~skipped,state if trace else None,lm if trace else None,risk if trace else None
