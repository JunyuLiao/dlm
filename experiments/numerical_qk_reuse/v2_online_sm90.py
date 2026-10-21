"""Explicitly built Hopper WGMMA online V2 with independent output softmax.

Derived from the attributed experimental value-direction Hopper implementation.
Each query half computes QK/PZ once, and a cluster vote covers all Q128 rows.
Skipped tiles advance routing mass but never execute full PV. Full-dimensional
output has its own retained-only normalizer, and never consumes routing state.
"""
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import subprocess
import torch


class Params(ct.Structure):
    _fields_=[(x,ct.c_void_p) for x in ('q','k','v','z','reference','mask','output','skipped','eligible','log_normalizer','projected_state','risks')]+[
        (x,ct.c_int) for x in ('batch','heads','kv_heads','queries','keys','width','mask_kind','mask_heads','mode','trace')]+[
        ('scale',ct.c_float),('log_threshold',ct.c_float),('timings',ct.c_void_p),('projected_precision',ct.c_int),
        ('debug_scores',ct.c_void_p),('sensitivity',ct.c_void_p),('export_log_risk',ct.c_void_p),
        ('held',ct.c_void_p),('sticky',ct.c_float),('protect_tile',ct.c_int)]


def build(directory,*,fast=True,inline=True,production=False,dynamic_registers=False,bf16x3=False,lean=False,compact=False,bf16single=False,staged=False,onebarrier=False,defer_p=False):
    root=Path(__file__).resolve().parents[2]
    source=Path(__file__).with_name('csrc')/'v2_online_sm90.cu'
    header=source.with_suffix('.h')
    if lean:source=source.with_name('v2_online_lean_sm90.cu')
    if compact:source=source.with_name('v2_online_compact_sm90.cu')
    if staged:source=source.with_name('v2_online_staged_sm90.cu')
    cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
    trt=root/'reference/TensorRT-LLM-value-aware/cpp/kernels/fmha_v2/src'
    files=[source,header,trt/'fmha/hopper/utils_hgmma_bf16.h',trt/'fmha/hopper/utils_warpgroup.h']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    options=dict(fast=fast,inline=inline,float_operands=True,ptx_cluster=True,online_ratio=True,
                 production=production,dynamic_registers=dynamic_registers,bf16x3=bf16x3,lean=lean,compact=compact,bf16single=bf16single,staged=staged,onebarrier=onebarrier,defer_p=defer_p)
    builder_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key=hashlib.sha256(json.dumps([hashes,options,builder_sha],sort_keys=True).encode()).hexdigest()[:16]
    path=Path(directory)/key;path.mkdir(parents=True,exist_ok=True)
    library=path/'v2_online_sm90.so'
    if not library.exists():
        src=source.read_text()
        if defer_p:
            if not lean or compact or staged:raise ValueError('deferred P staging requires lean roles')
            begin=src.index('            output_m[r]=fmaxf(om[r],mx[r]);')
            end=src.index('\n        }\n        float mu[2][8]',begin)
            staging=src[begin:end]
            src=src[:begin]+src[end:]
            needle='''        bool take=!decision.skip;
        if(t==0&&half==0)'''
            if src.count(needle)!=1:raise RuntimeError('P staging patch drift')
            src=src.replace(needle,'''        bool take=!decision.skip;
        if(take){
            #pragma unroll
            for(int r=0;r<2;++r){
                int row=warp*16+lane/4+r*8;
'''+staging+'''
            }
            named_sync(1);proxy_fence();
            cluster_rendezvous();
        }
        if(t==0&&half==0)''')
            # The PV roles wait for P only on retained tiles. This adds a
            # rendezvous but eliminates output staging on rejected blocks.
            needle='''            if(a.mode==2)cluster_rendezvous();'''
            src=src.replace(needle,'''            cluster_rendezvous();''')
            src=src.replace('schedule>2','schedule>1')
        if onebarrier:
            if not lean or compact or staged:raise ValueError('double-buffered votes require lean roles')
            # The next cluster rendezvous guarantees all readers have consumed
            # the previous slot before the router can reuse it two tiles later.
            src=src.replace('float worst;\n    float limit;\n    int eligible;',
                            'float worst[2];\n    float limit;\n    int eligible[2];')
            import re
            src=re.sub(r's\.(worst|eligible)\b',r's.\1[j&1]',src)
            src=re.sub(r'(first|second)->(worst|eligible)\b',r'\1->\2[j&1]',src)
            needle='''    cluster_rendezvous();
    // v5: mandatory tiles'''
            if src.count(needle)!=1:raise RuntimeError('cluster vote patch drift')
            src=src.replace(needle,'''    // Double-buffered vote: the next tile's rendezvous protects slot reuse.
    // v5: mandatory tiles''')
        if production:
            src=src.replace('int const mask_kind=a.mask_kind;','int const mask_kind=0;')
            src=src.replace('if(a.trace)','if constexpr(false)')
            src=src.replace('if(a.export_log_risk &&','if(false &&')
            src=src.replace('if(a.sensitivity &&','if(false &&')
            src=src.replace('uint64_t started=a.timings?timer():0;','uint64_t started=0;')
            src=src.replace('a.timings ? timer():0','0')
            src=src.replace('a.timings?timer():0','0')
            src=src.replace('if(a.timings) {','if(false) {')
            # Production launch has no trace/state export; wrappers reject trace.
        if dynamic_registers:
            needle='''    if(t<128) {
        if constexpr(!FIXED)'''
            replacement='''#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    if(t>=128)asm volatile("setmaxnreg.dec.sync.aligned.u32 136;");
    else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
#endif
    if(t<128) {
        if constexpr(!FIXED)'''
            if src.count(needle)!=1:raise RuntimeError('register-role patch drift')
            src=src.replace(needle,replacement)
        (path/source.name).write_text(src)
        (path/header.name).write_bytes(header.read_bytes())
        cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC',
             '-DVD_F32_OPERANDS=1','-DVD_PTX_CLUSTER=1','-DVD_ONLINE_RATIO=1','-lineinfo','--ptxas-options=-v',
             '-I'+str(cutlass),'-I'+str(trt),'-I/usr/local/cuda/include/cccl',str(path/source.name),
             '-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
        if fast:cmd.insert(1,'-DVD_FAST_SFU=1')
        if inline:cmd.insert(1,'-DVD_INLINE_ROLES=1')
        if bf16x3:cmd.insert(1,'-DV2_PZ_BF16X3=1')
        if bf16single:cmd.insert(1,'-DV2_PZ_BF16_SINGLE=1')
        p=subprocess.run(cmd,text=True,capture_output=True)
        (path/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,
             builder_sha256=builder_sha,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
        if p.returncode:raise RuntimeError(p.stderr[-9000:])
    return library


class Kernel:
    def __init__(self,path):
        self.path=Path(path)
        self.options=json.loads((self.path.parent/'build_receipt.json').read_text())['options']
        self.production=self.options.get('production',False)
        self.lib=ct.CDLL(str(self.path))
        self.lib.v2_online_sm90_params_size.restype=ct.c_uint32
        if self.lib.v2_online_sm90_params_size()!=ct.sizeof(Params):raise RuntimeError('ABI size mismatch')
        self.fn=self.lib.v2_online_sm90_sm90
        self.fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int];self.fn.restype=ct.c_int
        self.tma_fn=getattr(self.lib,'v2_online_sm90_sm90_tma',None)
        if self.tma_fn is not None:self.tma_fn.argtypes=self.fn.argtypes;self.tma_fn.restype=ct.c_int

    def __call__(self,q,k,v,z,nu,scale,threshold,*,prefix_tiles,held=None,sticky=0.,schedule=1,trace=False,tma=False):
        if trace and self.production:raise ValueError('production binary has no diagnostic state export')
        if self.options.get('defer_p',False) and schedule==2:raise ValueError('deferred P staging is unsupported by split roles')
        b,h,n,d=q.shape;hk,nk=k.shape[1:3];qb=(n+127)//128;jt=(nk+63)//64
        if d!=512 or b!=1 or h%hk or k.shape!=v.shape or not all(x.is_contiguous() for x in (q,k,v,z,nu)):
            raise ValueError('contiguous native D512 batch-one QKV/Z required')
        if any(x.device!=q.device for x in (k,v,z,nu)) or q.dtype!=torch.bfloat16 or z.dtype!=torch.float32:
            raise ValueError('BF16 QKV and current FP32 sketch required')
        if held is not None and (held.numel()!=h*qb*jt or held.dtype!=torch.bool or held.device!=q.device):
            raise ValueError('matching current held map required')
        out=torch.empty_like(q);skipped=torch.empty((1,h,qb,jt),device=q.device,dtype=torch.bool)
        eligible=torch.empty_like(skipped);normalizer=torch.empty((1,h,n),device=q.device)
        state=torch.empty((1,h,n,32) if trace else (1,),device=q.device)
        risks=torch.empty(skipped.shape if trace else (1,),device=q.device)
        if held is not None:held=held.reshape_as(skipped).contiguous()
        p=Params(*(x.data_ptr() if x is not None else 0 for x in (q,k,v,z,nu,None,out,skipped,eligible,normalizer,state,risks)),
            b,h,hk,n,nk,d,0,1,1,int(trace),scale,math.log(threshold) if threshold else -math.inf,
            0,2,0,0,0,0 if held is None else held.data_ptr(),sticky,prefix_tiles)
        if tma and (self.tma_fn is None or schedule==0):raise ValueError('TMA entrypoint unavailable')
        status=(self.tma_fn if tma else self.fn)(ct.byref(p),torch.cuda.current_stream(q.device).cuda_stream,schedule)
        if status:raise RuntimeError(f'CUDA launch error {status}')
        return out.transpose(1,2),~skipped,state if trace else None,normalizer if trace else None,risks if trace else None
