"""Stream-safe ctypes harness for the allocation-free TensorRT FMHA C ABI.

Build is explicit, never performed during generation or CUDA graph capture.
This harness is not by itself TensorRT-LLM runtime integration.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from .fused import Output
from .masks import PackedMask

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).parent/'csrc'
BUILD=ROOT/'results/value_direction_hopper_v1/build'
_TORCH_BINARY=None


def source_digest(sources, path):
    """Accept migrated symlink aliases only when they identify the same file."""
    resolved=Path(path).resolve()
    matches={digest for name,digest in sources.items() if Path(name).resolve()==resolved}
    if len(matches)>1:raise RuntimeError('Conflicting provenance hashes for one resolved source')
    return next(iter(matches),None)


class Params(ct.Structure):
    _fields_=[(x,ct.c_void_p) for x in ('q','k','v','z','reference','mask','output','skipped','eligible','log_normalizer','projected_state','risks')]+[
        (x,ct.c_int) for x in ('batch','heads','kv_heads','queries','keys','width','mask_kind','mask_heads','mode','trace')]+[
        ('scale',ct.c_float),('log_threshold',ct.c_float),('timings',ct.c_void_p),('projected_precision',ct.c_int),('debug_scores',ct.c_void_p),
        ('sensitivity',ct.c_void_p)]


class ParamsV3(ct.Structure):
    """Frozen historical ABI; old binaries remain usable for unweighted controls."""
    _fields_=Params._fields_[:-1]


def build(*, fast_sfu=False,inline_roles=False,online_ratio=False):
    trt=ROOT/'reference/TensorRT-LLM-value-aware'
    cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
    files=[SOURCE/'value_direction.cu',SOURCE/'value_direction.h',
           trt/'cpp/kernels/fmha_v2/src/fmha/hopper/utils_hgmma_bf16.h',
           trt/'cpp/kernels/fmha_v2/src/fmha/hopper/utils_warpgroup.h']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    options={'fast_sfu':fast_sfu,'inline_roles':inline_roles,'online_ratio':online_ratio}
    name=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16]
    BUILD.mkdir(parents=True,exist_ok=True)
    dest=BUILD/f'value_direction_{name}.so'
    if not dest.exists():
        archive=BUILD/name
        archive.mkdir(exist_ok=True)
        for source in files:
            (archive/source.name).write_bytes(source.read_bytes())
        cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC',
             '-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I'+str(trt/'cpp/kernels/fmha_v2/src'),
             '-I/usr/local/cuda/include/cccl',str(SOURCE/'value_direction.cu'),'-o',str(dest),
             '-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
        if fast_sfu: cmd.insert(1,'-DVD_FAST_SFU=1')
        if inline_roles: cmd.insert(1,'-DVD_INLINE_ROLES=1')
        if online_ratio: cmd.insert(1,'-DVD_ONLINE_RATIO=1')
        result=subprocess.run(cmd,text=True,capture_output=True)
        proof=dict(command=cmd,sources=hashes,options=options,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        (BUILD/f'{name}.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(result.stdout+result.stderr,flush=True)
        result.check_returncode()
    print(dest,flush=True)
    return dest


class Kernel:
    def __init__(self,path,*,allow_legacy=False,torch_library=None):
        self.path=Path(path)
        self.torch_op=None
        self.lib=ct.CDLL(str(self.path))
        try:
            version=self.lib.value_direction_abi_version
            version.restype=ct.c_uint32
            size=self.lib.value_direction_params_size
            size.restype=ct.c_uint32
            self.abi_version=version()
            if self.abi_version==2 and allow_legacy:
                pass
            elif self.abi_version not in (3,4) or size()!=ct.sizeof(Params if self.abi_version==4 else ParamsV3): raise RuntimeError('Incompatible kernel ABI')
            self.timing_fields=6
        except AttributeError:
            if not allow_legacy: raise RuntimeError('Unversioned kernel; rebuild or explicitly select legacy benchmarking')
            self.timing_fields=4
            self.abi_version=1
        self.fn=self.lib.value_direction_sm90
        self.fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]
        self.fn.restype=ct.c_int
        self.tma_fn=getattr(self.lib,'value_direction_sm90_tma',None)
        if self.tma_fn is not None:
            self.tma_fn.argtypes=self.fn.argtypes;self.tma_fn.restype=ct.c_int
        if torch_library is not None:
            global _TORCH_BINARY
            bridge=Path(torch_library).resolve()
            if _TORCH_BINARY is not None and _TORCH_BINARY!=bridge:raise RuntimeError('Use separate processes for different ATen bridge binaries')
            proof=json.loads((bridge.parent/'provenance.json').read_text())
            expected=source_digest(proof['sources'],self.path)
            if expected!=hashlib.sha256(self.path.read_bytes()).hexdigest() or proof['torch']!=torch.__version__:
                raise RuntimeError('ATen bridge kernel/runtime provenance mismatch')
            if _TORCH_BINARY is None:torch.ops.load_library(str(bridge));_TORCH_BINARY=bridge
            self.torch_op=torch.ops.value_direction_hopper.attention

    def pack_mask(self,mask):
        """CUDA-only packing, also usable when the TRT image has no Triton JIT."""
        if mask.ndim!=4 or mask.dtype!=torch.bool or not mask.is_cuda or not mask.is_contiguous():
            raise ValueError('Contiguous B,H,Q,K bool CUDA mask required')
        b,h,q,k=mask.shape
        if min(b,h,q,k)<1:raise ValueError('Nonempty mask required')
        result=torch.empty((b,h,q,(k+63)//64),device=mask.device,dtype=torch.uint64)
        fn=self.lib.value_direction_pack_mask
        fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        fn.restype=ct.c_int
        with torch.cuda.device(mask.device):
            status=fn(mask.data_ptr(),result.data_ptr(),b,h,q,k,torch.cuda.current_stream(mask.device).cuda_stream)
        if status:raise RuntimeError(f'value_direction_pack_mask CUDA error {status}')
        return PackedMask(result,k)

    def __call__(self,q,k,v,z,reference,*,mask=None,scale=None,log_threshold=-float('inf'),mode='value',trace=False,overlap=True,timings=None,precision='ieee',debug_scores=None,split_pv=False,tma=False,sensitivity=None):
        if sensitivity is not None:
            if self.abi_version<4:raise ValueError('Query sensitivity requires the v4 kernel')
            if mode!='value' or sensitivity.shape!=(q.shape[0],q.shape[-2]) or sensitivity.dtype!=torch.float32 or not sensitivity.is_cuda or not sensitivity.is_contiguous() or sensitivity.device!=q.device:
                raise ValueError('Sensitivity must be contiguous FP32 B,Q on the Q device in value mode')
        if self.torch_op is not None and timings is None and debug_scores is None:
            if split_pv and (q.shape[-1]!=512 or mode!='value' or precision not in ('tf32x3_register','tf32x3_shared')):raise ValueError('Invalid split-PV specialization')
            packed=isinstance(mask,PackedMask)
            if packed and mask.keys!=k.shape[-2]:raise ValueError('Packed mask length mismatch')
            kind=3 if packed else 0 if mask is None else 1 if mask.dtype==torch.bool else 2
            tensor=mask.tensor if packed else q if mask is None else mask
            args=(q,k,v,z,reference,tensor,kind,q.shape[-1]**-.5 if scale is None else scale,
                log_threshold,{'dense':0,'value':1,'blasst':2}[mode],{'ieee':0,'tf32x3':1,'tf32x3_register':2,'tf32x3_shared':3}[precision],trace,tma,2 if split_pv else int(overlap))
            return Output(*self.torch_op(*args,*((sensitivity,) if self.abi_version>=4 else ())))
        if q.ndim!=4 or k.shape!=v.shape or k.ndim!=4: raise ValueError('Expected B,H,N,D')
        b,h,nq,d=q.shape
        bk,hk,nk,dk=k.shape
        if d not in (256,512) or bk!=b or dk!=d or h%hk or min(b,h,hk,nq,nk)<1: raise ValueError('Unsupported shape')
        if z.shape!=(b,hk,nk,32) or reference.shape!=(b,hk): raise ValueError('Gaussian32 native-head sketches required')
        if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype or z.dtype!=torch.float32 or reference.dtype!=torch.float32: raise ValueError('BF16 QKV, FP32 Z/ref required')
        if not q.is_cuda: raise ValueError('CUDA required')
        if torch.cuda.get_device_capability(q.device)!=(9,0): raise ValueError('SM90 H100/H200 required')
        for x in (q,k,v,z,reference):
            if x.device!=q.device or not x.is_contiguous(): raise ValueError('Contiguous tensors on one device required')
        kind,mh=0,1
        if isinstance(mask,PackedMask):
            packed=mask
            mask=packed.tensor
            if packed.keys!=nk or mask.shape!=(b,mask.shape[1],nq,(nk+63)//64) or mask.shape[1] not in (1,h) or mask.dtype!=torch.uint64 or mask.device!=q.device or not mask.is_contiguous():raise ValueError('Invalid packed mask')
            kind,mh=3,mask.shape[1]
        elif mask is not None:
            if mask.shape[0]!=b or mask.ndim!=4 or mask.shape[1] not in (1,h) or mask.shape[2:]!=(nq,nk): raise ValueError('Invalid mask shape')
            if mask.dtype not in (torch.bool,torch.bfloat16) or mask.device!=q.device or not mask.is_contiguous(): raise ValueError('Contiguous bool/BF16 device mask required')
            kind,mh=(1 if mask.dtype==torch.bool else 2),mask.shape[1]
        qb,kt=(nq+127)//128,(nk+63)//64
        if split_pv and (d!=512 or mode!='value' or precision not in ('tf32x3_register','tf32x3_shared') or kind!=3):raise ValueError('Split-PV specialization requires D512, packed masks, and register/shared TF32x3 value routing')
        if tma and (self.tma_fn is None or not overlap or precision not in ('tf32x3_register','tf32x3_shared') or kind!=3):raise ValueError('TMA specialization requires overlap, packed masks, and register/shared TF32x3 precision selection')
        if debug_scores is not None and (self.abi_version!=3 or debug_scores.shape!=(b,h,nq,nk) or debug_scores.dtype!=torch.float32 or debug_scores.device!=q.device or not debug_scores.is_contiguous()): raise ValueError('Invalid diagnostic score buffer or ABI')
        if timings is not None and (timings.shape!=(b,h,qb*(4 if split_pv else 2),kt,self.timing_fields) or timings.dtype!=torch.uint64 or timings.device!=q.device or not timings.is_contiguous()): raise ValueError('Invalid timing buffer')
        output=torch.empty_like(q)
        skipped=torch.empty((b,h,qb,kt),device=q.device,dtype=torch.bool)
        eligible=torch.empty_like(skipped)
        normalizer=torch.empty((b,h,nq),device=q.device,dtype=torch.float32)
        state=torch.empty((b,h,nq,32) if trace else (1,),device=q.device,dtype=torch.float32)
        risks=torch.empty(skipped.shape if trace else (1,),device=q.device,dtype=torch.float32)
        parameters=Params if self.abi_version>=4 else ParamsV3
        params=parameters(*(x.data_ptr() if x is not None else 0 for x in (q,k,v,z,reference,mask,output,skipped,eligible,normalizer,state,risks)),
            b,h,hk,nq,nk,d,kind,mh,{'dense':0,'value':1,'blasst':2}[mode],int(trace),d**-.5 if scale is None else scale,log_threshold,0 if timings is None else timings.data_ptr(),{'ieee':0,'tf32x3':1,'tf32x3_register':2,'tf32x3_shared':3}[precision],0 if debug_scores is None else debug_scores.data_ptr())
        if self.abi_version>=4:params.sensitivity=0 if sensitivity is None else sensitivity.data_ptr()
        with torch.cuda.device(q.device):
            status=(self.tma_fn if tma else self.fn)(ct.byref(params),torch.cuda.current_stream(q.device).cuda_stream,2 if split_pv else int(overlap))
        if status: raise RuntimeError(f'value_direction_sm90 CUDA error {status}')
        return Output(output,skipped,eligible,normalizer,state,risks)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fast-sfu',action='store_true')
    parser.add_argument('--inline-roles',action='store_true')
    parser.add_argument('--online-ratio',action='store_true')
    args=parser.parse_args()
    build(fast_sfu=args.fast_sfu,inline_roles=args.inline_roles,online_ratio=args.online_ratio)
