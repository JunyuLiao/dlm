"""Native DiffusionGemma binding with request-local, producer-owned sketch leases.

No alternate decoding implementation. The historical router's structural masks
are retained explicitly, including its per-query local-window convention.
"""
from contextlib import contextmanager
import math

import torch
from dllm.attention.blasst.core import _attention_type,_attention_validity
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from .cuda import Kernel
from .masks import geometry,pack


class Sketches:
    """Never infer unchanged inference tensors from a data pointer alone.

    An encoder-entry hook invalidates every lease before any cache mutation.
    Decoder-entry hooks identify the actual prefix object; the inspected native
    decoder only reads it. Different shape/object/version also invalidates.
    Boundary and canvas tokens always refresh. Lifetime is one request; external
    mutation/concurrent requests on a bound model are not supported.
    """
    def __init__(self,adapter,projections=None,*,fused=False):
        self.projections=projections or Projections();self.entries={};self.sources={};self.handles=[]
        self.fused=fused
        self.epoch=0;self.projected_tokens=0;self.reused_tokens=0
        self.peak_storage_bytes=0;self.projection_madds=0
        for name,module in adapter.model.named_modules():
            if type(module).__name__=='DiffusionGemmaEncoderModel':
                self.handles.append(module.register_forward_pre_hook(self.invalidate))
            if adapter.is_blasst_attention_module(name,module):
                self.handles.append(module.register_forward_pre_hook(self.identify,with_kwargs=True))
        if not self.handles:raise ValueError('No native DiffusionGemma hooks installed')

    def invalidate(self,*_):
        self.epoch+=1;self.entries.clear();self.sources.clear()

    def identify(self,module,args,kwargs):
        cache=kwargs.get('past_key_values',args[3] if len(args)>3 else None)
        source=None if cache is None else cache.layers[int(module.layer_idx)].values
        self.sources[int(module.layer_idx)]=source

    @staticmethod
    def version(source):
        if source is None or torch.is_inference(source):return None
        return source._version

    def get(self,layer,value,valid,prefix):
        b,h,n,d=value.shape;end=max(0,prefix)//64*64
        source=self.sources.get(layer);old=self.entries.get(layer)
        identity=(self.epoch,self.version(source),tuple(value.shape),end)
        reuse=end>0 and source is not None and old is not None and old['source'] is source and old['identity']==identity
        matrix=self.projections.get(layer,h,d,'gaussian',32,1729,value.device)
        if not reuse:
            old=dict(source=source,identity=identity,z=torch.empty((b,h,n,32),device=value.device,dtype=torch.float32),norm=torch.empty((b,h,n),device=value.device,dtype=torch.float32))
            self.entries[layer]=old
        start=end if reuse else 0
        if self.fused:
            from .projection import refresh
            ref=refresh(value,matrix,old['z'],old['norm'],valid,start)
        else:
            x=value[...,start:,:].float()
            torch.matmul(x,matrix,out=old['z'][...,start:,:])
            old['norm'][...,start:]=x.norm(dim=-1).square()
            ref=(old['norm'].masked_fill(~valid,0.).sum(-1)/valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
        self.projected_tokens+=b*h*(n-start);self.reused_tokens+=b*h*start
        self.projection_madds+=b*h*(n-start)*d*32
        self.peak_storage_bytes=max(self.peak_storage_bytes,sum(e['z'].numel()*4+e['norm'].numel()*4 for e in self.entries.values()))
        return old['z'],ref

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.entries.clear();self.sources.clear()


class Attention:
    def __init__(self,adapter,library,thresholds,*,precision='tf32x3_register',collect=True,tma=True,mode='value',projections=None,profile=False,projection='fused',torch_library=None,blasst_tma=False):
        if mode not in ('value','blasst'):raise ValueError('Unsupported production routing mode')
        if projection not in ('fused','torch'):raise ValueError('Unsupported projection backend')
        self.kernel=Kernel(library,torch_library=torch_library);self.thresholds=thresholds;self.precision=precision
        self.cache=Sketches(adapter,projections,fused=projection=='fused');self.geometry={};self.pending=[];self.collect=collect;self.calls=0
        self.tma=tma;self.blasst_tma=blasst_tma;self.mode=mode;self.dummy={};self.lambda_records=[];self.profile=profile;self.events=[]
        self.policy_selector=None
        self.query_sensitivity=None

    @torch.no_grad()
    def __call__(self,module,q,k,v,mask,*,dropout=0.,scaling=None,is_causal=None,sliding_window=None,**kwargs):
        if dropout or module.training:raise ValueError('Inference/eval only')
        b,h,nq,d=q.shape;nk=k.shape[-2];hk=k.shape[1]
        if self.profile:
            begin,ready,end=(torch.cuda.Event(enable_timing=True) for _ in range(3));begin.record()
        causal=bool(is_causal) if is_causal is not None else mask is None and nq>1
        if mask is None:
            key=(b,nq,nk,q.device,causal,sliding_window)
            if key not in self.geometry:
                self.geometry[key]=geometry(b,nq,nk,device=q.device,causal=causal,window=sliding_window or 0)
            packed,validkv=self.geometry[key]
            validkv=validkv[None,None,:].expand(b,hk,nk)
            length=nk-max(0,nk-nq-int(sliding_window)+1 if sliding_window else 0)
        else:
            valid=_attention_validity(mask,q,k,is_causal=causal,sliding_window=sliding_window)
            validkv=valid.reshape(b,hk,h//hk,nq,nk).any((2,3))
            # The frozen BLASST rule uses the union of valid keys across heads
            # and queries. Arbitrary masks need an explicit reduction; native
            # DiffusionGemma's compact geometry uses the exact analytic length.
            length=int(valid.any((1,2)).sum(-1).item()) if self.mode=='blasst' else nk
            if mask.dtype==torch.bool:packed=pack(valid.contiguous())
            else:
                packed=mask.expand(b,h,nq,nk).contiguous().masked_fill(~valid,-torch.inf)
                if packed.dtype!=torch.bfloat16:raise ValueError('Non-BF16 additive masks require an explicit supported specialization')
        layer=int(module.layer_idx);kind=_attention_type(module,sliding_window)
        policy=self.thresholds[kind]
        if self.policy_selector is not None:
            selected=self.policy_selector(int(module._blasst_2d_runtime.current_denoising_iteration),kind)
            if selected is None:
                selected={'log_threshold': -float('inf')} if self.mode=='value' else {'log_scale': -float('inf'),'cap_one':False}
            policy=selected
        if self.mode=='value':
            z,ref=self.cache.get(layer,v,validkv,max(0,nk-nq))
            threshold=policy['log_threshold']
        else:
            key=(b,hk,nk,q.device)
            if key not in self.dummy:
                self.dummy[key]=(torch.empty((b,hk,nk,32),device=q.device,dtype=torch.float32),torch.ones((b,hk),device=q.device))
            z,ref=self.dummy[key]
            threshold=policy['log_scale']-math.log(max(1,length)) if 'log_scale' in policy else policy['log_threshold']
            if policy.get('cap_one'):threshold=min(0.,threshold)
            if policy.get('unattainable') and policy.get('cap_one'):threshold=0.
            self.lambda_records.append(dict(layer=layer,attention_type=kind,valid_kv_length=length,log_lambda=threshold))
        if self.profile:ready.record()
        result=self.kernel(q.contiguous(),k.contiguous(),v.contiguous(),z,ref,mask=packed,
            scale=scaling,log_threshold=threshold,precision=self.precision,mode=self.mode,
            tma=self.tma and (self.mode=='value' or self.blasst_tma) and mask is None,
            sensitivity=self.query_sensitivity if self.mode=='value' else None)
        if self.profile:end.record();self.events.append((layer,kind,begin,ready,end))
        self.calls+=1
        if self.collect:
            self.pending.append((layer,int(module._blasst_2d_runtime.current_denoising_iteration),kind,nk-nq,result.skipped,result.eligible))
        return result.output.transpose(1,2).contiguous(),None

    def profile_records(self):
        # Call only after generation has synchronized. CUDA-event instrumentation
        # is a separate profiling run, never the end-to-end latency run.
        return [dict(layer=l,attention_type=k,preparation_ms=a.elapsed_time(b),kernel_and_launch_ms=b.elapsed_time(c))
                for l,k,a,b,c in self.events]

    def records(self):
        rows=[]
        # A single device->host transfer after generation; no per-tile host votes.
        arrays=[x for item in self.pending for x in item[-2:]]
        flat=torch.cat([x.flatten() for x in arrays]).cpu() if arrays else torch.empty(0,dtype=torch.bool)
        offset=0
        for layer,step,kind,prefix,skip,eligible in self.pending:
            n=skip.numel();ss=flat[offset:offset+n].view(skip.shape);ee=flat[offset+n:offset+2*n].view(eligible.shape);offset+=2*n
            if (ss&~ee).any():raise AssertionError('Skipped structurally ineligible tile')
            starts=torch.arange(skip.shape[-1])*64
            for head in range(skip.shape[1]):
                row=dict(layer=layer,head=head,step=step,attention_type=kind,eligible=int(ee[:,head].sum()),skipped=int(ss[:,head].sum()))
                for region,selector in [('prefix',starts+64<=prefix),('canvas',starts>=prefix),('boundary',(starts<prefix)&(starts+64>prefix))]:
                    row[region+'_eligible']=int((ee[:,head]&selector).sum());row[region+'_skipped']=int((ss[:,head]&selector).sum())
                rows.append(row)
        return rows


@contextmanager
def install(adapter,library,thresholds,**kwargs):
    if getattr(adapter.model,'_value_direction_lease',False):raise RuntimeError('Concurrent binding on one model is unsupported')
    adapter.model._value_direction_lease=True
    binding=None;router=None
    try:
        binding=_install_dense(adapter)
        router=Attention(adapter,library,thresholds,**kwargs)
        binding.runtime.attention_override=router
        yield binding,router
    finally:
        if router is not None:router.cache.close()
        if binding is not None:binding.close()
        del adapter.model._value_direction_lease
