"""Last-GLOBAL forensic observation; no native call is replaced or repeated."""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import re


class Probe:
    def __init__(self,state,destination,modes,raw_calls):
        import torch
        from humaneval_gpu_fingerprint import fingerprint_cuda
        from graph_fingerprint import cpu_fingerprints
        self.torch=torch;self.fingerprint=fingerprint_cuda;self.cpu_fingerprints=cpu_fingerprints
        self.state,self.destination,self.modes,self.raw_calls=state,destination,modes,raw_calls
        self.bindings={}
        state.update(last_active=False,last_samples=0,last_events=[],last_raw=[],last_current=None,last_anchor=None)

    def fp(self,value):
        if value is None:return None
        if not value.numel():return dict(shape=list(value.shape),dtype=str(value.dtype),empty=True)
        flat=value.detach().reshape(-1);padding=(-flat.numel()*flat.element_size())%4
        if padding:flat=self.torch.nn.functional.pad(flat.contiguous().view(self.torch.uint8),(0,padding))
        return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),gpu128=self.fingerprint(flat))

    def tensor(self,value,groups=None):
        if value is None:return None
        pointer=value.untyped_storage().data_ptr()
        group=groups.setdefault(pointer,len(groups)) if groups is not None else None
        return dict(value=value.detach().clone(memory_format=self.torch.contiguous_format),
            shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
            storage_group=group,storage_offset=value.storage_offset(),alignment=value.data_ptr()%256)

    def lists(self,value,raw=False):
        if value is None:return None
        function=self.tensor if raw else self.fp
        return dict(block_size=list(value.block_size),**{key:function(getattr(value,key)) for key in
            ('full_block_cnt','full_block_idx','mask_block_cnt','mask_block_idx')})

    def logical(self,k,v,kwargs,nk):
        table=kwargs.get('page_table')
        if table is None:return k.permute(0,2,1,3),v.permute(0,2,1,3)
        page=k.shape[1];ids=table[0,:(nk+page-1)//page].long()
        def gather(x):return x[ids].reshape(-1,x.shape[2],x.shape[3])[:nk].transpose(0,1)[None]
        return gather(k),gather(v)

    def capture_fwd(self,q,k,v,kwargs,nk):
        groups={};result=dict(q=self.tensor(q,groups),scalar_kwargs={},tensor_kwargs={},aux=[])
        table=kwargs.get('page_table')
        if table is None:
            result.update(layout='contiguous',k=self.tensor(k,groups),v=self.tensor(v,groups))
        else:
            page=k.shape[1];ids=table[:,:((nk+page-1)//page)].flatten().long()
            # Metadata preserves the full source geometry and K/V alias relation.
            # Payload stores only referenced physical pages, including last-page padding.
            def metadata(x):
                pointer=x.untyped_storage().data_ptr();group=groups.setdefault(pointer,len(groups))
                return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
                    storage_group=group,storage_offset=x.storage_offset(),alignment=x.data_ptr()%256)
            result.update(layout='paged',k_metadata=metadata(k),v_metadata=metadata(v),
                page_ids=self.tensor(ids),k_pages=self.tensor(k[ids]),v_pages=self.tensor(v[ids]))
        for key,value in kwargs.items():
            if key=='aux_tensors':
                # aux[0] is the observer's write-only log-mass output.
                for i,x in enumerate(value or []):
                    result['aux'].append(dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
                        input=None if i==0 else self.tensor(x)))
            elif key=='block_sparse_tensors':result[key]=self.lists(value,True)
            elif isinstance(value,self.torch.Tensor):result['tensor_kwargs'][key]=self.tensor(value)
            elif value is None or isinstance(value,(str,int,float,bool,tuple,list)):result['scalar_kwargs'][key]=value
            else:raise TypeError('unsupported actual FA4 argument '+key)
        return result

    def wrap_fwd(self,original,label):
        @functools.wraps(original)
        def call(q,k,v,*args,**kwargs):
            event=self.state['last_current']
            if event is None:return original(q,k,v,*args,**kwargs)
            if args:raise RuntimeError('actual FA4 arguments must have named optional fields')
            nk=event['context']['seq_len'];table=kwargs.get('page_table')
            pages=None
            if table is not None:
                ids=table[0,:(nk+k.shape[1]-1)//k.shape[1]].long()
                pk,pv=k[ids],v[ids];pages=dict(k=self.fp(pk),v=self.fp(pv))
                logical_k=pk.reshape(-1,k.shape[2],k.shape[3])[:nk].transpose(0,1)[None]
                logical_v=pv.reshape(-1,v.shape[2],v.shape[3])[:nk].transpose(0,1)[None]
            else:logical_k,logical_v=self.logical(k,v,kwargs,nk)
            record=dict(kind=label,q=self.fp(q),logical_k=self.fp(logical_k),logical_v=self.fp(logical_v),
                lists=self.lists(kwargs.get('block_sparse_tensors')),
                page_table=self.fp(kwargs.get('page_table')),seqused_k=self.fp(kwargs.get('seqused_k')),
                num_splits=kwargs.get('num_splits',0),paged_pages_including_padding=pages,
                scalar_kwargs={key:value for key,value in kwargs.items() if value is None or isinstance(value,(str,int,float,bool))},
                other_tensor_inputs={key:self.fp(value) for key,value in kwargs.items() if
                    isinstance(value,self.torch.Tensor) and key not in ('page_table','seqused_k')},
                aux_inputs=[None if i==0 else self.fp(value) for i,value in enumerate(kwargs.get('aux_tensors') or [])])
            capture=event['sampler_index'] in self.raw_calls.get(event['arm'],[])
            raw=self.capture_fwd(q,k,v,kwargs,event['context']['seq_len']) if capture else None
            result=original(q,k,v,**kwargs)
            record.update(output=self.fp(result[0]),lse=self.fp(result[1]) if len(result)>1 else None)
            if kwargs.get('aux_tensors'):record['observed_mass']=self.fp(kwargs['aux_tensors'][0])
            if raw is not None:
                raw.update(kind='fa4',consumer=label,sampler_index=event['sampler_index'],arm=event['arm'],
                    output=self.tensor(result[0]),lse=self.tensor(result[1]) if len(result)>1 else None,
                    aux_after=[self.tensor(x) for x in kwargs.get('aux_tensors') or []])
                self.state['last_raw'].append(raw)
            event.setdefault('consumers',[]).append(record)
            return result
        return call

    def wrap_moments(self,original):
        self.bindings['moments']=self.binding(original)
        @functools.wraps(original)
        def call(q,k,v,*args,**kwargs):
            anchor=self.state['last_anchor']
            if anchor is None:return original(q,k,v,*args,**kwargs)
            event=dict(kind='moments',ordinal=len(anchor['moments']),q=self.fp(q),k=self.fp(k),v=self.fp(v))
            result=original(q,k,v,*args,**kwargs)
            event['outputs']={key:self.fp(getattr(result,key)) for key in
                ('mass','numerator','tail_mass','tail_numerator','log_scale')}
            anchor['moments'].append(event)
            return result
        return call

    @staticmethod
    def binding(function):
        original=inspect.unwrap(function)
        try:path=inspect.getsourcefile(original)
        except TypeError:path=None
        path=Path(path) if path else None
        return dict(module=getattr(original,'__module__',None),name=getattr(original,'__qualname__',type(original).__name__),
            source_file=path.name if path else None,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None)

    def install(self):
        torch=self.torch;state=self.state
        import vllm.model_executor.models.diffusion_gemma as dg
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from experiments.numerical_qk_reuse import vllm_adapter,v31_value_fast_runtime as fast
        from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
        add,step,sample=LLMEngine.add_request,LLMEngine.step,dg._compiled_sample_step
        install_adapter=vllm_adapter.install_vllm_patches
        initialize,apply,merge=fast.FastRuntime.initialize,ResidualDirectionCache.apply_cuda,vllm_adapter.VllmMethodAdapter._merge_alias2
        fast.moments_full=self.wrap_moments(fast.moments_full)

        @functools.wraps(add)
        def begin(engine,*args,**kwargs):
            result=add(engine,*args,**kwargs)
            mode=self.modes[state['index']]
            if mode not in (None,'full_swap8','residual_direction_guard'):raise RuntimeError('unknown last-GLOBAL probe arm')
            state.update(last_active=mode is not None,last_arm=mode,last_samples=0,last_events=[],last_raw=[],last_current=None,last_anchor=None)
            return result

        @functools.wraps(sample)
        def sampler(*args,**kwargs):
            result=sample(*args,**kwargs)
            if state['last_active']:state['last_samples']+=1
            return result

        @functools.wraps(initialize)
        def anchor(runtime,adapter,q,b,scale,prefix,n,out,identity):
            event=state['last_current']
            if event is None:return initialize(runtime,adapter,q,b,scale,prefix,n,out,identity)
            if runtime.mode!=event['arm']:raise RuntimeError('actual full-value runtime mode differs from probe schedule')
            entry=dict(kind='anchor',sampler_index=event['sampler_index'],identity=identity,moments=[],
                q=self.fp(q),k=self.fp(b['k']),v=self.fp(b['v']),dense_native=self.fp(out),mask_before=self.fp(adapter.mage_state[29]['kept']))
            raw=dict(kind='anchor_inputs',sampler_index=event['sampler_index'],identity=identity,arm=event['arm'],scale=float(scale),prefix=prefix,
                q=self.tensor(q),k=self.tensor(b['k']),v=self.tensor(b['v']),dense_native=self.tensor(out),mask_before=self.tensor(adapter.mage_state[29]['kept']))
            state['last_anchor']=entry
            try:result=initialize(runtime,adapter,q,b,scale,prefix,n,out,identity)
            finally:state['last_anchor']=None
            entry['mask_after']=self.fp(adapter.mage_state[29]['kept']);raw['mask_after']=self.tensor(adapter.mage_state[29]['kept'])
            cache=(runtime.state or {}).get('direction_cache')
            if cache is not None:
                entry['cache']={key:self.fp(getattr(cache,key)) for key in ('anchor_selected','residual','dual','residual_squared')}
                raw['cache']={key:self.tensor(getattr(cache,key)) for key in ('anchor_selected','residual','dual','residual_squared')}
            if not any(x['kind']=='weights' and x['arm']==event['arm'] for x in state['last_raw']):
                weights=dict(kind='weights',arm=event['arm'])
                if getattr(runtime,'grams',None):weights['grams']={key:self.tensor(value) for key,value in runtime.grams.items()}
                if hasattr(runtime,'projection'):weights['projection']=self.tensor(runtime.projection)
                state['last_raw'].append(weights)
            state['last_raw'].append(raw);event.setdefault('anchors',[]).append(entry)
            return result

        @functools.wraps(apply)
        def correction(cache,current,identity):
            event=state['last_current']
            if event is None:return apply(cache,current,identity)
            entry=dict(current=self.fp(current),cache={key:self.fp(getattr(cache,key)) for key in
                ('anchor_selected','residual','dual','residual_squared')})
            result,alpha=apply(cache,current,identity)
            entry.update(output=self.fp(result),alpha=self.fp(alpha));event['correction']=entry
            if event['sampler_index'] in self.raw_calls.get(event['arm'],[]):
                state['last_raw'].append(dict(kind='correction',sampler_index=event['sampler_index'],arm=event['arm'],identity=identity,
                    current=self.tensor(current),cache={key:self.tensor(getattr(cache,key)) for key in
                    ('anchor_selected','residual','dual','residual_squared')},output=self.tensor(result),alpha=self.tensor(alpha)))
            return result,alpha

        @functools.wraps(merge)
        def merging(adapter,partials,lse):
            event=state['last_current']
            if event is None:return merge(adapter,partials,lse)
            entry=dict(partials=self.fp(partials),lse=self.fp(lse))
            result=merge(adapter,partials,lse)
            cached=adapter._merge_cache[(partials.shape[2],partials.shape[1],str(partials.device))]
            entry.update(order=self.fp(cached.order),output=self.fp(result));event['merge']=entry
            if event['sampler_index'] in self.raw_calls.get(event['arm'],[]):
                state['last_raw'].append(dict(kind='merge',sampler_index=event['sampler_index'],arm=event['arm'],partials=self.tensor(partials),
                    lse=self.tensor(lse),order=self.tensor(cached.order),output=self.tensor(result)))
            return result

        def installed(adapter):
            result=install_adapter(adapter)
            # Native mask fixes are installed by the base worker before this point.
            from experiments.numerical_qk_reuse import v27_fa4,v31_fa4_observe
            v27_fa4.load()
            self.bindings['paged_sparse_fwd']=self.binding(v27_fa4._FWD)
            self.bindings['observing_dense_fwd']=self.binding(v31_fa4_observe._flash_attn_fwd)
            v27_fa4._FWD=self.wrap_fwd(v27_fa4._FWD,'paged_sparse_fwd')
            v31_fa4_observe._flash_attn_fwd=self.wrap_fwd(v31_fa4_observe._flash_attn_fwd,'observing_dense_fwd')
            forward=FlashAttentionImpl.forward
            @functools.wraps(forward)
            def attention(impl,layer,query,key,value,kv_cache,metadata,output,*args,**kwargs):
                name=getattr(layer,'layer_name','')
                if not state['last_active'] or metadata is None or not re.search(r'\.layers\.29\.',name):
                    return forward(impl,layer,query,key,value,kv_cache,metadata,output,*args,**kwargs)
                if torch.cuda.is_current_stream_capturing():raise RuntimeError('last-GLOBAL observer captured')
                context=dict(adapter.step_ctx or {})
                if context.get('encoder',True):
                    return forward(impl,layer,query,key,value,kv_cache,metadata,output,*args,**kwargs)
                event=dict(kind='attention',sampler_index=state['last_samples'],context=context,canvas=adapter.canvas_id,
                    arm=state['last_arm'],
                    query=self.fp(query[:metadata.num_actual_tokens]),key=self.fp(key[:metadata.num_actual_tokens]) if key is not None else None,
                    value=self.fp(value[:metadata.num_actual_tokens]) if value is not None else None)
                state['last_current']=event
                try:result=forward(impl,layer,query,key,value,kv_cache,metadata,output,*args,**kwargs)
                finally:state['last_current']=None
                event['output']=self.fp(output[:metadata.num_actual_tokens])
                b=adapter.buffers.get(29);st=adapter.mage_state.get(29)
                if b is not None:event['contiguous_cache']={key:self.fp(b[key]) for key in ('k','v')}
                if st is not None:event['kept']=self.fp(st.get('kept'));event['held_lists']=self.lists(st.get('lists'))
                state['last_events'].append(event)
                return result
            FlashAttentionImpl.forward=attention
            return result

        def raw_cpu(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().contiguous()
            if isinstance(value,dict):return {key:raw_cpu(v) for key,v in value.items()}
            if isinstance(value,list):return [raw_cpu(v) for v in value]
            return value

        @functools.wraps(step)
        def finish(engine,*args,**kwargs):
            active,index=state['last_active'],state['index']
            result=step(engine,*args,**kwargs)
            if active and any(o.finished for o in result):
                self.destination.mkdir(parents=True,exist_ok=True)
                with (self.destination/f'request{index:03d}.json').open('x') as f:
                    json.dump(self.cpu_fingerprints(dict(events=state['last_events'],bindings=self.bindings,
                        extra_forwards=0,layer=29,clean_timing=False)),f,sort_keys=True)
                path=self.destination/f'request{index:03d}.private.pt';assert not path.exists()
                torch.save(raw_cpu(state['last_raw']),path)
                state.update(last_active=False,last_events=[],last_raw=[],last_current=None)
            return result
        LLMEngine.add_request,LLMEngine.step=begin,finish
        dg._compiled_sample_step=sampler
        fast.FastRuntime.initialize=anchor;ResidualDirectionCache.apply_cuda=correction
        vllm_adapter.VllmMethodAdapter._merge_alias2=merging
        vllm_adapter.install_vllm_patches=installed


def install(state,destination,modes,raw_calls):
    probe=Probe(state,destination,modes,raw_calls)
    probe.install()
    return probe
