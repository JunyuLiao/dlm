"""Named native held/carry execution path with resident direct paged K/V.

The inherited selection/settledness/carry lifecycle is preserved. Only ordinary
held attention bypasses the historical per-call KV-copy/page/list preparation.
No selector, compact-metadata construction or prefix/canvas copy on held calls.
"""
import types
import math
import os
import torch
from .resident_sparse import PagedContext,ResidentMask,GraphPagedBucket

def install_allkept(adapter):
    """Full physical support using the selected resident held consumer."""
    if adapter.arm != 'allkept':
        raise ValueError('explicit all-kept control required')
    original_begin, original_receipt = adapter.begin_request, adapter._value_receipt
    state = dict(contexts={}, calls=0, events=[], metadata_builds=0)
    def begin(self,*args,**kwargs):
        result=original_begin(*args,**kwargs)
        state.update(contexts={},calls=0,events=[],metadata_builds=0)
        return result
    def forward(self,impl,layer,query,kv_cache,metadata,output):
        n=int(metadata.num_actual_tokens);nk=self.step_ctx['seq_len'];prefix=nk-n
        key,value=kv_cache.transpose(1,2).split(impl.head_size,dim=-1)
        tag=(self.canvas_id,nk,n,key.data_ptr(),metadata.block_table.data_ptr())
        previous=state['contexts'].get(layer)
        if previous is None or previous[0]!=tag:
            splits=1 if nk<=4096 else 2
            context=PagedContext(key,value,metadata.block_table[0],nk,splits=splits)
            keep=torch.ones((1,query.shape[1],(n+127)//128,(nk+63)//64),device=query.device,dtype=torch.bool)
            mask=ResidentMask.create(keep,splits=splits,prefix_tiles=prefix//64,audit=False)
            previous=(tag,context,mask);state['contexts'][layer]=previous;state['metadata_builds']+=1
        h=query.shape[1];units=h*((n+127)//128);eligible=units*((nk+63)//64)
        self.calls['global_calls']+=1;self._cur_layer=layer
        self._global_tiles+=units*(prefix//64)
        self._global_canvas_tiles+=eligible-units*(prefix//64)
        events=None
        if self.value_audit:
            events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];events[0].record()
        q=query[:n].transpose(0,1).unsqueeze(0)
        out=previous[1](q,previous[2],float(impl.scale),out=output[:n].view(1,n,h,query.shape[-1]))
        if events is not None:
            events[1].record();state['events'].append(events)
            kept=torch.tensor(eligible,device=query.device,dtype=torch.float64)
            self._value_global_step_phases.setdefault((self.canvas_id,self._canvas_step),set()).add('dense')
            for mapping,k in [(self._value_phases,'dense'),(self._value_canvas_phases,(self.canvas_id,'dense'))]:
                row=mapping.setdefault(k,[0,0,[]]);row[0]+=1;row[1]+=eligible;row[2].append(kept)
            self._value_nonfinite_outputs.append(~torch.isfinite(out).all())
        state['calls']+=1
        return output
    def receipt(self):
        result=original_receipt()
        if state['events']:torch.cuda.synchronize()
        result.update(resident_allkept_execution=True,resident_allkept_calls=state['calls'],
            resident_allkept_metadata_builds=state['metadata_builds'],resident_allkept_kv_copy_calls=0,
            resident_allkept_gpu_ms=sum(a.elapsed_time(b) for a,b in state['events']) if state['events'] else None,
            resident_allkept_consumer='native_paged_fa4_direct_output_persistent_metadata',
            resident_allkept_extra_forwards=0)
        return result
    adapter.begin_request=types.MethodType(begin,adapter)
    adapter.forward=types.MethodType(forward,adapter)
    adapter._value_receipt=types.MethodType(receipt,adapter)
    return state

def install(adapter,*,splits=None,graph_buckets=False,v31_paged_refresh=False):
    from . import v27_fa4
    if adapter.arm!='mage' or adapter.cached_refresh_policy is not None or adapter.stable_refresh_config is not None:
        raise ValueError('qualified inherited mage lifecycle required')
    if adapter.mage_pool is not None or adapter.mage_row_weight is not None or adapter.drop_guard is not None or adapter.residual is not None:
        raise ValueError('unqualified extra routing/output variant')
    original_forward=adapter.forward;original_select=adapter._select_initial_refresh
    original_buffers=adapter._buffers
    original_begin=adapter.begin_request;original_receipt=adapter._value_receipt
    state={'contexts':{},'masks':{},'held_calls':0,'carried_calls':0,'metadata_builds':0,'events':[],'graphs':{},'request_index':-1,'warm_captures':0,'jit_compiles':[],'paged_refresh_calls':0,'jit_events':[]}
    from triton.runtime.jit import JITFunction
    original_compile=JITFunction._do_compile
    def compile_record(function,*args,**kwargs):
        if adapter.bound:state['jit_compiles'].append(function.__name__)
        return original_compile(function,*args,**kwargs)
    JITFunction._do_compile=compile_record
    state['compile_hook_installed']=True
    from vllm.utils import jit_monitor
    original_jit_event=jit_monitor._handle_jit_event
    def jit_event(**event):
        if adapter.bound:state['jit_events'].append(event)
        return original_jit_event(**event)
    jit_monitor._handle_jit_event=jit_event
    def split_bucket(nk):return splits if splits is not None else (1 if nk<=4096 else 2)
    def graph_key(layer,key,n,h,bucket):return (layer,key.data_ptr(),n,h,bucket)

    def begin(self,*args,**kwargs):
        result=original_begin(*args,**kwargs)
        state.update(contexts={},masks={},held_calls=0,carried_calls=0,metadata_builds=0,events=[],jit_compiles=[],paged_refresh_calls=0,jit_events=[])
        state['request_index']+=1
        return result

    def select(self,layer,q,b,scale,prefix,n):
        out,keep=original_select(layer,q,b,scale,prefix,n)
        lists=v27_fa4.block_sparse_tensors(keep)
        if split_bucket(prefix+n)==1:
            resident=ResidentMask.from_lists(keep,lists,prefix_tiles=prefix//64,audit=not self.value_clean_timing)
        else:resident=ResidentMask.create(keep,splits=split_bucket(prefix+n),prefix_tiles=prefix//64,audit=not self.value_clean_timing)
        state['masks'][id(lists)]=(lists,resident)
        state['metadata_builds']+=1
        if graph_buckets:
            context=state['contexts'][layer]
            context[1].prepare(resident,context[2],prefix+n)
            context[1].prepared_mask=resident
        return out,keep

    def forward(self,impl,layer,query,kv_cache,metadata,output):
        ctx=self.step_ctx;n=int(metadata.num_actual_tokens);nk=ctx['seq_len'];prefix=nk-n
        if n!=ctx['n']:raise ValueError('native query geometry drift')
        key,value=kv_cache.transpose(1,2).split(impl.head_size,dim=-1)
        current_splits=split_bucket(nk)
        bucket=2**max(10,math.ceil(math.log2(nk)))
        if graph_buckets and graph_key(layer,key,n,query.shape[1],bucket) not in state['graphs']:
            # The explicit warm request is first. Prime all length buckets for
            # this native query/cache geometry there; timed requests may only
            # reuse these graphs. A miss fails instead of capturing silently.
            maximum=int(os.environ.get('MAX_MODEL_LEN','131072'))
            for power in range(10,max(10,math.ceil(math.log2(maximum)))+1):
                size=2**power;gkey=graph_key(layer,key,n,query.shape[1],size)
                if gkey not in state['graphs']:
                    if state['request_index']!=0:raise RuntimeError('unwarmed sparse execution graph bucket')
                    state['graphs'][gkey]=GraphPagedBucket(key,value,metadata.block_table[0],size,n,query.shape[1],float(impl.scale),splits=split_bucket(size))
                    state['warm_captures']+=1
        tag=(self.canvas_id,nk,n,key.data_ptr(),key.stride(),metadata.block_table.data_ptr())
        resident_context=state['contexts'].get(layer)
        built_context=resident_context is None or resident_context[0]!=tag
        if built_context:
            consumer=state['graphs'][graph_key(layer,key,n,query.shape[1],bucket)] if graph_buckets else PagedContext(key,value,metadata.block_table[0],nk,splits=current_splits)
            resident_context=(tag,consumer,metadata.block_table[0])
            state['contexts'][layer]=resident_context
            state['metadata_builds']+=1
        st=self.mage_state.get(layer);current=st is not None and st['canvas']==self.canvas_id and st['nk']==nk
        count=self._mage_count.get(layer)
        index=0 if count is None or count[:2]!=[self.canvas_id,nk] else count[2]+1
        due=(index in self.mage_reselect if self.mage_reselect is not None else self._trig_at==(self.canvas_id,index))
        if current and not due:
            if built_context:raise RuntimeError('page metadata changed on an ordinary held call')
            lists=st['lists'];entry=state['masks'].get(id(lists))
            if entry is None or entry[0] is not lists:raise RuntimeError('held resident metadata absent at refresh')
            resident=entry[1];phase='held';self.calls['mage_reused_calls']+=1
        else:
            # Carry maps are prepared once at a canvas transition, never an
            # ordinary held step. Other calls retain the exact original path.
            warm=self._mage_warm.get(layer)
            carry_due=not current and self.mage_select_step==1 and self.mage_carry_first and (warm is None or warm[:2]!=[self.canvas_id,nk] or warm[2]==0)
            lists=self._mage_carried(st,prefix,n) if carry_due else None
            if lists is None:
                # The original lifecycle advances counts/selects exactly once.
                # Only an actual V31 discovery call may bypass contiguous KV.
                warm_count=warm[2] if warm is not None and warm[:2]==[self.canvas_id,nk] else 0
                selection_due=(current and due) or (not current and warm_count>=self.mage_select_step)
                if v31_paged_refresh and selection_due and nk>4096:
                    self._resident_v31_paged=(key,value,metadata.block_table[0])
                    try:result=original_forward(impl,layer,query,kv_cache,metadata,output)
                    finally:self._resident_v31_paged=None
                    state['paged_refresh_calls']+=1
                    return result
                return original_forward(impl,layer,query,kv_cache,metadata,output)
            from .v31_residual import kept_from_lists
            keep=kept_from_lists(lists)
            if keep.ndim==3:keep=keep[None]
            resident=ResidentMask.create(keep.contiguous(),splits=current_splits,prefix_tiles=prefix//64,audit=not self.value_clean_timing)
            state['metadata_builds']+=1;phase='carried';self._mage_warm[layer]=[self.canvas_id,nk,1]
            self.calls['mage_carried_calls']=self.calls.get('mage_carried_calls',0)+1
        self._mage_count[layer]=[self.canvas_id,nk,index]
        self.calls['global_calls']+=1;self._cur_layer=layer
        h=query.shape[1];qb=(n+127)//128;eligible=h*qb*((nk+63)//64);canvas=h*qb*((nk+63)//64-prefix//64)
        self._global_tiles+=h*qb*(prefix//64);self._global_canvas_tiles+=canvas
        events=None
        if self.value_audit:events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];events[0].record()
        q=query[:n].transpose(0,1).unsqueeze(0)
        if graph_buckets:
            if phase=='carried' or built_context:resident_context[1].prepare(resident,metadata.block_table[0],nk)
            # Refresh can change a mask without changing the context. Associate
            # its prepared identity at selection time below as well.
            elif getattr(resident_context[1],'prepared_mask',None) is not resident:
                raise RuntimeError('graph metadata was not prepared at refresh')
            resident_context[1].prepared_mask=resident
            out=resident_context[1](q,out=output[:n].view(1,n,h,query.shape[-1]))
        else:out=resident_context[1](q,resident,float(impl.scale),out=output[:n].view(1,n,h,query.shape[-1]))
        if events is not None:events[1].record();state['events'].append((phase,*events))
        if phase=='held':state['held_calls']+=1
        else:state['carried_calls']+=1
        if not self.value_clean_timing:
            kept=resident.kept_prefix
            self._kept_prefix=kept if self._kept_prefix is None else self._kept_prefix+kept;self._prefix_total+=resident.eligible_prefix
            self._sparse_kept=kept if self._sparse_kept is None else self._sparse_kept+kept;self._sparse_total+=resident.eligible_prefix
        if self.value_audit:
            full_kept=kept+canvas
            self._value_global_step_phases.setdefault((self.canvas_id,self._canvas_step),set()).add(phase)
            for mapping,key_ in [(self._value_phases,phase),(self._value_canvas_phases,(self.canvas_id,phase))]:
                row=mapping.setdefault(key_,[0,0,[]]);row[0]+=1;row[1]+=eligible;row[2].append(full_kept)
            self._value_nonfinite_outputs.append(~torch.isfinite(out).all())
        return output

    def receipt(self):
        result=original_receipt();timing={}
        consumer='native_paged_fa4_warm_bucket_graph_q_stage_output_copy' if graph_buckets else 'native_paged_fa4_direct_output_persistent_metadata'
        if state['events']:torch.cuda.synchronize()
        for phase,a,b in state['events']:
            row=timing.setdefault(phase,dict(calls=0,gpu_ms=0.));row['calls']+=1;row['gpu_ms']+=a.elapsed_time(b)
        result.update(resident_sparse_execution=True,resident_sparse_splits=splits if splits is not None else 'static_1_through_4096_else_2',
            resident_sparse_held_calls=state['held_calls'],resident_sparse_carried_calls=state['carried_calls'],
            resident_sparse_metadata_builds=state['metadata_builds'],resident_sparse_phase_timing=timing,
            resident_sparse_held_kv_copy_calls=0,resident_sparse_held_selector_calls=0,resident_sparse_held_list_builds=0,
            resident_sparse_held_metadata_builds=0,resident_sparse_output_copy_calls=state['held_calls']+state['carried_calls'] if graph_buckets else 0,
            resident_sparse_q_staging_calls=state['held_calls']+state['carried_calls'] if graph_buckets else 0,
            resident_sparse_graph_buckets=graph_buckets,resident_sparse_warm_captures=state['warm_captures'],
            resident_sparse_request_triton_compiles=len(state['jit_compiles']),resident_sparse_request_triton_compile_names=state['jit_compiles'],
            resident_sparse_compile_counter_scope='JITFunction._do_compile while adapter.bound; graph warmup and engine startup excluded',
            resident_sparse_request_jit_events=state['jit_events'],
            resident_sparse_direct_paged_refresh_calls=state['paged_refresh_calls'],
            resident_sparse_consumer=consumer,optimized_refresh_held_consumer=consumer,resident_sparse_extra_forwards=0)
        return result
    def buffers(self,layer,key,value,table,prefix,n):
        if getattr(self,'_resident_v31_paged',None) is not None:
            # Shape/stride-only views satisfy the inherited selector contract;
            # their values are never consumed by paged observation/recovery.
            hk,d=key.shape[2:];nk=prefix+n
            k=key.as_strided((1,hk,nk,d),(0,key.stride(2),0,1))
            v=value.as_strided(k.shape,k.stride())
            return dict(k=k,v=v,prefix=prefix,nk=nk,pk=k[:,:,:prefix],pv=v[:,:,:prefix])
        return original_buffers(layer,key,value,table,prefix,n)
    adapter.begin_request=types.MethodType(begin,adapter);adapter._select_initial_refresh=types.MethodType(select,adapter)
    adapter.forward=types.MethodType(forward,adapter);adapter._value_receipt=types.MethodType(receipt,adapter)
    adapter._buffers=types.MethodType(buffers,adapter)
    return state
