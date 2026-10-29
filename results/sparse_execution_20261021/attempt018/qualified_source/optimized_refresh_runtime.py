"""Native integration of the frozen optimized V31 and true-online V2 refreshes.

Yuhan's reuse lifecycle and held/carry FA4 consumer are inherited unchanged.
Junyu's Gaussian32 router is executed online on CURRENT Q/K/V for V2. The
selected experimental SM90 kernel returns the model's refresh output directly.
"""
import hashlib
import os
from pathlib import Path
import types
import torch


def install(adapter, *, method, library, observer_directory, schedule=2):
    from . import v27_fa4
    from .refresh_v31_fast import Observer
    from .v2_online_warpspec import PackedRefresh
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections
    if method not in ('v2','v31'):raise ValueError('named optimized refresh method required')
    if adapter.mage_granularity!='qblock_max' or adapter.mage_sink or adapter.mage_recent_tiles:
        raise ValueError('qualified qblock_max without additional protection required')
    if adapter.cached_refresh_policy is not None or adapter.mage_pool is not None or adapter.mage_row_weight is not None:
        raise ValueError('unqualified reuse variant')
    op=PackedRefresh(Path(library),schedule) if method=='v2' else Observer(observer_directory)
    projections=Projections()
    old_begin,old_receipt=adapter.begin_request,adapter._value_receipt
    old_list=v27_fa4.block_sparse_tensors
    state={'keep':None,'lists':None,'calls':0,'events':[],'budget_errors':[],'units':0,'shapes':set()}
    binary_hash=hashlib.sha256(Path(library).read_bytes()).hexdigest() if method=='v2' else None

    def lists(mask):
        return state['lists'] if mask is state['keep'] else old_list(mask)
    v27_fa4.block_sparse_tensors=lists

    def begin(self,*args,**kw):
        result=old_begin(*args,**kw)
        state.update(keep=None,lists=None,calls=0,events=[],budget_errors=[],units=0,shapes=set())
        return result

    def select(self,layer,q,b,scale,prefix,n):
        h,hk,nk,d=q.shape[1],b['k'].shape[1],b['k'].shape[2],q.shape[-1]
        qb=(n+127)//128;pt=prefix//64
        if d!=512 or nk!=prefix+n:raise ValueError('native GLOBAL geometry drift')
        held=self._mage_held
        events=None
        if self.value_audit:
            events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];events[0].record()
        # Native Q is a transposed view; contiguous conversion is charged here.
        if method=='v2':
            matrix=projections.get(layer,hk,d,'gaussian',32,1729,q.device)
            out,keep,packed=op(q.contiguous(),b['k'].contiguous(),b['v'].contiguous(),matrix,scale,pt,
                self.value_threshold,held=held,sticky=0. if held is None else self.mage_sticky)
            # Physical refresh PV work is sparse as well; no hidden dense output.
            if not self.value_clean_timing:
                kept=keep[...,:pt].sum().double()
                self._sparse_kept=kept if self._sparse_kept is None else self._sparse_kept+kept
                self._sparse_total+=h*qb*pt
                self._kept_prefix=kept if self._kept_prefix is None else self._kept_prefix+kept
                self._prefix_total+=h*qb*pt
            self.calls['value_selection_calls']=self.calls.get('value_selection_calls',0)+1
        else:
            budget=self._k_override if self._k_override is not None else self.mage_k
            paged=getattr(self,'_resident_v31_paged',None)
            if os.environ.get('RESIDENT_SPARSE_EXECUTION')=='1':
                op.warmup_selectors(q,b['k'],scale,int(os.environ['MAX_MODEL_LEN']),budget,self.mage_sticky or 0.,paged=paged)
            if paged is None:
                out,keep,packed=op.refresh(q,b['k'],b['v'],scale,prefix,budget,held=held,
                    sticky=0. if held is None else self.mage_sticky,overlap=False,num_splits=1,dynamic_causal=False)
            else:
                key,value,table=paged
                out,keep,packed=op.refresh_paged(q,key,value,table,nk,scale,prefix,budget,held=held,
                    sticky=0. if held is None else self.mage_sticky,num_splits=0,dynamic_causal=True)
            if self.value_audit:
                state['budget_errors'].append((keep[...,:pt].sum(-1)!=min(max(1,budget//64),pt)).sum())
                state['units']+=h*qb
            self.calls['mage_kept_prefix_tiles']+=min(max(1,budget//64),pt)*h*qb
        if events is not None:
            events[1].record()
            previous=self.mage_state.get(layer)
            phase='refresh' if previous and previous['canvas']==self.canvas_id and previous['nk']==nk else 'initial'
            state['events'].append((phase,nk,*events))
        state.update(keep=keep,lists=packed,calls=state['calls']+1)
        state['shapes'].add((n,nk))
        self.calls['mage_prefix_tiles']+=pt*h*qb
        if held is not None:self.calls['mage_sticky_units']=self.calls.get('mage_sticky_units',0)+1
        return out,keep

    def receipt(self):
        result=old_receipt()
        elapsed=None
        if state['events']:
            torch.cuda.synchronize();elapsed=sum(a.elapsed_time(b) for _,_,a,b in state['events'])
        by_phase={}
        for phase,nk,a,b in state['events']:
            row=by_phase.setdefault(phase,dict(calls=0,gpu_ms=0.))
            row['calls']+=1;row['gpu_ms']+=a.elapsed_time(b)
        result.update(optimized_refresh_phase_timing=by_phase,optimized_refresh_method=method,optimized_refresh_calls=state['calls'],
            optimized_refresh_kernel=('q128_fold_bf16x3_fastmath_sm90' if method=='v2' else 'fa4_exp_scores_fused_qblock_topk'),
            optimized_refresh_library_sha256=binary_hash,optimized_refresh_schedule=schedule if method=='v2' else None,
            optimized_refresh_observer_sha256=op.source_sha256 if method=='v31' else None,
            optimized_refresh_gpu_ms=elapsed,optimized_refresh_shapes=sorted(state['shapes']),
            optimized_refresh_budget_errors=(int(torch.stack(state['budget_errors']).sum()) if state['budget_errors'] else 0) if self.value_audit else None,
            optimized_refresh_budget_units=state['units'] if self.value_audit else None,
            optimized_refresh_extra_model_forwards=0,
            optimized_refresh_projection_current_v=method=='v2',
            optimized_refresh_held_consumer='inherited_paged_split_fa4',
            optimized_refresh_threshold=self.value_threshold if method=='v2' else None,
            value_output_path='online_sparse_refresh_current_qkv' if method=='v2' else 'dense_fa4_optimized_observation',
            value_state_role='online_router_and_sparse_refresh_output' if method=='v2' else 'selector_only',
            value_discovery_work='full QK and Gaussian32 PZ; conditional full PV; four channel groups repeat QK/routing' if method=='v2' else 'FA4 score observation plus fused topk/list',
            value_projection_manifest=projections.manifest if method=='v2' else {},
            value_budget_policy='online_variable_threshold' if method=='v2' else 'fixed_qblock_prefix_budget')
        return result

    adapter.begin_request=types.MethodType(begin,adapter)
    adapter._select_initial_refresh=types.MethodType(select,adapter)
    adapter._value_receipt=types.MethodType(receipt,adapter)
    return state
