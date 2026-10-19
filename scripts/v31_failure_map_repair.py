"""Audit-only first-held-call output/support factorial; no extra model forward."""
from __future__ import annotations
import os
import runpy
import types

from experiments.numerical_qk_reuse.v31_refresh_diagnostic import (
    Runtime, FACTORIAL, fingerprint, _rng, _same_rng, _summary,
)


class RepairRuntime(Runtime):
    def __init__(self, adapter, list_fn=None):
        super().__init__(adapter, audit=True)
        self.list_fn = list_fn

    def mage(self, layer, q, b, scale, prefix, n):
        import torch
        a=self.adapter
        if self.policy=='fixed16':raise RuntimeError('repair does not change refresh schedule')
        call=self._call(layer,prefix+n);identity=(a.canvas_id,call)
        st=a.mage_state.get(layer)
        same=st is not None and st.get('canvas')==a.canvas_id and st.get('nk')==prefix+n and st.get('prefix')==prefix
        due=a._trig_at==identity
        if self.episode is None and layer==self.scope[0] and same and not due and call>a.mage_select_step and prefix//64>a.mage_k//64:
            self._support(st,prefix,n);self.episode=identity
            if self.before_event_callback is None:raise RuntimeError('repair requires native prefix binding')
            self.prefix_metadata=self.before_event_callback(a,layer,q,b,prefix,n)
        treated=self.episode==identity and layer in self.scope
        if self.episode is not None and not treated and layer in self.scope and layer not in self.next_calls:
            before=dict(a.calls);support=fingerprint(st.get('kept')) if st else None
            out=self.original_mage(layer,q,b,scale,prefix,n)
            self.next_calls[layer]=dict(canvas=identity[0],call=call,support_at_entry_sha256=support,
                held_consumer=a.calls.get('mage_reused_calls',0)>before.get('mage_reused_calls',0),
                refresh_before_consumption=a.calls.get('mage_reselections',0)>before.get('mage_reselections',0))
            return out
        if not treated:return self.original_mage(layer,q,b,scale,prefix,n)
        if not same or due:raise RuntimeError('repair requires held sparse call in every GLOBAL layer')
        old,budget=self._support(st,prefix,n);old=old.clone();old_lists=st['lists']
        rng=_rng(q.device);counts=dict(a.calls)
        sparse=self.original_mage(layer,q,b,scale,prefix,n)
        if a.calls.get('mage_reused_calls',0)!=counts.get('mage_reused_calls',0)+1:
            raise RuntimeError('repair did not execute original held consumer')
        if self.policy=='control':
            if layer==self.scope[0]:
                self.control_anchor=dict(layer=layer,canvas=identity[0],call=call,prefix_budget=budget,
                    old_support_sha256=fingerprint(old),next_support_sha256=fingerprint(old),
                    returned_output_sha256=fingerprint(sparse),original_mage_calls=1,rng_unchanged=_same_rng(rng,q.device))
            return sparse
        counts_after=dict(a.calls)
        saved={name:getattr(a,name,None) for name in ('_mage_units_w','_mage_held','_k_override','_last_pool')}
        a._mage_units_w=None;a._mage_held=old if a.mage_sticky is not None else None;a._k_override=None
        try:
            dense,fresh=a._select_initial_refresh(layer,q,b,scale,prefix,n)
        finally:
            # Extra observation is reported separately, never as native sparse work.
            a.calls.clear();a.calls.update(counts_after)
            for name,value in saved.items():setattr(a,name,value)
        self._support(dict(kept=fresh),prefix,n)
        if self.policy.endswith('_fresh'):
            if self.list_fn is None:
                from experiments.numerical_qk_reuse import v27_fa4
                lists=v27_fa4.block_sparse_tensors(fresh)
            else:lists=self.list_fn(fresh)
            st.update(kept=fresh,lists=lists)
        else:st.update(kept=old,lists=old_lists)
        out=dense if self.policy.startswith('dense_') else sparse
        if dense.shape!=sparse.shape or dense.dtype!=sparse.dtype or not bool(torch.isfinite(dense).all() & torch.isfinite(sparse).all()):
            raise RuntimeError('repair alternatives mismatch or nonfinite')
        if not _same_rng(rng,q.device):raise RuntimeError('repair consumed RNG')
        rectangles=q.shape[1]*((n+127)//128)*((prefix+n+63)//64)
        for field in ('observer_qk_rectangles','observer_pv_rectangles'):
            self.extra_work[field]=self.extra_work.get(field,0)+rectangles
        self.extra_work['dense_observer_calls']=self.extra_work.get('dense_observer_calls',0)+1
        self.events.append(dict(layer=layer,canvas=identity[0],call=call,prefix=prefix,query_rows=n,prefix_budget=budget,
            old_support_sha256=fingerprint(old),fresh_support_sha256=fingerprint(fresh),next_support_sha256=fingerprint(st['kept']),
            changed_prefix_blocks=int((old^fresh)[...,:prefix//64].sum()),current_output='dense' if self.policy.startswith('dense_') else 'sparse_old_support',
            next_support='fresh' if self.policy.endswith('_fresh') else 'held',dense_output=_summary(dense),
            sparse_old_output=_summary(sparse),returned_output_sha256=fingerprint(out),original_mage_calls=1,rng_unchanged=True))
        return out

    def receipt(self):
        r=super().receipt()['refresh_diagnostic']
        r.update(intervention='first held sparse call after initial selection',
            physical_accounting='original sparse consumer ledger plus separately counted dense observation',timing_valid=False,
            fresh_selection='same K and V31 sticky bonus; current exact QK observations',sham_policy='sparse_hold')
        return dict(refresh_diagnostic=r)


def install(adapter):
    r=RepairRuntime(adapter);adapter._refresh_diagnostic_runtime=r
    adapter._mage=types.MethodType(lambda self,*args,**kw:r.mage(*args,**kw),adapter)
    adapter.begin_request=types.MethodType(lambda self,*args,**kw:r.begin_request(*args,**kw),adapter)
    adapter._value_receipt=types.MethodType(lambda self:dict(r.original_receipt(),**r.receipt()),adapter)
    return r


def install_budget_audit(adapter):
    """Observe every selected mask in audit passes; no reductions in clean runs."""
    import torch
    begin, select, receipt = adapter.begin_request, adapter._select_initial_refresh, adapter._value_receipt
    state = dict(checks=[], units=0)
    def start(self, *args, **kwargs):
        state.update(checks=[], units=0)
        return begin(*args, **kwargs)
    def selection(self, layer, q, b, scale, prefix, n):
        out, mask = select(layer, q, b, scale, prefix, n)
        pt = prefix // 64
        state['checks'].append((mask[..., :pt].sum(-1) != min(self.mage_k//64, pt)).sum())
        state['checks'].append((~mask[..., pt:]).sum())
        state['units'] += mask.shape[0]*mask.shape[1]*mask.shape[2]
        return out, mask
    def read(self):
        return dict(receipt(), failure_map_budget_units=state['units'],
            failure_map_budget_errors=int(torch.stack(state['checks']).sum()) if state['checks'] else 0)
    adapter.begin_request=types.MethodType(start,adapter)
    adapter._select_initial_refresh=types.MethodType(selection,adapter)
    adapter._value_receipt=types.MethodType(read,adapter)


def main():
    from experiments.numerical_qk_reuse import v31_refresh_diagnostic, vllm_adapter
    if os.environ.get('FAILURE_MAP_REPAIR')=='1':
        if os.environ.get('VALUE_AUDIT')!='1' or os.environ.get('VALUE_CLEAN_TIMING')!='0':
            raise RuntimeError('repair requires explicit audit-only execution')
        v31_refresh_diagnostic.install=lambda adapter,audit=True:install(adapter)
    if os.environ.get('VALUE_AUDIT')=='1':
        original=vllm_adapter.install_vllm_patches
        def audited(adapter):
            result=original(adapter)
            if adapter.arm=='mage':install_budget_audit(adapter)
            return result
        vllm_adapter.install_vllm_patches=audited
    runpy.run_module('scripts.v31_refresh_diagnostic_worker',run_name='__main__')


if __name__=='__main__':main()
