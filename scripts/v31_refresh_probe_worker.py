"""Process-local CPU annotations or one-event snapshots; native worker unchanged."""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import runpy


def install(adapter,mode,output):
    import torch
    from vllm.v1.attention.backends import flash_attn as fa
    from experiments.numerical_qk_reuse import v31_fa4_observe as observe
    from scripts.v31_refresh_headroom_profile import attention_scope
    from scripts.v31_refresh_operator_replay import NativeProbe,tensor_meta,digest,outputs
    state=dict(request=-1,active=False,phase='prefill',seen=False,captured=False,event=None)
    spans=[];captures=[];stack=[]
    @contextmanager
    def scope(label):
        if mode!='nvtx' or not state['active']:yield None;return
        row=dict(id=len(spans),request=state['request'],label=label,parent=stack[-1]['id'] if stack else None)
        spans.append(row);stack.append(row)
        torch.cuda.nvtx.range_push('refresh_probe:'+str(row['id']))
        try:yield row
        finally:torch.cuda.nvtx.range_pop();stack.pop()
    begin,end,prepare=adapter.begin_request,adapter.end_request,adapter.on_prepare
    def start(*a,**kw):
        result=begin(*a,**kw)
        state.update(request=state['request']+1,active=True,phase='prefill',seen=False,captured=False,event=None)
        return result
    def finish(*a,**kw):
        result=end(*a,**kw);state['active']=False;return result
    def prepared(encoder,*a,**kw):
        result=prepare(encoder,*a,**kw)
        if state['active']:
            state['phase']=('commit' if state['seen'] else 'prefill') if encoder else 'denoise'
            state['seen']|=not encoder
        return result
    adapter.begin_request,adapter.end_request,adapter.on_prepare=start,finish,prepared
    original_attention=fa.FlashAttentionImpl.forward
    @wraps(original_attention)
    def attention(*a,**kw):
        if mode!='nvtx' or not state['active']:return original_attention(*a,**kw)
        layer=a[1] if len(a)>1 else kw.get('layer');name=getattr(layer,'layer_name','')
        before=dict(adapter.calls);phase=state['phase']
        with scope(attention_scope(name,phase)) as row:
            result=original_attention(*a,**kw)
            row['label']=attention_scope(name,phase,before,adapter.calls)
            row['layer_name']=name
        return result
    fa.FlashAttentionImpl.forward=attention
    select=adapter._select_initial_refresh
    def selection(layer,q,b,scale,prefix,n):
        st=adapter.mage_state.get(layer)
        genuine=st is not None and st['canvas']==adapter.canvas_id and st['nk']==prefix+n
        capture=(mode=='capture' and state['request']>=2 and layer==5 and genuine and not state['captured'])
        if capture:state['event']=dict(request=state['request'],canvas=adapter.canvas_id,layer=layer,prefix=prefix,n=n,call=adapter._mage_count[layer][2])
        try:
            with scope(state['phase']+'/selection'):return select(layer,q,b,scale,prefix,n)
        finally:state['event']=None
    adapter._select_initial_refresh=selection
    original_observe=observe.observe_dense
    @wraps(original_observe)
    def observed(*a,**kw):
        event=state['event']
        if event is None:
            with scope(state['phase']+'/observation'):return original_observe(*a,**kw)
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        with NativeProbe() as probe:result=original_observe(*a,**kw)
        config,tensors=probe.extract()
        # Keep original caller layouts distinct from any native canonicalization.
        for key,t in zip(('q','k','v'),a[:3]):
            tensors['native_'+key]=tensors[key];tensors[key]=t
        metadata={k:tensor_meta(t) for k,t in tensors.items()}
        values={k:t.detach().cpu().clone() for k,t in tensors.items() if k in ('q','k','v','dynamic_causal','aux1')}
        hashes={k:digest(t) for k,t in values.items()}
        saved=dict(event=dict(event),scale=float(a[3]),metadata=metadata,values=values,hashes=hashes,config=config,outputs=outputs(tensors))
        filename='snapshot%03d.pt'%len(captures)
        torch.save(saved,output/'private'/filename)
        captures.append(dict(file=filename,event=dict(event),metadata=metadata,hashes=hashes,config=config,output_hash=digest(result),
            rng_unchanged=bool(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()))))
        state['captured']=True
        return result
    observe.observe_dense=observed
    for name,label in (('_buffers','kv_buffer'),('_mage_units','mask_rank')):
        original=getattr(adapter,name)
        def wrapped(*a,_fn=original,_label=label,**kw):
            with scope(state['phase']+'/'+_label):return _fn(*a,**kw)
        setattr(adapter,name,wrapped)
    return lambda: (output/'probe.json').write_text(json.dumps(dict(mode=mode,scopes=spans,captures=captures),indent=2)+'\n')


def main():
    from experiments.numerical_qk_reuse import vllm_adapter
    output=Path(os.environ['PROBE_OUTPUT']);mode=os.environ['PROBE_MODE']
    original=vllm_adapter.install_vllm_patches;finish=[]
    def patched(adapter):
        result=original(adapter);finish.append(install(adapter,mode,output));return result
    vllm_adapter.install_vllm_patches=patched
    try:runpy.run_path(os.environ['PROBE_ORIGINAL_WORKER'],run_name='__main__')
    finally:
        for callback in finish:callback()


if __name__=='__main__':main()
