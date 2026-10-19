import json
import sys
from types import ModuleType,SimpleNamespace
import torch
from scripts.v31_humaneval_last_global_probe import Probe


def make_probe(tmp_path,monkeypatch):
    module=ModuleType('humaneval_gpu_fingerprint')
    def fp(x):
        assert bool(torch.isfinite(x).all()),'read uninitialized output'
        return torch.tensor([int(x.double().sum().abs()),x.numel()],dtype=torch.uint64)
    module.fingerprint_cuda=fp;monkeypatch.setitem(sys.modules,module.__name__,module)
    from scripts.v31_humaneval_graph_fingerprint import cpu_fingerprints
    helper=ModuleType('graph_fingerprint');helper.cpu_fingerprints=cpu_fingerprints
    monkeypatch.setitem(sys.modules,helper.__name__,helper)
    return Probe({},tmp_path,[],{'full_swap8':[0]})


def test_fwd_called_once_aux_output_read_only_after_native_write(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);calls=[]
    q=torch.ones(1,2,1,4);k=torch.ones(1,7,1,4);v=k+1
    aux=torch.full((1,1,1,128),float('nan'));out=torch.full_like(q,float('nan'))
    result=(out,None)
    def native(q,k,v,**kwargs):
        calls.append('native');out.copy_(q+1);aux.fill_(3);return result
    probe.state['last_current']=dict(context=dict(seq_len=7),sampler_index=0,arm='full_swap8')
    wrapped=probe.wrap_fwd(native,'observing_dense_fwd')
    assert wrapped(q,k,v,aux_tensors=[aux,torch.ones(1)]) is result
    assert calls==['native'] and bool(torch.isfinite(out).all())
    raw=probe.state['last_raw'][0]
    assert raw['aux'][0]['input'] is None and torch.equal(raw['aux_after'][0]['value'],aux)
    assert raw['layout']=='contiguous' and raw['kind']=='fa4'
    probe.state['last_current']=None
    assert wrapped(q,k,v,aux_tensors=[aux]) is result and calls==['native','native']


def test_actual_paged_operands_keep_ids_alias_metadata_and_logical_order(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch)
    pool=torch.arange(4*64*2*16,dtype=torch.float32).reshape(4,64,2,16)
    k,v=pool[...,:8],pool[...,8:]
    q=torch.ones(1,4,4,8).expand(2,-1,-1,-1)
    table=torch.tensor([[3,1],[3,1]],dtype=torch.int32);kwargs=dict(page_table=table,num_splits=1)
    logical_k,logical_v=probe.logical(k,v,kwargs,67)
    assert torch.equal(logical_k,torch.cat((k[3],k[1,:3])).transpose(0,1)[None])
    assert torch.equal(logical_v,torch.cat((v[3],v[1,:3])).transpose(0,1)[None])
    raw=probe.capture_fwd(q,k,v,kwargs,67)
    assert raw['layout']=='paged' and raw['q']['stride'][0]==0
    assert raw['k_metadata']['storage_group']==raw['v_metadata']['storage_group']
    assert raw['v_metadata']['storage_offset']-raw['k_metadata']['storage_offset']==8
    assert raw['page_ids']['value'].tolist()==[3,1,3,1]
    assert torch.equal(raw['k_pages']['value'],k[torch.tensor([3,1,3,1])])


def test_moments_body_called_once_and_return_identity_preserved(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);calls=[]
    fields=('mass','numerator','tail_mass','tail_numerator','log_scale')
    expected=SimpleNamespace(**{key:torch.ones(2,4) for key in fields})
    def native(q,k,v,*a,**kw):calls.append((q,k,v));return expected
    probe.state['last_anchor']=dict(moments=[])
    q=torch.ones(2,4);saved=q.clone();wrapped=probe.wrap_moments(native)
    assert wrapped(q,q,q,128,.1) is expected and len(calls)==1 and torch.equal(q,saved)
    assert set(probe.state['last_anchor']['moments'][0]['outputs'])==set(fields)
    probe.state['last_anchor']=None
    assert wrapped(q,q,q,128,.1) is expected and len(calls)==2


def test_layer_wrapper_binds_after_adapter_and_independent_mode(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);state=probe.state;state.update(index=-1,active=False)
    probe.modes=[None,'full_swap8'];calls=[];installed=[]
    import vllm.model_executor.models.diffusion_gemma as dg
    from experiments import numerical_qk_reuse as package
    from experiments.numerical_qk_reuse import vllm_adapter,v31_value_fast_runtime as fast
    from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
    class Engine:
        def add_request(self,*a,**k):state['index']+=1;return 'request'
        def step(self):return [SimpleNamespace(finished=True)]
    class Attention:pass
    em=ModuleType('vllm.v1.engine.llm_engine');em.LLMEngine=Engine
    am=ModuleType('vllm.v1.attention.backends.flash_attn');am.FlashAttentionImpl=Attention
    monkeypatch.setitem(sys.modules,em.__name__,em);monkeypatch.setitem(sys.modules,am.__name__,am)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(dg,'_compiled_sample_step',lambda *a,**k:'sample')
    monkeypatch.setattr(fast,'moments_full',lambda *a,**k:None)
    monkeypatch.setattr(fast.FastRuntime,'initialize',lambda *a,**k:None)
    monkeypatch.setattr(ResidualDirectionCache,'apply_cuda',lambda *a,**k:None)
    monkeypatch.setattr(vllm_adapter.VllmMethodAdapter,'_merge_alias2',lambda *a,**k:None)
    fwd=ModuleType('experiments.numerical_qk_reuse.v27_fa4')
    observer=ModuleType('experiments.numerical_qk_reuse.v31_fa4_observe')
    def native_fwd(q,k,v,**kwargs):calls.append('consumer');return q+1,None
    fwd._FWD=native_fwd;observer._flash_attn_fwd=native_fwd
    def load():assert installed;return fwd._FWD
    fwd.load=load
    for m in (fwd,observer):
        monkeypatch.setitem(sys.modules,m.__name__,m)
        monkeypatch.setattr(package,m.__name__.rsplit('.',1)[1],m,raising=False)
    def native(self,layer,q,k,v,cache,metadata,output,*a,**kw):
        value,_=fwd.load()(q[None],torch.ones(1,7,1,4),torch.ones(1,7,1,4))
        output.copy_(value[0]);return output
    def old_install(adapter):installed.append(True);Attention.forward=native;return 'installed'
    monkeypatch.setattr(vllm_adapter,'install_vllm_patches',old_install)
    probe.install()
    lists=SimpleNamespace(block_size=(128,64),**{key:torch.zeros(1,1,1,1,dtype=torch.int32) for key in
        ('full_block_cnt','full_block_idx','mask_block_cnt','mask_block_idx')})
    adapter=SimpleNamespace(step_ctx=dict(encoder=False,seq_len=7,n=2),canvas_id=0,
        buffers={29:dict(k=torch.ones(1,1,7,4),v=torch.ones(1,1,7,4))},
        mage_state={29:dict(kept=torch.ones(1,1,1,1,dtype=torch.bool),lists=lists)})
    assert vllm_adapter.install_vllm_patches(adapter)=='installed'
    e=Engine();e.add_request('off');assert state['last_active'] is False
    e.add_request('probe');assert state['last_active'] and state['active'] is False
    q=torch.ones(2,1,4);out=torch.full_like(q,float('nan'))
    assert Attention().forward(SimpleNamespace(layer_name='model.layers.29.self_attn.attn'),q,q,q,None,
        SimpleNamespace(num_actual_tokens=2),out) is out
    assert calls==['consumer'] and torch.equal(out,q+1)
    assert dg._compiled_sample_step()=='sample' and state['last_samples']==1
    e.step()
    trace=json.loads((tmp_path/'request001.json').read_text())
    assert trace['layer']==29 and trace['extra_forwards']==0 and len(trace['events'])==1
    assert (tmp_path/'request001.private.pt').exists()
