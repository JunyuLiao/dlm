import json
import sys
from types import ModuleType,SimpleNamespace

import torch
import pytest
import vllm.model_executor.models.diffusion_gemma as dg
from experiments import numerical_qk_reuse as package
from experiments.numerical_qk_reuse import vllm_adapter
from scripts import v31_humaneval_attention_fingerprint as observer


@pytest.mark.parametrize('mode',['pre_post','post_only'])
def test_attention_wrapper_binds_after_adapter_and_preserves_native_result(tmp_path,monkeypatch,mode):
    calls=[];hashes=[];order=[]
    modes=tmp_path/'modes.json';modes.write_text(json.dumps({'0':mode}))
    monkeypatch.setenv('HUMANEVAL_ATTENTION_MODES',str(modes))
    class Engine:
        def add_request(self,*args,**kwargs):return 'native_request'
        def step(self):return [SimpleNamespace(finished=True)]
    class Attention:pass
    engine_module=ModuleType('vllm.v1.engine.llm_engine');engine_module.LLMEngine=Engine
    attention_module=ModuleType('vllm.v1.attention.backends.flash_attn');attention_module.FlashAttentionImpl=Attention
    monkeypatch.setitem(sys.modules,engine_module.__name__,engine_module)
    monkeypatch.setitem(sys.modules,attention_module.__name__,attention_module)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    fingerprint=ModuleType('humaneval_gpu_fingerprint')
    def digest(value):
        order.append('hash')
        hashes.append(value.shape)
        return torch.tensor([int(value.numel()),0],dtype=torch.uint64)
    fingerprint.fingerprint_cuda=digest
    monkeypatch.setitem(sys.modules,fingerprint.__name__,fingerprint)
    sparse=ModuleType('experiments.numerical_qk_reuse.v27_fa4');sparse.sparse_lists=lambda *a,**k:None
    dense=ModuleType('experiments.numerical_qk_reuse.v31_fa4_observe');dense.observe_dense=lambda *a,**k:None
    monkeypatch.setattr(package,'v27_fa4',sparse,raising=False)
    monkeypatch.setattr(package,'v31_fa4_observe',dense,raising=False)
    monkeypatch.setitem(sys.modules,sparse.__name__,sparse);monkeypatch.setitem(sys.modules,dense.__name__,dense)
    result_token=object()
    monkeypatch.setattr(dg,'_compiled_sample_step',lambda *a,**k:result_token)
    def native(self,layer,query,key,value,cache,metadata,output,*args,**kwargs):
        calls.append('native');order.append('native');output.copy_(query+1);return output
    def install(adapter):Attention.forward=native;return 'installed'
    monkeypatch.setattr(vllm_adapter,'install_vllm_patches',install)
    state=dict(active=True,index=0)
    observer.install(state,tmp_path)
    adapter=SimpleNamespace(step_ctx=dict(encoder=False,n=2,seq_len=2),canvas_id=0,
        mage_state={},buffers={},_trig_at=None,_trig_count=0)
    assert vllm_adapter.install_vllm_patches(adapter)=='installed'
    engine=Engine();assert engine.add_request('observed')=='native_request'
    q=torch.arange(4,dtype=torch.float32).reshape(2,1,2);before=q.clone();out=torch.empty_like(q)
    result=Attention().forward(SimpleNamespace(layer_name='model.layers.5.self_attn.attn'),q,q,q,None,
                               SimpleNamespace(num_actual_tokens=2),out)
    assert result is out and torch.equal(out,q+1) and torch.equal(q,before)
    assert calls==['native'] and len(hashes)==4
    assert order==(['hash']*3+['native','hash'] if mode=='pre_post' else ['native']+['hash']*4)
    assert dg._compiled_sample_step() is result_token and state['attention_samples']==1
    engine.step()
    trace=json.loads(next(tmp_path.glob('request*.json')).read_text())
    assert len(trace['events'])==1 and trace['events'][0]['layer']==5
    assert trace['events'][0]['sampler_index']==0 and trace['extra_forwards']==0
    state['active']=False;engine.add_request('unobserved')
    Attention().forward(SimpleNamespace(layer_name='model.layers.5.self_attn.attn'),q,q,q,None,
                        SimpleNamespace(num_actual_tokens=2),out)
    assert len(hashes)==4 and calls==['native','native']
