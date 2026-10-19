"""CPU binding tests for native graph instrumentation; no CUDA/model work."""
import json
import sys
from types import ModuleType,SimpleNamespace
import pytest
import torch
from scripts import v31_humaneval_graph_fingerprint as observer


@pytest.mark.parametrize('active,mode',[(False,'off'),(True,'full'),(False,'graph_only')])
def test_native_dispatch_called_once_and_result_preserved(tmp_path,monkeypatch,active,mode):
    calls=[];order=[];result_token=object()
    class Engine:
        def add_request(self,*a,**k):return result_token
        def step(self):return [SimpleNamespace(finished=True)]
    class Graph:
        runtime_mode=SimpleNamespace(name='PIECEWISE')
        def __call__(self,*a,**k):calls.append((a,k));order.append('native');return result_token
    engine=ModuleType('vllm.v1.engine.llm_engine');engine.LLMEngine=Engine
    graph=ModuleType('vllm.compilation.cuda_graph');graph.CUDAGraphWrapper=Graph
    context=ModuleType('vllm.forward_context')
    context.is_forward_context_available=lambda:True
    context.get_forward_context=lambda:SimpleNamespace(batch_descriptor=None,
        cudagraph_runtime_mode=SimpleNamespace(name='PIECEWISE'))
    import vllm.model_executor.models.diffusion_gemma as dg
    fingerprint=ModuleType('humaneval_gpu_fingerprint')
    fingerprint.fingerprint_cuda=lambda x:pytest.fail('CPU tensor must not launch GPU work')
    for m in (engine,graph,fingerprint,context):monkeypatch.setitem(sys.modules,m.__name__,m)
    monkeypatch.setattr(dg,'_compiled_sample_step',lambda *a,**k:result_token)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    schedule=tmp_path/'modes.json';schedule.write_text(json.dumps([mode]))
    monkeypatch.setenv('HUMANEVAL_GRAPH_MODES',str(schedule))
    state=dict(active=active,index=0)
    observer.install(state,tmp_path)
    value=torch.ones(2,4);saved=value.clone()
    assert Graph()(value) is result_token  # Native startup before any request.
    assert dg._compiled_sample_step() is result_token and state['graph_mode']=='off'
    calls.clear();order.clear()
    request=Engine();assert request.add_request('request') is result_token
    assert Graph()(value,flag=2) is result_token
    assert len(calls)==1 and calls[0][0][0] is value and calls[0][1]==dict(flag=2)
    assert torch.equal(value,saved)
    assert dg._compiled_sample_step() is result_token
    request.step()
    files=list(tmp_path.glob('request*.json'))
    assert len(files)==int(mode!='off')
    if mode!='off':
        trace=json.loads(files[0].read_text())
        assert [e['point'] for e in trace['events']]==['before','after']
        assert all(e['native_runtime_mode']=='PIECEWISE' for e in trace['events'])
        assert trace['native_graph_dispatch_preserved'] and trace['graph_bypasses']==0
        assert trace['extra_forwards']==0 and state['graph_samples']==1


def test_capture_rejected_before_native_dispatch(tmp_path,monkeypatch):
    class Engine:
        def add_request(self,*a,**k):pass
        def step(self):pass
    class Graph:
        def __call__(self,*a,**k):pytest.fail('captured diagnostic must not dispatch')
    import vllm.model_executor.models.diffusion_gemma as dg
    modules={'vllm.v1.engine.llm_engine':('LLMEngine',Engine),
        'vllm.compilation.cuda_graph':('CUDAGraphWrapper',Graph),
        'humaneval_gpu_fingerprint':('fingerprint_cuda',lambda x:None)}
    for name,(key,value) in modules.items():
        module=ModuleType(name);setattr(module,key,value);monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(dg,'_compiled_sample_step',lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    observer.install(dict(active=True,index=0),tmp_path)
    Engine().add_request('request')
    with pytest.raises(RuntimeError,match='entered capture'):Graph()()


def test_activation_fingerprints_read_native_output_after_dispatch(tmp_path,monkeypatch):
    order=[]
    class FakeCuda(torch.Tensor):
        @property
        def is_cuda(self):return True
    value=torch.ones(2,4).as_subclass(FakeCuda)
    output=torch.full((2,4),float('nan')).as_subclass(FakeCuda)
    class Engine:
        def add_request(self,*a,**k):pass
        def step(self):return [SimpleNamespace(finished=True)]
    class Graph:
        runtime_mode=SimpleNamespace(name='PIECEWISE')
        def __call__(self,x):order.append('native');output.copy_(x+1);return output
    import vllm.model_executor.models.diffusion_gemma as dg
    def fingerprint(x):
        order.append('hash')
        assert bool(torch.isfinite(x).all()),'output read before native write'
        return torch.tensor([int(x.sum()),0],dtype=torch.uint64)
    modules={'vllm.v1.engine.llm_engine':('LLMEngine',Engine),
        'vllm.compilation.cuda_graph':('CUDAGraphWrapper',Graph),
        'humaneval_gpu_fingerprint':('fingerprint_cuda',fingerprint)}
    for name,(key,v) in modules.items():
        module=ModuleType(name);setattr(module,key,v);monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(dg,'_compiled_sample_step',lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    observer.install(dict(active=True,index=0),tmp_path)
    e=Engine();e.add_request('request')
    assert Graph()(value) is output and order==['hash','native','hash']
    e.step()
    events=json.loads(next(tmp_path.glob('request*.json')).read_text())['events']
    assert events[0]['values'][0]['gpu128']==[8,0]
    assert events[1]['values'][0]['gpu128']==[16,0]
