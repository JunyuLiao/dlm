import json
import sys
from types import ModuleType,SimpleNamespace

import torch
from experiments.numerical_qk_reuse.v31_value_fast_runtime import FastRuntime
from experiments.numerical_qk_reuse.v31_value_reuse_guard import ResidualDirectionCache
from scripts import v31_humaneval_residual_repro_worker as observer


def test_residual_observer_excludes_oracle_and_preserves_native_calls(tmp_path,monkeypatch):
    class Engine:
        def add_request(self,*args,**kwargs):return 'native_request'
        def step(self):return [SimpleNamespace(finished=True)]
    module=ModuleType('vllm.v1.engine.llm_engine');module.LLMEngine=Engine
    monkeypatch.setitem(sys.modules,'vllm.v1.engine.llm_engine',module)
    monkeypatch.setenv('VALUE_AUDIT','1');monkeypatch.setenv('VALUE_CLEAN_TIMING','0')
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    selected=torch.tensor([[[1.,2.]],[[3.,4.]]]);dense=selected+.25
    projection=torch.eye(2);identity=(1,64,2);calls=[]
    def initialize(self,adapter,q,b,scale,prefix,n,out,key):
        calls.append(key)
        self.state={'direction_cache':ResidualDirectionCache.build(dense,selected,projection,key)}
    monkeypatch.setattr(FastRuntime,'initialize',initialize)
    monkeypatch.setattr(ResidualDirectionCache,'apply_cuda',ResidualDirectionCache.apply)
    monkeypatch.setattr(ResidualDirectionCache,'build',ResidualDirectionCache.__dict__['build'])
    runtime=FastRuntime('residual_direction_guard')
    observer.install(['clean','observe','clean'],tmp_path)
    engine=Engine();adapter=SimpleNamespace(mage_state={29:{'kept':torch.ones(1,1,1,2,dtype=torch.bool)}})
    current=selected.to(torch.bfloat16);before=current.clone()
    for label in ('unobserved','observed','unobserved'):
        assert engine.add_request(label)=='native_request'
        # The independent diagnostic oracle builds the same cache outside the
        # production initialization scope; it must not enter production traces.
        ResidualDirectionCache.build(dense,selected,projection,identity)
        runtime.initialize(adapter,current[None],{},1.,64,2,current[None],identity)
        cache=runtime.state['direction_cache'];expected=cache.apply(current,identity)
        actual=cache.apply_cuda(current,identity)
        assert torch.equal(actual[0],expected[0]) and torch.equal(actual[1],expected[1])
        engine.step()
    assert calls==[identity]*3 and torch.equal(current,before)
    paths=list(tmp_path.glob('*.json'));assert len(paths)==1
    trace=json.loads(paths[0].read_text())
    assert [r['kind'] for r in trace['events']]==['anchor_inputs','projection','anchor_cache','held_correction']
    assert trace['extra_forwards']==0 and trace['clean_timing'] is False
    assert trace['events'][2]['cache']['residual_squared']['shape']==[2]
