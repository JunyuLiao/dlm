"""The line observer must capture completed outputs without modifying inputs."""
import inspect
import functools
import sys
from types import SimpleNamespace

import torch


def test_trajectory_observer_adds_no_sampler_call_or_rng_consumption(tmp_path,monkeypatch):
    from scripts.v31_humaneval_repro_trajectory_worker import install
    from vllm.v1.engine.llm_engine import LLMEngine
    import vllm.model_executor.models.diffusion_gemma as dg
    calls = []
    def sample(*args, **kwargs):
        calls.append(1)
        args[7].add_(1)
        args[5].add_(1)
        return args[0]
    monkeypatch.setattr(dg,'_compiled_sample_step',sample)
    monkeypatch.setattr(LLMEngine,'add_request',lambda *a,**k:None)
    monkeypatch.setattr(LLMEngine,'step',lambda *a,**k:[])
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.zeros(4,dtype=torch.uint8))
    install(['clean','observe','clean'],tmp_path)
    state = inspect.getclosurevars(dg._compiled_sample_step).nonlocals['state']
    args = [torch.zeros(1),torch.tensor([0]),None,None,None,torch.zeros(1,4),
            torch.zeros(1,4),torch.zeros(1,dtype=torch.int32),torch.zeros(1,dtype=torch.int32),
            None,torch.zeros(1,4,3)]
    rng_before = torch.get_rng_state().clone()
    for name,repetitions in [('clean',1),('observe',2),('clean',1)]:
        LLMEngine.add_request(None,name)
        for _ in range(repetitions):
            assert dg._compiled_sample_step(*args) is args[0]
        assert len(state['entries']) == (2 if name == 'observe' else 0)
        if name == 'observe':
            assert state['initial']['step'].item() == 1
            assert state['entries'][-1]['step'].item() == 3
    assert len(calls) == 4
    assert args[7].item() == 4
    assert torch.equal(torch.get_rng_state(),rng_before)


def test_native_helper_recompilation_keeps_exact_python_functions(monkeypatch):
    from scripts.v31_humaneval_repro_policy_worker import recompile_native_helpers
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.platforms import current_platform
    names = ('_softcap_logits', '_compute_num_rejected', '_compiled_sample_step')
    native = {name:getattr(dg,name)._torchdynamo_orig_callable for name in names}
    calls = []
    def compile_existing(fn, **options):
        calls.append((fn,options))
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapped._torchdynamo_orig_callable = fn
        return wrapped
    monkeypatch.setattr(torch,'compile',compile_existing)
    monkeypatch.setattr(current_platform,'simple_compile_backend','inductor')
    for name in names:
        monkeypatch.setattr(dg,name,getattr(dg,name))
    receipts = recompile_native_helpers()
    assert [fn for fn,options in calls] == list(native.values())
    for name in names:
        assert getattr(dg,name)._torchdynamo_orig_callable is native[name]
        assert receipts[name]['original_python_callable_preserved']
    assert all(options == dict(dynamic=True,backend='inductor',
        options=dict(deterministic=True,benchmark_combo_kernel=False)) for _,options in calls)


def test_compiler_policy_preserves_native_graph_and_other_options():
    from scripts.v31_humaneval_repro_policy_worker import compilation_options
    original = dict(cudagraph_mode='PIECEWISE',inductor_compile_config=dict(combo_kernels=True))
    fixed = compilation_options(original)
    assert fixed['cudagraph_mode'] == 'PIECEWISE'
    assert fixed['inductor_compile_config'] == dict(combo_kernels=True,deterministic=True,
                                                   benchmark_combo_kernel=False)
    assert original['inductor_compile_config'] == dict(combo_kernels=True)
    assert 'cudagraph_mode' not in compilation_options(None)


def test_segment_dispatch_preserves_downstream_captured_addresses():
    from scripts.v31_humaneval_repro_segment_worker import copy_to_captured_outputs
    fresh = (torch.arange(6).reshape(2, 3), [torch.ones(2), None])
    captured = (torch.zeros(2, 3, dtype=torch.int64), [torch.zeros(2), None])
    addresses = [captured[0].data_ptr(), captured[1][0].data_ptr()]
    returned = copy_to_captured_outputs(fresh, captured)
    assert returned is captured
    assert [returned[0].data_ptr(), returned[1][0].data_ptr()] == addresses
    assert torch.equal(returned[0], fresh[0])
    assert torch.equal(returned[1][0], fresh[1][0])
    fresh[0].zero_()
    assert torch.equal(returned[0], torch.arange(6).reshape(2, 3))


def test_completed_kernel_snapshot_is_detached_and_inert(tmp_path, monkeypatch):
    from scripts.v31_humaneval_repro_segment_worker import install
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    import vllm.model_executor.models.diffusion_gemma as dg

    # Restore every runtime binding when the test exits.
    for owner, attr in [(LLMEngine, 'add_request'), (LLMEngine, 'step'),
                        (FlashAttentionImpl, 'forward'), (dg.DiffusionSampler, '_finish_prefills'),
                        (CUDAGraphWrapper, '__call__')]:
        monkeypatch.setattr(owner, attr, getattr(owner, attr))
    install([], tmp_path / 'unused')
    bindings = inspect.getclosurevars(FlashAttentionImpl.forward).nonlocals
    trace, state = bindings['trace'], bindings['state']
    state['segment'] = True
    folder = tmp_path / 'inductor_cache'
    folder.mkdir()
    path = folder / 'probe.py'
    path.write_text('def call(args):\n'
                    '    arg0_1, arg1_1 = args\n'
                    '    buf0 = torch.empty_like(arg0_1)\n'
                    '    extern_kernels.mm(arg0_1, arg1_1, out=buf0)\n'
                    '    return buf0\n')
    namespace = dict(torch=torch, extern_kernels=SimpleNamespace(mm=torch.mm))
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    b = torch.tensor([[1., 2.], [3., 4.]])
    old_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        out = namespace['call']([a, b])
    finally:
        sys.settrace(old_trace)
        state['segment'] = False
    assert len(state['entries']) == 1
    saved = state['entries'][0]['values']
    assert torch.equal(saved['buf0']['value'], a @ b)
    assert torch.equal(saved['arg0_1']['value'], a)
    assert torch.equal(a, torch.arange(4, dtype=torch.float32).reshape(2, 2))
    out.zero_()
    assert torch.equal(saved['buf0']['value'], a @ b)
