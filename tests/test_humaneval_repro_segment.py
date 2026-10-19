"""The line observer must capture completed outputs without modifying inputs."""
import inspect
import sys
from types import SimpleNamespace

import torch


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
