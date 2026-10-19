"""The line observer must capture completed outputs without modifying inputs."""
import inspect
import sys
from types import SimpleNamespace

import torch


def test_completed_kernel_snapshot_is_detached_and_inert(tmp_path, monkeypatch):
    from scripts.v31_humaneval_repro_segment_worker import install
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    import vllm.model_executor.models.diffusion_gemma as dg

    # Restore every runtime binding when the test exits.
    for owner, attr in [(LLMEngine, 'add_request'), (LLMEngine, 'step'),
                        (FlashAttentionImpl, 'forward'), (dg.DiffusionSampler, '_finish_prefills')]:
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
