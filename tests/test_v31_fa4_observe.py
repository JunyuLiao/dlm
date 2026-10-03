"""v31 FA4 in-kernel observation: z matches the FP32 reference and the Triton fused observation; the dense output is
bit-identical to the plain FA4 call (GPU only)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 needs CUDA')

H, HK, D, N = 16, 2, 512, 256


def _inputs(prefix, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    nk = prefix + N
    q = torch.randn(1, H, N, D, device='cuda', generator=g).to(torch.bfloat16)
    k = torch.randn(1, HK, nk, D, device='cuda', generator=g).to(torch.bfloat16)
    v = torch.randn(1, HK, nk, D, device='cuda', generator=g).to(torch.bfloat16)
    k[0, 0, 5 * 64:6 * 64] += q[0, :8].float().mean((0, 1)).to(torch.bfloat16) * 3   # a heavy tile
    return q, k, v


def _reference_z(q, k, scale, pt):
    s = torch.matmul(q[0].float().reshape(HK, H // HK, N, D), k[0, :, :pt * 64].float().transpose(-1, -2).unsqueeze(1))
    s = (s * scale).reshape(H, N, pt, 64)
    return torch.logsumexp(s, -1)                                       # [H, N, PT]


@pytest.mark.parametrize('prefix,scale', [(64 * 40 + 17, 0.05), (64 * 300, 1 / math.sqrt(D))])
def test_z_and_output(prefix, scale):
    from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    q, k, v = _inputs(prefix)
    pt = prefix // 64
    qb = -(-N // 128)
    z = torch.full((H, qb, pt, 128), float('nan'), device='cuda')
    out = observe_dense(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale, z)
    dyn = torch.tensor([False], device='cuda')
    plain = _flash_attn_fwd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scale, causal=True,
                            dynamic_causal=dyn, num_splits=0, pack_gqa=False)[0]
    assert torch.equal(out, plain)
    got = z.permute(0, 1, 3, 2).reshape(H, qb * 128, pt)[:, :N]          # [H, N, PT]
    torch.testing.assert_close(got, _reference_z(q, k, scale, pt), rtol=2e-5, atol=2e-4)


def test_matches_triton_fused_observation():
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe
    from experiments.numerical_qk_reuse.v31_fa4_observe import fused_observe_fa4
    prefix, scale = 64 * 120 + 33, 0.05
    q, k, v = _inputs(prefix, seed=2)
    nk, pt = prefix + N, prefix // 64
    sketch = torch.randn(1, HK, nk, 32, device='cuda')
    a = allocate_summary(1, H, 2, -(-nk // 64), pt, 0, 'cuda', ('t',))
    b = allocate_summary(1, H, 2, -(-nk // 64), pt, 0, 'cuda', ('t',))
    out_t, tail_t = fused_observe(q, k, v, sketch, scale, pt, a, splits=2, mu=False, mu_precision='bf16')
    out_f, tail_f = fused_observe_fa4(q, k, v, sketch, scale, pt, b)
    torch.testing.assert_close(b.z, a.z, rtol=2e-5, atol=2e-4)
    assert torch.equal(b.active, a.active) and torch.equal(b.bad, a.bad)
    torch.testing.assert_close(tail_f, tail_t, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(out_f.float(), out_t.float(), rtol=2e-2, atol=2e-2)   # bf16 outputs, two kernels
