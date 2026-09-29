"""v27 64-row / split-KV consumer: masked-dense reference parity (CUDA)."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA/Triton')


def _ref(q, k, v, skipped, scale):
    h, hk = q.shape[1], k.shape[1]
    kr, vr = k.repeat_interleave(h // hk, 1).float(), v.repeat_interleave(h // hk, 1).float()
    s = torch.einsum('bhqd,bhkd->bhqk', q.float(), kr) * scale
    keep = ~skipped.repeat_interleave(128, 2).repeat_interleave(64, 3)[:, :, :q.shape[2], :k.shape[2]]
    s = s.masked_fill(~keep, float('-inf'))
    return torch.einsum('bhqk,bhkd->bhqd', s.softmax(-1), vr).transpose(1, 2)


@pytest.mark.parametrize('keys,splits', [(700, 1), (700, 2), (1537, 4), (4096, 2)])
def test_consume64_matches_masked_dense(keys, splits):
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64, dense64
    g = torch.Generator(device='cuda').manual_seed(keys)
    q = torch.randn(1, 16, 256, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    kt = math.ceil(keys / 64)
    skipped = torch.rand((1, 16, 2, kt), device='cuda', generator=g) > .6
    skipped[..., 0] = False
    elig = torch.ones_like(skipped)
    got = consume64(q, k, v, skipped, elig, 512 ** -.5, splits=splits).float()
    torch.testing.assert_close(got, _ref(q, k, v, skipped, 512 ** -.5), atol=2e-2, rtol=2e-2)
    dense = dense64(q, k, v, 512 ** -.5, splits=splits).float()
    torch.testing.assert_close(dense, _ref(q, k, v, torch.zeros_like(skipped), 512 ** -.5), atol=2e-2, rtol=2e-2)
    assert dense.is_contiguous() and dense.shape == (1, 256, 16, 512)


def test_causal_dense64_matches_causal_reference():
    from experiments.numerical_qk_reuse.v27_consumer64 import dense64
    g = torch.Generator(device='cuda').manual_seed(3)
    q = torch.randn(1, 16, 700, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, 700, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, 700, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    want = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.repeat_interleave(8, 1).float(), v.repeat_interleave(8, 1).float(),
        is_causal=True, scale=512 ** -.5).transpose(1, 2)
    got = dense64(q, k, v, 512 ** -.5, splits=1, causal=True).float()
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
