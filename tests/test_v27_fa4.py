"""v27 FA4 wrappers vs FP32 references (CUDA + the vendored FA4 overlay; skipped otherwise)."""
import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('V27_FA4_OVERLAY'),
                                reason='requires CUDA and V27_FA4_OVERLAY')


def _ref(q, k, v, kept=None):
    h, hk = q.shape[1], k.shape[1]
    kr, vr = k.repeat_interleave(h // hk, 1).float(), v.repeat_interleave(h // hk, 1).float()
    s = torch.einsum('bhqd,bhkd->bhqk', q.float(), kr) * q.shape[-1] ** -.5
    if kept is not None:
        m = kept.repeat_interleave(128, 2).repeat_interleave(64, 3)[:, :, :q.shape[2], :k.shape[2]]
        s = s.masked_fill(~m, float('-inf'))
    return torch.einsum('bhqk,bhkd->bhqd', s.softmax(-1), vr).transpose(1, 2)


@pytest.mark.parametrize('keys', [1100, 4133])
def test_fa4_dense_and_block_sparse_match_reference(keys):
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(keys)
    q = torch.randn(1, 256, 16, 512, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)   # model views
    k = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
    torch.testing.assert_close(v27_fa4.dense(q, k, v, 512 ** -.5).float(), _ref(q, k, v), atol=2e-2, rtol=2e-2)
    kt = math.ceil(keys / 64)
    skipped = torch.rand(1, 16, 2, kt, device='cuda', generator=g) > .6
    skipped[..., 0] = False
    eligible = torch.ones_like(skipped)
    got = v27_fa4.sparse(q, k, v, skipped, eligible, 512 ** -.5).float()
    torch.testing.assert_close(got, _ref(q, k, v, ~skipped), atol=2e-2, rtol=2e-2)
