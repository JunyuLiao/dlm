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


def test_fa4_consumer_under_inference_mode_caches_by_map_object():
    from types import SimpleNamespace
    from experiments.numerical_qk_reuse.integration import Attention
    g = torch.Generator(device='cuda').manual_seed(9)
    keys = 2049
    fake = SimpleNamespace(consumer='fa4', trace=False, output_layout='model_major', _fa4_lists=[], fa4_list_builds=0)
    with torch.inference_mode():
        q = torch.randn(1, 256, 16, 512, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
        k = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
        skipped = torch.rand(1, 16, 2, math.ceil(keys / 64), device='cuda', generator=g) > .5
        skipped[..., 0] = False
        eligible = torch.ones_like(skipped)
        first = Attention._consume(fake, q, k, v, skipped, eligible, 512 ** -.5, None, False)
        again = Attention._consume(fake, q, k, v, skipped, eligible, 512 ** -.5, None, False)
        assert fake.fa4_list_builds == 1 and torch.equal(first.output, again.output)
        other = skipped.clone()
        other[..., 1] = ~other[..., 1]
        Attention._consume(fake, q, k, v, other, eligible, 512 ** -.5, None, False)
        assert fake.fa4_list_builds == 2
    torch.testing.assert_close(first.output.transpose(1, 2).float(), _ref(q, k, v, ~skipped), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('keys', [1100, 4133, 17284])
def test_fa4_dense_is_the_all_kept_path_and_bitwise_equals_plain_dense(keys):
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(keys + 1)
    with torch.inference_mode():
        q = torch.randn(1, 256, 16, 512, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
        k = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
        fast, plain = v27_fa4.dense(q, k, v, 1.0), v27_fa4.dense_plain(q, k, v, 1.0)
        kept = torch.ones(1, 16, 2, math.ceil(keys / 64), dtype=torch.bool, device='cuda')
        built = v27_fa4.sparse(q, k, v, ~kept, torch.ones_like(kept), 1.0)
    assert torch.equal(fast, plain) and torch.equal(fast, built)


@pytest.mark.parametrize('keys', [1279, 1280])
def test_fa4_block_sparse_at_the_local_layer_geometry(keys):
    # LOCAL layers (G75 S15/S30 route them): 16 Q / 8 KV heads, head_dim 256, sliding cache + canvas keys
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(keys)
    q = torch.randn(1, 256, 16, 256, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
    k = torch.randn(1, 8, keys, 256, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 8, keys, 256, device='cuda', dtype=torch.bfloat16, generator=g)
    torch.testing.assert_close(v27_fa4.dense(q, k, v, 256 ** -.5).float(), _ref(q, k, v), atol=2e-2, rtol=2e-2)
    kt = math.ceil(keys / 64)
    skipped = torch.rand(1, 16, 2, kt, device='cuda', generator=g) > .7
    skipped[..., 0] = False
    got = v27_fa4.sparse(q, k, v, skipped, torch.ones_like(skipped), 256 ** -.5).float()
    torch.testing.assert_close(got, _ref(q, k, v, ~skipped), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('nq,nk,window,d,hk', [(700, 700, 0, 512, 2), (256, 1500, 0, 512, 2),
                                                (900, 900, 256, 256, 8)])
def test_fa4_causal_prefill_and_append(nq, nk, window, d, hk):
    from experiments.numerical_qk_reuse import v27_fa4
    g = torch.Generator(device='cuda').manual_seed(nq + nk)
    q = torch.randn(1, nq, 16, d, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
    k = torch.randn(1, hk, nk, d, device='cuda', dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, hk, nk, d, device='cuda', dtype=torch.bfloat16, generator=g)
    rows = torch.arange(nq, device='cuda')[:, None] + (nk - nq)          # bottom-right aligned
    cols = torch.arange(nk, device='cuda')[None, :]
    allowed = cols <= rows
    if window:
        allowed &= cols > rows - window
    kr, vr = k.repeat_interleave(16 // hk, 1).float(), v.repeat_interleave(16 // hk, 1).float()
    s = torch.einsum('bhqd,bhkd->bhqk', q.float(), kr) * d ** -.5
    ref = torch.einsum('bhqk,bhkd->bhqd', s.masked_fill(~allowed, float('-inf')).softmax(-1), vr).transpose(1, 2)
    torch.testing.assert_close(v27_fa4.causal(q, k, v, d ** -.5, window).float(), ref, atol=2e-2, rtol=2e-2)
