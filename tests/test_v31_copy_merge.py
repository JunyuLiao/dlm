"""v31 check of the V29/V30 opt-in kernels used by KV_COPY=triton / MERGE=triton: identical to the adapter's torch
paths (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton kernels need CUDA')


def _adapter(**kw):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    return VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='allkept', **kw)


@pytest.mark.parametrize('prefix,n', [(32 * 300 + 7, 256), (32 * 64, 200)])
def test_paged_copy_matches_torch(prefix, n):
    g = torch.Generator(device='cuda').manual_seed(prefix)
    page, hk, d = 32, 2, 512
    nk = prefix + n
    pages = -(-nk // page) + 3
    key = torch.randn(pages + 5, page, hk, d, device='cuda', generator=g).to(torch.bfloat16)
    value = torch.randn(pages + 5, page, hk, d, device='cuda', generator=g).to(torch.bfloat16)
    table = torch.randperm(pages + 5, device='cuda', generator=g)[:pages].to(torch.int32)
    torch_a, triton_a = _adapter(), _adapter(kv_copy_backend='triton')
    for a in (torch_a, triton_a):
        a.buffers.clear()
        a._buffers(5, key, value, table, prefix, n)                      # prefix copy
        key[table[prefix // page].long(), prefix % page:] += 1           # the canvas changes between calls
        a._buffers(5, key, value, table, prefix, n)                      # canvas refresh
        key[table[prefix // page].long(), prefix % page:] -= 1
    assert torch.equal(torch_a.buffers[5]['k'], triton_a.buffers[5]['k'])
    assert torch.equal(torch_a.buffers[5]['v'], triton_a.buffers[5]['v'])


def test_lse_merge_matches_torch():
    g = torch.Generator(device='cuda').manual_seed(9)
    S, Q, H, D = 2, 256, 16, 512
    o = torch.randn(S, Q, H, D, device='cuda', generator=g).to(torch.bfloat16)
    lse = torch.randn(S, H, Q, device='cuda', generator=g) * 3
    lse[1, 3, 5] = float('-inf')                                          # an empty split for one row
    w = torch.softmax(lse, dim=0).permute(0, 2, 1)[..., None]
    ref = (o.float() * w).sum(0, keepdim=True).to(o.dtype)
    got = _adapter(merge_backend='triton')._merge_alias2(o, lse)
    torch.testing.assert_close(got.float(), ref.float(), rtol=0, atol=1e-2)
    assert (got != ref).float().mean() < 0.01                             # BF16 rounding of an FP32 sum only
