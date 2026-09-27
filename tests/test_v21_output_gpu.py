"""GPU qualification for v21 output arithmetic and storage (run on CUDA host)."""
import math

import pytest
import torch

from experiments.numerical_qk_reuse.cached_executor import (
    attention, guard_flags, preqk_attention,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def tensors(d, nq, nk, h=4, hk=2):
    gen = torch.Generator(device='cuda').manual_seed(2100 + d + nq + nk)
    # Transpose and slice retain nontrivial B/H/L strides, including V.
    q = torch.randn((1, nq, h, d), device='cuda', dtype=torch.bfloat16, generator=gen).transpose(1, 2)
    kv = torch.randn((1, nk, hk, 2*d), device='cuda', dtype=torch.bfloat16, generator=gen)
    k = kv[..., :d].transpose(1, 2)
    v = kv[..., d:].transpose(1, 2)
    return q, k, v


def reference(q, k, v, skipped, eligible, scale):
    """Independent FP32 QK with per-tile BF16 P and FP32 online PV."""
    b, h, nq, d = q.shape
    nk, hk = k.shape[2], k.shape[1]
    result = torch.zeros((b, h, nq, d), device=q.device, dtype=torch.float32)
    for batch in range(b):
        for head in range(h):
            kh = head // (h // hk)
            for row in range(nq):
                maximum = -math.inf
                denom = 0.
                accum = torch.zeros(d, device=q.device, dtype=torch.float32)
                for tile in range((nk + 63) // 64):
                    if not eligible[batch, head, row // 128, tile] or skipped[batch, head, row // 128, tile]:
                        continue
                    start, end = tile * 64, min((tile + 1) * 64, nk)
                    scores = (q[batch, head, row].float() @ k[batch, kh, start:end].float().T) * scale
                    next_max = max(maximum, scores.max().item())
                    old = math.exp(maximum - next_max) if maximum != -math.inf else 0.
                    weights = torch.exp(scores - next_max)
                    denom = old * denom + weights.sum().item()
                    accum = old * accum + weights.to(torch.bfloat16).float() @ v[batch, kh, start:end].float()
                    maximum = next_max
                if denom:
                    result[batch, head, row] = accum / denom
    return result.to(torch.bfloat16)


@pytest.mark.parametrize('d,nq,nk', [(64, 17, 65), (128, 129, 195),
                                     (256, 31, 129), (512, 19, 83)])
@pytest.mark.parametrize('sparse', [False, True])
def test_fp32_scores_against_independent_reference_and_exact_layout(d, nq, nk, sparse):
    q, k, v = tensors(d, nq, nk)
    shape = (1, q.shape[1], (nq + 127)//128, (nk + 63)//64)
    skipped = torch.zeros(shape, device='cuda', dtype=torch.bool)
    eligible = torch.ones_like(skipped)
    if sparse:
        skipped[..., 0] = True
    scale = d ** -.5
    outputs = {}
    for layout in ('head_major', 'model_major'):
        outputs[layout] = preqk_attention(q, k, v, skipped, eligible, scale=scale,
                                          variant='generic', output_score_precision='fp32_scores_bf16_pv',
                                          output_layout=layout)
    a, z = outputs['head_major'], outputs['model_major']
    assert torch.equal(a.output, z.output)
    assert torch.equal(a.invalid_scores, z.invalid_scores)
    assert z.output.transpose(1, 2).is_contiguous()
    assert z.output.stride() == (nq*q.shape[1]*d, d, q.shape[1]*d, 1)
    assert torch.allclose(a.output.float(), reference(q, k, v, skipped, eligible, scale).float(),
                          atol=.035, rtol=.025)
    assert guard_flags(z.output.transpose(1, 2), z.invalid_scores).all().item()


def test_legacy_arithmetic_and_pv_layout_are_exact():
    q, k, v = tensors(128, 21, 77)
    skip = torch.zeros((1, 4, 1, 2), device='cuda', dtype=torch.bool)
    elig = torch.ones_like(skip)
    a = preqk_attention(q, k, v, skip, elig, scale=128**-.5, variant='generic')
    z = preqk_attention(q, k, v, skip, elig, scale=128**-.5, variant='generic',
                        output_layout='model_major')
    assert torch.equal(a.output, z.output)
    # The score-consuming PV kernel also stores directly in model-major order.
    scores = ((q @ k.repeat_interleave(2, dim=1).transpose(-1, -2)) * (128**-.5)).float().contiguous()
    dense_v = v.contiguous()
    p = attention(scores, dense_v, skipped=skip, eligible=elig, variant='generic')
    pm = attention(scores, dense_v, skipped=skip, eligible=elig, variant='generic',
                   output_layout='model_major')
    assert torch.equal(p.output, pm.output)
    assert pm.output.transpose(1, 2).is_contiguous()


def test_invalid_retained_score_is_flagged_and_guard_rejects():
    q, k, v = tensors(64, 17, 65)
    q = q.clone()
    q[0, 0, 0, 0] = float('nan')
    skip = torch.zeros((1, 4, 1, 2), device='cuda', dtype=torch.bool)
    elig = torch.ones_like(skip)
    out = preqk_attention(q, k, v, skip, elig, scale=64**-.5, variant='generic',
                          output_score_precision='fp32_scores_bf16_pv', output_layout='model_major')
    assert out.invalid_scores[0, 0, 0].item()
    assert not guard_flags(out.output.transpose(1, 2), out.invalid_scores).all().item()
