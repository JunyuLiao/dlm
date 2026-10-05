"""v11 CP1 qualification of the preselected-support Hopper consumer.

Tolerances were FIXED before any result was inspected:
  * vs an independent FP32 reference on the same support/legality:
    max|new-ref| <= 1.5 * max|triton-ref| + 1e-3  and  ||new-ref||/||ref|| <= 1e-2,
    where triton = the frozen O preqk consumer on the identical support.
  * vs fresh-T output when fed fresh T's OWN support on the same Q/K/V/mask:
    the kept-tile arithmetic is the same, so equality is tested bit for bit;
    any difference is reported as a failure, not absorbed into a tolerance.
Work counters must equal the kept-tile count of each CTA's Q128 tile exactly.
"""
import json
import math
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
IDENTITY = os.environ.get('V11_SUPPORT_BUILD')
JUNYU = dict(library='/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/value_direction_db080045f7a5fbce.so',
             torch_library='/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/torch_4c65c048754f9fb7/'
                           'value_direction_torch_4c65c048754f9fb7.so')
LOCAL = dict(h=16, hk=8, d=256, window=1024)
GLOBAL = dict(h=16, hk=2, d=512, window=0)


@pytest.fixture(scope='module')
def support():
    from experiments.value_direction_hopper import support as s
    if not IDENTITY:
        pytest.skip('set V11_SUPPORT_BUILD to build_identity.json')
    s.load(IDENTITY)
    return s


def make(seed, geometry, prefix, nq=256, crop=True):
    g = torch.Generator(device='cuda').manual_seed(seed)
    h, hk, d, window = geometry['h'], geometry['hk'], geometry['d'], geometry['window']
    c = (max(0, prefix - window + 1) // 64 * 64) if (window and crop) else 0
    nk = prefix + nq - c
    q = (torch.randn(1, nq, h, d, generator=g, device='cuda', dtype=torch.bfloat16) * .3).transpose(1, 2)   # model view
    kv = torch.randn(1, hk, nk, 2 * d, generator=g, device='cuda', dtype=torch.bfloat16) * .3
    k, v = kv[..., :d], kv[..., d:]                         # strided views, like model slices
    return q, k, v, window, d ** -.5


def legal(nq, nk, window, device):
    qi = torch.arange(nq, device=device)[:, None]
    key = torch.arange(nk, device=device)[None, :]
    ok = torch.ones(nq, nk, dtype=torch.bool, device=device)
    if window:
        ok &= key >= qi + (nk - nq) - window + 1
    return ok


def reference(q, k, v, skipped, eligible, window, scale):
    b, h, nq, d = q.shape
    hk, nk = k.shape[1], k.shape[2]
    rep = h // hk
    kk = k.repeat_interleave(rep, 1).float()
    vv = v.repeat_interleave(rep, 1).float()
    s = torch.matmul(q.float(), kk.transpose(-1, -2))
    s = (s.to(torch.bfloat16).float() * scale).to(torch.bfloat16).float()
    keep = (eligible & ~skipped).repeat_interleave(128, 2)[:, :, :nq].repeat_interleave(64, 3)[..., :nk]
    ok = keep & legal(nq, nk, window, q.device)
    s = s.masked_fill(~ok, -math.inf)
    rows = ok.any(-1, keepdim=True)
    p = torch.softmax(s.masked_fill(~rows, 0.), -1) * rows
    return torch.matmul(p, vv)


def routed(q, k, v, window, scale, threshold, seed):
    from experiments.numerical_qk_reuse.integration import Attention
    from experiments.numerical_qk_reuse.cached_executor import route_only
    scores = Attention.observe_scores(q, k, None, scale, False, window or None, 0)
    g = torch.Generator(device='cuda').manual_seed(seed)
    z = torch.randn(1, k.shape[1], k.shape[2], 32, generator=g, device='cuda').contiguous()
    ref = (torch.rand(1, k.shape[1], generator=g, device='cuda') + .5).contiguous()
    sens = (1 + 3 * torch.rand(1, q.shape[2], generator=g, device='cuda')).contiguous()
    r = route_only(scores, z, ref, sensitivity=sens, log_threshold=threshold)
    return r.skipped, r.eligible


def triton_consumer(q, k, v, skipped, eligible, window, scale):
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention
    return preqk_attention(q, k, v, skipped, eligible, scale=scale, window=window or None,
                           variant='generic').output.float()


def check(support, q, k, v, skipped, eligible, window, scale):
    before = (skipped.clone(), eligible.clone())
    out, lse, invalid, counters = support.attention(q, k, v, skipped, eligible, scale=scale, window=window,
                                                    counters=True)
    torch.cuda.synchronize()
    assert torch.equal(before[0], skipped) and torch.equal(before[1], eligible)   # read-only
    ref = reference(q, k, v, skipped, eligible, window, scale)
    tri = triton_consumer(q, k, v, skipped, eligible, window, scale)
    new_err = (out.float() - ref).abs().max().item()
    tri_err = (tri - ref).abs().max().item()
    rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    assert new_err <= 1.5 * tri_err + 1e-3, (new_err, tri_err)
    assert rel <= 1e-2, rel
    assert not invalid.any()
    keep = (eligible & ~skipped).sum(-1)                         # [B,H,QB]
    expected = keep.repeat_interleave(2, -1)[..., :counters.shape[2]]
    for field in range(4):
        assert torch.equal(counters[..., field], expected.to(torch.int32)), field
    return dict(new_err=new_err, triton_err=tri_err, rel=rel)


CASES = [('local', LOCAL, p, t) for p, t in ((63, -1.0), (64, -1.0), (65, -1.0), (365, -1.0), (1023, -1.0), (1645, -1.0))] + \
        [('global', GLOBAL, p, t) for p, t in ((129, -3.1), (1645, -3.1), (5004, -3.1))] + \
        [('local_all_kept', LOCAL, 700, -math.inf), ('global_sparse', GLOBAL, 2000, 2.0)]


@pytest.mark.parametrize('name,geometry,prefix,threshold', CASES, ids=[f'{c[0]}-{c[2]}' for c in CASES])
def test_matches_fp32_reference_within_triton_envelope(support, name, geometry, prefix, threshold):
    q, k, v, window, scale = make(prefix, geometry, prefix)
    skipped, eligible = routed(q, k, v, window, scale, threshold, prefix)
    stats = check(support, q, k, v, skipped, eligible, window, scale)
    if name == 'global_sparse':
        assert skipped.any()


@pytest.mark.parametrize('pattern', ['all_dropped', 'one_kept_first', 'last_only', 'alternating', 'long_runs', 'tail_q'])
def test_constructed_support_patterns(support, pattern):
    nq = 77 if pattern == 'tail_q' else 256
    q, k, v, window, scale = make(7, GLOBAL, 1000, nq=nq)
    b, h, qb, kt = 1, 16, (nq + 127) // 128, (k.shape[2] + 63) // 64
    eligible = torch.ones(b, h, qb, kt, dtype=torch.bool, device='cuda')
    skipped = torch.zeros_like(eligible)
    j = torch.arange(kt, device='cuda')
    if pattern == 'all_dropped':
        skipped[:] = True
    elif pattern == 'one_kept_first':
        skipped[..., 1:] = True
    elif pattern == 'last_only':
        skipped[..., :-1] = True
    elif pattern == 'alternating':
        skipped[..., j % 2 == 1] = True
        eligible[:, 3] = False                                     # an ineligible head row
    elif pattern == 'long_runs':
        skipped[..., (j // 5) % 2 == 0] = True
    out, lse, invalid, counters = support.attention(q, k, v, skipped, eligible, scale=scale, window=window, counters=True)
    torch.cuda.synchronize()
    if pattern == 'all_dropped':
        assert not out.float().abs().any() and torch.isinf(lse).all() and not invalid.any()
        assert not counters.any()                                   # zero QK/K/V/PV work issued
        return
    check(support, q, k, v, skipped, eligible, window, scale)


def test_invalid_scores_flag_only_when_retained(support):
    q, k, v, window, scale = make(11, GLOBAL, 600)
    kt = (k.shape[2] + 63) // 64
    k = k.clone()
    k[0, 0, 70] = float('nan')                                     # key 70 -> tile 1, kv head 0
    eligible = torch.ones(1, 16, 2, kt, dtype=torch.bool, device='cuda')
    skipped = torch.zeros_like(eligible)
    out, lse, invalid, _ = support.attention(q, k, v, skipped, eligible, scale=scale, window=window)
    torch.cuda.synchronize()
    heads = 16 // 2
    assert invalid[0, :heads].all() and not invalid[0, heads:].any()
    assert not out[0, :heads].float().abs().any() and torch.isinf(lse[0, :heads]).all()
    skipped[..., 1] = True                                         # drop the poisoned tile
    out, lse, invalid, _ = support.attention(q, k, v, skipped, eligible, scale=scale, window=window)
    torch.cuda.synchronize()
    assert not invalid.any() and torch.isfinite(out.float()).all()


def test_nondefault_stream_and_model_layout(support):
    q, k, v, window, scale = make(3, LOCAL, 900)
    skipped, eligible = routed(q, k, v, window, scale, -1.0, 3)
    base = support.attention(q, k, v, skipped, eligible, scale=scale, window=window)[0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        side = support.attention(q, k, v, skipped, eligible, scale=scale, window=window, layout=1)[0]
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert side.stride(2) == 16 * 256 and torch.equal(base, side)   # [B,Q,H,D] storage, same values


@pytest.mark.parametrize('geometry,prefix', [(LOCAL, 1023), (LOCAL, 365), (GLOBAL, 1645), (GLOBAL, 129)])
def test_bit_identical_to_fresh_T_on_its_own_support(support, geometry, prefix):
    """Same Q/K/V, same packed legality: feed fresh T's decision to the new consumer."""
    from experiments.value_direction_hopper.cuda import Kernel
    from experiments.value_direction_hopper.masks import geometry as geom
    kernel = Kernel(JUNYU['library'], torch_library=JUNYU['torch_library'])
    q, k, v, window, scale = make(prefix, geometry, prefix, crop=False)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g = torch.Generator(device='cuda').manual_seed(prefix)
    z = torch.randn(1, k.shape[1], k.shape[2], 32, generator=g, device='cuda').contiguous()
    ref = (torch.rand(1, k.shape[1], generator=g, device='cuda') + .5).contiguous()
    sens = (1 + 3 * torch.rand(1, q.shape[2], generator=g, device='cuda')).contiguous()
    packed, _ = geom(1, q.shape[2], k.shape[2], device='cuda', window=window)
    threshold = -1.0 if window else 1.0          # must actually drop tiles for the test to mean anything
    fresh = kernel(q, k, v, z, ref, mask=packed, scale=scale, log_threshold=threshold, mode='value',
                   precision='tf32x3_register', tma=True, sensitivity=sens)
    out, lse, invalid, _ = support.attention(q, k, v, fresh.skipped, fresh.eligible, scale=scale, mask=packed.tensor)
    torch.cuda.synchronize()
    assert fresh.skipped.any() or prefix < 200
    assert torch.equal(out.view(torch.int16), fresh.output.view(torch.int16)), \
        (out.float() - fresh.output.float()).abs().max().item()


@pytest.mark.parametrize('geometry', [LOCAL, GLOBAL])
def test_dropped_tiles_are_never_loaded_or_multiplied_nan_poisoning(support, geometry):
    """If any K load/QK or V load/PV touched a dropped tile, NaN would reach the
    output (0 * NaN = NaN in PV). Output must be bit-identical to the clean run."""
    q, k, v, window, scale = make(5, geometry, 1500)
    skipped, eligible = routed(q, k, v, window, scale, -1.0 if window else 0.0, 5)
    assert skipped.any()
    clean = support.attention(q, k, v, skipped, eligible, scale=scale, window=window)[0].clone()
    # a key tile dropped for EVERY query head sharing a kv head, in every Q tile, is safe to poison
    hk = k.shape[1]
    drop = (skipped | ~eligible).reshape(1, hk, -1, skipped.shape[2], skipped.shape[3]).all(2).all(2)   # [1,HK,KT]
    assert drop.any()
    kp, vp = k.clone(), v.clone()
    for head in range(hk):
        for tile in drop[0, head].nonzero().flatten().tolist():
            kp[0, head, tile * 64:(tile + 1) * 64] = float('nan')
            vp[0, head, tile * 64:(tile + 1) * 64] = float('nan')
    out, lse, invalid, counters = support.attention(q, kp, vp, skipped, eligible, scale=scale, window=window, counters=True)
    torch.cuda.synchronize()
    assert not invalid.any() and torch.equal(out.view(torch.int16), clean.view(torch.int16))
