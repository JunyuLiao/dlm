"""v27 q64r (within-block query-row regrouping of the 64-row maps; idea credited to chw/value_aware) and the
q64 carried call-0 map."""
import math
import os

import pytest
import torch

from tests.test_v27_q64 import BASE, _brute, _mainline, _state


def _row_need_ref(state, ref, sens, thr):
    b, h, qb, pt, _ = state.lognorm.shape
    kh = torch.arange(h) // (h // ref.shape[1])
    risk = state.lognorm - torch.log(ref)[:, kh][:, :, None, None, None] + torch.log(sens).view(1, 1, qb, 1, 128)
    return (risk >= thr).permute(0, 1, 2, 4, 3).reshape(b, h, qb * 128, pt)


def _groups(order):
    """group index (0..3) of every original row, given the row order of the permuted query."""
    group = torch.empty_like(order)
    rank = torch.arange(order.shape[-1], device=order.device).div(64, rounding_mode='floor')
    return group.scatter_(2, order, rank.expand_as(order).contiguous())


def test_regrouped_map_covers_every_row_and_stays_inside_the_128_row_map():
    from experiments.numerical_qk_reuse.v27_dense_prefix import refine_q64
    pt, kt, thr = 40, 48, -3.87
    state = _state(pt, seed=5)
    ref = torch.tensor([[1.3, 0.8]])
    sens = 1 + 3 * torch.rand(1, 256, generator=torch.Generator().manual_seed(2))
    kept128 = torch.ones(1, 16, 2, kt, dtype=torch.bool)
    kept128[..., :pt] = _brute(state, ref, sens, thr, 128)
    kept128[..., pt + 2] = False                                                   # a skipped tail tile
    eligible = torch.ones_like(kept128)
    eligible[..., :pt] = state.eligible.bool()
    skipped = eligible & ~kept128
    kept, order = refine_q64(state, skipped, eligible, ref, sens, thr, 256, regroup=True)
    assert kept.shape == (1, 16, 4, kt) and order.shape == (1, 16, 256) and order.dtype == torch.int64
    srt = order.sort(-1).values
    assert torch.equal(srt, torch.arange(256).expand_as(srt))                      # a permutation per head
    assert (order[..., :128] < 128).all() and (order[..., 128:] >= 128).all()      # rows stay in their block
    rep = kept128.repeat_interleave(2, dim=2)
    assert not (kept & ~rep).any()                                                 # subset of the executed 128 map
    assert torch.equal(kept[..., pt:], rep[..., pt:])                              # tail copied
    need = _row_need_ref(state, ref, sens, thr) & kept128[..., :pt].repeat_interleave(128, dim=2)
    row_kept = torch.gather(kept[..., :pt], 2, _groups(order)[..., None].expand(-1, -1, -1, pt))
    assert not (need & ~row_kept).any()                                            # every row's need is covered
    brute = torch.stack([torch.stack([need[0, i, order[0, i, g * 64:(g + 1) * 64]].any(0) for i in range(16)])
                         for g in range(4)], 1)
    assert torch.equal(kept[0, ..., :pt], brute)                                   # exact union per group
    counts = need.sum(-1).view(1, 16, 2, 128)
    sorted_counts = torch.gather(need.sum(-1), 2, order).view(1, 16, 2, 128)
    assert (sorted_counts[..., 1:] >= sorted_counts[..., :-1]).all()                # sorted by need count
    assert torch.equal(sorted_counts.sort(-1).values, counts.sort(-1).values)


def test_regroup_is_the_natural_map_when_rows_are_already_sorted():
    from experiments.numerical_qk_reuse.v27_dense_prefix import refine_q64
    pt, kt, thr = 8, 12, -3.87
    state = _state(pt, seed=6)
    # every row needs the same tiles -> stable sort keeps the natural order
    state.lognorm[...] = state.lognorm[..., :1].clone()
    ref = torch.tensor([[1.0, 1.0]])
    eligible = torch.ones(1, 16, 2, kt, dtype=torch.bool)
    eligible[..., :pt] = state.eligible.bool()
    skipped = torch.zeros_like(eligible)
    kept, order = refine_q64(state, skipped, eligible, ref, None, thr, 256, regroup=True)
    assert torch.equal(order, torch.arange(256).expand(1, 16, 256))
    assert torch.equal(kept, refine_q64(state, skipped, eligible, ref, None, thr, 256))


def test_regroup_without_prefix_and_fallback():
    from types import SimpleNamespace
    from experiments.numerical_qk_reuse.v27_dense_prefix import refine_q64
    state = SimpleNamespace(lognorm=torch.zeros(1, 16, 2, 0, 128))                 # no prefix tiles yet
    eligible = torch.ones(1, 16, 2, 6, dtype=torch.bool)
    skipped = torch.zeros_like(eligible)
    kept, order = refine_q64(state, skipped, eligible, torch.ones(1, 2), None, -3.87, 256, regroup=True)
    assert kept.all() and torch.equal(order, torch.arange(256).expand(1, 16, 256))
    assert refine_q64(state, skipped, eligible, torch.ones(1, 2), None, -3.87, 200, regroup=True) is None


def test_carried_64_row_map_extends_like_the_128_row_map():
    from experiments.numerical_qk_reuse.integration import carried_map, carried_q64
    g = torch.Generator().manual_seed(3)
    skipped = torch.rand(1, 16, 2, 30, generator=g) > .5
    eligible = torch.ones_like(skipped)
    kept64 = (~skipped).repeat_interleave(2, dim=2) & (torch.rand(1, 16, 4, 30, generator=g) > .3)
    prefix, keys = 1800, 1800 + 256 + 256
    new_s, new_e = carried_map(skipped, eligible, prefix, keys)
    out = carried_q64(kept64, prefix, new_s)
    old = prefix // 64
    assert out.shape == (1, 16, 4, new_s.shape[-1])
    assert torch.equal(out[..., :old], kept64[..., :old]) and out[..., old:].all()
    assert not (out & ~(new_e & ~new_s).repeat_interleave(2, dim=2)).any()         # subset of the carried 128 map
    with pytest.raises(ValueError):
        carried_q64(kept64[:, :, :2], prefix, new_s)


def test_q64r_config_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    cfg = v21.effective_config(dict(BASE), arm, scope, q_block=64, q_regroup=True, carry_first=True,
                               **_mainline(threshold_shift='minus_ln2'))
    v21.validate_effective(cfg, cfg['condition'])
    plain = v21.effective_config(dict(BASE), arm, scope, q_block=64, carry_first=True,
                                 **_mainline(threshold_shift='minus_ln2'))
    assert cfg['q_regroup'] is True and 'q_regroup' not in plain and cfg['fingerprint'] != plain['fingerprint']
    with pytest.raises(ValueError, match='q64r'):
        v21.effective_config(dict(BASE), arm, scope, q_regroup=True, **_mainline())


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('V27_FA4_OVERLAY'),
                    reason='requires CUDA and V27_FA4_OVERLAY')
def test_fa4_consumer_regrouped_rows_match_reference():
    from types import SimpleNamespace
    from experiments.numerical_qk_reuse.integration import Attention
    g = torch.Generator(device='cuda').manual_seed(11)
    keys = 4133
    kt = math.ceil(keys / 64)
    with torch.inference_mode():
        q = torch.randn(1, 256, 16, 512, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
        k = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 2, keys, 512, device='cuda', dtype=torch.bfloat16, generator=g)
        skipped = torch.zeros(1, 16, 2, kt, dtype=torch.bool, device='cuda')
        eligible = torch.ones_like(skipped)
        kept = torch.rand(1, 16, 4, kt, device='cuda', generator=g) > .6
        kept[..., 0] = True
        within = torch.argsort(torch.rand(1, 16, 2, 128, device='cuda', generator=g), -1)
        order = (within + torch.tensor([0, 128], device='cuda').view(1, 1, 2, 1)).reshape(1, 16, 256)
        fake = SimpleNamespace(consumer='fa4', trace=False, output_layout='model_major', _fa4_lists=[],
                               fa4_list_builds=0, _q64=[(skipped, eligible, kept, order)], q64_list_builds=0,
                               q64_regrouped_calls=0)
        got = Attention._consume(fake, q, k, v, skipped, eligible, 512 ** -.5, None, False).output   # [B,H,Q,D]
        assert fake.q64_list_builds == 1 and fake.q64_regrouped_calls == 1
    # reference: original row r uses the map of the 64-row group it was placed in
    row_map = torch.gather(kept, 2, _groups(order)[..., None].expand(-1, -1, -1, kt))   # [1,16,256,KT]
    h, hk = q.shape[1], k.shape[1]
    kr, vr = k.repeat_interleave(h // hk, 1).float(), v.repeat_interleave(h // hk, 1).float()
    s = torch.einsum('bhqd,bhkd->bhqk', q.float(), kr) * 512 ** -.5
    m = row_map.repeat_interleave(64, 3)[..., :keys]
    ref = torch.einsum('bhqk,bhkd->bhqd', s.masked_fill(~m, float('-inf')).softmax(-1), vr)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_q64_carry_is_opt_in_and_guarded():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    line = _mainline(threshold_shift='minus_ln2')
    cfg = v21.effective_config(dict(BASE), arm, scope, q_block=64, carry_first=True, q_carry64=True, **line)
    v21.validate_effective(cfg, cfg['condition'])
    plain = v21.effective_config(dict(BASE), arm, scope, q_block=64, carry_first=True, **line)
    assert cfg['q_carry64'] is True and 'q_carry64' not in plain and cfg['fingerprint'] != plain['fingerprint']
    for bad in (dict(q_block=64, q_carry64=True), dict(q_block=64, carry_first=True, q_regroup=True, q_carry64=True),
                dict(carry_first=True, q_carry64=True)):
        with pytest.raises(ValueError):
            v21.effective_config(dict(BASE), arm, scope, **bad, **line)
