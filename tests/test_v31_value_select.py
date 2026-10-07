"""CPU correctness gates for this study's value-aware v31 selectors.

Run: python -m pytest tests/test_v31_value_select.py -q      (CPU only, no CUDA, no Triton import)
"""
from __future__ import annotations

import math

import pytest
import torch

from experiments.numerical_qk_reuse import v31_value_select as vs

torch.manual_seed(20261007)


# --------------------------------------------------------------------------- helpers


def make_scores(h, rows, nk, g, device='cpu', dtype=torch.float64, dead=()):
    """Scaled masked logits [H, rows, nk]: normal, structural -inf on whole rows/keys in ``dead``."""
    s = torch.randn(h, rows, nk, device=device, dtype=dtype)
    for r, c in dead:
        s[r, :, c] = float('-inf')
    return s


def build_stats(h=4, nk=256, rows=128, g=2, chunk=2, dead=(), device='cpu', dtype=torch.float64,
                nu_scale=1.0):
    scores = make_scores(h, rows, nk, g, device=device, dtype=dtype, dead=dead)
    sketch = torch.randn(h // g, nk, vs.RANK, device=device, dtype=dtype) / math.sqrt(vs.RANK)
    row_ok = torch.ones((h, rows), dtype=torch.bool, device=device)
    z, mu = vs.tile_statistics(scores, sketch, row_ok, chunk_tiles=chunk)
    nu = ((sketch * sketch).sum(-1).mean(-1).sqrt() * nu_scale)[torch.arange(h, device=device) // g][:, None] \
        .expand(h, rows).contiguous()
    return vs.ValueStats(z=z, mu=mu, nu=nu, rows=row_ok, n=rows, kv_heads=h // g, group=g,
                         prefix_tiles=nk // 64, total_tiles=nk // 64), scores, sketch


def true_stats(scores, sketch, h):
    """(alpha, mu, O) of the full support from the raw masked scores, in tile-major layout.

    Independent of the module: the tile mass is the shifted within-tile exponential sum and mu is
    the ATTENTION-WEIGHTED within-tile mean of the projected values (not a pooled mean)."""
    n_pad, nk = scores.shape[1], scores.shape[2]
    tiles = -(-nk // 64)
    s = scores.reshape(h, n_pad, tiles, 64)
    # one ROW-level shift for the tile masses (proportional to sum_u exp(s_iu)), and one TILE-level
    # shift for the within-tile weights (mu is a normalized within-tile mean, so the shift cancels)
    row_max = s.amax(-1).amax(-1, keepdim=True)
    row_shifted = torch.where(torch.isfinite(s), torch.exp(s - row_max[..., None]),
                              torch.zeros_like(s))
    mass = row_shifted.sum(-1)                                      # [H, N, T], proportional to Z
    alpha = (mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)).transpose(1, 2)
    tile_max = s.amax(-1, keepdim=True)
    within = torch.where(torch.isfinite(s), torch.exp(s - tile_max), torch.zeros_like(s))
    per_head = sketch[torch.arange(h) // (h // sketch.shape[0])].reshape(h, 1, nk, vs.RANK)
    sk = per_head[:, :, :tiles * 64, :].reshape(h, 1, tiles, 64, vs.RANK)
    mu = (torch.einsum('hntc,hntcr->hntr', within, sk)
          / within.sum(-1).clamp_min(1e-12)[..., None]).transpose(1, 2)
    return alpha, mu, (alpha[..., None] * mu).sum(1)


def partial_last_block(h=2, n=160, g=2, device='cpu'):
    """Rows beyond ``n`` are a padded 128-block tail: they must never win an aggregation."""
    n_pad = -(-n // 128) * 128
    nk = 128
    scores = torch.randn(h, n_pad, nk, device=device, dtype=torch.float64)
    scores[:, n:, :] = float('-inf')
    sketch = torch.randn(h // g, nk, vs.RANK, device=device, dtype=torch.float64)
    row_ok = torch.zeros((h, n_pad), dtype=torch.bool, device=device)
    row_ok[:, :n] = True
    z, mu = vs.tile_statistics(scores, sketch, row_ok)
    nu = (sketch * sketch).sum(-1).mean(-1).sqrt()[torch.arange(h, device=device) // g][:, None] \
        .expand(h, n_pad).contiguous()
    return vs.ValueStats(z=z, mu=mu, nu=nu, rows=row_ok, n=n, kv_heads=h // g, group=g,
                         prefix_tiles=nk // 64, total_tiles=nk // 64), sketch


# --------------------------------------------------------------------------- projection identity


def test_gaussian32_bank_is_deterministic_and_pinned():
    a, ra = vs.gaussian32_bank(3, 2, 8, 'cpu')
    b, rb = vs.gaussian32_bank(3, 2, 8, 'cpu')
    assert a.shape == (2, 8, vs.RANK) and a.dtype == torch.float32
    assert torch.equal(a, b) and ra == rb
    assert {r['sha256'] for r in ra} == {vs.bank_digest(a[i]) for i in range(2)}
    # a different layer / KV head / width must give a different matrix
    c, _ = vs.gaussian32_bank(4, 2, 8, 'cpu')
    assert not torch.equal(a, c)
    d, _ = vs.gaussian32_bank(3, 2, 8, 'cpu', seed=2718)
    assert not torch.equal(a, d)
    # N(0, 1/32): the entry variance is 1/32
    assert abs(float(a.var()) - 1.0 / vs.RANK) < 0.02


def test_projection_and_reference_scale():
    v = torch.randn(1, 2, 5, 4, dtype=torch.float64)
    bank = torch.randn(2, 4, vs.RANK, dtype=torch.float64)
    z = vs.project_values(v, bank)
    assert torch.allclose(z[0, 0, 2], v[0, 0, 2] @ bank[0])
    valid = torch.ones(1, 2, 5, dtype=torch.bool)
    valid[0, 0, 3] = False
    ref = vs.valid_kv_reference(v, valid)
    rms = lambda head, keep: float((v[0, head][keep].norm(dim=-1).square().sum() / int(keep.sum())) ** 0.5)
    # the reference is the RMS of the VALID V norms, per native KV head: dropping key 3 of head 0
    # must change head 0's reference and leave head 1's alone
    assert float(ref[0]) == pytest.approx(rms(0, valid[0, 0]))
    assert float(ref[0]) != pytest.approx(rms(0, torch.ones(5, dtype=torch.bool)))
    assert float(ref[1]) == pytest.approx(rms(1, valid[0, 1]))
    assert float(ref[1]) != pytest.approx(rms(0, valid[0, 0]))


# --------------------------------------------------------------------------- statistics


def test_tile_statistics_match_a_direct_reference():
    h, nk, g = 2, 320, 2
    scores, sketch = make_scores(h, 128, nk, g), torch.randn(h // g, nk, vs.RANK, dtype=torch.float64)
    row_ok = torch.ones((h, 128), dtype=torch.bool)
    z, mu = vs.tile_statistics(scores, sketch, row_ok, chunk_tiles=3)
    tiles = -(-nk // 64)
    p = torch.softmax(scores.reshape(h, 128, tiles, 64), dim=-1)                  # [H, N, T, 64]
    zz = scores.reshape(h, 128, tiles, 64).logsumexp(-1).transpose(1, 2)
    torch.testing.assert_close(z, zz)
    per_head = sketch[torch.arange(h) // g].reshape(h, 1, nk, vs.RANK)          # [H, 1, nk, 32]
    sk = per_head[:, :, :tiles * 64, :].reshape(h, 1, tiles, 64, vs.RANK)
    ref = torch.einsum('hntc,hntcr->hntr', p, sk).transpose(1, 2)
    torch.testing.assert_close(mu, ref)


def test_tile_statistics_handle_dead_rows_and_tiles():
    h, nk, g = 2, 192, 2
    # a whole row with no legal key, and a whole 64-key tile illegal for every row
    scores = make_scores(h, 128, nk, g)
    scores[0, :, :] = float('-inf')
    scores[1, :, 64:128] = float('-inf')
    sketch = torch.randn(h // g, nk, vs.RANK, dtype=torch.float64)
    row_ok = torch.ones((h, 128), dtype=torch.bool)
    z, mu = vs.tile_statistics(scores, sketch, row_ok)
    assert torch.isneginf(z[0]).all(), 'a row with no legal key has no tile mass'
    assert torch.isneginf(z[1, 1]).all(), 'a wholly masked tile has no mass'
    assert torch.isfinite(z[1, 0]).all() and torch.isfinite(z[1, 2]).all()
    assert torch.isfinite(mu).all(), 'a zero-mass tile leaves mu at 0, never NaN'


def test_append_tail_tiles_extends_the_full_support():
    nk, h, g, n_tail = 128, 2, 2, 96
    scores, sketch = make_scores(h, 128, nk, g), torch.randn(h // g, nk, vs.RANK, dtype=torch.float64)
    row_ok = torch.ones((h, 128), dtype=torch.bool)
    z, mu = vs.tile_statistics(scores[:, :, :64], sketch[:, :64], row_ok)
    tail_z, tail_mu = vs.tile_statistics(scores[:, :, 64:], sketch[:, 64:], row_ok)
    zz, mm = vs.append_tail_tiles(z, mu, scores[:, :, 64:], sketch, total_tiles=2)
    assert zz.shape == (h, 2, 128) and mm.shape == (h, 2, 128, vs.RANK)
    torch.testing.assert_close(zz[:, :1], z)
    torch.testing.assert_close(zz[:, 1:], tail_z)
    torch.testing.assert_close(mm[:, 1:], tail_mu)
    with pytest.raises(ValueError):
        vs.append_tail_tiles(z, mu, scores[:, :, 64:], sketch, total_tiles=3)


# --------------------------------------------------------------------------- aggregation


def test_block_max_ignores_padded_rows():
    stats, _ = partial_last_block(h=2, n=160)
    per_row = torch.arange(2 * 256, dtype=torch.float64).reshape(2, 1, 256)
    agg = vs.block_max(per_row, stats.rows)
    assert agg.shape == (2, 1, 2)
    assert int(agg[0, 0, 0]) == 127                       # rows 0..127 are all real
    assert int(agg[0, 0, 1]) == 159                       # only rows 128..159 are real, so the
    assert int(agg[0, 0, 1]) != 255                       # padding rows 160..255 cannot win


def test_block_max_all_padding_is_minus_inf():
    rows = torch.zeros((1, 128), dtype=torch.bool)
    agg = vs.block_max(torch.ones((1, 1, 128)), rows)
    assert torch.isneginf(agg).all()


# --------------------------------------------------------------------------- V1 / V2


def test_v1_and_v2_match_a_simple_sequential_reference():
    """The batched reference must equal a plain per-unit sequential replay under identical decisions."""
    stats, scores, sketch = build_stats(h=4, nk=320, rows=128, g=2)
    budget, threshold = 3, 0.05
    mass = stats.finite_mass()                                             # [H, PT, N]
    for keep_mass, selector in ((False, 'v1'), (True, 'v2')):
        keep, _ = vs.scan_online(stats, budget, threshold, keep_mass)
        for h in range(stats.heads):
            pt = stats.prefix_tiles
            running = torch.zeros(128, dtype=mass.dtype)
            num = torch.zeros(128, vs.RANK, dtype=mass.dtype)
            optional, has_support = budget, False
            for i in range(pt):
                m = mass[h, i]
                eta = torch.where(running + m > 0, m / (running + m).clamp_min(1e-12),
                                  torch.zeros_like(m))
                est = num / running.clamp_min(1e-12)[:, None]
                rho = float((eta * (stats.mu[h, i] - est).norm(dim=-1)
                             / stats.nu[h].clamp_min(1e-12)).max())
                remaining = pt - i
                if optional <= 0:
                    take = False
                elif optional >= remaining:
                    take = True
                else:
                    take = bool(rho >= threshold) or not has_support
                if take:
                    optional -= 1
                    has_support = True
                    num = ((1 - eta)[:, None] * est + eta[:, None] * stats.mu[h, i]) * (running + m)[:, None]
                elif keep_mass:
                    num = est * (running + m)[:, None]
                running = running + (m if (take or keep_mass) else torch.zeros_like(m))
                assert bool(keep[0, h, 0, i]) == take, (selector, h, i, rho)


def test_v1_excludes_skipped_mass_but_v2_keeps_it():
    """V1's routing state only ever contains RETAINED support; V2's denominator also contains the
    mass of skipped blocks, while its normalized routing output stays put."""
    rows = torch.ones((1, 128), dtype=torch.bool)
    z = torch.zeros((1, 3, 128))
    mu = torch.zeros((1, 3, 128, vs.RANK), dtype=torch.float64)
    mu[0, 0, :, 0] = 1.0
    mu[0, 1, :, 0] = 4.0
    mu[0, 2, :, 0] = 1.0
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=3,
                       total_tiles=3)
    keep1, _, v1 = vs.scan_online(st, 2, 2.0, False, return_state=True)
    keep2, _, v2 = vs.scan_online(st, 2, 2.0, True, return_state=True)
    assert keep1[0, 0, 0].flatten().tolist() == [True, False, True]
    assert keep2[0, 0, 0].flatten().tolist() == [True, False, True]
    assert float(v1['running'][0, 0]) == pytest.approx(2.0), 'V1 counted only the two retained tiles'
    assert float(v2['running'][0, 0]) == pytest.approx(3.0), 'V2 also counted the skipped tile'
    # V1's routing output is the retained-support mean; V2's is held at the value it had at the
    # skip, i.e. it does NOT absorb the skipped block's direction
    assert float(v1['o_hat'][0, 0, 0]) == pytest.approx(1.0)
    assert float(v2['o_hat'][0, 0, 0]) == pytest.approx(1.0)
    # and the wrongly-updated variants would differ
    wrong_numerator = v2['numerator'][0, 0, 0] + 1.0 * 4.0            # o_hat * Z_ij * mu_ij
    assert not math.isclose(float(wrong_numerator / v2['running'][0, 0]), 1.0)


def test_v2_skip_advances_the_denominator_only():
    """V2's internal routing output must be UNCHANGED by a skip even when the running max moves."""
    rows = torch.ones((1, 128), dtype=torch.bool)
    z = torch.zeros((1, 2, 128))
    z[0, 1] = 3.0                                     # the second tile dominates the mass
    mu = torch.zeros((1, 2, 128, vs.RANK), dtype=torch.float64)
    mu[0, 0, :, 0] = 0.2
    mu[0, 1, :, 0] = 5.0
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=2,
                       total_tiles=2)
    keep, _, state = vs.scan_online(st, 1, 1e9, True, return_state=True)   # keep only tile 0
    assert bool(keep[0, 0, 0, 0]) and not bool(keep[0, 0, 0, 1])
    # the mass is scaled by the row's largest tile log-mass, so it is exp(-3) and 1
    assert float(state['running'][0, 0]) == pytest.approx(1.0 + math.exp(-3.0))
    torch.testing.assert_close(state['o_hat'][0, :128], mu[0, 0])
    # the WRONG update (advance only the denominator of a fixed numerator) does differ
    wrong = state['numerator'][0, :128] / (1.0 + math.exp(3.0))
    assert not torch.allclose(wrong, mu[0, 0])


def test_equal_mass_regression_v2_state_vs_masked_attention():
    """REGRESSION GATE. mu_A = mu_B = a, mu_C = b, skip B and retain C.

    V2's internal routing state can end at (2a + b)/3 while ordinary reused attention over {A, C}
    produces (a + b)/2. The two values must be distinguishable and NO executor compensation may
    turn one into the other."""
    rows = torch.ones((1, 128), dtype=torch.bool)
    z = torch.zeros((1, 3, 128))
    mu = torch.zeros((1, 3, 128, vs.RANK), dtype=torch.float64)
    a = torch.zeros(vs.RANK, dtype=torch.float64)
    b = torch.zeros(vs.RANK, dtype=torch.float64)
    b[0] = 6.0
    mu[0, 0, :, :] = a
    mu[0, 1, :, :] = a
    mu[0, 2, :, :] = b
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=3,
                       total_tiles=3)
    # V2 internals, replayed with the recorded decisions {A keep, B skip, C keep}
    m = torch.ones(128, dtype=torch.float64)
    P = m[:, None] * mu[0, 0]                             # after A: o_hat = a, Z = 1
    Z = m
    P = P + (P / Z[:, None]) * m[:, None]                  # the skip holds o_hat = a
    Z = Z + m
    o_hat = (P / Z[:, None])[:, 0]
    eta = m / (Z + m)                                       # the retain is the convex update
    P = P + eta[:, None] * (Z + m)[:, None] * mu[0, 2]
    Z = Z + m
    state = (P / Z[:, None])[0]
    assert torch.allclose(o_hat[0], a), 'the skip really left o_hat unchanged'
    alpha, c, ref, _ = vs.full_support(st)
    # the keep map the selector produced under this configuration: {A, C}, B skipped
    keep, _ = vs.scan_online(st, 2, 2.0, True)
    assert keep[0, 0, 0].flatten().tolist() == [True, False, True]
    # ordinary reused attention over the SAME {A, C} map renormalizes only the retained mass
    out, _ = vs.masked_attention_sketch(alpha, c, keep)
    reused = out[0, 0]
    assert torch.allclose(state, (2 * a + b) / 3)
    assert torch.allclose(reused, (a + b) / 2)
    assert not torch.allclose(state, reused)
    assert float((state - reused).abs().max()) > 0.5


def test_first_valid_support_is_always_kept():
    rows = torch.ones((1, 128), dtype=torch.bool)
    z = torch.full((1, 8, 128), -20.0)
    mu = torch.zeros((1, 8, 128, vs.RANK))
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=8,
                       total_tiles=8)
    keep, c = vs.scan_online(st, 4, 1e9, False)
    assert bool(keep[0, 0, 0, 0]), 'the first valid support is force-kept'
    assert int(keep[0, 0, 0, :].sum()) == 4           # and the fixed quota still holds exactly


def test_quota_forced_skip_and_forced_keep_and_ties():
    rows = torch.ones((1, 128), dtype=torch.bool)
    z = torch.zeros((1, 6, 128))
    mu = torch.zeros((1, 6, 128, vs.RANK))
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=6,
                       total_tiles=6)
    # every rho == 0, so `>= 0` keeps a boundary tie: 1 forced first support + 4 threshold keeps
    keep, c = vs.scan_online(st, 2, 0.0, False)
    assert int(keep[0, 0, 0, :6].sum()) == 2
    assert c.forced_keep == 1 and c.threshold_keeps == 1 and c.forced_skip == 4
    # a threshold nothing clears: the quota force-fills the last two optional slots
    keep, c = vs.scan_online(st, 5, 0.5, False)
    assert int(keep[0, 0, 0, :6].sum()) == 5
    assert c.threshold_keeps == 0 and c.forced_keep == 5 and c.forced_skip == 1
    # a threshold everything clears: the quota force-skips the tail
    keep, c = vs.scan_online(st, 2, -0.5, False)
    assert int(keep[0, 0, 0, :6].sum()) == 2
    assert c.threshold_keeps == 1 and c.forced_skip == 4


def test_partial_tiles_and_empty_rows():
    stats, _ = partial_last_block(h=2, n=160)
    keep, _ = vs.scan_online(stats, 1, -1e30, False)
    assert keep.shape == (1, 2, 2, 2)
    assert int(keep[0, :, :, :].sum()) == 2 * 2         # one optional tile per unit
    stats2, _sk = partial_last_block(h=2, n=64)
    keep2, _ = vs.scan_online(stats2, 1, 0.0, True)
    assert int(keep2[0, :, 0, :2].sum()) == 2


def test_protected_tiles_are_mandatory():
    rows = torch.ones((1, 128), dtype=torch.bool)
    z = torch.zeros((1, 8, 128))
    mu = torch.zeros((1, 8, 128, vs.RANK))
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=8,
                       total_tiles=8)
    prot = vs.protect_map(8, 8, 128, 128, 1, 1, z.device)
    keep, c = vs.scan_online(st, 4, -1e30, False, protect=prot)
    assert bool(keep[0, 0, 0, 0]) and bool(keep[0, 0, 0, 7])
    # sink tiles 0/1 and recent tiles 6/7 are mandatory and charged against the budget of 4
    assert int(keep[0, 0, 0, :8].sum()) == 4
    keep, _ = vs.scan_online(st, 1, -1e30, False, protect=prot)
    # a budget below the protected count still keeps every protected tile and nothing else
    assert int(keep[0, 0, 0, :8].sum()) == 4


def test_fixed_k_quota_is_never_exceeded():
    stats, _s, _k = build_stats(h=4, nk=512, rows=128, g=4)
    for budget in (1, 2, 5, 8):
        keep, _ = vs.scan_online(stats, budget, -1e30, True)
        assert int(keep[:, :, :].sum()) == 4 * budget
        keep, _ = vs.scan_online(stats, budget, 1e30, True)
        assert int(keep[:, :, :].sum()) == 4 * budget


def test_gqa_mapping_broadcasts_the_kv_head():
    """Every query head of a GQA group must see the same reference scale."""
    h, g = 6, 3
    nu_kv = torch.arange(1.0, g + 1.0, dtype=torch.float64)
    rows = torch.ones((h, 128), dtype=torch.bool)
    z = torch.zeros((h, 4, 128))
    mu = torch.zeros((h, 4, 128, vs.RANK))
    nu = nu_kv[torch.arange(h) // (h // g)][:, None].expand(h, 128).contiguous()
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=g, group=h // g, prefix_tiles=4,
                       total_tiles=4)
    # group size 2: heads 0/1 share KV head 0, heads 2/3 share KV head 1, heads 4/5 KV head 2
    assert torch.equal(st.nu[0], st.nu[1]) and torch.equal(st.nu[2], st.nu[3])
    assert not torch.equal(st.nu[0], st.nu[2])
    assert [float(st.nu[i, 0]) for i in range(6)] == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


# --------------------------------------------------------------------------- V3a / V3b


def test_masked_attention_sketch_matches_direct_recomputation():
    """``O_i(S)`` from the cached (alpha, c) must equal a direct masked recomputation."""
    stats, scores, sketch = build_stats(h=2, nk=256, rows=128, g=2)
    alpha, c, ref, live = vs.full_support(stats)
    kt = stats.total_tiles
    a_true, mu_true, ref_true = true_stats(scores, sketch, 2)
    torch.testing.assert_close(alpha, a_true)
    torch.testing.assert_close(stats.mu, mu_true)
    torch.testing.assert_close(ref, ref_true)
    keep = torch.rand((1, 2, 1, kt)) > 0.5
    out, den = vs.masked_attention_sketch(alpha, c, keep)
    w = keep[0, :, 0].to(alpha.dtype)                                     # [H, KT]
    wn = (c * w[:, :, None, None]).sum(1)                                 # [H, N, R]
    wd = (alpha * w[:, :, None]).sum(1).clamp_min(1e-12)                   # [H, N]
    torch.testing.assert_close(out, wn / wd[..., None])
    torch.testing.assert_close(den, wd)
    # the full-support objective of the all-kept map is exactly zero
    full = torch.ones((1, 2, 1, kt), dtype=torch.bool)
    torch.testing.assert_close(vs.masked_attention_sketch(alpha, c, full)[0], ref)


def test_v3a_matches_a_direct_singleton_deletion_ranking():
    stats, scores, sketch = build_stats(h=2, nk=256, rows=128, g=2)
    budget = 3
    keep, cnt, _ = vs.v3a_select(stats, budget)
    alpha, cc, ref, _ = vs.full_support(stats)
    a_true, mu_true, _ = true_stats(scores, sketch, 2)
    d = (alpha * (mu_true - ref[:, None, :, :]).norm(dim=-1)
         / ((1 - alpha).clamp_min(1e-12) * stats.nu[:, None, :]))          # [H, KT, N]
    pt = stats.prefix_tiles
    for h in range(2):
        want = set(torch.topk(d[h, :pt].amax(-1), budget).indices.tolist())
        got = set(torch.nonzero(keep[0, h, 0, :pt]).flatten().tolist())
        assert want == got, (h, sorted(want), sorted(got))
    assert cnt.evaluations == 2 * 1 * pt and cnt.passes == 1


def test_v3a_beats_the_mass_only_ranking_on_a_direction_separated_case():
    """A mass-only selector cannot see value direction.

    Four tiles, masses 4:3:2:1, projected means +a, +a, -a, -a. A mass-only top-2 keeps the two +a
    tiles and reproduces +a, far from the full-support mean; the value-aware objective prefers a set
    spanning both directions."""
    rows = torch.ones((1, 128), dtype=torch.bool)
    kt = 4
    a = torch.zeros(vs.RANK, dtype=torch.float64)
    a[0] = 1.0
    z = torch.log(torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float64))[:, :, None] \
        .expand(1, kt, 128).contiguous()
    mu = torch.zeros((1, kt, 128, vs.RANK), dtype=torch.float64)
    mu[0, 0, :, :] = a
    mu[0, 1, :, :] = a
    mu[0, 2, :, :] = -a
    mu[0, 3, :, :] = -a
    nu = torch.ones((1, 128), dtype=torch.float64)
    st = vs.ValueStats(z=z, mu=mu, nu=nu, rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=kt,
                       total_tiles=kt)
    keep, _, aux = vs.v3a_select(st, 2)
    chosen = set(torch.nonzero(keep[0, 0, 0]).flatten().tolist())
    mass_only = torch.zeros((1, 1, 1, kt), dtype=torch.bool)
    mass_only[0, 0, 0, :2] = True                                         # the mass-only top-2
    assert chosen != {0, 1}
    obj_v, _ = vs.objective(st, keep, **aux)
    obj_m, _ = vs.objective(st, mass_only, **aux)
    assert float(obj_v[0, 0]) < float(obj_m[0, 0]), (float(obj_v[0, 0]), float(obj_m[0, 0]))


def test_v3b_exact_matches_a_direct_cumulative_greedy():
    stats, scores, sketch = build_stats(h=2, nk=192, rows=128, g=2)
    budget = 3
    keep, cnt, aux = vs.v3b_greedy(stats, budget, exact=True)
    alpha, cc, ref, _ = vs.full_support(stats)
    pt = stats.prefix_tiles
    # direct replay of the cumulative rule over one head
    h = 0
    live = list(range(pt))
    a = alpha[h, :pt, 0].clone()                                  # [PT] for row 0
    g = (cc[h, :pt, 0] - alpha[h, :pt, 0, None] * ref[h, 0]).clone()
    nu = float(stats.nu[h, 0])
    while len(live) > budget:
        A = a[live].sum()
        r = g[live].sum(0)
        best, best_score = None, math.inf
        for j in live:
            A2 = A - a[j]
            score = float((r - g[j]).norm() / (A2 * nu))
            if score < best_score:
                best, best_score = j, score
        live.remove(best)
    got = set(torch.nonzero(keep[0, h, 0, :pt]).flatten().tolist())
    assert got == set(live)
    assert cnt.exact is True and cnt.passes == pt - budget


def test_v3b_shortlist_is_an_approximation_and_is_named():
    stats, _s, _k = build_stats(h=2, nk=384, rows=128, g=2)
    exact_keep, c_exact, aux = vs.v3b_greedy(stats, 2, exact=True)
    short_keep, c_short, _ = vs.v3b_greedy(stats, 2, exact=False, shortlist=4)
    assert c_exact.exact is True and c_short.exact is False
    assert c_short.extra['shortlist'] == 4 and c_short.passes <= c_exact.passes
    assert short_keep.shape == exact_keep.shape


def test_v3b_drop_and_refine_is_an_approximation_and_is_named():
    stats, _s, _k = build_stats(h=2, nk=384, rows=128, g=2)
    drop_keep, c_drop, _ = vs.v3b_greedy(stats, 2, exact=False, drop_fraction=0.25)
    assert c_drop.exact is False and c_drop.passes <= stats.prefix_tiles - 2
    assert int(drop_keep[:, :, :].sum()) == 2 * 2


def test_v3_never_drops_a_mandatory_tile_and_respects_the_budget():
    stats, _s, _k = build_stats(h=2, nk=256, rows=128, g=2)
    budget = 4
    for keep in (vs.v3a_select(stats, budget)[0], vs.v3b_greedy(stats, budget)[0],
                 vs.v3b_greedy(stats, budget, exact=False, drop_fraction=0.25)[0],
                 vs.v3b_greedy(stats, budget, exact=False, shortlist=3)[0]):
        assert keep.shape == (1, 2, 1, stats.total_tiles)
        assert bool(keep[..., stats.prefix_tiles:].all()), 'canvas/boundary tiles are never dropped'
        assert int(keep[:, :, :stats.prefix_tiles].sum()) == budget * 2
    with_protect = vs.protect_map(stats.prefix_tiles, stats.total_tiles, 128, 128, 2, 1, stats.z.device)
    keep = vs.v3a_select(stats, budget, protect=with_protect)[0]
    assert bool(keep[0, 0, 0, 0]) and bool(keep[0, 0, 0, stats.prefix_tiles - 1])
    assert int(keep[0, 0, 0, :].sum()) == budget


def test_identical_maps_give_identical_objective():
    stats, _s, _k = build_stats(h=2, nk=256, rows=128, g=2)
    keep, _, aux = vs.v3a_select(stats, 3)
    a, _ = vs.objective(stats, keep, **aux)
    b, _ = vs.objective(stats, keep.clone(), **aux)
    assert torch.equal(a, b)


def test_v3b_is_a_greedy_heuristic_not_a_subset_solver():
    """Documented limitation: the exact backward greedy may end above another set's objective."""
    stats, _s, _k = build_stats(h=1, nk=512, rows=128, g=1)
    keep, _, aux = vs.v3b_greedy(stats, 2, exact=True)
    best = None
    pt = stats.prefix_tiles
    for i in range(pt):
        for j in range(i + 1, pt):
            alt = torch.zeros((1, 1, 1, pt), dtype=torch.bool)
            alt[0, 0, 0, i] = True
            alt[0, 0, 0, j] = True
            v, _ = vs.objective(stats, alt, **aux)
            if best is None or float(v[0, 0]) < best:
                best = float(v[0, 0])
    got, _ = vs.objective(stats, keep, **aux)
    assert float(got[0, 0]) >= best - 1e-12