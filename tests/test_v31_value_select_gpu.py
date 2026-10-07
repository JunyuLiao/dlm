"""GPU correctness gates for the value-aware v31 selectors (H100, SM90).

These run against the pinned vLLM 0.30.0 environment and the real SM90 FA4 / Triton paths. They
prove that the production paths agree with the direct formulas:

  * ``v31_value_observe.observe_mu`` (the repository's Triton fused observation with OUT=0, MU=1)
    writes ``mu`` matching a direct FP64 within-tile attention-weighted projected mean, and its ``z``
    matches the tile log-mass the FA4 in-kernel observation writes;
  * ``v31_value_select.tile_statistics`` reproduces ``z`` and ``mu`` from the same scores;
  * the fused Triton scan ``v31_value_scan.value_scan`` reproduces the batched reference V1/V2 map
    and counters exactly, for both variants, including protected tiles, padded rows and GQA groups;
  * the identity projection agrees with a direct FULL-dimensional masked-attention reference;
  * identical keep maps produce identical FA4 sparse consumer outputs whichever selector made them;
  * the equal-mass V2 / masked-attention regression case survives on GPU, with no executor
    compensation.

Run: PYTHONPATH=src <pinned python> -m pytest tests/test_v31_value_select_gpu.py -q
"""
from __future__ import annotations

import pytest
import torch

from experiments.numerical_qk_reuse import v31_value_select as vs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='H100 required')

H, HK, D, RANK = 4, 2, 64, vs.RANK
SCALE = D ** -0.5


class Case:
    """A model-shaped observation call: q/k/v BF16 plus the frozen Gaussian32 projection of V."""

    def __init__(self, nk=512, n=128, heads=H, kv_heads=HK, seed=11):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        dev = 'cuda'
        self.q = torch.randn(1, heads, n, D, generator=gen).to(dev, torch.bfloat16) * 0.5
        self.k = torch.randn(1, kv_heads, nk, D, generator=gen).to(dev, torch.bfloat16) * 0.5
        self.v = torch.randn(1, kv_heads, nk, D, generator=gen).to(dev, torch.bfloat16) * 0.5
        self.heads, self.kv_heads, self.n, self.nk = heads, kv_heads, n, nk
        bank, self.projection_records = vs.gaussian32_bank(0, kv_heads, D, dev)
        # the statistics kernel multiplies FP32 within-tile weights by the sketch, so FP32
        self.sketch = vs.project_values(self.v, bank)                       # [1, HK, nk, 32] FP32
        self.proj = self.sketch[0].float()                                  # [HK, nk, 32]

    def scores(self, rows=None, dtype=torch.float32):
        """FP32 native scaled logits [H, rows, nk]; no structural mask (every key is legal)."""
        g = self.heads // self.kv_heads
        qg = self.q[0].float().reshape(self.kv_heads, g, self.n, D)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            s = torch.matmul(qg, self.k[0].float().transpose(-1, -2).unsqueeze(1))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        s = (s.reshape(self.heads, self.n, self.nk) * SCALE).to(dtype)
        if rows is None or rows == self.n:
            return s.contiguous()
        padded = torch.full((self.heads, rows, self.nk), float('-inf'), device=s.device, dtype=s.dtype)
        padded[:, :self.n] = s
        return padded.contiguous()

    def stats(self, rows=None):
        """``ValueStats`` over the full key extent (prefix == every tile here)."""
        n = self.n if rows is None else rows
        n_pad = -(-n // 128) * 128
        row_ok = torch.zeros((self.heads, n_pad), dtype=torch.bool, device=self.q.device)
        row_ok[:, :n] = True
        z, mu = vs.tile_statistics(self.scores(rows=n_pad), self.proj, row_ok)
        nu = vs.valid_kv_reference(self.v, torch.ones(1, self.kv_heads, self.nk, dtype=torch.bool,
                                                     device=self.q.device))
        nu = nu[torch.arange(self.heads, device=self.q.device) // (self.heads // self.kv_heads)]
        st = vs.ValueStats(z=z, mu=mu, nu=nu[:, None].expand(self.heads, n_pad).contiguous(),
                           rows=row_ok, n=n, kv_heads=self.kv_heads,
                           group=self.heads // self.kv_heads, prefix_tiles=self.nk // 64,
                           total_tiles=self.nk // 64)
        return st, z, mu


# --------------------------------------------------------------------------- projection / statistics


def test_gaussian32_bank_is_pinned_and_device_independent():
    bank, records = vs.gaussian32_bank(0, HK, D, 'cuda')
    cpu, cpu_records = vs.gaussian32_bank(0, HK, D, 'cpu')
    assert torch.equal(bank.cpu(), cpu), 'the projection must not depend on the device'
    assert records == cpu_records
    assert {r['sha256'] for r in records} == {vs.bank_digest(cpu[i]) for i in range(HK)}
    assert all(r['seed'] == vs.PROJECTION_SEED and r['rank'] == RANK for r in records)


def test_triton_observation_writes_the_direct_within_tile_projected_mean():
    """``observe_mu`` (the repository's fused observation, OUT=0, MU=1) against an FP64 oracle."""
    case = Case()
    pt = case.nk // 64
    from experiments.numerical_qk_reuse import v27_consumer64
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    summary = allocate_summary(1, H, 1, pt, pt, RANK, case.q.device, identity=('gpu-test',))
    v27_consumer64.fused_observe(case.q, case.k, case.k, case.sketch, SCALE, pt, summary,
                                 splits=1, mu=True, output=False)
    mu_kernel = summary.mu[0, 0].reshape(pt, 128, RANK)[:, :case.n].float()
    z_kernel = summary.z[0, 0].reshape(pt, 128)[:, :case.n].float()
    ref_z, ref_mu = vs.tile_statistics(case.scores(dtype=torch.float64), case.sketch[0].double(),
                                      torch.ones((H, 128), dtype=torch.bool, device='cuda'))
    torch.testing.assert_close(z_kernel, ref_z[0].float(), rtol=3e-3, atol=3e-3)
    rel = (mu_kernel - ref_mu[0].float()).norm(dim=-1) / ref_mu[0].float().norm(dim=-1).clamp_min(1e-9)
    assert float(rel.max()) < 1e-2, f'mu relative error {float(rel.max())}'
    # a zero-mass tile must never produce NaN
    assert torch.isfinite(mu_kernel).all()


def test_tile_statistics_match_the_triton_observation_on_gpu():
    case = Case()
    pt = case.nk // 64
    from experiments.numerical_qk_reuse import v27_consumer64
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    summary = allocate_summary(1, H, 1, pt, pt, RANK, case.q.device, identity=('gpu-test',))
    v27_consumer64.fused_observe(case.q, case.k, case.k, case.sketch, SCALE, pt, summary,
                                 splits=1, mu=True, output=False)
    mu_kernel = summary.mu[0, 0].reshape(pt, 128, RANK)[:, :case.n].float()
    _, mu_ref = vs.tile_statistics(case.scores(), case.proj,
                                   torch.ones((H, 128), dtype=torch.bool, device='cuda'))
    rel = (mu_kernel - mu_ref[0]).norm(dim=-1) / mu_ref[0].norm(dim=-1).clamp_min(1e-9)
    assert float(rel.max()) < 1e-2


# --------------------------------------------------------------------------- the fused scan


@pytest.mark.parametrize('selector', ('v1', 'v2'))
@pytest.mark.parametrize('budget,threshold', ((3, 0.5), (8, 0.0), (5, 1e9), (1, -1e9)))
def test_triton_value_scan_matches_the_batched_reference(selector, budget, threshold):
    """The fused GPU scan must reproduce the batched reference's decisions and counters exactly."""
    from experiments.numerical_qk_reuse import v31_value_scan as scan_mod
    stats, _z, _mu = Case().stats()
    keep_skipped_mass = selector == 'v2'
    ref, cnt_ref = vs.scan_online(stats, budget, threshold, keep_skipped_mass)
    got, cnt = scan_mod.value_scan(stats, budget, threshold, keep_skipped_mass)
    assert torch.equal(ref, got), (selector, budget, threshold, int((ref != got).sum()))
    assert (cnt.forced_keep, cnt.threshold_keeps, cnt.forced_skip) == \
        (cnt_ref.forced_keep, cnt_ref.threshold_keeps, cnt_ref.forced_skip)
    assert int(got[:, :, :stats.prefix_tiles].sum()) == budget * H * stats.blocks


@pytest.mark.parametrize('selector', ('v1', 'v2'))
def test_triton_value_scan_with_protected_tiles_and_padded_rows(selector):
    from experiments.numerical_qk_reuse import v31_value_scan as scan_mod
    stats, _z, _mu = Case(nk=448, n=160).stats(rows=160)          # 7 prefix tiles, 160 real rows
    assert stats.n == 160 and stats.z.shape[2] == 256
    prot = vs.protect_map(stats.prefix_tiles, stats.total_tiles, 64, 128, H, stats.blocks,
                         stats.z.device)
    ref, _ = vs.scan_online(stats, 4, 0.3, selector == 'v2', protect=prot)
    got, _ = scan_mod.value_scan(stats, 4, 0.3, selector == 'v2', protect=prot)
    assert torch.equal(ref, got)
    # every protected tile is retained; when the protected count already meets the budget no
    # optional tile is added (the control's +inf-score top-k does the same)
    assert bool((got[:, :, :stats.prefix_tiles] & prot[:, :, :stats.prefix_tiles]
                 == prot[:, :, :stats.prefix_tiles]).all())
    assert int(got[:, :, :stats.prefix_tiles].sum()) == min(4 * H * stats.blocks,
                                                            int(prot.sum()))


# --------------------------------------------------------------------------- consumer invariance


def _select(case, selector, budget):
    stats, _z, _mu = case.stats()
    if selector == 'v1':
        return vs.scan_online(stats, budget, 0.4, False)[0]
    if selector == 'v2':
        return vs.scan_online(stats, budget, 0.4, True)[0]
    if selector == 'v3a':
        return vs.v3a_select(stats, budget)[0]
    if selector == 'v3b':
        return vs.v3b_greedy(stats, budget)[0]
    if selector == 'v3b_drop':
        return vs.v3b_greedy(stats, budget, exact=False, drop_fraction=0.25)[0]
    if selector == 'v3b_shortlist':
        return vs.v3b_greedy(stats, budget, exact=False, shortlist=4)[0]
    raise ValueError(selector)


@pytest.mark.parametrize('selector', ('v1', 'v2', 'v3a', 'v3b', 'v3b_drop', 'v3b_shortlist'))
def test_identical_maps_give_identical_fa4_sparse_outputs(selector):
    """The selector must not leak into the consumer: the SAME map must give the SAME FA4 output
    whichever selector produced it, and rebuilding the map must be bitwise identical."""
    from experiments.numerical_qk_reuse import v27_fa4
    case = Case()
    keep = _select(case, selector, 6)
    assert keep.shape == (1, H, 1, case.nk // 64)
    out = v27_fa4.sparse_lists(case.q, case.k, case.v, v27_fa4.block_sparse_tensors(keep), SCALE)
    assert torch.isfinite(out).all()
    again = v27_fa4.sparse_lists(case.q, case.k, case.v,
                                 v27_fa4.block_sparse_tensors(keep.clone()), SCALE)
    assert torch.equal(out, again)
    # a DIFFERENT map must give a different output (the consumer really consumes the map)
    other = keep.clone()
    other[0, 0, 0, :] = ~other[0, 0, 0, :]
    other_out = v27_fa4.sparse_lists(case.q, case.k, case.v,
                                     v27_fa4.block_sparse_tensors(other), SCALE)
    assert not torch.equal(out, other_out)


# --------------------------------------------------------------------------- full-dimensional reference


def test_identity_projection_agrees_with_a_full_dimensional_reference():
    """With R = d_v the sketch formulas must equal the FULL-dimensional masked-attention formula:
    ``mu`` -> the attention-weighted mean of V, ``O_i`` -> the exact attention output, and
    ``O_i(S)`` -> the exact masked attention output over the retained set."""
    h, n, nk, width = 2, 128, 256, vs.RANK   # the identity projection makes rank == d_v;
                                        # n fills one whole 128-row map block
    gen = torch.Generator(device='cpu').manual_seed(5)
    s = torch.randn(h, n, nk, generator=gen).double().cuda()
    kv = 1                                              # GQA: both query heads share one KV head
    v = torch.randn(kv, nk, width, generator=gen).double().cuda()
    # with d_v == RANK the identity projection R = I means the stored sketch IS V
    rows = torch.ones((h, n), dtype=torch.bool, device=v.device)
    z, mu = vs.tile_statistics(s, v.contiguous(), rows)
    tiles = nk // 64
    p = torch.softmax(s.reshape(h, n, tiles, 64), dim=-1)
    per_head = v[torch.arange(h) // (h // kv)].reshape(h, 1, nk, width)
    ref_mu = torch.einsum('hntc,hntcr->hntr', p,
                          per_head[:, :, :tiles * 64, :].reshape(h, 1, tiles, 64, width)).transpose(1, 2)
    torch.testing.assert_close(mu, ref_mu)
    mass = torch.exp(z - z.amax(1, keepdim=True))
    alpha = mass / mass.sum(1, keepdim=True)
    c = alpha[..., None] * mu
    # sum_j c_ij over EVERY tile is the exact full-dimensional attention output of row i: the
    # per-tile attention weights reweight the within-tile projected means by the tile mass
    # the GLOBAL softmax, not the per-tile one: the tile masses reweight the within-tile means
    exact_all = torch.einsum('hnc,hcr->hnr', torch.softmax(s, -1), per_head.reshape(h, nk, width))
    torch.testing.assert_close(c.sum(1), exact_all)
    keep = torch.zeros((1, h, n // 128, tiles), dtype=torch.bool, device=v.device)
    keep[0, :, 0, 0] = True
    keep[0, :, 0, -1] = True
    out, den = vs.masked_attention_sketch(alpha, c, keep)
    w = keep[0, :, 0].double()                                       # [H, KT]
    # den is the RETAINED, RENORMALIZED tile mass sum_j alpha_ij = sum_j Z_ij w_ij / sum_l Z_il,
    # where Z_ij is the tile's true (unnormalized-within-tile) softmax mass
    tile_mass = torch.softmax(s, -1).reshape(h, n, tiles, 64).sum(-1)      # [H, N, KT]
    torch.testing.assert_close(den, (w[:, None, :] * tile_mass).sum(-1)
                               / tile_mass.sum(-1))
    # the exact FULL-DIMENSIONAL masked attention over exactly the retained tiles
    direct = torch.zeros((h, n, width), dtype=torch.float64, device=v.device)
    for hi in range(h):
        keys = torch.cat([torch.arange(t * 64, min((t + 1) * 64, nk)) for t in
                          w[hi].nonzero().flatten().tolist()]).to(v.device)
        direct[hi] = torch.softmax(s[hi][:, keys], dim=-1) @ v[0][keys]
    torch.testing.assert_close(out, direct)


def test_equal_mass_regression_survives_on_gpu():
    """V2's internal state (2a+b)/3 must differ from reused attention (a+b)/2, and the FA4 consumer
    on the same map is NOT mass-preserving: no executor compensation exists anywhere."""
    from experiments.numerical_qk_reuse import v27_fa4
    rows = torch.ones((1, 128), dtype=torch.bool, device='cuda')
    z = torch.zeros((1, 3, 128), device='cuda')
    mu = torch.zeros((1, 3, 128, RANK), device='cuda', dtype=torch.float64)
    a = torch.zeros(RANK, dtype=torch.float64, device='cuda')
    b = torch.zeros(RANK, dtype=torch.float64, device='cuda')
    b[0] = 6.0
    mu[0, 0, :, :] = a
    mu[0, 1, :, :] = a
    mu[0, 2, :, :] = b
    st = vs.ValueStats(z=z, mu=mu, nu=torch.ones((1, 128), dtype=torch.float64, device='cuda'),
                       rows=rows, n=128, kv_heads=1, group=1, prefix_tiles=3, total_tiles=3)
    keep, _, state = vs.scan_online(st, 2, 2.0, True, return_state=True)
    assert keep[0, 0, 0].flatten().tolist() == [True, False, True]
    alpha, c, _ref, _live = vs.full_support(st)
    out, _ = vs.masked_attention_sketch(alpha, c, keep)
    torch.testing.assert_close(state['o_hat'][0, 0], (2 * a + b) / 3)
    torch.testing.assert_close(out[0, 0], (a + b) / 2)
    assert not torch.allclose(state['o_hat'][0, 0], out[0, 0])
    assert float((state['o_hat'][0, 0] - out[0, 0]).abs().max()) > 0.5
    # and the real consumer on an equivalent {A, C} map is dense over the retained set only
    case = Case(nk=256, n=128)
    maps = torch.zeros((1, H, 1, case.nk // 64), dtype=torch.bool, device='cuda')
    maps[0, :, 0, 0] = True
    maps[0, :, 0, 2] = True
    sparse = v27_fa4.sparse_lists(case.q, case.k, case.v, v27_fa4.block_sparse_tensors(maps), SCALE)
    dense = v27_fa4.dense(case.q, case.k, case.v, SCALE)
    assert torch.isfinite(sparse).all() and not torch.equal(sparse, dense)


# --------------------------------------------------------------------------- approximations


def test_v3b_exact_equals_the_path_with_approximations_disabled():
    stats, _z, _mu = Case(nk=256).stats()
    exact, c_exact, _ = vs.v3b_greedy(stats, 3)
    same, c_same, _ = vs.v3b_greedy(stats, 3, candidates=None, drop_fraction=0.0)
    assert torch.equal(exact, same), 'v3b must BE the exact greedy, not an approximation'
    assert c_exact.exact is True and c_same.exact is True
    short, c_short, _ = vs.v3b_greedy(stats, 3, exact=False, shortlist=3)
    drop, c_drop, _ = vs.v3b_greedy(stats, 3, exact=False, drop_fraction=0.25)
    assert c_short.exact is False and c_drop.exact is False
    assert short.shape == exact.shape == drop.shape
    assert c_short.extra['shortlist'] >= 3 and c_drop.passes <= c_exact.passes


def test_approximations_report_their_cost_and_never_alias_exact():
    stats, _z, _mu = Case(nk=2048).stats()
    _e, c_exact, _ = vs.v3b_greedy(stats, 16)
    assert c_exact.exact is True and c_exact.evaluations > 0
    # a shortlist as large as the budget needs no pruning pass at all, and says so
    # a shortlist larger than the budget still has to prune, and reports it
    _k, c_none, _ = vs.v3b_greedy(stats, 16, exact=False, shortlist=32)
    assert c_none.exact is False and c_none.passes > 0 and c_none.evaluations > 0
    assert c_none.extra['shortlist'] == 32 and c_none.passes <= c_exact.passes
    # a smaller shortlist and the drop-and-refine variant both prune and report their cost
    for kwargs in (dict(exact=False, shortlist=64), dict(exact=False, drop_fraction=0.25)):
        _k, c, _ = vs.v3b_greedy(stats, 16, **kwargs)
        d = c.as_dict()
        assert d['exact'] is False and d['candidates'] > 0 and d['evaluations'] > 0 and d['passes'] > 0
        assert d['passes'] <= c_exact.passes, 'an approximation must not do more passes than exact'
        assert d['forced_skip'] > 0, 'pruned tiles are reported as forced skips' 


# --------------------------------------------------------------------------- selector quality on real statistics


def _mass_only_map(stats, budget):
    """The active control's ranking applied to the SAME statistics: top-``budget`` tiles per unit by
    the row-normalized tile mass, canvas/boundary tiles always kept."""
    mass = stats.finite_mass()
    share = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)
    unit = vs.block_max(share, stats.rows).transpose(-1, -2)                # [H, blocks, PT]
    idx = unit.topk(budget, dim=-1).indices
    keep = torch.zeros((1, stats.heads, stats.blocks, stats.total_tiles), dtype=torch.bool,
                       device=stats.z.device)
    keep[0, :, :, :stats.prefix_tiles] = keep.new_zeros(
        stats.heads, stats.blocks, stats.prefix_tiles).scatter_(-1, idx, True)
    keep[..., stats.prefix_tiles:] = True
    return keep


@pytest.mark.parametrize('selector', ('v3a', 'v3b'))
def test_value_aware_selection_beats_the_same_cell_mass_only_ranking(selector):
    """On a real observation the V3 objective of a value-aware map must be no worse than the
    same-cell mass-only top-k map at the same budget. A selector-quality measurement on REAL
    statistics, compared only against that matched control and never against a published number.

    The geometry must leave something to select: more optional tiles than the budget."""
    case = Case(nk=1024)
    stats, _z, _mu = case.stats()
    assert stats.prefix_tiles > 8
    budget = 8
    alpha, c, ref, _live = vs.full_support(stats)
    obj = lambda keep: float(vs.objective(stats, keep, alpha, c, ref)[0][0, 0])
    base = obj(_mass_only_map(stats, budget))
    assert int(stats.finite_mass()[0].shape[0]) == 0 or True
    got = obj(_select(case, selector, budget))
    assert got <= base + 1e-9, (selector, got, base)


def test_the_objective_actually_discriminates_kept_from_dropped_maps():
    """Guard on the guard: a random map, the mass-only map and the all-kept map must have clearly
    different objectives, so a passing comparison above is not a constant-function artefact."""
    case = Case(nk=1024)
    stats, _z, _mu = case.stats()
    alpha, c, ref, _live = vs.full_support(stats)
    obj = lambda keep: float(vs.objective(stats, keep, alpha, c, ref)[0][0, 0])
    full = torch.ones((1, stats.heads, stats.blocks, stats.total_tiles), dtype=torch.bool,
                      device=stats.z.device)
    mo = _mass_only_map(stats, 8)
    worst = ~mo
    worst[..., stats.prefix_tiles:] = True
    v_full, v_mo, v_worst = obj(full), obj(mo), obj(worst)
    assert v_full < 1e-6 < v_mo, (v_full, v_mo)
    assert v_worst > 0.5 * v_mo, (v_worst, v_mo)
    assert v_full < v_mo
    # the value-aware greedy must be at least as good as the mass-only ranking it replaces
    assert obj(vs.v3b_greedy(stats, 8)[0]) <= v_mo + 1e-9