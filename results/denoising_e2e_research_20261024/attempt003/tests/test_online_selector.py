import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from selector import removal, mass_lower_bound, candidate_tiles, candidate_supports, choose_unit


def test_tile_bias_and_q_response_match_recomputed_attention():
    gen = torch.Generator().manual_seed(17)
    q = torch.randn(3, 5, generator=gen, dtype=torch.float64)
    k = torch.randn(192, 5, generator=gen, dtype=torch.float64)
    v = torch.randn(192, 7, generator=gen, dtype=torch.float64)
    u = torch.randn(q.shape, generator=gen, dtype=q.dtype) * .1
    scale = .4
    _, o, rows = removal(q, k, v, scale, [1], u)
    row = rows[0]

    def calculate(t):
        z = (q + t*u) @ k.T * scale
        a = z.softmax(-1) @ v
        z[:, 64:128] = -torch.inf
        s = z.softmax(-1) @ v
        return a, s

    a, s = calculate(0)
    assert torch.allclose(o, a)
    assert torch.allclose(row['bias'], s-a, atol=1e-12)
    ap, sp = calculate(1e-5)
    am, sm = calculate(-1e-5)
    assert torch.allclose(row['J_A'], (ap-am)/2e-5, atol=1e-9)
    assert torch.allclose(row['J_S'], (sp-sm)/2e-5, atol=1e-9)
    assert torch.allclose(row['residual'], ((sp-ap)-(sm-am))/2e-5, atol=1e-9)


def test_retained_mass_certificate_accounts_for_changing_canvas_keys():
    gen = torch.Generator().manual_seed(18)
    q0 = torch.randn(4, 6, generator=gen, dtype=torch.float64)
    prefix = torch.randn(128, 6, generator=gen, dtype=torch.float64)
    q = q0 + .03*torch.randn(q0.shape, generator=gen, dtype=q0.dtype)
    # Deliberately very different current canvas keys. No old canvas K is used.
    current_canvas = torch.randn(32, 6, generator=gen, dtype=q0.dtype)*3
    scale = .25
    z0 = q0 @ prefix.T * scale
    kept0 = z0[:, 64:].logsumexp(-1)
    tailZ = (q @ current_canvas.T * scale).logsumexp(-1)
    bound = mass_lower_bound(q0, q, prefix.norm(dim=-1).max(), z0.logsumexp(-1), kept0, tailZ, scale)
    z = q @ torch.cat([prefix, current_canvas]).T * scale
    actual = z.softmax(-1)[:, 64:].sum(-1)
    assert bool((bound <= actual + 1e-12).all())
    exact = mass_lower_bound(q0, q0, prefix.norm(dim=-1).max(), z0.logsumexp(-1), kept0, tailZ, scale)
    reference = torch.logaddexp(kept0, tailZ) - torch.logaddexp(z0.logsumexp(-1), tailZ)
    assert torch.allclose(exact, reference.exp())


def test_candidate_proposals_do_not_consume_native_rng_and_enumerate_short():
    before = torch.get_rng_state().clone()
    assert candidate_tiles(torch.arange(5.), 5) == list(range(5))
    a = candidate_tiles(torch.arange(50.), 50, unit=7)
    b = candidate_tiles(torch.arange(50.), 50, unit=7)
    assert a == b and len(set(a)) == 12
    assert torch.equal(before, torch.get_rng_state())


def test_all_scoring_rules_share_feasible_unit_work_and_preserve_canvas():
    gen = torch.Generator().manual_seed(19)
    q = torch.randn(4, 3, generator=gen) * .05
    k = torch.randn(192, 3, generator=gen) * .05
    v = torch.ones(192, 5) + torch.randn(192, 5, generator=gen) * .01
    modes = ['current_error_fixed4', 'response_guard_qdirection', 'margin_guard_weighted_bias',
             'validity_reuse_mass_certificate', 'sham_fixed4']
    banks = []
    for mode in modes:
        keep, bank, cert = choose_unit(q, k, v, .1, 2, q*.9,
            torch.ones(4), torch.ones(4, dtype=torch.bool), mode, 0, guards=(.5, .1))
        assert keep.shape == (2,) and int(keep.sum()) == 1
        banks.append([(x['tile'], x['feasible']) for x in bank['candidates']])
        assert all('k' != key and 'v' != key for key in cert)
    assert all(b == banks[0] for b in banks)


def test_empty_feasible_set_is_allkept_and_not_ranking_success():
    q = torch.ones(3, 2)
    k = torch.ones(128, 2)
    v = torch.arange(128.).view(128, 1)
    keep, bank, _ = choose_unit(q, k, v, 1., 1, q, None, None,
                               'current_error_fixed4', 0, guards=(0., 0.))
    assert keep.all() and bank['chosen'] is None
    assert not bank['ranking_identifying']


def test_runtime_begin_clears_canvas_certificates_and_query_state():
    from online_runtime import Runtime
    rt = Runtime('validity_reuse_mass_certificate', False)
    rt.certificates = [object()]
    rt.previous_q = torch.ones(2)
    rt.keep = torch.zeros(2, dtype=torch.bool)
    rt.begin()
    assert not rt.certificates and rt.previous_q is None and rt.keep is None


def test_larger_support_bank_has_equal_ten_percent_work():
    a = candidate_supports(torch.arange(100.), 100, 10, unit=3)
    assert len(a) == len(set(a)) == 12
    assert all(len(set(x)) == 10 for x in a)
    assert all(0 <= j < 100 for row in a for j in row)


def test_multi_tile_bias_matches_full_sparse_operator():
    gen = torch.Generator().manual_seed(22)
    q = torch.randn(3, 5, generator=gen, dtype=torch.float64)
    k = torch.randn(256, 5, generator=gen, dtype=torch.float64)
    v = torch.randn(256, 7, generator=gen, dtype=torch.float64)
    u = torch.randn(q.shape, generator=gen, dtype=q.dtype)*.1
    _, dense, rows = removal(q, k, v, .3, [(0, 2)], u)
    def sparse(t):
        z = (q+t*u)@k.T*.3
        z[:, :64] = -torch.inf; z[:, 128:192] = -torch.inf
        return z.softmax(-1)@v
    assert torch.allclose(rows[0]['bias'], sparse(0)-dense, atol=1e-12)
    assert torch.allclose(rows[0]['J_S'], (sparse(1e-5)-sparse(-1e-5))/2e-5, atol=1e-9)
