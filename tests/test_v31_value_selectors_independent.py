"""Independent formula checks, not comparisons to a previous implementation."""
import math

import pytest
import torch

from experiments.numerical_qk_reuse.v31_value_selectors import (
    Stats, SELECTORS, statistics_reference, online_reference, full_support,
    masked_output, singleton_reference, greedy_reference, select, mandatory_map)


def small(means, log_mass=None, prefix=None):
    mean = torch.tensor(means, dtype=torch.float32).reshape(1, -1, 1, 1)
    j = mean.shape[1]
    z = torch.zeros((1, j, 1)) if log_mass is None else torch.tensor(log_mass).float().reshape(1, j, 1)
    return Stats(z, mean, torch.ones(1, 1), torch.zeros_like(z, dtype=torch.bool),
                 j if prefix is None else prefix, 1, 1)


def direct_output(s, keep):
    # Direct softmax on retained tiles, independent of alpha/c/g implementation.
    z = s.log_mass.masked_fill(~keep[..., None], -math.inf)
    weights = torch.softmax(z, dim=1)
    return (torch.nan_to_num(weights)[..., None]*s.mean).sum(1)


@pytest.mark.parametrize('preserve', [False, True])
def test_equal_mass_skip_is_selector_state_only(preserve):
    s = small([1., 1., 4.])
    keep, routing, lm, _ = online_reference(s, 0.9, preserve)
    assert keep.tolist() == [[True, False, True]]
    assert routing.item() == pytest.approx(2. if preserve else 2.5)
    assert direct_output(s, keep).item() == pytest.approx(2.5)
    assert lm.exp().item() == pytest.approx(3. if preserve else 2.)


def test_v2_skip_changes_running_max_but_not_normalized_output():
    s = small([2., 2.], [0., 100.])
    keep, routing, lm, _ = online_reference(s, 0.1, True)
    assert keep.tolist() == [[True, False]]
    assert routing.item() == 2.
    assert lm.item() == pytest.approx(100.)
    _, _, discard_lm, _ = online_reference(s, 0.1, False)
    assert discard_lm.item() == 0.


def test_future_mass_does_not_underflow_first_support():
    s = small([3., 3.], [0., 1000.])
    keep, routing, lm, _ = online_reference(s, 0.1, True)
    assert keep.tolist() == [[True, False]]
    assert routing.item() == 3.
    assert lm.item() == 1000.


def test_threshold_tie_and_no_quota():
    s = small([0., 2., 6.])
    keep, _, _, risks = online_reference(s, 1., False)
    assert keep[0, 1]  # eta=1/2, distance=2, rho=1: strict tie keeps
    assert risks[0, 1].item() == pytest.approx(0.)
    assert select(s, SELECTORS[0], budget=1, threshold=0.)[0].sum() == 3


def test_first_support_is_per_row_and_mandatory_not_a_quota():
    s = small([0., 0., 0.])
    s.log_mass = torch.tensor([[[-math.inf, 0.], [0., 0.], [0., 0.]]])
    s.mean = torch.zeros(1, 3, 2, 1)
    s.nu = torch.ones(1, 2)
    s.invalid = torch.zeros(1, 3, 2, dtype=torch.bool)
    mandatory = torch.tensor([[False, False, True]])
    keep, _, _, _ = online_reference(s, 10., True, mandatory)
    assert keep.tolist() == [[True, True, True]]


def test_statistics_identity_gqa_partial_keys_and_empty_rows():
    torch.manual_seed(31)
    scores = torch.randn(4, 5, 9)
    scores[0, 2] = -math.inf
    values = torch.randn(2, 9, 3)
    s = statistics_reference(scores, values, torch.ones(2), 2, q_block=4, k_block=4)
    keep = torch.ones(8, 3, dtype=torch.bool)
    result = direct_output(s, keep).reshape(4, 8, 3)[:, :5]
    expected = torch.nan_to_num(torch.softmax(scores, -1)) @ values.repeat_interleave(2, 0)
    assert torch.allclose(result, expected, atol=2e-6)
    assert not s.rows.reshape(4, 8)[0, 2]
    assert not s.rows.reshape(4, 8)[:, 5:].any()


@pytest.mark.parametrize('selector', SELECTORS[:4])
def test_invalid_score_forces_dense_query_unit(selector):
    scores = torch.zeros(1, 2, 5)
    scores[0, 0, 1] = math.nan
    s = statistics_reference(scores, torch.ones(1, 5, 2), torch.ones(1), 2, q_block=2, k_block=2)
    keep, _ = select(s, selector, budget=2, threshold=10.)
    assert keep.all()


def test_singleton_scores_equal_direct_deleted_attention():
    torch.manual_seed(17)
    scores = torch.randn(2, 4, 13)
    sketch = torch.randn(1, 13, 3)
    s = statistics_reference(scores, sketch, torch.tensor([2.]), 3, q_block=4, k_block=4)
    keep, scores = singleton_reference(s, 2)
    allkept = torch.ones(2, 4, dtype=torch.bool)
    full = direct_output(s, allkept)
    for j in range(3):
        deleted = allkept.clone()
        deleted[:, j] = False
        error = ((direct_output(s, deleted)-full).norm(dim=-1)/2.).amax(-1)
        assert torch.allclose(scores[:, j], error, atol=2e-6)
    assert (keep[:, :3].sum(-1) == 2).all()
    assert keep[:, 3].all()


def direct_greedy(s, budget, mandatory):
    keep = s.valid.any(-1) | mandatory
    full = direct_output(s, keep)
    for unit in range(keep.shape[0]):
        while keep[unit, :s.prefix_tiles].sum() > budget:
            options = []
            for j in range(s.prefix_tiles):
                if keep[unit, j] and not mandatory[unit, j]:
                    candidate = keep.clone()
                    candidate[unit, j] = False
                    remaining = s.valid[unit] & candidate[unit, :, None]
                    if bool((s.rows[unit] & ~remaining.any(0)).any()):
                        continue
                    error = ((direct_output(s, candidate)[unit]-full[unit]).norm(dim=-1)/s.nu[unit]).amax().item()
                    options.append((error, j))
            if not options:
                raise ValueError('no support')
            keep[unit, min(options)[1]] = False
    return keep


@pytest.mark.parametrize('seed', range(5))
def test_exact_greedy_matches_direct_recomputation(seed):
    torch.manual_seed(seed)
    s = statistics_reference(torch.randn(2, 4, 17), torch.randn(1, 17, 3), torch.ones(1),
                             4, q_block=4, k_block=4)
    mandatory = mandatory_map(s, sink=1)
    got, count = greedy_reference(s, 2, mandatory)
    expected = direct_greedy(s, 2, mandatory)
    assert torch.equal(got, expected)
    assert count > 0


def test_last_row_support_is_inadmissible_and_fixed_budget_ties_are_stable():
    s = small([1., 1., 1.])
    assert singleton_reference(s, 1)[0].tolist() == [[True, False, False]]
    assert greedy_reference(s, 1)[0].tolist() == [[False, False, True]]
    # Only tile 0 supports the second row, so neither algorithm may delete it.
    s.log_mass = torch.tensor([[[0., 0.], [0., -math.inf], [0., -math.inf]]])
    s.mean = torch.ones(1, 3, 2, 1)
    s.nu = torch.ones(1, 2)
    s.invalid = torch.zeros(1, 3, 2, dtype=torch.bool)
    assert singleton_reference(s, 1)[0][0, 0]
    assert greedy_reference(s, 1)[0][0, 0]


def test_same_mask_has_same_native_operator_for_all_selector_states():
    s = small([1., 1., 4.])
    v1 = online_reference(s, .9, False)
    v2 = online_reference(s, .9, True)
    assert torch.equal(v1[0], v2[0])
    assert not torch.equal(v1[1], v2[1])
    assert torch.equal(direct_output(s, v1[0]), direct_output(s, v2[0]))


@pytest.mark.parametrize('cuda', [False, True])
def test_singleton_joint_support_constraint(cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    # Naive top-3 chooses A,B,E, losing C/D's row. Scores stay fixed while
    # support constraints require retaining A,C,E instead.
    z = torch.full((1, 5, 128), -math.inf)
    z[:, :2, 0] = 0.
    z[:, 2:4, 1] = 0.
    z[:, 4, 2] = 0.
    s = Stats(z, torch.ones(1, 5, 128, 32), torch.ones(1, 128),
              torch.zeros_like(z, dtype=torch.bool), 5, 1, 1)
    if cuda:
        s = Stats(*(x.cuda() for x in (s.log_mass, s.mean, s.nu, s.invalid)), 5, 1, 1)
    keep, _ = select(s, SELECTORS[2], budget=3)
    assert keep.cpu().reshape(-1).tolist() == [True, False, True, False, True]


def test_adapter_value_scope_guard():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    options = dict(arm='mage', mage_select='fa4', mage_granularity='qblock_max',
                   value_selector=SELECTORS[0], value_threshold=.1)
    types = ['sliding_attention']*5+['full_attention']
    assert VllmMethodAdapter(types, **options).local_router is None
    for extra in (dict(local_kv_budget=512), dict(mage_keep_frac=.1), dict(mage_pool=2)):
        with pytest.raises(ValueError):
            VllmMethodAdapter(types, **(options | extra))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('selector', SELECTORS)
def test_cuda_selectors_against_independent_oracles(selector):
    from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
    torch.manual_seed(71)
    q = torch.randn(1, 4, 133, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 701, 64, dtype=torch.bfloat16)
    z = torch.randn(1, 2, 701, 32)
    nu = torch.tensor([5., 7.])
    reference = statistics_reference((q[0].float() @ k[0].float().repeat_interleave(2, 0).transpose(-1, -2))/8., z[0], nu, 10)
    stats = statistics_cuda(q.cuda(), k.cuda(), z.cuda(), nu.cuda(), .125, 10)
    assert torch.allclose(stats.log_mass.cpu(), reference.log_mass, atol=3e-5)
    assert torch.allclose(stats.mean.cpu(), reference.mean, atol=3e-5)
    expected, _ = select(reference, selector, budget=2, threshold=.01)
    got, _ = select(stats, selector, budget=2, threshold=.01)
    assert torch.equal(got.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('preserve', [False, True])
def test_cuda_online_state_with_large_running_max_change(preserve):
    from experiments.numerical_qk_reuse.v31_value_kernels import online_cuda
    s = statistics_reference(torch.zeros(1, 128, 192), torch.ones(1, 192, 32), torch.ones(1), 3)
    s.log_mass[:, 1] += 1000.
    expected = online_reference(s, 1., preserve)
    gpu = Stats(*(x.cuda() for x in (s.log_mass, s.mean, s.nu, s.invalid)), 3, 1, 1)
    got = online_cuda(gpu, 1., preserve, mandatory_map(gpu), torch.zeros(1, 3, device='cuda', dtype=torch.bool), 0.)
    for a, b in zip(got, expected):
        assert torch.allclose(a.cpu(), b, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_identical_online_maps_use_identical_current_fa4_output():
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.v31_value_kernels import online_cuda
    s = statistics_reference(torch.zeros(16, 256, 192),
        torch.cat((torch.ones(2, 128, 32), torch.full((2, 64, 32), 4.)), dim=1),
        torch.full((2,), math.sqrt(32)), 3)
    gpu = Stats(*(x.cuda() for x in (s.log_mass, s.mean, s.nu, s.invalid)), 3, 16, 2)
    mandatory = torch.zeros(32, 3, dtype=torch.bool, device='cuda')
    v1 = online_cuda(gpu, .9, False, mandatory, mandatory, 0.)
    v2 = online_cuda(gpu, .9, True, mandatory, mandatory, 0.)
    assert torch.equal(v1[0], v2[0])
    assert not torch.equal(v1[1], v2[1])
    torch.manual_seed(31)
    q = torch.randn(1, 16, 256, 512, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 2, 192, 512, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    outputs = [v27_fa4.sparse_lists(q, k, v,
        v27_fa4.block_sparse_tensors(result[0].reshape(1, 16, 2, 3)), 512**-.5) for result in (v1, v2)]
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('selector', SELECTORS[:4])
def test_cuda_invalid_and_empty_rows_match_safe_reference(selector):
    s = statistics_reference(torch.zeros(1, 128, 192), torch.ones(1, 192, 32), torch.ones(1), 2)
    s.log_mass[:, :, 3] = -math.inf
    s.invalid[:, 0, 7] = True
    gpu = Stats(*(x.cuda() for x in (s.log_mass, s.mean, s.nu, s.invalid)), 2, 1, 1)
    keep, _ = select(gpu, selector, budget=1, threshold=1.)
    assert keep.all()
