import torch

from experiments.numerical_qk_reuse.multi_model_attention import budget_map


def test_budget_includes_protected_tiles_and_stable_ties():
    score = torch.ones((2, 8))
    eligible = torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool)
    mandatory = torch.zeros_like(eligible)
    mandatory[:, 4] = True
    keep = budget_map(score, mandatory, eligible)
    assert keep[0].tolist() == [True, True, True, False, True, False, False, False]
    assert keep[1].sum() == 2
    assert keep[:, 4].all()
    assert not (keep & ~eligible).any()


def test_infeasible_protection_is_preserved():
    score = torch.tensor([[.2, .1, .3]])
    mandatory = torch.tensor([[True, True, True]])
    assert budget_map(score, mandatory, torch.ones_like(mandatory)).all()
