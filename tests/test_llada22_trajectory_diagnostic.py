import torch

from scripts.llada22_eos_trajectory_diagnostic import eos_stats, output_metrics
from scripts.llada22_snapshot_attention_diagnostic import mass_count_matched


def test_mass_and_output_error_use_token_probabilities_and_sparse_renormalization():
    # Equal logits, GQA, two physical tiles: retaining half the tiles loses
    # exactly half the probability and changes output from 2 to 1.
    q = torch.zeros(1, 4, 2, 2)
    k = torch.zeros(1, 2, 128, 2)
    v = torch.ones(1, 2, 128, 2)
    v[:, :, 64:] = 3
    keep = torch.tensor([[[[True, False]]]*4])
    metrics, full, sparse = output_metrics(q, k, v, keep, 1.)
    assert metrics['retained_mass']['mean'] == .5
    assert torch.allclose(full, torch.full_like(full, 2.))
    assert torch.allclose(sparse, torch.ones_like(sparse))
    assert abs(metrics['relative_error_over_dense_output']['mean']-.5) < 1e-6


def test_allkept_operator_is_exactly_dense():
    gen = torch.Generator().manual_seed(211)
    q = torch.randn(1, 4, 3, 4, generator=gen)
    k = torch.randn(1, 2, 70, 4, generator=gen)
    v = torch.randn(1, 2, 70, 4, generator=gen)
    keep = torch.ones(1, 4, 1, 2, dtype=torch.bool)
    metrics, full, sparse = output_metrics(q, k, v, keep, .5)
    assert torch.allclose(full, sparse, atol=1e-6)
    assert metrics['error_over_value_rms']['max'] < 1e-6


def test_eos_probability_margin_and_rank_are_shift_invariant():
    logits = torch.tensor([[3., 1., 2.], [1., 5., 2.]])
    stats = eos_stats(logits, 1)
    shifted = eos_stats(logits+13, 1)
    assert stats['eos_rank'] == [3, 1]
    assert stats['eos_margin'] == [-2., 3.]
    assert stats['eos_argmax'] == [False, True]
    assert stats == shifted


def test_matched_count_mass_control_protects_mandatory_and_exact_head_counts():
    scores = torch.tensor([[1.,3.,2.],[3.,1.,2.]])
    mandatory = torch.tensor([[True,False,False],[False,False,True]])
    keep = mass_count_matched(scores,mandatory,torch.tensor([2,1]))
    assert keep.tolist() == [[True,True,False],[False,False,True]]
