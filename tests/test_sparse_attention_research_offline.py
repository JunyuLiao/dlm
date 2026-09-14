import numpy as np
import torch

from experiments.diffusion_gemma_sparse_attention_research.offline_analysis import (
    average_precision,
    binary_auc,
    mass_support,
    neighbor_mean,
    physical_online_mask,
    query_union_curve,
    top_budget_mask,
)
from experiments.diffusion_gemma_sparse_attention_research.collect_rich import summarize_tile_supervisors


def test_rank_metrics_and_mass_support():
    labels = np.array([False, False, True, True])
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    assert binary_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    support = mass_support(np.array([0.6, 0.3, 0.1]), 0.9)
    assert support.tolist() == [True, True, False]


def test_top_budget_is_physical_and_nonempty():
    keep = top_budget_mask(np.arange(10), 0.75)
    assert keep.sum() == 3
    assert keep[-3:].all()
    assert top_budget_mask(np.arange(3), 0.99).sum() == 1


def test_online_physical_mask_uses_query_union():
    maxima = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    valid = np.ones_like(maxima, dtype=bool)
    # At lambda=1 each row retains only new records/ties. Their union retains
    # every physical KV tile even though each row alone skips one tile.
    assert physical_online_mask(maxima, valid, 1.0).tolist() == [True, True, True]


def test_query_union_reduces_executable_sparsity():
    mass = np.array([[0.9, 0.1], [0.1, 0.9]])
    valid = np.ones_like(mass, dtype=bool)
    curve = query_union_curve(mass, valid, 0.5, (1, 2))
    assert curve["logical"]["sparsity"] == 0.5
    assert curve["1"]["sparsity"] == 0.5
    assert curve["2"]["sparsity"] == 0.0


def test_neighbor_mean_has_no_self_signal():
    result = neighbor_mean(np.array([1.0, 2.0, 7.0]))
    assert np.allclose(result, [2.0, 4.0, 2.0])


def test_rich_tile_supervisors_mass_entropy_and_value_effect():
    scores = torch.tensor([[2.0, 0.0, -1.0, -2.0], [0.0, 2.0, -2.0, -1.0]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    probabilities = scores.softmax(-1)
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    result = summarize_tile_supervisors(scores, valid, probabilities, values, tile_size=2)
    assert result["eligible"].tolist() == [True, True]
    assert torch.allclose(result["tile_mass"].sum(), torch.tensor(2.0))
    assert result["tile_value_rms"][1] > result["tile_value_rms"][0]
    assert torch.isfinite(result["tile_drop_output_l2"]).all()
    assert result["query_mass_support90"].shape == (2, 2)
