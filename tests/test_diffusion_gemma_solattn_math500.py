from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.diffusion_attention_threshold_modeling.routing import (
    FreshRoutingAttention,
    FreshRoutingConfig,
)
from experiments.diffusion_gemma_solattn_math500.config import (
    ANALYTIC_BETAS,
    conditions,
)
from experiments.diffusion_gemma_solattn_math500.metrics import aggregate_calls


def test_math500_condition_grid_has_dense_and_eight_analytic_conditions():
    grid = conditions()
    assert len(grid) == 9
    assert grid[0].name == "dense"
    assert {item.region for item in grid[1:]} == {"prefix_only", "all"}
    assert {item.target_sparsity for item in grid[1:]} == set(ANALYTIC_BETAS)
    for item in grid[1:]:
        assert item.beta == ANALYTIC_BETAS[item.target_sparsity]
        assert item.retained_density == pytest.approx(1.0 - item.target_sparsity)


def test_count_weighted_sparsity_sums_tiles_instead_of_averaging_calls():
    result = aggregate_calls([{
        "per_call": [
            {
                "attention_type": "local", "physical_total_tiles": 2,
                "physical_candidate_tiles": 2, "physical_retained_tiles": 1,
                "physical_skipped_tiles": 1, "valid_rows": 1,
                "retained_dense_attention_mass": 0.9,
            },
            {
                "attention_type": "local", "physical_total_tiles": 10,
                "physical_candidate_tiles": 10, "physical_retained_tiles": 1,
                "physical_skipped_tiles": 9, "valid_rows": 9,
                "retained_dense_attention_mass": 0.5,
            },
        ]
    }])
    # (1 + 9) / (2 + 10), not mean(1/2, 9/10) = 0.7.
    assert result["whole_model"]["full_model_tile_sparsity"] == pytest.approx(10 / 12)
    assert result["local"]["full_model_tile_sparsity"] == pytest.approx(10 / 12)
    assert result["whole_model"]["retained_dense_attention_mass"] == pytest.approx(0.54)


def test_prefix_only_counts_dense_canvas_in_full_model_denominator():
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((
        torch.ones(1, 1, 64, 1),
        torch.full((1, 1, 64, 1), 2.0),
        torch.full((1, 1, 64, 1), 3.0),
    ), dim=-2)
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="gaussian", target_density=0.5, region="prefix_only",
        q_block_size=64, kv_block_size=64, combined_region_population=True,
    ))
    router(module, query, key, key, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert summary["physical_total_tiles"] == 3
    assert summary["physical_candidate_tiles"] == 2
    assert summary["physical_skipped_tiles"] == 1
    assert summary["full_model_physical_sparsity"] == pytest.approx(1 / 3)
    assert summary["all_region_counts"]["canvas"] == {
        "total_tiles": 1, "candidate_tiles": 0, "retained_tiles": 1, "skipped_tiles": 0,
    }


def test_prefix_only_degenerate_row_keeps_every_candidate():
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    query = torch.ones(1, 1, 64, 1)
    key = torch.ones(1, 1, 128, 1)
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="gaussian", target_density=0.1, region="prefix_only",
        q_block_size=64, kv_block_size=64, combined_region_population=True,
    ))
    router(module, query, key, key, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert summary["degenerate_rows"] == 1
    assert summary["physical_skipped_tiles"] == 0
