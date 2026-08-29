from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments.math500_prefix_proxy_distribution.collector import (
    PrefixProxyConfig,
    PrefixProxyShardCollector,
)
from experiments.math500_prefix_proxy_distribution.plots import generate_plots
from experiments.diffusion_attention_threshold_modeling.datasets import load_ruler


def test_prefix_collector_keeps_raw_prefix_logits_and_excludes_canvas(tmp_path):
    collector = PrefixProxyShardCollector(PrefixProxyConfig(reservoir_per_row=8))
    collector.begin_prompt(request_id="math500-000", problem_index=0)
    module = nn.Module()
    module.layer_idx = 3
    module.num_key_value_groups = 1
    query = torch.ones(1, 1, 128, 1)
    prefix = torch.cat((torch.full((1, 1, 64, 1), 2.0), torch.full((1, 1, 64, 1), 4.0)), dim=-2)
    canvas = torch.full((1, 1, 128, 1), 100.0)
    key = torch.cat((prefix, canvas), dim=-2)
    collector(module, query, key, key, None, scaling=1.0, is_causal=False)
    path = collector.export_shard(tmp_path / "prefix.npz")
    with np.load(path) as payload:
        assert payload["prefix_tiles"].tolist() == [2, 2]
        assert payload["prefix_length"].tolist() == [128, 128]
        assert payload["prefix_mean"].tolist() == pytest.approx([3.0, 3.0])
        assert payload["prefix_reservoir"].shape == (2, 8)
        assert np.sort(payload["prefix_reservoir"][0, :2]) == pytest.approx([2.0, 4.0])
        assert payload["hierarchical_row_weight"].tolist() == pytest.approx([0.5, 0.5])
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["population"] == "prefix_only"
    assert "no row standardization" in sidecar["score_transform"]


def test_prefix_distribution_plot_handles_global_only_fast_model(tmp_path):
    shard_root = tmp_path / "shards" / "fast_dllm_v2"
    shard_root.mkdir(parents=True)
    np.savez_compressed(
        shard_root / "00_math500-000.npz",
        request_id=np.asarray(["math500-000", "math500-000"]),
        problem_index=np.asarray([0, 0]), denoising_call=np.asarray([0, 0]),
        layer=np.asarray([0, 0]), head=np.asarray([0, 1]),
        attention_type=np.asarray(["global", "global"]), query_block=np.asarray([0, 0]),
        query_start=np.asarray([0, 0]), query_size=np.asarray([64, 64]),
        prefix_length=np.asarray([128, 128]), prefix_tiles=np.asarray([2, 2]),
        prefix_mean=np.asarray([1.0, 2.0]), prefix_std=np.asarray([1.0, 1.0]),
        prefix_min=np.asarray([0.0, 1.0]), prefix_max=np.asarray([2.0, 3.0]),
        prefix_reservoir=np.asarray([[0.0, 2.0, np.nan], [1.0, 3.0, np.nan]], dtype=np.float32),
        hierarchical_row_weight=np.asarray([0.5, 0.5]),
    )
    summary = generate_plots(tmp_path)
    assert summary["groups"]["fast_dllm_v2|global"]["rows"] == 2
    assert summary["groups"]["fast_dllm_v2|local"]["rows"] == 0
    assert summary["score_transform"].startswith("row-standardized")
    assert summary["groups"]["fast_dllm_v2|global"]["weighted_mean"] == pytest.approx(0.0)
    assert summary["groups"]["fast_dllm_v2|global"]["weighted_skewness"] == pytest.approx(0.0)
    assert summary["groups"]["fast_dllm_v2|global"]["weighted_excess_kurtosis"] == pytest.approx(-2.0)
    assert summary["groups"]["fast_dllm_v2|global"]["asymmetric_tail_probabilities"] == {
        "gt_2": 0.0, "lt_minus_2": 0.0, "gt_3": 0.0, "lt_minus_3": 0.0,
    }
    assert (tmp_path / "plots" / "01_standardized_prefix_proxy_distributions.png").is_file()
    assert (tmp_path / "plots" / "02_row_mean_prefix_proxy_distributions.png").is_file()


def test_ruler16k_selection_is_ten_balanced_examples():
    rows = load_ruler(
        "results/blasst/diffusion_gemma/context_tile_sweep_lambda_0p003/manifests/16384/samples.jsonl",
        10,
        20260824,
        corpus="ruler16k",
    )
    assert len(rows) == 10
    assert {row["corpus"] for row in rows} == {"ruler16k"}
    assert len({row["request_id"] for row in rows}) == 10
    assert {row["task"] for row in rows} == {
        "niah_multikey_1", "niah_multivalue", "niah_multiquery", "vt"
    }
