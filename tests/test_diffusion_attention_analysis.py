from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from experiments.diffusion_attention_analysis.hooks import install_attention_observer
from experiments.diffusion_attention_analysis.analyze import analyze_result_directory
from experiments.diffusion_attention_analysis.proxy import (
    AttentionObservationConfig,
    AttentionObserver,
    attention_scores_and_validity,
)
from experiments.diffusion_attention_analysis.statistics import add_standardized_proxies
from experiments.diffusion_attention_analysis.replay import (
    ReplayAttention,
    ReplayConfig,
    expand_routing_mask,
    routing_mask,
)


ALL_ATTENTION_FUNCTIONS = {}


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 3
        self.num_key_value_groups = 1

    def forward(self, query, key, value, mask):
        output, weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            self,
            query,
            key,
            value,
            mask,
            scaling=0.5,
            is_causal=False,
            sliding_window=None,
        )
        return output, weights


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = ToyAttention()

    def forward(
        self,
        decoder_input_ids=None,
        self_conditioning_logits=None,
        *,
        query,
        key,
        value,
        mask,
    ):
        return self.attention(query, key, value, mask)


def native_attention(module, query, key, value, mask, *, scaling, **kwargs):
    del module, kwargs
    scores = query @ key.transpose(-2, -1) * scaling
    scores = scores + mask
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return (probabilities @ value).transpose(1, 2).contiguous(), None


def test_attention_score_semantics_include_scale_mask_and_gqa():
    module = nn.Module()
    module.num_key_value_groups = 2
    query = torch.tensor([[[[1.0], [1.0]], [[2.0], [2.0]]]])
    key = torch.tensor([[[[3.0], [4.0]]]])
    mask = torch.tensor([[[[0.0, -torch.inf], [0.0, 0.0]]]])
    repeated, scores, valid = attention_scores_and_validity(
        module,
        query,
        key,
        mask,
        scaling=0.5,
        is_causal=False,
        sliding_window=None,
    )
    assert repeated.shape == (1, 2, 2, 1)
    assert scores[0, 0, 0, 0].item() == 1.5
    assert torch.isneginf(scores[0, 0, 0, 1])
    assert scores[0, 1, 1].tolist() == [3.0, 4.0]
    assert valid[0, 0, 0].tolist() == [True, False]


def test_observer_records_region_aware_tiles_and_state(tmp_path):
    config = AttentionObservationConfig(q_block_size=2, kv_block_size=2)
    observer = AttentionObserver(config)
    observer.begin_request("request-0", num_denoising_steps=2)
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    query = torch.tensor([[[[1.0], [2.0]]]])
    key = torch.tensor([[[[1.0], [3.0], [2.0], [4.0]]]])
    value = torch.ones_like(key)
    observer.begin_forward(None, None)
    observer.observe(
        module,
        query,
        key,
        value,
        None,
        scaling=1.0,
        is_causal=False,
    )
    observer.begin_forward(None, torch.zeros(1))
    observer.observe(
        module,
        query * 2,
        torch.cat((key[..., :2, :], key[..., 2:, :] * 2), dim=-2),
        value,
        None,
        scaling=1.0,
        is_causal=False,
    )
    assert len(observer.tile_records) == 4
    first_prefix = observer.tile_records[0]
    assert first_prefix["kv_region"] == "prefix"
    assert first_prefix["prefix_length"] == 2
    assert first_prefix["proxy_mean"] == pytest.approx(3.0)
    assert first_prefix["proxy_max"] == pytest.approx(6.0)
    assert len(observer.state_records) == 1
    assert observer.state_records[0]["q_cosine"] == pytest.approx(1.0)
    observer.export(tmp_path, {"purpose": "unit-test"})
    with np.load(tmp_path / "tile_records.npz") as payload:
        assert payload["proxy_mean"].shape == (4,)
        assert set(payload["kv_region"].tolist()) == {"prefix", "canvas"}


def test_standardization_uses_complete_kv_row_not_each_region():
    import pandas as pd

    common = {
        "request_id": "r",
        "denoising_step": 0,
        "layer": 0,
        "head": 0,
        "attention_type": "global",
        "query_block": 0,
    }
    rows = pd.DataFrame([
        {**common, "kv_region": "prefix", "proxy_mean": 1.0, "proxy_max": 2.0},
        {**common, "kv_region": "canvas", "proxy_mean": 3.0, "proxy_max": 6.0},
    ])
    result = add_standardized_proxies(rows)
    assert result.proxy_z_mean.tolist() == pytest.approx([-1.0, 1.0])
    assert result.proxy_z_max.tolist() == pytest.approx([-1.0, 1.0])


def test_mean_proxy_is_literal_mean_q_dot_mean_k_on_partial_mask():
    observer = AttentionObserver(AttentionObservationConfig(q_block_size=2, kv_block_size=2))
    observer.begin_request("partial")
    observer.begin_forward(None, None)
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    query = torch.tensor([[[[1.0], [10.0]]]])
    key = torch.tensor([[[[1.0], [10.0]]]])
    mask = torch.tensor([[[[0.0, -torch.inf], [-torch.inf, 0.0]]]])
    observer.observe(module, query, key, key, mask, scaling=1.0, is_causal=False)
    assert observer.tile_records[0]["proxy_mean"] == pytest.approx(30.25)
    assert observer.tile_records[0]["proxy_max"] == pytest.approx(100.0)


def test_read_only_binding_preserves_native_attention_exactly():
    ALL_ATTENTION_FUNCTIONS["sdpa"] = native_attention
    model = ToyModel()
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 2, 4)
    kwargs = dict(
        decoder_input_ids=torch.ones(1, 2, dtype=torch.long),
        query=query,
        key=key,
        value=value,
        mask=mask,
    )
    baseline = model(**kwargs)[0]
    observer = AttentionObserver(AttentionObservationConfig(q_block_size=2, kv_block_size=2))
    observer.begin_request("parity")
    with install_attention_observer(model, observer, attention_class_name="ToyAttention"):
        observed = model(**kwargs)[0]
    assert torch.equal(baseline, observed)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is native_attention
    assert len(observer.tile_records) == 2


def test_end_to_end_offline_analysis_writes_report_tables_and_plots(tmp_path):
    config = AttentionObservationConfig(
        q_block_size=1,
        kv_block_size=1,
        physical_q_tile_size=2,
        physical_kv_tile_size=1,
    )
    observer = AttentionObserver(config)
    observer.begin_request("analysis", num_denoising_steps=4)
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    value = torch.ones(1, 1, 6, 2)
    for step in range(4):
        query = torch.tensor([[[[1.0 + step, 0.5], [0.5, 2.0 + step]]]])
        prefix = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]]]])
        canvas = torch.tensor([[[[1.0, 1.0], [2.0, -1.0], [-1.0, 2.0], [0.5, 0.5]]]])
        canvas = torch.roll(canvas, shifts=step, dims=-2)
        observer.begin_forward(None, None if step == 0 else torch.zeros(1))
        observer.observe(
            module,
            query,
            torch.cat((prefix, canvas), dim=-2),
            value,
            None,
            scaling=0.5,
            is_causal=False,
        )
    observer.export(
        tmp_path,
        {
            "model": "synthetic",
            "model_revision": "unit-test",
            "git_commit": "unit-test",
            "hardware": "cpu",
            "dtype": "float32",
            "dataset": "synthetic",
            "num_samples": 1,
        },
    )
    decisions = analyze_result_directory(tmp_path)
    assert set(decisions) == {
        "sol_attn_gaussian_threshold",
        "mean_proxy",
        "previous_step_replay",
        "borderline_revalidation",
        "query_alignment",
        "fixed_prefix_cache",
        "live_replay_correctness",
        "skipped_block_approximation",
    }
    assert (tmp_path / "report.md").is_file()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 20
    assert len(list((tmp_path / "tables").glob("*.csv"))) == 17


def test_request_shards_are_loaded_by_offline_analysis(tmp_path):
    config = AttentionObservationConfig(q_block_size=1, kv_block_size=1)
    observer = AttentionObserver(config)
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    for request_index in range(2):
        observer.begin_request(f"shard-{request_index}", num_denoising_steps=2)
        for step in range(2):
            observer.begin_forward(None, None if step == 0 else torch.zeros(1))
            query = torch.tensor([[[[1.0 + step]]]])
            key = torch.tensor([[[[1.0], [2.0], [3.0]]]])
            observer.observe(module, query, key, key, None, scaling=1.0, is_causal=False)
        observer.export_shard(tmp_path / "raw", f"{request_index:04d}")
    observer.export(
        tmp_path,
        {
            "model": "synthetic",
            "model_revision": "test",
            "git_commit": "test",
            "hardware": "cpu",
            "dtype": "float32",
            "dataset": "synthetic",
            "num_samples": 2,
        },
    )
    decisions = analyze_result_directory(tmp_path)
    assert "previous_step_replay" in decisions
    assert not (tmp_path / "tile_records.npz").exists()
    assert len(list((tmp_path / "raw").glob("tile_records_*.npz"))) == 2


def test_replay_routing_keeps_high_proxy_block_and_expands_to_valid_pairs():
    query = torch.ones(1, 1, 2, 1)
    key = torch.tensor([[[[1.0], [1.0], [5.0], [5.0]]]])
    scores = query @ key.transpose(-2, -1)
    valid = torch.ones_like(scores, dtype=torch.bool)
    keep, eligible, tiles = routing_mask(
        query,
        key,
        scores,
        valid,
        proxy="mean",
        beta=0.0,
        q_block_size=2,
        kv_block_size=2,
        scaling=1.0,
    )
    assert eligible.tolist() == [[[[True, True]]]]
    assert keep.tolist() == [[[[False, True]]]]
    allowed = expand_routing_mask(keep, tiles, valid, 2)
    assert allowed[0, 0, 0].tolist() == [False, False, True, True]


def test_empirical_quantile_routing_keeps_exact_per_row_density():
    query = torch.ones(1, 1, 2, 1)
    key = torch.tensor([[[[1.0], [1.0], [3.0], [3.0], [5.0], [5.0]]]])
    scores = query @ key.transpose(-2, -1)
    valid = torch.ones_like(scores, dtype=torch.bool)
    keep, eligible, _ = routing_mask(
        query,
        key,
        scores,
        valid,
        proxy="mean",
        beta=1.28,
        threshold_mode="empirical_quantile",
        target_density=0.5,
        q_block_size=2,
        kv_block_size=2,
        scaling=1.0,
    )
    assert eligible.sum().item() == 3
    assert keep.sum().item() == 2
    assert keep.tolist() == [[[[False, True, True]]]]


def test_previous_replay_applies_prior_fresh_mask():
    module = nn.Module()
    module.layer_idx = 0
    module.num_key_value_groups = 1
    replay = ReplayAttention(
        ReplayConfig(mode="previous", proxy="max", beta=0.0, q_block_size=2, kv_block_size=2)
    )
    replay.begin_request("r")
    query = torch.ones(1, 1, 2, 1)
    value = torch.tensor([[[[1.0], [1.0], [10.0], [10.0]]]])
    replay.begin_forward(None, None)
    first, _ = replay.attention_forward(
        module,
        query,
        torch.tensor([[[[1.0], [1.0], [5.0], [5.0]]]]),
        value,
        None,
        scaling=1.0,
        is_causal=False,
    )
    replay.begin_forward(None, torch.zeros(1))
    second, _ = replay.attention_forward(
        module,
        query,
        torch.tensor([[[[5.0], [5.0], [1.0], [1.0]]]]),
        value,
        None,
        scaling=1.0,
        is_causal=False,
    )
    # Both applied masks retain the second KV block, even though the second
    # step's fresh maximum moved to the first block.
    assert first[0, 0, 0].item() == pytest.approx(10.0)
    assert second[0, 0, 0].item() == pytest.approx(10.0)
    summary = replay.stats.summary()
    assert summary["attention_regimes"]["global"]["fresh_applied_mask_agreement"] == pytest.approx(0.5)
