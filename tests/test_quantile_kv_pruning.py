from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/ruler/kv_pruning/quantile_kv_pruning_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("quantile_kv_pruning_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
OracleTilePruner = MODULE.OracleTilePruner
OracleStats = MODULE.OracleStats


class Attention(torch.nn.Module):
    num_key_value_groups = 1
    layer_idx = 3
    is_sliding = False


def _inputs():
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[4.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]])
    value = torch.tensor([[[[4.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 1.0]]]])
    return query, key, value


def test_dense_override_matches_dense_attention():
    query, key, value = _inputs()
    pruner = OracleTilePruner("dense", 0.0, kv_tile_size=2)
    output, _ = pruner(Attention(), query, key, value, None, scaling=1.0, is_causal=False)
    probability = torch.softmax(query @ key.transpose(-2, -1), dim=-1)
    expected = (probability @ value).transpose(1, 2)
    torch.testing.assert_close(output, expected)


def test_block_max_drops_lowest_half_and_renormalizes():
    query, key, value = _inputs()
    pruner = OracleTilePruner("block_max", 0.5, kv_tile_size=2)
    output, _ = pruner(Attention(), query, key, value, None, scaling=1.0, is_causal=False)
    probability = torch.softmax(torch.tensor([4.0, 3.0]), dim=-1)
    expected = probability[0] * value[0, 0, 0] + probability[1] * value[0, 0, 1]
    torch.testing.assert_close(output[0, 0, 0], expected)
    summary = pruner.stats.summary()["overall"]
    assert summary["eligible_tiles"] == 2
    assert summary["dropped_tiles"] == 1
    assert summary["actual_tile_drop_fraction"] == 0.5


def test_attention_mass_and_output_norm_obey_structural_mask():
    query, key, value = _inputs()
    mask = torch.tensor([[[[True, True, True, False]]]])
    outputs = []
    for method in ("attention_mass", "output_norm"):
        pruner = OracleTilePruner(method, 0.5, kv_tile_size=2, output_norm_query_chunk=1)
        output, _ = pruner(
            Attention(), query, key, value, mask, scaling=1.0, is_causal=False
        )
        outputs.append(output)
        summary = pruner.stats.summary()["overall"]
        assert summary["eligible_tiles"] == 2
        assert summary["dropped_tiles"] == 1
    assert all(torch.isfinite(output).all() for output in outputs)


def test_never_drops_only_valid_tile():
    query, key, value = _inputs()
    mask = torch.tensor([[[[True, True, False, False]]]])
    pruner = OracleTilePruner("block_max", 0.75, kv_tile_size=2)
    output, _ = pruner(Attention(), query, key, value, mask, scaling=1.0, is_causal=False)
    assert torch.isfinite(output).all()
    assert pruner.stats.summary()["overall"]["dropped_tiles"] == 0


def test_aggressive_block_max_uses_complement_and_keeps_highest_tile():
    query = torch.tensor([[[[1.0]]]])
    key = torch.tensor([[[[4.0], [3.0], [2.0], [1.0]]]])
    value = torch.tensor([[[[40.0], [30.0], [20.0], [10.0]]]])
    pruner = OracleTilePruner("block_max", 0.75, kv_tile_size=1)
    output, _ = pruner(
        Attention(), query, key, value, None, scaling=1.0, is_causal=False
    )
    torch.testing.assert_close(output[0, 0, 0], torch.tensor([40.0]))
    summary = pruner.stats.summary()["overall"]
    assert summary["eligible_tiles"] == 4
    assert summary["dropped_tiles"] == 3


def test_aggressive_complement_respects_per_row_valid_tile_counts():
    query = torch.tensor([[[[1.0], [1.0]]]])
    key = torch.tensor([[[[4.0], [3.0], [2.0], [1.0]]]])
    value = torch.tensor([[[[40.0], [30.0], [20.0], [10.0]]]])
    mask = torch.tensor([[[[True, True, True, True], [True, False, False, False]]]])
    pruner = OracleTilePruner("block_max", 0.75, kv_tile_size=1)
    output, _ = pruner(
        Attention(), query, key, value, mask, scaling=1.0, is_causal=False
    )
    torch.testing.assert_close(output[0, 0, 0], torch.tensor([40.0]))
    torch.testing.assert_close(output[0, 1, 0], torch.tensor([40.0]))
    summary = pruner.stats.summary()["overall"]
    assert summary["eligible_tiles"] == 5
    assert summary["dropped_tiles"] == 3


def test_random_policy_is_reproducible_and_independent_of_global_rng():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.tensor([[[
        [4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0],
    ]]])
    value = key.clone()

    outputs = []
    for global_seed in (7, 999):
        torch.manual_seed(global_seed)
        pruner = OracleTilePruner("random", 0.5, kv_tile_size=2, random_seed=123)
        pruner.set_sample_seed(456)
        output, _ = pruner(
            Attention(), query, key, value, None, scaling=1.0, is_causal=False
        )
        outputs.append(output)
        assert pruner.stats.summary()["overall"]["dropped_tiles"] == 4
    torch.testing.assert_close(outputs[0], outputs[1])


def test_random_policy_changes_with_sample_seed():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    value = torch.flip(key, dims=(-2,))
    outputs = []
    for sample_seed in (1, 2):
        pruner = OracleTilePruner("random", 0.5, kv_tile_size=2, random_seed=123)
        pruner.set_sample_seed(sample_seed)
        output, _ = pruner(
            Attention(), query, key, value, None, scaling=0.01, is_causal=False
        )
        outputs.append(output)
    assert not torch.equal(outputs[0], outputs[1])


def test_attention_statistics_round_trip_for_resumable_runs():
    stats = OracleStats()
    stats.update(
        attention_type="global",
        layer=2,
        query_rows=4,
        eligible_tiles=16,
        dropped_tiles=8,
        discarded_mass_sum=0.8,
        discarded_mass_sq_sum=0.2,
    )
    restored = OracleStats.from_summary(stats.summary())
    combined = OracleStats()
    combined.add(restored)
    combined.add(restored)
    summary = combined.summary()["overall"]
    assert summary["query_rows"] == 8
    assert summary["eligible_tiles"] == 32
    assert summary["dropped_tiles"] == 16
    assert abs(summary["discarded_dense_attention_mass_mean"] - 0.2) < 1e-7
