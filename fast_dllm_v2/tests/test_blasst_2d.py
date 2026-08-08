from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from sparse_attention.blasst_2d import (  # noqa: E402
    Blasst2DConfig,
    Blasst2DRuntime,
    Blasst2DStats,
    apply_blasst_2d,
    blasst_2d_attention_forward,
    evaluate_blasst_thresholds,
    slow_blasst_2d,
)


def config(**kwargs) -> Blasst2DConfig:
    return Blasst2DConfig(
        enable_blasst_2d=True,
        q_tile_size=kwargs.pop("q_tile_size", 2),
        kv_tile_size=kwargs.pop("kv_tile_size", 2),
        **kwargs,
    )


def run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Blasst2DConfig,
    **kwargs,
) -> torch.Tensor:
    module = SimpleNamespace(
        _blasst_2d_runtime=Blasst2DRuntime(cfg),
        num_key_value_groups=kwargs.pop("num_key_value_groups", 1),
        layer_idx=0,
        training=False,
    )
    output, _ = blasst_2d_attention_forward(
        module,
        query,
        key,
        value,
        mask,
        scaling=1.0,
        is_causal=False,
        **kwargs,
    )
    return output.transpose(1, 2)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        Blasst2DConfig(blasst_lambda=0)
    with pytest.raises(ValueError):
        Blasst2DConfig(blasst_lambda=1)
    with pytest.raises(ValueError):
        Blasst2DConfig(q_tile_size=0)
    with pytest.raises(ValueError):
        Blasst2DConfig(kv_tile_size=-1)
    assert math.isclose(Blasst2DConfig().log_lambda, math.log(0.5))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_slow_matches_vectorized_random(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator(device=device).manual_seed(7)
    scores = torch.randn((2, 3, 5, 7), generator=generator, device=device)
    valid = torch.rand((2, 1, 5, 7), generator=generator, device=device) > 0.25
    active = torch.tensor(
        [[True, True, True, True, False], [True, True, False, False, False]],
        device=device,
    )
    cfg = config(q_tile_size=3, kv_tile_size=2)
    vector_scores, vector = apply_blasst_2d(scores, valid, active, cfg)
    slow_scores, slow = slow_blasst_2d(scores, valid, active, cfg)
    assert torch.equal(vector.skip_mask, slow.skip_mask)
    assert torch.equal(vector.eligible_mask, slow.eligible_mask)
    assert torch.equal(vector.row_skippable, slow.row_skippable)
    assert torch.equal(vector.row_total, slow.row_total)
    assert torch.equal(vector_scores, slow_scores)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_threshold_sweep_matches_individual_evaluation(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator(device=device).manual_seed(19)
    scores = torch.randn((2, 3, 7, 13), generator=generator, device=device)
    valid = torch.rand((2, 1, 7, 13), generator=generator, device=device) > 0.2
    active = torch.tensor(
        [[True, True, True, True, True, False, False]] * 2,
        device=device,
    )
    lambdas = (1.0e-4, 3.0e-3, 0.1, 0.5)
    swept = evaluate_blasst_thresholds(
        scores,
        valid,
        active,
        lambdas,
        q_tile_size=4,
        kv_tile_size=5,
    )
    previous_skips = -1
    previous_row_votes = -1
    for lambda_value in lambdas:
        _, individual = apply_blasst_2d(
            scores,
            valid,
            active,
            config(
                blasst_lambda=lambda_value,
                q_tile_size=4,
                kv_tile_size=5,
            ),
        )
        shared = swept[lambda_value]
        for field in (
            "skip_mask",
            "eligible_mask",
            "structural_mask",
            "row_skippable",
            "row_total",
            "skipped_valid_elements",
            "valid_elements",
        ):
            assert torch.equal(getattr(shared, field), getattr(individual, field))
        skipped = int(shared.skip_mask.sum())
        row_votes = int(shared.row_skippable.sum())
        assert skipped >= previous_skips
        assert row_votes >= previous_row_votes
        previous_skips = skipped
        previous_row_votes = row_votes


def test_all_rows_skippable_skips_physical_tile() -> None:
    scores = torch.tensor([[[[5.0, 5.0, 0.0, 0.0], [4.0, 4.0, -1.0, -1.0]]]])
    _, decisions = apply_blasst_2d(scores, None, None, config())
    assert decisions.skip_mask[0, 0, 0].tolist() == [False, True]


def test_one_unskippable_row_retains_complete_tile() -> None:
    scores = torch.tensor([[[[5.0, 5.0, 0.0, 0.0], [0.0, 0.0, 6.0, 6.0]]]])
    masked, decisions = apply_blasst_2d(scores, None, None, config())
    assert decisions.skip_mask[0, 0, 0].tolist() == [False, False]
    assert torch.equal(masked, scores)


def test_retained_tile_preserves_every_rows_contribution() -> None:
    query = torch.tensor([[[[1.0], [1.0]]]])
    key = torch.tensor([[[[5.0], [5.0], [0.0], [6.0]]]])
    value = torch.tensor([[[[1.0], [2.0], [100.0], [10.0]]]])
    output = run_attention(query, key, value, None, config())
    dense_scores = query @ key.transpose(-2, -1)
    expected = F.softmax(dense_scores, dim=-1) @ value
    assert torch.allclose(output, expected)
    # The low score in row 0's retained second tile must still contribute.
    assert output[0, 0, 0, 0] > 1.5


def test_first_valid_kv_tile_is_never_skipped() -> None:
    scores = torch.randn(2, 4, 5, 9)
    valid = torch.zeros_like(scores, dtype=torch.bool)
    valid[..., 2:4] = True
    _, decisions = apply_blasst_2d(scores, valid, None, config(kv_tile_size=2))
    assert not decisions.skip_mask[..., 1].any()


def test_causal_rows_without_keys_vote_true_without_nans() -> None:
    scores = torch.tensor([[[[4.0, 0.0, -2.0, -2.0], [4.0, 0.0, 3.5, 3.5]]]])
    valid = torch.tensor(
        [[[[True, True, False, False], [True, True, True, True]]]]
    )
    masked, decisions = apply_blasst_2d(scores, valid, None, config())
    assert not torch.isnan(masked[torch.isfinite(masked)]).any()
    # Row 0 has no key in tile 1 and votes true, row 1 vetoes the skip.
    assert decisions.row_total[0, 0, 0, 1] == 1
    assert not decisions.skip_mask[0, 0, 0, 1]


def test_bidirectional_denoising_block_mask() -> None:
    # Prefix block 0 is visible to all current-block query rows, while all
    # current-block rows attend bidirectionally within block 1.
    scores = torch.tensor([[[[4.0, 4.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    _, decisions = apply_blasst_2d(scores, valid, None, config())
    assert decisions.eligible_mask.all()
    assert decisions.skip_mask[0, 0, 0].tolist() == [False, True]


def test_padding_tail_tiles_exclude_inactive_rows() -> None:
    scores = torch.tensor(
        [[[[5.0, 5.0, 0.0, 0.0], [5.0, 5.0, 6.0, 6.0], [5.0, 5.0, 9.0, 9.0]]]]
    )
    active = torch.tensor([[True, False, False]])
    _, decisions = apply_blasst_2d(
        scores, None, active, config(q_tile_size=4, kv_tile_size=2)
    )
    assert decisions.skip_mask[0, 0, 0, 1]
    assert decisions.row_total[0, 0, 0, 1] == 1


def test_fully_masked_tile_is_structural() -> None:
    scores = torch.zeros((1, 1, 2, 4))
    valid = torch.tensor([[[[True, True, False, False], [True, True, False, False]]]])
    _, decisions = apply_blasst_2d(scores, valid, None, config())
    assert decisions.structural_mask[0, 0, 0, 1]
    assert not decisions.eligible_mask[0, 0, 0, 1]
    assert not decisions.skip_mask[0, 0, 0, 1]


def test_numerical_stability_with_bool_mask() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn((2, 2, 3, 4), generator=generator)
    key = torch.randn((2, 2, 5, 4), generator=generator)
    value = torch.randn((2, 2, 5, 4), generator=generator)
    valid = torch.ones((2, 1, 3, 5), dtype=torch.bool)
    valid[:, :, 0, 3:] = False
    valid[1, :, 2, :] = False
    output = run_attention(query, key, value, valid, config())
    assert torch.isfinite(output).all()
    assert torch.equal(output[1, :, 2], torch.zeros_like(output[1, :, 2]))


def test_batch_and_head_decisions_are_independent() -> None:
    scores = torch.zeros((2, 2, 1, 4))
    scores[..., :2] = 5.0
    scores[..., 2:] = 0.0
    scores[1, 0, 0, 2:] = 6.0
    scores[0, 1, 0, 2:] = 7.0
    _, decisions = apply_blasst_2d(
        scores, None, None, config(q_tile_size=1, kv_tile_size=2)
    )
    assert decisions.skip_mask[:, :, 0, 1].tolist() == [
        [True, False],
        [False, True],
    ]


def test_gqa_mapping_matches_explicit_repeat() -> None:
    query = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
    key = torch.tensor([[[[1.0], [0.0]], [[0.0], [1.0]]]])
    value = torch.tensor([[[[1.0], [2.0]], [[10.0], [20.0]]]])
    output = run_attention(
        query,
        key,
        value,
        None,
        config(q_tile_size=1, kv_tile_size=1),
        num_key_value_groups=2,
    )
    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    scores = query @ repeated_key.transpose(-2, -1)
    # First KV tile is retained. Compute the same physical skips explicitly.
    masked, _ = apply_blasst_2d(scores, None, None, config(q_tile_size=1, kv_tile_size=1))
    expected = F.softmax(masked, dim=-1) @ repeated_value
    assert torch.allclose(output, expected)


def test_stats_consistency_and_exports(tmp_path: Path) -> None:
    scores = torch.tensor([[[[5.0, 5.0, 0.0, 0.0]]]])
    _, decisions = apply_blasst_2d(
        scores, None, None, config(q_tile_size=1, kv_tile_size=2)
    )
    stats = Blasst2DStats()
    stats.record(
        decisions,
        layer=3,
        query_length=2,
        sequence_length=4,
        metadata={
            "denoising_step": 2,
            "mask_ratio": 0.5,
            "masked_tokens": 1,
            "valid_kv_length": 3,
        },
        dump_trace=True,
    )
    summary = stats.summary()
    assert summary["eligible_tiles"] == (
        summary["skipped_tiles"] + summary["retained_tiles"]
    )
    for name in (
        "physical_tile_sparsity",
        "row_vote_sparsity",
        "valid_element_sparsity",
    ):
        assert 0 <= summary[name] <= 1
    cfg = config(collect_blasst_stats=True, dump_blasst_trace=True)
    stats.export(tmp_path, cfg, {"seed": 7})
    for filename in (
        "summary.json",
        "per_step.csv",
        "per_layer.csv",
        "per_head.csv",
        "run_config.json",
        "trace.json",
    ):
        assert (tmp_path / filename).is_file()
    assert json.loads((tmp_path / "run_config.json").read_text())["seed"] == 7
    assert next(iter(stats.per_step.values()))["valid_kv_length"] == 3
    assert next(iter(stats.per_layer.values()))["denoising_step"] == 2
    assert next(iter(stats.per_head.values()))["mask_ratio"] == 0.5


def test_stats_can_disable_high_cardinality_layer_and_head_rows() -> None:
    scores = torch.tensor([[[[5.0, 5.0, 0.0, 0.0]]]])
    _, decisions = apply_blasst_2d(
        scores, None, None, config(q_tile_size=1, kv_tile_size=2)
    )
    stats = Blasst2DStats(record_layers=False, record_heads=False)
    stats.record(
        decisions,
        layer=4,
        query_length=1,
        sequence_length=4,
        metadata={
            "experiment": "context_length",
            "sweep_value": 512,
            "benchmark": "gsm8k",
            "example_id": "gsm8k_test_1",
        },
    )
    assert stats.per_step
    row = next(iter(stats.per_step.values()))
    assert row["benchmark"] == "gsm8k"
    assert row["example_id"] == "gsm8k_test_1"
    assert not stats.per_layer
    assert not stats.per_head


def test_vectorized_per_head_counts_match_literal_slices() -> None:
    generator = torch.Generator().manual_seed(73)
    scores = torch.randn((2, 3, 5, 9), generator=generator)
    valid = torch.rand((2, 1, 5, 9), generator=generator) > 0.2
    _, decisions = apply_blasst_2d(
        scores,
        valid,
        None,
        config(q_tile_size=3, kv_tile_size=4),
    )
    vectorized = Blasst2DStats._per_head_tensor_counts(decisions)
    literal = [
        Blasst2DStats._tensor_counts(
            decisions,
            (slice(0, decisions.skip_mask.shape[0]), head),
        )
        for head in range(decisions.skip_mask.shape[1])
    ]
    assert vectorized == literal


def test_feature_disabled_dispatch_calls_original_dense() -> None:
    sentinel = torch.randn(1, 2, 3)
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel, None

    runtime = Blasst2DRuntime(Blasst2DConfig(enable_blasst_2d=False))
    module = SimpleNamespace(_blasst_2d_runtime=runtime)

    def dispatch(module, *args, **kwargs):
        tagged = getattr(module, "_blasst_2d_runtime", None)
        if tagged is None or not tagged.config.enable_blasst_2d:
            return original(module, *args, **kwargs)
        return blasst_2d_attention_forward(module, *args, **kwargs)

    output, _ = dispatch(module, "dense-input", marker=True)
    assert output is sentinel
    assert calls and calls[0][0][1] == "dense-input"
