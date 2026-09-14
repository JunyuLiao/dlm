from __future__ import annotations

import json
import csv

import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.calibration import (
    DenseMarginTraceCollector,
    candidate_lambdas,
    evaluate_margin_trace,
    fit_blasst_policy,
)
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.config import (
    ANALYTIC_BETAS,
    ExperimentConfig,
    canonical_conditions,
)
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.dataset import build_disjoint_manifests
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import (
    aggregate_routing_stats,
    positional_token_id_agreement,
    retained_dense_attention_mass,
)
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.report import EXPECTED_CONDITIONS, build_report
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.routing import (
    blasst_route,
    mean_pooled_tile_proxies,
    sol_gaussian_route,
)
from experiments.diffusion_attention_threshold_modeling.proxy import sol_attention_proxy_rows
from dllm.attention.blasst.core import (
    Blasst2DConfig,
    Blasst2DStats,
    apply_blasst_2d,
    evaluate_blasst_thresholds,
    slow_blasst_2d,
)


def test_analytic_betas_and_skip_direction():
    assert ANALYTIC_BETAS == {0.25: -0.674490, 0.5: 0.0, 0.75: 0.674490, 0.9: 1.281552}
    q = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.ones(1, 1, 64, 1), torch.full((1, 1, 64, 1), 10.0)), dim=-2)
    valid = torch.ones(1, 1, 64, 128, dtype=torch.bool)
    routed = sol_gaussian_route(q, key, valid, target_sparsity=0.5, prefix_length=0, scaling=1.0)
    assert routed.stats["skipped_tiles"] == 1
    assert routed.proxies[0, 0, 0, 1] > routed.proxies[0, 0, 0, 0]
    assert bool(routed.allowed[..., 64:].any())


def test_mean_proxy_partial_mask_and_gqa_expansion():
    q = torch.tensor([[[[1.0], [3.0]], [[2.0], [4.0]]]])
    key = torch.tensor([[[[2.0], [10.0], [4.0], [8.0]]]])
    expanded = key[:, :, None].expand(1, 1, 2, 4, 1).reshape(1, 2, 4, 1)
    valid = torch.ones(1, 2, 2, 4, dtype=torch.bool)
    valid[..., 0, 3] = False
    proxies, eligible = mean_pooled_tile_proxies(q, expanded, valid, scaling=0.5)
    assert proxies[0, 0, 0, 0].item() == 6.0
    assert proxies[0, 1, 0, 0].item() == 9.0
    assert bool(eligible.all())


def test_sol_standardizes_prefix_and_canvas_as_one_population_and_handles_degenerate_rows():
    q = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.ones(1, 1, 64, 1), torch.full((1, 1, 64, 1), 3.0)), dim=-2)
    valid = torch.ones(1, 1, 64, 128, dtype=torch.bool)
    result = sol_gaussian_route(q, key, valid, target_sparsity=0.5, prefix_length=64, scaling=1.0)
    assert result.stats["prefix"]["eligible_tiles"] == 1
    assert result.stats["canvas"]["eligible_tiles"] == 1
    assert result.stats["skipped_tiles"] == 1
    degenerate = sol_gaussian_route(q, torch.ones_like(key), valid, target_sparsity=0.9, prefix_length=64, scaling=1.0)
    assert degenerate.stats["degenerate_rows"] == 1
    assert degenerate.stats["skipped_tiles"] == 0


def test_canonical_sol_global_kv_tiling_keeps_non_aligned_prefix_tile_indivisible():
    module = SimpleNamespace(num_key_value_groups=1)
    query = torch.ones(1, 1, 2, 1)
    key = torch.arange(130, dtype=torch.float32).reshape(1, 1, 130, 1)
    rows = sol_attention_proxy_rows(
        module,
        query,
        key,
        key,
        None,
        q_block_size=2,
        kv_block_size=64,
        prefix_length=65,
        scaling=1.0,
        is_causal=False,
        global_kv_tiling=True,
    )
    assert len(rows) == 1
    # The globally aligned tile [64, 128) straddles the 65-token prefix
    # boundary and is classified by its start position for diagnostics; it is
    # still one indivisible candidate in the shared population.
    assert rows[0]["prefix_spans"] == [(0, 64), (64, 128)]
    assert rows[0]["canvas_spans"] == [(128, 130)]
    assert len(rows[0]["prefix"]) + len(rows[0]["canvas"]) == 3


def test_blasst_max_pooling_monotonicity_and_physical_vote():
    scores = torch.tensor([[[[5.0, 4.0, 0.0, 0.0], [5.0, 4.0, 0.0, 0.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    low = blasst_route(scores, valid, 0.1, q_tile_size=2, kv_tile_size=2)
    high = blasst_route(scores, valid, 0.99, q_tile_size=2, kv_tile_size=2)
    assert high.stats["skipped_tiles"] >= low.stats["skipped_tiles"]
    assert high.stats["valid_qk_element_sparsity"] >= low.stats["valid_qk_element_sparsity"]
    assert high.stats["eligible_tiles"] == 2


def test_disjoint_manifests_are_deterministic_and_balanced():
    tasks = ("niah_multikey_1", "niah_multivalue", "niah_multiquery", "vt", "fwe")
    rows = {task: [{"source_id": index, "prompt": f"{task}-{index}", "token_count": 4, "tokens_to_generate": 2} for index in range(12)] for task in tasks}
    first = build_disjoint_manifests(rows, calibration_per_task=2, final_per_task=10, split_seed=42)
    second = build_disjoint_manifests(rows, calibration_per_task=2, final_per_task=10, split_seed=42)
    assert first == second
    calibration, final, audit = first
    assert len(calibration) == 10 and len(final) == 50 and audit["disjoint"]
    assert all(sum(row["task"] == task for row in calibration) == 2 for task in tasks)
    assert all(sum(row["task"] == task for row in final) == 10 for task in tasks)


def test_calibration_grid_and_policy_are_shared_by_attention_type():
    trace = {
        "attention_type": "global",
        "valid_kv_length": 128,
        "margins": [[float("inf"), -2.0, -0.5, 0.5]],
        "valid_rows": [[True, True, True, True]],
        "eligible_tiles": [True, True, True, True],
        "valid_elements": [[64, 64, 64, 64]],
    }
    assert len(candidate_lambdas()) == 41
    assert evaluate_margin_trace(trace, 0.5)["physical_sparsity"] >= 0.0
    policy = fit_blasst_policy([trace, {**trace, "attention_type": "local"}])
    assert set(policy["lambda_global"]) == {"0.25", "0.5", "0.75", "0.9"}
    assert set(policy["lambda_local"]) == {"0.25", "0.5", "0.75", "0.9"}


def test_dense_margin_observer_accepts_registry_callback_shape():
    collector = DenseMarginTraceCollector()
    module = nn.Module()
    module.num_key_value_groups = 1
    query = torch.ones(1, 1, 2, 1)
    key = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
    value = key.clone()
    collector(module, query, key, value, None, scaling=1.0, is_causal=False, blasst_metadata={"attention_type": "global"})
    assert len(collector.traces) == 1
    assert collector.traces[0]["attention_type"] == "global"
    assert len(collector.traces[0]["margins"]) == 2


def test_dense_margin_observer_uses_runtime_attention_type_metadata():
    collector = DenseMarginTraceCollector()
    module = nn.Module()
    module.num_key_value_groups = 1
    module._blasst_2d_runtime = SimpleNamespace(metadata={"attention_type": "local"})
    q = torch.ones(1, 1, 2, 1)
    k = torch.ones(1, 1, 4, 1)
    collector(module, q, k, k, None, scaling=1.0, is_causal=False)
    assert collector.traces[0]["attention_type"] == "local"


def test_metrics_cover_mass_token_agreement_and_region_aggregation():
    scores = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]]]])
    retained = torch.tensor([[[[True, False], [False, True]]]])
    assert retained_dense_attention_mass(scores, retained) > 0.5
    assert positional_token_id_agreement([1, 2, 3], [1, 2]) == 2 / 3
    aggregate = aggregate_routing_stats([
        {"attention_type": "global", "eligible_tiles": 4, "skipped_tiles": 1, "retained_tiles": 3, "valid_qk_elements": 16, "skipped_valid_qk_elements": 4, "routing_rows": 2, "prefix": {"eligible_tiles": 2, "skipped_tiles": 1}},
        {"attention_type": "local", "eligible_tiles": 2, "skipped_tiles": 1, "retained_tiles": 1, "valid_qk_elements": 8, "skipped_valid_qk_elements": 2, "routing_rows": 1, "canvas": {"eligible_tiles": 2, "skipped_tiles": 1}},
    ])
    assert aggregate["overall"]["full_tile_sparsity"] == 2 / 6
    assert aggregate["global"]["full_tile_sparsity"] == 0.25
    assert aggregate["regions"]["prefix"]["skipped_tiles"] == 1


def test_report_regenerates_local_global_stats_and_audits_shared_identity(tmp_path):
    for condition in EXPECTED_CONDITIONS:
        condition_dir = tmp_path / condition
        condition_dir.mkdir()
        predictions = [
            {
                "sample_id": f"sample-{index}",
                "prompt_hash": f"prompt-{index}",
                "inference_seed": index,
                "completion_tokens": [1, 2],
            }
            for index in range(50)
        ]
        (condition_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )
        condition_metadata = {"name": condition}
        if condition.startswith("sol_"):
            condition_metadata["beta_used"] = 0.0
        elif condition.startswith("blasst_"):
            condition_metadata.update(lambda_local=0.2, lambda_global=0.3)
        (condition_dir / "summary.json").write_text(
            json.dumps({"condition": condition_metadata, "official_ruler_accuracy": 0.5}),
            encoding="utf-8",
        )
        if condition.startswith("blasst_"):
            stats_dir = condition_dir / "attention_stats"
            stats_dir.mkdir()
            with (stats_dir / "per_step.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "attention_type", "eligible_tiles", "skipped_tiles", "retained_tiles",
                        "valid_elements", "skipped_valid_elements", "retained_dense_attention_mass",
                        "retained_attention_mass_rows", "region_counts",
                    ),
                )
                writer.writeheader()
                writer.writerow({
                    "attention_type": "global", "eligible_tiles": 4, "skipped_tiles": 1,
                    "retained_tiles": 3, "valid_elements": 16, "skipped_valid_elements": 4,
                    "retained_dense_attention_mass": 0.9, "retained_attention_mass_rows": 2,
                    "region_counts": "{'prefix': {'eligible_tiles': 2, 'skipped_tiles': 1, 'retained_tiles': 1}}",
                })
                writer.writerow({
                    "attention_type": "local", "eligible_tiles": 2, "skipped_tiles": 1,
                    "retained_tiles": 1, "valid_elements": 8, "skipped_valid_elements": 2,
                    "retained_dense_attention_mass": 0.8, "retained_attention_mass_rows": 1,
                    "region_counts": "{'canvas': {'eligible_tiles': 2, 'skipped_tiles': 1, 'retained_tiles': 1}}",
                })
    result = build_report(tmp_path)
    assert result["audit"]["all_nine_conditions_present"]
    assert result["audit"]["all_conditions_have_50_examples"]
    assert result["audit"]["all_conditions_share_prompts_and_seeds"]
    blasst = next(row for row in result["conditions"] if row["condition"] == "blasst_calibrated_s25")
    assert blasst["global"]["full_tile_sparsity"] == 0.25
    assert blasst["local"]["full_tile_sparsity"] == 0.5
    dense = next(row for row in result["conditions"] if row["condition"] == "dense")
    assert dense["retained_attention_mass"] == 1.0
    assert (tmp_path / "sol_vs_blasst.csv").exists()
    assert (tmp_path / "global_local.csv").exists()
    assert any(path.endswith("target_vs_achieved_sparsity.png") for path in result["plots"])


def test_blasst_stats_export_round_trip_preserves_mass_and_region_counts(tmp_path):
    config = Blasst2DConfig(q_tile_size=2, kv_tile_size=2, collect_blasst_stats=True)
    scores = torch.tensor([[[[3.0, 2.0, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    _, decisions = apply_blasst_2d(scores, valid, None, config, blasst_lambda=0.5)
    stats = Blasst2DStats()
    stats.record(
        decisions,
        layer=0,
        query_length=2,
        sequence_length=4,
        metadata={"attention_type": "global"},
        retained_dense_attention_mass=0.75,
        retained_mass_rows=2,
        region_counts={"prefix": {"eligible_tiles": 1, "skipped_tiles": 0, "retained_tiles": 1}},
    )
    stats.export(tmp_path, config)
    loaded = Blasst2DStats.load_export(tmp_path)
    assert loaded.summary()["retained_dense_attention_mass"] == 0.75
    assert loaded.summary()["retained_attention_mass_rows"] == 2
    assert loaded.summary()["region_counts"]["prefix"]["retained_tiles"] == 1
    assert len(loaded.per_step) == 1
    loaded.export(tmp_path / "resumed", config)
    assert (tmp_path / "resumed" / "per_step.csv").exists()


def test_blasst_core_compares_against_preceding_running_maximum():
    scores = torch.tensor([[[[10.0, 9.0, 9.0, 8.5, 8.0, 7.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    decisions = evaluate_blasst_thresholds(
        scores, valid, None, [0.3], q_tile_size=1, kv_tile_size=2
    )[0.3]
    assert decisions.skip_mask.tolist() == [[[[False, False, True]]]]


def test_active_query_masks_trim_cached_prefix_positions():
    scores = torch.zeros(1, 1, 2, 4)
    valid = torch.ones_like(scores, dtype=torch.bool)
    decisions = evaluate_blasst_thresholds(
        scores, valid, torch.tensor([[False, False, True, True]]), [0.5], q_tile_size=1, kv_tile_size=2
    )[0.5]
    assert decisions.row_total.sum().item() == 4


def test_blasst_element_sparsity_counts_only_wholly_skipped_tiles():
    scores = torch.tensor([[[[5.0, 4.0, 0.0, 0.0], [1.0, 0.0, 4.0, 3.0]]]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    masked, decisions = apply_blasst_2d(
        scores,
        valid,
        None,
        Blasst2DConfig(q_tile_size=2, kv_tile_size=2),
        blasst_lambda=0.5,
    )
    assert not bool(decisions.skip_mask[..., 1].any())
    assert decisions.skipped_valid_elements.sum().item() == 0
    # Row votes remain diagnostic, but both execution and element counts
    # follow the physical mask: this mixed-vote tile is fully retained.
    assert torch.equal(masked, scores)
    assert decisions.row_skip_mask[0, 0, 0, 1]
    assert not decisions.row_skip_mask[0, 0, 1, 1]
    slow_masked, slow_decisions = slow_blasst_2d(
        scores,
        valid,
        None,
        Blasst2DConfig(q_tile_size=2, kv_tile_size=2, blasst_lambda=0.5),
    )
    assert torch.equal(masked, slow_masked)
    assert torch.equal(decisions.skip_mask, slow_decisions.skip_mask)
    assert torch.equal(decisions.skipped_valid_elements, slow_decisions.skipped_valid_elements)


def test_blasst_attention_forward_keeps_all_rows_of_mixed_vote_tile():
    query = torch.eye(2).reshape(1, 1, 2, 2)
    key = torch.tensor([[[[5.0, 1.0], [4.0, 0.0], [0.0, 4.0], [0.0, 3.0]]]])
    value = torch.tensor([[[[1.0], [1.0], [100.0], [100.0]]]])
    module = SimpleNamespace(
        _blasst_2d_runtime=SimpleNamespace(
            config=Blasst2DConfig(
                q_tile_size=2,
                kv_tile_size=2,
                blasst_lambda=0.5,
                apply_blasst_mask=True,
                collect_blasst_stats=True,
            ),
            active_query_mask=None,
            dense_kv_prefix_extractor=None,
            metadata={},
            stats=Blasst2DStats(),
        ),
        num_key_value_groups=1,
        layer_idx=0,
        training=False,
    )
    output, _ = __import__("dllm.attention.blasst.core", fromlist=["blasst_2d_attention_forward"]).blasst_2d_attention_forward(
        module, query, key, value, None, scaling=1.0, is_causal=False
    )
    # Row zero votes to skip the second tile, but row one retains it.
    # Both tiles must be dense for both rows, including their value products.
    dense = torch.softmax(query @ key.transpose(-2, -1), dim=-1) @ value
    assert torch.equal(output, dense.transpose(1, 2))
    summary = module._blasst_2d_runtime.stats.summary()
    assert summary['skipped_tiles'] == summary['skipped_valid_elements'] == 0
    assert summary['skippable_row_votes'] > 0
    assert abs(summary['retained_dense_attention_mass'] - 1.) < 1.e-6
