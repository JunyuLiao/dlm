from __future__ import annotations

import numpy as np
import json
import pytest
import torch
from types import SimpleNamespace
from scipy import stats
from torch import nn

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.attention.blasst.core import (
    _finish_eager_attention,
    _prepare_attention_scores,
)
from dllm.models.adapters.fast_dllm_v2 import FastDLLMV2Adapter
from fast_dllm_v2.generation_functions import _filter_finished_cache
from experiments.diffusion_attention_threshold_modeling.datasets import (
    balanced_ruler_selection,
    stratified_split,
)
from experiments.diffusion_attention_threshold_modeling.collect import (
    _distribution_prefix_extractor,
)
from experiments.diffusion_attention_threshold_modeling.fitting import (
    evaluate_candidate,
    fit_family,
    fit_gaussian_mixture,
    weighted_quantile,
    fit_threshold_models,
)
from experiments.diffusion_attention_threshold_modeling.math500 import (
    condition_grid,
    paired_bootstrap_delta,
)
from experiments.diffusion_attention_threshold_modeling.proxy import (
    CollectorConfig,
    PromptShardCollector,
    deterministic_reservoir,
    hierarchical_row_weights,
    sol_attention_proxy_rows,
)
from experiments.diffusion_attention_threshold_modeling.routing import (
    FreshRoutingAttention,
    FreshRoutingConfig,
)
from experiments.diffusion_attention_threshold_modeling.report import build_report
from experiments.diffusion_attention_threshold_modeling.ruler_report import build_ruler_report
from experiments.diffusion_attention_threshold_modeling.canonical_report import build_canonical_report


def test_fast_dllm_finished_short_prompt_without_prefix_cache():
    finished = torch.tensor([True, False])
    assert _filter_finished_cache(None, finished) is None

    class Cache:
        key_cache = [torch.tensor([[1], [2]])]
        value_cache = [torch.tensor([[3], [4]])]

        def __len__(self):
            return 1

    cache = Cache()
    _filter_finished_cache(cache, finished)
    assert cache.key_cache[0].tolist() == [[2]]
    assert cache.value_cache[0].tolist() == [[4]]


def test_distribution_collection_overrides_legacy_diffusion_gemma_prefix_policy():
    class Adapter:
        def blasst_dense_kv_prefix(self, *args, **kwargs):
            return 7

    adapter = Adapter()
    assert _distribution_prefix_extractor("diffusion_gemma", adapter) is None
    extractor = _distribution_prefix_extractor("fast_dllm_v2", adapter)
    assert extractor is not None
    assert extractor() == 7


def test_sol_proxy_exact_gqa_masking_and_prefix_separation():
    module = nn.Module()
    module.num_key_value_groups = 2
    query = torch.tensor([[[[1.0], [3.0]], [[2.0], [4.0]]]])
    key = torch.tensor([[[[2.0], [10.0], [4.0], [8.0]]]])
    value = torch.ones_like(key)
    mask = torch.zeros(1, 1, 2, 4)
    mask[..., 0, 3] = -torch.inf
    rows = sol_attention_proxy_rows(
        module, query, key, value, mask,
        q_block_size=64, kv_block_size=64, prefix_length=2,
        scaling=0.5, is_causal=False,
    )
    assert len(rows) == 2  # one real expanded-GQA row per query head
    assert rows[0]["prefix_spans"] == [(0, 2)]
    assert rows[0]["canvas_spans"] == [(2, 4)]
    # mean Q=2, mean prefix K=6, scale=.5
    assert rows[0]["prefix"] == pytest.approx([6.0])
    # K=8 is excluded because it has no valid pair for q0 but remains valid for
    # q1, so both Q/K positions participate: mean K=6.
    assert rows[0]["canvas"] == pytest.approx([6.0])
    assert rows[1]["prefix"] == pytest.approx([9.0])


def test_collector_exports_compact_exact_histograms_and_coverage(tmp_path):
    collector = PromptShardCollector(CollectorConfig(reservoir_per_row=1))
    collector.begin_prompt(request_id="p", corpus="math500", task="algebra", split="calibration")
    module = nn.Module(); module.layer_idx = 2; module.num_key_value_groups = 1
    query = torch.ones(1, 1, 192, 1)
    key = torch.cat((torch.ones(1, 1, 64, 1), torch.arange(1, 193).view(1, 1, 192, 1).float()), dim=-2)
    collector(module, query, key, key, None, scaling=1.0, is_causal=False)
    path = collector.export_shard(tmp_path / "p.npz")
    with np.load(path) as payload:
        assert payload["canvas_tiles"].tolist() == [3, 3, 3]
        assert payload["prefix_tiles"].tolist() == [1, 1, 1]
        assert payload["canvas_reservoir"].shape == (3, 1)
        assert int(payload["canvas_histogram"].sum() + payload["canvas_underflow"].sum() + payload["canvas_overflow"].sum()) == 9
        assert payload["hierarchical_row_weight"].tolist() == pytest.approx([1 / 3] * 3)


def test_fast_dllm_v2_short_prompt_denoising_and_dense_prefix():
    adapter = FastDLLMV2Adapter("checkpoint", device="cpu")
    mask = 151665
    ids = torch.tensor([[10, 11, mask, mask]])
    metadata = {"generation_block_index": 0, "denoising_iteration": 0}
    kwargs = {
        "input_ids": ids,
        "past_key_values": None,
        "update_past_key_values": False,
        "use_block_cache": False,
        "blasst_metadata": metadata,
    }
    assert adapter.blasst_call_is_eligible(None, (), kwargs)
    query = torch.empty(1, 1, 4, 2)
    key = torch.empty(1, 1, 12, 2)
    assert adapter.blasst_dense_kv_prefix(
        None, query, key, key, None, {"blasst_metadata": metadata}
    ) == 10  # eight cached tokens plus two prompt-remainder tokens
    assert not adapter.blasst_call_is_eligible(
        None, (), {**kwargs, "blasst_metadata": None, "update_past_key_values": True}
    )

    # A new generation's step zero must reset the block-local remainder.
    next_ids = torch.tensor([[12, mask, mask, mask]])
    adapter.blasst_call_is_eligible(None, (), {**kwargs, "input_ids": next_ids})
    assert adapter.blasst_dense_kv_prefix(
        None, query, key, key, None, {"blasst_metadata": metadata}
    ) == 9


def test_fast_dllm_v2_rejects_unsupported_transformers(monkeypatch):
    monkeypatch.setattr(
        "dllm.models.adapters.fast_dllm_v2.transformers.__version__", "5.12.1"
    )
    adapter = FastDLLMV2Adapter("checkpoint", device="cpu")
    with pytest.raises(ImportError, match="transformers==4.53.1"):
        adapter.load()


def test_hierarchical_weights_equalize_every_level():
    base = dict(layer=0, head=0, query_block=0)
    rows = [
        {**base, "request_id": "a", "denoising_call": 0},
        {**base, "request_id": "a", "denoising_call": 1},
        {**base, "request_id": "a", "denoising_call": 1, "layer": 1},
        {**base, "request_id": "b", "denoising_call": 0},
    ]
    assert hierarchical_row_weights(rows).tolist() == pytest.approx([0.25, 0.125, 0.125, 0.5])


def test_reservoir_balancing_and_stratified_splits_are_deterministic():
    values = np.arange(100.0)
    first = deterministic_reservoir(values, 8, identity=("row", 1), seed=7)
    second = deterministic_reservoir(values, 8, identity=("row", 1), seed=7)
    assert np.array_equal(first, second)
    rows = [
        {"request_id": f"{task}-{index}", "prompt": "x", "task": task, "corpus": "ruler8k"}
        for task in ("a", "b") for index in range(10)
    ]
    selected = balanced_ruler_selection(rows, 10, 42)
    assert [row["task"] for row in selected] == ["a", "b"] * 5
    split1 = stratified_split(selected, 9)
    split2 = stratified_split(selected, 9)
    assert [(r["request_id"], r["split"]) for r in split1] == [(r["request_id"], r["split"]) for r in split2]
    assert sum(row["split"] == "validation" for row in split1) == 2


def test_empirical_quantile_and_synthetic_family_recovery():
    assert weighted_quantile(np.array([0.0, 1.0, 2.0]), np.ones(3), 0.5) == pytest.approx(1.0)
    rng = np.random.default_rng(4)
    gaussian = rng.normal(size=3000)
    weights = np.ones(len(gaussian)) / len(gaussian)
    fitted = fit_family("gaussian", gaussian, weights, (0.25, 0.5, 0.75))
    assert fitted["betas"]["0.75"] < 0  # conservative density allows negative beta
    empirical = fit_family("empirical_cdf", gaussian, weights, (0.25,))
    assert empirical["betas"]["0.25"] == pytest.approx(np.quantile(gaussian, .75), abs=.03)
    skewed = stats.jf_skew_t.rvs(7, 3, size=2500, random_state=rng)
    skew_fit = fit_family("jones_faddy_skew_t", skewed, np.ones(len(skewed)), (0.25,))
    assert skew_fit["parameters"]["a"] > skew_fit["parameters"]["b"]
    mixture_values = np.r_[rng.normal(-2, .3, 1600), rng.normal(2, .5, 2400)]
    mixture = fit_gaussian_mixture(mixture_values)
    assert mixture.means == pytest.approx((-2, 2), abs=.12)
    assert mixture.weights == pytest.approx((.4, .6), abs=.05)


def _synthetic_record(prompt, split, corpus, values):
    edges = np.linspace(-8, 8, 321)
    hist = np.histogram(values, bins=edges)[0]
    return {
        "adapter": "model", "corpus": corpus, "task": "task", "split": split,
        "request_id": prompt, "attention_type": "global", "denoising_call": 0,
        "layer": 0, "head": 0, "query_block": 0, "canvas_tiles": len(values),
        "row_weight": 1.0, "reservoir": np.asarray(values), "histogram_edges": edges,
        "histogram": hist, "underflow": 0, "overflow": 0, "row_max_z": max(values),
    }


def test_density_gate_passes_well_calibrated_gaussian_rows():
    values = stats.norm.ppf((np.arange(200) + .5) / 200)
    records = []
    for corpus in ("ruler8k", "math500"):
        for index in range(10):
            records.append(_synthetic_record(f"{corpus}-{index}", "calibration" if index < 8 else "validation", corpus, values))
    _, evaluations, passed = evaluate_candidate(
        records, "model", "gaussian", (0.25, 0.5, 0.75), bootstrap_repeats=200, seed=1
    )
    assert passed
    assert all(row["absolute_error"] <= 0.01 for row in evaluations)


def test_fresh_oracle_routing_keeps_prefix_dense_and_exact_canvas_density():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    query = torch.ones(1, 1, 128, 1)
    key = torch.cat((torch.full((1, 1, 2, 1), 10.0), torch.ones(1, 1, 64, 1), torch.full((1, 1, 64, 1), 4.0)), dim=-2)
    value = key.clone()
    router = FreshRoutingAttention(FreshRoutingConfig(mode="oracle", target_density=.5))
    output, _ = router(module, query, key, value, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert summary["logical_density"] == 0.5
    assert 0.0 < summary["retained_dense_attention_mass"] <= 1.0
    assert output.shape == (1, 128, 1, 1)


def test_fresh_all_kv_routing_reports_regions_fallback_and_no_empty_rows():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.full((1, 1, 64, 1), 10.0), torch.ones(1, 1, 64, 1)), dim=-2)
    value = key.clone()
    router = FreshRoutingAttention(FreshRoutingConfig(mode="gaussian", target_density=.1, region="all"))
    output, _ = router(module, query, key, value, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert torch.isfinite(output).all()
    assert set(summary["region_counts"]) == {"prefix", "canvas"}
    assert summary["logical_candidate_tiles"] == 2
    assert summary["logical_retained_tiles"] >= 2  # deterministic top-one fallback per row/region
    assert summary["fallback_rows"] == 0
    assert summary["degenerate_rows"] == 2


def test_fresh_combined_region_population_standardizes_prefix_and_canvas_together():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.ones(1, 1, 64, 1), torch.full((1, 1, 64, 1), 3.0)), dim=-2)
    value = key.clone()
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="gaussian", target_density=.5, region="all", combined_region_population=True
    ))
    router(module, query, key, value, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert summary["routing_rows"] == 1
    assert summary["logical_candidate_tiles"] == 2
    assert summary["logical_retained_tiles"] == 1


def test_prefix_only_routes_prefix_and_keeps_canvas_dense():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((
        torch.ones(1, 1, 64, 1),
        torch.full((1, 1, 64, 1), 2.0),
        torch.full((1, 1, 64, 1), 7.0),
    ), dim=-2)
    value = torch.arange(192.0).view(1, 1, 192, 1)
    module._blasst_2d_runtime = SimpleNamespace(
        dense_kv_prefix_extractor=lambda *args: 128,
    )
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="oracle", target_density=.1, region="prefix_only"
    ))
    routed, _ = router(module, query, key, value, None, scaling=1.0, is_causal=False)

    expanded_key, expanded_value, scores, valid = _prepare_attention_scores(
        module, query, key, value, None, scaling=1.0, is_causal=False,
        sliding_window=None,
    )
    allowed = torch.zeros_like(valid)
    allowed[..., 64:192] = valid[..., 64:192]  # canvas remains dense
    allowed[..., 64:128] = valid[..., 64:128]  # top prefix block is retained
    expected, _ = _finish_eager_attention(
        query, expanded_value, scores.masked_fill(valid & ~allowed, -torch.inf),
        allowed, 0.0, False,
    )
    assert torch.allclose(routed, expected)
    assert router.stats.summary()["region_counts"] == {
        "prefix": {"candidate_tiles": 2, "retained_tiles": 1}
    }


def test_diffusion_gemma_prefix_classification_ignores_dense_sink_policy():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    module._blasst_2d_runtime = SimpleNamespace(
        dense_kv_prefix_extractor=lambda *args: 0,
        current_denoising_iteration=0,
    )
    query = torch.ones(1, 1, 64, 1)
    key = torch.cat((torch.ones(1, 1, 128, 1), torch.full((1, 1, 64, 1), 3.0)), dim=-2)
    value = key.clone()
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="oracle", target_density=.5, region="prefix_only",
        adapter="diffusion_gemma",
    ))
    router(module, query, key, value, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert summary["region_counts"] == {
        "prefix": {"candidate_tiles": 2, "retained_tiles": 1}
    }


def test_random_routing_is_deterministic_and_hits_exact_row_count():
    values = np.arange(10.0)
    config = FreshRoutingConfig(mode="random", target_density=.25, region="canvas", random_seed=19)
    from experiments.diffusion_attention_threshold_modeling.routing import _keep_vector
    first = _keep_vector(values, config, "global", random_identity=(1, 2, 3))[0]
    second = _keep_vector(values, config, "global", random_identity=(1, 2, 3))[0]
    assert np.array_equal(first, second)
    assert int(first.sum()) == 3


def test_physical_sparse_reference_agrees_with_logical_masked_attention():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(1, 2, 65, 8, generator=generator)
    key = torch.randn(1, 2, 129, 8, generator=generator)
    value = torch.randn(1, 2, 129, 8, generator=generator)
    logical = FreshRoutingAttention(FreshRoutingConfig(
        mode="oracle", target_density=.5, region="all", execution="logical"
    ))
    physical = FreshRoutingAttention(FreshRoutingConfig(
        mode="oracle", target_density=.5, region="all", execution="physical"
    ))
    logical_output, _ = logical(
        module, query, key, value, None, scaling=.25, is_causal=False
    )
    physical_output, _ = physical(
        module, query, key, value, None, scaling=.25, is_causal=False
    )
    assert torch.allclose(logical_output, physical_output, atol=2e-6, rtol=2e-6)
    assert logical.stats.logical_retained_tiles == physical.stats.logical_retained_tiles


def test_logical_and_physical_100_percent_retention_match_dense():
    module = nn.Module(); module.layer_idx = 0; module.num_key_value_groups = 1
    generator = torch.Generator().manual_seed(32)
    query = torch.randn(1, 1, 17, 4, generator=generator)
    key = torch.randn(1, 1, 17, 4, generator=generator)
    value = torch.randn(1, 1, 17, 4, generator=generator)
    _, expanded_value, scores, valid = _prepare_attention_scores(
        module, query, key, value, None, scaling=.5, is_causal=False,
        sliding_window=None,
    )
    dense, _ = _finish_eager_attention(query, expanded_value, scores, valid, 0.0, False)
    for execution in ("logical", "physical"):
        router = FreshRoutingAttention(FreshRoutingConfig(
            mode="gaussian", target_density=1.0, region="all", execution=execution
        ))
        output, _ = router(module, query, key, value, None, scaling=.5, is_causal=False)
        assert torch.allclose(output, dense, atol=2e-6, rtol=2e-6)
        assert router.stats.logical_retained_tiles == router.stats.logical_candidate_tiles


def test_fresh_router_records_consecutive_mask_overlap_without_reuse():
    module = nn.Module(); module.layer_idx = 3; module.num_key_value_groups = 1
    query = torch.ones(1, 1, 64, 2)
    key = torch.cat((torch.ones(1, 1, 64, 2), torch.full((1, 1, 64, 2), 2.0)), dim=-2)
    value = key.clone()
    router = FreshRoutingAttention(FreshRoutingConfig(
        mode="oracle", target_density=.5, region="all"
    ))
    router(module, query, key, value, None, scaling=1.0, is_causal=False)
    router(module, query, key, value, None, scaling=1.0, is_causal=False)
    summary = router.stats.summary()
    assert summary["mask_overlap_comparisons"] == 1
    assert summary["consecutive_mask_jaccard"] == 1.0
    assert summary["calls"] == 2
    assert summary["prefill_attention_calls"] == 2
    assert summary["denoising_attention_calls"] == 0


def test_ruler_report_handles_single_model_bundle(tmp_path):
    root = tmp_path / "smoke_model"
    dense = [{"sample_id": "a", "task": "niah_multikey_1", "task_base": "niah", "outputs": ["yes"], "prediction": "yes"}]
    routed = [{**dense[0], "routing_stats": {"calls": 1, "logical_candidate_tiles": 4, "logical_retained_tiles": 3, "physical_candidate_tiles": 4, "physical_retained_tiles": 3, "retained_dense_attention_mass": .9, "attention_output_relative_error": .1, "region_counts": {"prefix": {"candidate_tiles": 4, "retained_tiles": 3}}}}]
    from dllm.evaluation.ruler.io import write_json
    from dllm.evaluation.ruler.io import write_jsonl
    write_jsonl(root / "dense" / "predictions.jsonl", dense)
    write_json(root / "dense" / "summary.json", {"official_ruler_accuracy": 1.0})
    write_jsonl(root / "gaussian_rho75" / "predictions.jsonl", routed)
    write_json(root / "gaussian_rho75" / "summary.json", {"condition": {"name": "gaussian_rho75"}, "official_ruler_accuracy": 1.0, "routing_aggregate": {"logical_density": .75, "physical_density": .75, "retained_dense_attention_mass": .9, "attention_output_relative_error": .1}})
    result = build_ruler_report(root, "/tmp/NVIDIA-RULER")
    assert result["models"][0]["model"] == "smoke_model"
    assert (root / "ruler_report.md").exists()


def test_canonical_report_uses_revised_required_set_and_merges_revised_dirs(tmp_path):
    root = tmp_path / "bundle"
    dist = tmp_path / "ruler16k_prefix_proxy_distribution"
    from dllm.evaluation.ruler.io import write_json
    from dllm.evaluation.ruler.io import write_jsonl

    write_json(dist / "summary.json", {"population": "prefix_only", "groups": {}})
    write_json(dist / "audit.json", {"population": "prefix KV tiles only"})
    write_json(root / "profiled_threshold_model.json", {
        "simple_profiling_passed": False,
        "selected": {"scope": "model_attention", "family": "empirical_cdf", "fits": {}},
    })

    required = {
        "dense",
        "gaussian_rho75", "gaussian_rho50", "gaussian_rho25",
        "profiled_rho75", "profiled_rho50", "profiled_rho25",
        "oracle_rho75", "oracle_rho50", "oracle_rho25",
        "gaussian_rho75_prefix_only", "gaussian_rho50_prefix_only", "gaussian_rho25_prefix_only",
        "profiled_rho75_prefix_only", "profiled_rho50_prefix_only", "profiled_rho25_prefix_only",
        "oracle_rho75_prefix_only", "oracle_rho50_prefix_only", "oracle_rho25_prefix_only",
        "gaussian_rho10", "oracle_rho10",
    }
    rows = [{"sample_id": str(i)} for i in range(50)]
    for model in ("diffusion_gemma", "fast_dllm_v2"):
        for condition in required - {"oracle_rho50_prefix_only"}:
            write_jsonl(root / model / condition / "predictions.jsonl", rows)
            write_json(root / model / condition / "summary.json", {"condition": {"name": condition}, "actual_num_samples": 50})
        write_jsonl(root / model / "random_rho50" / "predictions.jsonl", rows[:7])
        write_jsonl(root / f"{model}_decision_revised" / "oracle_rho50_prefix_only" / "predictions.jsonl", rows)
        write_json(root / f"{model}_decision_revised" / "oracle_rho50_prefix_only" / "summary.json", {"condition": {"name": "oracle_rho50_prefix_only"}, "actual_num_samples": 50})

    result = build_canonical_report(root)
    assert result["routing_completeness"]["complete"] is True
    for status in result["routing"].values():
        assert not status["missing_conditions"]
        assert "random_rho50" in status["partial_conditions"]
        assert "random_rho50" in status["intentionally_skipped_conditions"]
        assert "oracle_rho50_prefix_only" in status["completed_conditions"]


ALL_ATTENTION_FUNCTIONS = {}


class BoundAttention(nn.Module):
    def __init__(self):
        super().__init__(); self.layer_idx = 0; self.num_key_value_groups = 1
    def forward(self, q, k, v, mask):
        return ALL_ATTENTION_FUNCTIONS["sdpa"](self, q, k, v, mask, scaling=.5, is_causal=False)[0]


class BoundModel(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = BoundAttention()
    def forward(self, input_ids=None, *, q, k, v, mask):
        return self.attn(q, k, v, mask)


def _native(module, q, k, v, mask, *, scaling, **kwargs):
    del module, kwargs
    probability = torch.softmax((q @ k.transpose(-2, -1) * scaling) + mask, dim=-1)
    return (probability @ v).transpose(1, 2), None


def test_generic_binding_observer_returns_native_output_bitwise():
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _native
    model = BoundModel()
    seen = []
    q = torch.randn(1, 1, 2, 3); k = torch.randn(1, 1, 4, 3); v = torch.randn_like(k); mask = torch.zeros(1, 1, 2, 4)
    baseline = model(input_ids=torch.ones(1, 2, dtype=torch.long), q=q, k=k, v=v, mask=mask)
    binding = install_blasst(
        model, Blasst2DConfig(enable_blasst_2d=True, collect_blasst_stats=False),
        attention_class_names=("BoundAttention",), attention_observer=lambda *a, **k: seen.append((a, k)),
        integration="registry",
    )
    observed = model(input_ids=torch.ones(1, 2, dtype=torch.long), q=q, k=k, v=v, mask=mask)
    binding.close()
    assert torch.equal(baseline, observed)
    assert len(seen) == 1
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is _native


def test_math500_grid_is_exactly_ten_and_paired_bootstrap_uses_problems():
    assert len(condition_grid()) == 10
    dense = np.array([1, 1, 0, 0])
    sparse = np.array([1, 0, 1, 0])
    delta, low, high = paired_bootstrap_delta(dense, sparse, seed=1, repeats=1000)
    assert delta == 0.0
    assert low <= 0 <= high


def test_synthetic_end_to_end_fit_plot_report_and_audit(tmp_path):
    shard_root = tmp_path / "shards" / "model" / "ruler8k"
    shard_root.mkdir(parents=True)
    values = stats.norm.ppf((np.arange(200) + .5) / 200)
    edges = np.linspace(-8, 8, 321)
    hist = np.histogram(values, bins=edges)[0]
    for index in range(10):
        split = "calibration" if index < 8 else "validation"
        path = shard_root / f"{index:04d}.npz"
        np.savez_compressed(
            path,
            request_id=np.asarray([f"p{index}"]), corpus=np.asarray(["ruler8k"]),
            task=np.asarray(["task"]), split=np.asarray([split]),
            attention_type=np.asarray(["global"]), denoising_call=np.asarray([0]),
            layer=np.asarray([0]), head=np.asarray([0]), query_block=np.asarray([0]),
            canvas_tiles=np.asarray([len(values)]), hierarchical_row_weight=np.asarray([1.0]),
            canvas_reservoir=values[np.linspace(0, len(values) - 1, 32, dtype=int)][None, :], histogram_edges=edges,
            canvas_histogram=hist[None, :], canvas_underflow=np.asarray([0]),
            canvas_overflow=np.asarray([0]), row_max_z=np.asarray([values.max()]),
            prefix_reservoir=np.full((1, 32), np.nan),
        )
        path.with_suffix(".json").write_text(json.dumps({"adapter": "model"}), encoding="utf-8")
    collection = tmp_path / "collection" / "model" / "ruler8k"
    collection.mkdir(parents=True)
    (collection / "metadata.json").write_text(json.dumps({
        "num_prompts": 10, "parity": {"passed": True}, "convergence": {"passed": True}
    }), encoding="utf-8")
    (collection / "convergence.json").write_text(json.dumps({"curves": [], "passed": True}), encoding="utf-8")
    fitted = fit_threshold_models(tmp_path, bootstrap_repeats=100, seed=1, require_complete_design=False)
    assert fitted["selected"]["family"] == "gaussian"
    report = build_report(tmp_path)
    assert report["audit"]["all_prompt_shards_present"]
    assert report["audit"]["fitting_uses_calibration_prompts_only"]
    assert (tmp_path / "report.md").is_file()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 13
