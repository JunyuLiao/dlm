import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import torch

from blasst.flash_attention import blasst_flash_attn_func
from proxy.proxy_blasst_reference import proxy_blasst_reference
from proxy.proxy_policy import (
    Decision, DisabledPolicy, MetadataStore, ProxyContext, ScoreThresholdPolicy,
    SourceMetadata, ThresholdTable, load_calibrated_thresholds,
)
from proxy.calibrate_proxy_thresholds import select_certified_threshold, wilson_upper_bound
from scripts.collect_proxy_traces import cyclic_contexts
from tracing import IncrementalTraceCollector, TraceContext, load_trace_directory


class FirstTileSkipPolicy:
    name = "first-tile"

    def decide(self, context, sources):
        # The predictor contract has coordinates but no target QK statistic.
        assert not hasattr(context, "score")
        assert not hasattr(context, "running_max")
        return Decision.PRE_SKIP if context.traversal_index == 0 else Decision.EXACT


class ResolverAuditPolicy:
    name = "audit"

    def __init__(self):
        self.relations = []

    def decide(self, context, sources):
        self.relations.append("previous_step")
        sources.resolve(context, "previous_step")
        return Decision.EXACT


class ProxyReferenceTest(unittest.TestCase):
    def tensors(self):
        generator = torch.Generator().manual_seed(8)
        q = torch.randn(2, 17, 2, 4, generator=generator)
        return q, torch.randn(q.shape, generator=generator), torch.randn(q.shape, generator=generator)

    def test_disabled_proxy_matches_existing_blasst(self):
        q, k, v = self.tensors()
        expected = blasst_flash_attn_func(q, k, v, blasst_lambda=0.3, q_block_size=8, kv_block_size=4)
        actual = proxy_blasst_reference(
            q, k, v, blasst_lambda=0.3, policy=DisabledPolicy(), q_block_size=8, kv_block_size=4
        )
        torch.testing.assert_close(actual.output, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(actual.stats.pre_skipped_tiles, 0)

    def test_pre_skip_does_not_update_running_max_or_apply_pv(self):
        q = torch.ones(1, 2, 1, 1)
        # Reverse traversal would see value 10 first. It is PRE_SKIPped, so
        # output must contain only the value from the remaining tile.
        k = torch.tensor([[[[0.0]], [[10.0]]]])
        v = torch.tensor([[[[3.0]], [[99.0]]]])
        result = proxy_blasst_reference(
            q, k, v, blasst_lambda=0.5, policy=FirstTileSkipPolicy(),
            q_block_size=2, kv_block_size=1,
        )
        torch.testing.assert_close(result.output, torch.full_like(result.output, 3.0))
        self.assertEqual(result.stats.pre_skipped_tiles, 1)
        self.assertEqual(result.stats.qk_tiles_computed, 1)

    def test_oracle_target_is_not_exposed_to_policy(self):
        q, k, v = self.tensors()
        oracle = MetadataStore()
        context = ProxyContext("0", 0, 0, 0, 0, "mid", 0, 0, 0, 1)
        oracle.write(context, SourceMetadata(score=123.0, skipped=False, introduced_new_max=True))
        policy = ResolverAuditPolicy()
        proxy_blasst_reference(
            q[:, :2, :1], k[:, :2, :1], v[:, :2, :1],
            blasst_lambda=0.3, policy=policy, oracle_targets=oracle,
            q_block_size=2, kv_block_size=2,
        )
        self.assertTrue(policy.relations)
        self.assertEqual(set(policy.relations), {"previous_step"})

    def test_gqa_exact_fallback_matches_dense_reference(self):
        generator = torch.Generator().manual_seed(9)
        q = torch.randn(1, 9, 4, 4, generator=generator)
        k = torch.randn(1, 9, 2, 4, generator=generator)
        v = torch.randn(1, 9, 2, 4, generator=generator)
        actual = proxy_blasst_reference(q, k, v, blasst_lambda=0.0, q_block_size=4, kv_block_size=3).output
        kr, vr = k.repeat_interleave(2, 2), v.repeat_interleave(2, 2)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2)
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


class TraceTest(unittest.TestCase):
    def test_incremental_trace_is_physical_and_versioned(self):
        context = TraceContext(
            sample_ids=["sample"], request_ids=["request"], seeds=[7], sequence_length=8,
            diffusion_block=1, denoising_iteration=2, remaining_mask_ratios=[0.5],
            query_heads=2, kv_heads=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            with IncrementalTraceCollector(
                directory, context, shard_records=1, q_block_size=4, kv_block_size=2,
                bitpack_masks=True, quantize_log_scores=4,
            ) as collector:
                q = torch.randn(1, 8, 2, 4)
                blasst_flash_attn_func(
                    q, torch.randn(1, 8, 1, 4), torch.randn(1, 8, 1, 4),
                    blasst_lambda=0.3, q_block_size=4, kv_block_size=2,
                    tile_trace_callback=lambda q_tile, kv_tile, trace: collector(3, q_tile, kv_tile, trace),
                )
            records = load_trace_directory(directory)
            self.assertEqual(len(records), 2 * 2 * 4)
            self.assertEqual(set(records["layer"].tolist()), {3})
            self.assertEqual(set(records["kv_group"].tolist()), {0})
            self.assertFalse(np.isnan(records["row_q50"]).any())
            self.assertEqual(set(records["noise_bucket"].tolist()), {"mid"})

    def test_short_corpus_is_rotated_and_repeated_to_native_context(self):
        rows = cyclic_contexts([1, 2, 3], context_length=8, num_contexts=3)
        self.assertEqual(rows.shape, (3, 8))
        self.assertEqual(rows[0].tolist(), [1, 2, 3, 1, 2, 3, 1, 2])
        self.assertNotEqual(rows[0].tolist(), rows[1].tolist())

    def test_spatial_sampling_is_shared_across_steps_layers_and_heads(self):
        context = TraceContext(
            sample_ids=["shared"], request_ids=["shared"], seeds=[1], sequence_length=8,
            diffusion_block=0, denoising_iteration=0, remaining_mask_ratios=[0.9],
            query_heads=2, kv_heads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = IncrementalTraceCollector(directory, context, sample_fraction=0.37)
            first = collector._selected("shared:2:3")
            # The selection identity intentionally contains no step/layer/head.
            self.assertEqual(first, collector._selected("shared:2:3"))
            collector.close()


class ConservativeCalibrationTest(unittest.TestCase):
    def test_wilson_bound_requires_real_support(self):
        self.assertGreater(wilson_upper_bound(0, 1000, 0.99), 1e-3)
        self.assertLess(wilson_upper_bound(0, 10000, 0.99), 1e-3)
        scores = np.linspace(0.0, 1.0, 10000)
        safe = np.ones(10000, dtype=bool)
        no_new_max = np.zeros(10000, dtype=bool)
        selection = select_certified_threshold(
            scores, safe, no_new_max, 1e-3,
            new_max_budget=1e-3, confidence=0.99, min_predictions=512,
        )
        self.assertTrue(selection.certified)
        self.assertEqual(selection.predictions, 10000)

    def test_transition_threshold_does_not_transfer_to_other_noise_pair(self):
        context = ProxyContext("s", 0, 0, 0, 1, "mid", 0, 0, 0, 1)
        store = MetadataStore()
        previous = ProxyContext("s", 0, 0, 0, 0, "high", 0, 0, 0, 1)
        store.write(previous, SourceMetadata(0.1, True, False, noise_bucket="high"))
        policy = ScoreThresholdPolicy(
            "previous_step", ThresholdTable({(0, 0, "high", "mid"): 0.2})
        )
        self.assertEqual(policy.decide(context, store), Decision.PRE_SKIP)
        wrong_transition = ProxyContext("s", 0, 0, 0, 1, "low", 0, 0, 0, 1)
        self.assertEqual(policy.decide(wrong_transition, store), Decision.EXACT)

    def test_loader_ignores_uncertified_thresholds(self):
        payload = {
            "deployment_schema_eligible": True,
            "deployable_thresholds": [
                {"relation": "previous_step", "false_skip_budget": 0.001, "certified": True,
                 "layer": -1, "head": -1, "source_noise_bucket": "high",
                 "target_noise_bucket": "mid", "threshold": 0.2},
                {"relation": "previous_step", "false_skip_budget": 0.001, "certified": False,
                 "layer": 0, "head": 0, "source_noise_bucket": "high",
                 "target_noise_bucket": "mid", "threshold": 0.9},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            table = load_calibrated_thresholds(path, relation="previous_step", false_skip_budget=1e-3)
        self.assertEqual(table.values, {(-1, -1, "high", "mid"): 0.2})

    def test_loader_rejects_legacy_trace_calibration(self):
        payload = {
            "deployment_schema_eligible": False,
            "deployable_thresholds": [
                {"relation": "previous_step", "false_skip_budget": 0.001, "certified": True,
                 "layer": -1, "head": -1, "source_noise_bucket": "high",
                 "target_noise_bucket": "mid", "threshold": 0.2},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            table = load_calibrated_thresholds(path, relation="previous_step", false_skip_budget=1e-3)
        self.assertFalse(table.values)


if __name__ == "__main__":
    unittest.main()
