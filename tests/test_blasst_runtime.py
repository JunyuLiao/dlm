import math
import unittest

import torch

from blasst.flash_attention import blasst_flash_attn_func, collect_blasst_stats
from blasst.runtime import (
    DenoisingMetadataStore,
    aggregate_step_metadata,
    build_kv_priority_order,
    build_query_permutation,
    structural_blasst_reference,
)


class RuntimePlanningTest(unittest.TestCase):
    def metadata(self, rows=133, tiles=65):
        skip = torch.zeros(3, rows, tiles, dtype=torch.bool)
        skip[:, ::2, ::3] = True
        scores = torch.arange(tiles).float()[None, None].expand_as(skip).clone()
        return aggregate_step_metadata(skip, scores)

    def test_signatures_use_multiple_words_and_aggregation_modes(self):
        skip = torch.zeros(3, 2, 65, dtype=torch.bool)
        skip[:2, 0, 64] = True
        scores = torch.randn(3, 2, 65)
        majority = aggregate_step_metadata(skip, scores, aggregation="majority")
        all_heads = aggregate_step_metadata(skip, scores, aggregation="all")
        any_head = aggregate_step_metadata(skip, scores, aggregation="any")
        self.assertEqual(majority.row_signature.shape, (2, 2))
        self.assertEqual(int(majority.row_signature[0, 1]), 1)
        self.assertEqual(int(all_heads.row_signature[0, 1]), 0)
        self.assertEqual(int(any_head.row_signature[0, 1]), 1)

    def test_local_permutation_never_crosses_parent_tiles_and_inverts(self):
        metadata = self.metadata()
        mask = torch.arange(133).remainder(3).eq(0)
        perm, inverse = build_query_permutation(
            mask, mode="previous_signature_local", previous=metadata
        )
        torch.testing.assert_close(inverse[perm], torch.arange(133))
        self.assertTrue(bool((perm[:128] < 128).all()))
        self.assertTrue(bool((perm[128:] >= 128).all()))

    def test_first_step_fallback_is_deterministic_and_keeps_valid_rows(self):
        mask = torch.tensor([True, False, True, False, False])
        first = build_query_permutation(mask, mode="previous_signature_local")
        second = build_query_permutation(mask, mode="previous_signature_local")
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1][first[0]], torch.arange(mask.numel()))

    def test_oracle_hamming_is_a_valid_diagnostic_permutation(self):
        metadata = self.metadata(rows=17, tiles=65)
        mask = torch.arange(17).remainder(2).bool()
        perm, inverse = build_query_permutation(
            mask, mode="oracle_hamming_debug", previous=metadata
        )
        torch.testing.assert_close(inverse[perm], torch.arange(17))

    def test_priority_prefix_is_unique_complete_and_zero_is_reverse(self):
        metadata = self.metadata(rows=128, tiles=65)
        self.assertEqual(
            build_kv_priority_order(5, mode="previous_step_priority", priority_prefix_k=0),
            [4, 3, 2, 1, 0],
        )
        order = build_kv_priority_order(
            65,
            mode="previous_step_priority_plus_local",
            priority_prefix_k=8,
            previous=metadata,
            visible_fraction=torch.linspace(0, 1, 65),
        )
        self.assertEqual(len(order), 65)
        self.assertEqual(sorted(order), list(range(65)))
        self.assertEqual(
            order,
            build_kv_priority_order(
                65,
                mode="previous_step_priority_plus_local",
                priority_prefix_k=8,
                previous=metadata,
                visible_fraction=torch.linspace(0, 1, 65),
            ),
        )

    def test_metadata_double_buffer_is_request_isolated_and_cleans_up(self):
        store = DenoisingMetadataStore()
        one, two = self.metadata(rows=4, tiles=2), self.metadata(rows=5, tiles=2)
        store.write_current("request-a", 0, one)
        self.assertIsNone(store.previous("request-a", 0))
        store.commit_step()
        self.assertIs(store.previous("request-a", 0), one)
        store.write_current("request-b", 0, two)
        store.commit_step()
        self.assertIsNone(store.previous("request-a", 0))
        self.assertIs(store.previous("request-b", 0), two)
        store.remove_request("request-b")
        self.assertIsNone(store.previous("request-b", 0))


class StructuralReferenceTest(unittest.TestCase):
    def tensors(self, batch=1, length=35, heads=2, dim=8):
        torch.manual_seed(11)
        q = torch.randn(batch, length, heads, dim)
        return q, torch.randn_like(q), torch.randn_like(q)

    def test_lambda_zero_permuted_and_reordered_matches_dense(self):
        q, k, v = self.tensors(batch=2)
        permutations = [torch.randperm(35), torch.randperm(35)]
        orders = [[[0]] for _ in range(2)]
        actual = structural_blasst_reference(
            q,
            k,
            v,
            blasst_lambda=0.0,
            query_permutations=permutations,
            kv_orders=orders,
            q_block_size=128,
            skip_group_rows=32,
        ).output
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertFalse(bool(torch.isnan(actual).any()))

    def test_features_disabled_match_calibrated_parent_tile_reference(self):
        q, k, v = self.tensors(length=67, heads=1)
        baseline = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.3, q_block_size=64, kv_block_size=16
        )
        collect_blasst_stats(reset=True)
        structural = structural_blasst_reference(
            q,
            k,
            v,
            blasst_lambda=0.3,
            q_block_size=64,
            kv_block_size=16,
            skip_group_rows=64,
        ).output
        torch.testing.assert_close(structural, baseline, atol=2e-5, rtol=2e-5)

    def test_strict_threshold_equality_keeps_group(self):
        q = torch.ones(1, 16, 1, 1)
        k = torch.cat((torch.full((1, 8, 1, 1), -2.0), torch.zeros(1, 8, 1, 1)), dim=1)
        v = torch.randn_like(k)
        result = structural_blasst_reference(
            q,
            k,
            v,
            blasst_lambda=math.exp(-2.0),
            kv_block_size=8,
            skip_group_rows=16,
        )
        self.assertFalse(bool(result.row_skip_by_head[0][:, :, 0].any()))

    def test_permuted_attention_bias_uses_original_query_rows(self):
        q, k, v = self.tensors(length=17, heads=1)
        bias = torch.randn(1, 1, 17, 17) * 0.1
        permutation = torch.randperm(17)
        actual = structural_blasst_reference(
            q,
            k,
            v,
            blasst_lambda=0.0,
            query_permutations=[permutation],
            attention_bias=bias,
        ).output
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1]) + bias
        expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_all_skip_group_sizes_and_non_divisible_lengths(self):
        for length in (7, 65, 129):
            q, k, v = self.tensors(length=length, heads=1)
            for group in (16, 32, 64, 128):
                result = structural_blasst_reference(
                    q, k, v, blasst_lambda=0.3, skip_group_rows=group
                )
                self.assertEqual(result.output.shape, q.shape)
                self.assertGreater(result.stats.microgroups_total, 0)

    def test_requested_lambda_sweep_is_finite_for_every_group_size(self):
        q, k, v = self.tensors(length=17, heads=1)
        lambdas = (0.001, 0.003, 0.01, 0.03, 0.04858582466840744, 0.1,
                   0.3, 0.4334796965122223, 1.0)
        for threshold in lambdas:
            for group in (16, 32, 64, 128):
                output = structural_blasst_reference(
                    q, k, v, blasst_lambda=threshold, skip_group_rows=group
                ).output
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_microgroup_sparsity_is_distinct_from_parent_and_v_sparsity(self):
        q = torch.tensor([[[[1.0]], [[-1.0]]] + [[[1.0]]] * 30])
        k = torch.cat((torch.full((1, 16, 1, 1), -4.0), torch.full((1, 16, 1, 1), 4.0)), dim=1)
        v = torch.randn_like(k)
        result = structural_blasst_reference(
            q, k, v,
            blasst_lambda=0.5, kv_block_size=16, skip_group_rows=16,
        )
        rates = result.stats.rates()
        self.assertIn("microgroup_skip_rate", rates)
        self.assertIn("full_parent_tile_skip_rate", rates)
        self.assertIn("v_load_skip_rate", rates)
        self.assertGreater(rates["microgroup_skip_rate"], rates["full_parent_tile_skip_rate"])

    def test_heterogeneous_lengths_leave_padding_untouched(self):
        q, k, v = self.tensors(batch=2, length=33)
        result = structural_blasst_reference(
            q, k, v, blasst_lambda=0.0, sequence_lengths=[17, 33], skip_group_rows=16
        )
        self.assertTrue(bool((result.output[0, 17:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
