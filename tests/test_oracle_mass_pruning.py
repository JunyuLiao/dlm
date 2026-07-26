import unittest

import torch

from blasst.oracle_mass_pruning import (
    OracleMassPruner,
    OraclePolicyConfig,
    aggregate_tile_mass,
    attention_error_metrics,
    candidate_tiles,
    collect_exact_tile_statistics,
    new_maximum_protected_tiles,
    oracle_mass_pruned_attention,
    select_cumulative_budget,
    select_independent,
    simulate_ordinary_blasst,
)


class OracleMassPruningTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260721)
        self.q = torch.randn(1, 5, 2, 4, generator=generator)
        self.k = torch.randn(1, 9, 1, 4, generator=generator)
        self.v = torch.randn(1, 9, 1, 3, generator=generator)
        self.stats = collect_exact_tile_statistics(
            self.q, self.k, self.v, kv_block_size=4
        )

    def test_final_mass_matches_complete_softmax(self):
        repeated_k = self.k.repeat_interleave(2, dim=2)
        score = torch.einsum("brhd,bkhd->bhrk", self.q, repeated_k) * 0.5
        probability = score.softmax(dim=-1)
        expected = torch.stack(
            (
                probability[..., :4].sum(-1),
                probability[..., 4:8].sum(-1),
                probability[..., 8:].sum(-1),
            ),
            dim=2,
        )
        torch.testing.assert_close(self.stats.final_mass, expected)
        torch.testing.assert_close(
            self.stats.final_mass.sum(dim=2),
            torch.ones_like(self.stats.final_mass[:, :, 0]),
        )

    def test_composition_matches_mask_logits_then_recompute_softmax(self):
        active = torch.tensor([[[True, False, True], [True, False, True]]])
        actual = self.stats.compose(active)
        repeated_k = self.k.repeat_interleave(2, dim=2)
        repeated_v = self.v.repeat_interleave(2, dim=2)
        score = torch.einsum("brhd,bkhd->bhrk", self.q, repeated_k) * 0.5
        score[..., 4:8] = -torch.inf
        expected = torch.einsum(
            "bhrk,bkhd->bhrd", score.softmax(dim=-1), repeated_v
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_candidate_and_aggregation_are_physical(self):
        decisions = simulate_ordinary_blasst(self.stats, 1.0)
        candidates = candidate_tiles(decisions, 5)
        self.assertEqual(candidates.shape, (1, 2, 3))
        all_max = aggregate_tile_mass(self.stats.final_mass, decisions, "all_max")
        veto_max = aggregate_tile_mass(self.stats.final_mass, decisions, "veto_max")
        self.assertEqual(all_max.shape, candidates.shape)
        self.assertTrue(bool((veto_max <= all_max).all()))
        removed = select_independent(candidates, all_max, 2.0)
        self.assertTrue(bool((removed <= decisions.keep).all()))

    def test_cumulative_budget_enforces_every_protected_row(self):
        decisions = simulate_ordinary_blasst(self.stats, 1.0)
        candidates = candidate_tiles(decisions, 5)
        score = aggregate_tile_mass(self.stats.final_mass, decisions, "all_max")
        selected, cumulative = select_cumulative_budget(
            candidates,
            self.stats.final_mass,
            budget=0.4,
            order="score",
            score=score,
        )
        self.assertTrue(bool((cumulative <= 0.4 + 1e-7).all()))
        self.assertTrue(bool((selected <= candidates).all()))

    def test_masked_new_max_protection_is_a_subset_of_all(self):
        decisions = simulate_ordinary_blasst(self.stats, 1.0)
        masked = torch.tensor([[True, False, False, True, False]])
        protected_all = new_maximum_protected_tiles(decisions, "all")
        protected_masked = new_maximum_protected_tiles(decisions, "masked", masked)
        self.assertTrue(bool((protected_masked <= protected_all).all()))

    def test_zero_removal_has_identity_metrics(self):
        output = self.stats.compose(torch.ones_like(self.stats.m[..., 0], dtype=torch.bool))
        metrics = attention_error_metrics(output, output)
        self.assertEqual(metrics.maximum_absolute_error, 0.0)
        self.assertEqual(metrics.mean_relative_error, 0.0)
        self.assertAlmostEqual(metrics.mean_row_cosine, 1.0, places=6)

    def test_end_to_end_oracle_path_matches_manual_selection(self):
        config = OraclePolicyConfig(
            k=4,
            aggregation="all_max",
            selection="independent",
            threshold=2.0,
        )
        pruner = OracleMassPruner(config)
        actual = oracle_mass_pruned_attention(
            self.q,
            self.k,
            self.v,
            blasst_threshold=1.0,
            pruner=pruner,
            q_block_size=5,
            kv_block_size=4,
        )
        decisions = simulate_ordinary_blasst(self.stats, 1.0)
        candidates = candidate_tiles(decisions, 4)
        expected = self.stats.compose(decisions.keep & ~candidates).transpose(1, 2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(pruner.stats.additional_skipped_tiles, int(candidates.sum()))
        self.assertEqual(
            sum(pruner.stats.by_layer_head.values()),
            pruner.stats.additional_skipped_tiles,
        )

    def test_protected_layer_disables_additional_pruning(self):
        pruner = OracleMassPruner(
            OraclePolicyConfig(
                k=4,
                aggregation="all_max",
                selection="independent",
                threshold=2.0,
                protected_layers=(7,),
            )
        )
        ordinary = simulate_ordinary_blasst(self.stats, 1.0)
        output = oracle_mass_pruned_attention(
            self.q,
            self.k,
            self.v,
            blasst_threshold=1.0,
            pruner=pruner,
            layer=7,
            q_block_size=5,
            kv_block_size=4,
        )
        expected = self.stats.compose(ordinary.keep).transpose(1, 2)
        torch.testing.assert_close(output, expected)
        self.assertEqual(pruner.stats.additional_skipped_tiles, 0)


if __name__ == "__main__":
    unittest.main()
