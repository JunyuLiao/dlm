import unittest

import torch

from blasst.certified_omission import (
    LogNormalizerInterval,
    advance_log_normalizer_interval,
    attention_output_after_omission,
    certified_tile_mass_upper,
    greedy_mass_budget_omission,
    refresh_log_normalizer_interval,
    temporal_score_perturbation_bound,
)
from blasst.tile_reuse import attention_tile_statistics, compose_attention_tile_statistics


class CertifiedOmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261005)

    def test_temporal_bound_contains_every_score_change(self) -> None:
        q0 = torch.randn(2, 5, 8)
        k0 = torch.randn(2, 3, 4, 8)
        q1 = q0 + 0.03 * torch.randn_like(q0)
        k1 = k0 + 0.04 * torch.randn_like(k0)
        scale = 8**-0.5
        bound = temporal_score_perturbation_bound(q1, q0, k1, k0, scale=scale)
        old = torch.einsum("brd,btkd->btrk", q0, k0) * scale
        new = torch.einsum("brd,btkd->btrk", q1, k1) * scale
        actual = (new - old).abs().amax(-1)
        self.assertTrue(torch.all(actual <= bound + 2e-6))

    def test_propagated_mass_bound_survives_omission_and_refresh(self) -> None:
        q0 = torch.randn(1, 6, 8)
        k0 = torch.randn(1, 4, 5, 8)
        scale = 8**-0.5

        def exact(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            scores = torch.einsum("brd,btkd->btrk", q, k) * scale
            return torch.logsumexp(scores, -1).transpose(-1, -2).transpose(-1, -2)

        # exact() already returns [batch, tile, row]; the explicit helper keeps
        # the test's shape contract obvious.
        z0 = exact(q0, k0)
        state = LogNormalizerInterval.exact(z0)
        q1, k1 = q0 + 0.02 * torch.randn_like(q0), k0 + 0.02 * torch.randn_like(k0)
        eps1 = temporal_score_perturbation_bound(q1, q0, k1, k0, scale=scale)
        candidate1 = advance_log_normalizer_interval(state, eps1)
        z1 = exact(q1, k1)
        self.assertTrue(torch.all(candidate1.lower <= z1 + 2e-6))
        self.assertTrue(torch.all(candidate1.upper >= z1 - 2e-6))
        mass1 = torch.softmax(z1, dim=-2)
        self.assertTrue(torch.all(mass1 <= certified_tile_mass_upper(candidate1) + 2e-6))
        omitted = torch.tensor([[True, False, True, False]])
        state = refresh_log_normalizer_interval(candidate1, z1, omitted)

        q2, k2 = q1 + 0.02 * torch.randn_like(q1), k1 + 0.02 * torch.randn_like(k1)
        eps2 = temporal_score_perturbation_bound(q2, q1, k2, k1, scale=scale)
        candidate2 = advance_log_normalizer_interval(state, eps2)
        z2 = exact(q2, k2)
        self.assertTrue(torch.all(candidate2.lower <= z2 + 3e-6))
        self.assertTrue(torch.all(candidate2.upper >= z2 - 3e-6))

    def test_greedy_selection_respects_aggregate_row_budget(self) -> None:
        mass_upper = torch.rand(3, 7, 5) * 0.003
        selected, accumulated = greedy_mass_budget_omission(mass_upper, 0.01)
        explicit = (mass_upper * selected[..., None]).sum(-2)
        torch.testing.assert_close(accumulated, explicit)
        self.assertTrue(torch.all(accumulated <= 0.01))

    def test_omission_composition_matches_direct_masked_attention(self) -> None:
        q = torch.randn(2, 6, 8)
        k = torch.randn(2, 20, 8)
        v = torch.randn(2, 20, 8)
        tiles = [
            attention_tile_statistics(q, k[:, start : start + 5], v[:, start : start + 5])
            for start in range(0, 20, 5)
        ]
        m = torch.stack([tile.m for tile in tiles], -2)
        l = torch.stack([tile.l for tile in tiles], -2)
        u = torch.stack([tile.u for tile in tiles], -3)
        omitted = torch.tensor([[False, True, False, True], [True, False, False, True]])
        actual = attention_output_after_omission(m, l, u, omitted)
        expected = torch.stack(
            [
                compose_attention_tile_statistics(
                    [tile for index, tile in enumerate(tiles) if not omitted[batch, index]]
                ).normalized_output()[batch]
                for batch in range(2)
            ]
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
