import unittest

import torch

from blasst.oracle_mass_pruning import candidate_tiles, simulate_ordinary_blasst
from blasst.tile_replacement_reference import (
    ReplacementComponents,
    collect_tile_replacement_statistics,
    compose_tile_replacements,
    exact_conditional_value,
    exact_log_z,
    exact_replacement_components,
    replace_physical_tiles,
    replacement_rows,
)


class TileReplacementReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260721)
        self.q = torch.randn(1, 6, 2, 4, generator=generator)
        self.k = torch.randn(1, 9, 1, 4, generator=generator)
        self.v = torch.randn(1, 9, 1, 3, generator=generator)
        self.stats = collect_tile_replacement_statistics(
            self.q, self.k, self.v, kv_block_size=4
        )
        self.decisions = simulate_ordinary_blasst(self.stats, 1.0)

    def explicit_output(self, active: torch.Tensor) -> torch.Tensor:
        k = self.k.repeat_interleave(2, dim=2)
        v = self.v.repeat_interleave(2, dim=2)
        score = torch.einsum("brhd,bkhd->bhrk", self.q, k) * 0.5
        for tile in range(active.shape[-1]):
            if not bool(active[0, 0, tile]):
                score[:, 0, :, tile * 4 : min((tile + 1) * 4, 9)] = -torch.inf
            if not bool(active[0, 1, tile]):
                score[:, 1, :, tile * 4 : min((tile + 1) * 4, 9)] = -torch.inf
        return torch.einsum("bhrk,bkhd->bhrd", score.softmax(-1), v)

    def test_no_replacement_is_identity(self):
        actual = compose_tile_replacements(self.stats, self.decisions.keep)
        expected = self.explicit_output(self.decisions.keep)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_exact_all_row_replacement_reconstructs_blasst(self):
        selected = candidate_tiles(self.decisions, 6)
        replacement = exact_replacement_components(
            self.stats, self.decisions, selected, scope="all_rows"
        )
        actual = replace_physical_tiles(
            self.stats, self.decisions, selected, replacement
        )
        expected = compose_tile_replacements(self.stats, self.decisions.keep)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_multiple_replacements_match_explicit_mask_and_restore(self):
        selected = self.decisions.keep.clone()
        selected[..., 0] = False
        rows = selected[..., None].expand_as(self.stats.m)
        replacement = ReplacementComponents(
            exact_log_z(self.stats), exact_conditional_value(self.stats), rows
        )
        actual = replace_physical_tiles(
            self.stats, self.decisions, selected, replacement
        )
        expected = compose_tile_replacements(self.stats, self.decisions.keep)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_veto_only_replacement_targets_exact_rows(self):
        candidates = candidate_tiles(self.decisions, 6)
        selected = torch.zeros_like(candidates)
        for head in range(candidates.shape[1]):
            indices = candidates[0, head].nonzero().flatten()
            if indices.numel():
                selected[0, head, indices[0]] = True
        rows = replacement_rows(self.decisions, selected, "veto_only")
        self.assertTrue(torch.equal(rows, selected[..., None] & self.decisions.veto_rows))
        replacement = exact_replacement_components(
            self.stats, self.decisions, selected, scope="veto_only"
        )
        actual = replace_physical_tiles(
            self.stats, self.decisions, selected, replacement
        )
        all_rows = exact_replacement_components(
            self.stats, self.decisions, selected, scope="all_rows"
        )
        upper = replace_physical_tiles(self.stats, self.decisions, selected, all_rows)
        intended = rows.any(dim=2)
        torch.testing.assert_close(actual[intended], upper[intended])
        self.assertTrue(bool((rows <= self.decisions.veto_rows).all()))


if __name__ == "__main__":
    unittest.main()
