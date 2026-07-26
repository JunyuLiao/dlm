import unittest

import torch

from blasst.tile_replacement_runtime import (
    ReplacementRuntimeConfig,
    TileReplacementRuntime,
)


class TileReplacementRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260721)
        self.q = torch.randn(1, 8, 2, 4, generator=generator)
        self.k = torch.randn(1, 8, 1, 4, generator=generator)
        self.v = torch.randn(1, 8, 1, 3, generator=generator)

    def test_no_replacement_matches_direct_sparse_reference(self):
        runtime = TileReplacementRuntime(
            ReplacementRuntimeConfig(
                policy="none", q_block_size=8, kv_block_size=4
            )
        )
        actual = runtime.attention(
            self.q,
            self.k,
            self.v,
            blasst_threshold=1.0,
            layer=0,
            row_masked=torch.ones(1, 8, dtype=torch.bool),
        )
        self.assertEqual(actual.shape, (1, 8, 2, 3))
        self.assertEqual(runtime.stats.replaced_tiles, 0)

    def test_exact_all_row_replacement_is_identity(self):
        baseline = TileReplacementRuntime(
            ReplacementRuntimeConfig(
                policy="none", q_block_size=8, kv_block_size=4
            )
        )
        oracle = TileReplacementRuntime(
            ReplacementRuntimeConfig(
                k=8,
                scope="all_rows",
                mass_method="exact",
                value_method="exact",
                policy="all_candidates",
                q_block_size=8,
                kv_block_size=4,
            )
        )
        kwargs = {
            "blasst_threshold": 1.0,
            "layer": 0,
            "row_masked": torch.ones(1, 8, dtype=torch.bool),
        }
        expected = baseline.attention(self.q, self.k, self.v, **kwargs)
        actual = oracle.attention(self.q, self.k, self.v, **kwargs)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
