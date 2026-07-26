import unittest

import torch

from blasst.tile_reuse import (
    attention_tile_statistics,
    compose_attention_tile_statistics,
    merge_attention_tile_statistics,
    query_tile_output_cache_footprint,
    tiled_attention_reference,
    two_dimensional_cache_footprint,
)


class TileReuseCompositionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261001)

    def test_fresh_tile_composition_matches_dense_attention(self) -> None:
        q = torch.randn(2, 3, 11, 8, dtype=torch.float64)
        k = torch.randn(2, 3, 17, 8, dtype=torch.float64)
        v = torch.randn(2, 3, 17, 6, dtype=torch.float64)
        scale = q.shape[-1] ** -0.5
        expected = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, -1) @ v
        actual = tiled_attention_reference(
            q, k, v, q_block_size=5, kv_block_size=4, scale=scale
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_composition_is_associative_with_float_tolerance(self) -> None:
        q = torch.randn(2, 7, 16)
        k = torch.randn(2, 15, 16)
        v = torch.randn(2, 15, 9)
        tiles = [
            attention_tile_statistics(q, k[:, start : start + 5], v[:, start : start + 5])
            for start in range(0, 15, 5)
        ]
        left = compose_attention_tile_statistics(tiles)
        right = merge_attention_tile_statistics(
            tiles[0], merge_attention_tile_statistics(tiles[1], tiles[2])
        )
        self.assertTrue(torch.allclose(left.m, right.m, atol=0, rtol=0))
        self.assertTrue(torch.allclose(left.l, right.l, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(left.u, right.u, atol=2e-6, rtol=2e-6))

    def test_mixed_cached_and_fresh_components_are_composable(self) -> None:
        q_now = torch.randn(1, 6, 8)
        q_previous = torch.randn(1, 6, 8)
        k_now = torch.randn(1, 12, 8)
        k_previous = torch.randn(1, 12, 8)
        v_now = torch.randn(1, 12, 5)
        v_previous = torch.randn(1, 12, 5)
        current = [
            attention_tile_statistics(q_now, k_now[:, start : start + 4], v_now[:, start : start + 4])
            for start in range(0, 12, 4)
        ]
        previous = [
            attention_tile_statistics(
                q_previous,
                k_previous[:, start : start + 4],
                v_previous[:, start : start + 4],
            )
            for start in range(0, 12, 4)
        ]
        mixed = compose_attention_tile_statistics([current[0], previous[1], current[2]])
        manual = merge_attention_tile_statistics(
            merge_attention_tile_statistics(current[0], previous[1]), current[2]
        )
        self.assertTrue(torch.allclose(mixed.normalized_output(), manual.normalized_output()))

    def test_cache_footprints_match_native_geometry(self) -> None:
        two_d = two_dimensional_cache_footprint(batch_size=1)
        query = query_tile_output_cache_footprint(batch_size=1)
        self.assertEqual(two_d.tiles, 2_097_152)
        self.assertEqual(two_d.bytes_per_tile, 33_792)
        self.assertAlmostEqual(two_d.gib, 66.0)
        self.assertEqual(query.tiles, 32_768)
        self.assertAlmostEqual(query.gib, 1.0)


if __name__ == "__main__":
    unittest.main()
