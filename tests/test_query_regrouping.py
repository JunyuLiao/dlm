import unittest

import torch

from blasst.regrouping import (
    aggregate_heads,
    build_permutation,
    build_head_group_permutations,
    build_windowed_permutation,
    evaluate_permutation,
    mean_jaccard,
    pack_keep_mask,
    unpack_keep_mask,
)


class QueryRegroupingTest(unittest.TestCase):
    def test_pack_round_trip_handles_partial_byte(self):
        torch.manual_seed(4)
        mask = torch.rand(2, 7, 13) > 0.5
        actual = unpack_keep_mask(pack_keep_mask(mask), 13)
        torch.testing.assert_close(actual, mask)

    def test_similar_rows_recover_physical_sparsity(self):
        keep = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]).bool()
        baseline = evaluate_permutation(keep, torch.arange(4), query_tile_size=2)
        perm = build_permutation(keep, "bitmap", group_size=2)
        grouped = evaluate_permutation(keep, perm, query_tile_size=2)
        self.assertEqual(grouped.physical_tile_sparsity, 0.5)
        self.assertGreater(grouped.physical_tile_sparsity, baseline.physical_tile_sparsity)

    def test_every_strategy_returns_a_permutation(self):
        torch.manual_seed(9)
        mask = torch.rand(37, 11) > 0.6
        for strategy, options in (
            ("identity", {}), ("bitmap", {}), ("topk", {"topk": 4}),
            ("simhash", {"signature_bits": 16}),
            ("bucket", {"signature_bits": 16, "bucket_bits": 6}),
            ("minhash", {"num_hashes": 4}),
            ("oracle_greedy", {}),
        ):
            with self.subTest(strategy=strategy):
                perm = build_permutation(mask, strategy, group_size=8, **options)
                torch.testing.assert_close(perm.sort().values, torch.arange(37))

    def test_state_aware_keeps_classes_contiguous(self):
        mask = torch.zeros(8, 4, dtype=torch.bool)
        state = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        perm = build_permutation(mask, "bitmap", state=state)
        self.assertEqual(state[perm].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_head_aggregation_and_jaccard(self):
        keep = torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]]).bool()
        torch.testing.assert_close(aggregate_heads(keep, None), torch.tensor([[1, 1], [0, 1]]).bool())
        torch.testing.assert_close(aggregate_heads(keep, 0.75), torch.tensor([[1, 0], [0, 0]]).bool())
        self.assertEqual(mean_jaccard(keep, keep), 1.0)

    def test_windowed_and_head_group_permutations_preserve_boundaries(self):
        torch.manual_seed(5)
        keep = torch.rand(4, 16, 7) > 0.5
        perm = build_windowed_permutation(keep[0], "bitmap", window_size=8, group_size=4)
        self.assertTrue(bool((perm[:8] < 8).all()))
        self.assertTrue(bool((perm[8:] >= 8).all()))
        grouped = build_head_group_permutations(
            keep, "bitmap", heads_per_group=2, group_size=4, window_size=8
        )
        torch.testing.assert_close(grouped[0], grouped[1])
        torch.testing.assert_close(grouped[2], grouped[3])


if __name__ == "__main__":
    unittest.main()
