import unittest

import torch

from blasst.tile_replacement_predictors import (
    analytic_log_z_predictions,
    oracle_row_centroids,
    per_instance_entropy_identity,
    summary_predictions,
    tiled_logits,
    uniform_pool_summaries,
)


class TileReplacementPredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(9)
        self.q = torch.randn(1, 5, 2, 4, generator=generator)
        self.k = torch.randn(1, 8, 1, 4, generator=generator)
        self.v = torch.randn(1, 8, 1, 3, generator=generator)
        self.logits = tiled_logits(self.q, self.k, self.v, kv_block_size=4)

    def test_exact_log_z_matches_direct(self):
        predictions = analytic_log_z_predictions(self.logits)
        torch.testing.assert_close(
            predictions["exact"], torch.logsumexp(self.logits.scores, -1)
        )

    def test_one_slot_uniform_summary_has_expected_shapes(self):
        summary = uniform_pool_summaries(
            self.k, self.v, query_heads=2, slots=1, kv_block_size=4
        )
        log_z, value = summary_predictions(self.q, summary)
        self.assertEqual(log_z.shape, (1, 2, 2, 5))
        self.assertEqual(value.shape, (1, 2, 2, 5, 3))

    def test_per_instance_entropy_identity_is_exact(self):
        log_z, value = per_instance_entropy_identity(self.logits)
        expected_log_z = torch.logsumexp(self.logits.scores, -1)
        expected_value = torch.einsum(
            "bhtrw,bhtwd->bhtrd", self.logits.scores.softmax(-1), self.logits.values
        )
        torch.testing.assert_close(log_z, expected_log_z, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(value, expected_value, atol=1e-6, rtol=1e-6)

    def test_oracle_row_centroids_preserve_shape(self):
        exact = torch.randn(1, 2, 2, 5, 3)
        actual = oracle_row_centroids(exact, 2)
        self.assertEqual(actual.shape, exact.shape)


if __name__ == "__main__":
    unittest.main()
