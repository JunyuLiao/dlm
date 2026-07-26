import math
import unittest

import torch

from blasst.flash_attention import blasst_flash_attn_func
from blasst.veto_pruning import KVetoPolicy


class VetoPolicyReferenceTest(unittest.TestCase):
    def test_k_veto_is_opt_in_and_new_max_protection_is_effective(self):
        q = torch.ones(1, 128, 1, 1)
        q[:, 0] = -1
        k = torch.cat(
            (torch.full((1, 64, 1, 1), -4.0), torch.full((1, 64, 1, 1), 4.0)),
            dim=1,
        )
        v = torch.cat(
            (torch.full((1, 64, 1, 1), 10.0), torch.zeros(1, 64, 1, 1)),
            dim=1,
        )
        baseline = blasst_flash_attn_func(
            q, k, v, blasst_lambda=0.5, q_block_size=128, kv_block_size=64
        )
        protected = KVetoPolicy(1, log_threshold=math.log(0.5), new_max_protection="all")
        protected_output = blasst_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=0.5,
            q_block_size=128,
            kv_block_size=64,
            veto_decision_callback=lambda q_tile, kv_tile, context: protected(
                0, q_tile, kv_tile, context
            ),
        )
        torch.testing.assert_close(protected_output, baseline)
        self.assertEqual(protected.stats.additional_skipped_tiles, 0)

        aggressive = KVetoPolicy(1, log_threshold=math.log(0.5), new_max_protection="none")
        aggressive_output = blasst_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=0.5,
            q_block_size=128,
            kv_block_size=64,
            veto_decision_callback=lambda q_tile, kv_tile, context: aggressive(
                0, q_tile, kv_tile, context
            ),
        )
        self.assertEqual(aggressive.stats.additional_skipped_tiles, 1)
        self.assertEqual(aggressive.stats.additional_new_max_tiles, 1)
        self.assertGreater(float((baseline - aggressive_output).abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()
