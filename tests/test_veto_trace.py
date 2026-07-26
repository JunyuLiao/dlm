import tempfile
import unittest

import numpy as np
import torch

from blasst.flash_attention import blasst_flash_attn_func
from tracing.veto_trace_collector import VetoTraceCollector, VetoTraceContext
from tracing.veto_trace_schema import iter_veto_trace_shards


class VetoTraceTest(unittest.TestCase):
    def test_rich_trace_mass_and_v_statistics(self):
        torch.manual_seed(7)
        q = torch.randn(1, 4, 2, 3)
        k = torch.randn(1, 4, 2, 3)
        v = torch.randn(1, 4, 2, 3)
        traces = []
        blasst_flash_attn_func(
            q,
            k,
            v,
            blasst_lambda=0.0,
            q_block_size=4,
            kv_block_size=2,
            query_delta=torch.zeros(1, 4, 2),
            rich_tile_trace_callback=lambda q_tile, kv_tile, trace: traces.append(
                (q_tile, kv_tile, trace)
            ),
        )
        self.assertEqual(len(traces), 2)
        traces.sort(key=lambda item: item[1])
        masses = torch.stack(
            [torch.exp(item[2].row_local_logsumexp - item[2].row_final_logsumexp) for item in traces]
        )
        self.assertTrue(torch.allclose(masses.sum(0), torch.ones_like(masses[0]), atol=1e-5))
        self.assertTrue(torch.equal(traces[0][2].row_query_delta, torch.zeros(1, 2, 4)))
        self.assertTrue((traces[0][2].row_final_output_norm > 0).all())
        self.assertTrue((traces[0][2].v_max_row_norm >= traces[0][2].v_mean_row_norm).all())

    def test_collector_writes_fixed_width_rows_and_stable_state(self):
        q = torch.ones(1, 3, 1, 2)
        k = torch.ones(1, 2, 1, 2)
        v = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
        with tempfile.TemporaryDirectory() as directory:
            context = VetoTraceContext(
                sample_ids=["sample"],
                request_ids=["request"],
                seeds=[3],
                sequence_length=3,
                diffusion_block=0,
                denoising_step=1,
                remaining_mask_ratios=[0.5],
                thresholds=[0.2],
                token_states=torch.tensor([[0, 1, 2]], dtype=torch.uint8),
                query_heads=1,
            )
            with VetoTraceCollector(
                directory,
                context,
                sample_fraction=1.0,
                q_block_size=4,
                kv_block_size=2,
                stable_query_delta=0.05,
            ) as collector:
                blasst_flash_attn_func(
                    q,
                    k,
                    v,
                    blasst_lambda=0.2,
                    q_block_size=4,
                    kv_block_size=2,
                    query_delta=torch.zeros(1, 3, 1),
                    rich_tile_trace_callback=lambda q_tile, kv_tile, trace: collector(
                        2, q_tile, kv_tile, trace
                    ),
                )
            shards = list(iter_veto_trace_shards(directory))
            self.assertEqual(len(shards), 1)
            payload = shards[0][1]
            self.assertEqual(payload["metadata"].shape, (1,))
            self.assertEqual(payload["row_score"].shape, (1, 4))
            np.testing.assert_array_equal(payload["row_token_state"][0], [0, 1, 3, 255])
            self.assertAlmostEqual(float(payload["metadata"]["threshold"][0]), 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
