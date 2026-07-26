import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tracing.tile_reuse_collector import ReuseTraceStep, TileReuseTraceCollector


class TileReuseTraceCollectorTest(unittest.TestCase):
    def test_paired_step_summary_and_raw_statistics(self) -> None:
        torch.manual_seed(20261002)
        with tempfile.TemporaryDirectory() as directory:
            collector = TileReuseTraceCollector(
                directory,
                heads=(0,),
                layers=(0,),
                raw_layers=(0,),
                raw_heads=(0,),
                q_block_size=4,
                kv_block_size=2,
            )
            states = torch.tensor([[4, 4, 0, 0, 2, 2, 2, 2]], dtype=torch.uint8)
            q = torch.randn(1, 8, 1, 4)
            k = torch.randn(1, 8, 1, 4)
            v = torch.randn(1, 8, 1, 4)
            output = torch.randn_like(q)
            collector.set_step(ReuseTraceStep(("sample",), 0, 0.5, 1, states))
            collector.record_layer(0, q, k, v, output)
            collector.set_step(ReuseTraceStep(("sample",), 1, 0.4, 1, states))
            collector.record_layer(0, q * 1.01, k * 0.99, v * 1.02, output * 1.01)
            summary_path = collector.flush()
            self.assertIsNotNone(summary_path)
            with np.load(summary_path) as payload:
                self.assertEqual(payload["request_id"].shape, (4,))
                self.assertEqual(payload["tile_output_error_max"].shape, (4,))
                self.assertTrue(np.isfinite(payload["q_fro"]).all())
                self.assertTrue((payload["step"] == 1).all())
                self.assertEqual(payload["cert_omit_b0p005"].shape, (4,))
                self.assertLessEqual(
                    float(payload["cert_group_interval_violation_max_b0p005"].max()),
                    2e-5,
                )
                self.assertLessEqual(
                    float(payload["cert_group_mass_violation_max_b0p005"].max()),
                    2e-5,
                )
            raw = sorted(Path(directory).glob("raw-*.pt"))
            self.assertEqual(len(raw), 2)
            payload = torch.load(raw[-1], weights_only=False)
            self.assertEqual(payload["u"].shape, (1, 1, 4, 4, 4))
            self.assertEqual(payload["m"].shape, (1, 1, 4, 4))


if __name__ == "__main__":
    unittest.main()
