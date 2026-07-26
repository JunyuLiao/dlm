#!/usr/bin/env python3
"""Generate a tiny multi-sample/multi-step trace without model downloads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.flash_attention import blasst_flash_attn_func
from blasst.triton_bidirectional import DiffusionLambdaSchedule
from tracing import IncrementalTraceCollector, TraceContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/proxy_smoke/traces"))
    args = parser.parse_args()
    schedule = DiffusionLambdaSchedule()
    batch, length, heads, dim = 6, 48, 4, 8
    sample_ids = [f"smoke-{index}" for index in range(batch)]
    generator = torch.Generator().manual_seed(31)
    base_q = torch.randn(batch, length, heads, dim, generator=generator)
    base_k = torch.randn(batch, length, heads, dim, generator=generator)
    base_v = torch.randn(batch, length, heads, dim, generator=generator)
    for iteration, ratio in enumerate((0.9, 0.5, 0.15)):
        context = TraceContext(
            sample_ids=sample_ids,
            request_ids=sample_ids,
            seeds=list(range(31, 31 + batch)),
            sequence_length=length,
            diffusion_block=0,
            denoising_iteration=iteration,
            remaining_mask_ratios=[ratio] * batch,
            query_heads=heads,
            kv_heads=heads,
        )
        with IncrementalTraceCollector(
            args.output_dir, context, shard_records=250, q_block_size=16, kv_block_size=8,
            bitpack_masks=True, quantize_log_scores=8,
        ) as collector:
            for layer in range(3):
                noise = torch.randn(base_q.shape, generator=generator) * (0.03 + 0.01 * layer)
                q = base_q + noise + iteration * 0.01 + layer * 0.02
                k = base_k + noise * 0.5 + layer * 0.01
                v = base_v + layer * 0.01
                blasst_flash_attn_func(
                    q, k, v,
                    blasst_lambda=schedule.threshold(ratio),
                    q_block_size=16,
                    kv_block_size=8,
                    tile_trace_callback=lambda q_tile, kv_tile, trace, layer=layer: collector(
                        layer, q_tile, kv_tile, trace
                    ),
                )
    print(args.output_dir)


if __name__ == "__main__":
    main()
