#!/usr/bin/env python3
"""Numerical analysis of reduced tile-result cache formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blasst.tile_reuse import AttentionTileStatistics, compose_attention_tile_statistics


def compose(payload: dict[str, object], u: torch.Tensor) -> torch.Tensor:
    m = payload["m"].float()  # type: ignore[union-attr]
    l = payload["l"].float()  # type: ignore[union-attr]
    tiles = [
        AttentionTileStatistics(m[:, :, tile], l[:, :, tile], u[:, :, tile].float())
        for tile in range(m.shape[2])
    ]
    return compose_attention_tile_statistics(tiles).normalized_output()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    relative = torch.linalg.vector_norm(difference, dim=-1) / (
        torch.linalg.vector_norm(expected, dim=-1) + 1e-6
    )
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "maximum_row_relative_error": float(relative.max()),
        "mean_row_relative_error": float(relative.mean()),
        "cosine": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.trace_dir.glob("context-*/raw-*.pt"))
    if not paths:
        raise ValueError("no raw tile traces found")
    results: dict[str, list[dict[str, float]]] = {
        "bf16_u_fp16_ml": [],
        "scaled_fp8_u": [],
        "per_row_scaled_fp8_u": [],
        "per_row_scaled_int8_u": [],
    }
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = payload["final_attention_output"].float()
        u = payload["u"]
        results["bf16_u_fp16_ml"].append(metrics(compose(payload, u), expected))
        maximum = u.float().abs().max().clamp_min(1e-12)
        scale = maximum / 448.0
        quantized = (u.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        dequantized = quantized.float() * scale
        results["scaled_fp8_u"].append(metrics(compose(payload, dequantized), expected))
        row_maximum = u.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
        row_scale_fp8 = row_maximum / 448.0
        row_fp8 = (u.float() / row_scale_fp8).clamp(-448, 448).to(torch.float8_e4m3fn)
        results["per_row_scaled_fp8_u"].append(
            metrics(compose(payload, row_fp8.float() * row_scale_fp8), expected)
        )
        row_scale_int8 = row_maximum / 127.0
        row_int8 = (u.float() / row_scale_int8).round().clamp(-127, 127).to(torch.int8)
        results["per_row_scaled_int8_u"].append(
            metrics(compose(payload, row_int8.float() * row_scale_int8), expected)
        )

    summary: dict[str, object] = {"files": len(paths), "formats": {}}
    for name, values in results.items():
        summary["formats"][name] = {  # type: ignore[index]
            metric: {
                "mean": float(np.mean([row[metric] for row in values])),
                "p99": float(np.quantile([row[metric] for row in values], 0.99)),
                "maximum": float(np.max([row[metric] for row in values])),
            }
            for metric in values[0]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
