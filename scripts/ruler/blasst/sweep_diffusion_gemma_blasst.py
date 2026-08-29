"""Aggregate and validate the DiffusionGemma BLASST tile/context sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CONTEXTS = (1024, 2048, 4096, 8192, 16384)
BLOCK_SIZES = (32, 64, 128, 256)
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    args = parser.parse_args()
    root = args.sweep_root.resolve()
    rows = []
    for context in CONTEXTS:
        manifest = json.loads(
            (root / "manifests" / str(context) / "manifest.json").read_text()
        )
        if manifest["requested_num_samples"] != 10 or manifest["actual_num_samples"] != 10:
            raise RuntimeError(f"context {context} manifest is not exact-count 10")
        if manifest["ruler"]["commit"] != RULER_REVISION:
            raise RuntimeError(f"context {context} uses an unexpected RULER revision")
        if manifest["tokenizer_revision"] != MODEL_REVISION:
            raise RuntimeError(f"context {context} uses an unexpected tokenizer revision")
        samples = Path(manifest["samples"]["path"]).read_text().splitlines()
        if len(samples) != 10:
            raise RuntimeError(f"context {context} samples file is not exact-count 10")
        for block_size in BLOCK_SIZES:
            run = root / "runs" / f"context_{context}" / f"block_{block_size}"
            summary = json.loads((run / "summary.json").read_text())
            config = json.loads((run / "run_config.json").read_text())
            predictions = [
                json.loads(line)
                for line in (run / "predictions.jsonl").read_text().splitlines()
            ]
            stats = summary["attention_sparsity"]
            if not (
                summary["requested_num_samples"]
                == summary["actual_num_samples"]
                == len(predictions)
                == 10
            ):
                raise RuntimeError(f"context={context}, block={block_size}: count mismatch")
            if not summary["all_completions_nonempty"]:
                raise RuntimeError(f"context={context}, block={block_size}: empty completion")
            if any(not str(row.get("prediction", "")).strip() for row in predictions):
                raise RuntimeError(f"context={context}, block={block_size}: empty prediction")
            if stats["eligible_tiles"] != stats["retained_tiles"] + stats["skipped_tiles"]:
                raise RuntimeError(f"context={context}, block={block_size}: invalid tile counts")
            expected_sparsity = stats["skipped_tiles"] / stats["eligible_tiles"]
            if not math.isclose(stats["physical_tile_sparsity"], expected_sparsity):
                raise RuntimeError(f"context={context}, block={block_size}: invalid sparsity")
            if not all(
                math.isfinite(float(value))
                for value in (
                    summary["official_ruler_accuracy"],
                    stats["physical_tile_sparsity"],
                )
            ):
                raise RuntimeError(f"context={context}, block={block_size}: non-finite metric")
            if config["q_tile_size"] != block_size or config["kv_tile_size"] != block_size:
                raise RuntimeError(f"context={context}, block={block_size}: tile config mismatch")
            if not (
                config["attention_backend"] == "blasst-reference"
                and config["blasst_lambda"] == 0.003
                and config["model_adapter"] == "diffusion_gemma"
                and config["resolved_model_revision"] == MODEL_REVISION
                and config["transformers_version"] == "5.11.0"
                and summary["full_checkpoint_on_cuda"]
                and summary["model_parameter_devices"] == ["cuda:0"]
            ):
                raise RuntimeError(f"context={context}, block={block_size}: run config mismatch")
            rows.append(
                {
                    "context_length": context,
                    "block_size": block_size,
                    "num_samples": 10,
                    "accuracy": summary["official_ruler_accuracy"],
                    "physical_sparsity": stats["physical_tile_sparsity"],
                    "eligible_tiles": stats["eligible_tiles"],
                    "skipped_tiles": stats["skipped_tiles"],
                    "retained_tiles": stats["retained_tiles"],
                    "valid_element_sparsity": stats["valid_element_sparsity"],
                    "elapsed_seconds": summary["total_elapsed_seconds"],
                    "peak_cuda_memory_allocated_bytes": summary.get(
                        "peak_cuda_memory_allocated_bytes"
                    ),
                    "resolved_model_revision": config.get("resolved_model_revision"),
                    "transformers_version": config.get("transformers_version"),
                }
            )

    fieldnames = list(rows[0])
    with (root / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "report.json").write_text(
        json.dumps(
            {
                "model": "google/diffusiongemma-26B-A4B-it",
                "attention_backend": "blasst-reference",
                "blasst_lambda": 0.003,
                "block_size_interpretation": "q_tile_size = kv_tile_size",
                "native_canvas_length": 256,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    lines = [
        "# DiffusionGemma BLASST tile/context sweep",
        "",
        "- BLASST lambda: `0.003`",
        "- Samples per case: `10` exact-count NVIDIA RULER samples",
        "- Block size: both BLASST query and KV tile size; native canvas remains 256",
        "",
        "| Context | Block | Accuracy | Physical sparsity | Skipped / eligible |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['context_length']} | {row['block_size']} | "
            f"{row['accuracy']:.4f} | {row['physical_sparsity']:.4%} | "
            f"{row['skipped_tiles']} / {row['eligible_tiles']} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
