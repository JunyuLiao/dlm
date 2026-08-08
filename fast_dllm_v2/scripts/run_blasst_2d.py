#!/usr/bin/env python3
"""Run Fast-dLLM v2 generation with optional reference 2D-BLASST."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import generate_one, load_examples, load_model, set_seed
from sparse_attention import Blasst2DConfig, Blasst2DStats, install_blasst_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="Efficient-Large-Model/Fast_dLLM_v2_7B",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(V2_ROOT / "data/alpaca/test/test_252.json"),
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--enable-blasst-2d", action="store_true")
    parser.add_argument("--blasst-lambda", type=float, default=0.5)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--collect-blasst-stats", action="store_true")
    parser.add_argument("--dump-blasst-trace", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--denoising-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path, args.device, args.precision)
    cfg = Blasst2DConfig(
        enable_blasst_2d=args.enable_blasst_2d,
        blasst_lambda=args.blasst_lambda,
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=args.collect_blasst_stats,
        dump_blasst_trace=args.dump_blasst_trace,
    )
    stats = Blasst2DStats()
    if args.enable_blasst_2d:
        install_blasst_2d(
            model,
            cfg,
            stats,
            mask_token_id=getattr(model.config, "mask_token_id", 151665),
            pad_token_id=tokenizer.pad_token_id,
            dual_cache_only=False,
        )

    examples = load_examples(args.dataset_path, args.num_samples)
    generations = []
    for index, example in enumerate(examples):
        set_seed(args.seed + index)
        result = generate_one(
            model,
            tokenizer,
            example["input"],
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            threshold=args.denoising_threshold,
        )
        result["reference"] = example["output"]
        generations.append(result)

    (output_dir / "generations.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=V2_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    run_config = {
        **vars(args),
        "repository_commit": commit,
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "actual_query_length_is_generation_block_size": True,
    }
    stats.export(output_dir, cfg, run_config)
    print(json.dumps({"config": asdict(cfg), "stats": stats.summary()}, indent=2))


if __name__ == "__main__":
    main()
