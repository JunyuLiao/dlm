#!/usr/bin/env python3
"""CLI for the DiffusionGemma Sol-Attn MATH500 region experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODEL, DEFAULT_REVISION

DEFAULT_OUTPUT = Path("results/diffusion_gemma_solattn_math500_prefix_vs_all")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--nemo-gym-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--split-seed", type=int, default=20260822)
    for command in ("smoke", "run"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", type=Path)
        item.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        item.add_argument("--model-path", default=DEFAULT_MODEL)
        item.add_argument("--revision", default=DEFAULT_REVISION)
        item.add_argument("--device", default="cuda")
        if command == "smoke":
            item.add_argument("--max-new-tokens", type=int, default=256)
        else:
            item.add_argument("--conditions", nargs="+")
    report = sub.add_parser("report")
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return root


def main() -> None:
    args = parser().parse_args()
    manifest = getattr(args, "manifest", None) or args.output_dir / "manifest.jsonl"
    if args.command == "prepare":
        from .dataset import prepare_manifest
        result = {"samples": len(prepare_manifest(args.nemo_gym_root, args.output_dir, args.split_seed))}
    elif args.command == "smoke":
        from .runner import cuda_smoke
        result = cuda_smoke(manifest, args.output_dir, model_path=args.model_path, revision=args.revision, device=args.device, max_new_tokens=args.max_new_tokens)
    elif args.command == "run":
        from .runner import run_sweep
        result = run_sweep(manifest, args.output_dir, model_path=args.model_path, revision=args.revision, selected_conditions=args.conditions, device=args.device)
    else:
        from .report import build_report
        result = build_report(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
