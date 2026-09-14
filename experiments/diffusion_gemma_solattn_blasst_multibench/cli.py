from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODEL, REVISION
from .controlled_dataset import prepare
from .controlled_report import build
from .controlled_runner import run, smoke
from .grading import grade_livecodebench


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact paired DiffusionGemma Sol-Attn/BLASST evaluation")
    parser.add_argument("command", choices=("prepare", "smoke", "run", "grade-livecodebench", "report"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_blasst_multibench_controlled"))
    parser.add_argument("--model-path", default=MODEL); parser.add_argument("--revision", default=REVISION); parser.add_argument("--conditions", nargs="*")
    args = parser.parse_args(); manifest = args.output_dir / "manifest.jsonl"
    if args.command == "prepare": result = prepare(args.output_dir, model_path=args.model_path)
    elif args.command == "smoke": result = smoke(manifest, args.output_dir, model_path=args.model_path, revision=args.revision)
    elif args.command == "run": result = run(manifest, args.output_dir, model_path=args.model_path, revision=args.revision, selected=args.conditions)
    elif args.command == "grade-livecodebench": result = grade_livecodebench(args.output_dir)
    else: result = build(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
