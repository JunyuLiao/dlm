"""Prefix-only raw/standardized Sol-Attn proxy distribution experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plots import generate_plots
from .report import build_report
from .runner import run_collection


DEFAULT_OUTPUT = Path("results/proxy_diagnostics/math500_prefix_distribution")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--adapter", choices=("diffusion_gemma", "fast_dllm_v2"), required=True)
    collect.add_argument("--model-path")
    collect.add_argument("--revision")
    collect.add_argument("--corpus", choices=("math500", "ruler16k"), default="math500")
    collect.add_argument("--nemo-gym-root", type=Path)
    collect.add_argument("--prompts-jsonl", type=Path)
    collect.add_argument("--num-problems", type=int, default=10)
    collect.add_argument("--parity-prompts", type=int, default=2)
    collect.add_argument("--reservoir-per-row", type=int, default=256)
    collect.add_argument("--split-seed", type=int, default=20260824)
    collect.add_argument("--base-seed", type=int, default=42)
    collect.add_argument("--max-new-tokens", type=int, default=2048)
    collect.add_argument("--generation-block-size", type=int, default=256)
    collect.add_argument("--steps", type=int)
    collect.add_argument("--temperature", type=float, default=0.0)
    collect.add_argument("--top-p", type=float, default=0.95)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--precision", default="bfloat16")
    collect.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--output-dir", type=Path, default=None)
    plot = sub.add_parser("plot")
    plot.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    report = sub.add_parser("report")
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "collect":
        if args.output_dir is None:
            args.output_dir = Path(f"results/{args.corpus}_prefix_proxy_distribution")
        if args.corpus == "math500" and args.nemo_gym_root is None:
            raise SystemExit("math500 collection requires --nemo-gym-root")
        if args.corpus != "math500" and args.prompts_jsonl is None:
            raise SystemExit("RULER collection requires --prompts-jsonl")
        result = run_collection(args)
    elif args.command == "plot":
        result = generate_plots(args.output_dir)
    else:
        result = build_report(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
