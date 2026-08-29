#!/usr/bin/env python3
"""Resumable distribution-first sparse-routing experiment CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = Path("results/attention/threshold_modeling/main")


def _densities(values: list[float]) -> list[float]:
    if values != [0.25, 0.5, 0.75]:
        raise argparse.ArgumentTypeError("the canonical protocol requires densities 0.25 0.5 0.75")
    return values


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="collect native-dense compact proxy shards")
    collect.add_argument("--adapter", choices=("diffusion_gemma", "fast_dllm_v2"), required=True)
    collect.add_argument("--model-path")
    collect.add_argument("--revision")
    collect.add_argument("--corpus", choices=("ruler8k", "math500"), required=True)
    collect.add_argument("--prompts-jsonl", type=Path)
    collect.add_argument("--nemo-gym-root", type=Path)
    collect.add_argument("--num-prompts", type=int, default=50)
    collect.add_argument("--auto-extend", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--max-distribution-prompts", type=int, default=100)
    collect.add_argument("--parity-prompts", type=int, default=2)
    collect.add_argument("--q-block-size", type=int, default=64)
    collect.add_argument("--kv-block-size", type=int, default=64)
    collect.add_argument("--reservoir-per-row", type=int, default=32)
    collect.add_argument("--split-seed", type=int, default=20260822)
    collect.add_argument("--base-seed", type=int, default=42)
    collect.add_argument("--densities", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    collect.add_argument("--max-new-tokens", type=int, default=2048)
    collect.add_argument("--generation-block-size", type=int, default=256)
    collect.add_argument("--steps", type=int)
    collect.add_argument("--temperature", type=float, default=0.0)
    collect.add_argument("--top-p", type=float, default=0.95)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--precision", default="bfloat16")
    collect.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    fit = sub.add_parser("fit", help="fit and gate threshold tables")
    fit.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    fit.add_argument("--densities", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    fit.add_argument("--bootstrap-repeats", type=int, default=2_000)
    fit.add_argument("--split-seed", type=int, default=20260822)

    evaluate = sub.add_parser("eval-math500", help="run ten paired fresh-routing conditions")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--nemo-gym-root", type=Path, required=True)
    evaluate.add_argument("--threshold-model", type=Path)
    evaluate.add_argument("--model-path")
    evaluate.add_argument("--revision")
    evaluate.add_argument("--split-seed", type=int, default=20260822)
    evaluate.add_argument("--base-seed", type=int, default=42)
    evaluate.add_argument("--bootstrap-repeats", type=int, default=20_000)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--precision", default="bfloat16")

    report = sub.add_parser("report", help="audit and regenerate plots/report")
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    ruler = sub.add_parser("eval-ruler-routing", help="run fresh logical tile-routing RULER sweep")
    ruler.add_argument("--adapter", choices=("diffusion_gemma", "fast_dllm_v2"), required=True)
    ruler.add_argument("--model-path", required=True)
    ruler.add_argument("--revision")
    ruler.add_argument("--manifest-path", type=Path, required=True)
    ruler.add_argument("--ruler-root", type=Path, required=True)
    ruler.add_argument("--output-dir", type=Path, required=True)
    ruler.add_argument("--num-samples", type=int, default=50)
    ruler.add_argument("--context-length", type=int, default=16384)
    ruler.add_argument("--max-new-tokens", type=int)
    ruler.add_argument("--block-size", type=int, default=16)
    ruler.add_argument("--temperature", type=float, default=0.0)
    ruler.add_argument("--device", default="cuda")
    ruler.add_argument("--precision", default="bfloat16")
    ruler.add_argument("--generation-extra")
    ruler.add_argument("--random-seed", type=int, default=20260825)
    ruler.add_argument("--no-prefix-ablation", action="store_true")
    ruler.add_argument("--conditions", nargs="+", help="run only these named conditions")
    ruler.add_argument("--threshold-model", type=Path, help="profiled threshold table JSON")
    ruler.add_argument("--routing-execution", choices=("logical", "physical"), default="logical")
    ruler.add_argument("--q-block-size", type=int, choices=(32, 64, 128), default=64)
    ruler.add_argument("--kv-block-size", type=int, choices=(64,), default=64)
    ruler.add_argument("--performance-warmups", type=int, default=0)
    ruler.add_argument("--performance-repeats", type=int, default=1)
    ruler.add_argument("--progress-every", type=int, default=0, help="print sparse progress every N examples; detailed progress is always logged")
    ruler.add_argument("--quiet", action="store_true", help="suppress progress printing")

    ruler_report = sub.add_parser("report-ruler", help="report completed fresh-routing RULER runs")
    ruler_report.add_argument("--output-dir", type=Path, required=True)
    ruler_report.add_argument("--ruler-root", type=Path, required=True)
    canonical = sub.add_parser("canonical-report", help="assemble the canonical RULER16K bundle audit")
    canonical.add_argument("--bundle-root", type=Path, required=True)
    audit = sub.add_parser("audit-ruler16k", help="write the final staged RULER16K audit")
    audit.add_argument("--bundle-root", type=Path, required=True)

    # Isolated DiffusionGemma Sol-Attn versus BLASST protocol.  These commands
    # intentionally live beside the historical distribution-first commands so
    # existing scripts remain byte-for-byte compatible.
    study = sub.add_parser("prepare", help="prepare the disjoint Sol-Attn/BLASST RULER16K manifests")
    study.add_argument("--raw-root", type=Path, required=True)
    study.add_argument("--ruler-root", type=Path, required=True)
    study.add_argument("--model-path", required=True)
    study.add_argument("--revision")
    study.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k"))
    study.add_argument("--split-seed", type=int, default=42)
    study.add_argument("--model-adapter", default="diffusion_gemma")
    study.add_argument("--calibration-per-task", type=int, default=2)
    study.add_argument("--final-per-task", type=int, default=10)
    calibrate_study = sub.add_parser("calibrate-blasst", help="calibrate shared local/global BLASST lambdas")
    calibrate_study.add_argument("--trace-path", type=Path)
    calibrate_study.add_argument("--study-manifest", type=Path)
    calibrate_study.add_argument("--adapter", default="diffusion_gemma")
    calibrate_study.add_argument("--model-path")
    calibrate_study.add_argument("--revision")
    calibrate_study.add_argument("--ruler-root", type=Path)
    calibrate_study.add_argument("--output-dir", type=Path, required=True)
    calibrate_study.add_argument("--device", default="cuda")
    calibrate_study.add_argument("--precision", default="bfloat16")
    run_study = sub.add_parser("run", help="run one or all nine isolated comparison conditions")
    run_study.add_argument("--study-manifest", type=Path, required=True)
    run_study.add_argument("--adapter", default="diffusion_gemma")
    run_study.add_argument("--model-path", required=True)
    run_study.add_argument("--revision")
    run_study.add_argument("--ruler-root", type=Path, required=True)
    run_study.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k"))
    run_study.add_argument("--threshold-policy", type=Path)
    run_study.add_argument("--conditions", nargs="*")
    run_study.add_argument("--device", default="cuda")
    run_study.add_argument("--precision", default="bfloat16")
    report_study = sub.add_parser("report-solattn-blasst", help="regenerate the isolated comparison report")
    report_study.add_argument("--output-dir", type=Path, default=Path("results/diffusion_gemma_solattn_vs_blasst_ruler16k"))
    report_study.add_argument("--ruler-root", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "densities"):
        _densities(list(args.densities))
    if args.command == "prepare":
        from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.cli import main as study_main
        import sys
        sys.argv[1:] = ["prepare", *sys.argv[2:]]
        study_main()
        return
    if args.command == "calibrate-blasst":
        from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.cli import main as study_main
        import sys
        sys.argv[1:] = ["calibrate-blasst", *sys.argv[2:]]
        study_main()
        return
    if args.command == "run":
        from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.cli import main as study_main
        import sys
        sys.argv[1:] = ["run", *sys.argv[2:]]
        study_main()
        return
    if args.command == "report-solattn-blasst":
        from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.cli import main as study_main
        import sys
        sys.argv[1:] = ["report", *sys.argv[2:]]
        study_main()
        return
    if args.command == "collect":
        from .collect import run_collection

        if args.num_prompts <= 0 or args.num_prompts > args.max_distribution_prompts or args.max_distribution_prompts > 100:
            raise ValueError("prompt counts must satisfy 0 < num-prompts <= max-distribution-prompts <= 100")
        result = run_collection(args)
    elif args.command == "fit":
        from .fitting import fit_threshold_models

        result = fit_threshold_models(args.output_dir, args.densities, bootstrap_repeats=args.bootstrap_repeats, seed=args.split_seed)
    elif args.command == "eval-math500":
        from .math500 import run_math500

        result = run_math500(args)
    elif args.command == "eval-ruler-routing":
        from .ruler import run_ruler_sweep

        result = run_ruler_sweep(args)
    elif args.command == "report-ruler":
        from .ruler_report import build_ruler_report

        result = build_ruler_report(args.output_dir, args.ruler_root)
    elif args.command == "canonical-report":
        from .canonical_report import build_canonical_report

        result = build_canonical_report(args.bundle_root)
    elif args.command == "audit-ruler16k":
        from .audit_ruler16k import build_final_audit

        result = build_final_audit(args.bundle_root)
    else:
        from .report import build_report

        result = build_report(args.output_dir)
    compact = {"command": args.command}
    if args.command == "eval-ruler-routing":
        compact.update({
            "output_dir": str(args.output_dir),
            "adapter": args.adapter,
            "conditions": len(result.get("conditions", [])),
            "completed_conditions": [item.get("condition", {}).get("name") for item in result.get("conditions", [])],
        })
    elif args.command in ("report-ruler", "canonical-report", "audit-ruler16k", "report"):
        compact["output_dir"] = str(getattr(args, "output_dir", getattr(args, "bundle_root", "")))
        if isinstance(result, dict):
            compact["keys"] = sorted(result.keys())
            if args.command == "audit-ruler16k":
                compact["passed"] = result.get("passed")
    elif args.command == "fit":
        compact.update({"output_dir": str(args.output_dir), "selected": result.get("selected"), "simple_profiling_passed": result.get("simple_profiling_passed")})
    elif args.command == "collect":
        compact.update({"output_dir": str(args.output_dir), "adapter": args.adapter, "corpus": args.corpus, "shards": len(result.get("shards", []))})
    elif args.command == "eval-math500":
        compact.update({"output_dir": str(args.output_dir), "conditions": len(result.get("conditions", []))})
    print(json.dumps(compact, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
