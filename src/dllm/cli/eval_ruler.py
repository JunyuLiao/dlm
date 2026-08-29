from __future__ import annotations

import argparse
import json
from pathlib import Path

from dllm.evaluation.ruler import RulerRunConfig, run_evaluation
from dllm.models import adapter_names


def _json_object(value: str | None) -> dict:
    if not value:
        return {}
    candidate = value.strip()
    if not candidate.startswith("{"):
        candidate = Path(candidate).read_text(encoding="utf-8")
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("generation config JSON must contain an object")
    return parsed


def _float_tuple(value: str | None) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(item) for item in value.split(","))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Universal exact-count RULER evaluation")
    parser.add_argument("--model-adapter", required=True, choices=adapter_names())
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ruler-root", required=True)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("dense", "eager-dense", "blasst-reference"),
        default="dense",
    )
    parser.add_argument("--blasst-lambda", type=float, default=0.003)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--revision")
    parser.add_argument("--collect-attention-stats", action="store_true")
    parser.add_argument(
        "--include-masked-kv-tiles-in-physical-stats",
        action="store_true",
        help=(
            "count masked padding and empty-cache KV tiles in the physical "
            "tile denominator"
        ),
    )
    parser.add_argument(
        "--stats-level", choices=("summary", "step", "layer", "head"), default="summary"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--generation-config-json",
        help="inline JSON object or path containing adapter-specific generation options",
    )
    parser.add_argument(
        "--blasst-policy-json",
        help=(
            "inline JSON object or path with local/global and phase-specific "
            "BLASST thresholds"
        ),
    )
    parser.add_argument(
        "--blasst-calibration-lambdas",
        help="comma-separated thresholds observed on the same dense eager trajectory",
    )
    parser.add_argument(
        "--verify-dense-after-blasst",
        action="store_true",
        help="after closing a BLASST binding, run one unrecorded dense cleanup smoke",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = run_evaluation(
        RulerRunConfig(
            model_adapter=args.model_adapter,
            model_path=args.model_path,
            manifest_path=args.manifest,
            ruler_root=args.ruler_root,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            context_length=args.context_length,
            attention_backend=args.attention_backend,
            blasst_lambda=args.blasst_lambda,
            q_tile_size=args.q_tile_size,
            kv_tile_size=args.kv_tile_size,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            threshold=args.threshold,
            temperature=args.temperature,
            device=args.device,
            precision=args.precision,
            revision=args.revision,
            collect_attention_stats=args.collect_attention_stats,
            include_masked_kv_tiles_in_physical_stats=(
                args.include_masked_kv_tiles_in_physical_stats
            ),
            blasst_policy=_json_object(args.blasst_policy_json),
            blasst_calibration_lambdas=_float_tuple(
                args.blasst_calibration_lambdas
            ),
            stats_level=args.stats_level,
            resume=not args.no_resume,
            generation_extra=_json_object(args.generation_config_json),
            verify_dense_after_blasst=args.verify_dense_after_blasst,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
