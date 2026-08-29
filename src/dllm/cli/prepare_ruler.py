from __future__ import annotations

import argparse
import json
from pathlib import Path

from dllm.evaluation.ruler.official import prepare_manifest
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare an exact-count RULER manifest")
    parser.add_argument("--ruler-root", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-adapter", required=True, choices=adapter_names())
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--length-mode",
        choices=("prompt", "total"),
        default="prompt",
        help="whether context length denotes prompt tokens or paper-style prompt+completion tokens",
    )
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruler-dependency-path")
    parser.add_argument("--nltk-data")
    parser.add_argument(
        "--generation-config-json",
        help="inline JSON object or path; prompt-affecting options are recorded in the manifest",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip()) or None
    path = prepare_manifest(
        ruler_root=args.ruler_root,
        tokenizer_path=args.tokenizer_path,
        model_adapter=args.model_adapter,
        context_length=args.context_length,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        tasks=tasks,
        revision=args.revision,
        dependency_path=args.ruler_dependency_path,
        nltk_data=args.nltk_data,
        generation_extra=_json_object(args.generation_config_json),
        length_mode=args.length_mode,
    )
    print(path)


if __name__ == "__main__":
    main()
