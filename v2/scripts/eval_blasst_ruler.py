#!/usr/bin/env python3
"""Pinned NVIDIA RULER generation, 2D-BLASST sweeps, and paired accuracy.

This pipeline deliberately stays separate from the mixed GSM8K/HumanEval
evaluation.  NVIDIA's generators produce every prompt and NVIDIA's synthetic
scoring functions score every prediction.  The only model-specific additions
are exact token-length validation and Fast-dLLM/2D-BLASST inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import yaml
from transformers import AutoTokenizer


V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import (  # noqa: E402
    generate_one,
    load_model,
    normalized_token_difference,
    set_seed,
)
from scripts.sweep_blasst_controlled import (  # noqa: E402
    COUNT_FIELDS,
    _aggregate,
    _masked_queries,
    _plot_curves,
    _stats_rows,
    _write_csv,
)
from sparse_attention import (  # noqa: E402
    Blasst2DConfig,
    Blasst2DStats,
    install_blasst_2d,
)


RULER_REPOSITORY = "https://github.com/NVIDIA/RULER.git"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384)
BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64)
LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.5)
MASK_RATIOS = (0.90, 0.70, 0.50, 0.30, 0.15)
PAPER_TASKS = (
    "niah_multikey_1",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
)
# The official VT one-shot and four-needle NIAH configurations cannot fit a
# genuine 512-token prompt. These are official, shorter RULER configurations.
SHORT_512_TASKS = ("niah_multikey_2", "fwe")
RULER_SOURCE_FILES = (
    "scripts/synthetic.yaml",
    "scripts/data/prepare.py",
    "scripts/data/tokenizer.py",
    "scripts/data/synthetic/constants.py",
    "scripts/data/synthetic/niah.py",
    "scripts/data/synthetic/variable_tracking.py",
    "scripts/data/synthetic/freq_words_extraction.py",
    "scripts/eval/synthetic/constants.py",
    "scripts/eval/evaluate.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "prepare", "sweeps", "accuracy", "report"),
        default="all",
    )
    parser.add_argument("--model-path", default="/tmp/fast_dllm_v2_7b")
    parser.add_argument("--ruler-root", default="/tmp/nvidia-ruler")
    parser.add_argument(
        "--ruler-dependency-path",
        default="/tmp/blasst-ruler",
        help="Isolated site-packages containing NVIDIA RULER dependencies.",
    )
    parser.add_argument("--nltk-data", default="/tmp/nltk_data")
    parser.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    parser.add_argument("--block-sizes", default=",".join(map(str, BLOCK_SIZES)))
    parser.add_argument("--lambdas", default=",".join(map(str, LAMBDAS)))
    parser.add_argument("--mask-ratios", default=",".join(map(str, MASK_RATIOS)))
    parser.add_argument("--context-samples", type=int, default=32)
    parser.add_argument("--accuracy-samples", type=int, default=100)
    parser.add_argument("--accuracy-context", type=int, default=8192)
    parser.add_argument("--accuracy-lambda", type=float, default=0.003)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--q-tile-size", type=int, default=128)
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--length-tolerance-fraction", type=float, default=0.06)
    parser.add_argument("--length-tolerance-min-tokens", type=int, default=32)
    parser.add_argument("--state-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--denoising-threshold", type=float, default=0.9)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/blasst_ruler"),
    )
    return parser.parse_args()


def _values(text: str, cast: Callable[[str], Any]) -> list[Any]:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _balanced_counts(tasks: tuple[str, ...], total: int) -> dict[str, int]:
    if total < len(tasks):
        raise ValueError("sample total must be at least the number of tasks")
    quotient, remainder = divmod(total, len(tasks))
    return {
        task: quotient + int(index < remainder)
        for index, task in enumerate(tasks)
    }


def _length_tolerance(target: int, args: argparse.Namespace) -> int:
    return max(
        int(args.length_tolerance_min_tokens),
        math.ceil(target * float(args.length_tolerance_fraction)),
    )


def _verify_ruler_checkout(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        raise FileNotFoundError(
            f"{root} is not an NVIDIA/RULER checkout; clone {RULER_REPOSITORY}"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != RULER_COMMIT:
        raise ValueError(
            f"RULER commit mismatch: {commit}; expected pinned {RULER_COMMIT}"
        )
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=root, text=True
    ).strip()
    if "NVIDIA/RULER" not in remote:
        raise ValueError(f"unexpected RULER origin: {remote}")
    source_hashes: dict[str, str] = {}
    for relative in RULER_SOURCE_FILES:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"missing pinned RULER source: {path}")
        working = path.read_bytes()
        upstream = subprocess.check_output(
            ["git", "show", f"{RULER_COMMIT}:{relative}"],
            cwd=root,
        )
        if working != upstream:
            raise ValueError(f"tracked RULER source was modified: {relative}")
        source_hashes[relative] = _sha256_bytes(working)
    corpus = (
        root
        / "scripts/data/synthetic/json/PaulGrahamEssays.json"
    )
    if not corpus.exists():
        raise FileNotFoundError(
            "Official RULER PaulGrahamEssays.json is missing; run "
            "scripts/data/synthetic/json/download_paulgraham_essay.py"
        )
    return {
        "repository": RULER_REPOSITORY,
        "commit": commit,
        "origin": remote,
        "source_sha256": source_hashes,
        "paul_graham_corpus_sha256": _sha256_file(corpus),
    }


def _load_ruler_configuration(
    ruler_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    customized = yaml.safe_load(
        (ruler_root / "scripts/synthetic.yaml").read_text(encoding="utf-8")
    )
    constants_path = ruler_root / "scripts/data/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location(
        "pinned_ruler_data_constants", constants_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {constants_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return customized, module.TASKS


def _load_official_scorers(
    ruler_root: Path,
) -> tuple[dict[str, Callable[[list[str], list[list[str]]], float]], str]:
    path = ruler_root / "scripts/eval/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location(
        "pinned_ruler_eval_constants", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name: config["metric_fn"] for name, config in module.TASKS.items()
    }, _sha256_file(path)


def _ruler_subprocess_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    additions = [
        str(args.ruler_dependency_path),
        str(Path("/tmp/blasst-transformers")),
    ]
    current = environment.get("PYTHONPATH")
    if current:
        additions.append(current)
    environment["PYTHONPATH"] = os.pathsep.join(additions)
    environment["NLTK_DATA"] = str(args.nltk_data)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _generate_official_shard(
    *,
    ruler_root: Path,
    cache_dir: Path,
    tokenizer_path: str,
    task: str,
    target_length: int,
    tokens_to_generate: int,
    samples: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Path, list[str]]:
    shard_root = cache_dir / "official_raw" / str(target_length)
    output = shard_root / task / "validation.jsonl"
    command = [
        sys.executable,
        str(ruler_root / "scripts/data/prepare.py"),
        "--save_dir",
        str(shard_root),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        tokenizer_path,
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(target_length + tokens_to_generate),
        "--model_template_type",
        "base",
        "--num_samples",
        str(samples),
        "--random_seed",
        str(seed),
        "--subset",
        "validation",
    ]
    if output.exists() and len(_read_jsonl(output)) == samples:
        return output, command
    completed = subprocess.run(
        command,
        cwd=ruler_root,
        env=_ruler_subprocess_environment(args),
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    # NVIDIA's prepare.py catches child failures and may still return zero.
    failure_text = "Error output:" in completed.stdout
    if (
        completed.returncode != 0
        or failure_text
        or not output.exists()
        or len(_read_jsonl(output)) != samples
    ):
        raise RuntimeError(
            "official RULER generation failed\n"
            f"command: {' '.join(command)}\n"
            f"returncode: {completed.returncode}\n"
            f"stdout: {completed.stdout[-4000:]}\n"
            f"stderr: {completed.stderr[-4000:]}"
        )
    return output, command


def _candidate_record(
    raw: Mapping[str, Any],
    *,
    task: str,
    task_base: str,
    target: int,
    row_index: int,
    generator_seed: int,
    inference_seed: int,
    tokenizer: Any,
    tolerance: int,
) -> dict[str, Any]:
    prompt = str(raw["input"]) + str(raw.get("answer_prefix", ""))
    actual = len(tokenizer(prompt)["input_ids"])
    return {
        "sample_id": f"ruler_{target}_{task}_{row_index:03d}",
        "task": task,
        "task_base": task_base,
        "generator_seed": generator_seed,
        "inference_seed": inference_seed,
        "target_length": target,
        "actual_prompt_length": actual,
        "length_tolerance_tokens": tolerance,
        "within_length_tolerance": abs(actual - target) <= tolerance,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "prompt": prompt,
        "input": str(raw["input"]),
        "answer_prefix": str(raw.get("answer_prefix", "")),
        "outputs": [str(value) for value in raw["outputs"]],
        "official_index": raw.get("index"),
        "official_reported_length_including_generation": raw.get("length"),
    }


def _prepare_samples(
    args: argparse.Namespace,
    output_dir: Path,
    ruler_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    ruler_root = Path(args.ruler_root)
    cache_dir = output_dir / "ruler_cache"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    customized, base_tasks = _load_ruler_configuration(ruler_root)
    context_records: list[dict[str, Any]] = []
    pool_by_length: dict[int, dict[str, list[dict[str, Any]]]] = {}
    commands: list[list[str]] = []

    for target in CONTEXTS:
        tasks = SHORT_512_TASKS if target == 512 else PAPER_TASKS
        context_counts = _balanced_counts(tasks, args.context_samples)
        pool_counts = dict(context_counts)
        if target == args.accuracy_context:
            accuracy_counts = _balanced_counts(PAPER_TASKS, args.accuracy_samples)
            pool_counts = {
                task: max(context_counts.get(task, 0), accuracy_counts[task])
                for task in PAPER_TASKS
            }
        tolerance = _length_tolerance(target, args)
        pool_by_length[target] = {}
        for task_index, task in enumerate(tasks if target != 8192 else PAPER_TASKS):
            desired = pool_counts[task]
            if task in (
                "niah_multivalue",
                "niah_multiquery",
            ):
                candidates = desired * 20
            elif task.startswith("niah_"):
                candidates = desired * 6
            elif task == "fwe":
                candidates = desired * 4
            else:
                candidates = desired + max(
                    8, math.ceil(desired * 0.50)
                )
            config = customized[task]
            task_base = str(config["task"])
            tokens_to_generate = int(base_tasks[task_base]["tokens_to_generate"])
            generator_seed = (
                args.seed * 100_000 + target + task_index * 1_000
            )
            shard, command = _generate_official_shard(
                ruler_root=ruler_root,
                cache_dir=cache_dir,
                tokenizer_path=args.model_path,
                task=task,
                target_length=target,
                tokens_to_generate=tokens_to_generate,
                samples=candidates,
                seed=generator_seed,
                args=args,
            )
            commands.append(command)
            records = [
                _candidate_record(
                    raw,
                    task=task,
                    task_base=task_base,
                    target=target,
                    row_index=row_index,
                    generator_seed=generator_seed,
                    inference_seed=(
                        args.seed * 1_000_000
                        + target * 100
                        + task_index * 10_000
                        + row_index
                    ),
                    tokenizer=tokenizer,
                    tolerance=tolerance,
                )
                for row_index, raw in enumerate(_read_jsonl(shard))
            ]
            valid = [row for row in records if row["within_length_tolerance"]]
            if len(valid) < desired:
                raise AssertionError(
                    f"RULER {task}/{target} supplied {len(valid)}/{desired} "
                    f"valid samples; actual lengths="
                    f"{[row['actual_prompt_length'] for row in records]}"
                )
            pool_by_length[target][task] = valid[:desired]

        # Round-robin selection keeps every prefix of the manifest balanced.
        selected_by_task = {
            task: pool_by_length[target][task][: context_counts[task]]
            for task in tasks
        }
        for row_index in range(max(context_counts.values())):
            for task in tasks:
                if row_index < len(selected_by_task[task]):
                    context_records.append(selected_by_task[task][row_index])

    context_path = cache_dir / "ruler_context_samples.jsonl"
    _write_jsonl(context_path, context_records)
    eight_k_context = [
        row
        for row in context_records
        if int(row["target_length"]) == args.accuracy_context
    ]
    block_path = cache_dir / "ruler_block_samples_8k.jsonl"
    _write_jsonl(block_path, eight_k_context)

    accuracy_counts = _balanced_counts(PAPER_TASKS, args.accuracy_samples)
    accuracy_records: list[dict[str, Any]] = []
    for row_index in range(max(accuracy_counts.values())):
        for task in PAPER_TASKS:
            if row_index < accuracy_counts[task]:
                accuracy_records.append(
                    pool_by_length[args.accuracy_context][task][row_index]
                )
    accuracy_path = cache_dir / "ruler_accuracy_samples_8k.jsonl"
    _write_jsonl(accuracy_path, accuracy_records)

    manifest = {
        "schema_version": 1,
        "ruler": dict(ruler_provenance),
        "model_tokenizer": args.model_path,
        "actual_length_measurement": (
            "len(AutoTokenizer(prompt)['input_ids']), including answer_prefix "
            "and tokenizer special tokens"
        ),
        "tolerance": {
            "fraction": args.length_tolerance_fraction,
            "minimum_tokens": args.length_tolerance_min_tokens,
            "rule": "max(minimum_tokens, ceil(target*fraction))",
        },
        "context_samples": len(context_records),
        "block_samples": len(eight_k_context),
        "accuracy_samples": len(accuracy_records),
        "context_tasks": {
            str(target): (
                list(SHORT_512_TASKS) if target == 512 else list(PAPER_TASKS)
            )
            for target in CONTEXTS
        },
        "generator_commands": commands,
        "files": {
            "context": {
                "path": str(context_path),
                "sha256": _sha256_file(context_path),
            },
            "block": {
                "path": str(block_path),
                "sha256": _sha256_file(block_path),
            },
            "accuracy": {
                "path": str(accuracy_path),
                "sha256": _sha256_file(accuracy_path),
            },
        },
    }
    (cache_dir / "ruler_sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _validate_sample_manifests(args, output_dir)
    return manifest


def _validate_sample_manifests(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    cache_dir = output_dir / "ruler_cache"
    context_path = cache_dir / "ruler_context_samples.jsonl"
    block_path = cache_dir / "ruler_block_samples_8k.jsonl"
    accuracy_path = cache_dir / "ruler_accuracy_samples_8k.jsonl"
    manifest_path = cache_dir / "ruler_sample_manifest.json"
    context = _read_jsonl(context_path)
    block = _read_jsonl(block_path)
    accuracy = _read_jsonl(accuracy_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    if len(context) != len(CONTEXTS) * args.context_samples:
        violations.append(f"context sample count {len(context)}")
    if len(block) != args.context_samples:
        violations.append(f"block sample count {len(block)}")
    if len(accuracy) != args.accuracy_samples:
        violations.append(f"accuracy sample count {len(accuracy)}")
    for target in CONTEXTS:
        subset = [row for row in context if int(row["target_length"]) == target]
        if len(subset) != args.context_samples:
            violations.append(f"context {target} count {len(subset)}")
        expected_tasks = SHORT_512_TASKS if target == 512 else PAPER_TASKS
        expected_counts = _balanced_counts(expected_tasks, args.context_samples)
        observed_counts = Counter(str(row["task"]) for row in subset)
        if observed_counts != Counter(expected_counts):
            violations.append(
                f"context {target} task balance {dict(observed_counts)}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    token_lengths: dict[str, int] = {}
    unique_records = {
        str(row["sample_id"]): row for row in context + block + accuracy
    }
    for row in unique_records.values():
        sample_id = str(row["sample_id"])
        prompt = str(row["prompt"])
        digest = _sha256_bytes(prompt.encode("utf-8"))
        if digest != str(row["prompt_sha256"]):
            violations.append(f"prompt hash mismatch {sample_id}")
        token_lengths.setdefault(
            digest, len(tokenizer(prompt)["input_ids"])
        )
        if token_lengths[digest] != int(row["actual_prompt_length"]):
            violations.append(f"tokenizer length mismatch {sample_id}")
        for seed_field in ("generator_seed", "inference_seed"):
            if not isinstance(row.get(seed_field), int):
                violations.append(f"missing integer {seed_field} {sample_id}")
        if not bool(row["within_length_tolerance"]):
            violations.append(f"length violation {sample_id}")
        if abs(
            int(row["actual_prompt_length"]) - int(row["target_length"])
        ) > int(row["length_tolerance_tokens"]):
            violations.append(f"length mismatch {sample_id}")

    for name, path in (
        ("context", context_path),
        ("block", block_path),
        ("accuracy", accuracy_path),
    ):
        if _sha256_file(path) != str(manifest["files"][name]["sha256"]):
            violations.append(f"manifest file hash mismatch {name}")

    for name, records in (
        ("context", context),
        ("block", block),
        ("accuracy", accuracy),
    ):
        ids = [str(row["sample_id"]) for row in records]
        if len(ids) != len(set(ids)):
            violations.append(f"duplicate sample IDs in {name}")

    context_8k = [
        row
        for row in context
        if int(row["target_length"]) == args.accuracy_context
    ]
    if block != context_8k:
        violations.append("block samples do not exactly reuse context 8K samples")
    if accuracy[: len(context_8k)] != context_8k:
        violations.append("context 8K samples are not reused by accuracy")
    expected_accuracy_counts = _balanced_counts(
        PAPER_TASKS, args.accuracy_samples
    )
    observed_accuracy_counts = Counter(str(row["task"]) for row in accuracy)
    if observed_accuracy_counts != Counter(expected_accuracy_counts):
        violations.append(
            f"accuracy task balance {dict(observed_accuracy_counts)}"
        )
    result = {
        "passed": not violations,
        "violations": violations,
        "context_samples": len(context),
        "block_samples": len(block),
        "accuracy_samples": len(accuracy),
        "block_reuses_context_8k": True,
        "accuracy_reuses_context_8k": True,
        "cached_file_hashes_verified": True,
        "prompt_hashes_verified": True,
        "model_tokenizer_lengths_recomputed": True,
        "task_balance_verified": True,
        "unique_sample_ids_verified": True,
        "integer_seeds_verified": True,
        "all_lengths_within_tolerance": True,
        "unrelated_alpaca_padding": False,
    }
    if violations:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _fixed_dense_state(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    seed: int,
    mask_token_id: int,
    block_size: int = 16,
    required_tokens: int = 64,
) -> list[int]:
    set_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    max_new_tokens = math.ceil(
        (required_tokens + block_size - 1) / block_size
    ) * block_size
    # Prompt/block alignment can consume up to block_size-1 positions.
    max_new_tokens += block_size
    generated = model.mdm_sample(
        encoded.clone(),
        tokenizer=tokenizer,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
        small_block_size=block_size,
        min_len=encoded.shape[1],
        seq_len=torch.tensor([encoded.shape[1]], device=model.device),
        mask_id=mask_token_id,
        stop_token=-1,
        use_block_cache=False,
        threshold=0.9,
        temperature=0.0,
    )[0]
    completion = generated[encoded.shape[1] :].detach().cpu().tolist()
    if len(completion) < required_tokens:
        raise AssertionError(
            f"fixed dense state has {len(completion)} < {required_tokens} tokens"
        )
    return [int(token) for token in completion[:required_tokens]]


def _dense_probe(
    model: Any,
    tokenizer: Any,
) -> torch.Tensor:
    probe = tokenizer(
        "RULER dense dispatch regression probe.",
        return_tensors="pt",
    )["input_ids"].to(model.device)
    if probe.shape[1] < 16:
        probe = torch.cat(
            (
                probe,
                torch.full(
                    (1, 16 - probe.shape[1]),
                    int(getattr(model.config, "mask_token_id", 151665)),
                    dtype=torch.long,
                    device=model.device,
                ),
            ),
            dim=1,
        )
    probe = probe[:, :16]
    with torch.no_grad():
        return model.model(
            input_ids=probe,
            use_cache=False,
            use_block_cache=False,
            block_size=16,
        ).last_hidden_state.detach()


def _dense_disabled_regression(
    before: torch.Tensor,
    after: torch.Tensor,
) -> dict[str, Any]:
    return {
        "exact_match": bool(torch.equal(before, after)),
        "all_finite": bool(
            torch.isfinite(before).all() and torch.isfinite(after).all()
        ),
        "max_abs_error": float((before - after).abs().max()),
    }


def _reset_sweep_stats(runtime: Any, lambdas: list[float]) -> None:
    runtime.sweep_lambdas = tuple(lambdas)
    runtime.sweep_stats = {
        value: Blasst2DStats(record_layers=False, record_heads=False)
        for value in lambdas
    }


def _load_state_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state_cache(path: Path, states: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(states), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _observe_sample(
    *,
    model: Any,
    tokenizer: Any,
    runtime: Any,
    config: Blasst2DConfig,
    sample: Mapping[str, Any],
    experiments: tuple[str, ...],
    block_sizes: list[int],
    ratios: list[float],
    lambdas: list[float],
    state_cache: dict[str, Any],
    state_cache_path: Path,
    mask_token_id: int,
) -> tuple[list[dict[str, Any]], int]:
    sample_id = str(sample["sample_id"])
    state_entry = state_cache.get(sample_id)
    if state_entry is None:
        runtime.config = replace(config, enable_blasst_2d=False)
        runtime.sweep_lambdas = ()
        state_tokens = _fixed_dense_state(
            model,
            tokenizer,
            str(sample["prompt"]),
            seed=int(sample["inference_seed"]),
            mask_token_id=mask_token_id,
            required_tokens=64,
        )
        state_entry = {
            "sample_id": sample_id,
            "prompt_sha256": sample["prompt_sha256"],
            "inference_seed": sample["inference_seed"],
            "state_tokens": state_tokens,
            "state_construction": (
                "first 64 tokens from deterministic dense block-16 diffusion "
                "generation with stop disabled"
            ),
        }
        state_cache[sample_id] = state_entry
        _save_state_cache(state_cache_path, state_cache)
    if state_entry["prompt_sha256"] != sample["prompt_sha256"]:
        raise AssertionError(f"state/prompt hash mismatch for {sample_id}")

    prompt_ids = tokenizer(
        str(sample["prompt"]), return_tensors="pt"
    )["input_ids"].to(model.device)
    if prompt_ids.shape[1] != int(sample["actual_prompt_length"]):
        raise AssertionError(f"prompt retokenization mismatch for {sample_id}")
    runtime.config = replace(config, enable_blasst_2d=False)
    runtime.sweep_lambdas = ()
    with torch.no_grad():
        prefix_output = model.model(
            input_ids=prompt_ids,
            use_cache=True,
            update_past_key_values=True,
            use_block_cache=False,
            block_size=1,
        )
    prefix_cache = prefix_output.past_key_values
    if prefix_cache.get_seq_length() != prompt_ids.shape[1]:
        raise AssertionError(f"ordinary cache length mismatch for {sample_id}")
    del prefix_output

    clean64 = torch.tensor(
        state_entry["state_tokens"],
        dtype=torch.long,
        device=model.device,
    )
    all_rows: list[dict[str, Any]] = []
    finite_outputs = 0
    for experiment in experiments:
        _reset_sweep_stats(runtime, lambdas)
        runtime.config = config
        sizes = [16] if experiment == "context_length" else block_sizes
        for block_size in sizes:
            states = _masked_queries(
                clean64,
                block_size,
                ratios,
                mask_token_id,
                int(sample["inference_seed"]),
            )
            for step, requested_ratio, noisy in states:
                metadata = {
                    "experiment": experiment,
                    "sweep_value": (
                        int(sample["target_length"])
                        if experiment == "context_length"
                        else block_size
                    ),
                    "benchmark": sample["task"],
                    "example_id": sample_id,
                    "denoising_step": step,
                    "mask_ratio": float(
                        (noisy == mask_token_id).sum().item() / block_size
                    ),
                    "requested_mask_ratio": requested_ratio,
                    "masked_tokens": int(
                        (noisy == mask_token_id).sum().item()
                    ),
                }
                with torch.no_grad():
                    output = model.model(
                        input_ids=noisy[None],
                        past_key_values=prefix_cache,
                        use_cache=True,
                        update_past_key_values=False,
                        use_block_cache=False,
                        block_size=block_size,
                        blasst_active_query_mask=torch.ones(
                            (1, block_size),
                            dtype=torch.bool,
                            device=model.device,
                        ),
                        blasst_metadata=metadata,
                    )
                if not torch.isfinite(output.last_hidden_state).all():
                    raise FloatingPointError(
                        f"non-finite RULER sweep output {sample_id}"
                    )
                finite_outputs += 1
                del output
        rows = _stats_rows(runtime, lambdas)
        expected = len(sizes) * len(ratios) * len(lambdas)
        if len(rows) != expected:
            raise AssertionError(
                f"{sample_id}/{experiment}: {len(rows)} != {expected} rows"
            )
        for row in rows:
            row_sparsity = float(row["row_vote_sparsity"])
            row["physical_to_row_sparsity_ratio"] = (
                float(row["physical_tile_sparsity"]) / row_sparsity
                if row_sparsity
                else 0.0
            )
            row["actual_prompt_length"] = int(sample["actual_prompt_length"])
            row["target_context_length"] = int(sample["target_length"])
            row["task"] = str(sample["task"])
            row["sample_id"] = sample_id
            row["prompt_sha256"] = str(sample["prompt_sha256"])
        all_rows.extend(rows)
    del prefix_cache, prompt_ids, clean64
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_rows, finite_outputs


def _join_prompt_length_summary(
    rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    axis: str,
) -> list[dict[str, Any]]:
    for row in rows:
        value = int(row[axis])
        experiment = (
            "context_length" if axis == "target_context_length" else "block_size"
        )
        subset = [
            raw
            for raw in raw_rows
            if raw["experiment"] == experiment
            and int(raw["sweep_value"]) == value
            and float(raw["lambda"]) == float(row["lambda"])
        ]
        prompt_lengths = {
            (raw["sample_id"], int(raw["actual_prompt_length"]))
            for raw in subset
        }
        valid_lengths = [int(raw["valid_kv_length"]) for raw in subset]
        row["samples"] = len(prompt_lengths)
        row["actual_prompt_length_min"] = min(
            length for _, length in prompt_lengths
        )
        row["actual_prompt_length_mean"] = sum(
            length for _, length in prompt_lengths
        ) / len(prompt_lengths)
        row["actual_prompt_length_max"] = max(
            length for _, length in prompt_lengths
        )
        row["observed_valid_kv_length_min"] = min(valid_lengths)
        row["observed_valid_kv_length_mean"] = (
            sum(valid_lengths) / len(valid_lengths)
        )
        row["observed_valid_kv_length_max"] = max(valid_lengths)
    return rows


def _rename_sweep_axis(
    rows: list[dict[str, Any]],
    axis: str,
) -> list[dict[str, Any]]:
    return [
        {
            axis: int(row["sweep_value"]),
            **{
                key: value
                for key, value in row.items()
                if key not in ("sweep_value", "experiment")
            },
        }
        for row in rows
    ]


def _validate_sweeps(
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    context_summary: list[dict[str, Any]],
    block_summary: list[dict[str, Any]],
    lambdas: list[float],
    dense_regression: Mapping[str, Any],
    finite_outputs: int,
) -> dict[str, Any]:
    violations: list[str] = []
    for row in raw_rows + context_summary + block_summary:
        for field in COUNT_FIELDS:
            if int(row[field]) < 0:
                violations.append(f"negative {field}: {row}")
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append(f"count identity failed: {row}")
        for metric in (
            "row_vote_sparsity",
            "physical_tile_sparsity",
            "valid_element_sparsity",
            "physical_to_row_sparsity_ratio",
        ):
            if not math.isfinite(float(row[metric])):
                violations.append(f"non-finite {metric}: {row}")
        eligible = int(row["eligible_tiles"])
        votes = int(row["valid_row_votes"])
        elements = int(row["valid_elements"])
        expected_ratios = {
            "row_vote_sparsity": (
                int(row["skippable_row_votes"]) / votes if votes else 0.0
            ),
            "physical_tile_sparsity": (
                int(row["skipped_tiles"]) / eligible if eligible else 0.0
            ),
            "valid_element_sparsity": (
                int(row["skipped_valid_elements"]) / elements
                if elements
                else 0.0
            ),
        }
        expected_ratios["physical_to_row_sparsity_ratio"] = (
            expected_ratios["physical_tile_sparsity"]
            / expected_ratios["row_vote_sparsity"]
            if expected_ratios["row_vote_sparsity"]
            else 0.0
        )
        for metric, expected in expected_ratios.items():
            if not math.isclose(
                float(row[metric]), expected, rel_tol=0.0, abs_tol=1e-15
            ):
                violations.append(f"ratio not derived from counts {metric}: {row}")

    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    stratum_fields = (
        "experiment",
        "sweep_value",
        "task",
        "sample_id",
        "denoising_step",
        "mask_ratio",
        "masked_tokens",
        "query_length",
        "valid_kv_length",
    )
    for row in raw_rows:
        strata.setdefault(
            tuple(row[field] for field in stratum_fields), []
        ).append(row)
        experiment = str(row["experiment"])
        query = int(row["query_length"])
        sweep_value = int(row["sweep_value"])
        expected_query = 16 if experiment == "context_length" else sweep_value
        if query != expected_query:
            violations.append(f"query-length mismatch: {row}")
        expected_kv = int(row["actual_prompt_length"]) + query
        if int(row["valid_kv_length"]) != expected_kv:
            violations.append(f"valid-KV mismatch: {row}")
        if query == 8 and not (
            experiment == "block_size" and sweep_value == 8
        ):
            violations.append(f"residual sub-block q=8: {row}")
    for key, group in strata.items():
        ordered = sorted(group, key=lambda row: float(row["lambda"]))
        if [float(row["lambda"]) for row in ordered] != sorted(lambdas):
            violations.append(f"incomplete lambda grid: {key}")
            continue
        for lower, upper in zip(ordered, ordered[1:]):
            for field in ("skipped_tiles", "skippable_row_votes"):
                if int(upper[field]) < int(lower[field]):
                    violations.append(f"non-monotonic {field}: {key}")
            for field in (
                "eligible_tiles",
                "valid_row_votes",
                "valid_elements",
                "structurally_masked_tiles",
            ):
                if int(upper[field]) != int(lower[field]):
                    violations.append(f"lambda-dependent denominator: {key}")

    if sorted(
        {int(row["target_context_length"]) for row in context_summary}
    ) != list(CONTEXTS):
        violations.append("context grid mismatch")
    if sorted({int(row["block_size"]) for row in block_summary}) != list(
        BLOCK_SIZES
    ):
        violations.append("block grid mismatch")
    if not dense_regression["exact_match"] or not dense_regression["all_finite"]:
        violations.append("dense-disabled regression failed")
    expected_raw_rows = (
        args.context_samples
        * (len(CONTEXTS) + len(BLOCK_SIZES))
        * len(MASK_RATIOS)
        * len(lambdas)
    )
    if len(raw_rows) != expected_raw_rows:
        violations.append(
            f"raw row count {len(raw_rows)} != {expected_raw_rows}"
        )

    # The 8K/context-q16 and block-q16 observations share exact prompts,
    # prompt caches, generated clean states, masks, and FP32 score traces.
    for value in lambdas:
        left = next(
            row
            for row in context_summary
            if int(row["target_context_length"]) == args.accuracy_context
            and float(row["lambda"]) == value
        )
        right = next(
            row
            for row in block_summary
            if int(row["block_size"]) == 16
            and float(row["lambda"]) == value
        )
        for field in COUNT_FIELDS:
            if int(left[field]) != int(right[field]):
                violations.append(
                    f"cross-experiment mismatch λ={value}, {field}"
                )

    checks = {
        "passed": not violations,
        "violations": violations,
        "dense_disabled_regression": dict(dense_regression),
        "finite_model_outputs_executed_this_invocation": finite_outputs,
        "finite_model_outputs_validated_before_checkpoint": (
            len(raw_rows) // len(lambdas)
        ),
        "raw_strata_rows": len(raw_rows),
        "expected_raw_strata_rows": expected_raw_rows,
        "same_fp32_maxima_reused_across_lambda": True,
        "monotonicity_checked_within_identical_score_traces": True,
        "aggregation": "ratios of summed integer counts",
        "structural_masking_excluded_from_eligible_denominators": True,
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "query_length_gate": (
            "context q=16; block q=configured block; q=8 only at block_size=8"
        ),
        "cross_experiment_8k_block16_identical": True,
    }
    if violations:
        raise AssertionError(json.dumps(checks, indent=2))
    return checks


def _run_sweeps(
    args: argparse.Namespace,
    output_dir: Path,
    lambdas: list[float],
    block_sizes: list[int],
    ratios: list[float],
) -> None:
    started = time.monotonic()
    manifest_checks = _validate_sample_manifests(args, output_dir)
    cache_dir = output_dir / "ruler_cache"
    context_samples = _read_jsonl(
        cache_dir / "ruler_context_samples.jsonl"
    )
    block_ids = {
        row["sample_id"]
        for row in _read_jsonl(cache_dir / "ruler_block_samples_8k.jsonl")
    }
    model, tokenizer = load_model(
        args.model_path, args.device, args.precision
    )
    mask_token_id = int(getattr(model.config, "mask_token_id", 151665))
    config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=lambdas[0],
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
    )
    dense_before = _dense_probe(model, tokenizer)
    runtime = install_blasst_2d(
        model,
        replace(config, enable_blasst_2d=False),
        mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ordinary_cache_queries_only=True,
    )
    runtime.config = replace(config, enable_blasst_2d=False)
    runtime.sweep_lambdas = ()
    dense_after = _dense_probe(model, tokenizer)
    dense_regression = _dense_disabled_regression(dense_before, dense_after)
    checkpoint = output_dir / "ruler_sweep_raw.jsonl"
    existing = _read_jsonl(checkpoint) if checkpoint.exists() else []
    checkpoint_rows_at_start = len(existing)
    completed = {
        (str(row["experiment"]), str(row["sample_id"])) for row in existing
    }
    state_cache_path = cache_dir / "ruler_sweep_states.json"
    state_cache = _load_state_cache(state_cache_path)
    finite_outputs = 0

    for index, sample in enumerate(context_samples, start=1):
        experiments = []
        sample_id = str(sample["sample_id"])
        if ("context_length", sample_id) not in completed:
            experiments.append("context_length")
        if (
            sample_id in block_ids
            and ("block_size", sample_id) not in completed
        ):
            experiments.append("block_size")
        if not experiments:
            continue
        rows, finite = _observe_sample(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            config=config,
            sample=sample,
            experiments=tuple(experiments),
            block_sizes=block_sizes,
            ratios=ratios,
            lambdas=lambdas,
            state_cache=state_cache,
            state_cache_path=state_cache_path,
            mask_token_id=mask_token_id,
        )
        _append_jsonl(checkpoint, rows)
        finite_outputs += finite
        print(
            f"completed RULER sweep sample {index}/{len(context_samples)}: "
            f"{sample_id} ({','.join(experiments)})",
            flush=True,
        )

    raw_rows = _read_jsonl(checkpoint)
    for row in raw_rows:
        row_sparsity = float(row["row_vote_sparsity"])
        row.setdefault(
            "physical_to_row_sparsity_ratio",
            (
                float(row["physical_tile_sparsity"]) / row_sparsity
                if row_sparsity
                else 0.0
            ),
        )
    context_raw = [
        row for row in raw_rows if row["experiment"] == "context_length"
    ]
    block_raw = [
        row for row in raw_rows if row["experiment"] == "block_size"
    ]
    context_summary = _rename_sweep_axis(
        _aggregate(context_raw, ("sweep_value", "lambda")),
        "target_context_length",
    )
    block_summary = _rename_sweep_axis(
        _aggregate(block_raw, ("sweep_value", "lambda")),
        "block_size",
    )
    context_summary = _join_prompt_length_summary(
        context_summary, raw_rows, "target_context_length"
    )
    block_summary = _join_prompt_length_summary(
        block_summary, raw_rows, "block_size"
    )
    context_per_task = _rename_sweep_axis(
        _aggregate(
            context_raw,
            ("sweep_value", "lambda", "benchmark"),
        ),
        "target_context_length",
    )
    block_per_task = _rename_sweep_axis(
        _aggregate(
            block_raw,
            ("sweep_value", "lambda", "benchmark"),
        ),
        "block_size",
    )
    for row in context_per_task + block_per_task:
        row["task"] = row.pop("benchmark", row.get("task", ""))
    context_per_example = _rename_sweep_axis(
        _aggregate(
            context_raw,
            ("sweep_value", "lambda", "benchmark", "example_id"),
        ),
        "target_context_length",
    )
    block_per_example = _rename_sweep_axis(
        _aggregate(
            block_raw,
            ("sweep_value", "lambda", "benchmark", "example_id"),
        ),
        "block_size",
    )
    for row in context_per_example + block_per_example:
        row["task"] = row.pop("benchmark", "")
        row["sample_id"] = row.pop("example_id")

    checks = _validate_sweeps(
        args=args,
        raw_rows=raw_rows,
        context_summary=context_summary,
        block_summary=block_summary,
        lambdas=lambdas,
        dense_regression=dense_regression,
        finite_outputs=finite_outputs,
    )
    checks["sample_manifest_validation"] = manifest_checks
    _write_csv(output_dir / "ruler_context_sweep.csv", context_summary)
    _write_csv(output_dir / "ruler_block_size_sweep.csv", block_summary)
    _write_csv(
        output_dir / "ruler_context_sweep_per_task.csv",
        context_per_task,
    )
    _write_csv(
        output_dir / "ruler_block_size_sweep_per_task.csv",
        block_per_task,
    )
    _write_csv(
        output_dir / "ruler_context_sweep_per_example.csv",
        context_per_example,
    )
    _write_csv(
        output_dir / "ruler_block_size_sweep_per_example.csv",
        block_per_example,
    )
    (output_dir / "ruler_sweep_correctness.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_curves(
        context_summary,
        "actual_prompt_length_mean",
        "row_vote_sparsity",
        lambdas,
        "2D-BLASST row-vote sparsity on RULER",
        "Mean actual RULER prompt length (model tokens)",
        output_dir / "ruler_context_row_sparsity",
    )
    _plot_curves(
        context_summary,
        "actual_prompt_length_mean",
        "physical_tile_sparsity",
        lambdas,
        "2D-BLASST physical tile sparsity on RULER",
        "Mean actual RULER prompt length (model tokens)",
        output_dir / "ruler_context_physical_sparsity",
    )
    _plot_curves(
        block_summary,
        "block_size",
        "row_vote_sparsity",
        lambdas,
        "RULER row-vote sparsity vs diffusion block size",
        "Diffusion block size / effective query length",
        output_dir / "ruler_block_row_sparsity",
    )
    _plot_curves(
        block_summary,
        "block_size",
        "physical_tile_sparsity",
        lambdas,
        "RULER physical tile sparsity vs diffusion block size",
        "Diffusion block size / effective query length",
        output_dir / "ruler_block_physical_sparsity",
    )
    runtime_path = output_dir / "ruler_sweep_runtime.json"
    prior_runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {}
    )
    sweep_runtime = {
        **prior_runtime,
        "last_invocation_wall_seconds": time.monotonic() - started,
        "finite_outputs_executed_last_invocation": finite_outputs,
        "checkpoint_rows": len(raw_rows),
    }
    if checkpoint_rows_at_start == 0 and finite_outputs:
        sweep_runtime["full_sweep_wall_seconds"] = sweep_runtime[
            "last_invocation_wall_seconds"
        ]
    sweep_runtime.pop("wall_seconds", None)
    sweep_runtime.pop("finite_outputs_this_invocation", None)
    runtime_path.write_text(
        json.dumps(sweep_runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _official_postprocess(prediction: str) -> str:
    # Exact logic from pinned scripts/eval/evaluate.py::postprocess_pred.
    prediction = prediction.strip()
    return re.sub(r"[\x00-\x1f]", "\n", prediction).strip()


def _score_one(
    scorer: Callable[[list[str], list[list[str]]], float],
    prediction: str,
    references: list[str],
) -> float:
    return float(scorer([_official_postprocess(prediction)], [references])) / 100.0


def _score_subset(
    scorers: Mapping[
        str, Callable[[list[str], list[list[str]]], float]
    ],
    rows: list[Mapping[str, Any]],
    prediction_field: str,
) -> float:
    """Run the official scorer once over a complete reporting subset."""
    if not rows:
        raise ValueError("cannot score an empty RULER subset")
    metric_functions = {
        scorers[str(row["task_base"])] for row in rows
    }
    if len(metric_functions) != 1:
        raise ValueError("RULER subset mixes incompatible official metrics")
    scorer = next(iter(metric_functions))
    predictions = [
        _official_postprocess(str(row[prediction_field])) for row in rows
    ]
    references = [
        [str(value) for value in row["outputs"]] for row in rows
    ]
    return float(scorer(predictions, references)) / 100.0


def _reference_hit_pattern(
    prediction: str,
    references: list[str],
) -> tuple[bool, ...]:
    normalized = _official_postprocess(prediction).lower()
    return tuple(reference.lower() in normalized for reference in references)


def _task_generation_budget(
    task: str,
    ruler_root: Path,
    block_size: int,
) -> int:
    customized, base = _load_ruler_configuration(ruler_root)
    tokens = int(base[customized[task]["task"]]["tokens_to_generate"])
    return math.ceil(tokens / block_size) * block_size


def _accuracy_row_counts(stats: Blasst2DStats) -> dict[str, Any]:
    summary = stats.summary()
    query_lengths = sorted(
        {int(row["query_length"]) for row in stats.per_step.values()}
    )
    if query_lengths != [16]:
        raise AssertionError(f"accuracy query lengths: {query_lengths}")
    return {**summary, "query_lengths": query_lengths}


def _run_accuracy(
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    started = time.monotonic()
    manifest_checks = _validate_sample_manifests(args, output_dir)
    ruler_root = Path(args.ruler_root)
    scorers, scorer_hash = _load_official_scorers(ruler_root)
    samples = _read_jsonl(
        output_dir / "ruler_cache/ruler_accuracy_samples_8k.jsonl"
    )
    model, tokenizer = load_model(
        args.model_path, args.device, args.precision
    )
    mask_token_id = int(getattr(model.config, "mask_token_id", 151665))
    config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=args.accuracy_lambda,
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=True,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
    )
    runtime = install_blasst_2d(
        model,
        replace(config, enable_blasst_2d=False),
        mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        ordinary_cache_queries_only=True,
    )
    runtime.sweep_lambdas = ()
    checkpoint = output_dir / "ruler_accuracy_8k_per_example.jsonl"
    rows = _read_jsonl(checkpoint) if checkpoint.exists() else []
    completed_at_start = len(rows)
    completed = {row["sample_id"] for row in rows}
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        task = str(sample["task"])
        task_base = str(sample["task_base"])
        scorer = scorers[task_base]
        generation_budget = _task_generation_budget(
            task, ruler_root, args.block_size
        )
        runtime.config = replace(config, enable_blasst_2d=False)
        runtime.stats = Blasst2DStats(record_layers=False, record_heads=False)
        set_seed(int(sample["inference_seed"]))
        dense = generate_one(
            model,
            tokenizer,
            str(sample["prompt"]),
            block_size=args.block_size,
            max_new_tokens=generation_budget,
            threshold=args.denoising_threshold,
        )
        runtime.config = config
        runtime.stats = Blasst2DStats(record_layers=False, record_heads=False)
        set_seed(int(sample["inference_seed"]))
        sparse = generate_one(
            model,
            tokenizer,
            str(sample["prompt"]),
            block_size=args.block_size,
            max_new_tokens=generation_budget,
            threshold=args.denoising_threshold,
        )
        sparse_counts = _accuracy_row_counts(runtime.stats)
        references = [str(value) for value in sample["outputs"]]
        dense_prediction = str(dense["completion"])
        sparse_prediction = str(sparse["completion"])
        dense_score = _score_one(scorer, dense_prediction, references)
        sparse_score = _score_one(scorer, sparse_prediction, references)
        dense_hits = _reference_hit_pattern(dense_prediction, references)
        sparse_hits = _reference_hit_pattern(sparse_prediction, references)
        row = {
            "sample_id": sample_id,
            "task": task,
            "task_base": task_base,
            "target_length": sample["target_length"],
            "actual_prompt_length": sample["actual_prompt_length"],
            "prompt_sha256": sample["prompt_sha256"],
            "inference_seed": sample["inference_seed"],
            "outputs": references,
            "generation_budget": generation_budget,
            "dense_prediction": dense_prediction,
            "sparse_prediction": sparse_prediction,
            "dense_completion_tokens": dense["completion_tokens"],
            "sparse_completion_tokens": sparse["completion_tokens"],
            "dense_accuracy": dense_score,
            "sparse_accuracy": sparse_score,
            "accuracy_change": sparse_score - dense_score,
            "dense_reference_hit_pattern": dense_hits,
            "sparse_reference_hit_pattern": sparse_hits,
            "final_answer_agreement_with_dense": int(
                dense_hits == sparse_hits
            ),
            "sequence_exact_match_with_dense": int(
                dense["completion_tokens"] == sparse["completion_tokens"]
            ),
            "normalized_token_difference_from_dense": (
                normalized_token_difference(
                    dense["completion_tokens"],
                    sparse["completion_tokens"],
                )
            ),
            **sparse_counts,
        }
        _append_jsonl(checkpoint, [row])
        rows.append(row)
        print(
            f"completed paired RULER accuracy {index}/{len(samples)}: "
            f"{sample_id}",
            flush=True,
        )

    if len(rows) != args.accuracy_samples:
        raise AssertionError(f"accuracy rows {len(rows)}")
    summary_rows: list[dict[str, Any]] = []
    for scope, task in [("overall", "all")] + [
        ("task", name) for name in PAPER_TASKS
    ]:
        subset = rows if scope == "overall" else [
            row for row in rows if row["task"] == task
        ]
        aggregated = _aggregate(subset, ())[0]
        prompt_lengths = [int(row["actual_prompt_length"]) for row in subset]
        dense_accuracy = _score_subset(
            scorers, subset, "dense_prediction"
        )
        sparse_accuracy = _score_subset(
            scorers, subset, "sparse_prediction"
        )
        accuracy_change = sparse_accuracy - dense_accuracy
        summary_rows.append(
            {
                "scope": scope,
                "task": task,
                "samples": len(subset),
                "dense_accuracy": dense_accuracy,
                "sparse_accuracy": sparse_accuracy,
                "accuracy_change": accuracy_change,
                "absolute_accuracy_change": abs(accuracy_change),
                "final_answer_agreement_with_dense": sum(
                    int(row["final_answer_agreement_with_dense"])
                    for row in subset
                )
                / len(subset),
                "sequence_exact_match_with_dense": sum(
                    int(row["sequence_exact_match_with_dense"])
                    for row in subset
                )
                / len(subset),
                "actual_prompt_length_min": min(prompt_lengths),
                "actual_prompt_length_mean": sum(prompt_lengths)
                / len(prompt_lengths),
                "actual_prompt_length_max": max(prompt_lengths),
                **{
                    key: value
                    for key, value in aggregated.items()
                    if key in COUNT_FIELDS
                    or key
                    in (
                        "row_vote_sparsity",
                        "physical_tile_sparsity",
                        "valid_element_sparsity",
                        "physical_to_row_sparsity_ratio",
                    )
                },
            }
        )
    _write_csv(output_dir / "ruler_accuracy_8k.csv", summary_rows)
    _write_csv(
        output_dir / "ruler_accuracy_8k_per_task.csv",
        [row for row in summary_rows if row["scope"] == "task"],
    )
    _write_csv(
        output_dir / "ruler_accuracy_8k_per_example.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "dense_completion_tokens",
                    "sparse_completion_tokens",
                )
            }
            for row in rows
        ],
    )
    violations: list[str] = []
    for row in rows:
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append(f"count identity {row['sample_id']}")
        if row["query_lengths"] != [16]:
            violations.append(f"query length {row['sample_id']}")
        for metric in (
            "dense_accuracy",
            "sparse_accuracy",
            "row_vote_sparsity",
            "physical_tile_sparsity",
            "valid_element_sparsity",
        ):
            if not math.isfinite(float(row[metric])):
                violations.append(f"non-finite {metric} {row['sample_id']}")
    expected_task_count = args.accuracy_samples // len(PAPER_TASKS)
    for row in summary_rows:
        if row["scope"] == "task" and int(row["samples"]) != expected_task_count:
            violations.append(f"unbalanced accuracy summary {row['task']}")
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append(f"summary count identity {row['task']}")
        for metric in (
            "dense_accuracy",
            "sparse_accuracy",
            "accuracy_change",
            "absolute_accuracy_change",
            "row_vote_sparsity",
            "physical_tile_sparsity",
            "valid_element_sparsity",
        ):
            if not math.isfinite(float(row[metric])):
                violations.append(f"non-finite summary {metric} {row['task']}")
    checks = {
        "passed": not violations,
        "violations": violations,
        "samples": len(rows),
        "paired_dense_sparse": True,
        "official_scorer_sha256": scorer_hash,
        "official_scorer_commit": RULER_COMMIT,
        "dense_agreement_is_diagnostic_only": True,
        "observed_sparse_query_lengths": sorted(
            {
                length
                for row in rows
                for length in row["query_lengths"]
            }
        ),
        "accuracy_aggregation": (
            "pinned official scorer invoked once over each complete reporting "
            "subset, matching RULER run_evaluation_per_task"
        ),
        "sparsity_aggregation": "ratios of summed integer counts",
        "balanced_task_samples": expected_task_count,
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "sample_manifest_validation": manifest_checks,
    }
    (output_dir / "ruler_accuracy_correctness.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if violations:
        raise AssertionError(json.dumps(checks, indent=2))
    runtime_path = output_dir / "ruler_accuracy_runtime.json"
    prior_runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {}
    )
    runtime_summary = {
        **prior_runtime,
        "last_invocation_wall_seconds": time.monotonic() - started,
        "new_pairs_executed_last_invocation": len(rows) - completed_at_start,
        "checkpoint_pairs": len(rows),
    }
    if completed_at_start == 0 and len(rows) == args.accuracy_samples:
        runtime_summary["full_paired_inference_wall_seconds"] = (
            runtime_summary["last_invocation_wall_seconds"]
        )
    (output_dir / "ruler_accuracy_runtime.json").write_text(
        json.dumps(
            runtime_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _length_distribution(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for target in sorted({int(row["target_length"]) for row in records}):
        subset = [
            int(row["actual_prompt_length"])
            for row in records
            if int(row["target_length"]) == target
        ]
        rows.append(
            {
                "target": target,
                "samples": len(subset),
                "minimum": min(subset),
                "mean": sum(subset) / len(subset),
                "maximum": max(subset),
            }
        )
    return rows


def _write_report(args: argparse.Namespace, output_dir: Path) -> None:
    context = _csv_rows(output_dir / "ruler_context_sweep.csv")
    block = _csv_rows(output_dir / "ruler_block_size_sweep.csv")
    accuracy = _csv_rows(output_dir / "ruler_accuracy_8k.csv")
    samples = _read_jsonl(
        output_dir / "ruler_cache/ruler_context_samples.jsonl"
    )
    provenance = json.loads(
        (
            output_dir / "ruler_cache/ruler_sample_manifest.json"
        ).read_text(encoding="utf-8")
    )["ruler"]
    lengths = _length_distribution(samples)
    context_lookup = {
        (int(row["target_context_length"]), float(row["lambda"])): row
        for row in context
    }
    block_lookup = {
        (int(row["block_size"]), float(row["lambda"])): row
        for row in block
    }
    previous_path = (
        REPO_ROOT
        / "results/blasst_controlled_sweeps/context_length_sweep.csv"
    )
    comparison_lines: list[str] = []
    if previous_path.exists():
        previous = _csv_rows(previous_path)
        previous_lookup = {
            (int(row["context_length"]), float(row["lambda"])): row
            for row in previous
        }
        for value in (0.001, 0.003, 0.5):
            deltas = []
            for target in CONTEXTS:
                current = context_lookup[(target, value)]
                old = previous_lookup[(target, value)]
                deltas.append(
                    float(current["physical_tile_sparsity"])
                    - float(old["physical_tile_sparsity"])
                )
            comparison_lines.append(
                f"λ={value:g}: mean physical-sparsity change "
                f"{sum(deltas)/len(deltas):+.1%}, range "
                f"[{min(deltas):+.1%}, {max(deltas):+.1%}]"
            )
    else:
        comparison_lines.append("Previous artificial-context CSV unavailable.")

    overall = next(row for row in accuracy if row["scope"] == "overall")
    lines = [
        "# RULER-based 2D-BLASST evaluation",
        "",
        "This replaces unrelated Alpaca concatenation with prompts generated by "
        "NVIDIA's official RULER implementation. It remains dense-QK observation "
        "and reference masking; no runtime speedup is measured or claimed.",
        "",
        "## Reproducibility",
        "",
        f"- Official repository: `{provenance['repository']}`.",
        f"- Pinned commit: `{provenance['commit']}`.",
        "- Tasks: NIAH MULTI (`niah_multikey_1`, `niah_multivalue`, "
        "`niah_multiquery`), variable tracking (`vt`), and frequent-word "
        "extraction (`fwe`). At 512 tokens only, official shorter NIAH "
        "multi-key configuration plus FWE are used because official VT and "
        "four-needle configurations cannot fit within the declared tolerance.",
        "- Samples are cached; block-size samples exactly reuse the 8K context "
        "samples, and the 100-sample accuracy set contains those same 32.",
        "- Sub-block splitting and dual block cache are disabled. Physical "
        "Q/KV tiles remain 128/64.",
        "",
        "## Actual prompt lengths",
        "",
        "| Target | Samples | Min | Mean | Max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in lengths:
        lines.append(
            f"| {row['target']} | {row['samples']} | {row['minimum']} | "
            f"{row['mean']:.1f} | {row['maximum']} |"
        )
    lines.extend(
        [
            "",
            "Every sample is within "
            f"`max({args.length_tolerance_min_tokens}, "
            f"ceil(target×{args.length_tolerance_fraction:g}))` model tokens.",
            "",
            "## Context-length effect",
            "",
        ]
    )
    for value in (0.0001, 0.001, 0.003, 0.01, 0.5):
        low = context_lookup[(512, value)]
        high = context_lookup[(16384, value)]
        lines.append(
            f"- λ={value:g}: row "
            f"{float(low['row_vote_sparsity']):.1%}→"
            f"{float(high['row_vote_sparsity']):.1%}; physical "
            f"{float(low['physical_tile_sparsity']):.1%}→"
            f"{float(high['physical_tile_sparsity']):.1%}."
        )
    lines.extend(
        [
            "",
            "## Diffusion-block unanimity at approximately 8K",
            "",
        ]
    )
    for value in (0.0001, 0.001, 0.003, 0.01, 0.5):
        one = block_lookup[(1, value)]
        sixty_four = block_lookup[(64, value)]
        lines.append(
            f"- λ={value:g}: row "
            f"{float(one['row_vote_sparsity']):.1%}→"
            f"{float(sixty_four['row_vote_sparsity']):.1%}; physical "
            f"{float(one['physical_tile_sparsity']):.1%}→"
            f"{float(sixty_four['physical_tile_sparsity']):.1%}."
        )
    lines.extend(
        [
            "",
            "## Paired 8K RULER accuracy",
            "",
            f"Setting: λ={args.accuracy_lambda:g}, diffusion block/effective "
            f"query length={args.block_size}.",
            "",
            "| Scope | Samples | Dense | Sparse | Δ (sparse−dense) | Absolute change | Dense-answer agreement | Row | Physical |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in accuracy:
        lines.append(
            f"| {row['task']} | {int(row['samples'])} | "
            f"{float(row['dense_accuracy']):.2%} | "
            f"{float(row['sparse_accuracy']):.2%} | "
            f"{float(row['accuracy_change']) * 100:+.2f} pp | "
            f"{float(row['absolute_accuracy_change']) * 100:.2f} pp | "
            f"{float(row['final_answer_agreement_with_dense']):.1%} | "
            f"{float(row['row_vote_sparsity']):.1%} | "
            f"{float(row['physical_tile_sparsity']):.1%} |"
        )
    lines.extend(
        [
            "",
            "Accuracy prompt lengths (model tokens): "
            f"min {int(overall['actual_prompt_length_min'])}, mean "
            f"{float(overall['actual_prompt_length_mean']):.1f}, max "
            f"{int(overall['actual_prompt_length_max'])}.",
            "",
            "RULER accuracy is computed by the pinned official task scorer. "
            "Dense-answer agreement compares the official reference-hit pattern "
            "and is diagnostic only.",
            "",
            "## Comparison with the previous artificial-context sweep",
            "",
            *[f"- {text}" for text in comparison_lines],
            "",
            "The numerical difference combines the change to meaningful RULER "
            "prompts with the changed task mixture; it must not be attributed to "
            "prompt construction alone. The new results are the appropriate basis "
            "for long-context claims.",
            "",
            "## Artifacts",
            "",
            "- Full aggregate, per-task, and per-example CSVs are adjacent to this report.",
            "- Cached prompts, seeds, task labels, target lengths, actual lengths, "
            "source hashes, and exact generator commands are under `ruler_cache/`.",
            "- PNG/PDF plots cover row-vote and physical sparsity for both sweeps.",
            "",
            f"Overall paired result: dense "
            f"{float(overall['dense_accuracy']):.2%}, sparse "
            f"{float(overall['sparse_accuracy']):.2%}, signed change "
            f"{float(overall['accuracy_change']) * 100:+.2f} pp "
            f"(absolute magnitude "
            f"{float(overall['absolute_accuracy_change']) * 100:.2f} pp).",
            "",
        ]
    )
    (output_dir / "ruler_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
    }


def _write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    ruler: Mapping[str, Any],
) -> None:
    config_path = output_dir / "ruler_run_config.json"
    prior = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    environment = _environment()
    phase_environments = dict(prior.get("phase_environments", {}))
    phase_environments[str(args.phase)] = environment
    model_environment = next(
        (
            phase_environments[phase]
            for phase in ("accuracy", "sweeps", "all")
            if phase in phase_environments
            and phase_environments[phase]["device_name"] != "cpu"
        ),
        environment,
    )
    config = {
        **prior,
        **vars(args),
        **model_environment,
        "contexts": list(CONTEXTS),
        "block_sizes": list(BLOCK_SIZES),
        "lambdas": list(LAMBDAS),
        "mask_ratios": list(MASK_RATIOS),
        "paper_tasks": list(PAPER_TASKS),
        "short_512_tasks": list(SHORT_512_TASKS),
        "ruler": dict(ruler),
        "sample_state": (
            "deterministic dense model continuation, five nested diffusion "
            "mask states; same prompt/state scores reused across lambda"
        ),
        "actual_accuracy_scoring": "pinned official NVIDIA RULER metric_fn",
        "sub_block_optimization": False,
        "dual_block_cache": False,
        "ordinary_kv_cache": True,
        "physical_attention_tiles": {
            "query": args.q_tile_size,
            "key_value": args.kv_tile_size,
        },
        "unrelated_alpaca_prompt_extension": False,
        "last_phase_environment": environment,
        "phase_environments": phase_environments,
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    contexts = _values(args.contexts, int)
    block_sizes = _values(args.block_sizes, int)
    lambdas = _values(args.lambdas, float)
    ratios = _values(args.mask_ratios, float)
    if contexts != list(CONTEXTS):
        raise ValueError(f"context grid must be {CONTEXTS}")
    if block_sizes != list(BLOCK_SIZES):
        raise ValueError(f"block grid must be {BLOCK_SIZES}")
    if lambdas != list(LAMBDAS):
        raise ValueError(f"lambda grid must be {LAMBDAS}")
    if ratios != list(MASK_RATIOS):
        raise ValueError(f"mask-ratio grid must be {MASK_RATIOS}")
    if args.context_samples != 32 or args.accuracy_samples != 100:
        raise ValueError("controlled evaluation requires 32/100 samples")
    if args.accuracy_context != 8192:
        raise ValueError("accuracy and block sweeps require 8192 target tokens")
    if args.block_size != 16:
        raise ValueError("context and accuracy runs require block_size=16")
    if args.q_tile_size != 128 or args.kv_tile_size != 64:
        raise ValueError("physical Q/KV tiles must remain 128/64")
    if args.state_tokens != 64:
        raise ValueError("shared block-sweep state must contain exactly 64 tokens")
    if not 0.0 < args.accuracy_lambda < 1.0:
        raise ValueError("accuracy lambda must be in (0,1)")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ruler = _verify_ruler_checkout(Path(args.ruler_root))
    _write_run_config(args, output_dir, ruler)
    if args.phase in ("all", "prepare"):
        _prepare_samples(args, output_dir, ruler)
    if args.phase in ("all", "sweeps"):
        _run_sweeps(args, output_dir, lambdas, block_sizes, ratios)
    if args.phase in ("all", "accuracy"):
        _run_accuracy(args, output_dir)
    if args.phase in ("all", "report"):
        _write_report(args, output_dir)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "output_dir": str(output_dir),
                "ruler_commit": ruler["commit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
