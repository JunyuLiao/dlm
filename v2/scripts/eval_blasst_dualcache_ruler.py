#!/usr/bin/env python3
"""Controlled RULER study of sub-block splitting and DualCache.

The experiment is deliberately separate from the no-sub-block RULER sweep.
It reuses that pipeline's exact cached 100-example manifest and pinned NVIDIA
RULER scorer, fixes lambda 0.003, and checkpoints each sample/configuration
independently. The DualCache curve fixes outer block size 64; its comparison
curve uses standalone outer/query block sizes 4, 8, 16, 32, and 64 without
DualCache. 2D-BLASST remains a dense-QK reference mask; this script does not
measure or claim kernel speedup.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
sys.path.insert(0, str(V2_ROOT))

from scripts.blasst_common import (  # noqa: E402
    generate_one,
    load_model,
    normalized_token_difference,
    set_seed,
)
from scripts.eval_blasst_ruler import (  # noqa: E402
    PAPER_TASKS,
    RULER_COMMIT,
    _load_official_scorers,
    _official_postprocess,
    _read_jsonl,
    _reference_hit_pattern,
    _score_one,
    _score_subset,
    _sha256_file,
    _task_generation_budget,
    _verify_ruler_checkout,
)
from sparse_attention import (  # noqa: E402
    Blasst2DConfig,
    Blasst2DStats,
    install_blasst_2d,
)


OUTER_BLOCK_SIZE = 64
SUB_BLOCK_SIZES = (4, 8, 16, 32)
BLASST_LAMBDA = 0.003
Q_TILE_SIZE = 128
KV_TILE_SIZE = 64
SAMPLE_COUNT = 100
TARGET_CONTEXT = 8192
COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
    "skippable_row_votes",
    "valid_row_votes",
    "skipped_valid_elements",
    "valid_elements",
)


def _standalone_configurations() -> list[dict[str, Any]]:
    return [
        {
            "configuration": f"sparse_no_dualcache_b{size}",
            "outer_block_size": size,
            "sub_block_size": size,
            "dual_cache_enabled": False,
        }
        for size in (*SUB_BLOCK_SIZES, OUTER_BLOCK_SIZE)
    ]


def _dual_configurations() -> list[dict[str, Any]]:
    return [
        {
            "configuration": f"dualcache_s{size}",
            "outer_block_size": OUTER_BLOCK_SIZE,
            "sub_block_size": size,
            "dual_cache_enabled": True,
        }
        for size in SUB_BLOCK_SIZES
    ]


def _sparse_configurations() -> list[dict[str, Any]]:
    return [*_standalone_configurations(), *_dual_configurations()]


def _all_configurations() -> list[dict[str, Any]]:
    return [
        {
            "configuration": "dense",
            "outer_block_size": OUTER_BLOCK_SIZE,
            "sub_block_size": OUTER_BLOCK_SIZE,
            "dual_cache_enabled": False,
        },
        *_sparse_configurations(),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "prepare", "run", "aggregate", "report"),
        default="all",
    )
    parser.add_argument("--model-path", default="/tmp/fast_dllm_v2_7b")
    parser.add_argument("--ruler-root", default="/tmp/nvidia-ruler")
    parser.add_argument(
        "--source-results-dir",
        default=str(REPO_ROOT / "results/blasst_ruler"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/blasst_dualcache_ruler"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--denoising-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument(
        "--configurations",
        default=",".join(row["configuration"] for row in _all_configurations()),
        help="Comma-separated subset used for resumable/pilot execution.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Allow fewer than 100 samples; production export rejects this.",
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ratios(counts: Mapping[str, Any]) -> dict[str, Any]:
    row = {field: int(counts[field]) for field in COUNT_FIELDS}
    eligible = row["eligible_tiles"]
    votes = row["valid_row_votes"]
    elements = row["valid_elements"]
    row["row_vote_sparsity"] = (
        row["skippable_row_votes"] / votes if votes else 0.0
    )
    row["physical_tile_sparsity"] = (
        row["skipped_tiles"] / eligible if eligible else 0.0
    )
    row["valid_element_sparsity"] = (
        row["skipped_valid_elements"] / elements if elements else 0.0
    )
    return row


def _sum_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {field: 0 for field in COUNT_FIELDS}
    for row in rows:
        for field in COUNT_FIELDS:
            totals[field] += int(row[field])
    return _ratios(totals)


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


def _selected_configuration_names(args: argparse.Namespace) -> list[str]:
    requested = [
        value.strip()
        for value in str(args.configurations).split(",")
        if value.strip()
    ]
    known = [row["configuration"] for row in _all_configurations()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("configuration selection must be unique and non-empty")
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"unknown configurations: {unknown}")
    return requested


def _source_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    source = Path(args.source_results_dir).resolve()
    return (
        source / "ruler_cache/ruler_accuracy_samples_8k.jsonl",
        source / "ruler_cache/ruler_sample_manifest.json",
    )


def _validate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    if len(samples) != SAMPLE_COUNT:
        violations.append(f"sample count {len(samples)}")
    ids = [str(row["sample_id"]) for row in samples]
    if len(ids) != len(set(ids)):
        violations.append("duplicate sample IDs")
    counts = Counter(str(row["task"]) for row in samples)
    expected = Counter({task: SAMPLE_COUNT // len(PAPER_TASKS) for task in PAPER_TASKS})
    if counts != expected:
        violations.append(f"task mixture {dict(counts)}")
    for row in samples:
        prompt = str(row["prompt"])
        if _sha256_bytes(prompt.encode("utf-8")) != row["prompt_sha256"]:
            violations.append(f"prompt hash {row['sample_id']}")
        if int(row["target_length"]) != TARGET_CONTEXT:
            violations.append(f"target length {row['sample_id']}")
        if not bool(row["within_length_tolerance"]):
            violations.append(f"length tolerance {row['sample_id']}")
        if not isinstance(row.get("inference_seed"), int):
            violations.append(f"inference seed {row['sample_id']}")
    result = {
        "passed": not violations,
        "violations": violations,
        "samples": len(samples),
        "task_counts": dict(sorted(counts.items())),
        "actual_prompt_length_min": min(
            int(row["actual_prompt_length"]) for row in samples
        ),
        "actual_prompt_length_mean": sum(
            int(row["actual_prompt_length"]) for row in samples
        )
        / len(samples),
        "actual_prompt_length_max": max(
            int(row["actual_prompt_length"]) for row in samples
        ),
    }
    if violations:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _prepare(args: argparse.Namespace, output_dir: Path) -> None:
    source_samples, source_manifest = _source_paths(args)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected = str(manifest["files"]["accuracy"]["sha256"])
    if _sha256_file(source_samples) != expected:
        raise AssertionError("source RULER accuracy cache hash mismatch")
    samples = _read_jsonl(source_samples)
    checks = _validate_samples(samples)
    destination = output_dir / "ruler_samples_8k.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_samples.read_bytes())
    if _sha256_file(destination) != expected:
        raise AssertionError("copied 100-sample manifest is not byte-identical")
    _write_json(
        output_dir / "dualcache_sample_manifest.json",
        {
            "source_path": str(source_samples),
            "source_manifest": str(source_manifest),
            "source_sha256": expected,
            "copied_path": str(destination),
            "copied_sha256": _sha256_file(destination),
            "byte_identical_to_existing_ruler_accuracy_manifest": True,
            "ruler": manifest["ruler"],
            **checks,
        },
    )


def _dense_probe(model: Any, tokenizer: Any) -> torch.Tensor:
    ids = tokenizer(
        "DualCache dense-disabled regression probe.",
        return_tensors="pt",
    )["input_ids"].to(model.device)
    mask_id = int(getattr(model.config, "mask_token_id", 151665))
    if ids.shape[1] < OUTER_BLOCK_SIZE:
        ids = torch.cat(
            (
                ids,
                torch.full(
                    (1, OUTER_BLOCK_SIZE - ids.shape[1]),
                    mask_id,
                    dtype=torch.long,
                    device=model.device,
                ),
            ),
            dim=1,
        )
    with torch.no_grad():
        return model.model(
            input_ids=ids[:, :OUTER_BLOCK_SIZE],
            use_cache=False,
            use_block_cache=False,
            block_size=OUTER_BLOCK_SIZE,
        ).last_hidden_state.detach()


def _shard_path(
    output_dir: Path,
    configuration: str,
    sample_id: str,
) -> Path:
    return output_dir / "raw" / configuration / f"{sample_id}.json.gz"


def _write_shard(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def _read_shard(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_sparse_trace(
    *,
    configuration: Mapping[str, Any],
    prediction: Mapping[str, Any],
    per_forward: list[dict[str, Any]],
    per_layer_head: list[dict[str, Any]],
) -> dict[str, Any]:
    violations: list[str] = []
    outer_block_size = int(configuration["outer_block_size"])
    if not per_forward:
        violations.append("no denoising forwards recorded")
    forward_ids = [int(row["forward_pass_id"]) for row in per_forward]
    if forward_ids != list(range(len(per_forward))):
        violations.append("forward-pass IDs are not contiguous")
    for row in per_forward + per_layer_head:
        if str(row["configuration"]) != configuration["configuration"]:
            violations.append("configuration metadata mismatch")
        if int(row["outer_block_size"]) != outer_block_size:
            violations.append("outer-block metadata mismatch")
        if int(row["sub_block_size"]) != int(
            configuration["sub_block_size"]
        ):
            violations.append("sub-block metadata mismatch")
        if bool(row["dual_cache_enabled"]) != bool(
            configuration["dual_cache_enabled"]
        ):
            violations.append("DualCache metadata mismatch")
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append("eligible count identity")
        for field in COUNT_FIELDS:
            if int(row[field]) < 0:
                violations.append(f"negative {field}")
        for field in (
            "row_vote_sparsity",
            "physical_tile_sparsity",
            "valid_element_sparsity",
        ):
            if not math.isfinite(float(row[field])):
                violations.append(f"non-finite {field}")
    expected_head_rows = len(per_forward) * 28 * 28
    if len(per_layer_head) != expected_head_rows:
        violations.append(
            f"layer/head rows {len(per_layer_head)} != {expected_head_rows}"
        )
    observed_q = sorted({int(row["query_length"]) for row in per_forward})
    if (
        not configuration["dual_cache_enabled"]
        and observed_q != [outer_block_size]
    ):
        violations.append(f"baseline query lengths {observed_q}")
    for row in per_forward:
        kind = str(row["forward_kind"])
        query = int(row["query_length"])
        if kind == "dual_refresh" and query != outer_block_size:
            violations.append(f"refresh Q={query}")
        if kind == "dual_subblock_update" and query != int(
            configuration["sub_block_size"]
        ):
            violations.append(f"sub-block update Q={query}")
        if kind == "ordinary_denoising" and query != outer_block_size:
            violations.append(f"ordinary denoising Q={query}")
    if prediction["query_lengths"] != observed_q:
        violations.append("prediction/query trace mismatch")
    result = {
        "passed": not violations,
        "violations": sorted(set(violations)),
        "forward_passes": len(per_forward),
        "layer_head_attention_calls": len(per_layer_head),
        "observed_query_lengths": observed_q,
        "forward_kind_counts": dict(
            sorted(Counter(str(row["forward_kind"]) for row in per_forward).items())
        ),
    }
    if violations:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _run(args: argparse.Namespace, output_dir: Path) -> None:
    started = time.monotonic()
    samples = _read_jsonl(output_dir / "ruler_samples_8k.jsonl")
    _validate_samples(samples)
    selected_names = _selected_configuration_names(args)
    selected_samples = samples[: int(args.num_samples)]
    configuration_by_name = {
        row["configuration"]: row for row in _all_configurations()
    }
    model, tokenizer = load_model(
        args.model_path, args.device, args.precision
    )
    before = _dense_probe(model, tokenizer)
    sparse_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=BLASST_LAMBDA,
        q_tile_size=Q_TILE_SIZE,
        kv_tile_size=KV_TILE_SIZE,
        collect_blasst_stats=True,
        collect_blasst_layer_stats=True,
        collect_blasst_head_stats=True,
    )
    runtime = install_blasst_2d(
        model,
        replace(sparse_config, enable_blasst_2d=False),
        mask_token_id=int(getattr(model.config, "mask_token_id", 151665)),
        pad_token_id=tokenizer.pad_token_id,
    )
    after = _dense_probe(model, tokenizer)
    dense_regression = {
        "exact_match": bool(torch.equal(before, after)),
        "all_finite": bool(
            torch.isfinite(before).all() and torch.isfinite(after).all()
        ),
        "max_abs_error": float((before - after).abs().max()),
    }
    if not dense_regression["exact_match"] or not dense_regression["all_finite"]:
        raise AssertionError(f"dense regression failed: {dense_regression}")
    _write_json(output_dir / "dualcache_dense_regression.json", dense_regression)

    ruler_root = Path(args.ruler_root)
    scorers, scorer_hash = _load_official_scorers(ruler_root)
    completed = 0
    for sample_index, sample in enumerate(selected_samples, start=1):
        for configuration_name in selected_names:
            configuration = configuration_by_name[configuration_name]
            sample_id = str(sample["sample_id"])
            shard = _shard_path(output_dir, configuration_name, sample_id)
            if shard.exists():
                payload = _read_shard(shard)
                if (
                    payload["prediction"]["prompt_sha256"]
                    != sample["prompt_sha256"]
                    or payload["prediction"]["configuration"]
                    != configuration_name
                ):
                    raise AssertionError(f"invalid completed shard {shard}")
                completed += 1
                continue

            is_dense = configuration_name == "dense"
            runtime.config = (
                replace(sparse_config, enable_blasst_2d=False)
                if is_dense
                else sparse_config
            )
            runtime.stats = Blasst2DStats(
                record_layers=True,
                record_heads=True,
            )
            runtime.forward_call_index = 0
            runtime.active_forward_call_index = 0
            runtime.dual_cache_only = bool(
                configuration["dual_cache_enabled"]
            )
            runtime.ordinary_cache_queries_only = (
                not is_dense
                and not bool(configuration["dual_cache_enabled"])
            )
            outer_block_size = int(configuration["outer_block_size"])
            runtime.metadata_context = {
                "experiment": "dualcache_ruler",
                "configuration": configuration_name,
                "benchmark": str(sample["task"]),
                "example_id": sample_id,
                "outer_block_size": outer_block_size,
                "sub_block_size": int(configuration["sub_block_size"]),
                "dual_cache_enabled": bool(
                    configuration["dual_cache_enabled"]
                ),
                "inference_seed": int(sample["inference_seed"]),
            }
            set_seed(int(sample["inference_seed"]))
            generation_budget = _task_generation_budget(
                str(sample["task"]),
                ruler_root,
                OUTER_BLOCK_SIZE,
            )
            generated = generate_one(
                model,
                tokenizer,
                str(sample["prompt"]),
                block_size=outer_block_size,
                small_block_size=int(configuration["sub_block_size"]),
                use_block_cache=bool(configuration["dual_cache_enabled"]),
                max_new_tokens=generation_budget,
                threshold=args.denoising_threshold,
            )
            references = [str(value) for value in sample["outputs"]]
            prediction_text = str(generated["completion"])
            scorer = scorers[str(sample["task_base"])]
            hit_pattern = _reference_hit_pattern(
                prediction_text, references
            )
            per_forward = (
                runtime.stats._rows(runtime.stats.per_step)
                if not is_dense
                else []
            )
            per_layer_head = (
                runtime.stats._rows(runtime.stats.per_head)
                if not is_dense
                else []
            )
            summary = (
                runtime.stats.summary()
                if not is_dense
                else {field: 0 for field in COUNT_FIELDS}
            )
            query_lengths = sorted(
                {int(row["query_length"]) for row in per_forward}
            )
            common_generation_spec = {
                "sample_id": sample_id,
                "prompt_sha256": str(sample["prompt_sha256"]),
                "inference_seed": int(sample["inference_seed"]),
                "outer_block_size": outer_block_size,
                "generation_budget": generation_budget,
                "denoising_threshold": float(args.denoising_threshold),
                "temperature": 0.0,
                "mask_token_id": int(
                    getattr(model.config, "mask_token_id", 151665)
                ),
                "stop_token_id": int(
                    getattr(model.config, "eos_token_id", 151645)
                ),
            }
            first_masked_tokens = (
                outer_block_size
                - int(sample["actual_prompt_length"]) % outer_block_size
            )
            initial_state_spec = {
                **common_generation_spec,
                "first_block_masked_tokens": first_masked_tokens,
                "initial_block_state": (
                    "prompt followed by first_block_masked_tokens copies of "
                    "mask_token_id"
                ),
            }
            prediction = {
                "sample_id": sample_id,
                "task": str(sample["task"]),
                "task_base": str(sample["task_base"]),
                "configuration": configuration_name,
                "outer_block_size": outer_block_size,
                "sub_block_size": int(configuration["sub_block_size"]),
                "dual_cache_enabled": bool(
                    configuration["dual_cache_enabled"]
                ),
                "lambda": None if is_dense else BLASST_LAMBDA,
                "q_tile_size": Q_TILE_SIZE,
                "kv_tile_size": KV_TILE_SIZE,
                "inference_seed": int(sample["inference_seed"]),
                "target_length": int(sample["target_length"]),
                "actual_prompt_length": int(
                    sample["actual_prompt_length"]
                ),
                "prompt_sha256": str(sample["prompt_sha256"]),
                "outputs": references,
                "generation_budget": generation_budget,
                "common_generation_spec_sha256": _json_hash(
                    common_generation_spec
                ),
                "initial_denoising_state_sha256": _json_hash(
                    initial_state_spec
                ),
                "first_block_masked_tokens": first_masked_tokens,
                "prediction": prediction_text,
                "completion_tokens": generated["completion_tokens"],
                "per_example_ruler_accuracy": _score_one(
                    scorer, prediction_text, references
                ),
                "reference_hit_pattern": hit_pattern,
                "query_lengths": query_lengths,
                "forward_passes": len(per_forward),
                **summary,
            }
            trace_checks = (
                {
                    "passed": True,
                    "violations": [],
                    "forward_passes": 0,
                    "layer_head_attention_calls": 0,
                    "observed_query_lengths": [],
                    "forward_kind_counts": {},
                }
                if is_dense
                else _validate_sparse_trace(
                    configuration=configuration,
                    prediction=prediction,
                    per_forward=per_forward,
                    per_layer_head=per_layer_head,
                )
            )
            _write_shard(
                shard,
                {
                    "schema_version": 1,
                    "official_scorer_sha256": scorer_hash,
                    "prediction": prediction,
                    "trace_checks": trace_checks,
                    "per_forward": per_forward,
                    "per_layer_head": per_layer_head,
                },
            )
            completed += 1
            print(
                f"completed {completed}/"
                f"{len(selected_samples) * len(selected_names)}: "
                f"{configuration_name}/{sample_id}; "
                f"Q={query_lengths or 'dense'}",
                flush=True,
            )

    runtime_path = output_dir / "dualcache_runtime.json"
    prior = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.exists()
        else {}
    )
    history = list(prior.get("run_invocations", []))
    history.append(
        {
            "configurations": selected_names,
            "samples": len(selected_samples),
            "wall_seconds": time.monotonic() - started,
            "environment": _environment(),
        }
    )
    _write_json(runtime_path, {"run_invocations": history})


def _expected_shards(
    output_dir: Path,
    samples: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    return [
        (
            configuration,
            sample,
            _shard_path(
                output_dir,
                str(configuration["configuration"]),
                str(sample["sample_id"]),
            ),
        )
        for configuration in _all_configurations()
        for sample in samples
    ]


def _stream_detailed_csv(
    path: Path,
    shards: list[tuple[dict[str, Any], dict[str, Any], Path]],
) -> int:
    rows_written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter | None = None
        for configuration, _, shard in shards:
            if configuration["configuration"] == "dense":
                continue
            rows = _read_shard(shard)["per_layer_head"]
            if rows and writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            if writer is not None:
                writer.writerows(rows)
                rows_written += len(rows)
    if rows_written == 0:
        raise AssertionError("no detailed layer/head rows exported")
    return rows_written


def _accuracy_summary(
    *,
    scorers: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    configuration: str,
    scope: str,
    task: str,
) -> dict[str, Any]:
    subset = [
        row
        for row in predictions
        if row["configuration"] == configuration
        and (scope == "overall" or row["task"] == task)
    ]
    score = _score_subset(scorers, subset, "prediction")
    lengths = [int(row["actual_prompt_length"]) for row in subset]
    return {
        "configuration": configuration,
        "scope": scope,
        "task": task,
        "samples": len(subset),
        "ruler_accuracy": score,
        "actual_prompt_length_min": min(lengths),
        "actual_prompt_length_mean": sum(lengths) / len(lengths),
        "actual_prompt_length_max": max(lengths),
    }


def _aggregate(args: argparse.Namespace, output_dir: Path) -> None:
    if args.pilot or int(args.num_samples) != SAMPLE_COUNT:
        raise ValueError("production aggregation requires all 100 samples")
    samples = _read_jsonl(output_dir / "ruler_samples_8k.jsonl")
    manifest_checks = _validate_samples(samples)
    shards = _expected_shards(output_dir, samples)
    missing = [str(path) for _, _, path in shards if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} experiment shards; first={missing[0]}"
        )
    predictions: list[dict[str, Any]] = []
    per_forward: list[dict[str, Any]] = []
    trace_checks: list[dict[str, Any]] = []
    for configuration, sample, shard in shards:
        payload = _read_shard(shard)
        prediction = payload["prediction"]
        if prediction["prompt_sha256"] != sample["prompt_sha256"]:
            raise AssertionError(f"shard prompt mismatch: {shard}")
        predictions.append(prediction)
        per_forward.extend(payload["per_forward"])
        trace_checks.append(payload["trace_checks"])

    _write_csv(
        output_dir / "dualcache_per_forward_pass.csv",
        per_forward,
    )
    detailed_rows = _stream_detailed_csv(
        output_dir / "dualcache_per_forward_layer_head.csv.gz",
        shards,
    )

    ruler_root = Path(args.ruler_root)
    scorers, scorer_hash = _load_official_scorers(ruler_root)
    accuracy_rows: list[dict[str, Any]] = []
    for configuration in [
        row["configuration"] for row in _all_configurations()
    ]:
        accuracy_rows.append(
            _accuracy_summary(
                scorers=scorers,
                predictions=predictions,
                configuration=configuration,
                scope="overall",
                task="all",
            )
        )
        for task in PAPER_TASKS:
            accuracy_rows.append(
                _accuracy_summary(
                    scorers=scorers,
                    predictions=predictions,
                    configuration=configuration,
                    scope="task",
                    task=task,
                )
            )
    overall_accuracy = {
        row["configuration"]: float(row["ruler_accuracy"])
        for row in accuracy_rows
        if row["scope"] == "overall"
    }
    dense_accuracy = overall_accuracy["dense"]
    baseline_accuracy = overall_accuracy["sparse_no_dualcache_b64"]
    for row in accuracy_rows:
        accuracy = float(row["ruler_accuracy"])
        dense_match = next(
            candidate
            for candidate in accuracy_rows
            if candidate["configuration"] == "dense"
            and candidate["scope"] == row["scope"]
            and candidate["task"] == row["task"]
        )
        baseline_match = next(
            candidate
            for candidate in accuracy_rows
            if candidate["configuration"] == "sparse_no_dualcache_b64"
            and candidate["scope"] == row["scope"]
            and candidate["task"] == row["task"]
        )
        row["accuracy_change_from_dense"] = (
            accuracy - float(dense_match["ruler_accuracy"])
        )
        row["absolute_accuracy_change_from_dense"] = abs(
            row["accuracy_change_from_dense"]
        )
        row["accuracy_change_from_sparse_baseline"] = (
            accuracy - float(baseline_match["ruler_accuracy"])
        )
    _write_csv(
        output_dir / "dualcache_accuracy_per_task.csv",
        accuracy_rows,
    )

    dense_by_sample = {
        row["sample_id"]: row
        for row in predictions
        if row["configuration"] == "dense"
    }
    baseline_by_sample = {
        row["sample_id"]: row
        for row in predictions
        if row["configuration"] == "sparse_no_dualcache_b64"
    }
    per_example_rows: list[dict[str, Any]] = []
    for row in predictions:
        dense = dense_by_sample[row["sample_id"]]
        baseline = baseline_by_sample[row["sample_id"]]
        dense_hits = tuple(dense["reference_hit_pattern"])
        current_hits = tuple(row["reference_hit_pattern"])
        per_example_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in ("completion_tokens", "reference_hit_pattern")
            }
            | {
                "accuracy_change_from_dense": (
                    float(row["per_example_ruler_accuracy"])
                    - float(dense["per_example_ruler_accuracy"])
                ),
                "accuracy_change_from_sparse_baseline": (
                    float(row["per_example_ruler_accuracy"])
                    - float(baseline["per_example_ruler_accuracy"])
                ),
                "dense_answer_agreement": int(current_hits == dense_hits),
                "sequence_exact_match_with_dense": int(
                    row["completion_tokens"] == dense["completion_tokens"]
                ),
                "normalized_token_difference_from_dense": (
                    normalized_token_difference(
                        row["completion_tokens"],
                        dense["completion_tokens"],
                    )
                ),
            }
        )
    _write_csv(
        output_dir / "dualcache_accuracy_per_example.csv",
        per_example_rows,
    )

    per_q_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    configuration_lookup = {
        row["configuration"]: row for row in _all_configurations()
    }
    for configuration_name in [
        row["configuration"] for row in _sparse_configurations()
    ]:
        rows = [
            row
            for row in per_forward
            if row["configuration"] == configuration_name
        ]
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["query_length"])].append(row)
        total_forwards = len(rows)
        weighted = {
            "row_vote_sparsity": 0.0,
            "physical_tile_sparsity": 0.0,
            "valid_element_sparsity": 0.0,
        }
        distribution_parts = []
        for query_length, group in sorted(grouped.items()):
            counts = _sum_counts(group)
            weight = len(group) / total_forwards
            q_row = {
                "configuration": configuration_name,
                "outer_block_size": configuration_lookup[
                    configuration_name
                ]["outer_block_size"],
                "sub_block_size": configuration_lookup[
                    configuration_name
                ]["sub_block_size"],
                "dual_cache_enabled": configuration_lookup[
                    configuration_name
                ]["dual_cache_enabled"],
                "query_length": query_length,
                "forward_passes": len(group),
                "forward_pass_weight": weight,
                "forward_kind_counts": json.dumps(
                    dict(
                        sorted(
                            Counter(
                                str(row["forward_kind"]) for row in group
                            ).items()
                        )
                    ),
                    sort_keys=True,
                ),
                **counts,
            }
            for metric in weighted:
                contribution = weight * float(counts[metric])
                q_row[f"weighted_{metric}_contribution"] = contribution
                weighted[metric] += contribution
            per_q_rows.append(q_row)
            distribution_parts.append(
                f"Q={query_length}: {weight:.6%} ({len(group)})"
            )
        global_counts = _sum_counts(rows)
        config_predictions = [
            row
            for row in per_example_rows
            if row["configuration"] == configuration_name
        ]
        summary_rows.append(
            {
                "configuration": configuration_name,
                "outer_block_size": configuration_lookup[
                    configuration_name
                ]["outer_block_size"],
                "sub_block_size": configuration_lookup[
                    configuration_name
                ]["sub_block_size"],
                "dual_cache_enabled": configuration_lookup[
                    configuration_name
                ]["dual_cache_enabled"],
                "observed_query_lengths": ",".join(
                    str(value) for value in sorted(grouped)
                ),
                "query_length_distribution": "; ".join(
                    distribution_parts
                ),
                "forward_passes": total_forwards,
                "weighted_row_vote_sparsity": weighted[
                    "row_vote_sparsity"
                ],
                "weighted_physical_tile_sparsity": weighted[
                    "physical_tile_sparsity"
                ],
                "weighted_valid_element_sparsity": weighted[
                    "valid_element_sparsity"
                ],
                "global_row_vote_sparsity": global_counts[
                    "row_vote_sparsity"
                ],
                "global_physical_tile_sparsity": global_counts[
                    "physical_tile_sparsity"
                ],
                "global_valid_element_sparsity": global_counts[
                    "valid_element_sparsity"
                ],
                **{
                    field: global_counts[field] for field in COUNT_FIELDS
                },
                "ruler_accuracy": overall_accuracy[configuration_name],
                "accuracy_change_from_dense": (
                    overall_accuracy[configuration_name] - dense_accuracy
                ),
                "absolute_accuracy_change_from_dense": abs(
                    overall_accuracy[configuration_name] - dense_accuracy
                ),
                "accuracy_change_from_sparse_baseline": (
                    overall_accuracy[configuration_name] - baseline_accuracy
                ),
                "dense_answer_agreement": sum(
                    int(row["dense_answer_agreement"])
                    for row in config_predictions
                )
                / len(config_predictions),
                "sequence_exact_match_with_dense": sum(
                    int(row["sequence_exact_match_with_dense"])
                    for row in config_predictions
                )
                / len(config_predictions),
                "actual_prompt_length_min": min(
                    int(row["actual_prompt_length"])
                    for row in config_predictions
                ),
                "actual_prompt_length_mean": sum(
                    int(row["actual_prompt_length"])
                    for row in config_predictions
                )
                / len(config_predictions),
                "actual_prompt_length_max": max(
                    int(row["actual_prompt_length"])
                    for row in config_predictions
                ),
            }
        )
    summary_by_configuration = {
        row["configuration"]: row for row in summary_rows
    }
    for row in summary_rows:
        if bool(row["dual_cache_enabled"]):
            corresponding_name = (
                f"sparse_no_dualcache_b{row['sub_block_size']}"
            )
        else:
            corresponding_name = str(row["configuration"])
        corresponding = summary_by_configuration[corresponding_name]
        row["corresponding_no_dualcache_configuration"] = corresponding_name
        row["accuracy_change_from_corresponding_no_dualcache"] = (
            float(row["ruler_accuracy"])
            - float(corresponding["ruler_accuracy"])
        )
        row["weighted_physical_change_from_corresponding_no_dualcache"] = (
            float(row["weighted_physical_tile_sparsity"])
            - float(corresponding["weighted_physical_tile_sparsity"])
        )
        row["global_physical_change_from_corresponding_no_dualcache"] = (
            float(row["global_physical_tile_sparsity"])
            - float(corresponding["global_physical_tile_sparsity"])
        )
    _write_csv(
        output_dir / "dualcache_per_query_length.csv",
        per_q_rows,
    )
    _write_csv(
        output_dir / "dualcache_configuration_summary.csv",
        summary_rows,
    )

    violations: list[str] = []
    for sample in samples:
        sample_rows = [
            row
            for row in predictions
            if row["sample_id"] == sample["sample_id"]
        ]
        if len(sample_rows) != len(_all_configurations()):
            violations.append(
                f"configuration count for {sample['sample_id']}"
            )
            continue
        for field in (
            "prompt_sha256",
            "inference_seed",
            "generation_budget",
            "actual_prompt_length",
        ):
            if len({_json_hash(row[field]) for row in sample_rows}) != 1:
                violations.append(
                    f"cross-configuration {field} {sample['sample_id']}"
                )
        outer_64_rows = [
            row
            for row in sample_rows
            if int(row["outer_block_size"]) == OUTER_BLOCK_SIZE
        ]
        for field in (
            "common_generation_spec_sha256",
            "initial_denoising_state_sha256",
        ):
            if len({_json_hash(row[field]) for row in outer_64_rows}) != 1:
                violations.append(
                    f"outer-64 cross-configuration {field} "
                    f"{sample['sample_id']}"
                )
    for row in per_q_rows + summary_rows:
        if int(row["eligible_tiles"]) != (
            int(row["skipped_tiles"]) + int(row["retained_tiles"])
        ):
            violations.append(f"count identity {row['configuration']}")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                violations.append(f"non-finite {key} {row['configuration']}")
    for configuration_name in [
        row["configuration"] for row in _sparse_configurations()
    ]:
        q_rows = [
            row
            for row in per_q_rows
            if row["configuration"] == configuration_name
        ]
        weight_sum = sum(float(row["forward_pass_weight"]) for row in q_rows)
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-12):
            violations.append(f"weights {configuration_name}: {weight_sum}")
        summary = next(
            row
            for row in summary_rows
            if row["configuration"] == configuration_name
        )
        for metric in (
            "row_vote_sparsity",
            "physical_tile_sparsity",
            "valid_element_sparsity",
        ):
            reconstructed = sum(
                float(row[f"weighted_{metric}_contribution"])
                for row in q_rows
            )
            if not math.isclose(
                reconstructed,
                float(summary[f"weighted_{metric}"]),
                abs_tol=1e-12,
            ):
                violations.append(
                    f"weighted reconstruction {configuration_name}/{metric}"
                )
    if not all(bool(row["passed"]) for row in trace_checks):
        violations.append("one or more shard trace checks failed")
    dense_regression = json.loads(
        (output_dir / "dualcache_dense_regression.json").read_text(
            encoding="utf-8"
        )
    )
    if not dense_regression["exact_match"] or not dense_regression["all_finite"]:
        violations.append("dense-disabled regression")
    correctness = {
        "passed": not violations,
        "violations": violations,
        "samples": SAMPLE_COUNT,
        "configurations": len(_all_configurations()),
        "sparse_configurations": len(_sparse_configurations()),
        "prediction_rows": len(predictions),
        "per_forward_rows": len(per_forward),
        "per_layer_head_rows": detailed_rows,
        "sample_manifest_validation": manifest_checks,
        "dense_disabled_regression": dense_regression,
        "official_scorer_commit": RULER_COMMIT,
        "official_scorer_sha256": scorer_hash,
        "same_prompt_hashes_across_configurations": True,
        "same_prompts_seeds_and_generation_budgets_across_configurations": True,
        "same_common_generation_settings_within_outer_64_configurations": True,
        "same_initial_denoising_state_within_outer_64_configurations": True,
        "outer_block_sizes_compared": list(
            dict.fromkeys(
                int(row["outer_block_size"])
                for row in _all_configurations()
            )
        ),
        "trajectory_note": (
            "All outer-64 configurations start from the exact same masked "
            "block. Standalone b4/b8/b16/b32 configurations use their named "
            "outer-block boundaries by definition, so their initial masked "
            "blocks and subsequent trajectories differ. All configurations "
            "still share prompts, seeds, generation budgets, and scoring."
        ),
        "dualcache_outer_block_size": OUTER_BLOCK_SIZE,
        "lambda": BLASST_LAMBDA,
        "physical_tiles": {"query": Q_TILE_SIZE, "key_value": KV_TILE_SIZE},
        "structural_masking_excluded_from_sparsity_denominators": True,
        "forward_weights_sum_to_one": True,
        "weighted_aggregates_reconstructed_from_per_q": True,
        "global_aggregation": "ratios of total integer counts",
        "runtime_query_lengths_reported_without forcing": True,
        "dense_qk_reference_masking": True,
        "runtime_speedup_claimed": False,
    }
    _write_json(output_dir / "dualcache_correctness.json", correctness)
    if violations:
        raise AssertionError(json.dumps(correctness, indent=2))


def _plot_results(output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/blasst-matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = _read_csv(output_dir / "dualcache_configuration_summary.csv")
    baseline = next(
        row for row in rows if row["configuration"] == "sparse_no_dualcache_b64"
    )
    standalone = sorted(
        [
            row
            for row in rows
            if row["dual_cache_enabled"] == "False"
            and int(row["outer_block_size"]) in SUB_BLOCK_SIZES
        ],
        key=lambda row: int(row["outer_block_size"]),
    )
    dual = sorted(
        [row for row in rows if row["dual_cache_enabled"] == "True"],
        key=lambda row: int(row["sub_block_size"]),
    )

    def save(figure: Any, name: str) -> None:
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=300)
        figure.savefig(output_dir / f"{name}.pdf")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.3, 5.3))
    x = [int(row["sub_block_size"]) for row in dual]
    standalone_x = [int(row["outer_block_size"]) for row in standalone]
    axis.plot(
        x,
        [float(row["weighted_row_vote_sparsity"]) for row in dual],
        marker="o",
        label="DualCache row-vote (outer 64)",
    )
    axis.plot(
        x,
        [float(row["weighted_physical_tile_sparsity"]) for row in dual],
        marker="s",
        label="DualCache physical (outer 64)",
    )
    axis.plot(
        standalone_x,
        [float(row["weighted_row_vote_sparsity"]) for row in standalone],
        marker="o",
        linestyle="--",
        label="No DualCache row-vote (outer = x)",
    )
    axis.plot(
        standalone_x,
        [
            float(row["weighted_physical_tile_sparsity"])
            for row in standalone
        ],
        marker="s",
        linestyle="--",
        label="No DualCache physical (outer = x)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Update/sub-block size")
    axis.set_ylabel("Sparsity")
    axis.set_title(
        "2D-BLASST sparsity: DualCache versus standalone blocks"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "dualcache_weighted_sparsity")

    figure, axis = plt.subplots(figsize=(8.3, 5.3))
    axis.plot(
        x,
        [float(row["global_physical_tile_sparsity"]) for row in dual],
        marker="o",
        label="DualCache (outer 64)",
    )
    axis.plot(
        standalone_x,
        [float(row["global_physical_tile_sparsity"]) for row in standalone],
        marker="o",
        linestyle="--",
        label="No DualCache (outer = x)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Update/sub-block size")
    axis.set_ylabel("Globally count-aggregated physical sparsity")
    axis.set_title(
        "Physical sparsity: DualCache versus standalone blocks"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save(figure, "dualcache_global_physical_sparsity")

    figure, axis = plt.subplots(figsize=(8.3, 5.3))
    axis.plot(
        x,
        [float(row["ruler_accuracy"]) for row in dual],
        marker="o",
        label="DualCache (outer 64)",
    )
    axis.plot(
        standalone_x,
        [float(row["ruler_accuracy"]) for row in standalone],
        marker="o",
        linestyle="--",
        label="No DualCache (outer = x)",
    )
    dense_accuracy = float(
        next(
            row
            for row in _read_csv(
                output_dir / "dualcache_accuracy_per_task.csv"
            )
            if row["configuration"] == "dense"
            and row["scope"] == "overall"
        )["ruler_accuracy"]
    )
    axis.axhline(
        dense_accuracy,
        color="0.4",
        linewidth=1,
        linestyle=":",
        label="Dense outer-64 reference",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    accuracy_values = [
        *(float(row["ruler_accuracy"]) for row in dual),
        *(float(row["ruler_accuracy"]) for row in standalone),
        dense_accuracy,
    ]
    padding = max(0.01, (max(accuracy_values) - min(accuracy_values)) * 0.15)
    axis.set_ylim(
        max(0.0, min(accuracy_values) - padding),
        min(1.0, max(accuracy_values) + padding),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Update/sub-block size")
    axis.set_ylabel("Official RULER accuracy")
    axis.set_title(
        "RULER accuracy: DualCache versus standalone sparse blocks"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save(figure, "dualcache_ruler_accuracy")

    figure, axis = plt.subplots(figsize=(8.3, 5.3))
    for label, group, marker in (
        ("DualCache (outer 64)", dual, "o"),
        ("No DualCache (outer = x)", standalone, "s"),
    ):
        axis.scatter(
            [float(row["global_physical_tile_sparsity"]) for row in group],
            [float(row["ruler_accuracy"]) for row in group],
            s=65,
            marker=marker,
            label=label,
        )
        for row in group:
            prefix = "DC s" if bool(row["dual_cache_enabled"] == "True") else "noDC b"
            axis.annotate(
                f"{prefix}{row['sub_block_size']}",
                (
                float(row["global_physical_tile_sparsity"]),
                float(row["ruler_accuracy"]),
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Globally count-aggregated physical sparsity")
    axis.set_ylabel("Official RULER accuracy")
    axis.set_title("Accuracy–physical-sparsity trade-off")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    save(figure, "dualcache_accuracy_sparsity_tradeoff")


def _write_report(args: argparse.Namespace, output_dir: Path) -> None:
    summary = _read_csv(
        output_dir / "dualcache_configuration_summary.csv"
    )
    per_q = _read_csv(output_dir / "dualcache_per_query_length.csv")
    accuracy = _read_csv(output_dir / "dualcache_accuracy_per_task.csv")
    baseline = next(
        row
        for row in summary
        if row["configuration"] == "sparse_no_dualcache_b64"
    )
    standalone = sorted(
        [
            row
            for row in summary
            if row["dual_cache_enabled"] == "False"
            and int(row["outer_block_size"]) in SUB_BLOCK_SIZES
        ],
        key=lambda row: int(row["outer_block_size"]),
    )
    dual = sorted(
        [row for row in summary if row["dual_cache_enabled"] == "True"],
        key=lambda row: int(row["sub_block_size"]),
    )
    dense_accuracy = float(
        next(
            row
            for row in accuracy
            if row["configuration"] == "dense" and row["scope"] == "overall"
        )["ruler_accuracy"]
    )
    baseline_weighted = float(
        baseline["weighted_physical_tile_sparsity"]
    )
    baseline_global = float(baseline["global_physical_tile_sparsity"])
    baseline_accuracy = float(baseline["ruler_accuracy"])

    acceptable = [
        row for row in dual if float(row["ruler_accuracy"]) >= baseline_accuracy
    ]
    best = max(
        acceptable or dual,
        key=lambda row: (
            float(row["global_physical_tile_sparsity"]),
            float(row["ruler_accuracy"]),
        ),
    )
    largest_weighting_gap = max(
        [baseline, *dual],
        key=lambda row: abs(
            float(row["weighted_physical_tile_sparsity"])
            - float(row["global_physical_tile_sparsity"])
        ),
    )
    max_gap = (
        float(largest_weighting_gap["weighted_physical_tile_sparsity"])
        - float(largest_weighting_gap["global_physical_tile_sparsity"])
    )

    lines = [
        "# Sub-block splitting and DualCache with 2D-BLASST on RULER",
        "",
        "This is dense-QK observation/reference masking at λ=0.003 with "
        "physical Q/KV tiles 128/64. It does not measure or claim kernel "
        "speedup.",
        "",
        "## Fixed setup",
        "",
        f"- Pinned NVIDIA RULER commit: `{RULER_COMMIT}`.",
        "- Exact same 100 cached 8K prompts, task mixture, inference seeds, "
        "threshold, generation budgets, and deterministic decoding in every "
        "configuration.",
        "- DualCache curve: outer diffusion block size 64 with sub-block sizes "
        "4, 8, 16, and 32.",
        "- Standalone comparison curve: no DualCache, with outer/query block "
        "sizes 4, 8, 16, and 32.",
        "- The curves intentionally differ in outer-block semantics; they "
        "compare the sub-block optimization against decoding directly at the "
        "same effective update size.",
        "",
        "## Main results",
        "",
        "| Configuration | Outer | Sub-block | Observed Q distribution | Weighted row | Weighted physical | Global physical | RULER accuracy | Δ dense |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in [*standalone, baseline, *dual]:
        lines.append(
            f"| {row['configuration']} | {row['outer_block_size']} | "
            f"{row['sub_block_size']} | "
            f"{row['query_length_distribution']} | "
            f"{float(row['weighted_row_vote_sparsity']):.2%} | "
            f"{float(row['weighted_physical_tile_sparsity']):.2%} | "
            f"{float(row['global_physical_tile_sparsity']):.2%} | "
            f"{float(row['ruler_accuracy']):.2%} | "
            f"{float(row['accuracy_change_from_dense']) * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"Dense outer-block-64 reference accuracy: {dense_accuracy:.2%}. "
            "Dense-answer agreement is diagnostic only; the accuracy columns "
            "use the pinned official RULER scorer.",
            "",
            "## Direct sub-block comparison",
            "",
            "| Update size | No-DualCache physical | DualCache physical | Δ physical | No-DualCache accuracy | DualCache accuracy | Δ accuracy |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    standalone_by_size = {
        int(row["sub_block_size"]): row for row in standalone
    }
    for row in dual:
        comparison = standalone_by_size[int(row["sub_block_size"])]
        lines.append(
            f"| {row['sub_block_size']} | "
            f"{float(comparison['global_physical_tile_sparsity']):.2%} | "
            f"{float(row['global_physical_tile_sparsity']):.2%} | "
            f"{float(row['global_physical_change_from_corresponding_no_dualcache']) * 100:+.2f} pp | "
            f"{float(comparison['ruler_accuracy']):.2%} | "
            f"{float(row['ruler_accuracy']):.2%} | "
            f"{float(row['accuracy_change_from_corresponding_no_dualcache']) * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "On these measured points, DualCache does not improve the paired "
            "accuracy–sparsity outcome over decoding directly with the same "
            "update size. Standalone b4 and b8 are substantially better on "
            "both metrics; b16 is modestly better on both; at size 32, "
            "DualCache gains a small amount of physical sparsity but loses "
            "accuracy. This is a system-level comparison, not a pure cache "
            "on/off ablation, because the standalone curve changes the outer "
            "diffusion block boundary.",
            "",
            "## Where the physical-sparsity change comes from",
            "",
        ]
    )
    for row in dual:
        configuration = row["configuration"]
        q_rows = [
            q_row
            for q_row in per_q
            if q_row["configuration"] == configuration
        ]
        contributions = []
        for q_row in q_rows:
            contribution = float(q_row["forward_pass_weight"]) * (
                float(q_row["physical_tile_sparsity"]) - baseline_weighted
            )
            contributions.append(
                f"Q={q_row['query_length']} contributes "
                f"{contribution * 100:+.2f} pp"
            )
        lines.append(
            f"- `{configuration}`: weighted physical change "
            f"{(float(row['weighted_physical_tile_sparsity']) - baseline_weighted) * 100:+.2f} pp; "
            + "; ".join(contributions)
            + "."
        )
    lines.extend(
        [
            "",
            "## Answers to the requested questions",
            "",
            "1. **Physical sparsity increase.**",
        ]
    )
    for row in dual:
        lines.append(
            f"   - `{row['configuration']}`: forward-weighted "
            f"{(float(row['weighted_physical_tile_sparsity']) - baseline_weighted) * 100:+.2f} pp; "
            f"global-count "
            f"{(float(row['global_physical_tile_sparsity']) - baseline_global) * 100:+.2f} pp."
        )
    lines.extend(
        [
            "2. **Smaller-Q versus refresh calls.** The decomposition above "
            "uses each observed Q stratum's forward weight and its difference "
            "from the no-DualCache baseline; it sums exactly to the reported "
            "weighted change.",
            "3. **Accuracy beyond BLASST alone.** The no-DualCache sparse "
            f"baseline changes dense accuracy by "
            f"{(baseline_accuracy - dense_accuracy) * 100:+.2f} pp. "
            "DualCache's additional changes relative to that sparse baseline "
            "are shown in the main table.",
            "4. **Best measured DualCache-only trade-off.** Using the declared criterion "
            "\"highest global physical sparsity among configurations whose "
            "official accuracy is at least the sparse baseline,\" the selected "
            f"configuration is `{best['configuration']}`. "
            + (
                ""
                if acceptable
                else "No DualCache configuration met the accuracy constraint, "
                "so this is the highest-sparsity fallback. "
            ),
            "   Across both plotted curves, standalone b4 has the highest "
            "measured accuracy and physical sparsity, subject to the "
            "outer-block-semantics caveat above.",
            "5. **Forward-weighted versus global.** The largest physical "
            f"difference is {max_gap * 100:+.2f} pp for "
            f"`{largest_weighting_gap['configuration']}`. These metrics answer "
            "different questions: equal weight per denoising forward versus "
            "weight proportional to total eligible physical tiles.",
            "",
            "## Validation and artifacts",
            "",
            "- Query lengths were taken from runtime traces, not assumed.",
            "- Forward weights sum to one and weighted values reconstruct from "
            "the exported per-Q contributions.",
            "- Global sparsity is computed from summed integer counts; structural "
            "tiles are outside eligible denominators.",
            "- Exact dense-disabled dispatch, finite values, prompt identity, "
            "configuration metadata, and `eligible = skipped + retained` are "
            "checked in `dualcache_correctness.json`.",
            "- Per-forward rows are in `dualcache_per_forward_pass.csv`; "
            "layer/head rows are gzip-compressed CSV.",
            "- PNG/PDF plots cover weighted sparsity, global physical sparsity, "
            "accuracy, and the accuracy–sparsity trade-off.",
            "",
        ]
    )
    (output_dir / "dualcache_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_config(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "dualcache_run_config.json"
    prior = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {}
    )
    environments = dict(prior.get("phase_environments", {}))
    environments[str(args.phase)] = _environment()
    source_samples, _ = _source_paths(args)
    sample_copy = output_dir / "ruler_samples_8k.jsonl"
    config = {
        **prior,
        **vars(args),
        "dualcache_outer_block_size": OUTER_BLOCK_SIZE,
        "dualcache_sub_block_sizes": list(SUB_BLOCK_SIZES),
        "standalone_no_dualcache_block_sizes": [
            *SUB_BLOCK_SIZES,
            OUTER_BLOCK_SIZE,
        ],
        "blasst_lambda": BLASST_LAMBDA,
        "physical_attention_tiles": {
            "query": Q_TILE_SIZE,
            "key_value": KV_TILE_SIZE,
        },
        "expected_samples": SAMPLE_COUNT,
        "target_context": TARGET_CONTEXT,
        "configurations_spec": _all_configurations(),
        "selected_configurations_this_invocation": (
            _selected_configuration_names(args)
        ),
        "source_sample_manifest": str(source_samples),
        "source_sample_sha256": (
            _sha256_file(source_samples) if source_samples.exists() else ""
        ),
        "copied_sample_sha256": (
            _sha256_file(sample_copy) if sample_copy.exists() else ""
        ),
        "ruler_commit": RULER_COMMIT,
        "deterministic_decoding": {
            "temperature": 0.0,
            "threshold": args.denoising_threshold,
            "per_sample_inference_seed": True,
        },
        "dense_qk_reference_masking": True,
        "runtime_speedup_claimed": False,
        "phase_environments": environments,
    }
    _write_json(path, config)


def main() -> None:
    args = parse_args()
    if int(args.num_samples) <= 0 or int(args.num_samples) > SAMPLE_COUNT:
        raise ValueError("num-samples must be in [1, 100]")
    if not args.pilot and int(args.num_samples) != SAMPLE_COUNT:
        raise ValueError("production execution requires exactly 100 samples")
    if args.denoising_threshold != 0.9:
        raise ValueError("generation threshold must remain 0.9")
    _verify_ruler_checkout(Path(args.ruler_root))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "prepare"):
        _prepare(args, output_dir)
    _write_config(args, output_dir)
    if args.phase in ("all", "run"):
        _run(args, output_dir)
    if args.phase in ("all", "aggregate"):
        _aggregate(args, output_dir)
    if args.phase in ("all", "report"):
        _plot_results(output_dir)
        _write_report(args, output_dir)
    _write_config(args, output_dir)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "output_dir": str(output_dir),
                "ruler_commit": RULER_COMMIT,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
