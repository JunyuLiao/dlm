#!/usr/bin/env python3
"""Run original causal BLASST on Gemma 4 AR decode over RULER prompts.

This is intentionally independent from the diffusion-model BLASST 2D code.
Prompt prefill is dense SDPA.  Decode uses Q tiles of one token and 64-token
KV blocks, following Algorithm 1's chronological running-maximum test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from dllm.evaluation.ruler.io import append_jsonl, read_jsonl, sha256_file, write_json, write_jsonl
from dllm.evaluation.ruler.official import RULER_COMMIT, score_predictions


MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
DEFAULT_PARENT_MANIFEST = Path(
    "results/blasst/diffusion_gemma/paper_8k_n650/manifest/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("results/blasst/gemma4_ar/ruler8k_n260")
DEFAULT_LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1)
COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, length, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, length, head_dim)


def _valid_positions(
    attention_mask: torch.Tensor | None,
    scores: torch.Tensor,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = attention_mask
    if mask.shape[-2] != scores.shape[-2]:
        mask = mask[..., -scores.shape[-2] :, :]
    if mask.shape[-1] != scores.shape[-1]:
        mask = mask[..., -scores.shape[-1] :]
    if mask.dtype == torch.bool:
        valid = mask
    else:
        # Transformers causal masks use the finite dtype minimum rather than
        # necessarily -inf, while every valid entry is exactly zero.
        valid = mask > -1.0e4
    return valid.expand_as(scores)


def algorithm1_skip_mask(
    scores: torch.Tensor,
    valid: torch.Tensor,
    lambda_value: float,
    kv_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-head KV-block skip and eligibility masks for one Q row."""
    if scores.ndim != 4 or scores.shape[-2] != 1:
        raise ValueError("causal AR BLASST requires scores shaped [B,H,1,K]")
    if valid.shape != scores.shape:
        raise ValueError("valid mask must have the same shape as scores")
    if not 0.0 < lambda_value < 1.0:
        raise ValueError("lambda must be strictly between zero and one")
    if kv_tile_size <= 0:
        raise ValueError("kv_tile_size must be positive")

    key_length = scores.shape[-1]
    padding = (-key_length) % kv_tile_size
    masked_scores = scores.masked_fill(~valid, -torch.inf)
    if padding:
        masked_scores = F.pad(masked_scores, (0, padding), value=-torch.inf)
        valid = F.pad(valid, (0, padding), value=False)
    block_count = masked_scores.shape[-1] // kv_tile_size
    block_scores = masked_scores.reshape(*scores.shape[:-1], block_count, kv_tile_size)
    block_valid = valid.reshape(*scores.shape[:-1], block_count, kv_tile_size).any(dim=-1)
    block_max = block_scores.amax(dim=-1)
    running_max = torch.cummax(block_max, dim=-1).values
    skip = block_valid & ((block_max - running_max) < math.log(lambda_value))
    return skip.squeeze(-2), block_valid.squeeze(-2)


class CausalBlasstRuntime:
    def __init__(self, kv_tile_size: int, num_layers: int = 30) -> None:
        self.kv_tile_size = kv_tile_size
        self.num_layers = num_layers
        self.lambda_value: float | None = None
        self.apply_mask = False
        self.sample_id = ""
        self._counts: torch.Tensor | None = None
        self._calls = [0] * num_layers
        self._key_tokens = [0] * num_layers
        self._query_heads = [0] * num_layers
        self._attention_types = [""] * num_layers

    def configure(self, lambda_value: float | None, *, apply_mask: bool) -> None:
        self.lambda_value = lambda_value
        self.apply_mask = apply_mask

    def reset_sample(self, sample_id: str) -> None:
        self.sample_id = sample_id
        if self._counts is not None:
            self._counts.zero_()
        self._calls = [0] * self.num_layers
        self._key_tokens = [0] * self.num_layers
        self._query_heads = [0] * self.num_layers
        self._attention_types = [""] * self.num_layers

    def record(
        self,
        module: Any,
        skip: torch.Tensor,
        eligible: torch.Tensor,
        key_length: int,
    ) -> None:
        layer = int(module.layer_idx)
        if not 0 <= layer < self.num_layers:
            raise RuntimeError(f"unexpected Gemma 4 layer index: {layer}")
        if self._counts is None or self._counts.device != skip.device:
            self._counts = torch.zeros(
                (self.num_layers, len(COUNT_FIELDS)), dtype=torch.int64, device=skip.device
            )
        skipped = skip.sum(dtype=torch.int64)
        eligible_count = eligible.sum(dtype=torch.int64)
        total = torch.tensor(eligible.numel(), dtype=torch.int64, device=skip.device)
        values = torch.stack(
            (eligible_count, skipped, eligible_count - skipped, total - eligible_count)
        )
        self._counts[layer].add_(values)
        self._calls[layer] += 1
        self._key_tokens[layer] += int(key_length)
        self._query_heads[layer] = int(skip.shape[1])
        self._attention_types[layer] = "local" if module.is_sliding else "global"

    def sample_summary(self) -> dict[str, Any]:
        if self._counts is None:
            values = [[0] * len(COUNT_FIELDS) for _ in range(self.num_layers)]
        else:
            values = self._counts.detach().cpu().tolist()
        layers = []
        overall = {field: 0 for field in COUNT_FIELDS}
        for layer, counts in enumerate(values):
            if self._calls[layer] == 0:
                continue
            row = {
                "layer": layer,
                "attention_type": self._attention_types[layer],
                "decode_attention_calls": self._calls[layer],
                "summed_key_tokens": self._key_tokens[layer],
                "query_heads": self._query_heads[layer],
                **{field: int(counts[index]) for index, field in enumerate(COUNT_FIELDS)},
            }
            for field in COUNT_FIELDS:
                overall[field] += row[field]
            layers.append(row)
        overall["physical_sparsity"] = (
            overall["skipped_tiles"] / overall["eligible_tiles"]
            if overall["eligible_tiles"]
            else 0.0
        )
        return {"overall": overall, "layers": layers}


def make_attention_interface(
    runtime: CausalBlasstRuntime,
    dense_prefill: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    def gemma4_ar_blasst_attention(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float | int = 0.0,
        scaling: float | None = None,
        softcap: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.shape[-2] != 1:
            return dense_prefill(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                softcap=softcap,
                **kwargs,
            )

        if scaling is None:
            scaling = module.head_dim**-0.5
        key_states = _repeat_kv(key, module.num_key_value_groups)
        value_states = _repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            scores = torch.tanh(scores / softcap) * softcap
        valid = _valid_positions(attention_mask, scores)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, -torch.inf)
            else:
                scores = scores + attention_mask

        if runtime.lambda_value is not None:
            skip, eligible = algorithm1_skip_mask(
                scores, valid, runtime.lambda_value, runtime.kv_tile_size
            )
            runtime.record(module, skip, eligible, scores.shape[-1])
            if runtime.apply_mask:
                token_skip = skip.repeat_interleave(runtime.kv_tile_size, dim=-1)
                token_skip = token_skip[..., : scores.shape[-1]].unsqueeze(-2)
                scores = scores.masked_fill(token_skip, -torch.inf)

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=float(dropout), training=module.training)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights

    return gemma4_ar_blasst_attention


def prepare_manifest(parent_path: Path, output_dir: Path, samples_per_task: int) -> Path:
    parent_path = parent_path.resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("ruler", {}).get("commit") != RULER_COMMIT:
        raise RuntimeError("parent manifest is not from the pinned RULER revision")
    source_path = Path(parent["samples"]["path"])
    source_rows = read_jsonl(source_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["task"])].append(row)
    tasks = list(parent["tasks"])
    if any(len(grouped[task]) < samples_per_task for task in tasks):
        raise RuntimeError("parent manifest does not contain enough samples per task")
    selected = []
    for index in range(samples_per_task):
        selected.extend(grouped[task][index] for task in tasks)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = (output_dir / "samples.jsonl").resolve()
    write_jsonl(samples_path, selected)
    manifest = {
        **parent,
        "schema_version": max(3, int(parent.get("schema_version", 1))),
        "model_adapter": "gemma4_ar",
        "model_path": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_configuration": {"thinking": False},
        "requested_num_samples": len(selected),
        "actual_num_samples": len(selected),
        "task_counts": {task: samples_per_task for task in tasks},
        "parent_manifest": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "selection": f"first {samples_per_task} samples per task",
            "prompt_bytes_unchanged": True,
        },
        "samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    import transformers

    return {
        "repository_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _case_name(lambda_value: float | None) -> str:
    return "dense" if lambda_value is None else f"lambda_{lambda_value:g}".replace(".", "p")


def _aggregate_case(
    case_dir: Path,
    rows: list[dict[str, Any]],
    ruler_root: Path,
    lambda_value: float | None,
) -> dict[str, Any]:
    by_layer: dict[int, dict[str, Any]] = {}
    for prediction in rows:
        for source in prediction["attention_stats"]["layers"]:
            layer = int(source["layer"])
            target = by_layer.setdefault(
                layer,
                {
                    "layer": layer,
                    "attention_type": source["attention_type"],
                    "decode_attention_calls": 0,
                    "summed_key_tokens": 0,
                    "query_heads": int(source["query_heads"]),
                    **{field: 0 for field in COUNT_FIELDS},
                },
            )
            if target["attention_type"] != source["attention_type"]:
                raise RuntimeError(f"layer {layer} changed attention type")
            if target["query_heads"] != int(source["query_heads"]):
                raise RuntimeError(f"layer {layer} changed query-head count")
            for field in ("decode_attention_calls", "summed_key_tokens", *COUNT_FIELDS):
                target[field] += int(source[field])

    layer_rows = []
    for layer in sorted(by_layer):
        row = by_layer[layer]
        if row["eligible_tiles"] != row["skipped_tiles"] + row["retained_tiles"]:
            raise RuntimeError(f"tile count identity failed for layer {layer}")
        row["physical_sparsity"] = (
            row["skipped_tiles"] / row["eligible_tiles"] if row["eligible_tiles"] else 0.0
        )
        # A 1024-token sliding window can overlap at most 17 unaligned KV64
        # tiles. Old full-cache entries may still be present in the eager
        # reference tensors, but they must be structural and ineligible.
        max_local_eligible = (
            row["decode_attention_calls"] * row["query_heads"] * 17
        )
        if row["attention_type"] == "local" and row["eligible_tiles"] > max_local_eligible:
            raise RuntimeError(
                f"local layer {layer} exceeded Gemma 4's 1024-token eligible KV window"
            )
        layer_rows.append(row)
    overall = {field: sum(row[field] for row in layer_rows) for field in COUNT_FIELDS}
    by_attention = []
    for attention_type in ("global", "local"):
        subset = [row for row in layer_rows if row["attention_type"] == attention_type]
        counts = {field: sum(row[field] for row in subset) for field in COUNT_FIELDS}
        counts["physical_sparsity"] = (
            counts["skipped_tiles"] / counts["eligible_tiles"] if counts["eligible_tiles"] else 0.0
        )
        by_attention.append({"attention_type": attention_type, **counts})
    overall["physical_sparsity"] = (
        overall["skipped_tiles"] / overall["eligible_tiles"]
        if overall["eligible_tiles"]
        else 0.0
    )
    per_task, accuracy = score_predictions(rows, ruler_root)
    summary = {
        "schema_version": 1,
        "case": _case_name(lambda_value),
        "lambda": lambda_value,
        "samples": len(rows),
        "official_ruler_accuracy": accuracy,
        "per_task_accuracy": per_task,
        "attention_sparsity": overall,
        "sparsity_by_attention_type": by_attention,
    }
    write_json(case_dir / "summary.json", summary)
    if layer_rows:
        with (case_dir / "per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)
    return summary


def _load_diffusion_reference() -> dict[str, Any] | None:
    root = Path("results/blasst/diffusion_gemma/paper_8k_n650/screen/lambda_0p003")
    summary_path = root / "summary.json"
    layer_path = root / "attention_stats/per_layer.csv"
    if not summary_path.exists() or not layer_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with layer_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = counts[row["attention_type"]]
            group["eligible_tiles"] += int(row["eligible_tiles"])
            group["skipped_tiles"] += int(row["skipped_tiles"])
    return {
        "samples": summary["actual_num_samples"],
        "overall": summary["attention_sparsity"]["physical_tile_sparsity"],
        "by_attention_type": {
            name: value["skipped_tiles"] / value["eligible_tiles"]
            for name, value in counts.items()
        },
    }


def _write_report(root: Path, summaries: list[dict[str, Any]]) -> None:
    dense = next(row for row in summaries if row["lambda"] is None)
    diffusion = _load_diffusion_reference()
    rows = []
    for summary in summaries:
        groups = {row["attention_type"]: row for row in summary["sparsity_by_attention_type"]}
        rows.append(
            {
                "case": summary["case"],
                "lambda": summary["lambda"],
                "accuracy": summary["official_ruler_accuracy"],
                "accuracy_delta": summary["official_ruler_accuracy"] - dense["official_ruler_accuracy"],
                "overall_sparsity": summary["attention_sparsity"]["physical_sparsity"],
                "global_sparsity": groups["global"]["physical_sparsity"],
                "local_sparsity": groups["local"]["physical_sparsity"],
            }
        )
    with (root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison = None
    lambda_003 = next((row for row in rows if row["lambda"] == 0.003), None)
    if diffusion is not None and lambda_003 is not None:
        comparison = {
            "lambda": 0.003,
            "ar_overall": lambda_003["overall_sparsity"],
            "diffusion_overall": diffusion["overall"],
            "difference": lambda_003["overall_sparsity"] - diffusion["overall"],
            "ar_global": lambda_003["global_sparsity"],
            "diffusion_global": diffusion["by_attention_type"].get("global"),
            "ar_local": lambda_003["local_sparsity"],
            "diffusion_local": diffusion["by_attention_type"].get("local"),
            "diffusion_samples": diffusion["samples"],
        }
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "algorithm": "BLASST Algorithm 1 causal decode; dense prefill",
        "q_tile_size": 1,
        "kv_tile_size": 64,
        "aggregation": "globally summed skipped / eligible head tiles",
        "structural_tiles": "excluded",
        "sweep": rows,
        "diffusion_gemma_reference": comparison,
    }
    write_json(root / "report.json", report)

    lines = [
        "# Gemma 4 AR BLASST on 8K RULER",
        "",
        "Original causal BLASST Algorithm 1 is applied only during token-by-token AR decode (Q tile 1, KV tile 64). Prompt prefill is dense SDPA in every case.",
        "",
        "| Case | Accuracy | Delta vs dense | Overall sparsity | Global | Local |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = "dense" if row["lambda"] is None else f"lambda={row['lambda']:g}"
        lines.append(
            f"| {label} | {row['accuracy']:.2%} | {row['accuracy_delta']:+.2%} | "
            f"{row['overall_sparsity']:.2%} | {row['global_sparsity']:.2%} | {row['local_sparsity']:.2%} |"
        )
    if comparison is not None:
        lines.extend(
            [
                "",
                "## Matched lambda=0.003 comparison",
                "",
                f"Gemma 4 AR: {comparison['ar_overall']:.2%} overall; DiffusionGemma 2D reference: {comparison['diffusion_overall']:.2%}; difference: {comparison['difference']:+.2%}.",
                "",
                "The DiffusionGemma reference is the existing paper-aligned screen using Q128 x KV64 on the identical source prompt set (10 samples/task). The AR run uses the first 20 samples/task; its first half is the exact same prompt subset.",
            ]
        )
    lines.extend(
        [
            "",
            "Physical sparsity excludes structurally masked KV tiles and is the globally summed skipped/eligible ratio. The eager reference still computes QK to make Algorithm 1 decisions; skipped tiles represent avoided softmax/PV/V-loading work in a fused BLASST kernel, not measured wall-clock speedup.",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ruler-root", type=Path, default=Path("/tmp/NVIDIA-RULER"))
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--kv-tile-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--only-case", help="dense or a numeric lambda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.samples_per_task <= 0:
        raise ValueError("samples-per-task must be positive")
    if args.kv_tile_size != 64:
        raise ValueError("this experiment fixes the KV tile size to 64")
    lambdas = tuple(float(value) for value in args.lambdas.split(",") if value)
    if not lambdas or any(not 0.0 < value < 1.0 for value in lambdas):
        raise ValueError("lambdas must be a non-empty list of values between zero and one")
    if args.only_case is None:
        cases: tuple[float | None, ...] = (None, *lambdas)
    elif args.only_case == "dense":
        cases = (None,)
    else:
        selected = float(args.only_case)
        if selected not in lambdas:
            raise ValueError("--only-case lambda must also appear in --lambdas")
        cases = (selected,)

    root = args.output_root.resolve()
    manifest_path = prepare_manifest(
        args.parent_manifest, root / "manifest", args.samples_per_task
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = read_jsonl(Path(manifest["samples"]["path"]))
    expected = args.samples_per_task * len(manifest["tasks"])
    if len(samples) != expected:
        raise RuntimeError("derived manifest does not have the exact requested count")

    from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward

    runtime = CausalBlasstRuntime(args.kv_tile_size)
    dense_prefill = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)
    interface_name = "gemma4_ar_blasst_algorithm1"
    ALL_ATTENTION_FUNCTIONS.register(
        interface_name, make_attention_interface(runtime, dense_prefill)
    )
    # A custom attention name otherwise bypasses Transformers' causal and
    # sliding-window mask creation.  Reuse SDPA's boolean mask builder while
    # keeping the separate Algorithm 1 attention implementation above.
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        interface_name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    )
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation=interface_name,
        local_files_only=True,
    ).eval()
    if {str(parameter.device) for parameter in model.parameters()} != {"cuda:0"}:
        raise RuntimeError("the complete Gemma 4 checkpoint was not loaded on cuda:0")

    environment = _environment()
    summaries = []
    for lambda_value in cases:
        case_name = _case_name(lambda_value)
        case_dir = root / "runs" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "algorithm_version": 1,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "model": args.model,
            "revision": args.revision,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "samples": len(samples),
            "samples_per_task": args.samples_per_task,
            "context_length": 8192,
            "lambda": lambda_value,
            "q_tile_size": 1,
            "kv_tile_size": args.kv_tile_size,
            "prefill_backend": "dense-sdpa",
            "decode_backend": "eager-algorithm1-reference",
            "apply_sparse_mask": lambda_value is not None,
            "physical_denominator": "eligible non-structural per-head KV tiles",
            "thinking": False,
            "max_new_tokens_override": args.max_new_tokens,
            **environment,
        }
        config["fingerprint"] = _json_hash(config)
        config_path = case_dir / "run_config.json"
        if config_path.exists():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            if existing_config.get("fingerprint") != config["fingerprint"]:
                raise RuntimeError(f"{case_dir} belongs to a different run")
        write_json(config_path, config)

        predictions_path = case_dir / "predictions.jsonl"
        existing = read_jsonl(predictions_path)
        completed = {row["sample_id"]: row for row in existing}
        runtime.configure(lambda_value, apply_mask=lambda_value is not None)
        for index, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            torch.manual_seed(int(sample["inference_seed"]))
            runtime.reset_sample(sample_id)
            encoded = processor.apply_chat_template(
                [{"role": "user", "content": str(sample["prompt"])}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            input_length = int(encoded["input_ids"].shape[-1])
            generation_limit = int(sample["tokens_to_generate"])
            if args.max_new_tokens is not None:
                generation_limit = min(generation_limit, args.max_new_tokens)
            started = time.perf_counter()
            generated = model.generate(
                **encoded,
                max_new_tokens=generation_limit,
                do_sample=False,
                use_cache=True,
                past_key_values=DynamicCache(config=model.config.text_config),
            )
            elapsed = time.perf_counter() - started
            completion = generated[0, input_length:].detach().cpu()
            prediction = processor.decode(completion, skip_special_tokens=True)
            row = {
                **sample,
                "prediction": prediction,
                "completion_tokens": completion.tolist(),
                "runtime_prompt_length": input_length,
                "elapsed_seconds": elapsed,
                "attention_stats": runtime.sample_summary(),
            }
            append_jsonl(predictions_path, row)
            completed[sample_id] = row
            print(f"completed {case_name} {index}/{len(samples)}: {sample_id}", flush=True)

        ordered = [completed[str(sample["sample_id"])] for sample in samples]
        if len(ordered) != len(samples):
            raise RuntimeError(f"{case_name} did not produce all requested predictions")
        summaries.append(
            _aggregate_case(case_dir, ordered, args.ruler_root, lambda_value)
        )

    # A partial --only-case run is intentionally resumable but cannot yet make
    # a cross-lambda report.  If every default case exists, aggregate them now.
    all_summaries = []
    for lambda_value in (None, *lambdas):
        summary_path = root / "runs" / _case_name(lambda_value) / "summary.json"
        if not summary_path.exists():
            break
        all_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if len(all_summaries) == len(lambdas) + 1:
        _write_report(root, all_summaries)


if __name__ == "__main__":
    main()
