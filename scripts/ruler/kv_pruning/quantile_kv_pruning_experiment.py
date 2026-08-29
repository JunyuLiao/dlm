#!/usr/bin/env python3
"""Oracle bottom-k KV-tile pruning on 8K DiffusionGemma RULER.

For each attention head and query row, this experiment first computes the dense
QK matrix, scores every structurally valid KV tile, drops the lowest requested
fraction, and recomputes the normalized attention output over retained tiles.
It is an oracle policy study: dense QK and ranking are explicitly not speedups.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.attention.blasst.core import (
    _attention_type,
    _finish_eager_attention,
    _prepare_attention_scores,
)
from dllm.evaluation.ruler.official import score_predictions
from dllm.models import GenerationRequest, create_adapter


METHODS = ("block_max", "random", "attention_mass", "output_norm")
METHOD_LABELS = {
    "block_max": "Block maximum",
    "random": "Random tiles",
    "attention_mass": "Attention mass",
    "output_norm": "Block output norm",
    "dense": "Dense eager",
}
DEFAULT_K_VALUES = (0.10, 0.25, 0.50, 0.75)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _condition_name(method: str, k_fraction: float) -> str:
    if k_fraction == 0.0:
        return "dense_eager"
    return f"{method}_k{int(round(100 * k_fraction)):02d}"


@dataclass
class StatBucket:
    calls: int = 0
    query_rows: int | torch.Tensor = 0
    eligible_tiles: int | torch.Tensor = 0
    dropped_tiles: int | torch.Tensor = 0
    discarded_mass_sum: float | torch.Tensor = 0.0
    discarded_mass_sq_sum: float | torch.Tensor = 0.0

    def add(self, other: "StatBucket") -> None:
        self.calls += other.calls
        self.query_rows += other.query_rows
        self.eligible_tiles += other.eligible_tiles
        self.dropped_tiles += other.dropped_tiles
        self.discarded_mass_sum += other.discarded_mass_sum
        self.discarded_mass_sq_sum += other.discarded_mass_sq_sum

    def summary(self) -> dict[str, Any]:
        # Attention hooks run thousands of times per sample. Keep their scalar
        # reductions on-device and synchronize once here, at the sample boundary,
        # instead of calling .item() inside every hook invocation.
        query_rows = int(self.query_rows.item()) if torch.is_tensor(self.query_rows) else int(self.query_rows)
        eligible_tiles = int(self.eligible_tiles.item()) if torch.is_tensor(self.eligible_tiles) else int(self.eligible_tiles)
        dropped_tiles = int(self.dropped_tiles.item()) if torch.is_tensor(self.dropped_tiles) else int(self.dropped_tiles)
        discarded_mass_sum = (
            float(self.discarded_mass_sum.item())
            if torch.is_tensor(self.discarded_mass_sum)
            else float(self.discarded_mass_sum)
        )
        discarded_mass_sq_sum = (
            float(self.discarded_mass_sq_sum.item())
            if torch.is_tensor(self.discarded_mass_sq_sum)
            else float(self.discarded_mass_sq_sum)
        )
        mean_mass = discarded_mass_sum / query_rows if query_rows else 0.0
        variance = (
            discarded_mass_sq_sum / query_rows - mean_mass**2
            if query_rows
            else 0.0
        )
        return {
            "calls": self.calls,
            "query_rows": query_rows,
            "eligible_tiles": eligible_tiles,
            "dropped_tiles": dropped_tiles,
            "actual_tile_drop_fraction": (
                dropped_tiles / eligible_tiles if eligible_tiles else 0.0
            ),
            "discarded_dense_attention_mass_mean": mean_mass,
            "discarded_dense_attention_mass_std_over_rows": math.sqrt(max(0.0, variance)),
        }

    @classmethod
    def from_summary(cls, value: dict[str, Any]) -> "StatBucket":
        rows = int(value["query_rows"])
        mean = float(value["discarded_dense_attention_mass_mean"])
        std = float(value["discarded_dense_attention_mass_std_over_rows"])
        return cls(
            calls=int(value["calls"]),
            query_rows=rows,
            eligible_tiles=int(value["eligible_tiles"]),
            dropped_tiles=int(value["dropped_tiles"]),
            discarded_mass_sum=mean * rows,
            discarded_mass_sq_sum=(std**2 + mean**2) * rows,
        )


@dataclass
class OracleStats:
    overall: StatBucket = field(default_factory=StatBucket)
    by_attention_type: dict[str, StatBucket] = field(default_factory=dict)
    by_layer: dict[str, StatBucket] = field(default_factory=dict)

    def update(
        self,
        *,
        attention_type: str,
        layer: int,
        query_rows: int | torch.Tensor,
        eligible_tiles: int | torch.Tensor,
        dropped_tiles: int | torch.Tensor,
        discarded_mass_sum: float | torch.Tensor,
        discarded_mass_sq_sum: float | torch.Tensor,
    ) -> None:
        partial = StatBucket(
            calls=1,
            query_rows=query_rows,
            eligible_tiles=eligible_tiles,
            dropped_tiles=dropped_tiles,
            discarded_mass_sum=discarded_mass_sum,
            discarded_mass_sq_sum=discarded_mass_sq_sum,
        )
        self.overall.add(partial)
        self.by_attention_type.setdefault(attention_type, StatBucket()).add(partial)
        self.by_layer.setdefault(str(layer), StatBucket()).add(partial)

    def summary(self) -> dict[str, Any]:
        return {
            "overall": self.overall.summary(),
            "by_attention_type": {
                key: value.summary() for key, value in sorted(self.by_attention_type.items())
            },
            "by_layer": {
                key: value.summary()
                for key, value in sorted(self.by_layer.items(), key=lambda row: int(row[0]))
            },
        }

    def add(self, other: "OracleStats") -> None:
        self.overall.add(other.overall)
        for key, value in other.by_attention_type.items():
            self.by_attention_type.setdefault(key, StatBucket()).add(value)
        for key, value in other.by_layer.items():
            self.by_layer.setdefault(key, StatBucket()).add(value)

    @classmethod
    def from_summary(cls, value: dict[str, Any]) -> "OracleStats":
        return cls(
            overall=StatBucket.from_summary(value["overall"]),
            by_attention_type={
                key: StatBucket.from_summary(bucket)
                for key, bucket in value["by_attention_type"].items()
            },
            by_layer={
                key: StatBucket.from_summary(bucket)
                for key, bucket in value["by_layer"].items()
            },
        )


class OracleTilePruner:
    """Dense-first, row-wise KV-tile ranker and exact pruning forward."""

    def __init__(
        self,
        method: str,
        k_fraction: float,
        kv_tile_size: int,
        output_norm_query_chunk: int = 32,
        random_seed: int = 20260812,
    ) -> None:
        if method not in METHODS and not (method == "dense" and k_fraction == 0.0):
            raise ValueError(f"unsupported score assignment: {method}")
        if not 0.0 <= k_fraction < 1.0:
            raise ValueError("k_fraction must be in [0, 1)")
        self.method = method
        self.k_fraction = float(k_fraction)
        self.kv_tile_size = int(kv_tile_size)
        self.output_norm_query_chunk = int(output_norm_query_chunk)
        self.random_seed = int(random_seed)
        self._sample_random_seed = self.random_seed
        self._random_generator: torch.Generator | None = None
        self.stats = OracleStats()

    def set_sample_seed(self, inference_seed: int) -> None:
        """Reset the independent random-policy stream for one benchmark sample."""
        self._sample_random_seed = (
            (self.random_seed * 0x9E3779B97F4A7C15) ^ int(inference_seed)
        ) & ((1 << 63) - 1)
        self._random_generator = None

    def _random_scores(self, tile_valid: torch.Tensor) -> torch.Tensor:
        if self._random_generator is None:
            self._random_generator = torch.Generator(device=tile_valid.device)
            self._random_generator.manual_seed(self._sample_random_seed)
        return torch.rand(
            tile_valid.shape,
            dtype=torch.float32,
            device=tile_valid.device,
            generator=self._random_generator,
        )

    def _output_norm_scores(
        self,
        scores: torch.Tensor,
        valid: torch.Tensor,
        value: torch.Tensor,
        kv_tiles: int,
    ) -> torch.Tensor:
        batch, heads, q_len, _ = scores.shape
        tile = self.kv_tile_size
        value_blocks = value.float().reshape(batch, heads, kv_tiles, tile, value.shape[-1])
        pieces = []
        for start in range(0, q_len, self.output_norm_query_chunk):
            stop = min(q_len, start + self.output_norm_query_chunk)
            chunk_scores = scores[:, :, start:stop].float()
            chunk_valid = valid[:, :, start:stop]
            row_has_key = chunk_valid.any(-1, keepdim=True)
            normalized = torch.where(row_has_key, chunk_scores, torch.zeros_like(chunk_scores))
            probability = F.softmax(normalized, dim=-1, dtype=torch.float32)
            probability = torch.where(row_has_key, probability, torch.zeros_like(probability))
            probability = probability.reshape(batch, heads, stop - start, kv_tiles, tile)
            contribution = torch.einsum(
                "bhqtk,bhtkd->bhqtd", probability, value_blocks
            )
            pieces.append(torch.linalg.vector_norm(contribution, dim=-1))
            del probability, contribution, normalized, chunk_scores, chunk_valid
        return torch.cat(pieces, dim=2)

    @torch.no_grad()
    def __call__(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        sliding_window: int | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        _, value, scores, valid = _prepare_attention_scores(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        if self.k_fraction == 0.0:
            return _finish_eager_attention(
                query, value, scores, valid, dropout, module.training
            )

        batch, heads, q_len, kv_len = scores.shape
        tile = self.kv_tile_size
        kv_tiles = math.ceil(kv_len / tile)
        kv_pad = kv_tiles * tile - kv_len
        if kv_pad:
            scores = F.pad(scores, (0, kv_pad), value=-torch.inf)
            valid = F.pad(valid, (0, kv_pad), value=False)
            value = F.pad(value, (0, 0, 0, kv_pad), value=0.0)
        score_blocks = scores.float().reshape(batch, heads, q_len, kv_tiles, tile)
        valid_blocks = valid.reshape(batch, heads, q_len, kv_tiles, tile)
        masked_blocks = score_blocks.masked_fill(~valid_blocks, -torch.inf)
        tile_valid = valid_blocks.any(-1)
        block_lse = torch.logsumexp(masked_blocks, dim=-1)

        if self.method == "block_max":
            importance = masked_blocks.amax(-1)
        elif self.method == "random":
            importance = self._random_scores(tile_valid)
        elif self.method == "attention_mass":
            # Softmax has one row-wise denominator, so ranking exact block
            # mass is identical to ranking block log-sum-exp.
            importance = block_lse
        elif self.method == "output_norm":
            importance = self._output_norm_scores(scores, valid, value, kv_tiles)
        else:
            raise AssertionError(self.method)

        counts = tile_valid.sum(-1)
        drop_counts = torch.floor(counts.float() * self.k_fraction + 0.5).to(torch.long)
        drop_counts = torch.minimum(drop_counts, (counts - 1).clamp_min(0))
        # Only the lowest requested fraction is needed. A full argsort makes
        # small-k conditions needlessly expensive at 8K; selecting the largest
        # possible per-row drop count and then slicing it by each row's actual
        # count is equivalent (apart from exact score ties).
        selection_width = min(
            kv_tiles - 1,
            int(math.floor(kv_tiles * self.k_fraction + 0.5)),
        )
        dropped_tiles = torch.zeros_like(tile_valid)
        if selection_width:
            retention_width = max(1, kv_tiles - selection_width)
            if retention_width < selection_width:
                # Aggressive pruning retains a much smaller set than it drops.
                # Select that top set and invert it; this is the same ranking
                # policy (apart from exact score ties) and avoids asking topk
                # to return 75--95% of all tiles.
                retain_counts = counts - drop_counts
                order = torch.topk(
                    importance.masked_fill(~tile_valid, -torch.inf),
                    k=retention_width,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ).indices
                selected_keep = (
                    torch.arange(retention_width, device=scores.device)
                    .reshape(*([1] * (order.ndim - 1)), retention_width)
                    < retain_counts[..., None]
                )
                retained_tiles = torch.zeros_like(tile_valid)
                retained_tiles.scatter_(-1, order, selected_keep)
                dropped_tiles = tile_valid & ~retained_tiles
            else:
                order = torch.topk(
                    importance.masked_fill(~tile_valid, torch.inf),
                    k=selection_width,
                    dim=-1,
                    largest=False,
                    sorted=True,
                ).indices
                selected_drop = (
                    torch.arange(selection_width, device=scores.device)
                    .reshape(*([1] * (order.ndim - 1)), selection_width)
                    < drop_counts[..., None]
                )
                dropped_tiles.scatter_(-1, order, selected_drop)
                dropped_tiles &= tile_valid

        row_lse = torch.logsumexp(block_lse, dim=-1, keepdim=True)
        block_mass = torch.exp(block_lse - row_lse)
        block_mass = torch.where(tile_valid, block_mass, torch.zeros_like(block_mass))
        discarded_mass = (block_mass * dropped_tiles).sum(-1)
        row_valid = tile_valid.any(-1)
        layer = int(getattr(module, "layer_idx", -1))
        self.stats.update(
            attention_type=_attention_type(module, sliding_window),
            layer=layer,
            query_rows=row_valid.sum(),
            eligible_tiles=tile_valid.sum(),
            dropped_tiles=dropped_tiles.sum(),
            discarded_mass_sum=discarded_mass[row_valid].double().sum(),
            discarded_mass_sq_sum=discarded_mass[row_valid].double().square().sum(),
        )

        token_drop = dropped_tiles[..., None].expand(-1, -1, -1, -1, tile)
        pruned_valid = valid & ~token_drop.reshape_as(valid)
        pruned_scores = scores.masked_fill(~pruned_valid, -torch.inf)
        output, weights = _finish_eager_attention(
            query, value, pruned_scores, pruned_valid, dropout, module.training
        )
        del masked_blocks, score_blocks, valid_blocks, importance, block_mass
        return output, weights


def _validate_samples(
    samples: list[dict[str, Any]],
    count: int,
    context_length: int,
    length_mode: str,
) -> None:
    if len(samples) != count:
        raise ValueError(f"requested {count} samples, found {len(samples)}")
    task_counts = Counter(str(row["task"]) for row in samples)
    if max(task_counts.values()) - min(task_counts.values()) > 1:
        raise ValueError(f"sample subset is not task-balanced: {task_counts}")
    for row in samples:
        length = int(row["actual_prompt_length"])
        if length_mode == "prompt":
            if not 0.93 * context_length <= length <= 1.02 * context_length:
                raise ValueError(f"{row['sample_id']} is not an 8K prompt: {length}")
        elif length_mode == "total":
            total = int(row.get("actual_total_length", length + row["tokens_to_generate"]))
            target = int(row.get("target_length", context_length))
            if target != context_length or not 0 < length < total <= 1.02 * context_length:
                raise ValueError(
                    f"{row['sample_id']} violates the paper-style 8K total budget: "
                    f"prompt={length}, total={total}"
                )
        else:
            raise ValueError("length_mode must be prompt or total")


def _condition_grid(
    k_values: list[float], methods: list[str]
) -> list[tuple[str, float]]:
    return [("dense", 0.0)] + [
        (method, value) for method in methods for value in k_values
    ]


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.samples).resolve()
    samples = _read_jsonl(samples_path)[: args.num_samples]
    _validate_samples(samples, args.num_samples, args.context_length, args.length_mode)
    k_values = [float(value) for value in args.k_values]
    methods = [str(value) for value in args.methods]
    unknown_methods = set(methods) - set(METHODS)
    if unknown_methods:
        raise ValueError(f"unsupported methods: {sorted(unknown_methods)}")
    conditions = _condition_grid(k_values, methods)
    local_model_path = Path(args.model_path)
    local_model_hashes = {}
    if local_model_path.is_dir():
        for filename in ("config.json", "model.safetensors.index.json", "latest"):
            candidate = local_model_path / filename
            if candidate.is_file():
                local_model_hashes[filename] = _sha256(candidate)
    run_config = {
        "schema_version": 1,
        "experiment": "dense-first row-wise bottom-k KV-tile pruning",
        "model_adapter": args.model_adapter,
        "model_path": args.model_path,
        "revision": args.revision,
        "samples": str(samples_path),
        "samples_sha256": _sha256(samples_path),
        "num_samples": args.num_samples,
        "context_length": args.context_length,
        "length_mode": args.length_mode,
        "q_tile_size_metadata": args.q_tile_size,
        "kv_tile_size": args.kv_tile_size,
        "k_values": k_values,
        "score_assignments": methods,
        "rounding": "floor(N*k + 0.5), capped at N-1",
        "ranking_scope": "independent valid KV tiles for every layer/head/query row",
        "oracle_dense_qk_first": True,
        "random_policy_seed": args.random_seed,
        "attention_call_eligibility": (
            "ordinary cached denoising attention only; prefix and cache commits remain dense"
            if args.model_adapter == "fast_dllm_v2"
            else "DiffusionGemma decoder canvas attention"
        ),
        "local_model_identity_sha256": local_model_hashes,
        "block_size": args.block_size,
        "threshold": args.threshold,
        "temperature": 0.0,
        "repository_commit": _repository_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "model_path", "revision", "samples_sha256", "num_samples",
            "context_length", "length_mode", "q_tile_size_metadata",
            "kv_tile_size", "k_values", "block_size", "threshold",
        ):
            if prior.get(key) != run_config.get(key):
                raise ValueError(f"existing run configuration differs at {key}")
        prior_methods = [str(value) for value in prior["score_assignments"]]
        if not set(prior_methods).issubset(methods):
            raise ValueError(
                "existing score assignments must be retained when extending a run"
            )
        if "random" in prior_methods and prior.get("random_policy_seed") != args.random_seed:
            raise ValueError("existing run configuration differs at random_policy_seed")
        if prior_methods != methods or "random_policy_seed" not in prior:
            _atomic_json(config_path, run_config)
    else:
        _atomic_json(config_path, run_config)

    adapter = create_adapter(
        args.model_adapter,
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    for sample in samples:
        actual = len(adapter.encode_prompt(str(sample["prompt"])))
        if actual != int(sample["actual_prompt_length"]):
            raise ValueError(
                f"tokenizer mismatch for {sample['sample_id']}: "
                f"{actual} != {sample['actual_prompt_length']}"
            )

    blasst_config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=0.003,
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=False,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
        apply_blasst_mask=False,
    )
    if args.model_adapter == "fast_dllm_v2":
        from dllm.attention.blasst.core import install_blasst_2d

        runtime = install_blasst_2d(
            adapter.model,
            blasst_config,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            ordinary_cache_queries_only=True,
            attention_class_names=adapter.attention_class_names,
        )

        class _FastBinding:
            def __init__(self, value: Any) -> None:
                self.runtime = value

            def close(self) -> None:
                if self.runtime.hook_handle is not None:
                    self.runtime.hook_handle.remove()
                    self.runtime.hook_handle = None

        binding = _FastBinding(runtime)
    else:
        binding = install_blasst(
            adapter.model,
            blasst_config,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            attention_class_names=adapter.attention_class_names,
            module_selector=adapter.is_blasst_attention_module,
            query_ids_extractor=adapter.blasst_query_ids,
            filter_special_query_ids=adapter.blasst_filter_special_query_ids,
            dense_kv_prefix_extractor=adapter.blasst_dense_kv_prefix,
            integration=adapter.attention_integration,
        )
    try:
        for condition_index, (method, k_fraction) in enumerate(conditions, start=1):
            name = _condition_name(method, k_fraction)
            condition_dir = output / "conditions" / name
            predictions_path = condition_dir / "predictions.jsonl"
            stats_path = condition_dir / "attention_stats.json"
            existing = _read_jsonl(predictions_path)
            completed = {str(row["sample_id"]): row for row in existing}
            if len(completed) == len(samples) and stats_path.exists():
                print(f"resume condition {condition_index}/{len(conditions)}: {name}", flush=True)
                continue
            pruner = OracleTilePruner(
                method=method,
                k_fraction=k_fraction,
                kv_tile_size=args.kv_tile_size,
                output_norm_query_chunk=args.output_norm_query_chunk,
            )
            binding.runtime.attention_override = pruner
            condition_started = time.perf_counter()
            for sample_index, sample in enumerate(samples, start=1):
                if str(sample["sample_id"]) in completed:
                    continue
                pruner.stats = OracleStats()
                pruner.set_sample_seed(int(sample["inference_seed"]))
                binding.runtime.metadata_context = {
                    "benchmark": str(sample["task"]),
                    "example_id": str(sample["sample_id"]),
                    "inference_seed": int(sample["inference_seed"]),
                }
                budget = int(sample.get("tokens_to_generate", args.max_new_tokens))
                result = adapter.generate(
                    GenerationRequest(
                        prompt=str(sample["prompt"]),
                        max_new_tokens=budget,
                        block_size=args.block_size,
                        threshold=args.threshold,
                        temperature=0.0,
                        seed=int(sample["inference_seed"]),
                    )
                )
                row = {
                    **sample,
                    "prediction": result.text,
                    "completion_tokens": result.completion_tokens,
                    "elapsed_seconds": result.elapsed_seconds,
                    "model_evaluations": result.model_evaluations,
                    "termination_reason": result.termination_reason,
                    "oracle_attention_stats": pruner.stats.summary(),
                }
                _append_jsonl(predictions_path, row)
                print(
                    f"{name} {sample_index}/{len(samples)} "
                    f"({condition_index}/{len(conditions)}): "
                    f"{sample['sample_id']} {result.elapsed_seconds:.2f}s",
                    flush=True,
                )
            combined_stats = OracleStats()
            completed_rows = _read_jsonl(predictions_path)
            if len(completed_rows) != len(samples):
                raise RuntimeError(
                    f"{name} has {len(completed_rows)} predictions, expected {len(samples)}"
                )
            for row in completed_rows:
                combined_stats.add(
                    OracleStats.from_summary(row["oracle_attention_stats"])
                )
            _atomic_json(
                stats_path,
                {
                    "method": method,
                    "k_fraction": k_fraction,
                    "wall_seconds": time.perf_counter() - condition_started,
                    **combined_stats.summary(),
                },
            )
    finally:
        binding.runtime.attention_override = None
        binding.close()
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _bootstrap_interval(
    values: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    seed: int = 20260811,
    repeats: int = 20000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if strata is None:
        indices = rng.integers(0, len(values), size=(repeats, len(values)))
        means = values[indices].mean(axis=1)
    else:
        if len(strata) != len(values):
            raise ValueError("bootstrap strata must align with values")
        totals = np.zeros(repeats, dtype=np.float64)
        for label in np.unique(strata):
            positions = np.flatnonzero(strata == label)
            sampled = rng.integers(0, len(positions), size=(repeats, len(positions)))
            totals += values[positions[sampled]].sum(axis=1)
        means = totals / len(values)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    samples = _read_jsonl(Path(config["samples"]))[: int(config["num_samples"])]
    methods = [str(value) for value in config["score_assignments"]]
    conditions = _condition_grid(
        [float(value) for value in config["k_values"]], methods
    )
    rows = []
    sample_scores: dict[str, dict[str, float]] = {}
    task_strata = np.asarray([str(sample["task"]) for sample in samples])
    for method, k_fraction in conditions:
        name = _condition_name(method, k_fraction)
        condition_dir = output / "conditions" / name
        predictions = _read_jsonl(condition_dir / "predictions.jsonl")
        if len(predictions) != len(samples):
            raise ValueError(f"{name} has {len(predictions)} predictions, expected {len(samples)}")
        per_task, accuracy = score_predictions(predictions, args.ruler_root)
        individual = {
            str(row["sample_id"]): score_predictions([row], args.ruler_root)[1]
            for row in predictions
        }
        sample_scores[name] = individual
        ordered_scores = np.asarray(
            [individual[str(sample["sample_id"])] for sample in samples], dtype=np.float64
        )
        low, high = _bootstrap_interval(ordered_scores, strata=task_strata)
        stats = json.loads((condition_dir / "attention_stats.json").read_text(encoding="utf-8"))
        overall_stats = stats["overall"]
        rows.append(
            {
                "condition": name,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "k_fraction": k_fraction,
                "official_ruler_accuracy": accuracy,
                "accuracy_bootstrap_95_low": low,
                "accuracy_bootstrap_95_high": high,
                "accuracy_delta_vs_dense": 0.0,
                "actual_tile_drop_fraction": overall_stats["actual_tile_drop_fraction"],
                "discarded_dense_attention_mass_mean": overall_stats[
                    "discarded_dense_attention_mass_mean"
                ],
                "per_task_accuracy": per_task,
                "local": stats["by_attention_type"].get("local", StatBucket().summary()),
                "global": stats["by_attention_type"].get("global", StatBucket().summary()),
                "wall_seconds": stats["wall_seconds"],
            }
        )
    dense = next(row for row in rows if row["condition"] == "dense_eager")
    dense_scores = sample_scores["dense_eager"]
    for index, row in enumerate(rows):
        row["accuracy_delta_vs_dense"] = row["official_ruler_accuracy"] - dense[
            "official_ruler_accuracy"
        ]
        if row["condition"] == "dense_eager":
            row["paired_delta_bootstrap_95_low"] = 0.0
            row["paired_delta_bootstrap_95_high"] = 0.0
            continue
        deltas = np.asarray(
            [
                sample_scores[row["condition"]][str(sample["sample_id"])]
                - dense_scores[str(sample["sample_id"])]
                for sample in samples
            ]
        )
        low, high = _bootstrap_interval(
            deltas, strata=task_strata, seed=20260811 + index
        )
        row["paired_delta_bootstrap_95_low"] = low
        row["paired_delta_bootstrap_95_high"] = high

    _atomic_json(output / "summary.json", {"schema_version": 1, "conditions": rows})
    with (output / "accuracy_tradeoff.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "condition", "method", "k_fraction", "official_ruler_accuracy",
            "accuracy_delta_vs_dense", "accuracy_bootstrap_95_low",
            "accuracy_bootstrap_95_high", "paired_delta_bootstrap_95_low",
            "paired_delta_bootstrap_95_high", "actual_tile_drop_fraction",
            "discarded_dense_attention_mass_mean",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"block_max": "#0072B2", "random": "#CC79A7", "attention_mass": "#D55E00", "output_norm": "#009E73"}
    markers = {"block_max": "o", "random": "x", "attention_mass": "s", "output_norm": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda row: row["k_fraction"])
        x = [100 * row["k_fraction"] for row in subset]
        axes[0].plot(
            x, [100 * row["official_ruler_accuracy"] for row in subset],
            marker=markers[method], color=colors[method], label=METHOD_LABELS[method],
        )
        axes[1].plot(
            [100 * row["discarded_dense_attention_mass_mean"] for row in subset],
            [100 * row["official_ruler_accuracy"] for row in subset],
            marker=markers[method], color=colors[method], label=METHOD_LABELS[method],
        )
    for axis in axes:
        axis.axhline(100 * dense["official_ruler_accuracy"], color="0.35", lw=1, ls="--")
        axis.grid(alpha=0.25)
        axis.set_ylabel("Official RULER accuracy (%)")
    axes[0].set_xlabel("Requested bottom-k KV tiles dropped (%)")
    axes[1].set_xlabel("Mean dense attention mass discarded (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("DiffusionGemma 8K RULER: oracle row-wise KV-tile pruning")
    fig.tight_layout()
    fig.savefig(output / "accuracy_tradeoff.png", dpi=180)
    fig.savefig(output / "accuracy_tradeoff.pdf")
    plt.close(fig)

    lines = [
        "# Oracle row-wise bottom-k KV-tile pruning",
        "",
        f"Dense-first experiment on {len(samples)} balanced 8K RULER samples. "
        "Each layer/head/query independently ranks (or randomly samples) structurally valid kv64 tiles, "
        "drops `floor(kN+0.5)` (never all tiles), and renormalizes attention. "
        "Dense QK and ranking are performed, so this is an accuracy-potential study, not a speed benchmark.",
        "",
        "| Score assignment | k | Actual tile drop | Dense mass discarded | Accuracy | Δ vs dense | Paired 95% CI for Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {100*row['k_fraction']:.0f}% | "
            f"{100*row['actual_tile_drop_fraction']:.1f}% | "
            f"{100*row['discarded_dense_attention_mass_mean']:.2f}% | "
            f"{100*row['official_ruler_accuracy']:.2f}% | "
            f"{100*row['accuracy_delta_vs_dense']:+.2f} pp | "
            f"[{100*row['paired_delta_bootstrap_95_low']:+.2f}, "
            f"{100*row['paired_delta_bootstrap_95_high']:+.2f}] pp |"
        )
    task_order = sorted(dense["per_task_accuracy"])
    lines.extend([
        "",
        "## Accuracy by RULER task",
        "",
        "| Score assignment | k | " + " | ".join(task_order) + " |",
        "|---|---:|" + "---:|" * len(task_order),
    ])
    for row in rows:
        task_cells = " | ".join(
            f"{100*row['per_task_accuracy'][task]:.0f}%" for task in task_order
        )
        lines.append(
            f"| {row['method_label']} | {100*row['k_fraction']:.0f}% | {task_cells} |"
        )
    lines.extend([
        "",
        "## Discarded mass by attention type",
        "",
        "| Score assignment | k | Local tiles dropped | Local mass discarded | Global tiles dropped | Global mass discarded |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows[1:]:
        lines.append(
            f"| {row['method_label']} | {100*row['k_fraction']:.0f}% | "
            f"{100*row['local']['actual_tile_drop_fraction']:.1f}% | "
            f"{100*row['local']['discarded_dense_attention_mass_mean']:.2f}% | "
            f"{100*row['global']['actual_tile_drop_fraction']:.1f}% | "
            f"{100*row['global']['discarded_dense_attention_mass_mean']:.2f}% |"
        )
    lines.extend([
        "",
        f"The bootstrap intervals resample the {len(samples)} sample-level official scores within task and are descriptive; "
        "this small benchmark subset should not be read as a high-powered significance test.",
        "",
        "![Accuracy tradeoff](accuracy_tradeoff.png)",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model-adapter", default="diffusion_gemma")
    run_parser.add_argument("--model-path", default="google/diffusiongemma-26B-A4B-it")
    run_parser.add_argument("--revision", default="f7f5b7f5fa82ffc52addd066915886d497f5517b")
    run_parser.add_argument("--samples", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--num-samples", type=int, default=20)
    run_parser.add_argument("--context-length", type=int, default=8192)
    run_parser.add_argument("--length-mode", choices=("prompt", "total"), default="prompt")
    run_parser.add_argument("--q-tile-size", type=int, default=32)
    run_parser.add_argument("--kv-tile-size", type=int, default=64)
    run_parser.add_argument("--k-values", nargs="+", type=float, default=DEFAULT_K_VALUES)
    run_parser.add_argument("--methods", nargs="+", default=METHODS)
    run_parser.add_argument("--block-size", type=int, default=16)
    run_parser.add_argument("--threshold", type=float, default=0.9)
    run_parser.add_argument("--max-new-tokens", type=int, default=128)
    run_parser.add_argument("--output-norm-query-chunk", type=int, default=32)
    run_parser.add_argument("--random-seed", type=int, default=20260812)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--precision", default="bfloat16")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--output-dir", required=True)
    report_parser.add_argument("--ruler-root", default="/tmp/NVIDIA-RULER")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
