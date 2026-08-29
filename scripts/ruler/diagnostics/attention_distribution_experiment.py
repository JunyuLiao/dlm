#!/usr/bin/env python3
"""Dense 8K RULER attention-distribution experiment for BLASST analysis.

The collector treats every valid query row in every attention head and layer as
one sample.  It records shift-invariant block-max gaps, exact block softmax
mass concentration, and the q-tile-wide minimum gap that reproduces BLASST's
physical all-row vote.  Dense outputs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from dllm.attention.blasst import Blasst2DConfig, install_blasst
from dllm.attention.blasst.core import install_blasst_2d
from dllm.models import GenerationRequest, create_adapter


GAP_EDGES = np.concatenate((np.linspace(0.0, 20.0, 401), [np.inf]))
BLOCK_FRACTIONS = np.linspace(0.01, 1.0, 100)
RANK_FRACTIONS = np.linspace(0.0, 1.0, 101)
LAMBDAS = (0.003, 0.01, 0.1, 0.5)
GROUP_ORDER = ("diffusion_gemma_local", "diffusion_gemma_global", "fast_dllm_v2")
GROUP_LABELS = {
    "diffusion_gemma_local": "DiffusionGemma local",
    "diffusion_gemma_global": "DiffusionGemma global",
    "fast_dllm_v2": "Fast-dLLM v2 7B",
}
COLORS = {
    "diffusion_gemma_local": "#D55E00",
    "diffusion_gemma_global": "#CC79A7",
    "fast_dllm_v2": "#0072B2",
}


def _zeros(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.float64)


@dataclass
class Bucket:
    calls: int = 0
    queries: int = 0
    block_pairs: int = 0
    physical_tiles: int = 0
    online_query_hist: np.ndarray = field(default_factory=lambda: _zeros(len(GAP_EDGES) - 1))
    online_pair_hist: np.ndarray = field(default_factory=lambda: _zeros(len(GAP_EDGES) - 1))
    global_query_hist: np.ndarray = field(default_factory=lambda: _zeros(len(GAP_EDGES) - 1))
    physical_hist: np.ndarray = field(default_factory=lambda: _zeros(len(GAP_EDGES) - 1))
    concentration_sum: np.ndarray = field(default_factory=lambda: _zeros(len(BLOCK_FRACTIONS)))
    concentration_sq: np.ndarray = field(default_factory=lambda: _zeros(len(BLOCK_FRACTIONS)))
    rank_gap_sum: np.ndarray = field(default_factory=lambda: _zeros(len(RANK_FRACTIONS)))
    rank_gap_sq: np.ndarray = field(default_factory=lambda: _zeros(len(RANK_FRACTIONS)))
    metric_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    metric_sq: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    raw_logit_count: int = 0
    raw_logit_sum: float = 0.0
    raw_logit_sq: float = 0.0
    raw_logit_min: float = math.inf
    raw_logit_max: float = -math.inf

    def add(self, other: "Bucket") -> None:
        self.calls += other.calls
        self.queries += other.queries
        self.block_pairs += other.block_pairs
        self.physical_tiles += other.physical_tiles
        for name in (
            "online_query_hist",
            "online_pair_hist",
            "global_query_hist",
            "physical_hist",
            "concentration_sum",
            "concentration_sq",
            "rank_gap_sum",
            "rank_gap_sq",
        ):
            getattr(self, name)[:] += getattr(other, name)
        for name, value in other.metric_sum.items():
            self.metric_sum[name] += value
        for name, value in other.metric_sq.items():
            self.metric_sq[name] += value
        self.raw_logit_count += other.raw_logit_count
        self.raw_logit_sum += other.raw_logit_sum
        self.raw_logit_sq += other.raw_logit_sq
        self.raw_logit_min = min(self.raw_logit_min, other.raw_logit_min)
        self.raw_logit_max = max(self.raw_logit_max, other.raw_logit_max)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name, value in list(payload.items()):
            if isinstance(value, np.ndarray):
                payload[name] = value.tolist()
        payload["metric_sum"] = dict(self.metric_sum)
        payload["metric_sq"] = dict(self.metric_sq)
        if not math.isfinite(payload["raw_logit_min"]):
            payload["raw_logit_min"] = None
            payload["raw_logit_max"] = None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Bucket":
        result = cls()
        for name, value in payload.items():
            if name in (
                "online_query_hist",
                "online_pair_hist",
                "global_query_hist",
                "physical_hist",
                "concentration_sum",
                "concentration_sq",
                "rank_gap_sum",
                "rank_gap_sq",
            ):
                setattr(result, name, np.asarray(value, dtype=np.float64))
            elif name in ("metric_sum", "metric_sq"):
                setattr(result, name, defaultdict(float, value))
            elif name in ("raw_logit_min", "raw_logit_max") and value is None:
                setattr(result, name, math.inf if name.endswith("min") else -math.inf)
            else:
                setattr(result, name, value)
        return result


def _histogram(values: torch.Tensor, weights: torch.Tensor | None = None) -> np.ndarray:
    boundaries = torch.as_tensor(
        GAP_EDGES[1:-1], device=values.device, dtype=values.dtype
    )
    indices = torch.bucketize(values, boundaries)
    target = torch.zeros(
        len(GAP_EDGES) - 1, device=values.device, dtype=torch.float64
    )
    if weights is None:
        weights = torch.ones_like(values, dtype=torch.float64)
    else:
        weights = weights.to(torch.float64)
    target.scatter_add_(0, indices, weights)
    return target.cpu().numpy()


def _query_metrics(
    gap_global: torch.Tensor,
    block_mass: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    counts = eligible.sum(-1)
    safe_counts = counts.clamp_min(1)
    gaps = gap_global.masked_fill(~eligible, torch.inf)
    sorted_gaps = gaps.sort(dim=-1).values
    rank_positions = torch.floor(
        torch.as_tensor(RANK_FRACTIONS, device=gaps.device)[:, None]
        * (safe_counts[None, :] - 1)
    ).to(torch.long).transpose(0, 1)
    rank_curve = sorted_gaps.gather(-1, rank_positions)

    masses = block_mass.masked_fill(~eligible, 0.0)
    sorted_mass = masses.sort(dim=-1, descending=True).values
    cumulative = sorted_mass.cumsum(-1)
    concentration_positions = (
        torch.ceil(
            torch.as_tensor(BLOCK_FRACTIONS, device=gaps.device)[:, None]
            * safe_counts[None, :]
        ).to(torch.long) - 1
    ).clamp_min(0).transpose(0, 1)
    concentration = cumulative.gather(-1, concentration_positions)

    log_count = safe_counts.float().log().clamp_min(1.0e-12)
    entropy = -(masses * masses.clamp_min(1.0e-30).log()).sum(-1)
    valid_gaps = gap_global.masked_fill(~eligible, 0.0)
    gap_mean = valid_gaps.sum(-1) / safe_counts
    gap_var = (
        ((valid_gaps - gap_mean[:, None]) ** 2 * eligible).sum(-1)
        / safe_counts
    )
    metrics = {
        "normalized_block_entropy": entropy / log_count,
        "effective_block_fraction": entropy.exp() / safe_counts,
        "max_block_mass": masses.max(-1).values,
        "mean_global_gap": gap_mean,
        "std_global_gap": gap_var.clamp_min(0.0).sqrt(),
    }
    return metrics, concentration, rank_curve


class AttentionDistributionCollector:
    def __init__(self, model_label: str) -> None:
        self.model_label = model_label
        self.buckets: dict[str, Bucket] = {}
        self.completed_samples: list[str] = []
        self.sample_metadata: dict[str, dict[str, Any]] = {}
        self.wall_seconds = 0.0

    def _group(self, metadata: dict[str, Any]) -> str:
        if self.model_label == "fast_dllm_v2":
            return "fast_dllm_v2"
        attention_type = str(metadata.get("attention_type", "global"))
        return f"diffusion_gemma_{attention_type}"

    @staticmethod
    def _mask_bin(metadata: dict[str, Any]) -> str:
        value = float(metadata.get("mask_ratio", -1.0))
        if value < 0.0:
            return "unknown"
        if value < 1.0 / 3.0:
            return "late"
        if value < 2.0 / 3.0:
            return "middle"
        return "early"

    @torch.no_grad()
    def observe(
        self,
        *,
        scores: torch.Tensor,
        valid_pair_mask: torch.Tensor,
        active_query_mask: torch.Tensor | None,
        layer: int,
        metadata: dict[str, Any],
        q_tile_size: int,
        kv_tile_size: int,
        sparse_kv_start: int,
    ) -> None:
        if scores.ndim != 4:
            raise ValueError("attention scores must be [batch, head, query, key]")
        batch, heads, q_len, kv_len = scores.shape
        valid = torch.broadcast_to(
            valid_pair_mask.to(device=scores.device, dtype=torch.bool), scores.shape
        )
        if active_query_mask is None:
            active = torch.ones(
                (batch, heads, q_len), device=scores.device, dtype=torch.bool
            )
        else:
            active = active_query_mask.to(device=scores.device, dtype=torch.bool)
            if active.ndim == 2:
                active = active[:, None, :]
            active = torch.broadcast_to(active, (batch, heads, q_len))

        kv_tiles = math.ceil(kv_len / kv_tile_size)
        kv_pad = kv_tiles * kv_tile_size - kv_len
        score32 = scores.float()
        if kv_pad:
            score32 = torch.nn.functional.pad(score32, (0, kv_pad), value=-torch.inf)
            valid = torch.nn.functional.pad(valid, (0, kv_pad), value=False)
        score_blocks = score32.reshape(batch, heads, q_len, kv_tiles, kv_tile_size)
        valid_blocks = valid.reshape(batch, heads, q_len, kv_tiles, kv_tile_size)
        masked_blocks = score_blocks.masked_fill(~valid_blocks, -torch.inf)
        local_max = masked_blocks.amax(-1)
        block_lse = torch.logsumexp(masked_blocks, dim=-1)
        del masked_blocks, score_blocks, valid_blocks, score32

        eligible_tiles = (
            torch.arange(kv_tiles, device=scores.device) * kv_tile_size
            >= int(sparse_kv_start)
        )
        eligible = torch.isfinite(local_max) & active[..., None] & eligible_tiles
        row_valid = eligible.any(-1)
        if not bool(row_valid.any()):
            return
        running_max = torch.cummax(local_max, dim=-1).values
        final_max = local_max.amax(-1, keepdim=True)
        online_gap = running_max - local_max
        global_gap = final_max - local_max
        row_lse = torch.logsumexp(block_lse, dim=-1, keepdim=True)
        block_mass = torch.exp(block_lse - row_lse)

        flat_eligible = eligible[row_valid]
        flat_online = online_gap[row_valid]
        flat_global = global_gap[row_valid]
        flat_mass = block_mass[row_valid]
        flat_local_max = local_max[row_valid]
        counts = flat_eligible.sum(-1)
        values_online = flat_online[flat_eligible]
        values_global = flat_global[flat_eligible]
        per_pair_weights = (
            counts.float().reciprocal().to(torch.float64)[:, None]
            .expand_as(flat_online)[flat_eligible]
        )

        partial = Bucket(calls=1, queries=int(row_valid.sum().item()))
        partial.block_pairs = int(flat_eligible.sum().item())
        partial.online_query_hist = _histogram(values_online, per_pair_weights)
        partial.online_pair_hist = _histogram(values_online)
        partial.global_query_hist = _histogram(values_global, per_pair_weights)

        q_tiles = math.ceil(q_len / q_tile_size)
        q_pad = q_tiles * q_tile_size - q_len
        physical_gap_source = online_gap.masked_fill(~eligible, torch.inf)
        physical_eligible_source = eligible
        if q_pad:
            physical_gap_source = torch.nn.functional.pad(
                physical_gap_source, (0, 0, 0, q_pad), value=torch.inf
            )
            physical_eligible_source = torch.nn.functional.pad(
                physical_eligible_source, (0, 0, 0, q_pad), value=False
            )
        physical_gap = physical_gap_source.reshape(
            batch, heads, q_tiles, q_tile_size, kv_tiles
        ).amin(-2)
        physical_eligible = physical_eligible_source.reshape(
            batch, heads, q_tiles, q_tile_size, kv_tiles
        ).any(-2)
        physical_values = physical_gap[physical_eligible]
        partial.physical_tiles = int(physical_values.numel())
        partial.physical_hist = _histogram(physical_values)

        metrics, concentration, rank_curve = _query_metrics(
            flat_global, flat_mass, flat_eligible
        )
        partial.concentration_sum = concentration.sum(0).double().cpu().numpy()
        partial.concentration_sq = (concentration.double() ** 2).sum(0).cpu().numpy()
        partial.rank_gap_sum = rank_curve.sum(0).double().cpu().numpy()
        partial.rank_gap_sq = (rank_curve.double() ** 2).sum(0).cpu().numpy()
        for name, tensor in metrics.items():
            partial.metric_sum[name] = float(tensor.double().sum().item())
            partial.metric_sq[name] = float((tensor.double() ** 2).sum().item())
        for lambda_value in LAMBDAS:
            lambda_name = str(lambda_value).replace(".", "p")
            threshold = -math.log(lambda_value)
            row_fractions = (
                ((flat_online > threshold) & flat_eligible).sum(-1)
                / counts
            )
            partial.metric_sum[f"row_vote_lambda_{lambda_name}"] = float(
                row_fractions.double().sum().item()
            )
            partial.metric_sq[f"row_vote_lambda_{lambda_name}"] = float(
                (row_fractions.double() ** 2).sum().item()
            )
            partial.metric_sum[f"physical_skip_lambda_{lambda_name}"] = float(
                (physical_values > threshold).sum().item()
            )

        raw_values = flat_local_max[flat_eligible]
        partial.raw_logit_count = int(raw_values.numel())
        partial.raw_logit_sum = float(raw_values.double().sum().item())
        partial.raw_logit_sq = float((raw_values.double() ** 2).sum().item())
        partial.raw_logit_min = float(raw_values.min().item())
        partial.raw_logit_max = float(raw_values.max().item())

        group = self._group(metadata)
        sample_id = str(metadata.get("example_id", "unknown"))
        mask_bin = self._mask_bin(metadata)
        keys = (
            f"overall|{group}",
            f"sample|{group}|{sample_id}",
            f"layer|{group}|{layer}",
            f"mask_bin|{group}|{mask_bin}",
        )
        for key in keys:
            self.buckets.setdefault(key, Bucket()).add(partial)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_label": self.model_label,
            "gap_edges": [
                float(value) if math.isfinite(value) else None
                for value in GAP_EDGES
            ],
            "block_fractions": BLOCK_FRACTIONS.tolist(),
            "rank_fractions": RANK_FRACTIONS.tolist(),
            "completed_samples": self.completed_samples,
            "sample_metadata": self.sample_metadata,
            "wall_seconds": self.wall_seconds,
            "buckets": {key: value.to_dict() for key, value in self.buckets.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttentionDistributionCollector":
        result = cls(str(payload["model_label"]))
        result.completed_samples = list(payload.get("completed_samples", []))
        result.sample_metadata = dict(payload.get("sample_metadata", {}))
        result.wall_seconds = float(payload.get("wall_seconds", 0.0))
        result.buckets = {
            key: Bucket.from_dict(value) for key, value in payload["buckets"].items()
        }
        return result


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def collect(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "raw_statistics.json"
    samples_path = Path(args.samples).resolve()
    all_samples = _read_jsonl(samples_path)
    samples = all_samples[: args.num_samples]
    if len(samples) != args.num_samples:
        raise ValueError(f"requested {args.num_samples} samples, found {len(samples)}")
    task_counts = Counter(str(row["task"]) for row in samples)
    if max(task_counts.values()) - min(task_counts.values()) > 1:
        raise ValueError(f"RULER subset is not balanced: {task_counts}")

    collector = (
        AttentionDistributionCollector.from_dict(
            json.loads(state_path.read_text(encoding="utf-8"))
        )
        if state_path.exists()
        else AttentionDistributionCollector(args.model_label)
    )
    if collector.model_label != args.model_label:
        raise ValueError("existing output belongs to a different model")

    started = time.perf_counter()
    adapter = create_adapter(
        args.model_adapter,
        args.model_path,
        device=args.device,
        precision=args.precision,
        revision=args.revision,
    ).load()
    config = Blasst2DConfig(
        enable_blasst_2d=True,
        blasst_lambda=0.003,
        q_tile_size=args.q_tile_size,
        kv_tile_size=args.kv_tile_size,
        collect_blasst_stats=False,
        collect_blasst_layer_stats=False,
        collect_blasst_head_stats=False,
        apply_blasst_mask=False,
    )
    if args.model_label == "fast_dllm_v2":
        # Match the eligibility rule used by the original Fast-dLLM BLASST
        # experiments: observe only ordinary cached denoising queries, not the
        # dense prefix-prefill or cache-commit forwards.
        runtime = install_blasst_2d(
            adapter.model,
            config,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            ordinary_cache_queries_only=True,
            sweep_lambdas=(0.003,),
            attention_class_names=adapter.attention_class_names,
        )

        class _Binding:
            def __init__(self, value):
                self.runtime = value

            def close(self) -> None:
                if self.runtime.hook_handle is not None:
                    self.runtime.hook_handle.remove()

        binding = _Binding(runtime)
    else:
        binding = install_blasst(
            adapter.model,
            config,
            mask_token_id=adapter.mask_token_id,
            pad_token_id=adapter.pad_token_id,
            attention_class_names=adapter.attention_class_names,
            module_selector=adapter.is_blasst_attention_module,
            query_ids_extractor=adapter.blasst_query_ids,
            filter_special_query_ids=adapter.blasst_filter_special_query_ids,
            dense_kv_prefix_extractor=adapter.blasst_dense_kv_prefix,
            sweep_lambdas=(0.003,),
            integration=adapter.attention_integration,
        )
    binding.runtime.attention_observer = collector.observe

    try:
        for index, sample in enumerate(samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in collector.completed_samples:
                print(f"resume {index}/{len(samples)}: {sample_id}", flush=True)
                continue
            prompt_ids = adapter.encode_prompt(str(sample["prompt"]))
            declared = int(sample.get("actual_prompt_length", len(prompt_ids)))
            if len(prompt_ids) != declared:
                raise ValueError(
                    f"tokenizer mismatch for {sample_id}: {len(prompt_ids)} != {declared}"
                )
            if not 0.93 * args.context_length <= len(prompt_ids) <= 1.02 * args.context_length:
                raise ValueError(f"{sample_id} is not an 8K-context prompt: {len(prompt_ids)}")
            binding.runtime.metadata_context = {
                "benchmark": str(sample["task"]),
                "example_id": sample_id,
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
            collector.sample_metadata[sample_id] = {
                "task": str(sample["task"]),
                "actual_prompt_length": len(prompt_ids),
                "generation_budget": budget,
                "completion_tokens": len(result.completion_tokens),
                "model_evaluations": result.model_evaluations,
                "elapsed_seconds": result.elapsed_seconds,
            }
            collector.completed_samples.append(sample_id)
            collector.wall_seconds += time.perf_counter() - started
            started = time.perf_counter()
            _atomic_json(state_path, collector.to_dict())
            print(
                f"completed {index}/{len(samples)}: {sample_id} "
                f"({result.elapsed_seconds:.2f}s generation)",
                flush=True,
            )
    finally:
        binding.close()
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_config = {
        "schema_version": 1,
        "model_label": args.model_label,
        "model_adapter": args.model_adapter,
        "model_path": args.model_path,
        "revision": args.revision,
        "samples_path": str(samples_path),
        "samples_sha256": _sha256(samples_path),
        "num_samples": args.num_samples,
        "task_counts": dict(task_counts),
        "context_length": args.context_length,
        "q_tile_size": args.q_tile_size,
        "kv_tile_size": args.kv_tile_size,
        "generation_block_size": args.block_size,
        "default_max_new_tokens": args.max_new_tokens,
        "precision": args.precision,
        "device": args.device,
        "dense_outputs_unmodified": True,
        "attention_call_eligibility": (
            "ordinary cached denoising queries only"
            if args.model_label == "fast_dllm_v2"
            else "DiffusionGemma decoder canvas attention"
        ),
        "query_sample_definition": "one valid token query in one attention head and layer",
        "repository_commit": _repository_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "wall_seconds": collector.wall_seconds,
    }
    _atomic_json(output / "run_config.json", run_config)


def _survival(hist: np.ndarray) -> np.ndarray:
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return np.cumsum(hist[::-1])[::-1] / total


def _value_at_gap(hist: np.ndarray, gap: float) -> float:
    index = int(np.searchsorted(GAP_EDGES[1:], gap, side="right"))
    index = min(index, len(hist) - 1)
    return float(hist[index:].sum() / hist.sum()) if hist.sum() else math.nan


def _metric(bucket: Bucket, name: str) -> tuple[float, float]:
    count = bucket.queries
    if not count:
        return math.nan, math.nan
    mean = bucket.metric_sum[name] / count
    variance = max(0.0, bucket.metric_sq[name] / count - mean * mean)
    return mean, math.sqrt(variance)


def _exact_sparsity(bucket: Bucket, lambda_value: float, physical: bool) -> float:
    name = str(lambda_value).replace(".", "p")
    if physical:
        numerator = bucket.metric_sum[f"physical_skip_lambda_{name}"]
        denominator = bucket.physical_tiles
    else:
        numerator = bucket.metric_sum[f"row_vote_lambda_{name}"]
        denominator = bucket.queries
    return numerator / denominator if denominator else math.nan


def _bootstrap_curves(
    buckets: list[Bucket], extractor, *, seed: int = 42, draws: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = np.stack([extractor(bucket) for bucket in buckets])
    mean = curves.mean(0)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(curves), size=(draws, len(curves)))
    boot = curves[choices].mean(1)
    return mean, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def _bootstrap_scalar(values: list[float], seed: int = 42) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(array), size=(10000, len(array)))
    means = array[choices].mean(1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states = [
        AttentionDistributionCollector.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        for path in args.states
    ]
    buckets: dict[str, Bucket] = {}
    sample_metadata: dict[str, dict[str, Any]] = {}
    for state in states:
        buckets.update(state.buckets)
        sample_metadata.update(state.sample_metadata)
    missing = [group for group in GROUP_ORDER if f"overall|{group}" not in buckets]
    if missing:
        raise ValueError(f"missing result groups: {missing}")

    sample_buckets: dict[str, list[Bucket]] = {}
    for group in GROUP_ORDER:
        prefix = f"sample|{group}|"
        sample_buckets[group] = [
            value for key, value in buckets.items() if key.startswith(prefix)
        ]
        if len(sample_buckets[group]) != 20:
            raise ValueError(f"{group} has {len(sample_buckets[group])} samples, expected 20")

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    x_gap = GAP_EDGES[:-1]

    for group in GROUP_ORDER:
        color = COLORS[group]
        label = GROUP_LABELS[group]
        mean, low, high = _bootstrap_curves(
            sample_buckets[group], lambda b: _survival(b.online_query_hist)
        )
        axes[0, 0].plot(x_gap, mean, color=color, label=label, linewidth=2)
        axes[0, 0].fill_between(x_gap, low, high, color=color, alpha=0.14)

        mean, low, high = _bootstrap_curves(
            sample_buckets[group], lambda b: _survival(b.physical_hist)
        )
        axes[0, 1].plot(x_gap, mean, color=color, label=label, linewidth=2)
        axes[0, 1].fill_between(x_gap, low, high, color=color, alpha=0.14)

        mean, low, high = _bootstrap_curves(
            sample_buckets[group], lambda b: b.concentration_sum / b.queries
        )
        axes[1, 0].plot(BLOCK_FRACTIONS, mean, color=color, label=label, linewidth=2)
        axes[1, 0].fill_between(BLOCK_FRACTIONS, low, high, color=color, alpha=0.14)

        mean, low, high = _bootstrap_curves(
            sample_buckets[group], lambda b: b.rank_gap_sum / b.queries
        )
        axes[1, 1].plot(RANK_FRACTIONS, mean, color=color, label=label, linewidth=2)
        axes[1, 1].fill_between(RANK_FRACTIONS, low, high, color=color, alpha=0.14)

    for axis in axes[0]:
        for value, linestyle in ((-math.log(0.5), ":"), (-math.log(0.003), "--")):
            axis.axvline(value, color="#555555", linestyle=linestyle, linewidth=1)
        axis.set_xlim(0, 12)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("BLASST online gap $G^{run}_{ij}$")
        axis.grid(alpha=0.2)
    axes[0, 0].set_title("A. Equal-query survival of online block gaps")
    axes[0, 0].set_ylabel("$P(G^{run}_{ij} > g)$")
    axes[0, 1].set_title("B. Physical q32×kv64 tile survival")
    axes[0, 1].set_ylabel("Predicted physical sparsity at threshold $g$")
    axes[0, 1].text(-math.log(0.003) + 0.1, 0.02, "$\\lambda=0.003$", rotation=90, va="bottom")
    axes[0, 0].text(-math.log(0.5) + 0.1, 0.02, "$\\lambda=0.5$", rotation=90, va="bottom")

    axes[1, 0].plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1)
    axes[1, 0].set_title("C. Attention concentration across KV blocks")
    axes[1, 0].set_xlabel("Top fraction of blocks, ranked by softmax mass")
    axes[1, 0].set_ylabel("Cumulative attention mass")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1.01)
    axes[1, 0].grid(alpha=0.2)
    axes[1, 1].set_title("D. Mean within-query block-max profile")
    axes[1, 1].set_xlabel("Block rank percentile (most → least attended)")
    axes[1, 1].set_ylabel("Global gap $m_i-\\tilde m_{ij}$")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Dense attention distributions on 20 balanced RULER samples at 8K context\nShading: 95% sample-level bootstrap CI; logits are BF16 QK scores analyzed in FP32", fontsize=13)
    fig.savefig(output / "attention_distribution_comparison.png", bbox_inches="tight")
    fig.savefig(output / "attention_distribution_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    layer_groups: dict[str, list[tuple[int, Bucket]]] = {}
    for group in GROUP_ORDER:
        prefix = f"layer|{group}|"
        layer_groups[group] = sorted(
            (int(key.split("|")[-1]), value)
            for key, value in buckets.items()
            if key.startswith(prefix)
        )
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True, sharex=True)
    for axis, group in zip(axes, GROUP_ORDER):
        layers = layer_groups[group]
        matrix = np.full((layers[-1][0] + 1, len(GAP_EDGES) - 1), np.nan)
        for layer, bucket in layers:
            matrix[layer] = _survival(bucket.online_query_hist)
        color_map = matplotlib.colormaps["viridis"].copy()
        color_map.set_bad("#EEEEEE")
        image = axis.imshow(
            matrix[:, :241], aspect="auto", origin="lower", interpolation="nearest",
            extent=(0, 12, -0.5, layers[-1][0] + 0.5),
            norm=matplotlib.colors.LogNorm(vmin=1.0e-3, vmax=1.0), cmap=color_map,
        )
        axis.axvline(-math.log(0.003), color="white", linestyle="--", linewidth=1)
        axis.set_title(GROUP_LABELS[group])
        axis.set_xlabel("Online gap threshold $g$")
        axis.set_ylabel("Layer index")
    fig.colorbar(image, ax=axes, label="$P(G^{run}_{ij}>g)$$ within layer", shrink=0.85)
    fig.suptitle("Layer heterogeneity of dense attention block gaps")
    fig.savefig(output / "layer_gap_survival_heatmap.png", bbox_inches="tight")
    fig.savefig(output / "layer_gap_survival_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for group in GROUP_ORDER:
        bucket = buckets[f"overall|{group}"]
        entropy_mean, entropy_std = _metric(bucket, "normalized_block_entropy")
        effective_mean, effective_std = _metric(bucket, "effective_block_fraction")
        max_mass_mean, max_mass_std = _metric(bucket, "max_block_mass")
        row = {
            "group": group,
            "samples": len(sample_buckets[group]),
            "attention_calls": bucket.calls,
            "query_rows": bucket.queries,
            "query_block_pairs": bucket.block_pairs,
            "physical_q32_kv64_tiles": bucket.physical_tiles,
            "normalized_block_entropy_mean": entropy_mean,
            "normalized_block_entropy_std_over_queries": entropy_std,
            "effective_block_fraction_mean": effective_mean,
            "effective_block_fraction_std_over_queries": effective_std,
            "max_block_mass_mean": max_mass_mean,
            "max_block_mass_std_over_queries": max_mass_std,
        }
        for value in LAMBDAS:
            name = str(value).replace(".", "p")
            row[f"row_vote_sparsity_lambda_{name}"] = _exact_sparsity(
                bucket, value, False
            )
            row[f"physical_sparsity_lambda_{name}"] = _exact_sparsity(
                bucket, value, True
            )
        sample_physical = [
            _exact_sparsity(sample_bucket, 0.003, True)
            for sample_bucket in sample_buckets[group]
        ]
        ci_low, ci_high = _bootstrap_scalar(sample_physical)
        row["physical_sparsity_lambda_0p003_sample_mean"] = float(
            np.mean(sample_physical)
        )
        row["physical_sparsity_lambda_0p003_ci95_low"] = ci_low
        row["physical_sparsity_lambda_0p003_ci95_high"] = ci_high
        row["top_1pct_attention_mass"] = (
            bucket.concentration_sum[0] / bucket.queries
        )
        row["top_10pct_attention_mass"] = (
            bucket.concentration_sum[9] / bucket.queries
        )
        summary_rows.append(row)
    _write_csv(output / "summary.csv", summary_rows)

    per_sample_rows = []
    for key, bucket in buckets.items():
        if not key.startswith("sample|"):
            continue
        _, group, sample_id = key.split("|", 2)
        row = {
            "group": group,
            "sample_id": sample_id,
            "task": sample_metadata.get(sample_id, {}).get("task", ""),
            "query_rows": bucket.queries,
            "physical_tiles": bucket.physical_tiles,
        }
        for value in LAMBDAS:
            name = str(value).replace(".", "p")
            row[f"row_vote_sparsity_lambda_{name}"] = _exact_sparsity(bucket, value, False)
            row[f"physical_sparsity_lambda_{name}"] = _exact_sparsity(bucket, value, True)
        per_sample_rows.append(row)
    _write_csv(output / "per_sample_summary.csv", per_sample_rows)

    per_layer_rows = []
    for key, bucket in buckets.items():
        if not key.startswith("layer|"):
            continue
        _, group, layer = key.split("|", 2)
        row = {"group": group, "layer": int(layer), "query_rows": bucket.queries}
        for value in LAMBDAS:
            name = str(value).replace(".", "p")
            row[f"row_vote_sparsity_lambda_{name}"] = _exact_sparsity(bucket, value, False)
            row[f"physical_sparsity_lambda_{name}"] = _exact_sparsity(bucket, value, True)
        per_layer_rows.append(row)
    _write_csv(output / "per_layer_summary.csv", sorted(per_layer_rows, key=lambda r: (r["group"], r["layer"])))

    histogram_rows = []
    concentration_rows = []
    rank_rows = []
    for group in GROUP_ORDER:
        bucket = buckets[f"overall|{group}"]
        for index, left in enumerate(GAP_EDGES[:-1]):
            histogram_rows.append({
                "group": group,
                "gap_left": left,
                "gap_right": GAP_EDGES[index + 1],
                "online_equal_query_mass": bucket.online_query_hist[index] / bucket.queries,
                "online_pair_count": bucket.online_pair_hist[index],
                "global_equal_query_mass": bucket.global_query_hist[index] / bucket.queries,
                "physical_tile_count": bucket.physical_hist[index],
            })
        for fraction, total in zip(BLOCK_FRACTIONS, bucket.concentration_sum):
            concentration_rows.append({"group": group, "top_block_fraction": fraction, "mean_cumulative_attention_mass": total / bucket.queries})
        for fraction, total in zip(RANK_FRACTIONS, bucket.rank_gap_sum):
            rank_rows.append({"group": group, "block_rank_fraction": fraction, "mean_global_gap": total / bucket.queries})
    _write_csv(output / "gap_histograms.csv", histogram_rows)
    _write_csv(output / "attention_concentration.csv", concentration_rows)
    _write_csv(output / "block_rank_gap.csv", rank_rows)

    markdown = [
        "# Dense attention-distribution comparison",
        "",
        "Twenty balanced NVIDIA RULER samples were evaluated per model at 8K context. Dense attention outputs were left unmodified. Every valid token query in every head and layer is equally weighted in the query-level distributions. Physical sparsity uses the exact q32×kv64 all-row BLASST vote.",
        "",
        "| Group | Query rows | λ=.003 row vote | λ=.003 physical [95% sample CI] | Top 1% mass | Normalized block entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {GROUP_LABELS[row['group']]} | {row['query_rows']:,} | "
            f"{row['row_vote_sparsity_lambda_0p003']:.2%} | "
            f"{row['physical_sparsity_lambda_0p003']:.2%} "
            f"[{row['physical_sparsity_lambda_0p003_ci95_low']:.2%}, "
            f"{row['physical_sparsity_lambda_0p003_ci95_high']:.2%}] | "
            f"{row['top_1pct_attention_mass']:.2%} | "
            f"{row['normalized_block_entropy_mean']:.3f} |"
        )
    markdown.extend([
        "",
        "The online-gap survival curve maps directly to BLASST: a row votes to skip when $G^{run}>-\\log\\lambda$. The physical curve additionally requires every active query row in a q32 tile to vote for the same KV block. The concentration curve is independent of traversal order and measures exact dense softmax mass aggregated per KV block.",
        "",
        f"At $\\lambda=0.003$ ($-\\log\\lambda={-math.log(0.003):.3f}$), Fast-dLLM has a substantially longer gap tail than either DiffusionGemma attention type. Its top 1% of KV blocks carries {summary_rows[2]['top_1pct_attention_mass']:.1%} of attention mass on average, compared with {summary_rows[1]['top_1pct_attention_mass']:.1%} for DiffusionGemma global and {summary_rows[0]['top_1pct_attention_mass']:.1%} for local attention.",
        "",
        "Fast-dLLM collection follows the original BLASST experiment's eligibility rule and includes only ordinary cached denoising queries; prefix-prefill and cache-commit calls are excluded. DiffusionGemma collection covers its decoder canvas attention, split into native local and global layers. Structurally invalid KV blocks are excluded for every model.",
        "",
        "The two models use their own tokenizer-validated 8K RULER prompts. They share the same five-task balance and seed policy, but prompts are not assumed to be byte-identical because RULER length construction is tokenizer dependent.",
        "",
        "Uncertainty bands in the main figure are 95% bootstrap intervals over the 20 RULER samples, not over the much larger number of correlated query rows.",
    ])
    (output / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    _atomic_json(output / "report.json", {"summary": summary_rows, "states": [str(Path(path).resolve()) for path in args.states]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model-label", required=True, choices=("diffusion_gemma", "fast_dllm_v2"))
    collect_parser.add_argument("--model-adapter", required=True)
    collect_parser.add_argument("--model-path", required=True)
    collect_parser.add_argument("--revision")
    collect_parser.add_argument("--samples", required=True)
    collect_parser.add_argument("--num-samples", type=int, default=20)
    collect_parser.add_argument("--context-length", type=int, default=8192)
    collect_parser.add_argument("--q-tile-size", type=int, default=32)
    collect_parser.add_argument("--kv-tile-size", type=int, default=64)
    collect_parser.add_argument("--block-size", type=int, default=32)
    collect_parser.add_argument("--max-new-tokens", type=int, default=128)
    collect_parser.add_argument("--threshold", type=float, default=0.9)
    collect_parser.add_argument("--precision", default="bfloat16")
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--output-dir", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--states", nargs="+", required=True)
    report_parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
