"""Bounded prefix-proxy collection for Math500 or RULER 16K."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from experiments.diffusion_attention_threshold_modeling.proxy import (
    deterministic_reservoir,
    hierarchical_row_weights,
    sol_attention_proxy_rows,
)
from dllm.attention.blasst.core import _attention_type


@dataclass(frozen=True)
class PrefixProxyConfig:
    q_block_size: int = 64
    kv_block_size: int = 64
    reservoir_per_row: int = 256
    reservoir_seed: int = 20260824

    def __post_init__(self) -> None:
        if self.q_block_size != 64 or self.kv_block_size != 64:
            raise ValueError("the prefix proxy study requires 64x64 logical blocks")
        if self.reservoir_per_row <= 0:
            raise ValueError("reservoir_per_row must be positive")


def _moments(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


class PrefixProxyShardCollector:
    """Collect raw prefix proxy values while excluding canvas candidates."""

    def __init__(self, config: PrefixProxyConfig) -> None:
        self.config = config
        self.context: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self._reservoirs: list[np.ndarray] = []
        self.calls_seen = 0
        self.calls_recorded = 0

    def begin_prompt(self, **context: Any) -> None:
        if self.rows:
            raise RuntimeError("export the current prefix shard before beginning another prompt")
        self.context = dict(context)
        self.calls_seen = 0
        self.calls_recorded = 0

    @torch.no_grad()
    def __call__(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> None:
        self.calls_seen += 1
        runtime = getattr(module, "_blasst_2d_runtime", None)
        metadata = dict(getattr(runtime, "metadata", {}) or {})
        call = int(metadata.get("denoising_step", metadata.get("forward_pass_id", -1)))
        layer = int(getattr(module, "layer_idx", -1))
        attention_type = _attention_type(module, kwargs.get("sliding_window"))
        prefix_length = max(0, key.shape[-2] - query.shape[-2])
        if runtime is not None and runtime.dense_kv_prefix_extractor is not None:
            prefix_length = int(runtime.dense_kv_prefix_extractor(
                module, query, key, value, attention_mask, kwargs
            ))
        rows = sol_attention_proxy_rows(
            module, query, key, value, attention_mask,
            q_block_size=self.config.q_block_size,
            kv_block_size=self.config.kv_block_size,
            prefix_length=prefix_length,
            scaling=kwargs.get("scaling"),
            is_causal=kwargs.get("is_causal"),
            sliding_window=kwargs.get("sliding_window"),
        )
        if rows:
            self.calls_recorded += 1
        for proxy_row in rows:
            prefix = np.asarray(proxy_row["prefix"], dtype=np.float64)
            if not len(prefix):
                continue
            identity = (
                self.context.get("request_id"), call, layer,
                proxy_row["head"], proxy_row["query_block"], "prefix-raw",
            )
            summary = _moments(prefix)
            self.rows.append({
                "request_id": str(self.context.get("request_id", "unknown")),
                "problem_index": int(self.context.get("problem_index", -1)),
                "denoising_call": call,
                "layer": layer,
                "head": int(proxy_row["head"]),
                "attention_type": attention_type,
                "query_block": int(proxy_row["query_block"]),
                "query_start": int(proxy_row["query_start"]),
                "query_size": int(proxy_row["query_size"]),
                "prefix_length": int(prefix_length),
                "prefix_tiles": int(len(prefix)),
                "prefix_mean": summary["mean"],
                "prefix_std": summary["std"],
                "prefix_min": summary["min"],
                "prefix_max": summary["max"],
            })
            self._reservoirs.append(deterministic_reservoir(
                prefix, self.config.reservoir_per_row,
                identity=identity, seed=self.config.reservoir_seed,
            ).astype(np.float32))

    def export_shard(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"prefix shard already exists: {path}")
        weights = hierarchical_row_weights(self.rows)
        payload: dict[str, np.ndarray] = {}
        for name in (
            "request_id", "problem_index", "denoising_call", "layer", "head",
            "attention_type", "query_block", "query_start", "query_size",
            "prefix_length", "prefix_tiles", "prefix_mean", "prefix_std",
            "prefix_min", "prefix_max",
        ):
            dtype = np.float64 if name.startswith("prefix_") and name not in ("prefix_length", "prefix_tiles") else None
            payload[name] = np.asarray([row[name] for row in self.rows], dtype=dtype)
        reservoir = np.full(
            (len(self._reservoirs), self.config.reservoir_per_row), np.nan, dtype=np.float32
        )
        for index, values in enumerate(self._reservoirs):
            reservoir[index, :len(values)] = values
        payload["prefix_reservoir"] = reservoir
        payload["hierarchical_row_weight"] = weights
        np.savez_compressed(path, **payload)
        sidecar = {
            "schema_version": 1,
            "study": f"{self.context.get('corpus', 'unknown')}_prefix_proxy_distribution",
            "population": "prefix_only",
            "score_definition": "Mean(post-normalization/post-RoPE Q_i) @ Mean(post-normalization/post-RoPE K_j) * native scale",
            "score_transform": "raw pre-softmax block-proxy logits; no row standardization",
            "dense_output_semantics": "observer leaves native dense attention output unchanged",
            "config": asdict(self.config),
            "context": self.context,
            "calls_seen": self.calls_seen,
            "calls_recorded": self.calls_recorded,
            "rows": len(self.rows),
            "prefix_tiles": int(sum(row["prefix_tiles"] for row in self.rows)),
            "attention_types": sorted({row["attention_type"] for row in self.rows}),
            "metadata": dict(metadata or {}),
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n")
        self.rows.clear()
        self._reservoirs.clear()
        self.context.clear()
        return path
