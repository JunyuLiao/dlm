"""Per-row, per-step prefix proxy capture without cross-stratum aggregation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dllm.attention.blasst.core import _attention_type
from experiments.diffusion_attention_threshold_modeling.proxy import (
    deterministic_reservoir,
    sol_attention_proxy_rows,
)


@dataclass(frozen=True)
class DiagnosticConfig:
    q_block_size: int = 64
    kv_block_size: int = 64
    reservoir_per_row: int = 256
    reservoir_seed: int = 20260824

    def __post_init__(self) -> None:
        if (self.q_block_size, self.kv_block_size) != (64, 64):
            raise ValueError("the diagnostic uses 64x64 logical blocks")
        if self.reservoir_per_row <= 0:
            raise ValueError("reservoir_per_row must be positive")


def _moments(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


class StepwisePrefixCollector:
    """Capture one prompt's prefix rows, retaining each row independently."""

    def __init__(self, config: DiagnosticConfig) -> None:
        self.config = config
        self.context: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.reservoirs: list[np.ndarray] = []
        self.call_stats: dict[int, dict[str, Any]] = {}
        self.forward_state: dict[str, Any] = {}

    def begin_prompt(self, **context: Any) -> None:
        if self.rows:
            raise RuntimeError("collector already contains a prompt")
        self.context = dict(context)
        self.forward_state = {}

    def set_forward_state(self, *, masked_tokens: int, active_tokens: int, mask_ratio: float) -> None:
        """Receive read-only model-level token-state metadata before attention."""
        self.forward_state = {
            "masked_tokens": int(masked_tokens),
            "active_tokens": int(active_tokens),
            "mask_ratio": float(mask_ratio),
        }

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
        runtime = getattr(module, "_blasst_2d_runtime", None)
        metadata = dict(getattr(runtime, "metadata", {}) or {})
        call = int(metadata.get("denoising_step", metadata.get("forward_pass_id", -1)))
        prefix_length = max(0, int(key.shape[-2] - query.shape[-2]))
        if runtime is not None and runtime.dense_kv_prefix_extractor is not None:
            prefix_length = int(runtime.dense_kv_prefix_extractor(
                module, query, key, value, attention_mask, kwargs
            ))
        attention_type = _attention_type(module, kwargs.get("sliding_window"))
        mask_ratio = float(metadata.get("mask_ratio", self.forward_state.get("mask_ratio", -1.0)))
        masked_tokens = int(metadata.get("masked_tokens", self.forward_state.get("masked_tokens", -1)))
        active_mask = getattr(runtime, "active_query_mask", None)
        active_tokens = int(active_mask.sum().item()) if isinstance(active_mask, torch.Tensor) else int(self.forward_state.get("active_tokens", -1))
        call_record = self.call_stats.setdefault(call, {
            "denoising_call": call,
            "mask_ratio": mask_ratio,
            "masked_tokens": masked_tokens,
            "active_tokens": active_tokens,
            "rows": 0,
            "layers": set(),
            "heads": set(),
            "attention_types": set(),
        })
        if call_record["mask_ratio"] < 0 and mask_ratio >= 0:
            call_record["mask_ratio"] = mask_ratio
        if call_record["masked_tokens"] < 0 and masked_tokens >= 0:
            call_record["masked_tokens"] = masked_tokens
        if call_record["active_tokens"] < 0 and active_tokens >= 0:
            call_record["active_tokens"] = active_tokens
        rows = sol_attention_proxy_rows(
            module, query, key, value, attention_mask,
            q_block_size=self.config.q_block_size,
            kv_block_size=self.config.kv_block_size,
            prefix_length=prefix_length,
            scaling=kwargs.get("scaling"),
            is_causal=kwargs.get("is_causal"),
            sliding_window=kwargs.get("sliding_window"),
        )
        layer = int(getattr(module, "layer_idx", -1))
        for proxy_row in rows:
            prefix = np.asarray(proxy_row["prefix"], dtype=np.float64)
            if not len(prefix):
                continue
            mean, std = _moments(prefix)
            identity = (
                self.context.get("request_id"), call, layer,
                int(proxy_row["head"]), int(proxy_row["query_block"]), "stepwise-prefix",
            )
            sample = deterministic_reservoir(
                prefix, self.config.reservoir_per_row,
                identity=identity, seed=self.config.reservoir_seed,
            ).astype(np.float32)
            self.rows.append({
                "request_id": str(self.context.get("request_id", "unknown")),
                "denoising_call": call,
                "layer": layer,
                "head": int(proxy_row["head"]),
                "query_block": int(proxy_row["query_block"]),
                "query_start": int(proxy_row["query_start"]),
                "query_size": int(proxy_row["query_size"]),
                "attention_type": attention_type,
                "prefix_length": prefix_length,
                "prefix_tiles": int(len(prefix)),
                "prefix_mean": mean,
                "prefix_std": std,
                "prefix_min": float(prefix.min()),
                "prefix_max": float(prefix.max()),
                "mask_ratio": mask_ratio,
                "masked_tokens": masked_tokens,
                "active_tokens": active_tokens,
            })
            self.reservoirs.append(sample)
            call_record["rows"] += 1
            call_record["layers"].add(layer)
            call_record["heads"].add(int(proxy_row["head"]))
            call_record["attention_types"].add(attention_type)

    def export(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        fields = (
            "request_id", "denoising_call", "layer", "head", "query_block",
            "query_start", "query_size", "attention_type", "prefix_length",
            "prefix_tiles", "prefix_mean", "prefix_std", "prefix_min",
            "prefix_max", "mask_ratio", "masked_tokens", "active_tokens",
        )
        payload: dict[str, np.ndarray] = {}
        for field in fields:
            dtype = np.float64 if field.startswith("prefix_") or field == "mask_ratio" else None
            payload[field] = np.asarray([row[field] for row in self.rows], dtype=dtype)
        reservoir = np.full(
            (len(self.reservoirs), self.config.reservoir_per_row), np.nan, dtype=np.float32
        )
        for index, values in enumerate(self.reservoirs):
            reservoir[index, :len(values)] = values
        payload["prefix_reservoir"] = reservoir
        np.savez_compressed(path, **payload)
        calls = []
        for call in sorted(self.call_stats):
            item = dict(self.call_stats[call])
            item["layers"] = sorted(item["layers"])
            item["heads"] = sorted(item["heads"])
            item["attention_types"] = sorted(item["attention_types"])
            calls.append(item)
        sidecar = {
            "schema_version": 1,
            "study": "ruler16k_stepwise_proxy_diagnostic",
            "population": "prefix_only",
            "score_definition": "Mean(post-normalization/post-RoPE Q_i) @ Mean(post-normalization/post-RoPE K_j) * native scale",
            "score_transform": "raw per-row proxy retained; standardized z=(s-mu_row)/sigma_row for plots",
            "config": asdict(self.config),
            "context": self.context,
            "rows": len(self.rows),
            "calls": calls,
            "metadata": metadata or {},
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path
