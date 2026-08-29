"""Exact Sol-Attn proxy semantics and bounded per-prompt shard collection.

The callback is invoked at the registered attention boundary, after model Q/K
normalization and RoPE.  It never changes the attention result.  Each logical
row is summarized independently so correlated KV tiles are not later treated
as independent samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from scipy import stats

from dllm.attention.blasst.core import _attention_type, _prepare_attention_scores, _repeat_kv


ROW_QUANTILES = np.asarray((0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99))
DEFAULT_DENSITIES = (0.25, 0.50, 0.75)
ROW_META_COLUMNS = (
    "request_id", "corpus", "task", "split", "denoising_call", "layer",
    "head", "attention_type", "query_block", "query_start", "query_size",
    "prefix_length", "canvas_length", "canvas_tiles", "prefix_tiles",
)


@dataclass(frozen=True)
class CollectorConfig:
    q_block_size: int = 64
    kv_block_size: int = 64
    histogram_min: float = -8.0
    histogram_max: float = 8.0
    histogram_bins: int = 320
    reservoir_per_row: int = 32
    densities: tuple[float, ...] = DEFAULT_DENSITIES
    reservoir_seed: int = 20260822

    def __post_init__(self) -> None:
        if self.q_block_size != 64 or self.kv_block_size != 64:
            raise ValueError("the canonical Sol-Attn study requires 64x64 logical blocks")
        if self.histogram_bins <= 0 or self.histogram_max <= self.histogram_min:
            raise ValueError("histogram range and bin count are invalid")
        if self.reservoir_per_row <= 0:
            raise ValueError("reservoir_per_row must be positive")
        if not self.densities or any(not 0.0 < value < 1.0 for value in self.densities):
            raise ValueError("densities must lie strictly between zero and one")

    @property
    def histogram_edges(self) -> np.ndarray:
        return np.linspace(
            self.histogram_min,
            self.histogram_max,
            self.histogram_bins + 1,
            dtype=np.float64,
        )


def _stable_seed(parts: Iterable[Any], base_seed: int) -> int:
    digest = hashlib.sha256(
        (str(base_seed) + "|" + "|".join(map(str, parts))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def deterministic_reservoir(
    values: np.ndarray, size: int, *, identity: Iterable[Any], seed: int
) -> np.ndarray:
    """Return a stable, order-independent sample without replacement."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= size:
        return values.copy()
    rng = np.random.default_rng(_stable_seed(identity, seed))
    return values[np.sort(rng.choice(len(values), size=size, replace=False))]


def _unbiased_skewness_and_excess_kurtosis(
    values: np.ndarray,
) -> tuple[float, float]:
    """SciPy-equivalent bias-corrected moments without per-row dispatch overhead."""
    n = len(values)
    if n < 3:
        return math.nan, math.nan
    centered = values - values.mean()
    m2 = float(np.mean(centered * centered))
    if m2 <= 0.0:
        return math.nan, math.nan
    m3 = float(np.mean(centered**3))
    skewness = math.sqrt(n * (n - 1)) / (n - 2) * m3 / (m2**1.5)
    if n < 4:
        return skewness, math.nan
    m4 = float(np.mean(centered**4))
    biased_excess = m4 / (m2 * m2) - 3.0
    kurtosis = (n - 1) / ((n - 2) * (n - 3)) * (
        (n + 1) * biased_excess + 6.0
    )
    return skewness, kurtosis


def _tile_proxy(
    query: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    *,
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tile_valid = valid[..., q_start:q_end, kv_start:kv_end]
    eligible = tile_valid.any(dim=(-2, -1))
    q_participates = tile_valid.any(dim=-1)
    k_participates = tile_valid.any(dim=-2)
    q_vectors = query[..., q_start:q_end, :].float()
    k_vectors = key[..., kv_start:kv_end, :].float()
    q_mean = (q_vectors * q_participates[..., None]).sum(dim=-2)
    q_mean /= q_participates.sum(dim=-1, keepdim=True).clamp_min(1)
    k_mean = (k_vectors * k_participates[..., None]).sum(dim=-2)
    k_mean /= k_participates.sum(dim=-1, keepdim=True).clamp_min(1)
    return (q_mean * k_mean).sum(dim=-1) * float(scaling), eligible


def _validity_slice(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    *,
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
    kv_length: int,
    is_causal: bool,
    sliding_window: int | None,
) -> torch.Tensor:
    """Materialize only one query-block/region validity slice."""
    shape = (query.shape[0], query.shape[1], q_end - q_start, kv_end - kv_start)
    if attention_mask is None:
        valid = torch.ones(
            (query.shape[0], 1, q_end - q_start, kv_end - kv_start),
            dtype=torch.bool,
            device=query.device,
        )
    else:
        mask = attention_mask[..., q_start:q_end, kv_start:kv_end].to(query.device)
        if mask.dtype == torch.bool:
            valid = mask
        elif mask.is_floating_point():
            valid = torch.isfinite(mask) & (mask > -1.0e4)
        else:
            valid = mask != 0
    valid = torch.broadcast_to(valid, shape)
    if is_causal:
        q_positions = torch.arange(q_start, q_end, device=query.device) + (kv_length - query.shape[-2])
        k_positions = torch.arange(kv_start, kv_end, device=query.device)
        valid = valid & (k_positions[None, :] <= q_positions[:, None])
    if sliding_window is not None and (attention_mask is None or attention_mask.ndim < 4):
        q_positions = torch.arange(q_start, q_end, device=query.device) + (kv_length - query.shape[-2])
        k_positions = torch.arange(kv_start, kv_end, device=query.device)
        valid = valid & (k_positions[None, :] >= q_positions[:, None] - int(sliding_window) + 1)
    return valid


def _region_proxies(
    query: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    *,
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
    kv_block_size: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Vectorize literal proxy pooling across batch, heads, and KV blocks."""
    length = kv_end - kv_start
    block_count = (length + kv_block_size - 1) // kv_block_size
    padded_length = block_count * kv_block_size
    pad = padded_length - length
    if pad:
        valid = torch.nn.functional.pad(valid, (0, pad), value=False)
        key_region = torch.nn.functional.pad(key[..., kv_start:kv_end, :], (0, 0, 0, pad))
    else:
        key_region = key[..., kv_start:kv_end, :]
    valid_blocks = valid.reshape(
        query.shape[0], query.shape[1], q_end - q_start, block_count, kv_block_size
    )
    q_participates = valid_blocks.any(dim=-1).permute(0, 1, 3, 2)
    k_participates = valid_blocks.any(dim=-3).permute(0, 1, 2, 3)
    q_vectors = query[..., q_start:q_end, :].float()
    q_means = torch.matmul(q_participates.float(), q_vectors)
    q_means /= q_participates.sum(dim=-1, keepdim=True).clamp_min(1)
    key_blocks = key_region.reshape(
        key.shape[0], key.shape[1], block_count, kv_block_size, key.shape[-1]
    ).float()
    k_means = (key_blocks * k_participates[..., None]).sum(dim=-2)
    k_means /= k_participates.sum(dim=-1, keepdim=True).clamp_min(1)
    proxies = (q_means * k_means).sum(dim=-1) * float(scaling)
    eligible = valid_blocks.any(dim=(-3, -1))
    spans = [
        (kv_start + block * kv_block_size, min(kv_start + (block + 1) * kv_block_size, kv_end))
        for block in range(block_count)
    ]
    return proxies, eligible, spans


def sol_attention_proxy_rows(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    q_block_size: int = 64,
    kv_block_size: int = 64,
    prefix_length: int | None = None,
    scaling: float | None = None,
    is_causal: bool | None = None,
    sliding_window: int | None = None,
    prepared_key: torch.Tensor | None = None,
    valid_pair_mask: torch.Tensor | None = None,
    global_kv_tiling: bool = False,
) -> list[dict[str, Any]]:
    """Compute literal Mean(Q_i) Mean(K_j)^T scale for every eligible tile.

    GQA expansion and structural validity use the same helper as the reference
    attention backend. By default prefix and canvas blocks are tiled
    independently for the historical diagnostics path. When
    ``global_kv_tiling`` is true, one globally aligned KV grid is used and
    prefix/canvas tiles are placed into the same candidate population; this is
    the canonical Sol-Attn comparison mode.
    """
    if prepared_key is None or valid_pair_mask is None:
        repeats = (
            1
            if getattr(module, "_blasst_kv_already_repeated", False)
            else int(getattr(module, "num_key_value_groups", 1))
        )
        key = _repeat_kv(key, repeats)
        valid = None
    else:
        key, valid = prepared_key, valid_pair_mask
    scale = float(query.shape[-1] ** -0.5 if scaling is None else scaling)
    q_length, kv_length = query.shape[-2], key.shape[-2]
    if prefix_length is None:
        prefix_length = max(0, kv_length - q_length)
    if not 0 <= prefix_length <= kv_length:
        raise ValueError("prefix_length must be within the KV sequence")
    regions = (("prefix", 0, prefix_length), ("canvas", prefix_length, kv_length))
    rows: list[dict[str, Any]] = []
    row_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for q_start in range(0, q_length, q_block_size):
                q_end = min(q_start + q_block_size, q_length)
                row: dict[str, Any] = {
                    "batch": batch,
                    "head": head,
                    "query_block": q_start // q_block_size,
                    "query_start": q_start,
                    "query_size": q_end - q_start,
                    "prefix": [],
                    "canvas": [],
                    "prefix_spans": [],
                    "canvas_spans": [],
                }
                row_lookup[(batch, head, q_start)] = row
    causal = attention_mask is None and q_length > 1 if is_causal is None else bool(is_causal)

    if global_kv_tiling:
        # The candidate tiles are aligned to the complete KV sequence rather
        # than restarted at the prefix boundary. This matters whenever the
        # boundary is not a multiple of ``kv_block_size``: a physical tile is
        # indivisible and therefore must not be silently split into two proxy
        # populations.
        for q_start in range(0, q_length, q_block_size):
            q_end = min(q_start + q_block_size, q_length)
            for kv_start in range(0, kv_length, kv_block_size):
                kv_end = min(kv_start + kv_block_size, kv_length)
                if valid is not None:
                    tile_valid = valid[..., q_start:q_end, kv_start:kv_end]
                else:
                    tile_valid = _validity_slice(
                        attention_mask,
                        query,
                        q_start=q_start,
                        q_end=q_end,
                        kv_start=kv_start,
                        kv_end=kv_end,
                        kv_length=kv_length,
                        is_causal=causal,
                        sliding_window=sliding_window,
                    )
                tile_eligible = tile_valid.any(dim=(-2, -1))
                if not bool(tile_eligible.any().item()):
                    continue
                q_participates = tile_valid.any(dim=-1)
                k_participates = tile_valid.any(dim=-2)
                q_vectors = query[..., q_start:q_end, :].float()
                k_vectors = key[..., kv_start:kv_end, :].float()
                q_mean = (q_vectors * q_participates[..., None]).sum(dim=-2)
                q_mean /= q_participates.sum(dim=-1, keepdim=True).clamp_min(1)
                k_mean = (k_vectors * k_participates[..., None]).sum(dim=-2)
                k_mean /= k_participates.sum(dim=-1, keepdim=True).clamp_min(1)
                proxies = (q_mean * k_mean).sum(dim=-1) * scale
                region = "prefix" if kv_start < prefix_length else "canvas"
                span = (kv_start, kv_end)
                for batch in range(query.shape[0]):
                    for head in range(query.shape[1]):
                        if not bool(tile_eligible[batch, head]):
                            continue
                        row = row_lookup[(batch, head, q_start)]
                        row[region].append(float(proxies[batch, head].item()))
                        row[f"{region}_spans"].append(span)
        rows.extend(row for row in row_lookup.values() if row["canvas"] or row["prefix"])
        return rows

    for q_start in range(0, q_length, q_block_size):
        q_end = min(q_start + q_block_size, q_length)
        for region, region_start, region_end in regions:
            if region_start >= region_end:
                continue
            region_valid = (
                valid[..., q_start:q_end, region_start:region_end]
                if valid is not None
                else _validity_slice(
                    attention_mask,
                    query,
                    q_start=q_start,
                    q_end=q_end,
                    kv_start=region_start,
                    kv_end=region_end,
                    kv_length=kv_length,
                    is_causal=causal,
                    sliding_window=sliding_window,
                )
            )
            proxies, eligible, spans = _region_proxies(
                query,
                key,
                region_valid,
                q_start=q_start,
                q_end=q_end,
                kv_start=region_start,
                kv_end=region_end,
                kv_block_size=kv_block_size,
                scaling=scale,
            )
            proxies_cpu = proxies.detach().cpu().numpy()
            eligible_cpu = eligible.detach().cpu().numpy()
            for batch in range(query.shape[0]):
                for head in range(query.shape[1]):
                    row = row_lookup[(batch, head, q_start)]
                    for block, span in enumerate(spans):
                        if eligible_cpu[batch, head, block]:
                            row[region].append(float(proxies_cpu[batch, head, block]))
                            row[f"{region}_spans"].append(span)
    rows.extend(row for row in row_lookup.values() if row["canvas"] or row["prefix"])
    return rows


def hierarchical_row_weights(rows: list[Mapping[str, Any]]) -> np.ndarray:
    """Equal prompt/call/layer/head/query-block mass, in that order."""
    if not rows:
        return np.empty(0, dtype=np.float64)
    keys = ("request_id", "denoising_call", "layer", "head", "query_block")
    tuples = [tuple(row[name] for name in keys) for row in rows]
    weights = np.ones(len(rows), dtype=np.float64)
    parent_groups: dict[tuple[Any, ...], list[int]] = {(): list(range(len(rows)))}
    for depth in range(len(keys)):
        next_groups: dict[tuple[Any, ...], list[int]] = {}
        for parent, indices in parent_groups.items():
            children: dict[Any, list[int]] = {}
            for index in indices:
                children.setdefault(tuples[index][depth], []).append(index)
            for child, child_indices in children.items():
                weights[child_indices] /= len(children)
                next_groups[parent + (child,)] = child_indices
        parent_groups = next_groups
    return weights / weights.sum()


class PromptShardCollector:
    """Accumulate compact row summaries for one prompt at a time."""

    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.context: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self._hist_canvas: list[np.ndarray] = []
        self._hist_prefix: list[np.ndarray] = []
        self._reservoir_canvas: list[np.ndarray] = []
        self._reservoir_prefix: list[np.ndarray] = []
        self._outside_canvas: list[tuple[int, int]] = []
        self._outside_prefix: list[tuple[int, int]] = []
        self.calls_seen = 0
        self.calls_recorded = 0

    def begin_prompt(self, **context: Any) -> None:
        if self.rows:
            raise RuntimeError("export the current prompt shard before beginning another")
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
            prefix_length = int(
                runtime.dense_kv_prefix_extractor(
                    module, query, key, value, attention_mask, kwargs
                )
            )
        if not 0 <= prefix_length <= key.shape[-2]:
            raise ValueError("dense KV prefix extractor returned an invalid length")
        proxy_rows = sol_attention_proxy_rows(
            module,
            query,
            key,
            value,
            attention_mask,
            q_block_size=self.config.q_block_size,
            kv_block_size=self.config.kv_block_size,
            prefix_length=prefix_length,
            scaling=kwargs.get("scaling"),
            is_causal=kwargs.get("is_causal"),
            sliding_window=kwargs.get("sliding_window"),
        )
        if proxy_rows:
            self.calls_recorded += 1
        for proxy_row in proxy_rows:
            canvas = np.asarray(proxy_row["canvas"], dtype=np.float64)
            prefix = np.asarray(proxy_row["prefix"], dtype=np.float64)
            # Deployable thresholds are standardized on canvas candidates only.
            center = float(canvas.mean()) if len(canvas) else 0.0
            sigma = float(canvas.std(ddof=0)) if len(canvas) else 0.0
            denominator = max(sigma, 1.0e-8)
            canvas_z = (canvas - center) / denominator
            prefix_z = (prefix - center) / denominator
            identity = (
                self.context.get("request_id"), call, layer, proxy_row["head"],
                proxy_row["query_block"],
            )
            skewness, kurtosis = _unbiased_skewness_and_excess_kurtosis(canvas)
            row = {
                "request_id": str(self.context.get("request_id", "unknown")),
                "corpus": str(self.context.get("corpus", "unknown")),
                "task": str(self.context.get("task", "unclassified")),
                "split": str(self.context.get("split", "unknown")),
                "denoising_call": call,
                "layer": layer,
                "head": int(proxy_row["head"]),
                "attention_type": attention_type,
                "query_block": int(proxy_row["query_block"]),
                "query_start": int(proxy_row["query_start"]),
                "query_size": int(proxy_row["query_size"]),
                "prefix_length": prefix_length,
                "canvas_length": int(key.shape[-2] - prefix_length),
                "canvas_tiles": len(canvas),
                "prefix_tiles": len(prefix),
                "row_mean": center,
                "row_sigma": sigma,
                "row_skewness": skewness,
                "row_excess_kurtosis": kurtosis,
                "row_max_z": float(canvas_z.max()) if len(canvas_z) else math.nan,
            }
            quantiles = np.quantile(canvas_z, ROW_QUANTILES) if len(canvas_z) else np.full(len(ROW_QUANTILES), np.nan)
            row.update({f"q{int(q * 100):02d}": float(value) for q, value in zip(ROW_QUANTILES, quantiles)})
            for density in self.config.densities:
                beta = float(stats.norm.ppf(1.0 - density))
                row[f"gaussian_density_{density:g}"] = float(np.mean(canvas_z > beta)) if len(canvas_z) else math.nan
            self.rows.append(row)
            self._hist_canvas.append(
                np.histogram(canvas_z, bins=self.config.histogram_edges)[0].astype(np.uint8)
            )
            self._hist_prefix.append(
                np.histogram(prefix_z, bins=self.config.histogram_edges)[0].astype(np.uint8)
            )
            edges = self.config.histogram_edges
            self._outside_canvas.append((int(np.sum(canvas_z < edges[0])), int(np.sum(canvas_z > edges[-1]))))
            self._outside_prefix.append((int(np.sum(prefix_z < edges[0])), int(np.sum(prefix_z > edges[-1]))))
            self._reservoir_canvas.append(deterministic_reservoir(
                canvas_z, self.config.reservoir_per_row, identity=(*identity, "canvas"), seed=self.config.reservoir_seed
            ).astype(np.float32))
            self._reservoir_prefix.append(deterministic_reservoir(
                prefix_z, self.config.reservoir_per_row, identity=(*identity, "prefix"), seed=self.config.reservoir_seed
            ).astype(np.float32))

    def _reservoir_matrix(self, values: list[np.ndarray]) -> np.ndarray:
        result = np.full((len(values), self.config.reservoir_per_row), np.nan, dtype=np.float32)
        for index, row in enumerate(values):
            result[index, : len(row)] = row
        return result

    def export_shard(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"prompt shard already exists: {path}")
        weights = hierarchical_row_weights(self.rows)
        edges = self.config.histogram_edges
        payload: dict[str, np.ndarray] = {
            name: np.asarray([row[name] for row in self.rows]) for name in ROW_META_COLUMNS
        }
        for name in ("row_mean", "row_sigma", "row_skewness", "row_excess_kurtosis", "row_max_z"):
            payload[name] = np.asarray([row[name] for row in self.rows], dtype=np.float64)
        for q in ROW_QUANTILES:
            name = f"q{int(q * 100):02d}"
            payload[name] = np.asarray([row[name] for row in self.rows], dtype=np.float64)
        for density in self.config.densities:
            name = f"gaussian_density_{density:g}"
            payload[name] = np.asarray([row[name] for row in self.rows], dtype=np.float64)
        # A row has at most 4 canvas and 128 prefix tiles in the canonical
        # geometry, so uint8 retains exact counts while avoiding multi-GB
        # prompt shards for long Fast-dLLM-v2 denoising traces.
        canvas_hist = np.asarray(self._hist_canvas, dtype=np.uint8).reshape((-1, self.config.histogram_bins))
        prefix_hist = np.asarray(self._hist_prefix, dtype=np.uint8).reshape((-1, self.config.histogram_bins))
        payload.update(
            histogram_edges=edges,
            canvas_histogram=canvas_hist,
            prefix_histogram=prefix_hist,
            canvas_underflow=np.asarray([value[0] for value in self._outside_canvas], dtype=np.uint16),
            canvas_overflow=np.asarray([value[1] for value in self._outside_canvas], dtype=np.uint16),
            prefix_underflow=np.asarray([value[0] for value in self._outside_prefix], dtype=np.uint16),
            prefix_overflow=np.asarray([value[1] for value in self._outside_prefix], dtype=np.uint16),
            canvas_reservoir=self._reservoir_matrix(self._reservoir_canvas),
            prefix_reservoir=self._reservoir_matrix(self._reservoir_prefix),
            hierarchical_row_weight=weights,
        )
        np.savez_compressed(path, **payload)
        coverage: dict[str, Any] = {}
        for column in ("denoising_call", "layer", "head", "attention_type", "query_block"):
            values, counts = np.unique(payload[column], return_counts=True)
            coverage[column] = {str(value): int(count) for value, count in zip(values, counts)}
        coverage["kv_region"] = {
            "prefix_tiles": int(payload["prefix_tiles"].sum()),
            "canvas_tiles": int(payload["canvas_tiles"].sum()),
        }
        sidecar = {
            "schema_version": 1,
            "proxy_definition": "Mean(post-normalization/post-RoPE Q_i) @ Mean(post-normalization/post-RoPE K_j) * native scale",
            "dense_output_semantics": "observer returns the original native dense attention output unchanged",
            "config": asdict(self.config),
            "context": self.context,
            "calls_seen": self.calls_seen,
            "calls_recorded": self.calls_recorded,
            "rows": len(self.rows),
            "coverage": coverage,
            **dict(metadata or {}),
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        self.rows.clear()
        self._hist_canvas.clear()
        self._hist_prefix.clear()
        self._reservoir_canvas.clear()
        self._reservoir_prefix.clear()
        self._outside_canvas.clear()
        self._outside_prefix.clear()
        return path
