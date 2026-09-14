"""Fresh canvas-only keep/drop routing and paired attention diagnostics."""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from scipy import stats

from dllm.attention.blasst.core import (
    _attention_type,
    _finish_eager_attention,
    _prepare_attention_inputs,
    _prepare_attention_scores,
)

from .proxy import sol_attention_proxy_rows


@dataclass(frozen=True)
class FreshRoutingConfig:
    mode: str
    target_density: float
    q_block_size: int = 64
    kv_block_size: int = 64
    threshold_model: Mapping[str, Any] | None = None
    adapter: str = "diffusion_gemma"
    corpus: str = "math500"
    execution: str = "logical"
    combined_region_population: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("gaussian", "profiled", "oracle", "random"):
            raise ValueError("routing mode must be gaussian, profiled, oracle, or random")
        if not 0.0 < self.target_density <= 1.0:
            raise ValueError("target_density must lie in (0, 1]")
        if self.q_block_size not in (32, 64, 128) or self.kv_block_size != 64:
            raise ValueError("routing requires q blocks in {32,64,128} and 64-token KV blocks")
        if self.mode == "profiled" and not self.threshold_model:
            raise ValueError("profiled routing requires a fitted threshold model")
        if getattr(self, "region", "canvas") not in ("all", "canvas", "prefix_only"):
            raise ValueError("routing region must be all, canvas, or prefix_only")
        if self.execution not in ("logical", "physical"):
            raise ValueError("routing execution must be logical or physical")

    region: str = "canvas"
    random_seed: int = 20260825


@dataclass
class FreshRoutingStats:
    calls: int = 0
    logical_candidate_tiles: int = 0
    logical_retained_tiles: int = 0
    physical_candidate_tiles: int = 0
    physical_retained_tiles: int = 0
    physical_total_tiles: int = 0
    physical_skipped_tiles: int = 0
    dense_attention_mass_sum: float = 0.0
    dense_attention_mass_rows: int = 0
    output_relative_error_sum: float = 0.0
    output_relative_error_calls: int = 0
    fallback_rows: int = 0
    degenerate_rows: int = 0
    routing_rows: int = 0
    region_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    all_region_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    retained_tiles_per_row_histogram: dict[str, int] = field(default_factory=dict)
    by_attention_type: dict[str, dict[str, int]] = field(default_factory=dict)
    by_layer: dict[str, dict[str, int]] = field(default_factory=dict)
    by_head: dict[str, dict[str, int]] = field(default_factory=dict)
    by_step: dict[str, dict[str, int]] = field(default_factory=dict)
    mask_overlap_intersection: int = 0
    mask_overlap_union: int = 0
    mask_overlap_comparisons: int = 0
    proxy_mask_seconds: float = 0.0
    sparse_attention_seconds: float = 0.0
    per_call: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        prefill_calls = [item for item in self.per_call if int(item.get("denoising_step", -1)) < 0]
        denoising_calls = [item for item in self.per_call if int(item.get("denoising_step", -1)) >= 0]
        def _phase_sum(rows: list[dict[str, Any]], key: str) -> float:
            return float(sum(float(item.get(key) or 0.0) for item in rows))
        return {
            "calls": self.calls,
            "logical_candidate_tiles": self.logical_candidate_tiles,
            "logical_retained_tiles": self.logical_retained_tiles,
            "logical_density": self.logical_retained_tiles / self.logical_candidate_tiles if self.logical_candidate_tiles else math.nan,
            "physical_candidate_tiles": self.physical_candidate_tiles,
            "physical_retained_tiles": self.physical_retained_tiles,
            "physical_density": self.physical_retained_tiles / self.physical_candidate_tiles if self.physical_candidate_tiles else math.nan,
            "physical_total_tiles": self.physical_total_tiles,
            "physical_skipped_tiles": self.physical_skipped_tiles,
            "full_model_physical_sparsity": self.physical_skipped_tiles / self.physical_total_tiles if self.physical_total_tiles else math.nan,
            "retained_dense_attention_mass": self.dense_attention_mass_sum / self.dense_attention_mass_rows if self.dense_attention_mass_rows else None,
            "dense_attention_mass_rows": self.dense_attention_mass_rows,
            "attention_output_relative_error": self.output_relative_error_sum / self.output_relative_error_calls if self.output_relative_error_calls else None,
            "fallback_rows": self.fallback_rows,
            "degenerate_rows": self.degenerate_rows,
            "routing_rows": self.routing_rows,
            "fallback_row_fraction": self.fallback_rows / self.routing_rows if self.routing_rows else None,
            "degenerate_row_fraction": self.degenerate_rows / self.routing_rows if self.routing_rows else None,
            "region_counts": self.region_counts,
            "all_region_counts": self.all_region_counts,
            "retained_tiles_per_row_histogram": self.retained_tiles_per_row_histogram,
            "by_attention_type": self.by_attention_type,
            "by_layer": self.by_layer,
            "by_head": self.by_head,
            "by_step": self.by_step,
            "consecutive_mask_jaccard": self.mask_overlap_intersection / self.mask_overlap_union if self.mask_overlap_union else None,
            "mask_overlap_intersection": self.mask_overlap_intersection,
            "mask_overlap_union": self.mask_overlap_union,
            "mask_overlap_comparisons": self.mask_overlap_comparisons,
            "proxy_mask_seconds": self.proxy_mask_seconds,
            "sparse_attention_seconds": self.sparse_attention_seconds,
            "prefill_attention_calls": len(prefill_calls),
            "denoising_attention_calls": len(denoising_calls),
            "prefill_proxy_mask_seconds": _phase_sum(prefill_calls, "proxy_mask_seconds"),
            "denoising_proxy_mask_seconds": _phase_sum(denoising_calls, "proxy_mask_seconds"),
            "prefill_sparse_attention_seconds": _phase_sum(prefill_calls, "sparse_attention_seconds"),
            "denoising_sparse_attention_seconds": _phase_sum(denoising_calls, "sparse_attention_seconds"),
            "per_call": self.per_call,
        }


def _merge_counts(target: dict[str, dict[str, int]], key: str, **values: int) -> None:
    entry = target.setdefault(key, {name: 0 for name in values})
    for name, value in values.items():
        entry[name] = entry.get(name, 0) + int(value)


def _profiled_beta(model: Mapping[str, Any], config: FreshRoutingConfig, attention_type: str) -> float:
    selected = model.get("selected", model)
    scope = selected["scope"]
    if scope == "model":
        key = config.adapter
    elif scope == "model_attention":
        key = f"{config.adapter}|{attention_type}"
    elif scope == "model_corpus_attention":
        key = f"{config.adapter}|{config.corpus}|{attention_type}"
    else:
        raise ValueError(f"unsupported fitted scope: {scope}")
    try:
        return float(selected["fits"][key]["betas"][str(float(config.target_density))])
    except KeyError as error:
        raise KeyError(f"threshold table has no beta for {key} at {config.target_density}") from error


def _keep_vector(
    values: np.ndarray,
    config: FreshRoutingConfig,
    attention_type: str,
    *,
    random_identity: tuple[Any, ...] = (),
) -> tuple[np.ndarray, float, bool]:
    if not len(values):
        return np.zeros(0, dtype=bool), math.nan, False
    mean, sigma = float(values.mean()), float(values.std(ddof=0))
    z = (values - mean) / max(sigma, 1.0e-8)
    count = max(1, min(len(values), int(math.ceil(len(values) * config.target_density))))
    if config.mode in ("oracle", "random"):
        if config.mode == "random":
            # Match the realized count of the corresponding universal-
            # Gaussian decision, including its deterministic fallback.
            reference_beta = float(stats.norm.ppf(1.0 - config.target_density))
            reference_keep = z >= reference_beta
            if not reference_keep.any():
                reference_keep[int(np.argmax(values))] = True
            count = int(reference_keep.sum())
            digest = hashlib.sha256((str(config.random_seed) + "|" + "|".join(map(str, random_identity))).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            selected = rng.choice(len(values), size=count, replace=False)
        else:
            selected = np.argsort(values, kind="stable")[-count:]
        keep = np.zeros(len(values), dtype=bool)
        keep[selected] = True
        return keep, math.nan, False
    beta = (
        float(stats.norm.ppf(1.0 - config.target_density))
        if config.mode == "gaussian"
        else _profiled_beta(config.threshold_model or {}, config, attention_type)
    )
    keep = z >= beta
    fallback = False
    if not keep.any():
        keep[int(np.argmax(values))] = True
        fallback = True
    return keep, beta, fallback


def _physical_sparse_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Exact reference that evaluates QK and AV only for retained KV tokens."""
    output = torch.zeros(
        (query.shape[0], query.shape[-2], query.shape[1], value.shape[-1]),
        dtype=query.dtype,
        device=query.device,
    )
    additive = None
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        additive = torch.broadcast_to(
            attention_mask[..., : key.shape[-2]].to(query.device), allowed.shape
        )
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for query_index in range(query.shape[-2]):
                retained = torch.nonzero(
                    allowed[batch, head, query_index], as_tuple=False
                ).flatten()
                if not retained.numel():
                    continue
                logits = torch.matmul(
                    key[batch, head, retained].float(),
                    query[batch, head, query_index].float(),
                ) * float(scaling)
                if additive is not None:
                    logits = logits + additive[batch, head, query_index, retained].float()
                probabilities = torch.softmax(logits, dim=-1).to(query.dtype)
                if dropout:
                    probabilities = torch.nn.functional.dropout(
                        probabilities, p=dropout, training=training
                    )
                output[batch, query_index, head] = torch.matmul(
                    probabilities, value[batch, head, retained]
                )
    return output


def _physical_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float,
    training: bool,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Exact two-pass sparse reference that never materializes dropped QK tiles."""
    if dropout and training:
        raise ValueError("the physical reference does not support training dropout")
    output = torch.zeros(
        (query.shape[0], query.shape[-2], query.shape[1], value.shape[-1]),
        dtype=query.dtype, device=query.device,
    )
    additive = None
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        additive = torch.broadcast_to(
            attention_mask[..., : key.shape[-2]].to(query.device), allowed.shape
        )
    q_block, kv_block = block_size
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for q_start in range(0, query.shape[-2], q_block):
                q_end = min(q_start + q_block, query.shape[-2])
                q_part = query[batch, head, q_start:q_end].float()
                retained_spans = []
                for kv_start in range(0, key.shape[-2], kv_block):
                    kv_end = min(kv_start + kv_block, key.shape[-2])
                    if bool(allowed[batch, head, q_start:q_end, kv_start:kv_end].any()):
                        retained_spans.append((kv_start, kv_end))
                if not retained_spans:
                    continue
                row_max = torch.full((q_end - q_start,), -torch.inf, device=query.device)
                logits_by_span = []
                for kv_start, kv_end in retained_spans:
                    logits = q_part @ key[batch, head, kv_start:kv_end].float().transpose(0, 1)
                    logits = logits * float(scaling)
                    valid_tile = allowed[batch, head, q_start:q_end, kv_start:kv_end]
                    if additive is not None:
                        logits = logits + additive[batch, head, q_start:q_end, kv_start:kv_end].float()
                    logits = logits.masked_fill(~valid_tile, -torch.inf)
                    logits_by_span.append((kv_start, kv_end, logits))
                    row_max = torch.maximum(row_max, logits.max(dim=-1).values)
                row_sum = torch.zeros_like(row_max)
                weighted = torch.zeros((q_end - q_start, value.shape[-1]), dtype=torch.float32, device=query.device)
                for kv_start, kv_end, logits in logits_by_span:
                    weights = torch.exp(logits - row_max[:, None])
                    row_sum += weights.sum(dim=-1)
                    weighted += weights @ value[batch, head, kv_start:kv_end].float()
                block_output = weighted / row_sum.clamp_min(1.0e-12)[:, None]
                output[batch, q_start:q_end, head] = block_output.to(query.dtype)
    return output


class FreshRoutingAttention:
    """Attention override that recomputes a fresh logical mask on every call."""

    def __init__(self, config: FreshRoutingConfig) -> None:
        self.config = config
        self.stats = FreshRoutingStats()
        self._previous_retained_tiles: dict[int, set[tuple[Any, ...]]] = {}

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
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        timed_physical = self.config.execution == "physical" and query.is_cuda
        if timed_physical:
            torch.cuda.synchronize(query.device)
        timing_start = time.perf_counter()
        if self.config.execution == "logical":
            expanded_key, expanded_value, scores, valid = _prepare_attention_scores(
                module, query, key, value, attention_mask,
                scaling=scaling, is_causal=is_causal, sliding_window=sliding_window,
            )
            scale = float(query.shape[-1] ** -0.5 if scaling is None else scaling)
        else:
            expanded_key, expanded_value, valid, scale = _prepare_attention_inputs(
                module, query, key, value, attention_mask,
                scaling=scaling, is_causal=is_causal, sliding_window=sliding_window,
            )
            scores = None
        prefix_length = max(0, expanded_key.shape[-2] - query.shape[-2])
        runtime = getattr(module, "_blasst_2d_runtime", None)
        # DiffusionGemma's BLASST callback returns zero because it denotes an
        # always-dense sink policy, not the semantic prefix/canvas boundary.
        # Its decoder KV is encoder prefix followed by the active canvas, so
        # the actual region boundary is KV length minus query length. Fast-
        # dLLM's cache-aware callback does encode its semantic boundary.
        if (
            self.config.adapter != "diffusion_gemma"
            and runtime is not None
            and runtime.dense_kv_prefix_extractor is not None
        ):
            prefix_length = int(runtime.dense_kv_prefix_extractor(
                module, query, key, value, attention_mask, kwargs
            ))
        rows = sol_attention_proxy_rows(
            module, query, key, value, attention_mask,
            q_block_size=self.config.q_block_size,
            kv_block_size=self.config.kv_block_size,
            prefix_length=prefix_length,
            scaling=scaling, is_causal=is_causal, sliding_window=sliding_window,
            prepared_key=expanded_key, valid_pair_mask=valid,
            global_kv_tiling=self.config.combined_region_population,
        )
        # Count every structurally valid physical tile before applying the
        # region policy.  For prefix-only routing, canvas tiles remain dense
        # and therefore belong in the denominator but never in the skipped
        # numerator.  Summing these integer counters across calls correctly
        # weights later, longer decoding canvases.
        all_physical: set[tuple[int, int, int, int, int]] = set()
        all_region_totals: dict[str, int] = {"prefix": 0, "canvas": 0}
        for row in rows:
            batch, head = int(row["batch"]), int(row["head"])
            q_start = int(row["query_start"])
            for region_name in ("prefix", "canvas"):
                for kv_start, kv_end in row[f"{region_name}_spans"]:
                    all_physical.add((
                        batch,
                        head,
                        q_start // self.config.q_block_size,
                        int(kv_start),
                        int(kv_end),
                    ))
                    all_region_totals[region_name] += 1
        allowed = torch.zeros_like(valid)
        # A region-specific ablation routes only its named region; the other
        # region remains native dense.  The all-KV reproduction starts with no
        # implicit sink so both regions are selected by the same rule.
        if self.config.region == "canvas" and prefix_length:
            allowed[..., :prefix_length] = valid[..., :prefix_length]
        elif self.config.region == "prefix_only" and prefix_length < valid.shape[-1]:
            allowed[..., prefix_length:] = valid[..., prefix_length:]
        attention_type = _attention_type(module, sliding_window)
        logical_total = logical_kept = 0
        physical: dict[tuple[int, int, int, int, int], bool] = {}
        beta_values = []
        fallback_rows = 0
        degenerate_row_count = 0
        region_totals: dict[str, list[int]] = {}
        head_totals: dict[int, list[int]] = {}
        retained_tile_ids: set[tuple[Any, ...]] = set()
        for row in rows:
            batch, head = int(row["batch"]), int(row["head"])
            q_start = int(row["query_start"])
            q_end = q_start + int(row["query_size"])
            if self.config.combined_region_population and self.config.region == "all":
                entries: list[tuple[str, int, float, tuple[int, int]]] = []
                for region_name in ("prefix", "canvas"):
                    if region_name == "prefix" and not prefix_length:
                        continue
                    values = np.asarray(row[region_name], dtype=np.float64)
                    spans = row[f"{region_name}_spans"]
                    entries.extend(
                        (region_name, index, float(value), spans[index])
                        for index, value in enumerate(values)
                    )
                if not entries:
                    continue
                combined_values = np.asarray([item[2] for item in entries], dtype=np.float64)
                degenerate = len(combined_values) < 2 or float(combined_values.std(ddof=0)) < 1.0e-8
                if degenerate and self.config.mode in ("gaussian", "profiled"):
                    keep = np.ones(len(combined_values), dtype=bool)
                    beta = float("nan")
                    fallback = False
                else:
                    keep, beta, fallback = _keep_vector(
                        combined_values,
                        self.config,
                        attention_type,
                        random_identity=(self.stats.calls, int(getattr(module, "layer_idx", -1)), batch, head, q_start, "all"),
                    )
                if np.isfinite(beta):
                    beta_values.append(beta)
                fallback_rows += int(fallback)
                degenerate_row_count += int(degenerate)
                self.stats.degenerate_rows += int(degenerate)
                self.stats.routing_rows += 1
                retained_in_row = int(keep.sum())
                histogram_key = str(retained_in_row)
                self.stats.retained_tiles_per_row_histogram[histogram_key] = self.stats.retained_tiles_per_row_histogram.get(histogram_key, 0) + 1
                head_entry = head_totals.setdefault(head, [0, 0, 0, 0])
                head_entry[0] += len(entries)
                head_entry[1] += retained_in_row
                head_entry[2] += 1
                head_entry[3] += int(fallback)
                for index, (region_name, _region_index, _value, (kv_start, kv_end)) in enumerate(entries):
                    logical_total += 1
                    logical_kept += int(keep[index])
                    region_entry = region_totals.setdefault(region_name, [0, 0])
                    region_entry[0] += 1
                    region_entry[1] += int(keep[index])
                    if keep[index]:
                        retained_tile_ids.add((batch, head, q_start, region_name, int(kv_start), int(kv_end)))
                        allowed[batch, head, q_start:q_end, kv_start:kv_end] |= valid[batch, head, q_start:q_end, kv_start:kv_end]
                    physical_key = (batch, head, q_start // self.config.q_block_size, int(kv_start), int(kv_end))
                    physical[physical_key] = physical.get(physical_key, False) or bool(keep[index])
                continue
            region_names = ("prefix", "canvas") if self.config.region == "all" else (
                "prefix" if self.config.region == "prefix_only" else self.config.region,
            )
            for region_name in region_names:
                if region_name == "prefix" and not prefix_length:
                    continue
                values = np.asarray(row[region_name], dtype=np.float64)
                spans = row[f"{region_name}_spans"]
                if not len(values):
                    continue
                degenerate = len(values) < 2 or float(values.std(ddof=0)) < 1.0e-8
                if degenerate and self.config.mode in ("gaussian", "profiled"):
                    keep = np.ones(len(values), dtype=bool)
                    beta = float("nan")
                    fallback = False
                else:
                    keep, beta, fallback = _keep_vector(
                        values, self.config, attention_type,
                        random_identity=(self.stats.calls, int(getattr(module, "layer_idx", -1)), batch, head, q_start, region_name),
                    )
                if np.isfinite(beta):
                    beta_values.append(beta)
                fallback_rows += int(fallback)
                degenerate_row_count += int(degenerate)
                self.stats.degenerate_rows += int(degenerate)
                self.stats.routing_rows += 1
                retained_in_row = int(keep.sum())
                histogram_key = str(retained_in_row)
                self.stats.retained_tiles_per_row_histogram[histogram_key] = (
                    self.stats.retained_tiles_per_row_histogram.get(histogram_key, 0) + 1
                )
                head_entry = head_totals.setdefault(head, [0, 0, 0, 0])
                head_entry[0] += len(values)
                head_entry[1] += retained_in_row
                head_entry[2] += 1
                head_entry[3] += int(fallback)
                region_entry = region_totals.setdefault(region_name, [0, 0])
                region_entry[0] += len(values)
                region_entry[1] += int(keep.sum())
                for index, (kv_start, kv_end) in enumerate(spans):
                    logical_total += 1
                    logical_kept += int(keep[index])
                    if keep[index]:
                        retained_tile_ids.add((batch, head, q_start, region_name, int(kv_start), int(kv_end)))
                        allowed[batch, head, q_start:q_end, kv_start:kv_end] |= valid[
                            batch, head, q_start:q_end, kv_start:kv_end
                        ]
                    physical_key = (
                        batch,
                        head,
                        q_start // self.config.q_block_size,
                        int(kv_start),
                        int(kv_end),
                    )
                    physical[physical_key] = physical.get(physical_key, False) or bool(keep[index])
        if timed_physical:
            torch.cuda.synchronize(query.device)
        timing_mask_done = time.perf_counter()
        if scores is not None:
            sparse_scores = scores.masked_fill(valid & ~allowed, -torch.inf)
            dense_output, _ = _finish_eager_attention(query, expanded_value, scores, valid, dropout, module.training)
            sparse_output, _ = _finish_eager_attention(query, expanded_value, sparse_scores, allowed, dropout, module.training)
            has_key = valid.any(dim=-1, keepdim=True)
            dense_prob = torch.softmax(torch.where(has_key, scores, torch.zeros_like(scores)), dim=-1, dtype=torch.float32)
            dense_prob = torch.where(has_key, dense_prob, torch.zeros_like(dense_prob))
            retained_mass = (dense_prob * allowed).sum(dim=-1)
            active_rows = has_key.squeeze(-1)
            mass_sum = float(retained_mass[active_rows].sum().item())
            mass_rows = int(active_rows.sum().item())
            relative_error = float(
                torch.linalg.vector_norm((sparse_output - dense_output).float()).item()
                / max(torch.linalg.vector_norm(dense_output.float()).item(), 1.0e-12)
            )
        else:
            sparse_output = _physical_sparse_attention(
                query, expanded_key, expanded_value, allowed, attention_mask,
                scaling=scale, dropout=dropout, training=module.training,
                block_size=(self.config.q_block_size, self.config.kv_block_size),
            )
            mass_sum = 0.0
            mass_rows = 0
            relative_error = math.nan
        if timed_physical:
            torch.cuda.synchronize(query.device)
        timing_attention_done = time.perf_counter()
        self.stats.calls += 1
        self.stats.logical_candidate_tiles += logical_total
        self.stats.logical_retained_tiles += logical_kept
        self.stats.physical_candidate_tiles += len(physical)
        self.stats.physical_retained_tiles += sum(physical.values())
        self.stats.physical_total_tiles += len(all_physical)
        self.stats.physical_skipped_tiles += len(physical) - sum(physical.values())
        self.stats.dense_attention_mass_sum += mass_sum
        self.stats.dense_attention_mass_rows += mass_rows
        if math.isfinite(relative_error):
            self.stats.output_relative_error_sum += relative_error
            self.stats.output_relative_error_calls += 1
        self.stats.fallback_rows += fallback_rows
        if timed_physical:
            self.stats.proxy_mask_seconds += timing_mask_done - timing_start
            self.stats.sparse_attention_seconds += timing_attention_done - timing_mask_done
        layer = int(getattr(module, "layer_idx", -1))
        step = int(getattr(runtime, "current_denoising_iteration", -1)) if runtime is not None else -1
        previous = self._previous_retained_tiles.get(layer)
        overlap_jaccard = None
        if previous is not None:
            intersection = len(previous & retained_tile_ids)
            union = len(previous | retained_tile_ids)
            self.stats.mask_overlap_intersection += intersection
            self.stats.mask_overlap_union += union
            self.stats.mask_overlap_comparisons += 1
            overlap_jaccard = intersection / union if union else 1.0
        self._previous_retained_tiles[layer] = retained_tile_ids
        aggregate_counts = {
            "candidate_tiles": logical_total,
            "retained_tiles": logical_kept,
            "physical_total_tiles": len(all_physical),
            "physical_candidate_tiles": len(physical),
            "physical_retained_tiles": int(sum(physical.values())),
            "physical_skipped_tiles": int(len(physical) - sum(physical.values())),
            "routing_rows": sum(values[2] for values in head_totals.values()),
            "fallback_rows": fallback_rows,
        }
        _merge_counts(self.stats.by_attention_type, attention_type, **aggregate_counts)
        _merge_counts(self.stats.by_layer, str(layer), **aggregate_counts)
        _merge_counts(self.stats.by_step, str(step), **aggregate_counts)
        for head, (candidates, retained, rows_count, head_fallbacks) in head_totals.items():
            _merge_counts(
                self.stats.by_head, f"{layer}|{head}",
                candidate_tiles=candidates, retained_tiles=retained,
                routing_rows=rows_count, fallback_rows=head_fallbacks,
            )
        call_region_counts = {
            name: {"candidate_tiles": int(total), "retained_tiles": int(retained)}
            for name, (total, retained) in region_totals.items()
        }
        for region_name, counts in call_region_counts.items():
            entry = self.stats.region_counts.setdefault(region_name, {"candidate_tiles": 0, "retained_tiles": 0})
            entry["candidate_tiles"] += counts["candidate_tiles"]
            entry["retained_tiles"] += counts["retained_tiles"]
        call_all_region_counts: dict[str, dict[str, int]] = {}
        for region_name in ("prefix", "canvas"):
            total_tiles = int(all_region_totals.get(region_name, 0))
            routed, routed_retained = region_totals.get(region_name, (0, 0))
            skipped = int(routed - routed_retained)
            retained = int(total_tiles - skipped)
            if not total_tiles:
                continue
            entry = self.stats.all_region_counts.setdefault(region_name, {
                "total_tiles": 0,
                "candidate_tiles": 0,
                "retained_tiles": 0,
                "skipped_tiles": 0,
            })
            entry["total_tiles"] += total_tiles
            entry["candidate_tiles"] += int(routed)
            entry["retained_tiles"] += retained
            entry["skipped_tiles"] += skipped
            call_all_region_counts[region_name] = {
                "total_tiles": total_tiles,
                "candidate_tiles": int(routed),
                "retained_tiles": retained,
                "skipped_tiles": skipped,
            }
        self.stats.per_call.append({
            "call": self.stats.calls - 1,
            "layer": layer,
            "head_ids": sorted(int(head) for head in head_totals),
            "denoising_step": step,
            "attention_type": attention_type,
            "logical_candidate_tiles": logical_total,
            "logical_retained_tiles": logical_kept,
            "physical_candidate_tiles": len(physical),
            "physical_retained_tiles": int(sum(physical.values())),
            "physical_total_tiles": len(all_physical),
            "physical_skipped_tiles": int(len(physical) - sum(physical.values())),
            "retained_dense_attention_mass": mass_sum / mass_rows if mass_rows else math.nan,
            "valid_rows": mass_rows,
            "retained_attention_mass_rows": mass_rows,
            "attention_output_relative_error": relative_error if math.isfinite(relative_error) else None,
            "beta": float(np.mean(beta_values)) if beta_values else None,
            "region_counts": call_region_counts,
            "all_region_counts": call_all_region_counts,
            "fallback_rows": fallback_rows,
            "degenerate_rows": degenerate_row_count,
            "previous_call_mask_jaccard": overlap_jaccard,
            "proxy_mask_seconds": timing_mask_done - timing_start if timed_physical else None,
            "sparse_attention_seconds": timing_attention_done - timing_mask_done if timed_physical else None,
        })
        return sparse_output, None


def routing_condition_name(mode: str, density: float) -> str:
    return f"{mode}_rho{int(round(density * 100)):02d}"
