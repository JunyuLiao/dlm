"""Dense attention observations for block-proxy and temporal analysis.

The tensors received here are the tensors passed to the model's registered
attention implementation.  For DiffusionGemma this means Q and K have already
undergone the model-specific q_norm/k_norm and rotary positional embedding.
The attention function's scaling and mask are applied below before any proxy
or reference statistic is measured.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


TILE_COLUMNS = (
    "request_id",
    "canvas_id",
    "denoising_step",
    "num_denoising_steps",
    "layer",
    "head",
    "attention_type",
    "query_block",
    "query_start",
    "query_size",
    "kv_block",
    "kv_start",
    "kv_size",
    "kv_region",
    "proxy_mean",
    "proxy_max",
    "softmax_mass",
    "blasst_margin",
    "valid_pairs",
    "valid_queries",
    "sequence_length",
    "prefix_length",
)

STATE_COLUMNS = (
    "request_id",
    "denoising_step",
    "layer",
    "head",
    "attention_type",
    "q_relative_l2",
    "q_cosine",
    "k_relative_l2",
    "k_cosine",
)


@dataclass(frozen=True)
class AttentionObservationConfig:
    q_block_size: int = 64
    kv_block_size: int = 64
    physical_q_tile_size: int = 128
    physical_kv_tile_size: int = 64
    layers: tuple[int, ...] = ()
    heads: tuple[int, ...] = ()
    save_tile_stats: bool = True
    analyze_temporal: bool = True
    analyze_prefix: bool = True
    max_tile_records: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "q_block_size",
            "kv_block_size",
            "physical_q_tile_size",
            "physical_kv_tile_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(value < 0 for value in self.layers):
            raise ValueError("layer indices must be non-negative")
        if any(value < 0 for value in self.heads):
            raise ValueError("head indices must be non-negative")
        if self.max_tile_records is not None and self.max_tile_records <= 0:
            raise ValueError("max_tile_records must be positive")


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, heads, length, width = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, heads, repeats, length, width)
        .reshape(batch, heads * repeats, length, width)
    )


def attention_scores_and_validity(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float | None,
    is_causal: bool | None,
    sliding_window: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the native pre-softmax QK semantics without changing output."""
    repeats = int(getattr(module, "num_key_value_groups", 1))
    if key.shape[1] != query.shape[1]:
        if key.shape[1] * repeats != query.shape[1]:
            raise ValueError("query/KV head mapping does not match num_key_value_groups")
        key = _repeat_kv(key, repeats)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * float(scaling)
    batch, heads, q_length, kv_length = scores.shape
    valid = torch.ones_like(scores, dtype=torch.bool)
    if attention_mask is not None:
        mask = attention_mask[..., :kv_length].to(device=scores.device)
        if mask.dtype == torch.bool:
            mask_valid = mask
        elif mask.is_floating_point():
            mask_valid = torch.isfinite(mask) & (mask > -1.0e4)
            scores = scores + mask
        else:
            mask_valid = mask != 0
        valid &= torch.broadcast_to(mask_valid, scores.shape)
    causal = attention_mask is None and q_length > 1 if is_causal is None else is_causal
    if causal:
        diagonal = kv_length - q_length
        valid &= torch.ones(
            (q_length, kv_length), dtype=torch.bool, device=scores.device
        ).tril(diagonal=diagonal)
    # Transformers generally materializes the sliding mask.  Retain this
    # fallback for attention implementations which pass only the window size.
    if sliding_window is not None and (attention_mask is None or attention_mask.ndim < 4):
        q_positions = torch.arange(q_length, device=scores.device) + (kv_length - q_length)
        k_positions = torch.arange(kv_length, device=scores.device)
        valid &= k_positions[None, :] >= q_positions[:, None] - int(sliding_window) + 1
    scores = scores.masked_fill(~valid, -torch.inf)
    return key, scores, valid


def dense_probabilities(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    has_key = valid.any(dim=-1, keepdim=True)
    safe_scores = torch.where(has_key, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1, dtype=torch.float32)
    return torch.where(has_key, probabilities, torch.zeros_like(probabilities))


def _cosine_and_relative(current: torch.Tensor, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    current = current.float().flatten(start_dim=-2)
    previous = previous.float().flatten(start_dim=-2)
    difference = torch.linalg.vector_norm(current - previous, dim=-1)
    denominator = torch.linalg.vector_norm(previous, dim=-1).clamp_min(1.0e-12)
    relative = difference / denominator
    cosine = torch.nn.functional.cosine_similarity(current, previous, dim=-1, eps=1.0e-12)
    return relative, cosine


class AttentionObserver:
    """Collect compact tile and state records while native dense attention runs."""

    def __init__(self, config: AttentionObservationConfig) -> None:
        self.config = config
        self.tile_records: list[dict[str, Any]] = []
        self.state_records: list[dict[str, Any]] = []
        self.metadata_context: dict[str, Any] = {}
        self.current_step = -1
        self.current_iteration = -1
        self._previous_state: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.calls_seen = 0
        self.calls_recorded = 0
        self.total_tile_records = 0
        self.total_state_records = 0

    def begin_request(self, request_id: str, *, num_denoising_steps: int | None = None) -> None:
        self.metadata_context = {
            "request_id": str(request_id),
            "canvas_id": str(request_id),
            "num_denoising_steps": -1 if num_denoising_steps is None else int(num_denoising_steps),
        }
        self.current_step = -1
        self.current_iteration = -1
        self._previous_state.clear()

    def begin_forward(self, decoder_input_ids: torch.Tensor | None, self_conditioning_logits: Any) -> None:
        del decoder_input_ids
        self.current_step += 1
        if self_conditioning_logits is None:
            self.current_iteration = 0
        else:
            self.current_iteration = max(1, self.current_iteration + 1)

    def _selected(self, layer: int) -> bool:
        return not self.config.layers or layer in self.config.layers

    @staticmethod
    def _attention_type(module: torch.nn.Module, sliding_window: int | None) -> str:
        return "local" if (
            bool(getattr(module, "is_sliding", False))
            or getattr(module, "layer_type", "") == "sliding_attention"
            or sliding_window is not None
        ) else "global"

    def observe(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        scaling: float | None = None,
        is_causal: bool | None = None,
        sliding_window: int | None = None,
        **_: Any,
    ) -> None:
        del value
        self.calls_seen += 1
        layer = int(getattr(module, "layer_idx", -1))
        if not self._selected(layer):
            return
        self.calls_recorded += 1
        key, scores, valid = attention_scores_and_validity(
            module,
            query,
            key,
            attention_mask,
            scaling=scaling,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        attention_type = self._attention_type(module, sliding_window)
        prefix_length = max(0, key.shape[-2] - query.shape[-2])
        if self.config.analyze_temporal:
            self._record_state(query, key[..., prefix_length:, :], layer, attention_type)
        if self.config.save_tile_stats:
            self._record_tiles(
                query,
                key,
                scores,
                valid,
                layer,
                attention_type,
                prefix_length,
                float(scaling if scaling is not None else query.shape[-1] ** -0.5),
            )

    def _record_state(
        self,
        query: torch.Tensor,
        canvas_key: torch.Tensor,
        layer: int,
        attention_type: str,
    ) -> None:
        request_id = str(self.metadata_context.get("request_id", "unknown"))
        state_key = (request_id, layer)
        previous = self._previous_state.get(state_key)
        if previous is not None and previous[0].shape == query.shape and previous[1].shape == canvas_key.shape:
            q_relative, q_cosine = _cosine_and_relative(query, previous[0].to(query.device))
            k_relative, k_cosine = _cosine_and_relative(canvas_key, previous[1].to(canvas_key.device))
            head_indices = self._head_indices(query.shape[1])
            values = torch.stack((q_relative, q_cosine, k_relative, k_cosine), dim=-1)
            values = values[:, head_indices].detach().cpu().numpy()
            for batch_index in range(values.shape[0]):
                for output_index, head in enumerate(head_indices):
                    row = {
                        "request_id": request_id,
                        "denoising_step": self.current_step,
                        "layer": layer,
                        "head": head,
                        "attention_type": attention_type,
                        "q_relative_l2": float(values[batch_index, output_index, 0]),
                        "q_cosine": float(values[batch_index, output_index, 1]),
                        "k_relative_l2": float(values[batch_index, output_index, 2]),
                        "k_cosine": float(values[batch_index, output_index, 3]),
                    }
                    self.state_records.append(row)
                    self.total_state_records += 1
        # Only the evolving canvas is retained; cached prefix K is read-only and
        # would dominate memory for long-context observations.
        self._previous_state[state_key] = (query.detach().cpu(), canvas_key.detach().cpu())

    def _head_indices(self, count: int) -> list[int]:
        if not self.config.heads:
            return list(range(count))
        invalid = [head for head in self.config.heads if head >= count]
        if invalid:
            raise ValueError(f"head selection exceeds available heads: {invalid}")
        return list(self.config.heads)

    def _record_tiles(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        layer: int,
        attention_type: str,
        prefix_length: int,
        scaling: float,
    ) -> None:
        probabilities = dense_probabilities(scores, valid)
        batch, heads, query_length, sequence_length = scores.shape
        head_indices = self._head_indices(heads)
        q_block = self.config.q_block_size
        kv_block = self.config.kv_block_size
        regions = []
        if self.config.analyze_prefix and prefix_length:
            regions.append(("prefix", 0, prefix_length))
        canvas_start = prefix_length if self.config.analyze_prefix else 0
        if canvas_start < sequence_length:
            regions.append(("canvas", canvas_start, sequence_length))
        request = {
            "request_id": str(self.metadata_context.get("request_id", "unknown")),
            "canvas_id": str(self.metadata_context.get("canvas_id", "unknown")),
            "denoising_step": self.current_step,
            "num_denoising_steps": int(self.metadata_context.get("num_denoising_steps", -1)),
            "layer": layer,
            "attention_type": attention_type,
            "sequence_length": sequence_length,
            "prefix_length": prefix_length,
        }
        for q_start in range(0, query_length, q_block):
            q_end = min(q_start + q_block, query_length)
            for region, region_start, region_end in regions:
                for kv_start in range(region_start, region_end, kv_block):
                    kv_end = min(kv_start + kv_block, region_end)
                    tile_valid = valid[:, head_indices, q_start:q_end, kv_start:kv_end]
                    counts = tile_valid.sum(dim=(-2, -1))
                    eligible = counts > 0
                    if not bool(eligible.any()):
                        continue
                    tile_scores = scores[:, head_indices, q_start:q_end, kv_start:kv_end]
                    # Literal Sol-Attn proxy. At a partially masked local-window
                    # boundary, only Q/K positions participating in at least one
                    # valid pair are pooled, then their means are dotted. This
                    # preserves Mean(Q_i)Mean(K_j)^T rather than substituting a
                    # mean over the irregular set of valid pairwise logits.
                    q_participates = tile_valid.any(dim=-1)
                    k_participates = tile_valid.any(dim=-2)
                    q_vectors = query[:, head_indices, q_start:q_end, :].float()
                    k_vectors = key[:, head_indices, kv_start:kv_end, :].float()
                    q_means = (
                        q_vectors * q_participates[..., None]
                    ).sum(dim=-2) / q_participates.sum(dim=-1, keepdim=True).clamp_min(1)
                    k_means = (
                        k_vectors * k_participates[..., None]
                    ).sum(dim=-2) / k_participates.sum(dim=-1, keepdim=True).clamp_min(1)
                    means = (q_means * k_means).sum(dim=-1) * scaling
                    maxima = tile_scores.masked_fill(~tile_valid, -torch.inf).amax(dim=(-2, -1)).float()
                    mass = probabilities[:, head_indices, q_start:q_end, kv_start:kv_end]
                    valid_queries = tile_valid.any(dim=-1).sum(dim=-1)
                    mass = mass.sum(dim=(-2, -1)) / valid_queries.clamp_min(1).float()
                    # Exact existing BLASST score: local tile maximum relative
                    # to the running per-query maximum preceding this KV tile.
                    preceding = scores[:, head_indices, q_start:q_end, :kv_start]
                    preceding_valid = valid[:, head_indices, q_start:q_end, :kv_start]
                    if kv_start:
                        running = preceding.masked_fill(~preceding_valid, -torch.inf).amax(dim=-1)
                        local_per_query = tile_scores.masked_fill(~tile_valid, -torch.inf).amax(dim=-1)
                        margins = (local_per_query - running).masked_fill(~tile_valid.any(dim=-1), -torch.inf)
                        blasst_margin = margins.amax(dim=-1).float()
                    else:
                        blasst_margin = torch.full_like(maxima, torch.inf)
                    packed = torch.stack(
                        (
                            means,
                            maxima,
                            mass,
                            blasst_margin,
                            counts.float(),
                            valid_queries.float(),
                            eligible.float(),
                        ),
                        dim=-1,
                    ).detach().cpu().numpy()
                    for batch_index in range(batch):
                        for output_index, head in enumerate(head_indices):
                            if not packed[batch_index, output_index, 6]:
                                continue
                            row = {
                                **request,
                                "head": head,
                                "query_block": q_start // q_block,
                                "query_start": q_start,
                                "query_size": q_end - q_start,
                                "kv_block": kv_start // kv_block,
                                "kv_start": kv_start,
                                "kv_size": kv_end - kv_start,
                                "kv_region": region,
                                "proxy_mean": float(packed[batch_index, output_index, 0]),
                                "proxy_max": float(packed[batch_index, output_index, 1]),
                                "softmax_mass": float(packed[batch_index, output_index, 2]),
                                "blasst_margin": float(packed[batch_index, output_index, 3]),
                                "valid_pairs": int(packed[batch_index, output_index, 4]),
                                "valid_queries": int(packed[batch_index, output_index, 5]),
                            }
                            self.tile_records.append(row)
                            self.total_tile_records += 1
                            if (
                                self.config.max_tile_records is not None
                                and self.total_tile_records > self.config.max_tile_records
                            ):
                                raise RuntimeError("max_tile_records limit exceeded; narrow explicit sampling")

    @staticmethod
    def _columnar(records: Iterable[Mapping[str, Any]], columns: tuple[str, ...]) -> dict[str, np.ndarray]:
        records = list(records)
        result: dict[str, np.ndarray] = {}
        for column in columns:
            values = [row[column] for row in records]
            if values and isinstance(values[0], str):
                result[column] = np.asarray(values, dtype=np.str_)
            else:
                result[column] = np.asarray(values)
        return result

    def export(self, output_dir: str | Path, run_metadata: Mapping[str, Any]) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.tile_records or not (output / "raw").exists():
            np.savez_compressed(output / "tile_records.npz", **self._columnar(self.tile_records, TILE_COLUMNS))
        if self.state_records or not (output / "raw").exists():
            np.savez_compressed(output / "state_records.npz", **self._columnar(self.state_records, STATE_COLUMNS))
        import json

        metadata = {
            "schema_version": 1,
            "capture_boundary": (
                "registered attention input: post q_norm/k_norm and post RoPE; "
                "model scaling and native attention mask applied by observer"
            ),
            "mean_proxy_definition": (
                "Mean(participating post-norm/post-RoPE Q) @ Mean(participating "
                "post-norm/post-RoPE K) * model scale; invalid local-window positions excluded"
            ),
            "block_max_definition": "maximum valid scaled QK logit in each tile",
            "output_semantics": "native registered dense attention output returned unchanged",
            "calls_seen": self.calls_seen,
            "calls_recorded": self.calls_recorded,
            "tile_records": self.total_tile_records,
            "state_records": self.total_state_records,
            "sampling": asdict(self.config),
            **dict(run_metadata),
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def export_shard(self, output_dir: str | Path, shard_id: str) -> tuple[Path, Path]:
        """Persist and release one request's records for bounded-memory runs."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(shard_id))
        tile_path = output / f"tile_records_{safe_id}.npz"
        state_path = output / f"state_records_{safe_id}.npz"
        if tile_path.exists() or state_path.exists():
            raise FileExistsError(f"observation shard already exists: {safe_id}")
        np.savez_compressed(tile_path, **self._columnar(self.tile_records, TILE_COLUMNS))
        np.savez_compressed(state_path, **self._columnar(self.state_records, STATE_COLUMNS))
        self.tile_records.clear()
        self.state_records.clear()
        return tile_path, state_path
