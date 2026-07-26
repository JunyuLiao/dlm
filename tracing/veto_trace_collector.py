"""Bounded writer for sampled rich veto-row traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from blasst.flash_attention import RichPhysicalTileTrace
from .blasst_trace_collector import noise_bucket
from .veto_trace_schema import (
    METADATA_DTYPE,
    ROW_FLOAT_FIELDS,
    ROW_UINT8_FIELDS,
    SCHEMA_VERSION,
    validate_or_write_manifest,
)


@dataclass(frozen=True)
class VetoTraceContext:
    sample_ids: Sequence[str]
    request_ids: Sequence[str]
    seeds: Sequence[int]
    sequence_length: int
    diffusion_block: int
    denoising_step: int
    remaining_mask_ratios: Sequence[float]
    thresholds: Sequence[float]
    token_states: torch.Tensor
    query_heads: int

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        if not all(
            len(values) == count
            for values in (
                self.request_ids,
                self.seeds,
                self.remaining_mask_ratios,
                self.thresholds,
            )
        ):
            raise ValueError("all per-sample context fields must have equal length")
        if self.token_states.shape != (count, self.sequence_length):
            raise ValueError("token_states must have shape [batch, sequence_length]")


class VetoTraceCollector:
    """Collect sampled per-row statistics without affecting normal execution."""

    def __init__(
        self,
        output_dir: str | Path,
        context: VetoTraceContext,
        *,
        sample_fraction: float = 0.005,
        query_tiles_per_sample: int | None = None,
        q_block_size: int = 128,
        kv_block_size: int = 64,
        shard_records: int = 25_000,
        device_buffer_tiles: int = 32,
        stable_query_delta: float = 0.05,
    ) -> None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if q_block_size <= 0 or kv_block_size <= 0:
            raise ValueError("block sizes must be positive")
        if shard_records <= 0 or device_buffer_tiles <= 0:
            raise ValueError("buffer sizes must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.sample_fraction = sample_fraction
        self.query_tiles_per_sample = query_tiles_per_sample
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.shard_records = shard_records
        self.device_buffer_tiles = device_buffer_tiles
        self.stable_query_delta = stable_query_delta
        self._pending: list[tuple[int, int, int, RichPhysicalTileTrace, tuple[int, ...]]] = []
        self._metadata: list[tuple[object, ...]] = []
        self._row_data: dict[str, list[np.ndarray]] = {
            name: [] for name in (*ROW_FLOAT_FIELDS, *ROW_UINT8_FIELDS)
        }
        self._shard_index = len(list(self.output_dir.glob("veto-trace-*.npz")))
        num_query_tiles = math.ceil(context.sequence_length / q_block_size)
        if query_tiles_per_sample is not None:
            if not 1 <= query_tiles_per_sample <= num_query_tiles:
                raise ValueError("query_tiles_per_sample must be within the query-tile count")
            self._selected_query_tiles = {
                sample_id: frozenset(
                    sorted(
                        range(num_query_tiles),
                        key=lambda query_tile: hashlib.blake2b(
                            f"{sample_id}:{query_tile}".encode(), digest_size=8
                        ).digest(),
                    )[:query_tiles_per_sample]
                )
                for sample_id in context.sample_ids
            }
        else:
            self._selected_query_tiles = None
        validate_or_write_manifest(
            self.output_dir,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            sample_fraction=sample_fraction,
            query_tiles_per_sample=query_tiles_per_sample,
            row_storage="float16",
            stable_query_delta=stable_query_delta,
            sampling_unit="sample_id+query_tile+kv_tile shared across steps/layers/heads",
        )

    def _selected(self, identity: str) -> bool:
        if self.sample_fraction == 1.0:
            return True
        value = int.from_bytes(
            hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little"
        )
        return value / 2**64 < self.sample_fraction

    def wants_tile(self, query_tile: int, kv_tile: int) -> bool:
        if self._selected_query_tiles is not None:
            return any(
                query_tile in self._selected_query_tiles[sample_id]
                for sample_id in self.context.sample_ids
            )
        return any(
            self._selected(f"{sample_id}:{query_tile}:{kv_tile}")
            for sample_id in self.context.sample_ids
        )

    def __call__(
        self, layer: int, query_tile: int, kv_tile: int, trace: RichPhysicalTileTrace
    ) -> None:
        batch, heads = trace.exact_skip.shape
        if batch != len(self.context.sample_ids) or heads != self.context.query_heads:
            raise ValueError("trace tensor shape does not match VetoTraceContext")
        selected = tuple(
            sample
            for sample in range(batch)
            if (
                query_tile in self._selected_query_tiles[self.context.sample_ids[sample]]
                if self._selected_query_tiles is not None
                else self._selected(
                    f"{self.context.sample_ids[sample]}:{query_tile}:{kv_tile}"
                )
            )
        )
        if not selected:
            return
        self._pending.append((layer, query_tile, kv_tile, trace, selected))
        if len(self._pending) >= self.device_buffer_tiles:
            self._drain()

    @staticmethod
    def _padded(values: np.ndarray, rows: int, fill: float | int) -> np.ndarray:
        if values.shape[-1] == rows:
            return values
        result = np.full((*values.shape[:-1], rows), fill, dtype=values.dtype)
        result[..., : values.shape[-1]] = values
        return result

    def _drain(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        row_cpu = {
            name: [getattr(item[3], name).float().cpu().numpy() for item in pending]
            for name in ROW_FLOAT_FIELDS
        }
        row_cpu["row_keep_vote"] = [
            item[3].row_keep_vote.to(torch.uint8).cpu().numpy() for item in pending
        ]
        scalar_names = (
            "exact_skip",
            "introduced_new_max",
            "v_mean_row_norm",
            "v_max_row_norm",
            "v_frobenius_per_sqrt_rows",
            "v_centroid_norm",
            "v_residual_max_norm",
            "v_per_dimension_variance",
        )
        scalar_cpu = {
            name: [getattr(item[3], name).float().cpu().numpy() for item in pending]
            for name in scalar_names
        }
        num_kv_tiles = math.ceil(self.context.sequence_length / self.kv_block_size)
        for index, (layer, query_tile, kv_tile, trace, selected) in enumerate(pending):
            q_start = query_tile * self.q_block_size
            q_end = min(q_start + self.q_block_size, self.context.sequence_length)
            for sample in selected:
                base_states = self.context.token_states[sample, q_start:q_end].cpu().numpy()
                ratio = float(self.context.remaining_mask_ratios[sample])
                for head in range(self.context.query_heads):
                    self._metadata.append(
                        (
                            self.context.sample_ids[sample],
                            self.context.request_ids[sample],
                            self.context.seeds[sample],
                            self.context.sequence_length,
                            self.context.diffusion_block,
                            self.context.denoising_step,
                            ratio,
                            noise_bucket(ratio),
                            layer,
                            head,
                            query_tile,
                            kv_tile,
                            num_kv_tiles - 1 - kv_tile,
                            trace.valid_query_rows,
                            self.context.thresholds[sample],
                            int(bool(scalar_cpu["exact_skip"][index][sample, head])),
                            int(bool(scalar_cpu["introduced_new_max"][index][sample, head])),
                            *(
                                float(scalar_cpu[name][index][sample, head])
                                for name in scalar_names[2:]
                            ),
                        )
                    )
                    for name in ROW_FLOAT_FIELDS:
                        values = row_cpu[name][index][sample, head]
                        padded = self._padded(values, self.q_block_size, np.nan)
                        # +inf is the expected first-tile row score because the
                        # running maximum starts at -inf. Preserve it without a
                        # noisy finite-to-FP16 overflow warning.
                        with np.errstate(over="ignore", invalid="ignore"):
                            stored = padded.astype(np.float16, copy=False)
                        self._row_data[name].append(stored)
                    votes = row_cpu["row_keep_vote"][index][sample, head]
                    self._row_data["row_keep_vote"].append(
                        self._padded(votes, self.q_block_size, 0).astype(np.uint8, copy=False)
                    )
                    states = base_states.copy().astype(np.uint8, copy=False)
                    delta = row_cpu["row_query_delta"][index][sample, head]
                    stable = (states == 2) & np.isfinite(delta) & (
                        delta <= self.stable_query_delta
                    )
                    states[stable] = 3
                    self._row_data["row_token_state"].append(
                        self._padded(states, self.q_block_size, 255).astype(np.uint8, copy=False)
                    )
                    if len(self._metadata) >= self.shard_records:
                        self._flush_rows()

    def _flush_rows(self) -> None:
        if not self._metadata:
            return
        path = self.output_dir / f"veto-trace-{self._shard_index:06d}.npz"
        arrays = {
            name: np.stack(values)
            for name, values in self._row_data.items()
        }
        np.savez_compressed(
            path,
            schema_version=np.array(SCHEMA_VERSION),
            metadata=np.array(self._metadata, dtype=METADATA_DTYPE),
            **arrays,
        )
        self._metadata.clear()
        for values in self._row_data.values():
            values.clear()
        self._shard_index += 1

    def flush(self) -> None:
        self._drain()
        self._flush_rows()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "VetoTraceCollector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
