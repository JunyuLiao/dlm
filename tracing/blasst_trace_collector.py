"""Incremental exact trace collector for the readable BLASST path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from blasst.flash_attention import PhysicalTileTrace
from .trace_schema import SCHEMA_VERSION, TRACE_DTYPE, validate_or_write_manifest


@dataclass(frozen=True)
class TraceContext:
    sample_ids: Sequence[str]
    request_ids: Sequence[str]
    seeds: Sequence[int]
    sequence_length: int
    diffusion_block: int
    denoising_iteration: int
    remaining_mask_ratios: Sequence[float]
    query_heads: int
    kv_heads: int

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        if not all(len(values) == count for values in (self.request_ids, self.seeds, self.remaining_mask_ratios)):
            raise ValueError("all per-sample trace context fields must have the same length")
        if self.query_heads <= 0 or self.kv_heads <= 0 or self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be a positive multiple of kv_heads")


def noise_bucket(ratio: float) -> str:
    if ratio >= 0.75:
        return "high"
    if ratio >= 0.25:
        return "mid"
    return "low"


class IncrementalTraceCollector:
    """Write sampled physical-tile records as bounded NPZ shards.

    Sampling is a stable hash of sample and spatial tile only. Consequently a
    selected coordinate retains every step/layer/head needed for relationship
    joins. No full QK matrices are retained.
    """

    def __init__(
        self,
        output_dir: str | Path,
        context: TraceContext,
        *,
        sample_fraction: float = 1.0,
        shard_records: int = 100_000,
        q_block_size: int = 128,
        kv_block_size: int = 64,
        local_radius_tiles: int = 1,
        device_buffer_tiles: int = 256,
        bitpack_masks: bool = False,
        quantize_log_scores: int | None = None,
    ) -> None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if shard_records <= 0:
            raise ValueError("shard_records must be positive")
        if quantize_log_scores not in (None, 4, 8):
            raise ValueError("quantize_log_scores must be None, 4, or 8")
        if device_buffer_tiles <= 0:
            raise ValueError("device_buffer_tiles must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.sample_fraction = sample_fraction
        self.shard_records = shard_records
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.local_radius_tiles = local_radius_tiles
        self.device_buffer_tiles = device_buffer_tiles
        self.bitpack_masks = bitpack_masks
        self.quantize_log_scores = quantize_log_scores
        self._rows: list[tuple[object, ...]] = []
        self._pending: list[tuple[int, int, int, PhysicalTileTrace, tuple[int, ...]]] = []
        self._shard_index = len(list(self.output_dir.glob("trace-*.npz")))
        validate_or_write_manifest(
            self.output_dir,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            sample_fraction=sample_fraction,
            bitpack_masks=bitpack_masks,
            quantize_log_scores=quantize_log_scores,
            sampling_unit="sample_id+query_tile+kv_tile (shared across steps/layers/heads)",
            note="output_contribution_norm is NaN unless a future cheap kernel-side measurement is supplied",
        )

    def _selected(self, identity: str) -> bool:
        if self.sample_fraction == 1.0:
            return True
        value = int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little")
        return value / 2**64 < self.sample_fraction

    def wants_tile(self, query_tile: int, kv_tile: int) -> bool:
        """Return whether any batch sample selected this spatial coordinate."""
        return any(
            self._selected(f"{sample_id}:{query_tile}:{kv_tile}")
            for sample_id in self.context.sample_ids
        )

    def __call__(self, layer: int, query_tile: int, kv_tile: int, trace: PhysicalTileTrace) -> None:
        batch, heads = trace.score.shape
        if batch != len(self.context.sample_ids) or heads != self.context.query_heads:
            raise ValueError("trace tensor shape does not match TraceContext")
        selected_samples = tuple(
            sample
            for sample in range(batch)
            if self._selected(f"{self.context.sample_ids[sample]}:{query_tile}:{kv_tile}")
        )
        if not selected_samples:
            return
        self._pending.append((layer, query_tile, kv_tile, trace, selected_samples))
        if len(self._pending) >= self.device_buffer_tiles:
            self._drain_device_buffer()

    def _drain_device_buffer(self) -> None:
        if not self._pending:
            return
        names = (
            "score", "log_score", "exact_skip", "introduced_new_max",
            "row_max", "row_q50", "row_q90", "row_q99",
        )
        cpu = {
            name: torch.stack([getattr(item[3], name) for item in self._pending]).float().cpu().numpy()
            for name in names
        }
        pending, self._pending = self._pending, []
        for tile_index, (layer, query_tile, kv_tile, trace, selected_samples) in enumerate(pending):
            self._consume_cpu_tile(
                layer, query_tile, kv_tile, trace.valid_query_rows, selected_samples,
                tuple(cpu[name][tile_index] for name in names),
            )

    def _consume_cpu_tile(
        self,
        layer: int,
        query_tile: int,
        kv_tile: int,
        valid_query_rows: int,
        selected_samples: tuple[int, ...],
        cpu: tuple[np.ndarray, ...],
    ) -> None:
        heads = self.context.query_heads
        kv_per_group = heads // self.context.kv_heads
        num_kv_tiles = math.ceil(self.context.sequence_length / self.kv_block_size)
        traversal_index = num_kv_tiles - 1 - kv_tile
        q_to_kv_ratio = self.q_block_size // self.kv_block_size
        local_start = query_tile * q_to_kv_ratio
        diagonal = local_start <= kv_tile < local_start + q_to_kv_ratio
        local = (
            local_start - self.local_radius_tiles
            <= kv_tile
            < local_start + q_to_kv_ratio + self.local_radius_tiles
        )
        sink = kv_tile == 0
        distant = not (local or sink)
        for sample in selected_samples:
            ratio = float(self.context.remaining_mask_ratios[sample])
            for head in range(heads):
                self._rows.append(
                    (
                        self.context.sample_ids[sample],
                        self.context.request_ids[sample],
                        self.context.seeds[sample],
                        self.context.sequence_length,
                        self.context.diffusion_block,
                        self.context.denoising_iteration,
                        ratio,
                        noise_bucket(ratio),
                        layer,
                        head,
                        head // kv_per_group,
                        query_tile,
                        kv_tile,
                        traversal_index,
                        valid_query_rows,
                        *(float(value[sample, head]) for value in cpu[:2]),
                        int(bool(cpu[2][sample, head])),
                        int(bool(cpu[3][sample, head])),
                        *(float(value[sample, head]) for value in cpu[4:]),
                        float("nan"),
                        int(local),
                        int(diagonal),
                        int(sink),
                        int(distant),
                    )
                )
                if len(self._rows) >= self.shard_records:
                    self._flush_rows()

    def flush(self) -> None:
        self._drain_device_buffer()
        self._flush_rows()

    def _flush_rows(self) -> None:
        if not self._rows:
            return
        records = np.array(self._rows, dtype=TRACE_DTYPE)
        path = self.output_dir / f"trace-{self._shard_index:06d}.npz"
        extras: dict[str, np.ndarray] = {}
        if self.bitpack_masks:
            extras["exact_skip_packed"] = np.packbits(records["exact_skip"], bitorder="little")
            extras["new_max_packed"] = np.packbits(records["introduced_new_max"], bitorder="little")
        if self.quantize_log_scores is not None:
            finite = np.nan_to_num(records["log_score"], neginf=-32.0, posinf=32.0).clip(-32.0, 32.0)
            levels = (1 << self.quantize_log_scores) - 1
            extras[f"log_score_q{self.quantize_log_scores}"] = np.rint((finite + 32.0) * levels / 64.0).astype(np.uint8)
        np.savez_compressed(path, schema_version=np.array(SCHEMA_VERSION), records=records, **extras)
        self._rows.clear()
        self._shard_index += 1

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "IncrementalTraceCollector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
