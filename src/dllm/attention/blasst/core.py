"""Model-independent reference two-dimensional BLASST attention.

This module deliberately emulates tile skipping after dense QK computation.
It is an accuracy/sparsity reference, not a performance kernel.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import threading
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Blasst2DConfig:
    """Configuration for the reference 2D-BLASST path."""

    enable_blasst_2d: bool = False
    blasst_lambda: float = 0.5
    q_tile_size: int = 128
    kv_tile_size: int = 64
    collect_blasst_stats: bool = False
    dump_blasst_trace: bool = False
    collect_blasst_layer_stats: bool = True
    collect_blasst_head_stats: bool = True
    include_masked_kv_tiles_in_physical_stats: bool = False
    apply_blasst_mask: bool = True
    local_blasst_lambda: Optional[float] = None
    global_blasst_lambda: Optional[float] = None
    denoising_phase_starts: tuple[int, ...] = ()
    local_phase_lambdas: tuple[float, ...] = ()
    global_phase_lambdas: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.blasst_lambda < 1.0:
            raise ValueError("blasst_lambda must be strictly between 0 and 1")
        if self.q_tile_size <= 0:
            raise ValueError("q_tile_size must be positive")
        if self.kv_tile_size <= 0:
            raise ValueError("kv_tile_size must be positive")
        for name, value in (
            ("local_blasst_lambda", self.local_blasst_lambda),
            ("global_blasst_lambda", self.global_blasst_lambda),
        ):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if tuple(sorted(set(self.denoising_phase_starts))) != self.denoising_phase_starts:
            raise ValueError("denoising_phase_starts must be strictly increasing")
        if self.denoising_phase_starts and self.denoising_phase_starts[0] <= 0:
            raise ValueError("denoising phase starts must be positive iterations")
        expected_phases = len(self.denoising_phase_starts) + 1
        for name, values in (
            ("local_phase_lambdas", self.local_phase_lambdas),
            ("global_phase_lambdas", self.global_phase_lambdas),
        ):
            if values and len(values) != expected_phases:
                raise ValueError(f"{name} must contain {expected_phases} values")
            if any(not 0.0 < value < 1.0 for value in values):
                raise ValueError(f"all {name} values must be strictly between 0 and 1")

    @property
    def log_lambda(self) -> float:
        return math.log(self.blasst_lambda)

    def lambda_for(self, attention_type: str, denoising_iteration: int) -> float:
        if attention_type not in ("local", "global"):
            raise ValueError("attention_type must be local or global")
        phases = (
            self.local_phase_lambdas
            if attention_type == "local"
            else self.global_phase_lambdas
        )
        if phases:
            phase = bisect_right(self.denoising_phase_starts, denoising_iteration)
            return phases[phase]
        value = (
            self.local_blasst_lambda
            if attention_type == "local"
            else self.global_blasst_lambda
        )
        return self.blasst_lambda if value is None else value

    def phase_for(self, denoising_iteration: int) -> int:
        return bisect_right(self.denoising_phase_starts, denoising_iteration)


@dataclass
class BlasstTileDecisions:
    """Tile masks and incremental counters produced by one attention call."""

    skip_mask: torch.Tensor
    eligible_mask: torch.Tensor
    structural_mask: torch.Tensor
    masked_only_mask: torch.Tensor
    row_skippable: torch.Tensor
    row_total: torch.Tensor
    skipped_valid_elements: torch.Tensor
    valid_elements: torch.Tensor
    # Boolean row-granular votes with shape [B, H, Q, kv_tiles].  The
    # physical ``skip_mask`` remains tile-granular for reporting: a tile is
    # physically skipped only when every valid active query row votes to skip.
    row_skip_mask: Optional[torch.Tensor] = None

    @property
    def retained_mask(self) -> torch.Tensor:
        return self.eligible_mask & ~self.skip_mask

    @property
    def row_retained_mask(self) -> torch.Tensor:
        if self.row_skip_mask is None:
            # Backward-compatible fallback for callers constructing the old
            # eight-field decision tuple. Without the query tile size there is
            # no lossless way to expand physical tiles to rows, so expose the
            # physical retained mask rather than returning a misleading extra
            # singleton dimension. Native BLASST paths always provide
            # row_skip_mask.
            return ~self.skip_mask
        return ~self.row_skip_mask


_COUNT_FIELDS = (
    "eligible_tiles",
    "skipped_tiles",
    "retained_tiles",
    "structurally_masked_tiles",
    "masked_only_tiles",
    "skippable_row_votes",
    "valid_row_votes",
    "skipped_valid_elements",
    "valid_elements",
)

_TRACE_METADATA_FIELDS = (
    "configuration",
    "outer_block_size",
    "sub_block_size",
    "dual_cache_enabled",
    "forward_pass_id",
    "forward_kind",
    "generation_block_index",
    "sub_block_index",
    "denoising_iteration",
    "denoising_phase",
    "attention_type",
    "effective_blasst_lambda",
    "inference_seed",
)


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in _COUNT_FIELDS}


def _with_ratios(counts: Mapping[str, int]) -> dict[str, Any]:
    row = dict(counts)
    eligible = int(row["eligible_tiles"])
    row_votes = int(row["valid_row_votes"])
    elements = int(row["valid_elements"])
    row["physical_tile_sparsity"] = (
        int(row["skipped_tiles"]) / eligible if eligible else 0.0
    )
    row["row_vote_sparsity"] = (
        int(row["skippable_row_votes"]) / row_votes if row_votes else 0.0
    )
    row["valid_element_sparsity"] = (
        int(row["skipped_valid_elements"]) / elements if elements else 0.0
    )
    return row


@dataclass
class Blasst2DStats:
    """Thread-safe, bounded-memory statistics accumulator."""

    totals: dict[str, int] = field(default_factory=_empty_counts)
    per_step: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    per_layer: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    per_head: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)
    retained_attention_mass_sum: float = 0.0
    retained_attention_mass_rows: int = 0
    region_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    record_layers: bool = True
    record_heads: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def _add(target: dict[str, int], source: Mapping[str, int]) -> None:
        for name in _COUNT_FIELDS:
            target[name] += int(source[name])

    @staticmethod
    def _tensor_counts(
        decisions: BlasstTileDecisions,
        selector: Optional[tuple[Any, ...]] = None,
    ) -> dict[str, int]:
        tensors = {
            "eligible_tiles": decisions.eligible_mask,
            "skipped_tiles": decisions.skip_mask,
            "retained_tiles": decisions.retained_mask,
            "structurally_masked_tiles": decisions.structural_mask,
            "masked_only_tiles": decisions.masked_only_mask,
            "skippable_row_votes": decisions.row_skippable,
            "valid_row_votes": decisions.row_total,
            "skipped_valid_elements": decisions.skipped_valid_elements,
            "valid_elements": decisions.valid_elements,
        }
        if selector is not None:
            tensors = {name: value[selector] for name, value in tensors.items()}
        values = torch.stack(
            [value.sum(dtype=torch.int64) for value in tensors.values()]
        ).cpu().tolist()
        return {
            name: int(values[index])
            for index, name in enumerate(tensors)
        }

    @staticmethod
    def _per_head_tensor_counts(
        decisions: BlasstTileDecisions,
    ) -> list[dict[str, int]]:
        """Reduce every counter for every head with one device synchronization."""
        tensors = {
            "eligible_tiles": decisions.eligible_mask,
            "skipped_tiles": decisions.skip_mask,
            "retained_tiles": decisions.retained_mask,
            "structurally_masked_tiles": decisions.structural_mask,
            "masked_only_tiles": decisions.masked_only_mask,
            "skippable_row_votes": decisions.row_skippable,
            "valid_row_votes": decisions.row_total,
            "skipped_valid_elements": decisions.skipped_valid_elements,
            "valid_elements": decisions.valid_elements,
        }
        reduced = []
        for name in _COUNT_FIELDS:
            value = tensors[name]
            dimensions = tuple(
                dimension
                for dimension in range(value.ndim)
                if dimension != 1
            )
            reduced.append(value.sum(dim=dimensions, dtype=torch.int64))
        # [counter, head] -> one CPU transfer -> [head, counter].
        matrix = torch.stack(reduced).transpose(0, 1).cpu().tolist()
        return [
            {
                name: int(values[index])
                for index, name in enumerate(_COUNT_FIELDS)
            }
            for values in matrix
        ]

    def record(
        self,
        decisions: BlasstTileDecisions,
        *,
        layer: int,
        query_length: int,
        sequence_length: int,
        metadata: Optional[Mapping[str, Any]] = None,
        dump_trace: bool = False,
        retained_dense_attention_mass: Optional[float] = None,
        retained_mass_rows: int = 0,
        region_counts: Optional[Mapping[str, Mapping[str, int]]] = None,
    ) -> None:
        metadata = dict(metadata or {})
        step = metadata.get("denoising_step", -1)
        mask_ratio = metadata.get("mask_ratio", -1.0)
        masked_tokens = metadata.get("masked_tokens", -1)
        valid_kv_length = metadata.get("valid_kv_length", sequence_length)
        benchmark = metadata.get("benchmark", "")
        example_id = metadata.get("example_id", "")
        experiment = metadata.get("experiment", "")
        sweep_value = metadata.get("sweep_value", -1)
        trace_metadata = {
            name: metadata.get(name, -1 if name.endswith(("_size", "_index", "_id", "_iteration", "_seed")) else "")
            for name in _TRACE_METADATA_FIELDS
        }
        batch_size, heads = decisions.skip_mask.shape[:2]
        per_head_counts = (
            self._per_head_tensor_counts(decisions)
            if self.record_heads
            else []
        )

        with self._lock:
            all_counts = self._tensor_counts(decisions)
            self._add(self.totals, all_counts)
            if retained_dense_attention_mass is not None and retained_mass_rows > 0:
                self.retained_attention_mass_sum += float(retained_dense_attention_mass) * int(retained_mass_rows)
                self.retained_attention_mass_rows += int(retained_mass_rows)
            if region_counts:
                for name, values in region_counts.items():
                    target = self.region_counts.setdefault(
                        str(name), {"eligible_tiles": 0, "skipped_tiles": 0, "retained_tiles": 0}
                    )
                    eligible = int(values.get("eligible_tiles", values.get("candidate_tiles", 0)))
                    skipped = int(values.get("skipped_tiles", values.get("skipped", 0)))
                    retained = int(values.get("retained_tiles", values.get("retained", eligible - skipped)))
                    target["eligible_tiles"] += eligible
                    target["skipped_tiles"] += skipped
                    target["retained_tiles"] += retained

            step_key = (
                experiment,
                sweep_value,
                benchmark,
                example_id,
                step,
                mask_ratio,
                masked_tokens,
                query_length,
                sequence_length,
                valid_kv_length,
                *(trace_metadata[name] for name in _TRACE_METADATA_FIELDS),
            )
            step_row = self.per_step.setdefault(
                step_key,
                {
                    "experiment": experiment,
                    "sweep_value": sweep_value,
                    "benchmark": benchmark,
                    "example_id": example_id,
                    "denoising_step": step,
                    "mask_ratio": mask_ratio,
                    "masked_tokens": masked_tokens,
                    "query_length": query_length,
                    "sequence_length": sequence_length,
                    "valid_kv_length": valid_kv_length,
                    **trace_metadata,
                    **_empty_counts(),
                },
            )
            self._add(step_row, all_counts)
            if retained_dense_attention_mass is not None and retained_mass_rows > 0:
                step_row["retained_dense_attention_mass_sum"] = float(step_row.get("retained_dense_attention_mass_sum", 0.0)) + float(retained_dense_attention_mass) * int(retained_mass_rows)
                step_row["retained_attention_mass_rows"] = int(step_row.get("retained_attention_mass_rows", 0)) + int(retained_mass_rows)
                step_row["retained_dense_attention_mass"] = step_row["retained_dense_attention_mass_sum"] / step_row["retained_attention_mass_rows"]
            if region_counts:
                step_regions = step_row.setdefault("region_counts", {})
                for name, values in region_counts.items():
                    target = step_regions.setdefault(str(name), {"eligible_tiles": 0, "skipped_tiles": 0, "retained_tiles": 0})
                    eligible = int(values.get("eligible_tiles", values.get("candidate_tiles", 0)))
                    skipped = int(values.get("skipped_tiles", values.get("skipped", 0)))
                    retained = int(values.get("retained_tiles", values.get("retained", eligible - skipped)))
                    target["eligible_tiles"] += eligible
                    target["skipped_tiles"] += skipped
                    target["retained_tiles"] += retained

            if self.record_layers:
                layer_key = (
                    layer,
                    experiment,
                    sweep_value,
                    benchmark,
                    example_id,
                    step,
                    mask_ratio,
                    masked_tokens,
                    query_length,
                    sequence_length,
                    valid_kv_length,
                    *(trace_metadata[name] for name in _TRACE_METADATA_FIELDS),
                )
                layer_row = self.per_layer.setdefault(
                    layer_key,
                    {
                        "layer": layer,
                        "experiment": experiment,
                        "sweep_value": sweep_value,
                        "benchmark": benchmark,
                        "example_id": example_id,
                        "denoising_step": step,
                        "mask_ratio": mask_ratio,
                        "masked_tokens": masked_tokens,
                        "query_length": query_length,
                        "sequence_length": sequence_length,
                        "valid_kv_length": valid_kv_length,
                        **trace_metadata,
                        **_empty_counts(),
                    },
                )
                self._add(layer_row, all_counts)

            if self.record_heads:
                for head in range(heads):
                    counts = per_head_counts[head]
                    head_key = (
                        layer,
                        head,
                        experiment,
                        sweep_value,
                        benchmark,
                        example_id,
                        step,
                        mask_ratio,
                        masked_tokens,
                        query_length,
                        sequence_length,
                        valid_kv_length,
                        *(trace_metadata[name] for name in _TRACE_METADATA_FIELDS),
                    )
                    head_row = self.per_head.setdefault(
                        head_key,
                        {
                            "layer": layer,
                            "head": head,
                            "experiment": experiment,
                            "sweep_value": sweep_value,
                            "benchmark": benchmark,
                            "example_id": example_id,
                            "denoising_step": step,
                            "mask_ratio": mask_ratio,
                            "masked_tokens": masked_tokens,
                            "query_length": query_length,
                            "sequence_length": sequence_length,
                            "valid_kv_length": valid_kv_length,
                            **trace_metadata,
                            **_empty_counts(),
                        },
                    )
                    self._add(head_row, counts)

            if dump_trace:
                eligible = decisions.eligible_mask.nonzero().cpu().tolist()
                skipped = decisions.skip_mask.nonzero().cpu().tolist()
                structural = decisions.structural_mask.nonzero().cpu().tolist()
                self.traces.append(
                    {
                        "layer": layer,
                        "query_length": query_length,
                        "sequence_length": sequence_length,
                        **metadata,
                        "eligible_tiles": eligible,
                        "skipped_tiles": skipped,
                        "structural_tiles": structural,
                    }
                )

    def summary(self) -> dict[str, Any]:
        summary = _with_ratios(self.totals)
        if summary["eligible_tiles"] != (
            summary["skipped_tiles"] + summary["retained_tiles"]
        ):
            raise AssertionError("eligible tile count is inconsistent")
        summary["retained_dense_attention_mass"] = (
            self.retained_attention_mass_sum / self.retained_attention_mass_rows
            if self.retained_attention_mass_rows
            else None
        )
        summary["retained_attention_mass_rows"] = self.retained_attention_mass_rows
        summary["region_counts"] = self.region_counts
        return summary

    @staticmethod
    def _parse_export_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value == "":
            return value
        try:
            if value[:1] in "[{" and value[-1:] in "]}":
                return json.loads(value)
            return float(value) if any(char in value for char in ".eE") else int(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value

    @classmethod
    def load_export(cls, output_dir: str | Path) -> "Blasst2DStats":
        """Load an earlier attention-stat export for resumable accumulation."""

        output = Path(output_dir)
        summary_path = output / "summary.json"
        if not summary_path.exists():
            return cls()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stats = cls()
        for name in _COUNT_FIELDS:
            stats.totals[name] = int(summary.get(name, 0))
        rows = int(summary.get("retained_attention_mass_rows", 0))
        mass = summary.get("retained_dense_attention_mass")
        if mass is not None and rows:
            stats.retained_attention_mass_sum = float(mass) * rows
            stats.retained_attention_mass_rows = rows
        region_counts = summary.get("region_counts")
        if isinstance(region_counts, Mapping):
            stats.region_counts = {
                str(name): {
                    "eligible_tiles": int(values.get("eligible_tiles", 0)),
                    "skipped_tiles": int(values.get("skipped_tiles", 0)),
                    "retained_tiles": int(values.get("retained_tiles", 0)),
                }
                for name, values in region_counts.items()
                if isinstance(values, Mapping)
            }

        def load_rows(filename: str) -> dict[tuple[int], dict[str, Any]]:
            path = output / filename
            if not path.exists():
                return {}
            loaded: dict[tuple[int], dict[str, Any]] = {}
            with path.open(newline="", encoding="utf-8") as handle:
                for index, raw in enumerate(csv.DictReader(handle)):
                    row = {key: cls._parse_export_value(value) for key, value in raw.items()}
                    for name in _COUNT_FIELDS:
                        if name in row:
                            row[name] = int(row[name])
                    loaded[(index,)] = row
            return loaded

        stats.per_step = load_rows("per_step.csv")
        stats.per_layer = load_rows("per_layer.csv")
        stats.per_head = load_rows("per_head.csv")
        trace_path = output / "trace.json"
        if trace_path.exists():
            try:
                stats.traces = json.loads(trace_path.read_text(encoding="utf-8"))
            except (TypeError, ValueError, json.JSONDecodeError):
                stats.traces = []
        return stats

    @staticmethod
    def _rows(values: Mapping[tuple[Any, ...], Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            _with_ratios(values[key])
            for key in sorted(values, key=lambda item: repr(item))
        ]

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row.keys()
        )) if rows else [
            "eligible_tiles",
            "skipped_tiles",
            "retained_tiles",
            "structurally_masked_tiles",
            "masked_only_tiles",
            "skippable_row_votes",
            "valid_row_votes",
            "skipped_valid_elements",
            "valid_elements",
            "physical_tile_sparsity",
            "row_vote_sparsity",
            "valid_element_sparsity",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                serialized = {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
                writer.writerow(serialized)

    def export(
        self,
        output_dir: str | Path,
        config: Blasst2DConfig,
        run_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "summary.json").write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        merged_config = {**asdict(config), **dict(run_config or {})}
        (output_path / "run_config.json").write_text(
            json.dumps(merged_config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        self._write_csv(output_path / "per_step.csv", self._rows(self.per_step))
        self._write_csv(output_path / "per_layer.csv", self._rows(self.per_layer))
        self._write_csv(output_path / "per_head.csv", self._rows(self.per_head))
        if config.dump_blasst_trace:
            (output_path / "trace.json").write_text(
                json.dumps(self.traces, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


@dataclass
class Blasst2DRuntime:
    """State explicitly attached to attention modules by the installer."""

    config: Blasst2DConfig
    stats: Blasst2DStats = field(default_factory=Blasst2DStats)
    dual_cache_only: bool = False
    ordinary_cache_queries_only: bool = False
    active_call: bool = True
    active_query_mask: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_context: dict[str, Any] = field(default_factory=dict)
    forward_call_index: int = 0
    active_forward_call_index: int = 0
    current_denoising_iteration: int = -1
    hook_handle: Any = field(default=None, repr=False)
    sweep_lambdas: tuple[float, ...] = ()
    sweep_stats: dict[float, Blasst2DStats] = field(default_factory=dict)
    attention_observer: Optional[Callable[..., None]] = field(
        default=None, repr=False
    )
    attention_override: Optional[Callable[..., Any]] = field(
        default=None, repr=False
    )
    dense_kv_prefix_extractor: Optional[Callable[..., int]] = field(
        default=None, repr=False
    )


def _expand_valid_mask(
    valid_pair_mask: Optional[torch.Tensor],
    scores: torch.Tensor,
) -> torch.Tensor:
    if valid_pair_mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = valid_pair_mask.to(device=scores.device)
    if mask.dtype == torch.bool:
        valid = mask
    elif mask.is_floating_point():
        # Hugging Face additive masks use zero for valid pairs and a large
        # negative finite value (or -inf) for invalid pairs.
        valid = torch.isfinite(mask) & (mask > -1.0e4)
    else:
        valid = mask != 0
    return torch.broadcast_to(valid, scores.shape)


def _expand_active_rows(
    active_query_mask: Optional[torch.Tensor],
    scores: torch.Tensor,
) -> torch.Tensor:
    batch, heads, q_len, _ = scores.shape
    if active_query_mask is None:
        return torch.ones((batch, heads, q_len), dtype=torch.bool, device=scores.device)
    active = active_query_mask.to(device=scores.device, dtype=torch.bool)
    if active.ndim == 2:
        active = active[:, None, :]
    if active.shape[-1] != q_len:
        if active.shape[-1] < q_len:
            raise ValueError(
                "active_query_mask is shorter than the attention query sequence"
            )
        active = active[..., -q_len:]
    return torch.broadcast_to(active, (batch, heads, q_len))


def _decision_tensors(
    scores: torch.Tensor,
    valid_pair_mask: Optional[torch.Tensor],
    active_query_mask: Optional[torch.Tensor],
    config: Blasst2DConfig,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if scores.ndim != 4:
        raise ValueError("scores must have shape [batch, heads, query, key]")
    valid = _expand_valid_mask(valid_pair_mask, scores)
    active = _expand_active_rows(active_query_mask, scores)
    batch, heads, q_len, kv_len = scores.shape
    q_tiles = math.ceil(q_len / config.q_tile_size)
    kv_tiles = math.ceil(kv_len / config.kv_tile_size)
    return valid, active, q_tiles, kv_tiles


@torch.no_grad()
def evaluate_blasst_thresholds(
    scores: torch.Tensor,
    valid_pair_mask: Optional[torch.Tensor],
    active_query_mask: Optional[torch.Tensor],
    lambdas: list[float] | tuple[float, ...],
    *,
    q_tile_size: int = 128,
    kv_tile_size: int = 64,
    sparse_kv_start: int = 0,
    include_masked_kv_tiles_in_physical_stats: bool = False,
) -> dict[float, BlasstTileDecisions]:
    """Evaluate many thresholds from one shared set of QK tile maxima.

    The model state and dense scores are identical for every lambda. Local
    maxima and FP32 running maxima are computed once, then thresholded.
    """

    lambda_values = tuple(float(value) for value in lambdas)
    if not lambda_values:
        raise ValueError("at least one lambda is required")
    for value in lambda_values:
        if not 0.0 < value < 1.0:
            raise ValueError("all lambda values must be strictly between 0 and 1")
    if q_tile_size <= 0 or kv_tile_size <= 0:
        raise ValueError("tile sizes must be positive")

    probe_config = Blasst2DConfig(
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
        include_masked_kv_tiles_in_physical_stats=(
            include_masked_kv_tiles_in_physical_stats
        ),
    )
    valid, active, q_tiles, kv_tiles = _decision_tensors(
        scores, valid_pair_mask, active_query_mask, probe_config
    )
    batch, heads, q_len, kv_len = scores.shape
    shape = (batch, heads, q_tiles, kv_tiles)
    decisions_by_lambda: dict[float, BlasstTileDecisions] = {}
    buffers: dict[float, dict[str, torch.Tensor]] = {}
    for value in lambda_values:
        bools = torch.zeros(shape, dtype=torch.bool, device=scores.device)
        counts = torch.zeros(shape, dtype=torch.int64, device=scores.device)
        buffers[value] = {
            "skip": bools,
            "row_skip_mask": torch.zeros(
                (batch, heads, q_len, kv_tiles), dtype=torch.bool, device=scores.device
            ),
            "eligible": torch.zeros_like(bools),
            "row_skip": counts,
            "row_total": torch.zeros_like(counts),
            "skip_elements": torch.zeros_like(counts),
            "valid_elements": torch.zeros_like(counts),
            "semantic_eligible": torch.zeros_like(bools),
        }

    sparse_tiles = (
        torch.arange(kv_tiles, device=scores.device) * kv_tile_size
        >= int(sparse_kv_start)
    )

    for query_tile in range(q_tiles):
        q_start = query_tile * q_tile_size
        q_end = min(q_start + q_tile_size, q_len)
        tile_scores = scores[:, :, q_start:q_end].float()
        tile_valid = valid[:, :, q_start:q_end]
        active_rows = active[:, :, q_start:q_end]
        pad = kv_tiles * kv_tile_size - kv_len
        if pad:
            tile_scores = F.pad(tile_scores, (0, pad), value=-torch.inf)
            tile_valid = F.pad(tile_valid, (0, pad), value=False)
        score_blocks = tile_scores.reshape(
            batch, heads, q_end - q_start, kv_tiles, kv_tile_size
        )
        valid_blocks = tile_valid.reshape(
            batch, heads, q_end - q_start, kv_tiles, kv_tile_size
        )
        local_max = score_blocks.masked_fill(
            ~valid_blocks, -torch.inf
        ).amax(-1)
        cumulative_max = torch.cummax(local_max, dim=-1).values
        running_before = F.pad(cumulative_max[..., :-1], (1, 0), value=-torch.inf)
        margin = local_max - running_before
        row_has_valid = valid_blocks.any(-1)
        active_valid = valid_blocks & active_rows[..., None, None]
        semantic_eligible = active_valid.any(dim=(-3, -1)) & sparse_tiles
        if probe_config.include_masked_kv_tiles_in_physical_stats:
            eligible = active_rows.any(dim=-1, keepdim=True) & sparse_tiles
        else:
            eligible = semantic_eligible
        valid_count = active_valid.sum(dim=(-3, -1)) * sparse_tiles
        valid_vote = active_rows[..., None] & row_has_valid & sparse_tiles

        for value in lambda_values:
            vote = (~row_has_valid) | (margin < math.log(value))
            physical_skip = (
                vote | ~active_rows[..., None]
            ).all(dim=-2) & eligible
            target = buffers[value]
            target["eligible"][:, :, query_tile] = eligible
            target["semantic_eligible"][:, :, query_tile] = semantic_eligible
            target["skip"][:, :, query_tile] = physical_skip
            target["row_skip_mask"][:, :, q_start:q_end] = vote & valid_vote
            target["row_skip"][:, :, query_tile] = (
                vote & valid_vote
            ).sum(-2)
            target["row_total"][:, :, query_tile] = valid_vote.sum(-2)
            target["valid_elements"][:, :, query_tile] = valid_count
            # Secondary BLASST sparsity is row-granular: count valid QK
            # elements removed by individual row votes, even when another
            # query row retains the same physical tile.
            target["skip_elements"][:, :, query_tile] = (
                (active_valid & (vote[..., None])).sum(dim=(-3, -1))
            )

    for value, target in buffers.items():
        decisions_by_lambda[value] = BlasstTileDecisions(
            skip_mask=target["skip"],
            eligible_mask=target["eligible"],
            structural_mask=~target["eligible"],
            masked_only_mask=(
                target["eligible"] & ~target["semantic_eligible"]
            ),
            row_skippable=target["row_skip"],
            row_total=target["row_total"],
            skipped_valid_elements=target["skip_elements"],
            valid_elements=target["valid_elements"],
            row_skip_mask=target["row_skip_mask"],
        )
    return decisions_by_lambda


@torch.no_grad()
def apply_blasst_2d(
    scores: torch.Tensor,
    valid_pair_mask: Optional[torch.Tensor],
    active_query_mask: Optional[torch.Tensor],
    config: Blasst2DConfig,
    *,
    sparse_kv_start: int = 0,
    blasst_lambda: Optional[float] = None,
) -> tuple[torch.Tensor, BlasstTileDecisions]:
    """Vectorized-over-batch/head reference 2D-BLASST decision path."""
    value = config.blasst_lambda if blasst_lambda is None else blasst_lambda
    decisions = evaluate_blasst_thresholds(
        scores,
        valid_pair_mask,
        active_query_mask,
        [value],
        q_tile_size=config.q_tile_size,
        kv_tile_size=config.kv_tile_size,
        sparse_kv_start=sparse_kv_start,
        include_masked_kv_tiles_in_physical_stats=(
            config.include_masked_kv_tiles_in_physical_stats
        ),
    )[value]
    # Expand only the final block decision mask; all score analysis above stays
    # block-shaped. This removes the Python loop over every KV tile, which is
    # especially important for fast 8K benchmark iteration.
    # ``apply_blasst_2d`` remains a tile-mask oracle for compatibility with
    # callers that inspect the physical mask directly.  The attention forward
    # path applies the row-granular mask below before softmax.
    element_skip_mask = decisions.skip_mask.repeat_interleave(
        config.q_tile_size, dim=-2
    ).repeat_interleave(config.kv_tile_size, dim=-1)
    element_skip_mask = element_skip_mask[..., : scores.shape[-2], : scores.shape[-1]]
    valid, _, _, _ = _decision_tensors(
        scores, valid_pair_mask, active_query_mask, config
    )
    masked_scores = scores.masked_fill(element_skip_mask & valid, -torch.inf)
    return masked_scores, decisions


@torch.no_grad()
def slow_blasst_2d(
    scores: torch.Tensor,
    valid_pair_mask: Optional[torch.Tensor],
    active_query_mask: Optional[torch.Tensor],
    config: Blasst2DConfig,
    *,
    sparse_kv_start: int = 0,
) -> tuple[torch.Tensor, BlasstTileDecisions]:
    """Literal loop implementation used as a correctness oracle in tests."""

    valid, active, q_tiles, kv_tiles = _decision_tensors(
        scores, valid_pair_mask, active_query_mask, config
    )
    batch, heads, q_len, kv_len = scores.shape
    shape = (batch, heads, q_tiles, kv_tiles)
    skip = torch.zeros(shape, dtype=torch.bool, device=scores.device)
    eligible = torch.zeros_like(skip)
    row_skip = torch.zeros(shape, dtype=torch.int64, device=scores.device)
    row_total = torch.zeros_like(row_skip)
    skip_elements = torch.zeros_like(row_skip)
    valid_elements = torch.zeros_like(row_skip)
    masked_only = torch.zeros_like(skip)
    row_skip_mask = torch.zeros(
        (batch, heads, q_len, kv_tiles), dtype=torch.bool, device=scores.device
    )
    masked_scores = scores.clone()

    for batch_idx in range(batch):
        for head in range(heads):
            for query_tile in range(q_tiles):
                q_start = query_tile * config.q_tile_size
                q_end = min(q_start + config.q_tile_size, q_len)
                running = [-math.inf] * (q_end - q_start)
                for kv_tile in range(kv_tiles):
                    kv_start = kv_tile * config.kv_tile_size
                    kv_end = min(kv_start + config.kv_tile_size, kv_len)
                    votes: list[bool] = []
                    valid_rows = 0
                    skippable_rows = 0
                    tile_elements = 0
                    tile_skipped_elements = 0
                    for local_row, query_idx in enumerate(range(q_start, q_end)):
                        if not bool(active[batch_idx, head, query_idx]):
                            continue
                        valid_scores = [
                            float(scores[batch_idx, head, query_idx, key_idx])
                            for key_idx in range(kv_start, kv_end)
                            if bool(valid[batch_idx, head, query_idx, key_idx])
                        ]
                        tile_elements += len(valid_scores)
                        if not valid_scores:
                            votes.append(True)
                            continue
                        local_max = max(valid_scores)
                        previous_max = running[local_row]
                        vote = (local_max - previous_max) < config.log_lambda
                        running[local_row] = max(previous_max, local_max)
                        votes.append(vote)
                        if kv_start >= sparse_kv_start:
                            row_skip_mask[batch_idx, head, query_idx, kv_tile] = vote
                            if vote:
                                tile_skipped_elements += len(valid_scores)
                        valid_rows += 1
                        skippable_rows += int(vote)

                    tile_eligible = (
                        bool(votes)
                        and kv_start >= sparse_kv_start
                        and (
                            config.include_masked_kv_tiles_in_physical_stats
                            or tile_elements > 0
                        )
                    )
                    tile_skip = tile_eligible and all(votes)
                    eligible[batch_idx, head, query_tile, kv_tile] = tile_eligible
                    masked_only[batch_idx, head, query_tile, kv_tile] = (
                        tile_eligible and tile_elements == 0
                    )
                    skip[batch_idx, head, query_tile, kv_tile] = tile_skip
                    if tile_eligible:
                        row_skip[batch_idx, head, query_tile, kv_tile] = skippable_rows
                        row_total[batch_idx, head, query_tile, kv_tile] = valid_rows
                        valid_elements[batch_idx, head, query_tile, kv_tile] = tile_elements
                    skip_elements[batch_idx, head, query_tile, kv_tile] = (
                        tile_skipped_elements if kv_start >= sparse_kv_start else 0
                    )

    element_skip_mask = skip.repeat_interleave(
        config.q_tile_size, dim=-2
    ).repeat_interleave(config.kv_tile_size, dim=-1)
    element_skip_mask = element_skip_mask[..., :q_len, :kv_len]
    masked_scores = masked_scores.masked_fill(element_skip_mask & valid, -torch.inf)
    return masked_scores, BlasstTileDecisions(
        skip_mask=skip,
        eligible_mask=eligible,
        structural_mask=~eligible,
        masked_only_mask=masked_only,
        row_skippable=row_skip,
        row_total=row_total,
        skipped_valid_elements=skip_elements,
        valid_elements=valid_elements,
        row_skip_mask=row_skip_mask,
    )


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, kv_heads, length, head_dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, kv_heads, repeats, length, head_dim)
        .reshape(batch, kv_heads * repeats, length, head_dim)
    )


def _attention_validity(
    attention_mask: Optional[torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    is_causal: bool,
    sliding_window: Optional[int],
) -> torch.Tensor:
    batch, heads, q_len, _ = query.shape
    kv_len = key.shape[-2]
    if attention_mask is None:
        valid = torch.ones(
            (batch, heads, q_len, kv_len), dtype=torch.bool, device=query.device
        )
    else:
        mask = attention_mask[..., :kv_len].to(query.device)
        if mask.dtype == torch.bool:
            valid = mask
        elif mask.is_floating_point():
            valid = torch.isfinite(mask) & (mask > -1.0e4)
        else:
            valid = mask != 0
        valid = torch.broadcast_to(valid, (batch, heads, q_len, kv_len))

    if is_causal:
        diagonal = kv_len - q_len
        causal = torch.ones(
            (q_len, kv_len), dtype=torch.bool, device=query.device
        ).tril(diagonal=diagonal)
        valid = valid & causal
    if sliding_window is not None and (
        attention_mask is None or attention_mask.ndim < 4
    ):
        q_positions = torch.arange(q_len, device=query.device) + (kv_len - q_len)
        k_positions = torch.arange(kv_len, device=query.device)
        window = k_positions[None, :] >= (
            q_positions[:, None] - int(sliding_window) + 1
        )
        valid = valid & window
    return valid


def _prepare_attention_inputs(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: Optional[float],
    is_causal: Optional[bool],
    sliding_window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    repeats = (
        1
        if getattr(module, "_blasst_kv_already_repeated", False)
        else int(getattr(module, "num_key_value_groups", 1))
    )
    key = _repeat_kv(key, repeats)
    value = _repeat_kv(value, repeats)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    if is_causal is None:
        is_causal = attention_mask is None and query.shape[-2] > 1
    valid = _attention_validity(
        attention_mask,
        query,
        key,
        is_causal=bool(is_causal),
        sliding_window=sliding_window,
    )
    return key, value, valid, float(scaling)


def _prepare_attention_scores(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: Optional[float],
    is_causal: Optional[bool],
    sliding_window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key, value, valid, scale = _prepare_attention_inputs(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        scores = scores + attention_mask[..., : key.shape[-2]]
    scores = scores.masked_fill(~valid, -torch.inf)
    return key, value, scores, valid


def _attention_type(module: torch.nn.Module, sliding_window: Optional[int]) -> str:
    is_local = bool(
        getattr(module, "is_sliding", False)
        or getattr(module, "layer_type", "") == "sliding_attention"
        or sliding_window is not None
    )
    return "local" if is_local else "global"


def _finish_eager_attention(
    query: torch.Tensor,
    value: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    dropout: float,
    training: bool,
) -> tuple[torch.Tensor, None]:
    row_has_any_key = valid.any(dim=-1, keepdim=True)
    softmax_scores = torch.where(row_has_any_key, scores, torch.zeros_like(scores))
    probabilities = F.softmax(
        softmax_scores, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    probabilities = torch.where(
        row_has_any_key, probabilities, torch.zeros_like(probabilities)
    )
    if dropout:
        probabilities = F.dropout(probabilities, p=dropout, training=training)
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return output, None


@torch.no_grad()
def _retained_dense_mass(
    dense_scores: torch.Tensor,
    decisions: BlasstTileDecisions,
    valid: torch.Tensor,
    active_query_mask: Optional[torch.Tensor],
    config: Blasst2DConfig,
) -> tuple[float, int]:
    """Measure dense softmax mass retained by the physical tile mask.

    The reference path computes dense scores before masking, so this diagnostic
    is exact.  Rows that are inactive or have no valid key are excluded from
    the denominator, matching the attention output semantics.
    """

    row_skip = decisions.row_skip_mask
    if row_skip is None:
        row_skip = decisions.skip_mask.unsqueeze(-2)
    element_skip = row_skip.repeat_interleave(config.kv_tile_size, dim=-1)
    element_skip = element_skip[..., : dense_scores.shape[-2], : dense_scores.shape[-1]]
    valid_rows = valid.any(dim=-1)
    active = _expand_active_rows(active_query_mask, dense_scores)
    rows = valid_rows & active
    safe_scores = torch.where(rows[..., None], dense_scores, torch.zeros_like(dense_scores))
    probabilities = F.softmax(safe_scores, dim=-1, dtype=torch.float32)
    probabilities = torch.where(rows[..., None], probabilities, torch.zeros_like(probabilities))
    mass = (probabilities * (~element_skip) * valid).sum(dim=-1)
    count = int(rows.sum().item())
    return (float(mass[rows].sum().item()) if count else 0.0), count


def _region_tile_counts(
    decisions: BlasstTileDecisions,
    *,
    kv_tile_size: int,
    sequence_length: int,
    prefix_length: int,
) -> dict[str, dict[str, int]]:
    """Reduce physical tile counters into prefix/canvas diagnostics."""

    regions: dict[str, dict[str, int]] = {
        "prefix": {"eligible_tiles": 0, "skipped_tiles": 0, "retained_tiles": 0},
        "canvas": {"eligible_tiles": 0, "skipped_tiles": 0, "retained_tiles": 0},
    }
    kv_tiles = decisions.eligible_mask.shape[-1]
    for tile in range(kv_tiles):
        start = tile * int(kv_tile_size)
        end = min(start + int(kv_tile_size), int(sequence_length))
        name = "prefix" if start < int(prefix_length) else "canvas"
        eligible = int(decisions.eligible_mask[..., tile].sum().item())
        skipped = int(decisions.skip_mask[..., tile].sum().item())
        target = regions[name]
        target["eligible_tiles"] += eligible
        target["skipped_tiles"] += skipped
        target["retained_tiles"] += eligible - skipped
    return regions


def dense_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    sliding_window: Optional[int] = None,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    """Dense eager path sharing BLASST's QK, mask, softmax, and PV ordering."""
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
    return _finish_eager_attention(
        query, value, scores, valid, dropout, module.training
    )


@torch.no_grad()
def _observe_blasst_sweep(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    sliding_window: Optional[int] = None,
    **kwargs: Any,
) -> None:
    runtime: Blasst2DRuntime = module._blasst_2d_runtime
    key, _, scores, valid = _prepare_attention_scores(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )
    active_query_mask = kwargs.get(
        "blasst_active_query_mask", runtime.active_query_mask
    )
    sparse_kv_start = 0
    if runtime.dense_kv_prefix_extractor is not None:
        sparse_kv_start = runtime.dense_kv_prefix_extractor(
            module, query, key, value, attention_mask, kwargs
        )
    metadata = {
        **runtime.metadata,
        **dict(kwargs.get("blasst_metadata", {}) or {}),
    }
    metadata.setdefault("attention_type", _attention_type(module, sliding_window))
    valid_lengths = valid.any(dim=(1, 2)).sum(-1)
    if valid_lengths.numel() == 1:
        metadata["valid_kv_length"] = int(valid_lengths.item())
    elif torch.equal(valid_lengths, valid_lengths[:1].expand_as(valid_lengths)):
        metadata["valid_kv_length"] = int(valid_lengths[0].item())
    else:
        raise ValueError(
            "sweep statistics require one shared valid KV length per batch"
        )
    if runtime.attention_observer is not None:
        runtime.attention_observer(
            scores=scores,
            valid_pair_mask=valid,
            active_query_mask=active_query_mask,
            layer=int(getattr(module, "layer_idx", -1)),
            metadata=metadata,
            q_tile_size=runtime.config.q_tile_size,
            kv_tile_size=runtime.config.kv_tile_size,
            sparse_kv_start=sparse_kv_start,
        )
    decisions = evaluate_blasst_thresholds(
        scores,
        valid,
        active_query_mask,
        runtime.sweep_lambdas,
        q_tile_size=runtime.config.q_tile_size,
        kv_tile_size=runtime.config.kv_tile_size,
        sparse_kv_start=sparse_kv_start,
        include_masked_kv_tiles_in_physical_stats=(
            runtime.config.include_masked_kv_tiles_in_physical_stats
        ),
    )
    for lambda_value, tile_decisions in decisions.items():
        runtime.sweep_stats[lambda_value].record(
            tile_decisions,
            layer=int(getattr(module, "layer_idx", -1)),
            query_length=query.shape[-2],
            sequence_length=key.shape[-2],
            metadata=metadata,
            dump_trace=runtime.config.dump_blasst_trace,
        )


def blasst_2d_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    sliding_window: Optional[int] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Hugging Face attention-interface compatible reference function."""

    runtime: Blasst2DRuntime = module._blasst_2d_runtime
    config = runtime.config
    # Match the dense eager path's ordering: QK in the model dtype, scaling,
    # then all pre-softmax masks. Comparisons and running maxima are FP32.
    key, value, scores, valid = _prepare_attention_scores(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        is_causal=is_causal,
        sliding_window=sliding_window,
    )

    active_query_mask = kwargs.pop(
        "blasst_active_query_mask", runtime.active_query_mask
    )
    sparse_kv_start = 0
    if runtime.dense_kv_prefix_extractor is not None:
        sparse_kv_start = runtime.dense_kv_prefix_extractor(
            module, query, key, value, attention_mask, kwargs
        )
    if not isinstance(sparse_kv_start, int) or not 0 <= sparse_kv_start <= key.shape[-2]:
        raise ValueError("dense KV prefix extractor returned an invalid length")
    # ``sparse_kv_start`` controls eligibility, and canonical DiffusionGemma
    # deliberately returns zero so prompt/encoder KV tiles remain skippable.
    # For prefix-versus-canvas diagnostics infer the semantic boundary from
    # the cached KV/query lengths when no explicit eligibility boundary was
    # supplied. Cache-aware backends that provide a nonzero boundary retain
    # that explicit value.
    region_prefix_length = sparse_kv_start
    if region_prefix_length == 0:
        region_prefix_length = max(0, int(key.shape[-2]) - int(query.shape[-2]))
    metadata = {
        **runtime.metadata,
        **dict(kwargs.pop("blasst_metadata", {}) or {}),
    }
    attention_type = _attention_type(module, sliding_window)
    denoising_iteration = int(metadata.get("denoising_iteration", -1))
    effective_lambda = config.lambda_for(attention_type, denoising_iteration)
    metadata.update(
        attention_type=attention_type,
        denoising_phase=config.phase_for(denoising_iteration),
        effective_blasst_lambda=effective_lambda,
    )
    valid_lengths = valid.any(dim=(1, 2)).sum(-1)
    if valid_lengths.numel() == 1:
        metadata.setdefault("valid_kv_length", int(valid_lengths.item()))
    dense_scores = scores
    # Keep the public ``apply_blasst_2d`` helper a physical-tile oracle for
    # compatibility with callers that inspect or compare the tile mask.  The
    # actual attention path follows BLASST's row-granular rule: each query row
    # can skip a KV tile independently, while physical-tile sparsity is only
    # counted when all valid rows vote to skip it.
    _, decisions = apply_blasst_2d(
        scores,
        valid,
        active_query_mask,
        config,
        sparse_kv_start=sparse_kv_start,
        blasst_lambda=effective_lambda,
    )
    if config.apply_blasst_mask:
        row_skip = decisions.row_skip_mask
        if row_skip is None:
            row_skip = decisions.skip_mask.unsqueeze(-2)
        element_skip_mask = row_skip.repeat_interleave(
            config.kv_tile_size, dim=-1
        )
        element_skip_mask = element_skip_mask[
            ..., : dense_scores.shape[-2], : dense_scores.shape[-1]
        ]
        scores = dense_scores.masked_fill(element_skip_mask & valid, -torch.inf)
    else:
        scores = dense_scores

    if config.collect_blasst_stats:
        retained_mass, retained_rows = _retained_dense_mass(
            dense_scores,
            decisions,
            valid,
            active_query_mask,
            config,
        )
        region_counts = _region_tile_counts(
            decisions,
            kv_tile_size=config.kv_tile_size,
            sequence_length=key.shape[-2],
            prefix_length=region_prefix_length,
        )
        runtime.stats.record(
            decisions,
            layer=int(getattr(module, "layer_idx", -1)),
            query_length=query.shape[-2],
            sequence_length=key.shape[-2],
            metadata=metadata,
            dump_trace=config.dump_blasst_trace,
            retained_dense_attention_mass=(retained_mass / retained_rows if retained_rows else None),
            retained_mass_rows=retained_rows,
            region_counts=region_counts,
        )
    return _finish_eager_attention(
        query, value, scores, valid, dropout, module.training
    )


def install_blasst_2d(
    model: torch.nn.Module,
    config: Blasst2DConfig,
    stats: Optional[Blasst2DStats] = None,
    *,
    mask_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    dual_cache_only: bool = False,
    ordinary_cache_queries_only: bool = False,
    sweep_lambdas: Optional[list[float] | tuple[float, ...]] = None,
    attention_class_names: Optional[tuple[str, ...]] = None,
) -> Blasst2DRuntime:
    """Install BLASST on a loaded model using the HF attention registry.

    The remote model hard-codes the ``sdpa`` entry from its imported
    ``ALL_ATTENTION_FUNCTIONS`` registry. We replace that registry entry with
    a dispatcher that calls the original function for all modules except the
    explicitly tagged modules in ``model``. Thus disabled/baseline models and
    unrelated models retain the original dense implementation.
    """

    runtime_stats = stats or Blasst2DStats(
        record_layers=config.collect_blasst_layer_stats,
        record_heads=config.collect_blasst_head_stats,
    )
    runtime = Blasst2DRuntime(
        config=config,
        stats=runtime_stats,
        dual_cache_only=dual_cache_only,
        ordinary_cache_queries_only=ordinary_cache_queries_only,
        sweep_lambdas=tuple(float(value) for value in (sweep_lambdas or ())),
    )
    runtime.sweep_stats = {
        value: Blasst2DStats(
            record_layers=config.collect_blasst_layer_stats,
            record_heads=config.collect_blasst_head_stats,
        )
        for value in runtime.sweep_lambdas
    }
    if attention_class_names is None:
        attention_class_names = ("Fast_dLLM_QwenAttention",)
    attention_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ in attention_class_names
    ]
    if not attention_modules:
        expected = ", ".join(attention_class_names)
        raise ValueError(f"No supported attention modules found (expected: {expected})")

    modeling_module = importlib.import_module(attention_modules[0].__class__.__module__)
    registry = modeling_module.ALL_ATTENTION_FUNCTIONS
    original: Callable[..., Any] = getattr(
        modeling_module, "_blasst_2d_original_sdpa", registry["sdpa"]
    )
    setattr(modeling_module, "_blasst_2d_original_sdpa", original)

    def dispatch(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        tagged_runtime = getattr(module, "_blasst_2d_runtime", None)
        dense_call = (
            tagged_runtime is None
            or not tagged_runtime.config.enable_blasst_2d
            or not tagged_runtime.active_call
            or (
                tagged_runtime.dual_cache_only
                and not kwargs.get("use_block_cache", False)
            )
        )
        if dense_call:
            return original(module, *args, **kwargs)
        if tagged_runtime.attention_override is not None:
            return tagged_runtime.attention_override(module, *args, **kwargs)
        if tagged_runtime.sweep_lambdas:
            _observe_blasst_sweep(module, *args, **kwargs)
            return original(module, *args, **kwargs)
        return blasst_2d_attention_forward(module, *args, **kwargs)

    registry["sdpa"] = dispatch
    for attention in attention_modules:
        attention._blasst_2d_runtime = runtime

    base_model = getattr(model, "model", model)

    def capture_forward_state(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            runtime.active_query_mask = None
            call_metadata = {}
        else:
            active = torch.ones_like(input_ids, dtype=torch.bool)
            if pad_token_id is not None:
                active = input_ids != pad_token_id
            masked_tokens = (
                int((input_ids == mask_token_id).sum().item())
                if mask_token_id is not None
                else -1
            )
            active_tokens = int(active.sum().item())
            runtime.active_query_mask = active
            call_metadata = {
                "masked_tokens": masked_tokens,
                "mask_ratio": (
                    masked_tokens / active_tokens
                    if masked_tokens >= 0 and active_tokens
                    else -1.0
                ),
            }
        past_key_values = kwargs.get("past_key_values")
        update_cache = bool(kwargs.get("update_past_key_values", False))
        use_block_cache = bool(kwargs.get("use_block_cache", False))
        block_past_key_values = kwargs.get("block_past_key_values")
        if runtime.ordinary_cache_queries_only:
            runtime.active_call = (
                past_key_values is not None
                and not update_cache
                and not use_block_cache
            )
        elif runtime.dual_cache_only:
            runtime.active_call = (
                past_key_values is not None
                and not update_cache
                and use_block_cache
            )
        else:
            runtime.active_call = True
        if use_block_cache:
            forward_kind = (
                "dual_refresh"
                if block_past_key_values is None
                else "dual_subblock_update"
            )
        elif update_cache:
            forward_kind = "cache_commit"
        elif past_key_values is not None:
            forward_kind = "ordinary_denoising"
        else:
            forward_kind = "prefix_prefill"
        runtime.metadata = {
            **runtime.metadata_context,
            **call_metadata,
            "forward_call_index": runtime.forward_call_index,
            "forward_kind": forward_kind,
        }
        if runtime.active_call:
            runtime.metadata["forward_pass_id"] = (
                runtime.active_forward_call_index
            )
            runtime.metadata["denoising_step"] = (
                runtime.active_forward_call_index
            )
            runtime.active_forward_call_index += 1
        else:
            runtime.metadata["forward_pass_id"] = -1
            runtime.metadata["denoising_step"] = -1
        runtime.forward_call_index += 1

    runtime.hook_handle = base_model.register_forward_pre_hook(
        capture_forward_state, with_kwargs=True
    )
    return runtime
