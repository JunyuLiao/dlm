"""Physical-tile threshold calibration helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class PhysicalCalibrationRegression:
    """Summary of the selected physical-tile calibration measurement."""

    lambdas: dict[float, float]
    physical_sparsity: dict[float, float]
    dense_agreement: dict[float, float]
    overall_physical_sparsity: float
    skipped_tiles: int
    total_tiles: int


def select_largest_eligible_lambda(
    candidates: list[tuple[float, float]], *, minimum_agreement: float = 0.95
) -> float:
    """Select using the calibration's exact, tolerance-free comparison."""
    eligible = [threshold for threshold, agreement in candidates if agreement >= minimum_agreement]
    return max(eligible, default=0.0)


def load_physical_calibration(path: str | Path) -> PhysicalCalibrationRegression:
    """Parse the existing calibration artifact without recalibrating it."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload["selected_measurement"]
    rows = selected["per_mask_ratio"]
    return PhysicalCalibrationRegression(
        lambdas={float(key): float(value) for key, value in payload["selected_lambda_by_mask_ratio"].items()},
        physical_sparsity={float(row["mask_ratio"]): float(row["achieved_physical_sparsity"]) for row in rows},
        dense_agreement={float(row["mask_ratio"]): float(row["agreement_with_lambda_zero"]) for row in rows},
        overall_physical_sparsity=float(selected["physical_block_sparsity"]),
        skipped_tiles=int(selected["skipped_physical_blocks"]),
        total_tiles=int(selected["total_physical_blocks"]),
    )


def calibration_matches_runtime_defaults(
    calibration: PhysicalCalibrationRegression,
    *,
    high_noise_lambda: float,
    mid_noise_lambda: float,
    low_noise_lambda: float,
) -> bool:
    """Require bit-for-bit Python-float equality with the selected entries."""
    return calibration.lambdas == {
        0.15: low_noise_lambda,
        0.5: mid_noise_lambda,
        0.9: high_noise_lambda,
    }


class PhysicalTileScoreCollector:
    """Collect dense-pass R_j scores without synchronizing once per tile."""

    def __init__(self) -> None:
        self._records: list[tuple[int, torch.Tensor]] = []

    def __call__(self, layer: int, q_tile: int, kv_tile: int, scores: torch.Tensor) -> None:
        del q_tile, kv_tile
        # Each record is [batch, head]. Retaining detached GPU tensors keeps
        # collection cheap; scores are copied to CPU only when requested.
        self._records.append((layer, scores.detach()))

    def bucketed_scores(
        self,
        sample_buckets: list[object],
        *,
        by_layer: bool = False,
        by_head: bool = False,
    ) -> dict[tuple[object, ...], torch.Tensor]:
        grouped: dict[tuple[object, ...], list[torch.Tensor]] = defaultdict(list)
        if not self._records:
            return {}
        if self._records[0][1].shape[0] != len(sample_buckets):
            raise ValueError("sample_buckets must contain one bucket per batch row")
        # Stack on the source device and perform one bulk transfer. A 4096-token
        # calibration contains millions of scores, so per-score Python work or
        # per-tile device synchronization would dominate the dense pass.
        layers = torch.tensor([layer for layer, _ in self._records])
        stacked = torch.stack([scores.float() for _, scores in self._records]).cpu()
        unique_layers = layers.unique(sorted=True).tolist() if by_layer else [None]
        heads = range(stacked.shape[2]) if by_head else [None]
        for sample, bucket in enumerate(sample_buckets):
            for layer in unique_layers:
                layer_scores = stacked if layer is None else stacked[layers == layer]
                for head in heads:
                    key = (bucket,)
                    if layer is not None:
                        key += (layer,)
                    if head is not None:
                        key += (head,)
                    values = layer_scores[:, sample] if head is None else layer_scores[:, sample, head]
                    grouped[key].append(values.reshape(-1))
        return {key: torch.cat(values) for key, values in grouped.items()}


def lambda_for_physical_sparsity(scores: torch.Tensor, target: float) -> float:
    """Return the target quantile of physical scores, clamped to kernel range."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty vector")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target physical sparsity must be in [0, 1]")
    # The kernel accepts lambda <= 1. Scores above one are unskippable for any
    # legal threshold and naturally collapse to that upper boundary.
    return float(torch.quantile(scores.clamp_max(1.0), target).item())


def measured_physical_sparsity(scores: torch.Tensor, threshold: float) -> float:
    """Apply the kernel's strict R_j < lambda rule to collected scores."""
    return float((scores < threshold).float().mean().item())
