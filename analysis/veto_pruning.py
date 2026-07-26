"""Offline features and metrics for denoising-aware veto-row pruning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


STATE_MASKED = 0
STATE_NEWLY_REVEALED = 1
STATE_PREVIOUSLY_REVEALED = 2
STATE_STABLE_VISIBLE = 3
STATE_PREFIX = 4


@dataclass(frozen=True)
class PolicyMetrics:
    policy: str
    split: str
    total_tiles: int
    baseline_skipped_tiles: int
    baseline_kept_tiles: int
    additional_skipped_tiles: int
    baseline_physical_sparsity: float
    new_physical_sparsity: float
    additional_physical_sparsity: float
    downstream_work_reduction: float
    unsafe_removals: int
    unsafe_removal_rate: float
    unsafe_wilson95_upper: float
    new_max_removals: int
    affected_masked_rows: int
    maximum_masked_relative_error: float
    p99_masked_relative_error: float
    mean_masked_relative_error: float
    pv_tiles_eliminated: int
    v_bytes_avoided: int
    softmax_elements_avoided: int
    pv_flops_avoided: int
    qk_flops_avoided: int
    optimistic_kernel_speedup: float
    half_downstream_kernel_speedup: float

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def wilson_upper(errors: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 1.0
    p = errors / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return (center + radius) / denominator


def online_attention_mass(
    local_logsumexp: np.ndarray,
    running_max: np.ndarray,
    running_sum: np.ndarray,
) -> np.ndarray:
    """Mass relative to the prefix seen so far plus the current tile."""
    running_lse = np.where(
        running_sum > 0,
        running_max + np.log(np.maximum(running_sum, 1e-30)),
        -np.inf,
    )
    denominator = np.logaddexp(running_lse, local_logsumexp)
    return np.exp(np.clip(local_logsumexp - denominator, -80.0, 0.0))


def state_weight(states: np.ndarray, beta: float) -> np.ndarray:
    result = np.full(states.shape, beta, dtype=np.float32)
    result[(states == STATE_MASKED) | (states == STATE_PREFIX)] = 1.0
    return result


def analytical_safe_mask(
    relative_error: np.ndarray,
    states: np.ndarray,
    valid: np.ndarray,
    introduced_new_max: np.ndarray,
    limits: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Conservative per-tile screening label, not an end-quality claim."""
    if limits is None:
        limits = {
            STATE_MASKED: 0.005,
            STATE_NEWLY_REVEALED: 0.0075,
            STATE_PREVIOUSLY_REVEALED: 0.02,
            STATE_STABLE_VISIBLE: 0.05,
            STATE_PREFIX: 0.005,
        }
    row_limits = np.full(states.shape, np.inf, dtype=np.float32)
    for state, value in limits.items():
        row_limits[states == state] = value
    row_safe = (~valid) | (relative_error <= row_limits)
    return row_safe.all(axis=1) & ~introduced_new_max


def calibrated_thresholds(
    score: np.ndarray,
    safe: np.ndarray,
    eligible: np.ndarray,
    group: np.ndarray,
    calibration: np.ndarray,
    *,
    max_wilson_upper: float = 0.01,
    minimum_support: int = 256,
) -> dict[int, float]:
    """Largest per-group score threshold satisfying an unsafe-event bound."""
    result: dict[int, float] = {}
    for value in np.unique(group):
        selected = calibration & eligible & (group == value) & np.isfinite(score)
        indices = np.flatnonzero(selected)
        if indices.size < minimum_support:
            result[int(value)] = -math.inf
            continue
        order = indices[np.argsort(score[indices], kind="stable")]
        unsafe = (~safe[order]).astype(np.int64)
        cumulative = np.cumsum(unsafe)
        trials = np.arange(1, order.size + 1)
        p = cumulative / trials
        z = 1.959963984540054
        denominator = 1.0 + z * z / trials
        upper = (
            p
            + z * z / (2.0 * trials)
            + z
            * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        ) / denominator
        acceptable = np.flatnonzero((trials >= minimum_support) & (upper <= max_wilson_upper))
        if not acceptable.size:
            result[int(value)] = -math.inf
            continue
        last = int(acceptable[-1])
        result[int(value)] = float(np.nextafter(score[order[last]], math.inf))
    return result


def apply_group_thresholds(
    score: np.ndarray, group: np.ndarray, thresholds: Mapping[int, float]
) -> np.ndarray:
    selected = np.zeros(score.shape, dtype=bool)
    for value, threshold in thresholds.items():
        selected |= (group == value) & (score < threshold)
    return selected


def policy_metrics(
    policy: str,
    split: str,
    selected: np.ndarray,
    split_mask: np.ndarray,
    baseline_skip: np.ndarray,
    safe: np.ndarray,
    new_max: np.ndarray,
    masked_error_max: np.ndarray,
    masked_error_sum: np.ndarray,
    affected_masked_rows: np.ndarray,
    *,
    kv_tile_size: int = 64,
    q_tile_size: int = 128,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> PolicyMetrics:
    scope = split_mask
    removed = scope & selected & ~baseline_skip
    total = int(scope.sum())
    skipped = int((scope & baseline_skip).sum())
    kept = total - skipped
    additional = int(removed.sum())
    unsafe = int((removed & ~safe).sum())
    newmax = int((removed & new_max).sum())
    errors = masked_error_max[removed]
    baseline_sparsity = skipped / total if total else 0.0
    new_sparsity = (skipped + additional) / total if total else 0.0
    work_reduction = additional / kept if kept else 0.0
    optimistic = 1.0 / max(1.0 - work_reduction, 1e-12)
    half_downstream = 1.0 / max(1.0 - 0.5 * work_reduction, 1e-12)
    return PolicyMetrics(
        policy=policy,
        split=split,
        total_tiles=total,
        baseline_skipped_tiles=skipped,
        baseline_kept_tiles=kept,
        additional_skipped_tiles=additional,
        baseline_physical_sparsity=baseline_sparsity,
        new_physical_sparsity=new_sparsity,
        additional_physical_sparsity=additional / total if total else 0.0,
        downstream_work_reduction=work_reduction,
        unsafe_removals=unsafe,
        unsafe_removal_rate=unsafe / additional if additional else 0.0,
        unsafe_wilson95_upper=wilson_upper(unsafe, additional),
        new_max_removals=newmax,
        affected_masked_rows=int(affected_masked_rows[removed].sum()),
        maximum_masked_relative_error=float(errors.max(initial=0.0)),
        p99_masked_relative_error=float(np.quantile(errors, 0.99)) if errors.size else 0.0,
        mean_masked_relative_error=float(
            masked_error_sum[removed].sum()
            / max(affected_masked_rows[removed].sum(), 1)
        ),
        pv_tiles_eliminated=additional,
        v_bytes_avoided=additional * kv_tile_size * head_dim * dtype_bytes,
        softmax_elements_avoided=additional * q_tile_size * kv_tile_size,
        pv_flops_avoided=additional * 2 * q_tile_size * kv_tile_size * head_dim,
        qk_flops_avoided=0,
        optimistic_kernel_speedup=optimistic,
        half_downstream_kernel_speedup=half_downstream,
    )

