"""Cached-statistics partial refresh for the V31 GLOBAL selector.

This module contains only selector state and a trusted reference
implementation.  It knows nothing about FA4, K/V caches, or the attention
output.  A dense observation supplies raw per-row prefix-tile log masses; a
cheap event supplies fresh masses for the currently retained tiles.  The
returned map is installed for the next attention call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import torch


DEFAULT_DECAY_LAMBDA = math.log(2.0) / 2.0
DEFAULT_DECAY_FLOOR = 0.25
DEFAULT_EPS = 1.0e-6


def _check_score(score: torch.Tensor) -> None:
    if score.ndim != 2:
        raise ValueError("score must have shape [U, J]")


def _check_raw(raw: torch.Tensor) -> None:
    if raw.ndim != 3:
        raise ValueError("raw log-mass must have shape [U, J, R]")


def _check_map(selected: torch.Tensor, shape: tuple[int, int], name: str = "selected") -> None:
    if selected.ndim != 2 or tuple(selected.shape) != shape or selected.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape {shape}")


def log_decay(age: torch.Tensor, decay_lambda: float = 0.0,
              floor: float = DEFAULT_DECAY_FLOOR) -> torch.Tensor:
    """Return ``log(max(floor, exp(-lambda * age)))`` elementwise."""
    if not 0.0 < float(floor) <= 1.0:
        raise ValueError("decay floor must be in (0, 1]")
    if float(decay_lambda) < 0.0 or not math.isfinite(float(decay_lambda)):
        raise ValueError("decay lambda must be finite and nonnegative")
    agef = age.to(torch.float32)
    if float(decay_lambda) == 0.0:
        return torch.zeros_like(agef)
    return torch.clamp(-float(decay_lambda) * agef, min=math.log(float(floor)))


def normalize_complete_support(raw: torch.Tensor) -> torch.Tensor:
    """Normalize over all prefix tiles for each query row, without NaNs."""
    _check_raw(raw)
    denom = torch.logsumexp(raw, dim=1, keepdim=True)
    finite_denom = torch.isfinite(denom)
    safe = torch.where(finite_denom, denom, torch.zeros_like(denom))
    normalized = raw - safe
    return torch.where(torch.isfinite(normalized) & finite_denom,
                       normalized, torch.full_like(normalized, -math.inf))


def hybrid_log_mass(cached_raw: torch.Tensor, fresh_raw: torch.Tensor,
                    retained: torch.Tensor) -> torch.Tensor:
    """Replace cached statistics only for retained physical tiles."""
    _check_raw(cached_raw)
    _check_raw(fresh_raw)
    if fresh_raw.shape != cached_raw.shape:
        raise ValueError("cached and fresh statistics must have identical geometry")
    _check_map(retained, cached_raw.shape[:2], "retained")
    return torch.where(retained[..., None], fresh_raw, cached_raw)


def hybrid_qblock_score(dense_z: torch.Tensor, fresh_z: Optional[torch.Tensor] = None,
                        fresh_mask: Optional[torch.Tensor] = None,
                        current: Optional[torch.Tensor] = None,
                        ages: Optional[torch.Tensor] = None,
                        decay_lambda: float = 0.0,
                        decay_floor: float = DEFAULT_DECAY_FLOOR) -> torch.Tensor:
    """Return V31's worst-row normalized prefix-tile mass score.

    The optional ``current`` and ``ages`` arguments apply decay outside the
    current map.  They are retained for compatibility with the adapter's
    first implementation; the state object below is the preferred API.
    """
    z = dense_z
    if fresh_z is not None or fresh_mask is not None:
        if fresh_z is None or fresh_mask is None:
            raise ValueError("fresh_z and fresh_mask must be supplied together")
        z = hybrid_log_mass(dense_z, fresh_z, fresh_mask)
    normalized = normalize_complete_support(z)
    score = normalized.amax(dim=-1)
    score = torch.where(torch.isfinite(score), score,
                        torch.full_like(score, -math.inf))
    if current is not None or ages is not None:
        if current is None or ages is None:
            raise ValueError("current and ages must be supplied together")
        _check_map(current, score.shape, "current")
        if ages.shape != score.shape:
            raise ValueError("ages/current geometry mismatch")
        score = score + torch.where(~current,
                                    log_decay(ages, decay_lambda, decay_floor),
                                    torch.zeros_like(score))
    return score


def _stable_order(values: torch.Tensor, descending: bool) -> torch.Tensor:
    # Columns are physical ids in ascending order.  Stable sorting therefore
    # provides deterministic lower-id tie breaking on CPU and CUDA.
    return torch.argsort(values, dim=-1, descending=descending, stable=True)


def _row_topk(score: torch.Tensor, k: int, value_score: Optional[torch.Tensor] = None,
              mass_tolerance: float = 0.0) -> torch.Tensor:
    j = score.numel()
    k = max(0, min(int(k), j))
    out = torch.zeros(j, dtype=torch.bool, device=score.device)
    if k == 0:
        return out
    order = _stable_order(score, True)
    if value_score is None or float(mass_tolerance) <= 0.0:
        out[order[:k]] = True
        return out
    if value_score.shape != score.shape:
        raise ValueError("value score geometry mismatch")
    boundary = score[order[k - 1]]
    near = torch.isfinite(score) & (torch.abs(score - boundary) <= float(mass_tolerance))
    base = order[~near[order]]
    selected = list(base[:k].tolist())
    ids = torch.nonzero(near, as_tuple=False).flatten()
    if ids.numel() and len(selected) < k:
        vo = _stable_order(value_score[ids], True)
        selected.extend(ids[vo[:k - len(selected)]].tolist())
    if len(selected) < k:
        for idx in order.tolist():
            if idx not in selected:
                selected.append(idx)
                if len(selected) == k:
                    break
    out[torch.tensor(selected[:k], device=score.device)] = True
    return out


def topk_fixed(score: torch.Tensor, k: int, value_score: Optional[torch.Tensor] = None,
               mass_tolerance: float = 0.0) -> torch.Tensor:
    """Select exactly ``k`` physical tiles per query block."""
    _check_score(score)
    if value_score is not None and value_score.shape != score.shape:
        raise ValueError("value score geometry mismatch")
    if score.is_cuda and value_score is None:
        k = max(0, min(int(k), score.shape[-1]))
        out = torch.zeros_like(score, dtype=torch.bool)
        if k:
            ids = _stable_order(score, True)[:, :k]
            out.scatter_(1, ids, True)
        return out
    return torch.stack([_row_topk(row, k, None if value_score is None else value_score[i],
                                  mass_tolerance)
                        for i, row in enumerate(score)], dim=0)


def bounded_swaps(score: torch.Tensor, current: torch.Tensor, k: int,
                  frontier_m: int, swap_budget: int,
                  value_score: Optional[torch.Tensor] = None,
                  mass_tolerance: float = 0.0) -> torch.Tensor:
    """Apply at most ``B`` swaps from a total top-``M`` frontier."""
    _check_score(score)
    _check_map(current, tuple(score.shape), "current")
    if value_score is not None and value_score.shape != score.shape:
        raise ValueError("value score geometry mismatch")
    if float(mass_tolerance) < 0.0:
        raise ValueError("mass tolerance must be nonnegative")
    u, j = score.shape
    k = max(0, min(int(k), j))
    m = max(k, min(int(frontier_m), j))
    b = max(0, int(swap_budget))
    if score.is_cuda:
        counts = current.sum(-1)
        if counts.numel() and not bool(torch.all(counts == k)):
            current = topk_fixed(score, k)
        frontier = _stable_order(score, True)[:, :m]
        held = current.clone()
        used = torch.zeros((u, m), dtype=torch.bool, device=score.device)
        for _ in range(b):
            in_held = held.gather(1, frontier)
            available = ~in_held & ~used
            any_candidate = available.any(-1)
            pos = available.to(torch.int32).argmax(-1)
            cand = frontier.gather(1, pos[:, None]).squeeze(1)
            weak_score = score.masked_fill(~held, float("inf"))
            weak = weak_score.argmin(-1)
            cm = score.gather(1, cand[:, None]).squeeze(1)
            wm = score.gather(1, weak[:, None]).squeeze(1)
            diff = cm - wm
            better = any_candidate & (diff > float(mass_tolerance))
            if value_score is not None:
                cv = value_score.gather(1, cand[:, None]).squeeze(1)
                wv = value_score.gather(1, weak[:, None]).squeeze(1)
                better = better | (any_candidate & (diff.abs() <= float(mass_tolerance))
                                   & torch.isfinite(cv) & torch.isfinite(wv) & (cv > wv))
            used.scatter_(1, pos[:, None], True)
            held.scatter_(1, weak[:, None], ~better[:, None])
            held.scatter_(1, cand[:, None], better[:, None])
        return held
    rows = []
    for i in range(u):
        cur = current[i].clone()
        if int(cur.sum()) != k:
            cur = topk_fixed(score[i:i + 1], k)[0]
        frontier = _stable_order(score[i], True)[:m]
        candidates = [int(x) for x in frontier.tolist() if not bool(cur[int(x)])]
        swaps = 0
        for cand in candidates:
            if swaps >= b:
                break
            held = torch.nonzero(cur, as_tuple=False).flatten()
            if not held.numel():
                break
            weakest = held[_stable_order(score[i, held], False)[0]]
            diff = float((score[i, cand] - score[i, weakest]).item())
            better = diff > float(mass_tolerance)
            if not better and abs(diff) <= float(mass_tolerance) and value_score is not None:
                cv, wv = value_score[i, cand], value_score[i, weakest]
                if bool(torch.isfinite(cv) & torch.isfinite(wv)):
                    better = bool(cv > wv)
            if better:
                cur[weakest] = False
                cur[cand] = True
                swaps += 1
        rows.append(cur)
    return torch.stack(rows, dim=0) if rows else current.new_zeros((0, j))


def advance_ages(previous: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Reset selected ages and increment ages that remain outside the map."""
    if previous.shape != selected.shape:
        raise ValueError("age/map geometry mismatch")
    _check_map(selected, tuple(previous.shape), "selected")
    age = previous.to(torch.int32).clamp_min(0)
    age = torch.where(selected, torch.zeros_like(age), torch.clamp(age + 1, max=32767))
    return age.to(torch.int16)


def map_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("map geometry mismatch")
    _check_map(a, tuple(a.shape), "a")
    _check_map(b, tuple(b.shape), "b")
    return (a & b).sum(-1).float() / (a | b).sum(-1).clamp_min(1).float()


def drift_metric(fresh: torch.Tensor, cached: torch.Tensor, retained: torch.Tensor,
                 eps: float = DEFAULT_EPS) -> tuple[torch.Tensor, bool]:
    """Return max retained relative drift and a non-finite flag."""
    _check_raw(fresh)
    _check_raw(cached)
    if fresh.shape != cached.shape:
        raise ValueError("fresh/cached geometry mismatch")
    _check_map(retained, tuple(cached.shape[:2]), "retained")
    active = retained[..., None]
    finite = torch.isfinite(fresh) & torch.isfinite(cached)
    matching_structural_inf = (~torch.isfinite(fresh) & ~torch.isfinite(cached)
                               & (fresh == cached))
    bad = bool((active & ~finite & ~matching_structural_inf).any().item())
    rel = (fresh - cached).abs() / cached.abs().clamp_min(float(eps))
    rel = torch.where(active & finite, rel, torch.zeros_like(rel))
    value = rel.amax() if rel.numel() else fresh.new_zeros(())
    if bad:
        value = torch.tensor(float("inf"), device=fresh.device)
    return value, bad


def full_information_omission_error(raw_log_mass: torch.Tensor, tile_values: torch.Tensor,
                                    dense_output: torch.Tensor,
                                    eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Offline full-information single-tile omission error diagnostic."""
    _check_raw(raw_log_mass)
    if tile_values.ndim != 4 or tile_values.shape[:3] != raw_log_mass.shape:
        raise ValueError("tile_values must have shape [U, J, R, D]")
    if dense_output.ndim != 3 or dense_output.shape[:2] != raw_log_mass.shape[::2]:
        raise ValueError("dense_output must have shape [U, R, D]")
    if dense_output.shape[-1] != tile_values.shape[-1]:
        raise ValueError("dense_output value dimension mismatch")
    alpha = normalize_complete_support(raw_log_mass).exp()
    total_mass = alpha.sum(1)
    total_num = (alpha[..., None] * tile_values).sum(1)
    errors = []
    for j in range(raw_log_mass.shape[1]):
        remain = total_mass - alpha[:, j]
        omitted = (total_num - alpha[:, j, :, None] * tile_values[:, j]) / remain.clamp_min(float(eps))[..., None]
        errors.append((dense_output - omitted).norm(dim=-1))
    return torch.stack(errors, dim=1)


def strict_intersection(control: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Return the unmatched-budget intersection diagnostic."""
    if control.shape != other.shape:
        raise ValueError("map geometry mismatch")
    _check_map(control, tuple(control.shape), "control")
    _check_map(other, tuple(other.shape), "other")
    return control & other


def filled_intersection(control: torch.Tensor, other: torch.Tensor, score: torch.Tensor,
                        k: int) -> torch.Tensor:
    """Fill a strict intersection back to ``k`` by stable hybrid mass."""
    inter = strict_intersection(control, other)
    _check_score(score)
    if score.shape != inter.shape:
        raise ValueError("score/intersection geometry mismatch")
    out = inter.clone()
    for i in range(out.shape[0]):
        missing = max(0, int(k) - int(out[i].sum()))
        if missing:
            order = _stable_order(score[i].masked_fill(out[i], -math.inf), True)
            out[i, order[:missing]] = True
    return out


@dataclass
class RefreshResult:
    selected: torch.Tensor
    score: torch.Tensor
    hybrid_raw: torch.Tensor
    fallback: bool = False
    nonfinite: bool = False
    no_op: bool = False
    swaps: int = 0
    mean_iou: float = 1.0
    max_drift: float = 0.0
    stale_selected_fraction: float = 0.0


@dataclass
class CachedRefreshState:
    """Selector-only state held between native attention calls."""
    raw_log_mass: torch.Tensor
    selected: torch.Tensor
    age: torch.Tensor
    dense_output: Optional[torch.Tensor] = None
    value_statistics: Optional[torch.Tensor] = None
    cheap_events: int = 0
    dense_refreshes: int = 1
    no_op_updates: int = 0
    swap_count: int = 0
    fallback_count: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0
    max_drift: float = 0.0
    age_histogram: dict = field(default_factory=dict)

    @classmethod
    def from_dense(cls, raw_log_mass: torch.Tensor, selected: torch.Tensor,
                   dense_output: Optional[torch.Tensor] = None,
                   value_statistics: Optional[torch.Tensor] = None) -> "CachedRefreshState":
        _check_raw(raw_log_mass)
        _check_map(selected, tuple(raw_log_mass.shape[:2]), "selected")
        counts = selected.sum(-1)
        if counts.numel() and not bool((counts == counts[0]).all()):
            raise ValueError("fixed-budget map must have one K for every query block")
        return cls(raw_log_mass.detach().clone(), selected.detach().clone(),
                   torch.zeros_like(selected, dtype=torch.int16),
                   None if dense_output is None else dense_output.detach().clone(),
                   None if value_statistics is None else value_statistics.detach().clone())

    def dense_refresh(self, raw_log_mass: torch.Tensor, selected: torch.Tensor,
                      dense_output: Optional[torch.Tensor] = None,
                      value_statistics: Optional[torch.Tensor] = None) -> None:
        fresh = self.from_dense(raw_log_mass, selected, dense_output, value_statistics)
        self.raw_log_mass, self.selected, self.age = fresh.raw_log_mass, fresh.selected, fresh.age
        self.dense_output, self.value_statistics = fresh.dense_output, fresh.value_statistics
        self.dense_refreshes += 1

    def refresh(self, fresh_raw: torch.Tensor, *, policy: str = "bounded_swaps",
                frontier_m: Optional[int] = None, swap_budget: int = 8,
                decay_lambda: float = 0.0, decay_floor: float = DEFAULT_DECAY_FLOOR,
                value_score: Optional[torch.Tensor] = None,
                mass_tolerance: float = 0.0,
                drift_threshold: Optional[float] = None,
                eps: float = DEFAULT_EPS,
                phase_protected: bool = False) -> RefreshResult:
        """Run one cheap event and update state for the next attention call."""
        if policy not in ("global_topk", "bounded_swaps"):
            raise ValueError("policy must be global_topk or bounded_swaps")
        if fresh_raw.shape != self.raw_log_mass.shape:
            raise ValueError("fresh statistics geometry mismatch")
        retained = self.selected
        drift, nonfinite = drift_metric(fresh_raw, self.raw_log_mass, retained, eps)
        drift_value = float(drift.item())
        self.max_drift = max(self.max_drift, drift_value)
        if nonfinite or (drift_threshold is not None and drift_value > float(drift_threshold)):
            self.fallback_count += 1
            return RefreshResult(retained.clone(), hybrid_qblock_score(self.raw_log_mass),
                                 self.raw_log_mass.clone(), fallback=True,
                                 nonfinite=nonfinite, max_drift=drift_value)
        hybrid = hybrid_log_mass(self.raw_log_mass, fresh_raw, retained)
        score = hybrid_qblock_score(hybrid)
        k = int(retained.sum(-1).min().item()) if retained.numel() else 0
        if policy == "global_topk":
            selected = topk_fixed(score, k, value_score, mass_tolerance)
        else:
            m = score.shape[-1] if frontier_m is None else int(frontier_m)
            selected = bounded_swaps(score, retained, k, m, swap_budget,
                                     value_score, mass_tolerance)
        if phase_protected and self.cheap_events == 0:
            selected = retained.clone()
        stale = ((selected & (self.age > 0)).sum().float() /
                 selected.sum().clamp_min(1)).item()
        new_age = advance_ages(self.age, selected)
        iou = map_iou(retained, selected)
        swaps = int((retained & ~selected).sum().item())
        no_op = bool(torch.equal(retained, selected))
        # Newly admitted tiles intentionally keep cached statistics until the
        # next cheap event; this is the defined age/freshness behavior.
        self.raw_log_mass = torch.where(retained[..., None], fresh_raw, self.raw_log_mass)
        self.selected, self.age = selected, new_age
        self.cheap_events += 1
        self.swap_count += swaps
        self.iou_sum += float(iou.mean().item())
        self.iou_count += int(iou.numel())
        if no_op:
            self.no_op_updates += 1
        hist = torch.bincount(new_age.to(torch.int64).flatten(), minlength=1)
        self.age_histogram = {int(i): int(v) for i, v in enumerate(hist.tolist()) if v}
        return RefreshResult(selected.clone(), score, hybrid, no_op=no_op, swaps=swaps,
                             mean_iou=float(iou.mean().item()), max_drift=drift_value,
                             stale_selected_fraction=float(stale))

    def receipt(self) -> dict:
        return {
            "dense_refresh_count": int(self.dense_refreshes),
            "cheap_refresh_count": int(self.cheap_events),
            "no_op_count": int(self.no_op_updates),
            "swap_count": int(self.swap_count),
            "forced_dense_fallback_count": int(self.fallback_count),
            "mean_iou": self.iou_sum / self.iou_count if self.iou_count else None,
            "max_qk_statistic_drift": float(self.max_drift),
            "age_histogram": dict(self.age_histogram),
            "state_role": "selector_only",
            "attention_output_path": "unchanged_native_fa4",
        }
