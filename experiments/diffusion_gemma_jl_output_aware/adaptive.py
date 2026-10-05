"""Uncertainty-adaptive nested-sketch routing.

This module deliberately keeps the value-direction criterion identical to the
fixed-rank/full-dimensional router.  It only changes how many coordinates are
used before a tile decision is resolved.  The implementation is a trusted
PyTorch reference used by the frozen-history harness and by the native router
when ``Config(method='adaptive')`` is selected.  It is intentionally free of
model- or benchmark-specific features.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_STAGES = (2, 8, 32, 64)


@dataclass(frozen=True)
class CascadeConfig:
    """Frozen cascade policy.

    ``interval_delta`` is a *total* error budget for one tile.  It is divided
    over valid query rows and tested ranks, so the advertised interval is not
    an uncorrected repeatedly-applied 95% interval.
    """

    stages: Tuple[int, ...] = DEFAULT_STAGES
    interval_delta: float = 1e-3
    deterministic_screening: bool = False
    exact_fallback: bool = True
    band_name: str = "analytic"

    def __post_init__(self):
        stages = tuple(int(r) for r in self.stages)
        if not stages or tuple(sorted(set(stages))) != stages:
            raise ValueError("cascade stages must be strictly increasing")
        if stages[0] < 1:
            raise ValueError("cascade rank must be positive")
        if not (0.0 < self.interval_delta < 1.0):
            raise ValueError("interval_delta must lie in (0,1)")


@dataclass
class CascadeResult:
    """Decision plus auditable work and uncertainty counters."""

    skip: torch.Tensor
    rank_used: torch.Tensor
    interval_lower: torch.Tensor
    interval_upper: torch.Tensor
    projected_score: torch.Tensor
    exact_score: Optional[torch.Tensor]
    diagnostics: Dict[str, float] = field(default_factory=dict)


def chi_square_bounds(rank: int, delta: float, *, device=None,
                      dtype=torch.float32) -> Tuple[float, float]:
    """Return lower/upper multiplicative bounds for a projected norm.

    For a fixed vector, ``r * q_hat**2 / q**2`` is chi-square(r).  Quantiles
    are evaluated outside kernels and returned as Python scalars so callers
    can cache them.  SciPy is used here only for policy construction; the
    routing path performs tensor arithmetic with the cached constants.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0,1)")
    from scipy.stats import chi2
    # Caller normally supplies a per-row/per-stage budget.  The bounds are
    # factors multiplying the observed norm.
    lo_q = float(chi2.ppf(delta / 2.0, rank))
    hi_q = float(chi2.ppf(1.0 - delta / 2.0, rank))
    if not (0.0 < lo_q < hi_q):
        raise ValueError("invalid chi-square quantiles")
    return math.sqrt(rank / hi_q), math.sqrt(rank / lo_q)


def interval_factors(stages: Sequence[int], rows: int,
                     interval_delta: float) -> Dict[int, Tuple[float, float]]:
    """Bonferroni factors for all tested rows and rank stages."""
    rows = max(1, int(rows))
    stages = tuple(stages)
    per_test = interval_delta / (rows * len(stages))
    return {int(r): chi_square_bounds(int(r), per_test) for r in stages}


def deterministic_upper_bound(alpha: torch.Tensor, block_norm: torch.Tensor,
                              retained_norm: torch.Tensor, reference: torch.Tensor,
                              threshold: float, active: torch.Tensor,
                              supported: torch.Tensor) -> torch.Tensor:
    """Sufficient zero-projection skip test for the unchanged criterion.

    Inputs are per-query-row tensors.  A tile is safe to screen only when all
    valid supported rows have an upper bound below ``threshold``.  First
    support is never screened.
    """
    bound = alpha * (block_norm + retained_norm) / reference.clamp_min(1e-12)
    row = active & supported
    eligible = active.any(dim=-1)
    safe = (~row) | (bound < threshold)
    return eligible & supported.any(dim=-1) & safe.all(dim=-1)


def _tile_max(value: torch.Tensor, active: torch.Tensor,
              supported: torch.Tensor) -> torch.Tensor:
    """Max over valid rows, with unsupported rows excluded."""
    row = active & supported
    return value.masked_fill(~row, -torch.inf).amax(dim=-1)


def _update_state(previous_logz: torch.Tensor, previous: torch.Tensor,
                  logz: torch.Tensor, mu: torch.Tensor, include: torch.Tensor):
    """Stable online-softmax update for retained physical tiles."""
    addz = logz.masked_fill(~include, -torch.inf)
    newz = torch.logaddexp(previous_logz, addz)
    safe = torch.where(torch.isfinite(newz), newz, torch.zeros_like(newz))
    a = torch.exp(previous_logz - safe)
    b = torch.exp(addz - safe)
    return newz, a[..., None] * previous + b[..., None] * mu


def _bands_for(c: CascadeConfig, factors, stage: int, estimate: torch.Tensor,
               empirical_bands: Optional[Mapping[int, Tuple[float, float]]]):
    if empirical_bands is not None and stage in empirical_bands:
        lo, hi = empirical_bands[stage]
    else:
        lo, hi = factors[stage]
    return estimate * float(lo), estimate * float(hi)


def route(state: Mapping[str, torch.Tensor], reference: torch.Tensor,
          threshold: float, config: CascadeConfig = CascadeConfig(),
          *, exact_state: Optional[Mapping[str, torch.Tensor]] = None,
          empirical_bands: Optional[Mapping[int, Tuple[float, float]]] = None,
          block_norm: Optional[torch.Tensor] = None,
          retained_norm: Optional[torch.Tensor] = None) -> CascadeResult:
    """Route one or more physical query tiles with a nested sketch.

    ``state['mu']`` contains the *maximum* projected rank and is assumed to be
    formed from genuinely nested columns.  The function only reads the prefix
    needed at each stage.  Running state is kept at max rank, which guarantees
    correct rank promotion; ``state_coord_updates`` separately records the
    coordinates that a lazy implementation would need to maintain.

    ``exact_state`` is optional and used only for explicit maximum-rank
    resolution.  It is never consulted before a tile remains unresolved at
    the last requested sketch rank.
    """
    logz = state['logz']
    mu = state['mu'].float()
    count = state.get('count', state['active'].to(torch.int32))
    active = state.get('active', count > 0)
    if logz.ndim != 4 or mu.ndim != 5:
        raise ValueError('state must be [B,H,Q,T] and mu [B,H,Q,T,R]')
    b, h, q, tiles = logz.shape
    max_rank = mu.shape[-1]
    if max(config.stages) > max_rank:
        raise ValueError('state does not contain the requested maximum rank')
    if reference.shape != (b, h):
        raise ValueError('reference must be [B,H]')
    if exact_state is not None and exact_state['logz'].shape != logz.shape:
        raise ValueError('exact state shape mismatch')

    factors = interval_factors(config.stages, q, config.interval_delta)
    previous_logz = torch.full((b, h, q), -torch.inf, dtype=torch.float32,
                               device=logz.device)
    previous = torch.zeros((b, h, q, max_rank), dtype=torch.float32,
                           device=logz.device)
    exact_previous_logz = exact_previous = None
    if exact_state is not None:
        exact_mu = exact_state['mu'].float()
        exact_previous_logz = torch.full((b, h, q), -torch.inf, dtype=torch.float32,
                                         device=logz.device)
        exact_previous = torch.zeros((b, h, q, exact_mu.shape[-1]), dtype=torch.float32,
                                     device=logz.device)
    # Decisions are physical tiles: every valid query row must vote skip.
    skip = torch.zeros((b, h, tiles), dtype=torch.bool, device=logz.device)
    rank_used = torch.zeros((b, h, tiles), dtype=torch.int16, device=logz.device)
    lower = torch.zeros((b, h, tiles), dtype=torch.float32, device=logz.device)
    upper = torch.zeros_like(lower)
    projected = torch.zeros_like(lower)
    exact_scores = torch.full_like(lower, torch.nan, dtype=torch.float32) if exact_state is not None else None

    screened = escalated = fallback = 0
    state_coord_updates = projected_coord_evals = exact_work = 0
    for j in range(tiles):
        z = logz[..., j]
        n = count[..., j]
        a = active[..., j]
        combined = torch.logaddexp(previous_logz, z)
        alpha = torch.exp(z - torch.where(torch.isfinite(combined), combined,
                                          torch.zeros_like(combined)))
        alpha = torch.where(a, alpha, torch.zeros_like(alpha))
        supported = torch.isfinite(previous_logz)
        resolved = torch.zeros((b, h), dtype=torch.bool, device=logz.device)
        tile_skip = torch.zeros((b, h), dtype=torch.bool, device=logz.device)
        tile_keep = torch.zeros_like(tile_skip)
        chosen_lower = torch.full((b, h), -torch.inf, dtype=torch.float32, device=logz.device)
        chosen_upper = torch.full_like(chosen_lower, torch.inf)
        chosen_score = torch.full_like(chosen_lower, torch.inf)
        chosen_rank = torch.zeros((b, h), dtype=torch.int16, device=logz.device)

        # Optional sufficient bound before any projected PV.  It is deliberately
        # conservative and cannot suppress first-support tiles.
        if config.deterministic_screening and block_norm is not None and retained_norm is not None:
            screen = deterministic_upper_bound(alpha, block_norm[..., j, None], retained_norm,
                                               reference, threshold, a, supported)
            if screen.any():
                tile_skip |= screen
                resolved |= screen
                chosen_rank[screen] = 0
                screened += int(screen.sum())

        for rank in config.stages:
            unresolved = ~resolved
            if not unresolved.any():
                break
            # ``mu`` is stored as W/sqrt(max_rank).  Prefix columns become
            # R_r=W/sqrt(r) by this inexpensive scalar, preserving genuinely
            # nested projections without recomputing candidate means.
            scale = math.sqrt(max_rank / rank)
            delta = alpha[..., None] * ((mu[..., j, :rank] * scale) -
                                        (previous[..., :rank] * scale))
            estimate_rows = delta.norm(dim=-1) / reference[..., None].clamp_min(1e-12)
            estimate = _tile_max(estimate_rows, a, supported)
            lo, hi = _bands_for(config, factors, rank, estimate, empirical_bands)
            supported_tile = supported.any(dim=-1)
            eligible = a.any(dim=-1)
            can_skip = eligible & supported_tile & (hi < threshold)
            can_keep = eligible & supported_tile & (lo >= threshold)
            # Empty/first-support candidates are retained exactly as required.
            can_keep |= eligible & ~supported_tile
            take_skip = unresolved & can_skip
            take_keep = unresolved & can_keep
            take = take_skip | take_keep
            tile_skip |= take_skip
            tile_keep |= take_keep
            resolved |= take
            chosen_lower = torch.where(take, lo, chosen_lower)
            chosen_upper = torch.where(take, hi, chosen_upper)
            chosen_score = torch.where(take, estimate, chosen_score)
            chosen_rank = torch.where(take, torch.as_tensor(rank, device=logz.device, dtype=torch.int16), chosen_rank)
            projected_coord_evals += int(unresolved.sum()) * int(rank)

        unresolved = ~resolved
        if unresolved.any():
            escalated += int(unresolved.sum())
            if exact_state is not None and config.exact_fallback:
                ez = exact_state['logz'][..., j]
                emu = exact_state['mu'][..., j, :].float()
                ecombined = torch.logaddexp(exact_previous_logz, ez)
                ea = torch.exp(ez - torch.where(torch.isfinite(ecombined), ecombined,
                                                torch.zeros_like(ecombined)))
                eactive = exact_state.get('active')
                if eactive is None:
                    eactive = exact_state.get('count', torch.zeros_like(ez)[..., None]).to(torch.bool)
                eactive = eactive[..., j]
                ea = torch.where(eactive, ea, torch.zeros_like(ea))
                # The exact running state is maintained in parallel, so a
                # fallback is evaluated on exactly the same retained support,
                # even when the full value width differs from the sketch rank.
                eprev = exact_previous
                edelta = ea[..., None] * (emu - eprev)
                erow = edelta.norm(dim=-1) / reference[..., None].clamp_min(1e-12)
                esupported = torch.isfinite(exact_previous_logz)
                escore = _tile_max(erow, eactive, esupported)
                eeligible_all = exact_state.get('eligible')
                if eeligible_all is None:
                    eeligible_all = exact_state.get('active', eactive[..., None]).any(-2)
                eeligible = eeligible_all[..., j]
                take_skip = unresolved & eeligible & (escore < threshold)
                take_keep = unresolved & ~(take_skip)
                tile_skip |= take_skip
                tile_keep |= take_keep
                chosen_lower = torch.where(unresolved, escore, chosen_lower)
                chosen_upper = torch.where(unresolved, escore, chosen_upper)
                chosen_score = torch.where(unresolved, escore, chosen_score)
                chosen_rank = torch.where(unresolved, torch.as_tensor(-1, device=logz.device, dtype=torch.int16), chosen_rank)
                exact_scores[..., j] = escore
                fallback += int(unresolved.sum())
                exact_work += int(unresolved.sum()) * int(emu.shape[-1])
            else:
                # Without an exact oracle, unresolved max-rank decisions are
                # retained.  This is safer than silently turning uncertainty into
                # a skip and is explicitly reported as an unresolved keep.
                tile_keep |= unresolved
                chosen_rank = torch.where(unresolved, torch.as_tensor(max(config.stages), device=logz.device, dtype=torch.int16), chosen_rank)
                chosen_score = torch.where(unresolved, torch.full_like(chosen_score, float('nan')), chosen_score)

        tile_skip &= ~tile_keep
        valid_tile = a.any(dim=-1)
        tile_skip &= valid_tile
        skip[..., j] = tile_skip
        rank_used[..., j] = chosen_rank
        lower[..., j] = chosen_lower
        upper[..., j] = chosen_upper
        projected[..., j] = chosen_score

        include = a & ~tile_skip[..., None]
        previous_logz, previous = _update_state(previous_logz, previous, z, mu[..., j, :], include)
        if exact_state is not None:
            ez = exact_state['logz'][..., j]
            emu = exact_state['mu'][..., j, :].float()
            eactive = exact_state.get('active')
            if eactive is None:
                eactive = exact_state.get('count', torch.zeros_like(ez)).to(torch.bool)
            eactive = eactive[..., j]
            exact_previous_logz, exact_previous = _update_state(
                exact_previous_logz, exact_previous, ez, emu,
                eactive & ~tile_skip[..., None])
        # Every retained block must update all dormant coordinates before a later
        # rank promotion; this count prevents false claims of cheap expansion.
        state_coord_updates += int(include.sum()) * max_rank

    diagnostics = dict(screened_tiles=float(screened), escalated_tiles=float(escalated),
                       exact_fallback_tiles=float(fallback), exact_fallback_coordinates=float(exact_work),
                       projected_coordinate_evals=float(projected_coord_evals),
                       retained_state_coordinate_updates=float(state_coord_updates),
                       fallback_rate=float(fallback / max(1, escalated)),
        unresolved_keep_rate=float((rank_used == max(config.stages)).float().mean().item()))
    return CascadeResult(skip=skip, rank_used=rank_used, interval_lower=lower,
                          interval_upper=upper, projected_score=projected,
                          exact_score=exact_scores, diagnostics=diagnostics)


def fixed_route(state, reference, threshold, rank):
    """Fixed-rank control using the same trusted online criterion."""
    from .config import Config
    from . import reference as trusted
    return trusted.route(state, reference, Config(method='centered', rank=rank,
        family='gaussian', log_threshold=math.log(threshold) if threshold > 0 else -math.inf))


def fit_empirical_bands(samples: Mapping[int, torch.Tensor], *, lower_q=.005,
                        upper_q=.995) -> Dict[int, Tuple[float, float]]:
    """Fit conservative multiplicative bands from development residuals.

    ``samples[r]`` contains ``true_score / projected_score`` for rank ``r``.
    The ratios are clipped only to keep malformed zero-score rows out of the
    log quantile; the returned factors are positive and serializable.
    """
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError('invalid empirical quantiles')
    out = {}
    for rank, values in samples.items():
        x = torch.as_tensor(values, dtype=torch.float64).flatten()
        x = x[torch.isfinite(x) & (x > 0)]
        if x.numel() < 8:
            raise ValueError(f'insufficient empirical band samples for rank {rank}: {x.numel()}')
        logs = x.log()
        lo, hi = torch.quantile(logs, torch.tensor([lower_q, upper_q], dtype=logs.dtype)).exp()
        out[int(rank)] = (float(lo), float(hi))
    return out


def decision_metrics(candidate: torch.Tensor, exact: torch.Tensor,
                     eligible: torch.Tensor) -> Dict[str, float]:
    """Physical-tile disagreement and false-skip/keep rates."""
    if candidate.shape != exact.shape or eligible.shape != exact.shape:
        raise ValueError('decision metric shape mismatch')
    c = candidate & eligible
    e = exact & eligible
    return dict(eligible_tiles=float(eligible.sum()), skipped_tiles=float(c.sum()),
                exact_skipped_tiles=float(e.sum()), disagreement=float((c ^ e).sum()),
                false_skips=float((c & ~e).sum()), false_keeps=float((~c & e).sum()),
                disagreement_rate=float(((c ^ e).sum() / eligible.sum().clamp_min(1)).item()))
