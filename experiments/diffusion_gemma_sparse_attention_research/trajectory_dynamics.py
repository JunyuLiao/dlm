"""Pure reference diagnostics for native DiffusionGemma trajectory studies.

These functions deliberately do not run the model, alter the sampler, or choose
task labels.  They provide small NumPy or Python oracles for the statistics that
the native diagnostic protocol records: stable/confident position attribution,
attention response, prefix renormalization, token-margin safety, correction and
break flow, and within-canvas support validity.

The module is intentionally independent of CUDA and vLLM so its identities can
be tested before adding a native observer or a Hopper kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class StableConfidenceAttribution:
    """Retrospective position-level labels; native sampling is not changed."""

    stable: Array
    confident: Array
    stable_and_confident: Array
    first_stable_call: Array
    first_confident_call: Array
    terminal_stable: Array
    terminal_confident: Array


def stable_confidence_attribution(
    argmax_history: Array,
    entropy_history: Array,
    confidence_threshold: float,
    *,
    dwell: int = 1,
) -> StableConfidenceAttribution:
    """Compute retrospective stable/confident labels from native logits.

    Parameters
    ----------
    argmax_history:
        Integer clean-argmax tokens with shape ``[calls, positions]``.
    entropy_history:
        Token entropies in nats with the same shape.
    confidence_threshold:
        Diagnostic per-token entropy threshold.  It never changes native
        acceptance or stopping behavior.
    dwell:
        Number of consecutive calls required for a stable/confident label.  A
        stable label at call ``n`` compares argmax at ``n`` with ``n-1`` and
        requires ``dwell`` consecutive equal transitions.

    The first-call arrays use ``-1`` where the condition is never observed.
    ``terminal_*`` are the final-call labels, not claims that native Diffusion-
    Gemma individually committed those positions.
    """

    a = np.asarray(argmax_history)
    h = np.asarray(entropy_history, dtype=float)
    if a.ndim != 2 or h.shape != a.shape:
        raise ValueError("argmax_history and entropy_history must both have shape [calls, positions]")
    if a.shape[0] == 0:
        raise ValueError("at least one call is required")
    if not isinstance(dwell, (int, np.integer)) or dwell < 1:
        raise ValueError("dwell must be a positive integer")
    if not isfinite(float(confidence_threshold)) or confidence_threshold < 0:
        raise ValueError("confidence_threshold must be finite and nonnegative")

    calls, positions = a.shape
    confident = h < float(confidence_threshold)
    stable = np.zeros((calls, positions), dtype=bool)
    if calls > 1:
        stable[1:] = a[1:] == a[:-1]

    def dwell_labels(base: Array) -> Array:
        out = np.zeros_like(base, dtype=bool)
        if dwell == 1:
            return base.copy()
        # A label at call n is available only after dwell consecutive true
        # transitions ending at n.  No label is fabricated for call zero.
        for n in range(dwell, calls):
            out[n] = np.all(base[n - dwell + 1 : n + 1], axis=0)
        return out

    stable = dwell_labels(stable)
    confident = dwell_labels(confident)
    both = stable & confident

    def first_call(labels: Array) -> Array:
        result = np.full(positions, -1, dtype=np.int32)
        seen = np.any(labels, axis=0)
        if np.any(seen):
            result[seen] = np.argmax(labels[:, seen], axis=0).astype(np.int32)
        return result

    return StableConfidenceAttribution(
        stable=stable,
        confident=confident,
        stable_and_confident=both,
        first_stable_call=first_call(stable),
        first_confident_call=first_call(confident),
        terminal_stable=stable[-1].copy(),
        terminal_confident=confident[-1].copy(),
    )


def attention_value_key_response(
    probabilities: Array,
    keys: Array,
    values: Array,
    *,
    scale: float = 1.0,
    directions: Array | None = None,
) -> dict[str, Array]:
    """Return the fixed-support value-key covariance response.

    ``probabilities`` has shape ``[queries, keys]`` and each row must sum to
    one. ``keys`` has shape ``[keys, dimension]`` and ``values`` has shape
    ``[keys, value_dimension]``.  The returned ``jacobian`` has shape
    ``[queries, value_dimension, dimension]`` and equals
    ``scale * Cov_p(v, k)``. If ``directions`` has shape ``[probes, dimension]``,
    ``directional`` contains the corresponding matrix products.
    """

    p = np.asarray(probabilities, dtype=float)
    k = np.asarray(keys, dtype=float)
    v = np.asarray(values, dtype=float)
    if p.ndim != 2 or k.ndim != 2 or v.ndim != 2 or p.shape[1] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("probabilities, keys and values have incompatible shapes")
    if not isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not np.isfinite(p).all() or not np.isfinite(k).all() or not np.isfinite(v).all():
        raise ValueError("attention inputs must be finite")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("each probability row must be nonnegative and sum to one")

    mean_k = p @ k
    mean_v = p @ v
    cross = np.einsum("qj,je,jd->qed", p, v, k)
    jacobian = float(scale) * (cross - mean_v[:, :, None] * mean_k[:, None, :])
    result: dict[str, Array] = {
        "mean_key": mean_k,
        "mean_value": mean_v,
        "jacobian": jacobian,
        "gain_frobenius": np.linalg.norm(jacobian, axis=(1, 2)),
    }
    if directions is not None:
        u = np.asarray(directions, dtype=float)
        if u.ndim != 2 or u.shape[1] != k.shape[1] or not np.isfinite(u).all():
            raise ValueError("directions must have shape [probes, key_dimension] and be finite")
        result["directional"] = np.einsum("qed,pd->qpe", jacobian, u)
        result["directional_gain"] = np.linalg.norm(result["directional"], axis=-1)
    return result


def deleted_mass_output(
    retained_output: Array,
    deleted_output: Array,
    deleted_mass: Array | float,
) -> Array:
    """Compute the exact same-state sparse-minus-dense output perturbation."""

    s = np.asarray(retained_output, dtype=float)
    d = np.asarray(deleted_output, dtype=float)
    alpha = np.asarray(deleted_mass, dtype=float)
    if s.shape != d.shape:
        raise ValueError("retained_output and deleted_output must have equal shape")
    if np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError("deleted_mass must lie in [0, 1)")
    return alpha[..., None] * (s - d)


def attention_state_directional_response(
    probabilities: Array, queries: Array, keys: Array, values: Array,
    query_delta: Array, key_delta: Array, value_delta: Array, *, scale: float = 1.0,
) -> dict[str, Array]:
    """Exact local attention differential including moving Q, K and V.

    This is a reference diagnostic, not a next-query predictor or a selector.
    It assumes fixed support for the differential; changing support requires
    direct masked recomputation. Shape conventions are Q:[rows,d], K:[keys,d],
    V:[keys,dv], p:[rows,keys], with deltas matching the corresponding state.
    """
    p, q, k, v, dq, dk, dv = [np.asarray(x, dtype=float) for x in
                              (probabilities, queries, keys, values, query_delta, key_delta, value_delta)]
    attention_value_key_response(p, k, v, scale=scale)
    if q.ndim != 2 or q.shape != dq.shape or q.shape != (p.shape[0], k.shape[1]):
        raise ValueError("query and query_delta shapes are incompatible")
    if dk.shape != k.shape or dv.shape != v.shape or not all(
            np.isfinite(x).all() for x in (q, dq, dk, dv)):
        raise ValueError("state deltas must match finite Q/K/V")
    mean_v = p @ v

    def covariance_product(ds):
        weighted = p * ds
        return weighted @ v - weighted.sum(-1, keepdims=True) * mean_v

    query_response = covariance_product(float(scale) * dq @ k.T)
    key_response = covariance_product(float(scale) * q @ dk.T)
    value_response = p @ dv
    return dict(query=query_response, key=key_response, value=value_response,
                total=query_response + key_response + value_response)


def canvas_mass_amplification(deleted_mass: Array | float) -> Array:
    """Return the current-canvas mass multiplier after prefix deletion."""

    alpha = np.asarray(deleted_mass, dtype=float)
    if np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError("deleted_mass must lie in [0, 1)")
    return 1.0 / (1.0 - alpha)


def retained_mass_lower_bound(
    anchor_retained_mass: Array | float,
    retained_logit_radius: Array | float,
    deleted_logit_radius: Array | float,
) -> Array:
    """Lower-bound retained prefix mass after bounded logit movement."""

    u = np.asarray(anchor_retained_mass, dtype=float)
    r_s = np.asarray(retained_logit_radius, dtype=float)
    r_d = np.asarray(deleted_logit_radius, dtype=float)
    u, r_s, r_d = np.broadcast_arrays(u, r_s, r_d)
    if np.any(u <= 0) or np.any(u >= 1) or np.any(r_s < 0) or np.any(r_d < 0):
        raise ValueError("anchor mass must be in (0,1), radii must be nonnegative")
    # Log-space evaluation avoids overflow for conservative radii.
    logit = np.log(u) - np.log1p(-u) - r_s - r_d
    out = np.empty_like(logit)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_logit = np.exp(logit[~positive])
    out[~positive] = exp_logit / (1.0 + exp_logit)
    return out


def query_key_radius(query_delta: Array, key_norm_max: Array | float, scale: float) -> Array:
    """Conservative scaled-dot-product logit radius for a support block."""

    dq = np.asarray(query_delta, dtype=float)
    kn = np.asarray(key_norm_max, dtype=float)
    if not np.isfinite(dq).all() or np.any(kn < 0) or not isfinite(float(scale)) or scale <= 0:
        raise ValueError("query delta/key norm/scale are invalid")
    return float(scale) * np.linalg.norm(dq, axis=-1) * kn


def margin_safe(margin: Array | float, logit_delta_inf: Array | float) -> Array:
    """Sufficient condition for preserving a token argmax at the next call."""

    m = np.asarray(margin, dtype=float)
    d = np.asarray(logit_delta_inf, dtype=float)
    if np.any(m < 0) or np.any(d < 0) or not np.isfinite(m).all() or not np.isfinite(d).all():
        raise ValueError("margin and logit_delta_inf must be finite and nonnegative")
    return d < (m / 2.0)


def correction_break_delta(error_fraction: Array | float, delta_fix: Array | float, delta_break: Array | float) -> Array:
    """Sparse-minus-reference one-step error-flow change.

    If ``e`` is the current wrong fraction, ``delta_fix`` is sparse minus dense
    correction probability, and ``delta_break`` is sparse minus dense break
    probability, the result is ``-e*delta_fix + (1-e)*delta_break``.
    """

    e = np.asarray(error_fraction, dtype=float)
    df = np.asarray(delta_fix, dtype=float)
    db = np.asarray(delta_break, dtype=float)
    e, df, db = np.broadcast_arrays(e, df, db)
    if np.any(e < 0) or np.any(e > 1) or not np.isfinite(e).all() or not np.isfinite(df).all() or not np.isfinite(db).all():
        raise ValueError("error fraction must be in [0,1] and all inputs finite")
    return -e * df + (1.0 - e) * db


def summarize_stop_bottleneck(
    stability_satisfied: Array,
    entropy_satisfied: Array,
    cap_reached: Array,
) -> dict[str, int]:
    """Count native stop strata from one boolean row per canvas."""

    stability = np.asarray(stability_satisfied, dtype=bool)
    entropy = np.asarray(entropy_satisfied, dtype=bool)
    cap = np.asarray(cap_reached, dtype=bool)
    if stability.shape != entropy.shape or stability.shape != cap.shape:
        raise ValueError("stop predicate arrays must have equal shape")
    return {
        "stability_only": int(np.sum(stability & ~entropy & ~cap)),
        "entropy_only": int(np.sum(~stability & entropy & ~cap)),
        "both": int(np.sum(stability & entropy & ~cap)),
        "cap": int(np.sum(cap)),
        "none": int(np.sum(~stability & ~entropy & ~cap)),
    }


def jsonable_summary(value: Any) -> Any:
    """Convert diagnostic scalars/arrays to JSON-safe values for receipts."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_summary(v) for v in value]
    return value

