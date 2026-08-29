"""Threshold-family fitting, held-out density gates, and model selection."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import optimize, stats


FAMILIES = ("gaussian", "empirical_cdf", "jones_faddy_skew_t", "gaussian_mixture_2")
SCOPES = ("model", "model_attention", "model_corpus_attention")
REQUIRED_DESIGN = {
    ("diffusion_gemma", "ruler8k"),
    ("diffusion_gemma", "math500"),
    ("fast_dllm_v2", "ruler8k"),
    ("fast_dllm_v2", "math500"),
}
SCOPE_COLUMNS = {
    "model": ("adapter",),
    "model_attention": ("adapter", "attention_type"),
    "model_corpus_attention": ("adapter", "corpus", "attention_type"),
}

# The on-disk shards deliberately retain one compact row for every eligible
# call/layer/head/query-block.  A full Fast-dLLM-v2 collection contains tens of
# millions of such rows, so expanding them into Python dictionaries is not a
# viable fitting representation.  We instead stream every row into a fine
# weighted reservoir histogram and retain one aggregate record per
# prompt/attention-type.  The 1/512-wide bins are substantially finer than the
# collector's density histogram and keep fitting deterministic and bounded.
RESERVOIR_AGGREGATION_EDGES = np.linspace(-16.0, 16.0, 16_385, dtype=np.float64)
SHARD_REDUCTION_CHUNK_ROWS = 8_192


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        raise ValueError("weighted quantile has no finite positive-weight samples")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(probability, cumulative, values, left=values[0], right=values[-1]))


@dataclass(frozen=True)
class GaussianMixture1D:
    weights: tuple[float, float]
    means: tuple[float, float]
    sigmas: tuple[float, float]

    def cdf(self, value: float | np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        return sum(
            weight * stats.norm.cdf((value - mean) / sigma)
            for weight, mean, sigma in zip(self.weights, self.means, self.sigmas)
        )

    def quantile(self, probability: float) -> float:
        lower = min(self.means) - 12 * max(self.sigmas)
        upper = max(self.means) + 12 * max(self.sigmas)
        return float(optimize.brentq(lambda x: float(self.cdf(x)) - probability, lower, upper))

    def as_dict(self) -> dict[str, list[float]]:
        return {"weights": list(self.weights), "means": list(self.means), "sigmas": list(self.sigmas)}


def fit_gaussian_mixture(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    variance_floor: float = 1.0e-4,
    max_iterations: int = 300,
) -> GaussianMixture1D:
    """Deterministic weighted two-component EM with fixed multi-starts."""
    x = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x)
    x = x[valid]
    if weights is None:
        sample_weight = np.ones(len(x), dtype=np.float64)
    else:
        sample_weight = np.asarray(weights, dtype=np.float64)[valid]
    sample_weight /= sample_weight.sum()
    starts = (
        (weighted_quantile(x, sample_weight, 0.25), weighted_quantile(x, sample_weight, 0.75)),
        (-1.0, 1.0),
        (weighted_quantile(x, sample_weight, 0.10), weighted_quantile(x, sample_weight, 0.90)),
    )
    best: tuple[float, GaussianMixture1D] | None = None
    base_sigma = max(float(np.sqrt(np.sum(sample_weight * (x - np.sum(sample_weight * x)) ** 2))), math.sqrt(variance_floor))
    for initial_means in starts:
        mixing = np.asarray((0.5, 0.5), dtype=np.float64)
        means = np.asarray(initial_means, dtype=np.float64)
        sigmas = np.asarray((base_sigma, base_sigma), dtype=np.float64)
        previous = -np.inf
        for _ in range(max_iterations):
            likelihood = np.column_stack([
                mixing[k] * stats.norm.pdf(x, means[k], sigmas[k]) for k in range(2)
            ]).clip(min=1.0e-300)
            normalizer = likelihood.sum(axis=1)
            responsibilities = likelihood / normalizer[:, None]
            effective = (sample_weight[:, None] * responsibilities).sum(axis=0).clip(min=1.0e-8)
            mixing = effective / effective.sum()
            means = (sample_weight[:, None] * responsibilities * x[:, None]).sum(axis=0) / effective
            variances = (
                sample_weight[:, None] * responsibilities * (x[:, None] - means) ** 2
            ).sum(axis=0) / effective
            sigmas = np.sqrt(np.maximum(variances, variance_floor))
            objective = float(np.sum(sample_weight * np.log(normalizer)))
            if abs(objective - previous) < 1.0e-10:
                break
            previous = objective
        order = np.argsort(means)
        model = GaussianMixture1D(
            tuple(float(v) for v in mixing[order]),
            tuple(float(v) for v in means[order]),
            tuple(float(v) for v in sigmas[order]),
        )
        if best is None or previous > best[0]:
            best = previous, model
    assert best is not None
    return best[1]


def _weighted_fit_sample(values: np.ndarray, weights: np.ndarray, size: int = 10_000) -> np.ndarray:
    """Deterministic systematic weighted sample for SciPy's unweighted MLE."""
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order]
    w = w / w.sum()
    positions = (np.arange(size, dtype=np.float64) + 0.5) / size
    return x[np.searchsorted(np.cumsum(w), positions, side="left")]


def fit_family(family: str, values: np.ndarray, weights: np.ndarray, densities: Iterable[float]) -> dict[str, Any]:
    densities = tuple(float(value) for value in densities)
    if family == "gaussian":
        return {"parameters": {}, "betas": {str(d): float(stats.norm.ppf(1.0 - d)) for d in densities}}
    if family == "empirical_cdf":
        return {"parameters": {}, "betas": {str(d): weighted_quantile(values, weights, 1.0 - d) for d in densities}}
    if family == "jones_faddy_skew_t":
        sample = _weighted_fit_sample(values, weights)
        a, b, loc, scale = stats.jf_skew_t.fit(sample)
        parameters = {"a": float(a), "b": float(b), "loc": float(loc), "scale": float(scale)}
        return {
            "parameters": parameters,
            "betas": {str(d): float(stats.jf_skew_t.ppf(1.0 - d, a, b, loc=loc, scale=scale)) for d in densities},
        }
    if family == "gaussian_mixture_2":
        model = fit_gaussian_mixture(values, weights)
        return {"parameters": model.as_dict(), "betas": {str(d): model.quantile(1.0 - d) for d in densities}}
    raise ValueError(f"unknown family: {family}")


def _weighted_histogram_from_reservoir(
    reservoir: np.ndarray,
    row_weights: np.ndarray,
    *,
    edges: np.ndarray = RESERVOIR_AGGREGATION_EDGES,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce row-balanced reservoirs without materializing repeated weights."""
    histogram = np.zeros(len(edges) - 1, dtype=np.float64)
    for start in range(0, len(reservoir), SHARD_REDUCTION_CHUNK_ROWS):
        values = reservoir[start : start + SHARD_REDUCTION_CHUNK_ROWS].astype(np.float64, copy=False)
        weights = row_weights[start : start + SHARD_REDUCTION_CHUNK_ROWS]
        valid = np.isfinite(values)
        counts = valid.sum(axis=1)
        per_value = np.divide(weights, counts, out=np.zeros_like(weights), where=counts > 0)
        clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
        histogram += np.histogram(
            clipped[valid], bins=edges, weights=np.broadcast_to(per_value[:, None], values.shape)[valid]
        )[0]
    occupied = histogram > 0
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers[occupied], histogram[occupied]


def _aggregate_canvas_histogram(
    histogram: np.ndarray,
    underflow: np.ndarray,
    overflow: np.ndarray,
    canvas_tiles: np.ndarray,
    row_weights: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Return a weighted marginal tile distribution with total mass one."""
    total_weight = float(row_weights.sum())
    if total_weight <= 0:
        raise ValueError("aggregate has no positive hierarchical row weight")
    mass = np.zeros(histogram.shape[1], dtype=np.float64)
    underflow_mass = 0.0
    overflow_mass = 0.0
    for start in range(0, len(histogram), SHARD_REDUCTION_CHUNK_ROWS):
        stop = start + SHARD_REDUCTION_CHUNK_ROWS
        counts = canvas_tiles[start:stop].astype(np.float64, copy=False)
        factors = np.divide(
            row_weights[start:stop], counts, out=np.zeros_like(row_weights[start:stop]), where=counts > 0
        )
        mass += factors @ histogram[start:stop].astype(np.float64, copy=False)
        underflow_mass += float(factors @ underflow[start:stop])
        overflow_mass += float(factors @ overflow[start:stop])
    return mass / total_weight, underflow_mass / total_weight, overflow_mass / total_weight, total_weight


def _load_shards(root: Path) -> list[dict[str, Any]]:
    """Load shards into bounded prompt/attention aggregates.

    This preserves the exact hierarchical weighted canvas histogram used by
    the density gates.  Reservoir values are represented by a fine weighted
    histogram that includes every finite per-row reservoir observation.
    """
    records: list[dict[str, Any]] = []
    for path in sorted((root / "shards").glob("*/*/*.npz")):
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path) as payload:
            attention = payload["attention_type"]
            prompt_weight = payload["hierarchical_row_weight"].astype(np.float64, copy=False)
            canvas_tiles = payload["canvas_tiles"]
            canvas_histogram = payload["canvas_histogram"]
            canvas_underflow = payload["canvas_underflow"]
            canvas_overflow = payload["canvas_overflow"]
            canvas_reservoir = payload["canvas_reservoir"]
            row_max_z = payload["row_max_z"]
            histogram_edges = payload["histogram_edges"].astype(np.float64, copy=False)
            for attention_type in np.unique(attention):
                indices = np.flatnonzero(attention == attention_type)
                weights = prompt_weight[indices]
                values, value_weights = _weighted_histogram_from_reservoir(canvas_reservoir[indices], weights)
                histogram, underflow, overflow, total_weight = _aggregate_canvas_histogram(
                    canvas_histogram[indices], canvas_underflow[indices], canvas_overflow[indices],
                    canvas_tiles[indices], weights,
                )
                finite_maxima = np.isfinite(row_max_z[indices]) & (weights > 0)
                maxima = row_max_z[indices][finite_maxima].astype(np.float64, copy=False)
                maximum_weights = weights[finite_maxima]
                if len(maxima):
                    maxima = _weighted_fit_sample(maxima, maximum_weights, size=min(2_048, len(maxima)))
                records.append({
                    "adapter": sidecar["adapter"],
                    "corpus": str(payload["corpus"][indices[0]]),
                    "task": str(payload["task"][indices[0]]),
                    "split": str(payload["split"][indices[0]]),
                    "request_id": str(payload["request_id"][indices[0]]),
                    "attention_type": str(attention_type),
                    "canvas_tiles": 1,
                    "supported_row_count": int(np.count_nonzero(canvas_tiles[indices] >= 4)),
                    "row_weight": total_weight,
                    "reservoir": values,
                    "reservoir_weights": value_weights / value_weights.sum(),
                    "histogram_edges": histogram_edges,
                    "histogram": histogram,
                    "underflow": underflow,
                    "overflow": overflow,
                    "histogram_is_mass": True,
                    "row_max_z": float(np.average(row_max_z[indices][finite_maxima], weights=maximum_weights)) if len(maxima) else math.nan,
                    "row_max_reservoir": maxima,
                })
    if not records:
        raise FileNotFoundError(f"no prompt shards under {root / 'shards'}")
    return records


def _scope_key(record: Mapping[str, Any], scope: str) -> tuple[str, ...]:
    return tuple(str(record[name]) for name in SCOPE_COLUMNS[scope])


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def _fit_values(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if rows and all("reservoir_weights" in row for row in rows):
        histogram = np.zeros(len(RESERVOIR_AGGREGATION_EDGES) - 1, dtype=np.float64)
        total = sum(row["row_weight"] for row in rows if len(row["reservoir"]))
        width = RESERVOIR_AGGREGATION_EDGES[1] - RESERVOIR_AGGREGATION_EDGES[0]
        for row in rows:
            if not len(row["reservoir"]):
                continue
            indices = np.rint(
                (np.asarray(row["reservoir"]) - (RESERVOIR_AGGREGATION_EDGES[0] + width * 0.5)) / width
            ).astype(np.int64)
            np.add.at(
                histogram, indices,
                np.asarray(row["reservoir_weights"], dtype=np.float64) * row["row_weight"] / total,
            )
        occupied = histogram > 0
        centers = (RESERVOIR_AGGREGATION_EDGES[:-1] + RESERVOIR_AGGREGATION_EDGES[1:]) * 0.5
        return centers[occupied], histogram[occupied]
    values, weights = [], []
    total = sum(row["row_weight"] for row in rows if len(row["reservoir"]))
    for row in rows:
        if not len(row["reservoir"]):
            continue
        values.append(np.asarray(row["reservoir"], dtype=np.float64))
        if "reservoir_weights" in row:
            weights.append(np.asarray(row["reservoir_weights"], dtype=np.float64) * row["row_weight"] / total)
        else:
            weights.append(np.full(len(row["reservoir"]), row["row_weight"] / len(row["reservoir"]) / total))
    return np.concatenate(values), np.concatenate(weights)


def histogram_survival(row: Mapping[str, Any], beta: float) -> float:
    """Estimate a row survival probability from exact fine-bin counts."""
    count = int(row["canvas_tiles"])
    if count == 0:
        return math.nan
    edges = row["histogram_edges"]
    hist = row["histogram"]
    retained = float(row["overflow"])
    if beta < edges[0]:
        retained += hist.sum()
        # Underflow locations are unknown; linearly include none/any only at boundary.
        retained += float(row["underflow"])
    elif beta < edges[-1]:
        index = min(len(hist) - 1, int(np.searchsorted(edges, beta, side="right") - 1))
        retained += hist[index + 1 :].sum()
        fraction = (edges[index + 1] - beta) / (edges[index + 1] - edges[index])
        retained += hist[index] * min(1.0, max(0.0, fraction))
    return retained if row.get("histogram_is_mass", False) else retained / count


def _hierarchical_mean(rows: list[dict[str, Any]], values: np.ndarray) -> float:
    weights = np.asarray([row["row_weight"] for row in rows], dtype=np.float64)
    valid = np.isfinite(values) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid]))


def _prompt_bootstrap_error(
    rows: list[dict[str, Any]], beta: float, target: float, *, seed: int, repeats: int
) -> tuple[float, float, float]:
    prompts = sorted({row["request_id"] for row in rows})
    prompt_values = []
    for prompt in prompts:
        selected = [row for row in rows if row["request_id"] == prompt]
        realized = np.asarray([histogram_survival(row, beta) for row in selected])
        prompt_values.append(_hierarchical_mean(selected, realized))
    values = np.asarray(prompt_values)
    if not len(values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    errors = np.abs(samples - target)
    return float(values.mean()), float(np.quantile(errors, 0.025)), float(np.quantile(errors, 0.975))


def _prompt_density_values(rows: list[dict[str, Any]], beta: float) -> np.ndarray:
    values = []
    for prompt in sorted({row["request_id"] for row in rows}):
        selected = [row for row in rows if row["request_id"] == prompt]
        realized = np.asarray([histogram_survival(row, beta) for row in selected])
        values.append(_hierarchical_mean(selected, realized))
    return np.asarray(values, dtype=float)


def _bootstrap_drift_upper(
    calibration: list[dict[str, Any]], validation: list[dict[str, Any]], beta: float, *, seed: int, repeats: int
) -> float:
    left = _prompt_density_values(calibration, beta)
    right = _prompt_density_values(validation, beta)
    if not len(left) or not len(right):
        return math.inf
    rng = np.random.default_rng(seed)
    left_means = left[rng.integers(0, len(left), size=(repeats, len(left)))].mean(axis=1)
    right_means = right[rng.integers(0, len(right), size=(repeats, len(right)))].mean(axis=1)
    return float(np.quantile(np.abs(left_means - right_means), 0.95))


def _covered_groups(records: list[dict[str, Any]], scope: str) -> list[tuple[str, ...]]:
    return sorted({_scope_key(row, scope) for row in records if row["split"] == "calibration"})


def _lookup_fit(fits: Mapping[tuple[str, ...], dict[str, Any]], row: Mapping[str, Any], scope: str) -> dict[str, Any]:
    return fits[_scope_key(row, scope)]


def evaluate_candidate(
    records: list[dict[str, Any]], scope: str, family: str, densities: Iterable[float], *, bootstrap_repeats: int, seed: int
) -> tuple[dict[tuple[str, ...], dict[str, Any]], list[dict[str, Any]], bool]:
    fits: dict[tuple[str, ...], dict[str, Any]] = {}
    for group in _covered_groups(records, scope):
        calibration = [row for row in records if row["split"] == "calibration" and _scope_key(row, scope) == group]
        values, weights = _fit_values(calibration)
        fitted = fit_family(family, values, weights, densities)
        if not _all_finite(fitted):
            raise RuntimeError(f"{family} produced non-finite parameters")
        fits[group] = fitted
    evaluations: list[dict[str, Any]] = []
    passed = True
    validation_strata = sorted({(row["adapter"], row["corpus"], row["attention_type"]) for row in records if row["split"] == "validation"})
    for stratum in validation_strata:
        validation = [row for row in records if row["split"] == "validation" and (row["adapter"], row["corpus"], row["attention_type"]) == stratum]
        calibration = [row for row in records if row["split"] == "calibration" and (row["adapter"], row["corpus"], row["attention_type"]) == stratum]
        if not validation or _scope_key(validation[0], scope) not in fits:
            passed = False
            continue
        fit = _lookup_fit(fits, validation[0], scope)
        for density in densities:
            beta = float(fit["betas"][str(float(density))])
            realized = np.asarray([histogram_survival(row, beta) for row in validation])
            validation_mean = _hierarchical_mean(validation, realized)
            calibration_mean = _hierarchical_mean(
                calibration, np.asarray([histogram_survival(row, beta) for row in calibration])
            )
            bootstrap_mean, _, upper = _prompt_bootstrap_error(
                validation, beta, float(density), seed=seed, repeats=bootstrap_repeats
            )
            supported_rows = sum(int(row.get("supported_row_count", row["canvas_tiles"] >= 4)) for row in validation)
            max_supported_stratum_error = abs(validation_mean - float(density)) if supported_rows else 0.0
            mean_error = abs(validation_mean - float(density))
            drift = abs(validation_mean - calibration_mean)
            drift_upper = _bootstrap_drift_upper(
                calibration, validation, beta, seed=seed, repeats=bootstrap_repeats
            )
            gate = mean_error <= 0.02 and upper <= 0.03 and max_supported_stratum_error <= 0.05 and drift <= 0.02 and drift_upper <= 0.03
            passed &= gate
            evaluations.append({
                "scope": scope,
                "family": family,
                "adapter": stratum[0],
                "corpus": stratum[1],
                "attention_type": stratum[2],
                "target_density": float(density),
                "beta": beta,
                "calibration_density": calibration_mean,
                "validation_density": validation_mean,
                "bootstrap_prompt_mean": bootstrap_mean,
                "absolute_error": mean_error,
                "bootstrap_95_upper_absolute_error": upper,
                "calibration_validation_drift": drift,
                "calibration_validation_drift_bootstrap_95_upper": drift_upper,
                "supported_rows": supported_rows,
                "passed": bool(gate),
            })
    return fits, evaluations, bool(passed and evaluations)


def _json_fits(fits: Mapping[tuple[str, ...], dict[str, Any]]) -> dict[str, Any]:
    return {"|".join(key): value for key, value in sorted(fits.items())}


def fit_threshold_models(
    output_dir: str | Path,
    densities: Iterable[float] = (0.25, 0.50, 0.75),
    *,
    bootstrap_repeats: int = 2_000,
    seed: int = 20260822,
    require_complete_design: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    densities = tuple(float(value) for value in densities)
    records = _load_shards(output_dir)
    observed_design = {(row["adapter"], row["corpus"]) for row in records}
    if require_complete_design and observed_design != REQUIRED_DESIGN:
        missing = sorted(REQUIRED_DESIGN - observed_design)
        extra = sorted(observed_design - REQUIRED_DESIGN)
        raise RuntimeError(f"fit requires all four model/corpus collections; missing={missing}, extra={extra}")
    candidates = []
    selected: dict[str, Any] | None = None
    for scope in SCOPES:
        for family in FAMILIES:
            try:
                fits, evaluations, passed = evaluate_candidate(
                    records, scope, family, densities, bootstrap_repeats=bootstrap_repeats, seed=seed
                )
                error = None
            except Exception as exception:
                fits, evaluations, passed = {}, [], False
                error = f"{type(exception).__name__}: {exception}"
            candidate = {
                "scope": scope,
                "family": family,
                "passed": passed,
                "fits": _json_fits(fits),
                "evaluations": evaluations,
                "error": error,
            }
            candidates.append(candidate)
            if selected is None and passed:
                selected = candidate
    portability_pass = bool(selected and selected["scope"] in ("model", "model_attention"))
    if selected is None:
        empirical = [row for row in candidates if row["family"] == "empirical_cdf"]
        selected = min(
            empirical,
            key=lambda row: np.mean([value["absolute_error"] for value in row["evaluations"]]) if row["evaluations"] else math.inf,
            default=None,
        )
    maximum_parts = [
        np.asarray(row.get("row_max_reservoir", [row["row_max_z"]]), dtype=np.float64)
        for row in records if row["split"] == "calibration"
    ]
    maxima = np.concatenate(maximum_parts) if maximum_parts else np.asarray([], dtype=np.float64)
    maxima = maxima[np.isfinite(maxima)]
    tail_diagnostics: dict[str, Any] = {}
    if len(maxima) >= 8 and float(maxima.std()) > 1.0e-8:
        gev = stats.genextreme.fit(maxima)
        gumbel = stats.gumbel_r.fit(maxima)
        if all(np.isfinite((*gev, *gumbel))):
            tail_diagnostics = {
                "population": "per-row standardized canvas maxima only; never used for marginal routing",
                "gev": {"shape": float(gev[0]), "loc": float(gev[1]), "scale": float(gev[2])},
                "gumbel": {"loc": float(gumbel[0]), "scale": float(gumbel[1])},
            }
    calibration_ids = sorted({f"{row['adapter']}|{row['corpus']}|{row['request_id']}" for row in records if row["split"] == "calibration"})
    validation_ids = sorted({f"{row['adapter']}|{row['corpus']}|{row['request_id']}" for row in records if row["split"] == "validation"})
    result = {
        "schema_version": 1,
        "observed_design": sorted([list(value) for value in observed_design]),
        "complete_design": observed_design == REQUIRED_DESIGN,
        "densities": densities,
        "selection_rule": "least granular passing scope, then Gaussian, empirical CDF, Jones-Faddy skew-t, GMM",
        "simple_profiling_passed": portability_pass,
        "decision": "pass" if portability_pass else "no-go",
        "selected": selected,
        "candidates": candidates,
        "maximum_tail_diagnostics": tail_diagnostics,
        "fit_prompt_ids": calibration_ids,
        "validation_prompt_ids": validation_ids,
        "calibration_only_audit": not bool(set(calibration_ids) & set(validation_ids)),
    }
    fit_dir = output_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "threshold_models.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    evaluation_rows = [value for candidate in candidates for value in candidate["evaluations"]]
    if evaluation_rows:
        with (fit_dir / "heldout_density_errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0]))
            writer.writeheader()
            writer.writerows(evaluation_rows)
    return result
