"""Canonical plots derived exclusively from final compact shards."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .fitting import _load_shards, histogram_survival


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scatter_metric(records, field: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = {}
    for row in records:
        groups.setdefault((row["adapter"], row["corpus"], row["attention_type"]), []).append(row[field])
    for name, values in groups.items():
        ax.plot(np.sort(np.asarray(values, dtype=float)), label="/".join(name), alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Hierarchical row order")
    ax.set_ylabel(field.replace("_", " "))
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def _record_fit_weights(row) -> np.ndarray:
    if "reservoir_weights" in row:
        return np.asarray(row["reservoir_weights"], dtype=float) * float(row["row_weight"])
    return np.full(len(row["reservoir"]), float(row["row_weight"]) / max(len(row["reservoir"]), 1))


def _stream_region_histograms(output_dir: Path, edges: np.ndarray) -> dict[str, np.ndarray]:
    result = {"prefix": np.zeros(len(edges) - 1), "canvas": np.zeros(len(edges) - 1)}
    for shard in sorted((output_dir / "shards").glob("*/*/*.npz")):
        with np.load(shard) as payload:
            row_weights = payload["hierarchical_row_weight"].astype(float, copy=False)
            for label, field in (("prefix", "prefix_reservoir"), ("canvas", "canvas_reservoir")):
                reservoir = payload[field]
                for start in range(0, len(reservoir), 8_192):
                    values = reservoir[start : start + 8_192].astype(float, copy=False)
                    valid = np.isfinite(values)
                    counts = valid.sum(axis=1)
                    weights = np.divide(
                        row_weights[start : start + 8_192], counts,
                        out=np.zeros_like(row_weights[start : start + 8_192]), where=counts > 0,
                    )
                    result[label] += np.histogram(
                        values[valid], bins=edges,
                        weights=np.broadcast_to(weights[:, None], values.shape)[valid],
                    )[0]
    return result


def _stream_heterogeneity(output_dir: Path, left: str, right: str) -> tuple[np.ndarray, list[int], list[int]]:
    sums: dict[tuple[int, int], float] = {}
    weights: dict[tuple[int, int], float] = {}
    for shard in sorted((output_dir / "shards").glob("*/*/*.npz")):
        with np.load(shard) as payload:
            a = payload[left].astype(np.int64, copy=False)
            b = payload[right].astype(np.int64, copy=False)
            values = payload["row_max_z"].astype(float, copy=False)
            row_weights = payload["hierarchical_row_weight"].astype(float, copy=False)
            valid = np.isfinite(values) & np.isfinite(row_weights) & (row_weights > 0)
            if not valid.any():
                continue
            pairs = np.column_stack((a[valid], b[valid]))
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            weighted = np.bincount(inverse, weights=values[valid] * row_weights[valid])
            mass = np.bincount(inverse, weights=row_weights[valid])
            for pair, value, weight in zip(unique, weighted, mass):
                key = (int(pair[0]), int(pair[1]))
                sums[key] = sums.get(key, 0.0) + float(value)
                weights[key] = weights.get(key, 0.0) + float(weight)
    left_values = sorted({key[0] for key in sums})
    right_values = sorted({key[1] for key in sums})
    matrix = np.full((len(left_values), len(right_values)), np.nan)
    left_index = {value: index for index, value in enumerate(left_values)}
    right_index = {value: index for index, value in enumerate(right_values)}
    for key, value in sums.items():
        matrix[left_index[key[0]], right_index[key[1]]] = value / weights[key]
    return matrix, left_values, right_values


def _stream_axis_convergence(
    output_dir: Path, densities: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Cumulative hierarchical means as steps, layers, or heads are added.

    This deliberately uses the precomputed per-row Gaussian-tail densities and
    hierarchical row weights, so the diagnostic has the same population
    semantics as prompt convergence without materializing tile observations.
    """
    axes = ("denoising_call", "layer", "head")
    sums = {axis: {density: {} for density in densities} for axis in axes}
    masses = {axis: {density: {} for density in densities} for axis in axes}
    for shard in sorted((output_dir / "shards").glob("*/*/*.npz")):
        with np.load(shard) as payload:
            weights = payload["hierarchical_row_weight"].astype(float, copy=False)
            coordinates = {
                axis: payload[axis].astype(np.int64, copy=False) for axis in axes
            }
            for density in densities:
                field = f"gaussian_density_{density:g}"
                if field not in payload.files:
                    continue
                values = payload[field].astype(float, copy=False)
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                for axis in axes:
                    coordinate = coordinates[axis][valid]
                    unique, inverse = np.unique(coordinate, return_inverse=True)
                    weighted = np.bincount(inverse, weights=values[valid] * weights[valid])
                    mass = np.bincount(inverse, weights=weights[valid])
                    for unit, value, weight in zip(unique, weighted, mass):
                        unit = int(unit)
                        sums[axis][density][unit] = sums[axis][density].get(unit, 0.0) + float(value)
                        masses[axis][density][unit] = masses[axis][density].get(unit, 0.0) + float(weight)
    result = {axis: {} for axis in axes}
    for axis in axes:
        for density in densities:
            units = np.asarray(sorted(sums[axis][density]), dtype=int)
            if not len(units):
                continue
            weighted = np.asarray([sums[axis][density][int(unit)] for unit in units])
            mass = np.asarray([masses[axis][density][int(unit)] for unit in units])
            result[axis][density] = (units, np.cumsum(weighted) / np.cumsum(mass))
    return result


def _row_survival(
    edges: np.ndarray,
    histogram_all: np.ndarray,
    underflow_all: np.ndarray,
    overflow_all: np.ndarray,
    canvas_tiles_all: np.ndarray,
    indices: np.ndarray,
    beta: float,
) -> np.ndarray:
    histogram = histogram_all[indices]
    retained = overflow_all[indices].astype(float)
    if beta < edges[0]:
        retained += histogram.sum(axis=1) + underflow_all[indices]
    elif beta < edges[-1]:
        index = min(len(edges) - 2, int(np.searchsorted(edges, beta, side="right") - 1))
        retained += histogram[:, index + 1 :].sum(axis=1)
        fraction = (edges[index + 1] - beta) / (edges[index + 1] - edges[index])
        retained += histogram[:, index] * min(1.0, max(0.0, fraction))
    counts = canvas_tiles_all[indices].astype(float)
    return np.divide(retained, counts, out=np.full_like(retained, np.nan), where=counts > 0)


def _selected_key(selected, adapter: str, corpus: str, attention_type: str) -> str:
    if selected["scope"] == "model":
        return adapter
    if selected["scope"] == "model_attention":
        return f"{adapter}|{attention_type}"
    return f"{adapter}|{corpus}|{attention_type}"


def _stream_row_density_bands(output_dir: Path, selected, densities) -> dict[float, tuple[float, float, float]]:
    bins = np.linspace(0.0, 1.0, 1_002)
    histograms = {float(density): np.zeros(len(bins) - 1, dtype=np.int64) for density in densities}
    sums = {float(density): 0.0 for density in densities}
    counts = {float(density): 0 for density in densities}
    for shard in sorted((output_dir / "shards").glob("*/*/*.npz")):
        sidecar = json.loads(shard.with_suffix(".json").read_text())
        with np.load(shard) as payload:
            adapter = sidecar["adapter"]
            corpus = str(payload["corpus"][0])
            attention = payload["attention_type"]
            edges = payload["histogram_edges"].astype(float, copy=False)
            histogram = payload["canvas_histogram"]
            underflow = payload["canvas_underflow"]
            overflow = payload["canvas_overflow"]
            canvas_tiles = payload["canvas_tiles"]
            for attention_type in np.unique(attention):
                all_indices = np.flatnonzero(attention == attention_type)
                key = _selected_key(selected, adapter, corpus, str(attention_type))
                if key not in selected["fits"]:
                    continue
                for start in range(0, len(all_indices), 8_192):
                    indices = all_indices[start : start + 8_192]
                    for density in densities:
                        density = float(density)
                        beta = selected["fits"][key]["betas"][str(density)]
                        realized = _row_survival(
                            edges, histogram, underflow, overflow, canvas_tiles, indices, beta
                        )
                        realized = realized[np.isfinite(realized)]
                        histograms[density] += np.histogram(realized, bins=bins)[0]
                        sums[density] += float(realized.sum())
                        counts[density] += len(realized)
    result = {}
    centers = (bins[:-1] + bins[1:]) * 0.5
    for density in map(float, densities):
        cumulative = np.cumsum(histograms[density])
        if not counts[density]:
            continue
        p10 = centers[np.searchsorted(cumulative, 0.1 * cumulative[-1])]
        p90 = centers[min(len(centers) - 1, np.searchsorted(cumulative, 0.9 * cumulative[-1]))]
        result[density] = (sums[density] / counts[density], float(p10), float(p90))
    return result


def generate_plots(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    records = _load_shards(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # 1. Reservoir-based standardized marginals, corpus/model/type separated.
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    all_values, all_weights = [], []
    for key in sorted({(r["adapter"], r["corpus"], r["attention_type"]) for r in records}):
        selected = [r for r in records if (r["adapter"], r["corpus"], r["attention_type"]) == key]
        values = np.concatenate([r["reservoir"] for r in selected])
        weights = np.concatenate([_record_fit_weights(r) for r in selected])
        ax.hist(values, weights=weights, bins=100, range=(-5, 5), density=True, histtype="step", label="/".join(key))
        all_values.append(values); all_weights.append(weights)
    x = np.linspace(-5, 5, 500)
    ax.plot(x, stats.norm.pdf(x), "k--", label="N(0,1)")
    ax.set(xlabel="row-standardized Sol-Attn proxy z", ylabel="weighted density", title="Canvas proxy distributions and Gaussian overlay")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    inset = ax.inset_axes([0.62, 0.50, 0.34, 0.30])
    tail_values = np.concatenate(all_values); tail_weights = np.concatenate(all_weights)
    order = np.argsort(tail_values); cumulative = np.cumsum(tail_weights[order]); cumulative /= cumulative[-1]
    inset.semilogy(tail_values[order], np.maximum(1.0 - cumulative, 1.0e-7))
    inset.set(title="untruncated upper tail", xlabel="z", ylabel="survival")
    path = plot_dir / "01_weighted_standardized_distributions.png"; _save(fig, path); paths.append(path)

    # 2. Prefix versus canvas diagnostic (prefix values are standardized by canvas rows).
    region_edges = np.linspace(-8, 8, 101)
    region_histograms = _stream_region_histograms(output_dir, region_edges)
    region_centers = (region_edges[:-1] + region_edges[1:]) * 0.5
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, histogram in region_histograms.items():
        if histogram.sum():
            ax.stairs(histogram / (histogram.sum() * np.diff(region_edges)), region_edges, label=label)
    ax.set(title="Prefix versus canvas proxy distribution", xlabel="z relative to canvas row", ylabel="density"); ax.legend(frameon=False)
    path = plot_dir / "02_prefix_vs_canvas.png"; _save(fig, path); paths.append(path)

    raw_values = np.concatenate(all_values); raw_weights = np.concatenate(all_weights)
    order = np.argsort(raw_values); raw_values, raw_weights = raw_values[order], raw_weights[order]
    cumulative = np.cumsum(raw_weights); cumulative /= cumulative[-1]
    positions = (np.arange(min(10_000, len(raw_values))) + .5) / min(10_000, len(raw_values))
    values = raw_values[np.searchsorted(cumulative, positions)]
    probabilities = (np.arange(len(values)) + 0.5) / len(values)
    ordered = np.sort(values)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    theoretical = stats.norm.ppf(probabilities)
    axes[0].plot(theoretical, ordered, ".", ms=1); axes[0].plot([-4, 4], [-4, 4], "k--"); axes[0].set(title="Gaussian Q-Q", xlabel="normal quantile", ylabel="proxy quantile")
    pit = stats.norm.cdf(values)
    axes[1].hist(pit, bins=30, density=True); axes[1].axhline(1, color="k", ls="--"); axes[1].set(title="Gaussian PIT", xlabel="Phi(z)")
    path = plot_dir / "03_qq_and_pit.png"; _save(fig, path); paths.append(path)

    convergence_paths = sorted((output_dir / "collection").glob("*/*/convergence.json"))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax = axes[0, 0]
    for source in convergence_paths:
        data = json.loads(source.read_text())
        for density, attention_type in sorted({(row["target_density"], row.get("attention_type", "all")) for row in data["curves"]}):
            rows = [row for row in data["curves"] if row["target_density"] == density and row.get("attention_type", "all") == attention_type]
            ax.plot([r["prompts"] for r in rows], [r["mean_realized_density"] for r in rows], marker="o", label=f"{source.parts[-3]}/{source.parts[-2]}/{attention_type} rho={density:g}")
    ax.set(title="Convergence over prompts", xlabel="prompts", ylabel="realized Gaussian-tail density")
    if ax.lines:
        ax.legend(fontsize=6, frameon=False)
    axis_curves = _stream_axis_convergence(output_dir)
    for ax, (field, label) in zip(axes.flat[1:], (
        ("denoising_call", "denoising steps"), ("layer", "layers"), ("head", "heads")
    )):
        for density, (units, cumulative) in axis_curves[field].items():
            ax.plot(np.arange(1, len(units) + 1), cumulative, label=f"rho={density:g}")
            ax.axhline(density, color="k", lw=0.5, alpha=0.25)
        ax.set(title=f"Convergence over {label}", xlabel=f"{label} included", ylabel="cumulative weighted density")
        if ax.lines:
            ax.legend(fontsize=7, frameon=False)
    path = plot_dir / "04_prompt_convergence.png"; _save(fig, path); paths.append(path)

    for index, (left, right, title) in enumerate((
        ("denoising_call", "layer", "step_layer"),
        ("layer", "head", "layer_head"),
        ("denoising_call", "head", "step_head"),
    ), start=5):
        matrix, left_values, right_values = _stream_heterogeneity(output_dir, left, right)
        fig, ax = plt.subplots(figsize=(8.0, 4.4)); image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set(title=f"{title.replace('_', '/')} heterogeneity heatmap", xlabel=right.replace("denoising_call", "step"), ylabel=left.replace("denoising_call", "step")); fig.colorbar(image, ax=ax, label="mean row maximum z")
        path = plot_dir / f"{index:02d}_{title}_heterogeneity_heatmap.png"; _save(fig, path); paths.append(path)

    fit_path = output_dir / "fit" / "threshold_models.json"
    fit = json.loads(fit_path.read_text()) if fit_path.exists() else {"candidates": [], "selected": None}
    evaluations = [row for candidate in fit["candidates"] for row in candidate["evaluations"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for family in sorted({row["family"] for row in evaluations}):
        rows = [row for row in evaluations if row["family"] == family]
        ax.scatter([r["target_density"] for r in rows], [r["validation_density"] for r in rows], label=family, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--"); ax.set(xlim=(0.15, .85), ylim=(0.15, .85), title="Predicted versus realized held-out density", xlabel="target", ylabel="realized"); ax.legend(fontsize=7, frameon=False)
    path = plot_dir / "08_predicted_vs_realized_density.png"; _save(fig, path); paths.append(path)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    labels = [f"{r['scope']}\n{r['family']}" for r in evaluations]
    ax.bar(np.arange(len(evaluations)), [100*r["absolute_error"] for r in evaluations])
    ax.axhline(2, color="k", ls="--"); ax.set(title="Held-out calibration error by family and scope", ylabel="absolute error (percentage points)", xticks=np.arange(len(labels)), xticklabels=labels); ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    path = plot_dir / "09_heldout_error_by_family_scope.png"; _save(fig, path); paths.append(path)

    maximum_parts = [np.asarray(row.get("row_max_reservoir", [row["row_max_z"]]), dtype=float) for row in records]
    maxima = np.concatenate(maximum_parts); maxima = maxima[np.isfinite(maxima)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.hist(maxima, bins=60, density=True, alpha=.45, label="row maxima")
    if len(maxima) >= 8 and float(maxima.std()) > 1.0e-8:
        x = np.linspace(maxima.min(), maxima.max(), 400); gev = stats.genextreme.fit(maxima); gumbel = stats.gumbel_r.fit(maxima)
        ax.plot(x, stats.genextreme.pdf(x, *gev), label="GEV"); ax.plot(x, stats.gumbel_r.pdf(x, *gumbel), label="Gumbel")
    ax.set(title="Maximum-tail diagnostics (not marginal routing models)", xlabel="per-row maximum z", ylabel="density"); ax.legend(frameon=False)
    path = plot_dir / "10_gev_gumbel_maxima.png"; _save(fig, path); paths.append(path)

    selected = fit.get("selected")
    fig, ax = plt.subplots(figsize=(8.0, 4.2)); ax.axis("off")
    text = "No fitted selection" if not selected else f"Decision: {fit['decision']}\nScope: {selected['scope']}\nFamily: {selected['family']}\n" + "\n".join(f"{key}: {value['betas']}" for key, value in selected["fits"].items())
    ax.text(0.01, .98, text, va="top", family="monospace", fontsize=8)
    path = plot_dir / "11_threshold_tables_and_decision.png"; _save(fig, path); paths.append(path)

    # Per-row p10-p90 density bands for the selected table, streamed from raw rows.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if selected:
        for density, (mean, p10, p90) in _stream_row_density_bands(output_dir, selected, fit["densities"]).items():
            ax.errorbar([density], [mean], yerr=[[max(0.0, mean-p10)], [max(0.0, p90-mean)]], fmt="o")
    ax.plot([0, 1], [0, 1], "k--"); ax.set(xlim=(.15,.85), ylim=(0,1), title="Realized density with per-row p10-p90 bands", xlabel="target", ylabel="realized")
    path = plot_dir / "12_density_row_bands.png"; _save(fig, path); paths.append(path)
    math_path = output_dir / "math500" / "summary.json"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if math_path.exists():
        conditions = json.loads(math_path.read_text())["conditions"]
        ax.scatter([100 * row["logical_density"] for row in conditions], [100 * row["symbolic_pass_at_1"] for row in conditions])
        label_offsets = {
            "dense": (-4, 4),
            "oracle_rho50": (4, -11),
            "profiled_rho50": (4, 5),
            "oracle_rho75": (4, -11),
            "profiled_rho75": (4, 5),
        }
        for row in conditions:
            offset = label_offsets.get(row["condition"], (4, 4))
            ax.annotate(
                row["condition"],
                (100 * row["logical_density"], 100 * row["symbolic_pass_at_1"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6,
                ha="right" if row["condition"] == "dense" else "left",
            )
    ax.set(title="Math500 accuracy versus realized logical density", xlabel="realized logical density (%)", ylabel="symbolic pass@1 (%)")
    path = plot_dir / "13_math500_accuracy_vs_realized_density.png"; _save(fig, path); paths.append(path)
    return paths
