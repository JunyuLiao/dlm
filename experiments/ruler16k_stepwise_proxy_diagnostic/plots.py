"""Plots and per-stratum statistics for the stepwise diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Z_EDGES = np.linspace(-6.0, 6.0, 121)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select(data: dict[str, np.ndarray], *, attention_type: str, layer: int, head: int, query_block: int) -> np.ndarray:
    return (
        (data["attention_type"].astype(str) == attention_type)
        & (data["layer"] == layer)
        & (data["head"] == head)
        & (data["query_block"] == query_block)
    )


def _standardized(data: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reservoir = data["prefix_reservoir"][mask].astype(np.float64)
    mean = data["prefix_mean"][mask].astype(np.float64)
    std = data["prefix_std"][mask].astype(np.float64)
    calls = data["denoising_call"][mask].astype(int)
    valid = np.isfinite(reservoir) & (std[:, None] > 0)
    z = (reservoir - mean[:, None]) / std[:, None]
    return z, valid, calls


def generate_plots(output_dir: str | Path, *, attention_type: str = "global", layer: int = 0, head: int = 0, query_block: int = 0) -> dict[str, Any]:
    root = Path(output_dir)
    shard = next(root.glob("*.npz"))
    with np.load(shard) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
    selected = _select(data, attention_type=attention_type, layer=layer, head=head, query_block=query_block)
    if not selected.any():
        raise ValueError("selected attention type/layer/head/query block has no rows")
    z, valid, calls = _standardized(data, selected)
    unique_calls = sorted(set(calls.tolist()))
    histograms = []
    quantiles = []
    step_stats = []
    for call in unique_calls:
        m = calls == call
        values = z[m][valid[m]]
        histograms.append(np.histogram(values, bins=Z_EDGES, density=True)[0])
        quantiles.append(np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99]))
        step_rows = selected.copy()
        step_rows[selected] &= data["denoising_call"][selected] == call
        step_stats.append({
            "denoising_call": call,
            "rows": int(step_rows.sum()),
            "mask_ratio": float(np.nanmedian(data["mask_ratio"][step_rows])),
            "masked_tokens": int(np.nanmedian(data["masked_tokens"][step_rows])),
            "active_tokens": int(np.nanmedian(data["active_tokens"][step_rows])),
            "z_mean": float(values.mean()),
            "z_std": float(values.std()),
            "z_p01": float(np.quantile(values, .01)),
            "z_p05": float(np.quantile(values, .05)),
            "z_p50": float(np.quantile(values, .50)),
            "z_p95": float(np.quantile(values, .95)),
            "z_p99": float(np.quantile(values, .99)),
            "p_z_gt_2": float(np.mean(values > 2)),
            "p_z_gt_3": float(np.mean(values > 3)),
        })
    histograms = np.asarray(histograms)
    quantiles = np.asarray(quantiles)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": (1.5, 1)})
    extent = [unique_calls[0] - .5, unique_calls[-1] + .5, Z_EDGES[0], Z_EDGES[-1]]
    axes[0].imshow(histograms.T, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap="magma")
    axes[0].set_ylabel("standardized z")
    axes[0].set_title(f"RULER 16K stepwise prefix proxy: {attention_type}, layer {layer}, head {head}, query block {query_block}")
    axes[0].set_yticks([-4, -2, 0, 2, 4, 6])
    axes[1].plot(unique_calls, quantiles[:, 0], label="p01")
    axes[1].plot(unique_calls, quantiles[:, 1], label="p05")
    axes[1].plot(unique_calls, quantiles[:, 2], label="median")
    axes[1].plot(unique_calls, quantiles[:, 3], label="p95")
    axes[1].plot(unique_calls, quantiles[:, 4], label="p99")
    axes[1].set_xlabel("denoising call")
    axes[1].set_ylabel("z quantile")
    axes[1].legend(ncol=5, frameon=False, fontsize=8)
    path1 = root / "01_stepwise_selected_row_distribution.png"
    _save(fig, path1)

    calls_all = data["denoising_call"].astype(int)
    types = data["attention_type"].astype(str)
    layers = sorted(set(data["layer"].astype(int)))
    heads = sorted(set(data["head"].astype(int)))
    tail = np.full((len(layers), len(heads)), np.nan)
    for li, layer_value in enumerate(layers):
        for hi, head_value in enumerate(heads):
            mask = (types == attention_type) & (data["layer"] == layer_value) & (data["head"] == head_value)
            if not mask.any(): continue
            zz, vv, _ = _standardized(data, mask)
            values = zz[vv]
            tail[li, hi] = np.mean(values > 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(tail, aspect="auto", origin="lower", vmin=0, vmax=max(.1, np.nanmax(tail)), cmap="viridis")
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"Per-layer/head standardized upper-tail rate P(z>2): {attention_type}")
    fig.colorbar(im, ax=ax, label="P(z > 2)")
    path2 = root / "02_layer_head_tail_heterogeneity.png"
    _save(fig, path2)

    path3 = root / "03_mask_ratio_and_step_stats.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot([x["denoising_call"] for x in step_stats], [x["mask_ratio"] for x in step_stats], marker="o")
    axes[0].set_ylabel("masked-token ratio")
    axes[0].set_title("Discrete denoising state for selected row")
    axes[1].plot([x["denoising_call"] for x in step_stats], [x["p_z_gt_2"] for x in step_stats], marker="o", label="P(z>2)")
    axes[1].plot([x["denoising_call"] for x in step_stats], [x["p_z_gt_3"] for x in step_stats], marker="o", label="P(z>3)")
    axes[1].set_xlabel("denoising call"); axes[1].set_ylabel("upper-tail probability"); axes[1].legend(frameon=False)
    _save(fig, path3)

    result = {
        "schema_version": 1,
        "attention_type": attention_type,
        "layer": layer,
        "head": head,
        "query_block": query_block,
        "rows_selected": int(selected.sum()),
        "calls": step_stats,
        "layer_head_tail_rate_min": float(np.nanmin(tail)),
        "layer_head_tail_rate_max": float(np.nanmax(tail)),
        "layer_head_tail_rate_std": float(np.nanstd(tail)),
        "plots": [path1.name, path2.name, path3.name],
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
