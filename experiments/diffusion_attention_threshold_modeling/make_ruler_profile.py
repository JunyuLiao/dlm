"""Build a compact model/attention profiled beta table from RULER16K shards."""

from __future__ import annotations

import json
import gc
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("niah_multikey_1", "niah_multivalue", "niah_multiquery", "vt", "fwe")


def _task(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        request_id = str(data["request_id"][0])
    for task in TASKS:
        if f"_{task}_" in request_id:
            return task
    return "unknown"


def _split_paths(paths: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_task(path), []).append(path)
    calibration, validation = [], []
    for task, members in sorted(groups.items()):
        members.sort(key=lambda path: hashlib.sha256(f"{seed}|{task}|{path.name}".encode()).digest())
        calibration_count = len(members) if len(members) == 1 else min(len(members) - 1, max(1, math.floor(.8 * len(members))))
        calibration.extend(members[:calibration_count])
        validation.extend(members[calibration_count:])
    return sorted(calibration), sorted(validation)


def _fit_adapter(paths: list[Path], adapter: str, densities: tuple[float, ...]) -> dict[str, Any]:
    fits: dict[str, Any] = {}
    edges = np.linspace(-12.0, 12.0, 49_153, dtype=np.float64)
    histograms: dict[str, np.ndarray] = {}
    row_counts: dict[str, int] = {}
    for path in paths:
        data = np.load(path, allow_pickle=False)
        attention_values = data["attention_type"]
        prefix_mean = data["prefix_mean"]
        prefix_std = data["prefix_std"]
        prefix_reservoir = data["prefix_reservoir"]
        hierarchical_row_weight = data["hierarchical_row_weight"]
        for attention_type in np.unique(attention_values):
            name = str(attention_type)
            histogram = histograms.setdefault(name, np.zeros(len(edges) - 1, dtype=np.float64))
            mask_indices = np.flatnonzero(attention_values == attention_type)
            row_counts[name] = row_counts.get(name, 0) + int(len(mask_indices))
            for start in range(0, len(mask_indices), 4096):
                indices = mask_indices[start : start + 4096]
                means = prefix_mean[indices].astype(np.float64)
                stds = np.maximum(prefix_std[indices].astype(np.float64), 1.0e-8)
                values = (prefix_reservoir[indices].astype(np.float64) - means[:, None]) / stds[:, None]
                finite = np.isfinite(values)
                counts = finite.sum(axis=1)
                row_weight = hierarchical_row_weight[indices].astype(np.float64)
                per_value = np.divide(row_weight, counts, out=np.zeros_like(row_weight), where=counts > 0)
                clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
                histogram += np.histogram(
                    clipped[finite], bins=edges,
                    weights=np.broadcast_to(per_value[:, None], values.shape)[finite],
                )[0]
        del data, prefix_mean, prefix_std, prefix_reservoir, hierarchical_row_weight
        gc.collect()
    for attention_type, histogram in histograms.items():
        if histogram.sum() <= 0:
            continue
        cdf = np.cumsum(histogram) / histogram.sum()
        centers = (edges[:-1] + edges[1:]) * 0.5
        betas = {}
        for density in densities:
            betas[str(float(density))] = float(np.interp(1.0 - density, cdf, centers))
        fits[f"{adapter}|{attention_type}"] = {
            "family": "empirical_cdf",
            "parameters": {"source": "ruler16k_prefix_only_shards", "rows": row_counts[attention_type]},
            "betas": betas,
        }
    return fits


def _validate_adapter(paths: list[Path], adapter: str, fits: dict[str, Any]) -> list[dict[str, Any]]:
    accumulators: dict[tuple[str, float], list[float]] = {}
    for path in paths:
        data = np.load(path, allow_pickle=False)
        attention_values = data["attention_type"]
        prefix_mean = data["prefix_mean"]
        prefix_std = data["prefix_std"]
        prefix_reservoir = data["prefix_reservoir"]
        hierarchical_row_weight = data["hierarchical_row_weight"]
        for attention_type in np.unique(attention_values):
            key = f"{adapter}|{attention_type}"
            if key not in fits:
                continue
            indices = np.flatnonzero(attention_values == attention_type)
            prompt_totals = {
                float(density_text): [0.0, 0.0]
                for density_text in fits[key]["betas"]
            }
            for start in range(0, len(indices), 4096):
                chunk = indices[start : start + 4096]
                means = prefix_mean[chunk].astype(np.float64)
                stds = np.maximum(prefix_std[chunk].astype(np.float64), 1.0e-8)
                z = (prefix_reservoir[chunk].astype(np.float64) - means[:, None]) / stds[:, None]
                finite = np.isfinite(z)
                finite_counts = finite.sum(axis=1)
                weights = hierarchical_row_weight[chunk].astype(np.float64)
                for density_text, beta in fits[key]["betas"].items():
                    density = float(density_text)
                    row_density = np.divide(
                        ((z >= beta) & finite).sum(axis=1), finite_counts,
                        out=np.zeros(len(chunk), dtype=np.float64), where=finite_counts > 0,
                    )
                    prompt_totals[density][0] += float(weights @ row_density)
                    prompt_totals[density][1] += float(weights.sum())
            for density, (numerator, denominator) in prompt_totals.items():
                accumulators.setdefault((str(attention_type), density), []).append(numerator / denominator)
        del data, prefix_mean, prefix_std, prefix_reservoir, hierarchical_row_weight
        gc.collect()
    results = []
    for (attention_type, density), values in sorted(accumulators.items()):
        errors = np.asarray(values, dtype=np.float64) - density
        seed = int.from_bytes(hashlib.sha256(f"{adapter}|{attention_type}|{density}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        bootstrap = errors[rng.integers(0, len(errors), size=(20_000, len(errors)))].mean(axis=1)
        mean_error = float(errors.mean())
        bootstrap_upper = float(np.quantile(np.abs(bootstrap), .95))
        max_prompt_error = float(np.max(np.abs(errors)))
        results.append({
            "adapter": adapter,
            "attention_type": attention_type,
            "target_density": density,
            "validation_prompt_densities": values,
            "mean_realized_density": float(np.mean(values)),
            "absolute_error": abs(mean_error),
            "prompt_bootstrap_95_upper_absolute_error": bootstrap_upper,
            "max_prompt_absolute_error": max_prompt_error,
            "passes": abs(mean_error) <= .02 and bootstrap_upper <= .03 and max_prompt_error <= .05,
        })
    return results


def build_profile(input_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(input_root)
    densities = (0.75, 0.50, 0.25, 0.10)
    fits = {}
    validation = []
    splits = {}
    for adapter in ("diffusion_gemma", "fast_dllm_v2"):
        paths = sorted((root / "shards" / adapter).glob("*.npz"))
        calibration_paths, validation_paths = _split_paths(paths, 20260825)
        adapter_fits = _fit_adapter(calibration_paths, adapter, densities)
        fits.update(adapter_fits)
        validation.extend(_validate_adapter(validation_paths, adapter, adapter_fits))
        splits[adapter] = {
            "calibration": [path.name for path in calibration_paths],
            "validation": [path.name for path in validation_paths],
        }
    result = {
        "schema_version": 1,
        "selected": {"scope": "model_attention", "family": "empirical_cdf", "fits": fits},
        "source": str(root),
        "population": "RULER16K prefix-only, 10 examples per adapter",
        "calibration_only_audit": True,
        "split_seed": 20260825,
        "splits": splits,
        "held_out_validation": validation,
        "simple_profiling_passed": bool(validation) and all(row["passes"] for row in validation),
    }
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_profile(args.input_root, args.output), indent=2, sort_keys=True))
