"""Prompt-level convergence gates for adaptive distribution collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


CHECKPOINTS = (1, 2, 5, 10, 20, 40, 50, 75, 100)


def prompt_tail_metrics(paths: Iterable[Path], densities: Iterable[float]) -> dict[tuple[float, str], list[float]]:
    result: dict[tuple[float, str], list[float]] = {}
    for path in sorted(paths):
        with np.load(path) as payload:
            weights = payload["hierarchical_row_weight"].astype(float)
            attention = payload["attention_type"].astype(str)
            for density in densities:
                values = payload[f"gaussian_density_{float(density):g}"].astype(float)
                for attention_type in sorted(set(attention)):
                    valid = np.isfinite(values) & (attention == attention_type)
                    result.setdefault((float(density), attention_type), []).append(
                        float(np.average(values[valid], weights=weights[valid]))
                    )
    return result


def convergence_summary(
    paths: Iterable[Path], densities: Iterable[float], *, seed: int, bootstrap_repeats: int = 4_000
) -> dict:
    paths = list(sorted(paths))
    metrics = prompt_tail_metrics(paths, densities)
    curves = []
    rng = np.random.default_rng(seed)
    failures = []
    for (density, attention_type), values_list in metrics.items():
        values = np.asarray(values_list, dtype=float)
        for count in CHECKPOINTS:
            if count > len(values):
                continue
            sample = values[:count]
            boot = sample[rng.integers(0, count, size=(bootstrap_repeats, count))].mean(axis=1)
            lower, upper = np.quantile(boot, (0.025, 0.975))
            curves.append({
                "target_density": density,
                "attention_type": attention_type,
                "prompts": count,
                "mean_realized_density": float(sample.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "ci95_half_width": float((upper - lower) / 2),
            })
        if len(values) >= 50:
            change = abs(float(values[:50].mean() - values[:40].mean()))
            latest = next(row for row in curves if row["target_density"] == density and row["attention_type"] == attention_type and row["prompts"] == len(values))
            failed = change > 0.01 or latest["ci95_half_width"] > 0.02
            failures.append({
                "target_density": density,
                "attention_type": attention_type,
                "change_40_to_50": change,
                "latest_ci95_half_width": latest["ci95_half_width"],
                "failed": failed,
            })
    return {
        "prompt_count": len(paths),
        "curves": curves,
        "gates": failures,
        "passed": bool(failures) and not any(row["failed"] for row in failures),
        "requires_extension": bool(failures) and any(row["failed"] for row in failures) and len(paths) < 100,
    }


def write_convergence(root: Path, adapter: str, corpus: str, densities: Iterable[float], seed: int) -> dict:
    shard_root = root / "shards" / adapter / corpus
    summary = convergence_summary(shard_root.glob("*.npz"), densities, seed=seed)
    output = root / "collection" / adapter / corpus / "convergence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
