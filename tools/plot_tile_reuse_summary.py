#!/usr/bin/env python3
"""Create the compact decision plot used by the tile-reuse report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.input_dir / "summary.json").read_text())
    causal = json.loads((args.input_dir / "causal_mass_policy.json").read_text())
    micro = json.loads((args.input_dir / "h100_microbenchmark.json").read_text())

    oracle = summary["oracle_reuse_heldout"]
    thresholds = np.asarray([float(value) for value in oracle])
    coverage = np.asarray([100.0 * oracle[value] for value in oracle])

    policy_fragment = "protection=none:max_age=unlimited:accumulation=sum"
    rows = {
        row["split"]: row
        for row in causal["all_splits"]
        if policy_fragment in row["policy"]
    }

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].semilogx(thresholds, coverage, marker="o", color="#1976d2")
    axes[0].axhspan(30, 40, color="#66bb6a", alpha=0.15, label="oracle gate")
    axes[0].set(title="Oracle tile stability", xlabel="single-tile output-error limit", ylabel="held-out reusable tiles (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    names = ["calibration", "heldout", "final_benchmark"]
    x = np.arange(len(names))
    reuse = [100.0 * rows[name]["reuse_fraction"] for name in names]
    unsafe = [100.0 * rows[name]["unsafe_wilson95_upper"] for name in names]
    width = 0.38
    axes[1].bar(x - width / 2, reuse, width, label="reuse", color="#1976d2")
    axes[1].bar(x + width / 2, unsafe, width, label="unsafe upper bound", color="#ef6c00")
    axes[1].axhline(1.0, color="#c62828", linestyle="--", linewidth=1, label="safety gate")
    axes[1].set(title="Best causal policy", ylabel="tiles (%)", xticks=x, xticklabels=["calib.", "held-out", "final"])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    labels = ["fresh", "cached BF16", "cached INT8", "mixed 14%"]
    mixed = next(
        row
        for row in micro["mixed"]
        if row["reuse_ratio"] == 0.14 and row["pattern"] == "contiguous"
    )
    latencies = [
        micro["fresh_all_ms"],
        micro["cached_all_ms"],
        micro["cached_int8_all_ms"],
        mixed["latency_ms"],
    ]
    axes[2].bar(np.arange(4), latencies, color=["#1976d2", "#8e24aa", "#6a1b9a", "#ef6c00"])
    axes[2].set(title="H100 tile-path diagnostic", ylabel="latency for 256 tiles (ms)", xticks=np.arange(4), xticklabels=labels)
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.25)

    figure.suptitle("2D attention-result caching: the oracle exists, but the online hardware path fails")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
