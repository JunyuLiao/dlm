#!/usr/bin/env python3
"""Generate the compact plots used by the query-regrouping report."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=pathlib.Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.analysis_dir / "per_record.csv").open()))
    selected = [
        row for row in rows
        if row["heads_per_group"] == "1" and row["window_size"] == "4096"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for source, linestyle in (("current", "-"), ("previous", "--")):
        for ratio in sorted({float(row["mask_ratio"]) for row in selected}, reverse=True):
            values = sorted(
                (int(row["layer"]), 100 * float(row["improvement"]))
                for row in selected if row["source"] == source and float(row["mask_ratio"]) == ratio
            )
            axes[0].plot(
                [value[0] for value in values], [value[1] for value in values],
                linestyle, label=f"{source}, mask={ratio:.2f}", linewidth=1.5,
            )
    axes[0].axhline(15, color="black", linewidth=1, alpha=0.5, label="15 pp gate")
    axes[0].set(xlabel="Layer", ylabel="Physical sparsity gain (percentage points)", title="Per-head bitmap regrouping")
    axes[0].legend(fontsize=7, ncol=2)
    temporal = list(csv.DictReader((args.analysis_dir / "temporal.csv").open()))
    ratios = sorted({float(row["mask_ratio"]) for row in temporal}, reverse=True)
    distributions = [[float(row["jaccard"]) for row in temporal if float(row["mask_ratio"]) == ratio] for ratio in ratios]
    axes[1].boxplot(distributions, tick_labels=[f"{ratio:.2f}" for ratio in ratios], showmeans=True)
    axes[1].set(xlabel="Remaining mask ratio", ylabel="Row/head Jaccard", title="Adjacent-step pattern stability", ylim=(0, 1.02))
    fig.savefig(args.analysis_dir / "regrouping_results.png", dpi=180)


if __name__ == "__main__":
    main()
