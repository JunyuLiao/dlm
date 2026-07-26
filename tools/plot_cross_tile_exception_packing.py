#!/usr/bin/env python3
"""Plot the Phase-A cross-tile exception-packing gate metrics."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=pathlib.Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.analysis_dir / "configurations.csv").open()))
    rows = [row for row in rows if row["grouping"] == "q-head"]
    taus = sorted({int(row["tau"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for pack_m in (64, 128):
        selected = {int(row["tau"]): row for row in rows if int(row["pack_m"]) == pack_m}
        axes[0].plot(
            taus, [100 * float(selected[tau]["weighted_utilization"]) for tau in taus],
            marker="o", label=f"PACK_M={pack_m}",
        )
        axes[1].plot(
            taus, [float(selected[tau]["v_load_factor"]) for tau in taus],
            marker="o", label=f"active-voter PACK_M={pack_m}",
        )
    axes[0].axhline(70, color="black", linestyle="--", linewidth=1, label="70% gate")
    axes[0].set(
        xlabel="Exception threshold τ", ylabel="Weighted utilization (%)",
        title="Counterfactual active-voter packability", xscale="log", xticks=taus,
    )
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].legend(fontsize=8)
    axes[1].axhline(2, color="black", linestyle="--", linewidth=1, label="2× gate")
    axes[1].axhline(1, color="tab:red", linestyle=":", linewidth=1.5, label="exact PACK_M=128")
    axes[1].axhline(0.5, color="tab:orange", linestyle=":", linewidth=1.5, label="exact PACK_M=64")
    axes[1].set(
        xlabel="Exception threshold τ", ylabel="Original / packed V loads",
        title="Theoretical versus semantics-preserving V reuse", xscale="log", xticks=taus,
    )
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(fontsize=8)
    fig.savefig(args.analysis_dir / "phase_a_gate.png", dpi=180)


if __name__ == "__main__":
    main()
