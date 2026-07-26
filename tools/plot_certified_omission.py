#!/usr/bin/env python3
"""Plot certified-omission coverage and exact aggregate error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    budgets = [row["mass_budget"] for row in summary["budgets"]]
    by_policy = {
        policy: [
            next(
                row
                for row in summary["bound_decomposition"]
                if row["policy"] == policy and row["mass_budget"] == budget
            )
            for budget in budgets
        ]
        for policy in (
            "temporal_upper_with_exact_current_denominator",
            "oracle_exact_current_mass",
        )
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(
        budgets,
        [100 * row["omission_fraction"] for row in summary["budgets"]],
        marker="o",
        label="causal certificate",
    )
    axes[0].plot(
        budgets,
        [100 * row["omission_fraction"] for row in by_policy["temporal_upper_with_exact_current_denominator"]],
        marker="o",
        label="exact denominator diagnostic",
    )
    axes[0].plot(
        budgets,
        [100 * row["omission_fraction"] for row in by_policy["oracle_exact_current_mass"]],
        marker="o",
        label="exact mass oracle",
    )
    axes[0].axhspan(25, 30, alpha=0.12, color="green", label="hardware gate")
    axes[0].set(xlabel="aggregate mass budget", ylabel="omitted physical tiles (%)", title="Coverage upper bounds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        budgets,
        [100 * row["maximum_group_output_error"] for row in summary["budgets"]],
        marker="o",
        label="causal certificate",
    )
    axes[1].plot(
        budgets,
        [100 * row["maximum_group_output_error"] for row in by_policy["temporal_upper_with_exact_current_denominator"]],
        marker="o",
        label="exact denominator diagnostic",
    )
    axes[1].plot(
        budgets,
        [100 * row["maximum_group_output_error"] for row in by_policy["oracle_exact_current_mass"]],
        marker="o",
        label="exact mass oracle",
    )
    axes[1].axhline(1.0, linestyle="--", color="red", linewidth=1, label="1% error target")
    axes[1].set(xlabel="aggregate mass budget", ylabel="maximum query-group relative error (%)", title="Exact simultaneous-omission error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Certified tile omission: safe, but far below hardware-useful coverage")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
