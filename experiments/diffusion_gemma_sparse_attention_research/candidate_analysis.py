"""Select simple pre-QK candidate scores on calibration prompts and test held-out prompts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .offline_analysis import OUTPUT, top_budget_mask
from .rich_analysis import SOURCE, load_rows


TARGETS = (0.25, 0.50, 0.75, 0.90)
WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float) / max(len(values) - 1, 1)


def candidate_score(item: dict, family: str, weight: float) -> np.ndarray:
    eligible = item["eligible"]
    proxy = rank01(item["sol_proxy"][eligible])
    if family == "sol_proxy":
        return proxy
    history_name = {
        "mass_history": "previous_step_mass",
        "drop_history": "previous_step_drop_error",
    }[family]
    if history_name not in item:
        return proxy
    history = rank01(item[history_name][eligible])
    return (1.0 - weight) * proxy + weight * history


def evaluate(rows: list[dict], attention_type: str, target: float, family: str, weight: float) -> dict:
    selected = [row for row in rows if row["attention_type"] == attention_type]
    totals = {"mass_kept": 0.0, "mass": 0.0, "drop_kept": 0.0, "drop": 0.0}
    kept_tiles = eligible_tiles = history_records = max_hits = 0
    prompt_values: dict[str, list[float]] = {}
    for item in selected:
        eligible = item["eligible"]
        score = candidate_score(item, family, weight)
        keep = top_budget_mask(score, target)
        mass = item["current_dense_mass"][eligible]
        drop = item["current_drop_error"][eligible]
        kept_tiles += int(keep.sum())
        eligible_tiles += len(keep)
        totals["mass_kept"] += float(mass[keep].sum())
        totals["mass"] += float(mass.sum())
        totals["drop_kept"] += float(drop[keep].sum())
        totals["drop"] += float(drop.sum())
        history_records += int("previous_step_mass" in item)
        max_hits += int(keep[np.argmax(mass)])
        values = prompt_values.setdefault(item["example_id"], [0.0, 0.0, 0.0, 0.0])
        values[0] += float(mass[keep].sum())
        values[1] += float(mass.sum())
        values[2] += float(drop[keep].sum())
        values[3] += float(drop.sum())
    retained_mass = totals["mass_kept"] / totals["mass"]
    retained_drop = totals["drop_kept"] / totals["drop"]
    prompt_mass = [v[0] / v[1] for v in prompt_values.values()]
    prompt_drop = [v[2] / v[3] for v in prompt_values.values()]
    return {
        "records": len(selected),
        "history_records": history_records,
        "history_coverage": history_records / len(selected),
        "target_sparsity": target,
        "actual_sparsity": 1.0 - kept_tiles / eligible_tiles,
        "retained_mass": retained_mass,
        "retained_drop_error_sum": retained_drop,
        "joint_calibration_utility": math.sqrt(retained_mass * retained_drop),
        "highest_mass_tile_recall": max_hits / len(selected),
        "prompt_retained_mass_mean": float(np.mean(prompt_mass)),
        "prompt_retained_mass_std": float(np.std(prompt_mass)),
        "prompt_retained_drop_mean": float(np.mean(prompt_drop)),
        "prompt_retained_drop_std": float(np.std(prompt_drop)),
    }


def evaluate_censored_history(
    rows: list[dict], attention_type: str, target: float, weight: float, exploration_bonus: float
) -> dict:
    """Causal dense-trajectory simulation: only retained tiles reveal exact mass."""
    selected = [row for row in rows if row["attention_type"] == attention_type]
    sequences: dict[tuple, list[dict]] = {}
    for item in selected:
        key = (item["example_id"], item["layer"], item["head"], item["query_start"], item["kv_length"])
        sequences.setdefault(key, []).append(item)
    totals = {"mass_kept": 0.0, "mass": 0.0, "drop_kept": 0.0, "drop": 0.0}
    kept_tiles = eligible_tiles = max_hits = 0
    observed_tile_decisions = all_tile_decisions = 0
    record_count = 0
    for sequence in sequences.values():
        sequence.sort(key=lambda item: item["call"])
        history = np.full(len(sequence[0]["eligible"]), np.nan)
        age = np.zeros(len(history), dtype=float)
        for item in sequence:
            eligible = item["eligible"]
            proxy = rank01(item["sol_proxy"][eligible])
            observed = np.isfinite(history[eligible])
            score = proxy.copy()
            if observed.any():
                history_rank = rank01(history[eligible][observed])
                score[observed] = (1.0 - weight) * proxy[observed] + weight * history_rank
                maximum_age = max(float(age[eligible].max()), 1.0)
                score += exploration_bonus * age[eligible] / maximum_age
            keep = top_budget_mask(score, target)
            mass = item["current_dense_mass"][eligible]
            drop = item["current_drop_error"][eligible]
            kept_tiles += int(keep.sum())
            eligible_tiles += len(keep)
            totals["mass_kept"] += float(mass[keep].sum())
            totals["mass"] += float(mass.sum())
            totals["drop_kept"] += float(drop[keep].sum())
            totals["drop"] += float(drop.sum())
            max_hits += int(keep[np.argmax(mass)])
            observed_tile_decisions += int(observed.sum())
            all_tile_decisions += len(observed)
            full_keep = np.zeros(len(history), dtype=bool)
            full_keep[np.flatnonzero(eligible)[keep]] = True
            age += 1.0
            history[full_keep] = item["current_dense_mass"][full_keep]
            age[full_keep] = 0.0
            record_count += 1
    retained_mass = totals["mass_kept"] / totals["mass"]
    retained_drop = totals["drop_kept"] / totals["drop"]
    return {
        "records": record_count,
        "target_sparsity": target,
        "actual_sparsity": 1.0 - kept_tiles / eligible_tiles,
        "retained_mass": retained_mass,
        "retained_drop_error_sum": retained_drop,
        "joint_calibration_utility": math.sqrt(retained_mass * retained_drop),
        "highest_mass_tile_recall": max_hits / record_count,
        "history_observed_tile_fraction_before_decision": observed_tile_decisions / all_tile_decisions,
    }


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    rows = load_rows(source)
    calibration = [row for row in rows if row["split"] == "calibration"]
    final = [row for row in rows if row["split"] == "final"]
    selected_policy = {}
    grid = []
    held_out = []
    censored_grid = []
    censored_policy = {}
    censored_held_out = []
    for attention_type in ("local", "global"):
        selected_policy[attention_type] = {}
        for target in TARGETS:
            choices = []
            for family in ("mass_history", "drop_history"):
                for weight in WEIGHTS:
                    metrics = evaluate(calibration, attention_type, target, family, weight)
                    choices.append((metrics["joint_calibration_utility"], family, weight, metrics))
                    grid.append({"attention_type": attention_type, "family": family, "weight": weight, **metrics})
            _, family, weight, calibration_metrics = max(choices, key=lambda value: (value[0], -value[2], value[1]))
            selected_policy[attention_type][f"s{int(target*100)}"] = {
                "family": family,
                "history_weight": weight,
                "selection_objective": "maximize geometric mean of retained mass and retained drop-one-tile importance on calibration prompts",
                "calibration": calibration_metrics,
            }
            for name, eval_family, eval_weight in (
                ("sol_proxy", "sol_proxy", 0.0),
                ("history_mass_with_sol_fallback", "mass_history", 1.0),
                ("history_drop_with_sol_fallback", "drop_history", 1.0),
                ("selected_blend", family, weight),
            ):
                held_out.append(
                    {
                        "attention_type": attention_type,
                        "target_sparsity": target,
                        "candidate": name,
                        "family": eval_family,
                        "history_weight": eval_weight,
                        **evaluate(final, attention_type, target, eval_family, eval_weight),
                    }
                )
            censored_choices = []
            for censored_weight in WEIGHTS:
                for exploration_bonus in (0.0, 0.10, 0.25):
                    metrics = evaluate_censored_history(
                        calibration, attention_type, target, censored_weight, exploration_bonus
                    )
                    censored_choices.append(
                        (metrics["joint_calibration_utility"], censored_weight, exploration_bonus, metrics)
                    )
                    censored_grid.append(
                        {
                            "attention_type": attention_type,
                            "history_weight": censored_weight,
                            "exploration_bonus": exploration_bonus,
                            **metrics,
                        }
                    )
            _, censored_weight, exploration_bonus, calibration_metrics = max(
                censored_choices, key=lambda value: (value[0], -value[1], -value[2])
            )
            censored_policy.setdefault(attention_type, {})[f"s{int(target*100)}"] = {
                "history_weight": censored_weight,
                "exploration_bonus": exploration_bonus,
                "calibration": calibration_metrics,
            }
            for name, chosen_weight, chosen_bonus in (
                ("sol_proxy", 0.0, 0.0),
                ("selected_censored_history", censored_weight, exploration_bonus),
            ):
                censored_held_out.append(
                    {
                        "attention_type": attention_type,
                        "candidate": name,
                        "history_weight": chosen_weight,
                        "exploration_bonus": chosen_bonus,
                        **evaluate_censored_history(final, attention_type, target, chosen_weight, chosen_bonus),
                    }
                )
    payload = {
        "protocol": {
            "selection_split": "8 calibration prompts",
            "evaluation_split": "8 disjoint final prompts",
            "budget": "deterministic top-k physical 64x64 tiles per snapshot, with at least one retained",
            "signals": "current Sol mean-pooled proxy plus previous-step exact dense mass or drop-one-tile importance",
            "availability": "dense-history blends are diagnostic upper bounds because skipped tiles would lose exact history; the censored simulation updates exact history only for retained tiles and is pre-current-QK causal",
            "objective": "geometric mean of retained current dense mass and retained current drop-one-tile importance",
            "limits": [
                "offline dense-trajectory evaluation, not a sparse rollout",
                "drop-one-tile importance is not additive multi-tile output error",
                "attention-type/target-specific scalar weight selected from five values",
            ],
        },
        "selected_policy": selected_policy,
        "calibration_grid": grid,
        "held_out": held_out,
        "censored_policy": censored_policy,
        "censored_calibration_grid": censored_grid,
        "censored_held_out": censored_held_out,
    }
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.0), sharey=True)
    for axis, attention_type in zip(axes, ("local", "global")):
        for candidate in ("sol_proxy", "history_mass_with_sol_fallback", "history_drop_with_sol_fallback", "selected_blend"):
            values = sorted([row for row in held_out if row["attention_type"] == attention_type and row["candidate"] == candidate], key=lambda row: row["actual_sparsity"])
            axis.plot([row["actual_sparsity"] for row in values], [row["retained_drop_error_sum"] for row in values], marker="o", label=candidate.replace("_", " "))
        axis.set_title(attention_type)
        axis.set_xlabel("Actual physical-tile sparsity")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Retained drop-one-tile importance sum")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    figure = figure_dir / "10_candidate_held_out_budget_curves.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    payload["figure"] = str(figure.relative_to(output))

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.0), sharey=True)
    for axis, attention_type in zip(axes, ("local", "global")):
        for candidate in ("sol_proxy", "selected_censored_history"):
            values = sorted(
                [row for row in censored_held_out if row["attention_type"] == attention_type and row["candidate"] == candidate],
                key=lambda row: row["actual_sparsity"],
            )
            axis.plot(
                [row["actual_sparsity"] for row in values],
                [row["retained_drop_error_sum"] for row in values],
                marker="o",
                label=candidate.replace("_", " "),
            )
        axis.set_title(attention_type)
        axis.set_xlabel("Actual physical-tile sparsity")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Retained drop-one-tile importance sum")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    causal_figure = figure_dir / "11_censored_history_held_out.png"
    fig.savefig(causal_figure, dpi=180)
    plt.close(fig)
    payload["censored_figure"] = str(causal_figure.relative_to(output))
    (output / "candidate_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = run(args.source, args.output)
    print(json.dumps({"censored_policy": payload["censored_policy"], "figure": payload["censored_figure"]}, indent=2))


if __name__ == "__main__":
    main()
