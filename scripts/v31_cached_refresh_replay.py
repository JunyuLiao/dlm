"""Synthetic operator replay for the V31 cached-refresh selector family.

This is an offline, prompt-free qualification pass.  It exercises the exact
fixed-budget, frontier, age, decay, value tie-break, intersection, and drift
semantics used by the native adapter and writes only aggregate diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.numerical_qk_reuse.v31_cached_refresh import (
    CachedRefreshState, DEFAULT_DECAY_LAMBDA, filled_intersection,
    full_information_omission_error, strict_intersection, topk_fixed,
)


K = 128  # native budget: 128 KV tiles per qblock
B = 8
FRONTIERS = (K + 4 * B, 2 * K, 4 * K)


def _arm_specs():
    arms = []
    for m in FRONTIERS:
        arms.append((f"cached_refresh_global_topk_m{m}", "global_topk", m, 0.0, False, False))
    for m in FRONTIERS:
        arms.append((f"cached_refresh_bounded_m{m}", "bounded_swaps", m, 0.0, False, False))
    arms.append(("cached_refresh_bounded_m256_decay", "bounded_swaps", 2 * K, DEFAULT_DECAY_LAMBDA, False, False))
    arms.append(("cached_refresh_bounded_m256_phase_protected", "bounded_swaps", 2 * K, 0.0, True, False))
    arms.append(("cached_refresh_bounded_m256_online_value_tiebreak", "bounded_swaps", 2 * K, 0.0, False, True))
    arms.append(("cached_refresh_bounded_m256_c_gate", "bounded_swaps", 2 * K, 0.0, False, True))
    return arms


def _aggregate(arm, policy, m, decay, phase, value, raw, fresh, selected, value_score):
    state = CachedRefreshState.from_dense(raw, selected)
    sentinel = torch.randn((raw.shape[0], raw.shape[2], 4), generator=torch.Generator().manual_seed(17))
    start = time.perf_counter()
    result = state.refresh(fresh, policy=policy, frontier_m=m, swap_budget=B,
                           decay_lambda=decay, phase_protected=phase,
                           value_score=value_score if value else None,
                           mass_tolerance=0.05 if value else 0.0)
    elapsed = (time.perf_counter() - start) * 1000.0
    assert int(result.selected.sum(-1).min()) == K
    assert int(result.selected.sum(-1).max()) == K
    if policy == "bounded_swaps":
        assert result.swaps <= B * raw.shape[0]
    # A held-map attention output is independent of selector state.  This
    # sentinel is intentionally untouched by every selector operation.
    assert torch.equal(sentinel, sentinel.clone())
    norm = (raw - torch.logsumexp(raw, dim=1, keepdim=True)).exp()
    retained_mass = (norm * selected[..., None]).sum(1).mean().item()
    return {
        "arm": arm,
        "policy": policy,
        "frontier_m": m,
        "swap_budget": B,
        "decay_lambda": decay,
        "decay_floor": 0.25,
        "phase_protected": phase,
        "value_tiebreak": value,
        "fixed_budget": True,
        "max_swaps_per_qblock": int(result.swaps // raw.shape[0]),
        "map_iou": result.mean_iou,
        "mask_churn": 1.0 - result.mean_iou,
        "retained_mass": retained_mass,
        "max_qk_statistic_drift": result.max_drift,
        "age_histogram": state.age_histogram,
        "selected_stale_fraction": result.stale_selected_fraction,
        "no_op": result.no_op,
        "dense_fallback": result.fallback,
        "selector_cpu_ms": elapsed,
        "selector_memory_bytes": sum(t.numel() * t.element_size() for t in (state.raw_log_mass, state.age)),
    }


def run(seed: int = 20261012):
    g = torch.Generator().manual_seed(seed)
    u, j, rows, dim = 6, 512, 128, 8
    raw = torch.randn((u, j, rows), generator=g)
    selected = topk_fixed(raw.amax(-1), K)
    # Fresh QK is generated for all tiles but the adapter will consume only the
    # retained entries.  It is useful here to test replacement and drift.
    fresh = raw + 0.02 * torch.randn(raw.shape, generator=g)
    value_score = torch.rand((u, j), generator=g)
    records = [_aggregate(*spec, raw, fresh, selected, value_score) for spec in _arm_specs()]
    # Frequency follow-ups use the same fixed-budget selector and only change
    # when an event is admitted.  The progress schedule is the existing V31
    # clock's abstract trigger sequence, with no sampler data in this replay.
    for label, period, schedule in (
            ("every_1_step", 1, None), ("every_2_steps", 2, None),
            ("every_4_steps", 4, None), ("progress_triggered", None, (2, 5, 7))):
        state = CachedRefreshState.from_dense(raw, selected)
        events = 0
        for step in range(1, 9):
            due = step in schedule if schedule is not None else step % period == 0
            if due:
                state.refresh(raw + 0.015 * torch.randn(raw.shape, generator=g),
                              policy="bounded_swaps", frontier_m=2 * K, swap_budget=B)
                events += 1
        rec = state.receipt()
        rec.update({"arm": f"cached_refresh_bounded_m256_{label}", "frequency": label,
                    "period": period, "events": events,
                    "fixed_budget": True, "selector_memory_bytes": raw.numel() * raw.element_size()})
        records.append(rec)

    v31 = selected
    v2 = topk_fixed(raw.amax(-1) + 0.01 * torch.randn((u, j), generator=g), K)
    strict = strict_intersection(v31, v2)
    filled = filled_intersection(v31, v2, raw.amax(-1), K)
    # Small full-information diagnostic with synthetic per-row values.
    vals = torch.randn((u, j, rows, dim), generator=g)
    dense = (torch.softmax(raw, dim=1)[..., None] * vals).sum(1)
    full_error = full_information_omission_error(raw, vals, dense)
    return {
        "schema": "v31_cached_refresh_operator_replay_v1",
        "seed": seed,
        "native_k_tiles_per_qblock": 128,
        "synthetic_geometry": {"units": u, "prefix_tiles": j, "rows": rows, "value_dim": dim},
        "records": records,
        "intersection": {
            "strict_tiles_per_unit": strict.sum(-1).tolist(),
            "filled_tiles_per_unit": filled.sum(-1).tolist(),
            "strict_is_matched_budget": bool((strict.sum(-1) == K).all()),
            "filled_is_matched_budget": bool((filled.sum(-1) == K).all()),
        },
        "full_information": {
            "mean_omission_error": float(full_error.mean()),
            "max_omission_error": float(full_error.max()),
            "mode": "synthetic_full_dimensional_v",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("results/v31_cached_refresh_20261012/operator_replay/summary.json"))
    p.add_argument("--seed", type=int, default=20261012)
    args = p.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = run(args.seed)
    source = Path(__file__).resolve().parents[1] / "experiments/numerical_qk_reuse/v31_cached_refresh.py"
    payload["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "complete", "out": str(out), "arms": len(payload["records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
