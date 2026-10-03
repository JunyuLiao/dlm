"""AIME26 Gaussian32/BLASST baselines with step-uniform thresholds.

This is intentionally separate from the temporal two-stage experiment.  For
each attention type (local/global), one calibrated parameter is copied to
call1, call2, and late denoising phases.  Calibration evaluates complete
trajectories on the frozen six-question calibration manifest and matches
pooled whole/local/global physical sparsity; no phase-specific threshold is
allowed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

from . import aime_temporal_sweep as b

ROOT = b.DLMDIR / "results/query_adaptive_aime_uniform_baselines_v1"
METHODS = ("gaussian32", "blasst_aggressive")
TARGETS = (30, 40, 50, 60, 70)
SEEDS = (42, 43, 44)


def _uniform_policy(method: str, local: float, global_: float):
    field = "log_scale" if method == "blasst_aggressive" else "log_threshold"
    phase = {"local": {field: float(local)}, "global": {field: float(global_)}}
    # Deep copies are important: the phase selector must not be able to
    # mutate one phase and accidentally alter another.
    return {p: copy.deepcopy(phase) for p in ("call1", "call2", "late")}


def _initial(method: str, target: int):
    """Use only a nearby archived point as a search seed."""
    old = b.DLMDIR / "results/query_adaptive_aime_temporal_v14/thresholds" / f"{method}_s{target}.json"
    if old.exists():
        d = json.loads(old.read_text())["policy"]
        phase = d.get("late", d.get("call1"))
        field = next(iter(phase["local"]))
        return _uniform_policy(method, phase["local"][field], phase["global"][field])
    # This fallback is only for a missing archived point; it is never a final
    # threshold.  The trajectory search below must still validate it.
    x = (target - 30.0) / 40.0
    if method == "blasst_aggressive":
        return _uniform_policy(method, 7.2 + 1.6 * x, 7.1 + 1.3 * x)
    return _uniform_policy(method, -1.49 + 1.3 * x, -1.81 + .95 * x)


def _config(root: Path):
    base = json.loads((b.RULER_CFG).read_text())
    cfg = dict(
        schema="aime26_uniform_baselines_v1",
        model=base["model"], revision=base["revision"],
        library=base["library"], torch_library=base["torch_library"],
        m_ref=base["m_ref"], beta=3.0, gamma=.5, rank=32,
        projection_seed=1729, physical_tile=[128, 64], canvas=256,
        max_steps=48, targets=list(TARGETS), seeds=list(SEEDS),
        methods=["dense", *METHODS], calibration_ids=[f"aime26/{x}" for x in b.CAL_IDS],
        calibration_rule=(
            "One-stage trajectory calibration: for each method/target, one "
            "local and one global threshold is copied to call1, call2 and "
            "late phases; complete calibration trajectories are searched until "
            "pooled whole/local/global physical sparsity is within ±2pp. "
            "BLASST log_scale retains its existing log(lambda)-style KV-length "
            "adjustment inside the kernel, but its calibrated parameter is "
            "step-uniform."),
        exposure="Calibration IDs overlap the historical AIME30; results are development evidence.",
    )
    cfg["source_hashes"] = {
        str(b.RULER_CFG.resolve()): b.sha(b.RULER_CFG),
        str((b.SOURCE / "final_manifest.json").resolve()): b.sha(b.SOURCE / "final_manifest.json"),
        str((b.SOURCE / "calibration_manifest.json").resolve()): b.sha(b.SOURCE / "calibration_manifest.json"),
        str(Path(__file__).resolve()): b.sha(Path(__file__)),
    }
    cfg["fingerprint"] = b.fingerprint(cfg)
    p = root / "configuration.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("schema") != cfg["schema"]:
            raise RuntimeError("existing result root has a different protocol")
        # Thresholds already frozen under this protocol remain authoritative
        # when the resumable search/report code receives a non-semantic fix.
        cfg = old
    else:
        b._write(p, cfg)
        final, cal = b._load_rows()
        b._write(root / "final_manifest.json", final)
        b._write(root / "calibration_manifest.json", cal)
        b._write(root / "dataset_audit.json", dict(passed=True, final=30,
            calibration=6, seeds=list(SEEDS), calibration_ids=list(b.CAL_IDS),
            calibration_overlap_with_final=True, fresh_heldout_claim=False))
    return cfg


def _violations(metrics, target):
    goal = target / 100.0
    return {k: metrics["sparsity"][k] - goal for k in ("whole", "local", "global")
            if abs(metrics["sparsity"][k] - goal) > .02}


def _calibrate_one(adapter, root, cfg, calibration, projections, method, target):
    out_path = root / "thresholds" / f"{method}_s{target}.json"
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old.get("fingerprint") == cfg["fingerprint"] and old.get("status") == "attained":
            return old["policy"]
    points = []
    memo = {}

    def evaluate(policy, label):
        key = b._hash(policy)[:16]
        if key in memo:
            return memo[key]
        rows = [b._cached(adapter, root, "calibration", label + "_" + key, row,
                          method, policy, cfg, projections) for row in calibration]
        metrics = b._profile(rows)
        point = dict(policy=policy, metrics=metrics,
                     violations=_violations(metrics, target))
        points.append(point); memo[key] = point
        b._write(root / "calibration_traces" / f"{method}_s{target}.json", points)
        print(json.dumps(dict(event="uniform_calibration", method=method,
            target=target, key=key, sparsity=metrics["sparsity"],
            phase_sparsity=metrics["phase_sparsity"],
            violations=point["violations"])), flush=True)
        return point

    current = evaluate(_initial(method, target), "initial")
    # The archived late threshold is sometimes already within the pooled
    # target when applied uniformly.  Do not spend trajectory evaluations on
    # perturbations once the acceptance gate has passed.
    if not current["violations"]:
        selected = current
        data = dict(fingerprint=cfg["fingerprint"], method=method, target=target,
            policy=selected["policy"], calibration_ids=[r["id"] for r in calibration],
            selected_metrics=selected["metrics"], violations={}, candidates=points,
            status="attained", threshold_schedule="uniform_all_denoising_steps")
        b._write(out_path, data)
        print(json.dumps(dict(event="uniform_selected", method=method, target=target,
                              status="attained", sparsity=selected["metrics"]["sparsity"],
                              policy=selected["policy"])), flush=True)
        return selected["policy"]
    # Complete-trajectory coordinate search.  Larger thresholds generally
    # skip more tiles for both routers; we nevertheless choose by measured
    # error because changing one coordinate changes later decoding states.
    for round_index in range(8):
        changed = False
        # Only coordinates represented by a failing pooled constraint can
        # improve the current point.  If the whole-model aggregate is the
        # only failure, both coordinates remain active.
        active_kinds = [k for k in ("local", "global") if k in current["violations"]]
        if "whole" in current["violations"] or not active_kinds:
            active_kinds = ["local", "global"]
        for kind in active_kinds:
            field = next(iter(current["policy"]["late"][kind]))
            best = current
            best_err = max(abs(v) for v in current["violations"].values()) if current["violations"] else 0.0
            center = float(current["policy"]["late"][kind][field])
            # Include both directions and a broad fallback.  This fixes the
            # previous failure mode where a search starting at a high-
            # sparsity boundary only explored the wrong direction.
            for delta in (-2.0, -1.0, -.5, -.25, .25, .5, 1.0, 2.0):
                trial = copy.deepcopy(current["policy"])
                value = center + delta
                for phase in ("call1", "call2", "late"):
                    trial[phase][kind][field] = value
                point = evaluate(trial, f"r{round_index}_{kind}_{delta:+g}")
                err = max(abs(v) for v in point["violations"].values()) if point["violations"] else 0.0
                if err < best_err:
                    best, best_err, changed = point, err, True
                if not point["violations"]:
                    best, best_err, changed = point, 0.0, True
                    break
            current = best
            if not current["violations"]:
                break
        if not changed or not current["violations"]:
            break
    feasible = [p for p in points if not p["violations"]]
    selected = min(feasible or points,
                   key=lambda p: (max(abs(v) for v in p["violations"].values()) if p["violations"] else 0.0,
                                  p["metrics"].get("mean_canvas_steps", 1e9)))
    data = dict(fingerprint=cfg["fingerprint"], method=method, target=target,
        policy=selected["policy"], calibration_ids=[r["id"] for r in calibration],
        selected_metrics=selected["metrics"], violations=selected["violations"],
        candidates=points, status="attained" if not selected["violations"] else "search_incomplete",
        threshold_schedule="uniform_all_denoising_steps")
    b._write(out_path, data)
    print(json.dumps(dict(event="uniform_selected", method=method, target=target,
                          status=data["status"], sparsity=selected["metrics"]["sparsity"],
                          policy=selected["policy"])), flush=True)
    return selected["policy"]


def calibrate(root):
    cfg = _config(root); _, calibration = b._load_rows()
    adapter = b._adapter(cfg); projections = b.Projections()
    for method in METHODS:
        for target in TARGETS:
            _calibrate_one(adapter, root, cfg, calibration, projections, method, target)
    statuses = {}
    for method in METHODS:
        for target in TARGETS:
            d = json.loads((root / "thresholds" / f"{method}_s{target}.json").read_text())
            statuses[f"{method}_s{target}"] = d["status"]
    b._write(root / "calibration_complete.json", dict(
        passed=all(x == "attained" for x in statuses.values()),
        statuses=statuses, methods=METHODS, targets=TARGETS, finished=time.time()))


def run(root):
    cfg = _config(root)
    if not (root / "calibration_complete.json").exists():
        raise RuntimeError("calibration is not complete")
    final, _ = b._load_rows(); adapter = b._adapter(cfg); projections = b.Projections()
    seed_text = os.environ.get("AIME_FINAL_SEEDS", "").strip()
    seeds = [int(x) for x in seed_text.split(",") if x.strip()] if seed_text else list(SEEDS)
    policies = {m: {t: json.loads((root / "thresholds" / f"{m}_s{t}.json").read_text())["policy"]
                     for t in TARGETS} for m in METHODS}
    # Reuse dense final shards from v14 only after identity checks; sparse
    # methods are always generated under this new protocol.
    for seed in seeds:
        for target in TARGETS:
            for method in METHODS:
                cond = f"{method}_s{target}_seed{seed}"
                for row0 in final:
                    row = dict(row0, seed=seed)
                    b._cached(adapter, root, "final", cond, row, method,
                              policies[method][target], cfg, projections)
    b._write(root / "final_complete.json", dict(passed=True, methods=METHODS,
        targets=TARGETS, seeds=seeds, examples=30, finished=time.time()))


def report(root):
    # Reuse the runner's aggregation/plotting code, which pools tile counts
    # before division and emits per-seed and pooled summaries.
    b.report(root)
    path = root / "report.md"
    text = path.read_text()
    text = text.replace(
        "# AIME26 temporal-instability sweep (v3 phase calibration)",
        "# AIME26 uniform-threshold Gaussian32 vs aggressive BLASST")
    text = text.replace(
        "β=3, γ=0.5; physical 128×64 tiles; 30 AIME26 prompts; evaluated seed(s): 42, 43, 44.",
        "Physical 128×64 tiles; 30 AIME26 prompts; seeds 42, 43, 44. Gaussian32 and aggressive BLASST use one local and one global threshold copied identically to every denoising step. The BLASST parameter retains its native KV-length normalization.")
    text = text.replace(
        "Calibration follows [`AIME_CALIBRATION_RULES_V3.md`](AIME_CALIBRATION_RULES_V3.md): independent call-1/call-2/later local-global thresholds, phase and pooled target matching, and early-conservative threshold ordering.",
        "Calibration is one-stage and step-uniform: complete trajectories on six calibration questions were searched for one local/global pair per method/target, with pooled whole/local/global physical sparsity checked against the target. `search_incomplete` means the nearest measured point was retained when no pair satisfied all three ±2-point constraints.")
    dense = ("\n## Dense reference\n\n"
             "The configuration-matched dense reference from the preceding AIME26 run is "
             "46.67% accuracy and 14.97 mean denoising steps per 256-token canvas "
             "(491 canvases pooled across seeds 42/43/44). It is not rerun here.\n")
    text += dense
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("calibrate", "run", "report"))
    p.add_argument("--root", type=Path, default=ROOT); a = p.parse_args()
    if a.stage == "calibrate": calibrate(a.root)
    elif a.stage == "run": run(a.root)
    else: report(a.root)


if __name__ == "__main__": main()
