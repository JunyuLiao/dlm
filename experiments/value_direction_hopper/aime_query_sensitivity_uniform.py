"""AIME26 evaluation of causal uniform-threshold query sensitivities.

This protocol evaluates the six causal variants implemented in
``query_adaptive.py``:

* ``T_prior``: renoising/acceptance prior plus historical flips;
* ``T_hybrid``: ``T_prior`` plus top-1 confidence drift;
* ``T_run``: consecutive-acceptance run length plus historical flips;
* ``M_prior``/``C_prior``: causal trajectory completion of margin/confidence;
* ``T_smooth``: causal temporal completion with a slower trajectory EMA;
* ``M_gate``, ``M_gate_norm``, ``C_gate``, and ``T_gate``: causal stable-run
  gates that delay relaxation until a query demonstrates stability.
* ``H_anchor``: a convex blend of completed-call trajectory uncertainty and
  processed entropy, with a 0.9 entropy weight.
* ``C_soft``: a confidence prior with a tempered trajectory contribution
  ``alpha*q + (1-alpha*q)*u_C`` (alpha 0.25).
* ``C_tail``: absolute confidence uncertainty plus a causal lift on the
  highest-uncertainty percentile tail.
* ``C_gate_soft``: the ``C_gate`` trajectory with a confidence-dependent
  relaxation floor.

Every method uses one local and one global log threshold copied to every
denoising call. Thresholds are fitted from complete trajectories on the six
historical calibration questions and then frozen for the final 30-question,
three-seed development evaluation. The calibration questions overlap the
final manifest, so this is not held-out evidence.

Run with the repository's CUDA environment, for example::

    conda run -n ljy_dlm python -m \
      experiments.value_direction_hopper.aime_query_sensitivity_uniform calibrate
    conda run -n ljy_dlm python -m \
      experiments.value_direction_hopper.aime_query_sensitivity_uniform run

Set ``AIME_TARGETS=50,60`` to limit the matched-sparsity targets. The default
is 40, 50, and 60 percent because the historical AIME failure appears in the
40--60 percent range. Set ``AIME_UNIFORM_METHODS=M_prior,C_prior,T_prior,T_smooth,T_hybrid,T_run``
to evaluate the complete causal family. ``T_smooth`` uses
``AIME_SMOOTH_TRAJECTORY_GAMMA`` (default 0.8); the other methods use
``AIME_TRAJECTORY_GAMMA`` (default 0.5).
The gated methods use ``AIME_GATE_TRAJECTORY_GAMMA`` (default 0.65),
``AIME_GATE_TAU`` (default 2.5), and ``AIME_MARGIN_SCALE`` (default 1.0).
``H_anchor`` uses ``AIME_ANCHOR_TRAJECTORY_GAMMA`` (default 0.65) and
``AIME_ANCHOR_LAMBDA`` (default 0.90).
``C_soft`` uses ``AIME_SOFT_PRIOR_ALPHA`` (default 0.25).
``C_tail`` uses ``AIME_TAIL_TAU`` (default 0.75) and
``AIME_TAIL_LAMBDA`` (default 0.5), with
``AIME_TAIL_TRAJECTORY_GAMMA`` (default 0.65).
``C_gate_soft`` uses ``AIME_SOFT_GATE_LAMBDA`` (default 0.2).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from . import aime_temporal_sweep as base
from .experiment import atomic, fingerprint, sha
from .query_adaptive import UNIFORM_METHODS


ROOT = Path(os.environ.get(
    "AIME_OUTPUT_ROOT", "/home/exouser/aime_query_sensitivity_uniform_v1"))
_method_text = os.environ.get("AIME_UNIFORM_METHODS", "T_prior,T_hybrid,T_run")
METHODS = tuple(x.strip() for x in _method_text.split(",") if x.strip())
if not METHODS or any(x not in UNIFORM_METHODS for x in METHODS):
    raise ValueError(f"AIME_UNIFORM_METHODS must be drawn from {UNIFORM_METHODS}")
TARGETS = tuple(int(x) for x in os.environ.get("AIME_TARGETS", "40,50,60").split(",") if x.strip())
SEEDS = tuple(int(x) for x in os.environ.get("AIME_SEEDS", "42,43,44").split(",") if x.strip())
CALIBRATION_IDS = (2, 8, 14, 20, 23, 30)
TRAJECTORY_GAMMA = float(os.environ.get("AIME_TRAJECTORY_GAMMA", "0.5"))
SMOOTH_TRAJECTORY_GAMMA = float(os.environ.get("AIME_SMOOTH_TRAJECTORY_GAMMA", "0.8"))
GATE_TRAJECTORY_GAMMA = float(os.environ.get("AIME_GATE_TRAJECTORY_GAMMA", "0.65"))
GATE_TAU = float(os.environ.get("AIME_GATE_TAU", "2.5"))
MARGIN_SCALE = float(os.environ.get("AIME_MARGIN_SCALE", "1.0"))
ANCHOR_TRAJECTORY_GAMMA = float(os.environ.get("AIME_ANCHOR_TRAJECTORY_GAMMA", "0.65"))
ANCHOR_LAMBDA = float(os.environ.get("AIME_ANCHOR_LAMBDA", "0.90"))
SOFT_PRIOR_ALPHA = float(os.environ.get("AIME_SOFT_PRIOR_ALPHA", "0.25"))
TAIL_TAU = float(os.environ.get("AIME_TAIL_TAU", "0.75"))
TAIL_LAMBDA = float(os.environ.get("AIME_TAIL_LAMBDA", "0.5"))
TAIL_TRAJECTORY_GAMMA = float(os.environ.get("AIME_TAIL_TRAJECTORY_GAMMA", "0.65"))
SOFT_GATE_LAMBDA = float(os.environ.get("AIME_SOFT_GATE_LAMBDA", "0.2"))
_eval_id_text = os.environ.get("AIME_EVAL_IDS", "").strip()
EVAL_IDS = tuple(x if x.startswith("aime26/") else f"aime26/{x}"
                 for x in _eval_id_text.split(",") if x.strip())


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _write(path, value):
    atomic(Path(path), value)


def _uniform_policy(local, global_):
    phase = {
        "local": {"log_threshold": float(local)},
        "global": {"log_threshold": float(global_)},
    }
    return {name: copy.deepcopy(phase) for name in ("call1", "call2", "late")}


def _load_rows():
    final, calibration = base._load_rows()
    if {int(r["source_id"]) for r in calibration} != set(CALIBRATION_IDS):
        raise ValueError("unexpected AIME calibration manifest")
    if EVAL_IDS:
        selected = set(EVAL_IDS)
        final = [row for row in final if row["id"] in selected]
        if len(final) != len(EVAL_IDS):
            raise ValueError(f"unknown or duplicate evaluation IDs: {EVAL_IDS}")
    return final, calibration


def _configuration(root):
    final, calibration = _load_rows()
    parent = json.loads(base.RULER_CFG.read_text())
    cfg = dict(
        schema="aime26_query_sensitivity_uniform_v2",
        model=parent["model"], revision=parent["revision"],
        library=parent["library"], torch_library=parent["torch_library"],
        m_ref=parent["m_ref"], beta=3., gamma=.5,
        trajectory_gamma=TRAJECTORY_GAMMA,
        smooth_trajectory_gamma=SMOOTH_TRAJECTORY_GAMMA,
        gate_trajectory_gamma=GATE_TRAJECTORY_GAMMA, gate_tau=GATE_TAU,
        anchor_trajectory_gamma=ANCHOR_TRAJECTORY_GAMMA, anchor_lambda=ANCHOR_LAMBDA,
        soft_prior_alpha=SOFT_PRIOR_ALPHA,
        tail_tau=TAIL_TAU, tail_lambda=TAIL_LAMBDA,
        tail_trajectory_gamma=TAIL_TRAJECTORY_GAMMA,
        soft_gate_lambda=SOFT_GATE_LAMBDA,
        margin_scale=MARGIN_SCALE, rank=32,
        projection_seed=1729, physical_tile=[128, 64], canvas=256,
        max_steps=48, methods=list(METHODS), targets=list(TARGETS),
        seeds=list(SEEDS),
        calibration_ids=[f"aime26/{x}" for x in CALIBRATION_IDS],
        evaluation_ids=list(EVAL_IDS) if EVAL_IDS else [r["id"] for r in final],
        calibration_rule=(
            "One local/global threshold pair shared by every denoising call; "
            "complete trajectory calibration targets pooled whole/local/global "
            "physical sparsity within ±2 percentage points, then selects the "
            "lowest mean canvas-step candidate among attained points."),
        exposure=("Calibration IDs are included in the historical AIME26 final "
                  "manifest; report is development evidence, not held-out."),
    )
    files = [Path(__file__), Path(base.__file__),
             Path(__file__).with_name("query_adaptive.py"),
             Path(__file__).with_name("query_sensitivity_uniform.py"),
             base.RULER_CFG, base.SOURCE / "final_manifest.json",
             base.SOURCE / "calibration_manifest.json"]
    cfg["source_hashes"] = {str(p.resolve()): sha(p) for p in files}
    cfg["fingerprint"] = fingerprint(cfg)
    path = Path(root) / "configuration.json"
    if path.exists():
        old = json.loads(path.read_text())
        semantic = set(cfg) - {"source_hashes", "fingerprint"}
        if any(old.get(key) != cfg[key] for key in semantic):
            raise ValueError("existing result root has different methods/settings; use a new root")
        # Only this search/report driver may change while resuming calibration.
        # Method, adapter and kernel changes require a fresh result root.
        driver = str(Path(__file__).resolve())
        changed = {path: digest for path, digest in cfg["source_hashes"].items()
                   if old.get("source_hashes", {}).get(path) != digest}
        if set(changed) - {driver}:
            raise ValueError(f"inference source changed; use a new result root: {list(changed)}")
        if changed:
            _write(Path(root) / "search_driver_revision.json", dict(
                original_source_hashes=old["source_hashes"],
                current_source_hashes=cfg["source_hashes"],
                reason="Resumable independent local/global calibration refinement and reporting",
                inference_unchanged=True))
        cfg = old
    else:
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        _write(path, cfg)
        _write(root / "final_manifest.json", final)
        _write(root / "calibration_manifest.json", calibration)
        _write(root / "dataset_audit.json", dict(
            passed=True, final=len(final), calibration=len(calibration),
            calibration_ids=list(CALIBRATION_IDS),
            calibration_overlap_with_final=True, fresh_heldout_claim=False))
    return cfg, final, calibration


def _initial_policy(method, target):
    """Use the old temporal late pair only as a search seed."""
    path = base.DLMDIR / "results/query_adaptive_aime_temporal_v14/thresholds" / f"temporal_s{target}.json"
    if path.exists():
        old = json.loads(path.read_text())["policy"]
        pair = old.get("late", old.get("early"))
        return _uniform_policy(pair["local"]["log_threshold"],
                               pair["global"]["log_threshold"])
    # Conservative fallback near the matched Gaussian32 RULER thresholds.
    x = max(0., min(1., (target - 40.) / 20.))
    return _uniform_policy(-1.15 + .45 * x, -1.55 + .45 * x)


def _metrics(rows):
    metrics = base._profile(rows)
    canvas_steps = [int(c['iterations']) for r in rows for c in r.get('canvases', [])]
    metrics['p90_canvas_steps'] = (float(base.np.quantile(canvas_steps, .9))
                                   if canvas_steps else None)
    return metrics


def _error(metrics, target):
    goal = target / 100.
    return max(abs(metrics["sparsity"][kind] - goal)
               for kind in ("whole", "local", "global"))


def _evaluate(adapter, root, cfg, calibration, projections, method, target, policy, tag, points):
    key = _hash(policy)[:16]
    for point in points:
        if point["key"] == key:
            return point
    # Cache keys are policy-based; reuse an interrupted candidate even if its
    # search tag changed. _cached still checks the full execution identity.
    parent = Path(root) / "calibration" / "calibration" / f"{method}_s{target}"
    existing = sorted(parent.glob(f"*_{key}"))
    condition = (str(existing[0].relative_to(Path(root) / "calibration"))
                 if existing else f"calibration/{method}_s{target}/{tag}_{key}")
    rows = [base._cached(adapter, root, "calibration", condition, row,
                         method, policy, cfg, projections) for row in calibration]
    metrics = _metrics(rows)
    point = dict(policy=policy, metrics=metrics, max_error=_error(metrics, target), key=key)
    points.append(point)
    _write(Path(root) / "calibration_traces" / f"{method}_s{target}.json", points)
    print(json.dumps(dict(event="uniform_candidate", method=method, target=target,
                          key=key, sparsity=metrics["sparsity"],
                          mean_steps=metrics["mean_canvas_steps"],
                          max_error=point["max_error"])), flush=True)
    return point


def _calibrate_one(adapter, root, cfg, calibration, projections, method, target):
    destination = Path(root) / "thresholds" / f"{method}_s{target}.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("fingerprint") != cfg["fingerprint"]:
            raise ValueError("threshold provenance mismatch")
        if old["status"] == "attained":
            return old
        if any((Path(root) / "final").glob(f"{method}_s{target}_seed*/*.json")):
            raise ValueError("cannot recalibrate after final evaluation")
    trace = Path(root) / "calibration_traces" / f"{method}_s{target}.json"
    points = json.loads(trace.read_text()) if trace.exists() else []
    def evaluate(policy, tag):
        return _evaluate(adapter, root, cfg, calibration, projections,
                         method, target, policy, tag, points)
    def error_objective(point):
        rates = point["metrics"]["sparsity"]
        return (point["max_error"], sum(abs(rates[k] - target/100.)
                                       for k in ("local", "global")))
    initial = cfg.get("initial_policy") or _initial_policy(method, target)
    evaluate(initial, "initial")
    if min(p["max_error"] for p in points) > .02 and len(points) < 2:
        pair = initial["late"]
        evaluate(_uniform_policy(pair["local"]["log_threshold"] + .5,
                                 pair["global"]["log_threshold"] + .5), "fast_shared_0.5")
    # Fit a local slope separately for each stratum from observed complete
    # trajectories. Slopes propose thresholds only; every candidate is then
    # verified by six new complete generations. No monotonicity or accuracy
    # assumption is needed to accept a point. Keep all prior measured points.
    import numpy as np
    for iteration in range(18):
        best = min(points, key=error_objective)
        if best["max_error"] <= .02:
            break
        pair = best["policy"]["late"]
        proposal = {}
        for kind in ("local", "global"):
            center = pair[kind]["log_threshold"]
            nearby = sorted(points, key=lambda p: abs(
                p["policy"]["late"][kind]["log_threshold"] - center))[:10]
            x = np.asarray([p["policy"]["late"][kind]["log_threshold"] for p in nearby])
            y = np.asarray([p["metrics"]["sparsity"][kind] for p in nearby])
            var = float(np.sum((x - x.mean())**2))
            slope = float(np.sum((x-x.mean())*(y-y.mean()))/var) if var > 1e-8 else .22
            slope = max(.08, min(.5, slope))
            delta = max(-.4, min(.4, (target/100. - best["metrics"]["sparsity"][kind])/slope))
            proposal[kind] = round(center + delta, 6)
        policy = _uniform_policy(proposal["local"], proposal["global"])
        if any(_hash(policy)[:16] == p["key"] for p in points):
            # A local response may be irregular. Explore a smaller coordinate
            # neighborhood instead of repeating or terminating at a coarse grid.
            alternatives = []
            for step in (.10, .05, .025, .20):
                for kind in ("local", "global"):
                    for sign in (-1, 1):
                        trial = {k: pair[k]["log_threshold"] for k in pair}
                        trial[kind] += sign*step
                        candidate = _uniform_policy(trial["local"], trial["global"])
                        if all(_hash(candidate)[:16] != p["key"] for p in points):
                            alternatives.append(candidate)
                if alternatives:
                    break
            if not alternatives:
                break
            policy = alternatives[0]
        evaluate(policy, f"refine_{iteration}")
    feasible = [point for point in points if point["max_error"] <= .02]
    if feasible:
        # Once aggregate sparsity is attained, prefer the least expensive
        # denoising trajectory. This keeps calibration from selecting a
        # needlessly slow point merely because its sparsity error is smaller.
        selected = min(feasible, key=lambda point: (
            point["metrics"].get("mean_canvas_steps", float("inf")),
            point["max_error"], error_objective(point)[1]))
    else:
        selected = min(points, key=error_objective)
    result = dict(fingerprint=cfg["fingerprint"], method=method, target=target,
                  policy=selected["policy"], selected_metrics=selected["metrics"],
                  max_error=selected["max_error"],
                  status="attained" if selected["max_error"] <= .02 else "search_incomplete",
                  calibration_ids=[r["id"] for r in calibration], candidates=points,
                  threshold_schedule="uniform_all_denoising_steps",
                  search="independent local/global slope proposals, complete trajectory verification",
                  search_driver_sha256=sha(Path(__file__)))
    _write(destination, result)
    return result


def calibrate(root=ROOT):
    if not base.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment; attach the H100 before calibration")
    cfg, _, calibration = _configuration(Path(root))
    adapter = base._adapter(cfg); projections = base.Projections()
    for method in METHODS:
        for target in TARGETS:
            _calibrate_one(adapter, Path(root), cfg, calibration, projections, method, target)
    statuses = {}
    for method in METHODS:
        for target in TARGETS:
            d = json.loads((Path(root) / "thresholds" / f"{method}_s{target}.json").read_text())
            statuses[f"{method}_s{target}"] = d["status"]
    _write(Path(root) / "calibration_complete.json", dict(
        passed=all(x == "attained" for x in statuses.values()), statuses=statuses,
        finished=time.time()))


def run(root=ROOT):
    if not base.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment; attach the H100 before evaluation")
    root = Path(root)
    cfg, final, _ = _configuration(root)
    complete = root / "calibration_complete.json"
    if not complete.exists() or not json.loads(complete.read_text()).get("passed"):
        raise RuntimeError("all methods must attain calibration before final evaluation")
    adapter = base._adapter(cfg); projections = base.Projections()
    policies = {(method, target): json.loads(
        (root / "thresholds" / f"{method}_s{target}.json").read_text())["policy"]
        for method in METHODS for target in TARGETS}
    for seed in SEEDS:
        for method in METHODS:
            for target in TARGETS:
                condition = f"{method}_s{target}_seed{seed}"
                for source in final:
                    row = dict(source, seed=seed)
                    base._cached(adapter, root, "final", condition, row, method,
                                 policies[(method, target)], cfg, projections)
    _write(root / "final_complete.json", dict(passed=True, methods=list(METHODS),
                                               targets=list(TARGETS), seeds=list(SEEDS),
                                               finished=time.time()))


def _read_final_rows(root):
    rows = []
    for method in METHODS:
        for target in TARGETS:
            for seed in SEEDS:
                condition = f"{method}_s{target}_seed{seed}"
                directory = Path(root) / "final" / condition
                values = [json.loads(path.read_text()) for path in directory.glob("*.json")]
                if len(values) != 30:
                    raise ValueError(f"incomplete {condition}: {len(values)} rows")
                metrics = _metrics(values)
                rows.append(dict(condition=condition, method=method, target=target,
                                 seed=seed, **{k: metrics[k] for k in (
                                     "accuracy", "sparsity", "mean_canvas_steps",
                                     "median_canvas_steps", "p90_canvas_steps", "total_canvases",
                                     "cap_canvases", "phase_sparsity")}))
    return rows


def _read_question_disagreement(root):
    """Return per-question correctness variation across the frozen seeds."""
    output = []
    for method in METHODS:
        for target in TARGETS:
            by_question = {}
            for seed in SEEDS:
                directory = Path(root) / "final" / f"{method}_s{target}_seed{seed}"
                values = [json.loads(path.read_text()) for path in directory.glob("*.json")]
                if len(values) != 30:
                    raise ValueError(f"incomplete {method}_s{target}_seed{seed}: {len(values)} rows")
                for value in values:
                    by_question.setdefault(value["id"], {})[seed] = int(value["score"])
            for question_id, scores in sorted(by_question.items()):
                ordered = [scores[seed] for seed in SEEDS]
                output.append(dict(method=method, target=target, id=question_id,
                                   correct_seed_count=sum(ordered),
                                   seed_disagreement=int(len(set(ordered)) > 1),
                                   **{f"correct_seed{seed}": scores[seed] for seed in SEEDS}))
    return output


def report(root=ROOT):
    root = Path(root); rows = _read_final_rows(root)
    fields = ["condition", "method", "target", "seed", "accuracy",
              "sparsity", "mean_canvas_steps", "median_canvas_steps",
              "p90_canvas_steps", "total_canvases", "cap_canvases", "phase_sparsity"]
    import csv
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow(row)
    disagreements = _read_question_disagreement(root)
    with (root / "per_question_disagreement.csv").open("w", newline="") as stream:
        disagreement_fields = ["method", "target", "id", "correct_seed_count",
                               "seed_disagreement"] + [f"correct_seed{seed}" for seed in SEEDS]
        writer = csv.DictWriter(stream, fieldnames=disagreement_fields)
        writer.writeheader()
        for row in disagreements: writer.writerow(row)

    pooled = []
    for method in METHODS:
        for target in TARGETS:
            values = []
            for seed in SEEDS:
                directory = root / "final" / f"{method}_s{target}_seed{seed}"
                values.extend(json.loads(path.read_text()) for path in directory.glob("*.json"))
            metrics = _metrics(values)
            seed_accuracy = [row["accuracy"] for row in rows
                             if row["method"] == method and row["target"] == target]
            pooled.append(dict(method=method, target=target,
                accuracy_std=(statistics.stdev(seed_accuracy)
                              if len(seed_accuracy) > 1 else None), **{
                k: metrics[k] for k in ("accuracy", "sparsity", "mean_canvas_steps",
                                         "median_canvas_steps", "p90_canvas_steps",
                                         "total_canvases", "cap_canvases", "phase_sparsity")}))
    pooled_fields = ["method", "target", "accuracy", "accuracy_std", "sparsity",
                     "mean_canvas_steps", "median_canvas_steps", "p90_canvas_steps",
                     "total_canvases", "cap_canvases", "phase_sparsity"]
    with (root / "pooled_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pooled_fields); writer.writeheader()
        for row in pooled: writer.writerow(row)
    lines = ["# AIME26 causal uniform query sensitivity", "",
             "All methods use one local/global threshold pair copied to every denoising call. "
             "Calibration questions overlap the historical AIME26 manifest; these are development results.", "",
             "| Condition | Accuracy | O/G/L sparsity | Mean steps/canvas | P90 | Caps |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        s = row["sparsity"]
        lines.append(f"| {row['condition']} | {100*row['accuracy']:.2f}% | "
                     f"{100*s['whole']:.2f}% / {100*s['global']:.2f}% / {100*s['local']:.2f}% | "
                     f"{row['mean_canvas_steps']:.2f} | {row['p90_canvas_steps']:.2f} | {row['cap_canvases']} |")
    lines += ["", "## Pooled across seeds", "",
              "| Method | Accuracy ± seed SD | O/G/L sparsity | Mean steps/canvas | P90 | Caps |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in pooled:
        s = row["sparsity"]
        lines.append(f"| {row['method']}_s{row['target']} | {100*row['accuracy']:.2f}% ± "
                     + (f"{100*row['accuracy_std']:.2f}%" if row["accuracy_std"] is not None else "N/A (one seed)") + " | "
                     f"{100*s['whole']:.2f}% / {100*s['global']:.2f}% / {100*s['local']:.2f}% | "
                     f"{row['mean_canvas_steps']:.2f} | {row['p90_canvas_steps']:.2f} | {row['cap_canvases']} |")

    lines += ["", "## Pooled phase sparsity", "",
              "| Method | Call 1 O/G/L | Call 2 O/G/L | Later O/G/L |",
              "|---|---:|---:|---:|"]
    for row in pooled:
        phases = row["phase_sparsity"]
        fmt = lambda phase: (f"{100*phase['whole']:.2f}% / {100*phase['global']:.2f}% / "
                             f"{100*phase['local']:.2f}%")
        lines.append(f"| {row['method']}_s{row['target']} | {fmt(phases['call1'])} | "
                     f"{fmt(phases['call2'])} | {fmt(phases['late'])} |")

    lines += ["", "## Per-question seed disagreement", "",
              "| Method | Questions with disagreement | Rate | Mean correct seeds/question |",
              "|---|---:|---:|---:|"]
    for method in METHODS:
        for target in TARGETS:
            subset = [x for x in disagreements if x["method"] == method and x["target"] == target]
            count = sum(x["seed_disagreement"] for x in subset)
            mean_correct = statistics.mean(x["correct_seed_count"] for x in subset)
            lines.append(f"| {method}_s{target} | {count}/{len(subset)} | "
                         f"{100*count/max(1,len(subset)):.1f}% | {mean_correct:.2f} |")

    lines += ["", "## Calibration", "",
              "| Method | Target | Local log τ | Global log τ | Calibration max error | Status |",
              "|---|---:|---:|---:|---:|---|"]
    for method in METHODS:
        for target in TARGETS:
            path = root / "thresholds" / f"{method}_s{target}.json"
            if not path.exists():
                continue
            threshold = json.loads(path.read_text())
            late = threshold["policy"]["late"]
            lines.append(f"| {method} | {target} | {late['local']['log_threshold']:.6f} | "
                         f"{late['global']['log_threshold']:.6f} | {100*threshold['max_error']:.2f} pp | "
                         f"{threshold['status']} |")

    lines += ["", "The matched historical dense, Gaussian32, and original temporal controls are in "
              "`results/query_adaptive_aime_temporal_v14/report.md` and its `pooled_summary.csv`. "]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == "prepare":
        _configuration(args.root)
    elif args.stage == "calibrate":
        calibrate(args.root)
    elif args.stage == "run":
        run(args.root)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
