"""AIME26 temporal-router sweep with explicit two-stage calibration.

Stage 1 calibrates the first two calls (where temporal sensitivity is one).
Stage 2 freezes those thresholds and calibrates the calls >=3 threshold to
the pooled whole/global/local physical sparsity target.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np

from . import aime_temporal_sweep as b

ROOT_DEFAULT = b.DLMDIR / "results" / "query_adaptive_aime_temporal_v12"

def _policy(field, early, late):
    return {p: {k: {field: float(v)} for k, v in zip(("local", "global"), pair)}
            for p, pair in (("call1", early), ("call2", early), ("late", late))}

def _eval(adapter, root, cfg, calibration, projections, method, target, policy, tag, points):
    key = b._hash(policy)[:16]
    cond = f"cal2_{method}_s{target}_{tag}_{key}"
    rows = [b._cached(adapter, root, "calibration", cond, r, method, policy, cfg, projections)
            for r in calibration]
    m = b._profile(rows)
    p = dict(policy=policy, metrics=m, key=key)
    points.append(p)
    b._write(root / "calibration_traces" / f"{method}_s{target}.json", points)
    print(json.dumps(dict(event="calibration_two_stage", method=method, target=target,
                          tag=tag, key=key, sparsity=m["sparsity"],
                          phase_sparsity=m["phase_sparsity"])), flush=True)
    return p

def _phase_error(m, target):
    # The first two calls are deliberately pooled: they share the same
    # threshold and must both remain close to the requested phase target.
    goal = target / 100.0
    vals = []
    for ph in ("call1", "call2"):
        for k in ("whole", "local", "global"):
            vals.append(abs(m["phase_sparsity"][ph][k] - goal))
    return max(vals)

def _pooled_error(m, target):
    goal = target / 100.0
    return max(abs(m["sparsity"][k] - goal) for k in ("whole", "local", "global"))

def _calibrate_temporal(adapter, root, cfg, calibration, projections, target):
    method = "temporal"
    prior = b._prior_policy(method, target)
    # v7/v11 roots used either {early,late} or {call1,call2,late}.
    # Normalize both spellings here so old thresholds remain valid seeds.
    if "call1" not in prior:
        prior = {"call1": prior.get("early", prior["late"]),
                 "call2": prior.get("early", prior["late"]),
                 "late": prior["late"]}
    field = next(iter(prior["call1"]["local"]))
    early0 = b._pair(prior, "call1")
    late0 = b._pair(prior, "late")
    points = []
    # Broad one-dimensional threshold probes. A larger log threshold skips
    # more; the range is intentionally much wider than the old v11 search.
    deltas = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
    early = list(early0)
    for coord, name in ((0, "early_local"), (1, "early_global")):
        cand = []
        for d in deltas:
            e = list(early); e[coord] = early0[coord] + d
            cand.append(_eval(adapter, root, cfg, calibration, projections, method, target,
                              _policy(field, e, late0), name, points))
        # Optimize this coordinate against the actual first-two-call rates;
        # retain the best complete trajectory, not an offline histogram.
        best = min(cand, key=lambda x: (_phase_error(x["metrics"], target),
                                         abs(x["metrics"]["phase_sparsity"]["call1"]["whole"] - target/100)))
        early[coord] = b._pair(best["policy"], "call1")[coord]
    # A small joint refinement around the selected early point catches the
    # local/global coupling caused by physical tile maxima.
    for dl in (-.6, 0.0, .6):
        for dg in (-.6, 0.0, .6):
            e = (early[0] + dl, early[1] + dg)
            _eval(adapter, root, cfg, calibration, projections, method, target,
                  _policy(field, e, late0), "early_joint", points)
    early_best = min(points, key=lambda x: (_phase_error(x["metrics"], target),
                                             _pooled_error(x["metrics"], target)))
    early = list(b._pair(early_best["policy"], "call1"))

    # Freeze early thresholds. Search late local/global independently and then
    # evaluate the small joint neighborhood using pooled physical counts.
    late = list(late0)
    for coord, name in ((0, "late_local"), (1, "late_global")):
        cand = []
        for d in deltas:
            lt = list(late); lt[coord] = late0[coord] + d
            cand.append(_eval(adapter, root, cfg, calibration, projections, method, target,
                              _policy(field, early, lt), name, points))
        best = min(cand, key=lambda x: (_pooled_error(x["metrics"], target),
                                         _phase_error(x["metrics"], target)))
        late[coord] = b._pair(best["policy"], "late")[coord]
    final_candidates = []
    for dl in (-.4, 0.0, .4):
        for dg in (-.4, 0.0, .4):
            lt = (late[0] + dl, late[1] + dg)
            final_candidates.append(_eval(adapter, root, cfg, calibration, projections, method, target,
                                          _policy(field, early, lt), "late_joint", points))
    selected = min(final_candidates, key=lambda x: (_pooled_error(x["metrics"], target),
                                                     _phase_error(x["metrics"], target),
                                                     x["metrics"].get("mean_canvas_steps", 999)))
    m = selected["metrics"]
    violations = []
    for k in ("whole", "local", "global"):
        if abs(m["sparsity"][k] - target/100) > .02: violations.append("pooled_"+k)
    for ph in ("call1", "call2"):
        for k in ("whole", "local", "global"):
            if abs(m["phase_sparsity"][ph][k] - target/100) > .02: violations.append(ph+"_"+k)
    out = dict(fingerprint=cfg["fingerprint"], method=method, target=target,
               policy=selected["policy"], calibration_ids=[r["id"] for r in calibration],
               selected_metrics=m, violations=violations, candidates=points,
               phase_target=dict(call1=target, call2=target, pooled=target),
               search="two-stage: broad complete-trajectory early solve, frozen early pooled late solve",
               status="attained" if not violations else "unattainable_under_guardrails")
    b._write(root / "thresholds" / f"{method}_s{target}.json", out)
    return selected["policy"]

def calibrate(root):
    cfg, _, calibration = b.prepare(root)
    (root / "CALIBRATION_RULES.md").write_text("""# AIME26 temporal two-stage calibration\n\n- beta = 3; gamma = 0.5; rank = 32; projection seed = 1729.\n- Physical tiles are 128 query positions by 64 KV positions; canvases are 256 positions.\n- Calls 1 and 2 use temporal sensitivity s=1. Their local and global thresholds are searched first using complete calibration generations.\n- Those early thresholds are then frozen. A separate local/global threshold pair is searched for calls 3 onward.\n- Selection minimizes pooled whole/local/global physical sparsity error against the requested target; phase-1 and phase-2 sparsity are reported separately.\n- Sparsity is summed skipped eligible tiles divided by summed eligible tiles, never an average of percentages.\n- Threshold candidates are evaluated by complete trajectories because threshold changes alter future states and denoising call counts.\n- The search uses broad log-threshold offsets (-4,-3,-2,-1,0,1,2,3,4) followed by local joint refinements.\n- A target is marked attained only when pooled whole/local/global and both early calls are within 2 percentage points; otherwise the measured closest point is marked unattainable.\n- Calibration uses six AIME IDs (2, 8, 14, 20, 23, 30), seed 42, and no final accuracy labels.\n""")
    adapter = b._adapter(cfg); projections = b.Projections()
    for target in b.TARGETS:
        existing = root / "thresholds" / f"temporal_s{target}.json"
        if existing.exists():
            print(json.dumps(dict(event="calibration_skip", target=target,
                                  reason="threshold_already_finalized")), flush=True)
            continue
        _calibrate_temporal(adapter, root, cfg, calibration, projections, target)
    # Baseline thresholds are reused from the prior temporal root if present;
    # they are not part of this temporal calibration task.
    b._write(root / "calibration_complete.json", dict(passed=True, targets=b.TARGETS,
                                                       methods=["temporal"], finished=time.time()))

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("stage", choices=("prepare","calibrate","run","report","launch-calibrate","launch-run")); ap.add_argument("--root", type=Path, default=ROOT_DEFAULT); a=ap.parse_args()
    if a.stage == "prepare": b.prepare(a.root)
    elif a.stage == "calibrate": calibrate(a.root)
    elif a.stage == "run": b.run_final(a.root)
    elif a.stage == "report": b.report(a.root)
    else: b.launch(a.root, a.stage.removeprefix("launch-"))
if __name__ == "__main__": main()
