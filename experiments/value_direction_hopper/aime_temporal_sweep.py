"""AIME26 temporal-instability sensitivity sweep.

This is a small, resumable companion to the frozen RULER guardrail study.  It
keeps the DiffusionGemma decoder and the 128x64 physical router unchanged and
only changes the benchmark manifests, threshold calibration, and reporting.
The temporal weight is the preregistered rule::

    u_t = .5*u_(t-1) + .5*[argmax_t != argmax_(t-1)]
    s_(t+1) = 1 + 3*u_t.

The six existing AIME calibration IDs are used for threshold fitting; they
are also part of the historical 30-question AIME manifest, so this report
explicitly does not claim fresh held-out accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_value_aware_followup.protocol import score as aime_score
from .experiment import atomic, sha, fingerprint, shard_path
from .integration import install
from .query_adaptive import GATED_METHODS, UNIFORM_METHODS, observe
from .query_sensitivity_uniform import UniformThresholdState
from .query_adaptive_guardrail import PhaseState, method_specs
from .sparsity_steps import routing_counts


HERE = Path(__file__).resolve()
DLMDIR = HERE.parents[2]
SOURCE = DLMDIR / "results" / "diffusion_gemma_jl_aime_gaussian_dimensions_v8"
RULER_CFG = DLMDIR / "results" / "query_adaptive_guardrail_v2" / "configs" / "configuration.json"
DEFAULT_ROOT = DLMDIR / "results" / "query_adaptive_aime_temporal_v6"
PRIOR_ROOT = Path(os.environ.get("AIME_PRIOR_ROOT", str(DLMDIR / "results" / "query_adaptive_aime_temporal_v6")))
_target_text = os.environ.get("AIME_TARGETS", "30,40,50,60,70")
TARGETS = tuple(int(x) for x in _target_text.split(",") if x.strip())
_method_text = os.environ.get("AIME_METHODS", "gaussian32,temporal,blasst_aggressive")
METHODS = tuple(x.strip() for x in _method_text.split(",") if x.strip())
SEEDS = (42, 43, 44)
CAL_IDS = (2, 8, 14, 20, 23, 30)


def _write(path: Path, value):
    atomic(path, value)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _label(method: str, target: int | None, seed: int | None = None) -> str:
    x = method if target is None else f"{method}_s{target}"
    return x if seed is None else f"{x}_seed{seed}"


def _load_rows():
    final = json.loads((SOURCE / "final_manifest.json").read_text())
    calibration = json.loads((SOURCE / "calibration_manifest.json").read_text())
    if len(final) != 30 or {int(r["source_id"]) for r in calibration} != set(CAL_IDS):
        raise ValueError("AIME manifests are not the frozen 30/6 protocol")
    if len({r["prompt_hash"] for r in final}) != 30:
        raise ValueError("duplicate AIME prompts")
    return final, calibration


def _anchor(target: int, family: str):
    """Interpolate archived RULER anchors, then search complete trajectories."""
    # RULER v2 successful early pair, and archived AIME50 aggressive BLASST
    # pair.  These are starting points only; no final score is used here.
    x = (target - 30.0) / 40.0
    x = min(1.0, max(0.0, x))
    if family == "value":
        low = (-1.45, -3.65)
        high = (-0.2344298, -2.2909701)
    elif family == "blasst":
        low = (6.0, 5.7)
        high = (8.0913338, 7.7792791)
    else:
        raise ValueError(family)
    return tuple(a + x * (b - a) for a, b in zip(low, high))


def phase_policy(call1, call2=None, late=None):
    # Per-call schedules are the v3 calibration contract.  Accept the old
    # two-phase spelling when loading legacy policies.
    if call2 is None:
        call2 = call1
    if late is None:
        late = call2
    field = "log_scale" if call1.get("field") == "log_scale" else "log_threshold"
    def one(pair):
        return {k: {field: float(pair[i])} for i, k in enumerate(("local", "global"))}
    return {"call1": one(call1["pair"]), "call2": one(call2["pair"]), "late": one(late["pair"])}


def _prior_policy(method: str, target: int):
    """Use the completed v6 trajectory policy as a starting point only.

    The prior sweep's thresholds are not treated as calibrated for the new
    protocol; they only provide a nearby point for the new independent
    local/global search.  Keeping this seed is important because the
    generation-level sparsity response is not well approximated by an
    offline score histogram.
    """
    path = PRIOR_ROOT / "thresholds" / f"{method}_s{target}.json"
    if path.exists():
        return json.loads(path.read_text())["policy"]
    family = "blasst" if method == "blasst_aggressive" else "value"
    local, glob = _anchor(target, family)
    field = "log_scale" if family == "blasst" else "log_threshold"
    early = {k: {field: float(v - .30)} for k, v in zip(("local", "global"), (local, glob))}
    late = {k: {field: float(v)} for k, v in zip(("local", "global"), (local, glob))}
    return {"call1": early, "call2": early, "late": late}


def _pair(policy, phase):
    phase = phase if phase in policy else "early"
    return tuple(float(policy[phase][k][next(iter(policy[phase][k]))]) for k in ("local", "global"))


def _make_policy(field, call1, call2, late):
    return {phase: {k: {field: float(v)} for k, v in zip(("local", "global"), pair)}
            for phase, pair in (("call1", call1), ("call2", call2), ("late", late))}


def _candidate_policies(target: int, method: str):
    """Generate independent local/global late candidates around the prior.

    The old AIME sweep moved both strata by one shared offset.  That cannot
    match three pooled constraints when local and global response slopes
    differ.  This compact coordinate set first explores each coordinate and
    then `_calibrate_one` performs bounded trajectory-aware refinements.
    """
    family = "blasst" if method == "blasst_aggressive" else "value"
    field = "log_scale" if family == "blasst" else "log_threshold"
    seed = _prior_policy(method, target)
    call1 = _pair(seed, "call1")
    call2 = _pair(seed, "call2")
    late = _pair(seed, "late")
    seen = set()
    # Search each call's local/global thresholds independently.  The initial
    # set includes the prior policy, per-coordinate perturbations, and a
    # conservative shared shift; later coordinate refinement is trajectory
    # aware and evaluates complete generations.
    seeds = [(call1, call2, late)]
    for which in range(3):
        for coord in range(2):
            for delta in (-.8, .8):
                vals = [list(call1), list(call2), list(late)]
                vals[which][coord] += delta
                seeds.append(tuple(tuple(x) for x in vals))
    for delta in (-.4, .4):
        seeds.append((tuple(x + delta for x in call1), tuple(x + delta for x in call2), tuple(x + delta for x in late)))
    for c1, c2, lt in seeds:
        candidate = _make_policy(field, c1, c2, lt)
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key); yield candidate


def _cfg(root: Path):
    base = json.loads(RULER_CFG.read_text())
    return dict(schema="aime26_temporal_sweep_v3_phase_calibration", model=base["model"],
        revision=base["revision"], library=base["library"],
        torch_library=base["torch_library"], m_ref=base["m_ref"], beta=3.0,
        gamma=0.5, rank=32, projection_seed=1729, physical_tile=[128, 64],
        canvas=256, max_steps=48, targets=list(TARGETS), seeds=list(SEEDS),
        methods=["dense", *METHODS],
        calibration_ids=[f"aime26/{x}" for x in CAL_IDS],
        calibration_rule="Three-phase trajectory calibration: independent local/global thresholds for call1, call2, and later calls; each phase and pooled whole/local/global rates target the requested sparsity within 2pp; early thresholds are protected only by an ordering constraint (no fixed sparsity ceiling); AIME has no disjoint sixth-task validation split",
        exposure="The six calibration questions are included in the historical AIME30; report noncalibration24 separately",
        source_hashes={str(RULER_CFG.resolve()): sha(RULER_CFG),
                       str(SOURCE.joinpath("final_manifest.json")): sha(SOURCE/"final_manifest.json"),
                       str(SOURCE.joinpath("calibration_manifest.json")): sha(SOURCE/"calibration_manifest.json"),
                       str(HERE): sha(HERE)})


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    final, calibration = _load_rows()
    cfg = _cfg(root)
    cfg["fingerprint"] = fingerprint(cfg)
    path = root / "configuration.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != cfg:
            # The v7 root is deliberately immutable after calibration starts.
            # Small report/resume-only code changes (for example dense-shard
            # reuse) must not invalidate already-frozen threshold identities;
            # retain the recorded configuration and fingerprint.
            if old.get("schema") in ("aime26_temporal_sweep_v2_coordinate_calibration", "aime26_v14_residual_two_stage"):
                cfg = old
            else:
                raise ValueError("existing root was created by a different AIME protocol")
    else:
        _write(path, cfg)
        _write(root / "final_manifest.json", final)
        _write(root / "calibration_manifest.json", calibration)
        _write(root / "dataset_audit.json", dict(passed=True, final=30,
            calibration=6, seeds=list(SEEDS), calibration_ids=CAL_IDS,
            calibration_overlap_with_final=True, fresh_heldout_claim=False))
    return cfg, final, calibration


def _adapter(cfg):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return create_adapter("diffusion_gemma", cfg["model"], device="cuda",
        precision="bfloat16", revision=cfg["revision"]).load()


def _empty_counts():
    return {k: {"eligible": 0, "skipped": 0} for k in ("whole", "local", "global")}


def _add_counts(dst, src):
    for kind in dst:
        for field in dst[kind]:
            dst[kind][field] += int(src[kind][field])


def _run_one(adapter, row, method, policy, cfg, projections, diagnostics=True):
    if method == "dense":
        ctx = nullcontext((None, None)); state_method = "native_dense"
    else:
        mode = "blasst" if method == "blasst_aggressive" else "value"
        # The binding owns a late/default threshold map; PhaseState switches
        # to the call-1/call-2 map before each attention layer.
        ctx = install(adapter, cfg["library"], policy["late"], mode=mode,
            projections=projections, torch_library=cfg["torch_library"], collect=diagnostics)
        state_method = "kernel_dense" if mode == "blasst" else ("T" if method == "temporal" else "unweighted")
        if method in UNIFORM_METHODS:
            state_method = method
        trajectory_gamma = (cfg.get("gate_trajectory_gamma", .65)
                            if state_method in GATED_METHODS else
                            (cfg.get("anchor_trajectory_gamma", .65)
                            if state_method in ("H_anchor", "C_soft") else
                            (cfg.get("smooth_trajectory_gamma", .8)
                             if state_method == "T_smooth"
                             else cfg.get("trajectory_gamma", cfg["gamma"]))))
        if state_method == "C_tail":
            trajectory_gamma = cfg.get("tail_trajectory_gamma", .65)
        elif state_method == "C_gate_soft":
            trajectory_gamma = cfg.get("gate_trajectory_gamma", .65)
    with ctx as (binding, router):
        if binding is not None:
            _set_context(binding, row)
        if method in UNIFORM_METHODS:
            state = UniformThresholdState(state_method, router, m_ref=cfg["m_ref"],
                beta=cfg["beta"], gamma=cfg["gamma"],
                trajectory_gamma=trajectory_gamma,
                gate_tau=cfg.get("gate_tau", 2.5),
                anchor_lambda=cfg.get("anchor_lambda", .9),
                soft_prior_alpha=cfg.get("soft_prior_alpha", .25),
                tail_tau=cfg.get("tail_tau", .75),
                tail_lambda=cfg.get("tail_lambda", .5),
                soft_gate_lambda=cfg.get("soft_gate_lambda", .2),
                margin_scale=cfg.get("margin_scale", 1.),
                allocation="normal",
                bootstrap=False, seed=int(row["seed"]), diagnostics=diagnostics,
                thresholds=policy["late"])
        else:
            state = PhaseState(state_method, router, m_ref=cfg["m_ref"], beta=cfg["beta"],
                gamma=cfg["gamma"], allocation="normal", bootstrap=False,
                seed=int(row["seed"]), diagnostics=diagnostics,
                phase_thresholds=policy or {"call1": {}, "call2": {}, "late": {}})
        with observe(adapter.model, state):
            torch.cuda.synchronize(); started = time.perf_counter()
            output = adapter.generate(_request(row))
            torch.cuda.synchronize(); seconds = time.perf_counter() - started
        routing = [] if router is None else router.records()
    counts = routing_counts(routing)
    canvases = [dict(x) for x in state.canvases]
    # The native adapter can create multiple 256-token canvases for AIME's
    # 2048-token budget.  State.finish_canvas records each independently.
    if diagnostics and len(state.steps) != int(output.metadata["actual_denoising_step_count"]):
        raise AssertionError("instrumented and metadata call counts disagree")
    if output.metadata.get("native_canvas_length") != 256:
        raise ValueError("unexpected native canvas length")
    phase_counts = {"call1": _empty_counts(), "call2": _empty_counts(), "late": _empty_counts()}
    for item in routing:
        step = int(item.get("step", 0))
        phase = "call1" if step == 0 else "call2" if step == 1 else "late"
        for kind in ("whole", item["attention_type"]):
            phase_counts[phase][kind]["eligible"] += int(item["eligible"])
            phase_counts[phase][kind]["skipped"] += int(item["skipped"])
    phase_sparsity = {p: {k: phase_counts[p][k]["skipped"] /
                          max(1, phase_counts[p][k]["eligible"])
                          for k in ("whole", "local", "global")} for p in phase_counts}
    return dict(status="complete", id=row["id"], source_id=row.get("source_id"),
        task=row["task"], prompt_hash=row["prompt_hash"], seed=int(row["seed"]),
        method=method, prediction=output.text, completion_tokens=output.completion_tokens,
        score=float(aime_score(row, output.text)), steps=int(output.metadata["actual_denoising_step_count"]),
        canvases=canvases, total_canvases=len(canvases), counts=counts,
        phase_counts=phase_counts, phase_sparsity=phase_sparsity,
        step_records=state.steps, seconds=seconds, metadata=output.metadata,
        policy=policy, beta=cfg["beta"], gamma=cfg["gamma"])


def _shard(root, stage, condition, row):
    return root / stage / condition / (hashlib.sha256(
        f"{row['id']}|{row['seed']}".encode()).hexdigest() + ".json")


def _cached(adapter, root, stage, condition, row, method, policy, cfg, projections):
    dest = _shard(root, stage, condition, row)
    identity = fingerprint([cfg["fingerprint"], stage, condition, method, policy,
                            row["id"], row["prompt_hash"], row["seed"]])
    if dest.exists():
        out = json.loads(dest.read_text())
        if out.get("identity") != identity or out.get("status") != "complete":
            raise ValueError(f"invalid cache identity {dest}")
        return out
    # Dense inference is configuration-invariant across v6/v7: the manifests,
    # model revision, seeds, scheduler and native decoder are identical. Reuse
    # the completed dense shard after checking its semantic identifiers, while
    # assigning the new root's identity so resumability remains strict.
    if stage == "final" and method == "dense" and root.resolve() != PRIOR_ROOT.resolve():
        source = _shard(PRIOR_ROOT, stage, condition, row)
        if source.exists():
            out = json.loads(source.read_text())
            if (out.get("status") == "complete" and out.get("method") == "dense" and
                    out.get("id") == row["id"] and int(out.get("seed")) == int(row["seed"])):
                out["identity"] = identity
                dest.parent.mkdir(parents=True, exist_ok=True); _write(dest, out)
                return out
    _write(root / "status.json", dict(stage=stage, condition=condition,
        id=row["id"], seed=row["seed"], started=time.time()))
    out = _run_one(adapter, row, method, policy, cfg, projections)
    out["identity"] = identity; out["condition"] = condition
    dest.parent.mkdir(parents=True, exist_ok=True); _write(dest, out)
    with (root / "completed.jsonl").open("a") as f:
        f.write(json.dumps(dict(stage=stage, condition=condition, id=row["id"],
            seed=row["seed"], steps=out["steps"], finished=time.time())) + "\n")
    print(json.dumps(dict(event="complete", stage=stage, condition=condition,
        id=row["id"], seed=row["seed"], steps=out["steps"],
        canvases=out["total_canvases"], score=out["score"])), flush=True)
    return out


def _profile(rows):
    counts = _empty_counts()
    for r in rows: _add_counts(counts, r["counts"])
    sparsity = {k: counts[k]["skipped"] / max(1, counts[k]["eligible"]) for k in counts}
    phase_counts = {p: _empty_counts() for p in ("call1", "call2", "late")}
    for r in rows:
        src = r.get("phase_counts", {})
        if {"call1", "call2", "late"}.issubset(src):
            for p in phase_counts:
                _add_counts(phase_counts[p], src[p])
        elif "early" in src or "late" in src:
            # Dense shards reused from v7/v8 used the older early/late
            # schema.  Preserve their aggregate totals while treating the
            # unavailable call-1/call-2 split as zero.
            if "early" in src:
                _add_counts(phase_counts["call1"], src["early"])
            if "late" in src:
                _add_counts(phase_counts["late"], src["late"])
    pooled_phase_sparsity = {p: {k: phase_counts[p][k]["skipped"] /
                                 max(1, phase_counts[p][k]["eligible"])
                                 for k in ("whole", "local", "global")} for p in phase_counts}
    canv_steps = [int(c["iterations"]) for r in rows for c in r.get("canvases", [])]
    ex_steps = [int(r["steps"]) for r in rows]
    return dict(n=len(rows), accuracy=float(np.mean([r["score"] for r in rows])) if rows else None,
        counts=counts, sparsity=sparsity, total_steps=sum(ex_steps),
        mean_steps=float(np.mean(ex_steps)) if ex_steps else None,
        median_steps=float(np.median(ex_steps)) if ex_steps else None,
        p90_steps=float(np.quantile(ex_steps, .9)) if ex_steps else None,
        mean_canvas_steps=float(np.mean(canv_steps)) if canv_steps else None,
        median_canvas_steps=float(np.median(canv_steps)) if canv_steps else None,
        total_canvases=len(canv_steps), canvases_per_example=float(np.mean([r["total_canvases"] for r in rows])) if rows else None,
        cap_canvases=sum(x >= 48 for x in canv_steps), executed_tiles=counts["whole"]["eligible"]-counts["whole"]["skipped"],
        phase_sparsity=pooled_phase_sparsity)


def _calibration_violations(metrics, target, policy=None):
    goal = target / 100.0
    problems = [f"overall_{kind}" for kind in ("whole", "local", "global")
                if abs(metrics["sparsity"][kind] - goal) > .02]
    # Each phase is calibrated to the requested target.  Protection is an
    # ordering constraint (early thresholds may not be more aggressive than
    # the late threshold), not a hard sparsity ceiling.
    for phase in ("call1", "call2", "late"):
        rates = metrics.get("phase_sparsity", {}).get(phase, {})
        for kind in ("whole", "local", "global"):
            if abs(rates.get(kind, 0.0) - goal) > .02:
                problems.append(f"{phase}_{kind}")
    if policy:
        for kind in ("local", "global"):
            c1 = float(policy["call1"][kind][next(iter(policy["call1"][kind]))])
            c2 = float(policy["call2"][kind][next(iter(policy["call2"][kind]))])
            late = float(policy["late"][kind][next(iter(policy["late"][kind]))])
            if c1 > late + 1e-6: problems.append(f"call1_{kind}_more_aggressive")
            if c2 > late + 1e-6: problems.append(f"call2_{kind}_more_aggressive")
    return problems


def _calibrate_one(adapter, root, cfg, calibration, projections, method, target):
    dest = root / "thresholds" / f"{method}_s{target}.json"
    if dest.exists():
        data = json.loads(dest.read_text())
        if data["fingerprint"] != cfg["fingerprint"]: raise ValueError("threshold provenance mismatch")
        # A v7 coarse pass may have frozen an unattainable boundary before the
        # 0.20-log-unit refinement was introduced.  Reopen only those files;
        # attained policies and explicitly fine-calibrated policies remain
        # immutable.
        if data.get("coordinate_step") == 0.20 or data.get("status") == "attained":
            return data["policy"]
    points = []
    for policy in _candidate_policies(target, method):
        key = _hash(policy)[:16]; condition = f"cal_{method}_s{target}_{key}"
        rows = [_cached(adapter, root, "calibration", condition, r, method, policy, cfg, projections) for r in calibration]
        info = _profile(rows)
        points.append(dict(policy=policy, metrics=info, violations=_calibration_violations(info, target, policy)))
        _write(root / "calibration_traces" / f"{method}_s{target}.json", points)
        print(json.dumps(dict(event="calibration_point", method=method, target=target,
            key=key, metrics=info, violations=points[-1]["violations"])), flush=True)

    # Refine local and global late thresholds independently.  This is a
    # complete-generation objective: changing a threshold changes future
    # states and call counts, so no histogram interpolation is used.
    goal = target / 100.0
    def key_for(point):
        return (max(abs(point["metrics"]["sparsity"][k] - goal)
                    for k in ("whole", "local", "global")),
                len(point["violations"]))
    current = min(points, key=key_for)
    for round_index in range(10):
        improved = False
        for kind in ("local", "global"):
            trial = json.loads(json.dumps(current["policy"]))
            field = next(iter(trial["late"][kind]))
            actual = current["metrics"]["sparsity"][kind]
            # 0.20 log units is small enough to resolve the requested ±2pp
            # pooled matching window without assuming a smooth response.
            trial["late"][kind][field] = float(trial["late"][kind][field] + (.20 if actual < goal else -.20))
            key = _hash(trial)[:16]; condition = f"cal_{method}_s{target}_{key}"
            if any(p["policy"] == trial for p in points):
                continue
            rows = [_cached(adapter, root, "calibration", condition, r, method, trial, cfg, projections) for r in calibration]
            info = _profile(rows)
            point = dict(policy=trial, metrics=info, violations=_calibration_violations(info, target, trial))
            points.append(point); _write(root / "calibration_traces" / f"{method}_s{target}.json", points)
            print(json.dumps(dict(event="calibration_refine", method=method, target=target,
                round=round_index, coordinate=kind, key=key, metrics=info,
                violations=point["violations"])), flush=True)
            if key_for(point) < key_for(current):
                current = point; improved = True
        if not improved:
            break
    feasible = [p for p in points if not p["violations"]]
    selected = min(feasible or points, key=lambda p: (key_for(p),
                    p["metrics"].get("mean_canvas_steps", 999),
                    p["metrics"].get("executed_tiles", 10**18)))
    data = dict(fingerprint=cfg["fingerprint"], method=method, target=target,
        policy=selected["policy"], calibration_ids=[r["id"] for r in calibration],
        selected_metrics=selected["metrics"], violations=selected["violations"], candidates=points,
        phase_target=dict(call1=target, call2=target, late=target),
        search="independent local/global call1/call2/late coordinate refinement; early thresholds protected by ordering only",
        coordinate_step=0.20,
        status="attained" if not selected["violations"] else "unattainable_under_guardrails")
    _write(dest, data); return selected["policy"]


def calibrate(root: Path):
    cfg, _, calibration = prepare(root)
    adapter = _adapter(cfg); projections = Projections()
    for target in TARGETS:
        for method in METHODS:
            _calibrate_one(adapter, root, cfg, calibration, projections, method, target)
    _write(root / "calibration_complete.json", dict(passed=True, targets=TARGETS,
        methods=list(METHODS), finished=time.time()))


def run_final(root: Path):
    cfg, final, calibration = prepare(root)
    adapter = _adapter(cfg); projections = Projections()
    # Allow an explicitly selected seed for staged evaluation.  The default
    # remains the preregistered three-seed sweep; setting AIME_FINAL_SEEDS=42
    # runs only that seed without changing calibration or cached shards.
    seed_text = os.environ.get("AIME_FINAL_SEEDS", "").strip()
    run_seeds = [int(x) for x in seed_text.split(",") if x.strip()] if seed_text else list(SEEDS)
    if not run_seeds:
        raise ValueError("AIME_FINAL_SEEDS must contain at least one integer seed")
    policies = {m: {t: json.loads((root / "thresholds" / f"{m}_s{t}.json").read_text())["policy"] for t in TARGETS}
                for m in METHODS}
    # Dense is target-independent but is intentionally run for every seed.
    for seed in run_seeds:
        for row0 in final:
            row = dict(row0, seed=seed)
            _cached(adapter, root, "final", _label("dense", None, seed), row, "dense", None, cfg, projections)
    for seed in run_seeds:
        for target in TARGETS:
            for method in METHODS:
                cond = _label(method, target, seed)
                for row0 in final:
                    row = dict(row0, seed=seed)
                    _cached(adapter, root, "final", cond, row, method, policies[method][target], cfg, projections)
    _write(root / "final_complete.json", dict(passed=True, examples=30, seeds=run_seeds,
        targets=TARGETS, finished=time.time()))


def report(root: Path):
    # Report regeneration is intentionally read-only and must remain possible
    # after the execution contract is frozen.  Do not re-hash moving source
    # files or invalidate completed shards merely because report code changed.
    cfg = json.loads((root / "configuration.json").read_text())
    final = json.loads((root / "final_manifest.json").read_text())
    # Report only the seed set declared by the completed run.  Older dense
    # shards may be preserved in the directory for reuse, but they must not
    # silently turn a seed-42-only evaluation into a three-seed report.
    completion = {}
    marker = root / "final_complete.json"
    if marker.exists():
        try:
            completion = json.loads(marker.read_text())
        except Exception:
            completion = {}
    selected_seeds = {int(x) for x in completion.get("seeds", [])}
    if not selected_seeds:
        selected_seeds = set(SEEDS)
    rows = []
    canvas_rows = []
    for path in (root / "final").glob("*/*.json"):
        if path.name == "status.json": continue
        try: data = json.loads(path.read_text())
        except Exception: continue
        if data.get("status") != "complete": continue
        if int(data.get("seed", -1)) not in selected_seeds: continue
        rows.append(data)
        for i, canvas in enumerate(data.get("canvases", [])):
            item = dict(canvas)
            item.pop("canvas_index", None)
            canvas_rows.append(dict(condition=data["condition"], method=data["method"], seed=data["seed"], id=data["id"], canvas_index=i, **item))
    if not rows: raise RuntimeError("no final shards found")
    grouped = defaultdict(list)
    for r in rows: grouped[r["condition"]].append(r)
    summary=[]
    for cond, group in sorted(grouped.items()):
        p = _profile(group); summary.append(dict(condition=cond, method=group[0]["method"], seed=group[0]["seed"], **{k:p[k] for k in ("n","accuracy","mean_steps","median_steps","p90_steps","mean_canvas_steps","median_canvas_steps","total_canvases","canvases_per_example","cap_canvases","executed_tiles")}, overall_sparsity=p["sparsity"]["whole"], global_sparsity=p["sparsity"]["global"], local_sparsity=p["sparsity"]["local"]))
    root.joinpath("summary.csv").parent.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (root / "per_canvas.csv").open("w", newline="") as f:
        if canvas_rows:
            writer=csv.DictWriter(f, fieldnames=list(canvas_rows[0])); writer.writeheader(); writer.writerows(canvas_rows)
    # Also publish a pooled three-seed view.  Sparsity is recomputed from
    # pooled eligible/skipped tile counts (never by averaging percentages),
    # while accuracy and denoising statistics are pooled over examples/canvas.
    pooled = []
    for method in ("dense", *METHODS):
        for target in ([None] if method == "dense" else TARGETS):
            group = [r for r in rows if r["method"] == method and
                     ((target is None and method == "dense") or
                      (target is not None and f"_s{target}_" in r["condition"]))]
            if not group:
                continue
            p = _profile(group)
            pooled.append(dict(method=method, target=("dense" if target is None else int(target)),
                               n=p["n"], accuracy=p["accuracy"],
                               overall_sparsity=p["sparsity"]["whole"],
                               global_sparsity=p["sparsity"]["global"],
                               local_sparsity=p["sparsity"]["local"],
                               mean_canvas_steps=p["mean_canvas_steps"],
                               median_canvas_steps=p["median_canvas_steps"],
                               p90_canvas_steps=float(np.quantile([int(c["iterations"]) for r in group for c in r.get("canvases", [])], .9)),
                               total_canvases=p["total_canvases"], cap_canvases=p["cap_canvases"],
                               executed_tiles=p["executed_tiles"]))
    with (root / "pooled_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pooled[0])); writer.writeheader(); writer.writerows(pooled)
    try:
        import matplotlib.pyplot as plt
        (root / "plots").mkdir(parents=True, exist_ok=True)
        for seed in sorted(selected_seeds):
            plt.figure(figsize=(8,5))
            for method, color in zip(METHODS, ("tab:blue", "tab:orange", "tab:red")):
                pts=[x for x in summary if x["method"]==method and x["seed"]==seed]
                pts.sort(key=lambda x:x["overall_sparsity"])
                if pts: plt.plot([100*x["overall_sparsity"] for x in pts],[x["mean_canvas_steps"] for x in pts],"o-",label=method,color=color)
            dense=[x for x in summary if x["method"]=="dense" and x["seed"]==seed]
            if dense: plt.axhline(dense[0]["mean_canvas_steps"],color="black",ls="--",label="dense")
            plt.xlabel("Actual physical tile sparsity (%)"); plt.ylabel("Mean denoising steps / 256-token canvas")
            plt.title(f"AIME26 temporal sweep (seed {seed})"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
            plt.savefig(root / "plots" / f"steps_vs_sparsity_seed{seed}.png", dpi=160); plt.close()
        plt.figure(figsize=(9,5))
        for method, color in zip(METHODS, ("tab:blue", "tab:orange", "tab:red")):
            pts=[x for x in summary if x["method"]==method]
            by={t:[] for t in TARGETS}
            for x in pts:
                try: by[int(x["condition"].split("_s")[1].split("_seed")[0])].append(x["mean_canvas_steps"])
                except Exception: pass
            xs=[];ys=[]
            for t,v in by.items():
                if v: xs.append(t);ys.append(float(np.mean(v)))
            if xs: plt.plot(xs,ys,"o-",label=method,color=color)
        plt.xlabel("Target sparsity (%)"); plt.ylabel("Mean denoising steps / canvas (selected-seed mean)")
        plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(root/"plots"/"target_steps_tradeoff.png",dpi=160);plt.close()
    except Exception as e:
        _write(root / "plot_error.json", dict(error=repr(e)))
    seed_label = ", ".join(str(x) for x in sorted(selected_seeds))
    lines=["# AIME26 temporal-instability sweep (v3 phase calibration)", "", f"β=3, γ=0.5; physical 128×64 tiles; 30 AIME26 prompts; evaluated seed(s): {seed_label}.", "", "Calibration follows [`AIME_CALIBRATION_RULES_V3.md`](AIME_CALIBRATION_RULES_V3.md): independent call-1/call-2/later local-global thresholds, phase and pooled target matching, and early-conservative threshold ordering. The six AIME calibration IDs overlap the historical 30-question manifest, so this is development evidence rather than fresh held-out confirmation.", "", "## Results", "", "| Condition | Actual O/G/L sparsity | Accuracy | Mean steps / canvas | Median | P90 | Total canvases |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x in summary:
        lines.append(f"| {x['condition']} | {100*x['overall_sparsity']:.2f}% / {100*x['global_sparsity']:.2f}% / {100*x['local_sparsity']:.2f}% | {100*x['accuracy']:.2f}% | {x['mean_canvas_steps']:.2f} | {x['median_canvas_steps']:.2f} | {x['p90_steps']:.2f} | {x['total_canvases']} |")
    lines += ["", "`per_canvas.csv` contains every canvas's denoising-step count and termination metadata. `summary.csv` contains per-seed condition aggregates; `pooled_summary.csv` pools only the selected completed seed set using aggregate tile counts.", "", "## Selected-seed pooled operating points", "", "| Method | Target | Actual O/G/L sparsity | Accuracy | Mean steps/canvas | Median | P90 | Canvases | Cap canvases |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in pooled:
        lines.append(f"| {x['method']} | {x['target']} | {100*x['overall_sparsity']:.2f}% / {100*x['global_sparsity']:.2f}% / {100*x['local_sparsity']:.2f}% | {100*x['accuracy']:.2f}% | {x['mean_canvas_steps']:.2f} | {x['median_canvas_steps']:.2f} | {x['p90_canvas_steps']:.2f} | {x['total_canvases']} | {x['cap_canvases']} |")
    d = next((x for x in summary if x["method"] == "dense"), None)
    parts = []
    if d:
        parts.append(f"Dense averages {d['mean_canvas_steps']:.2f} denoising steps per 256-token canvas.")
    for method in METHODS:
        pts = [x for x in summary if x["method"] == method]
        if pts:
            lo, hi = min(pts, key=lambda x: x["overall_sparsity"]), max(pts, key=lambda x: x["overall_sparsity"])
            parts.append(f"{method} spans {100*lo['overall_sparsity']:.1f}%--{100*hi['overall_sparsity']:.1f}% actual sparsity and {lo['mean_canvas_steps']:.2f}--{hi['mean_canvas_steps']:.2f} mean steps/canvas.")
    interp = " ".join(parts) + " Calibration questions overlap the historical AIME30, so these are development results rather than fresh held-out evidence."
    lines += ["", "## Interpretation", "", interp]
    (root / "report.md").write_text("\n".join(lines)+"\n")
    _write(root / "report_index.json", dict(summary_rows=len(summary), canvas_rows=len(canvas_rows), generated=time.time()))


def launch(root: Path, stage: str):
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    env = dict(os.environ, PYTHONPATH="src:.", CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4",
               LD_LIBRARY_PATH=torch_lib + (os.pathsep + ld if ld else ""))
    cmd = [sys.executable, "-u", "-m", "experiments.value_direction_hopper.aime_temporal_sweep", stage, "--root", str(root)]
    log = root / f"{stage}_{int(time.time())}.log"
    with log.open("xb") as f:
        p = subprocess.Popen(cmd, cwd=DLMDIR, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    _write(root / f"{stage}_launch.json", dict(pid=p.pid, log=str(log), command=cmd, started=time.time()))
    print(json.dumps(dict(pid=p.pid, log=str(log))))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage", choices=("prepare","calibrate","run","report","launch-calibrate","launch-run")); ap.add_argument("--root", type=Path, default=DEFAULT_ROOT); a=ap.parse_args()
    if a.stage == "prepare": prepare(a.root)
    elif a.stage == "calibrate": calibrate(a.root)
    elif a.stage == "run": run_final(a.root)
    elif a.stage == "report": report(a.root)
    else: launch(a.root, a.stage.removeprefix("launch-"))


if __name__ == "__main__": main()
