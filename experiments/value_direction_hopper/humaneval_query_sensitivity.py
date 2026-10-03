"""Small, paired HumanEval pass@1 study for query-sensitivity routers.

The study deliberately reuses the DiffusionGemma/AIME generation path.  It
freezes a seed-42 HumanEval subset, calibrates one local/global log-threshold
pair per sparse method on six tasks, and evaluates the remaining tasks.  The
calibration and evaluation tasks are disjoint so the result is useful for
iteration, while the manifest and all raw completions remain auditable.
"""
from __future__ import annotations

import argparse, ast, gzip, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from . import aime_temporal_sweep as base
from . import aime_query_sensitivity_uniform as uniform
from .query_adaptive import UNIFORM_METHODS
from fast_dllm_v2.scripts.eval_blasst_mixed_tasks import _execute_humaneval

ROOT_DEFAULT = Path("/home/exouser/ljy/dlm/results/humaneval_query_sensitivity_v1")
DATA_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
METHODS = ("dense", "gaussian32", "blasst_aggressive", "C_gate", "T_prior")
SPARSE = METHODS[1:]
TARGET = 0.50
N_TOTAL, N_CAL = 24, 6


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _download(path: Path) -> Path:
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def _prompt(item: Mapping[str, Any]) -> str:
    return ("Complete the following Python function. Return only valid Python code "
            "containing the complete function, without Markdown fences.\n\n" +
            str(item["prompt"]))


def _rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _download(root / "source_data" / "HumanEval.jsonl.gz")
    with gzip.open(source, "rt", encoding="utf-8") as f:
        all_items = [json.loads(x) for x in f if x.strip()]
    # Selection is frozen by seed 42 and independent of method outcomes.
    indices = sorted(random.Random(42).sample(range(len(all_items)), N_TOTAL))
    rows = []
    for pos, idx in enumerate(indices):
        item = all_items[idx]
        rows.append(dict(
            id=f"humaneval/{item['task_id']}", source_id=int(idx), task="humaneval",
            benchmark="humaneval",
            prompt=_prompt(item), sparsity_prompt=item["prompt"],
            prompt_hash=hashlib.sha256(item["prompt"].encode()).hexdigest(),
            prompt_tokens=0, generation_budget=512, seed=42,
            label={"canonical_solution": item["canonical_solution"],
                   "test": item["test"], "entry_point": item["entry_point"],
                   "prompt": item["prompt"]},
        ))
    return rows[N_CAL:], rows[:N_CAL]


def _score(row: Mapping[str, Any], text: str) -> float:
    result = _execute_humaneval(text, row["label"], timeout=5.0)
    return float(bool(result["passed"])), result


def _policy(local: float, global_: float) -> dict[str, Any]:
    return uniform._uniform_policy(local, global_)


def _initial(method: str) -> dict[str, Any]:
    if method == "blasst_aggressive":
        # Existing AIME BLASST s50 anchor; this is only a starting point.
        return _policy(7.0, 6.7)
    return _policy(-0.85, -1.25)


def _run_one(adapter, row, method, policy, cfg, projections):
    if method == "dense":
        out = base._run_one(adapter, row, method, None, cfg, projections, diagnostics=True)
    else:
        out = base._run_one(adapter, row, method, policy, cfg, projections, diagnostics=True)
    score, execution = _score(row, out["prediction"])
    out["score"] = score
    out["humaneval_execution"] = execution
    return out


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"accuracy": None, "sparsity": {k: 0.0 for k in ("whole", "local", "global")}}
    m = base._profile(rows)
    m["accuracy"] = sum(r["score"] for r in rows) / len(rows)
    return m


def _cache(root: Path, stage: str, method: str, key: str, row: Mapping[str, Any], fn):
    path = root / stage / method / key / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    value = fn()
    path.write_text(json.dumps(value, sort_keys=True, default=str) + "\n")
    return value


def _calibrate(root, adapter, cfg, calibration, projections, method):
    if method == "dense":
        return None
    destination = root / "thresholds" / f"{method}_s50.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("status") == "attained":
            return old
        # An incomplete search is deliberately resumable after expanding the
        # candidate range; cached complete trajectories are retained below.
        destination.unlink()
    initial = _initial(method)
    candidates = [initial]
    # Five complete-trajectory points are enough for this fast iteration: the
    # initial point, shared shifts, and independent local/global corrections.
    for dl, dg in ((.5, .5), (-.5, -.5), (.5, 0.), (0., .5), (-.25, .25),
                   (0., 1.0), (.25, 1.0), (0.5, 1.0), (1.0, 1.0),
                   (1.0, 1.5), (.75, 1.25), (2., 2.), (3., 3.),
                   (4., 4.), (2., 3.), (3., 2.), (4., 3.)):
        p = initial["late"]
        candidates.append(_policy(p["local"]["log_threshold"] + dl,
                                  p["global"]["log_threshold"] + dg))
    if method == "blasst_aggressive":
        # BLASST's log-scale convention is much steeper than value routing;
        # the AIME anchor is intentionally only a starting point.
        candidates = [_policy(7.0 + dl, 6.7 + dg)
                      for dl, dg in ((0., 0.), (-1., -1.), (-2., -2.),
                                     (-3., -3.), (-2., -1.), (-1., -2.),
                                     (-3., -2.), (-2., -3.), (-4., -4.),
                                     (-6., -6.), (-8., -8.), (-10., -10.),
                                     (-8., -6.), (-6., -8.), (-5.5, -5.5),
                                     (-5.25, -5.25), (-5.5, -5.25))]
        candidates.append(_policy(2.0, 1.7))
    elif method == "C_gate":
        candidates = [_policy(-0.85 + dl, -1.25 + dg)
                      for dl, dg in ((2., 2.), (2.5, 2.5), (3., 3.),
                                     (2.5, 3.), (3., 2.5), (1.75, 1.75),
                                     (1.8, 1.8))]
    elif method == "T_prior":
        candidates = [_policy(-0.85 + dl, -1.25 + dg)
                      for dl, dg in ((1.5, 1.5), (1.25, 1.5),
                                     (1.5, 1.25), (1.75, 1.75), (1.6, 1.6))]
    elif method == "gaussian32":
        candidates = [_policy(-0.85 + dl, -1.25 + dg)
                      for dl, dg in ((.5, .5), (.5, .75), (.75, .5))]
    points = []
    for ci, policy in enumerate(candidates):
        key = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
        values = [_cache(root, "calibration", method, key, row,
                         lambda row=row, policy=policy: _run_one(adapter, row, method, policy, cfg, projections))
                  for row in calibration]
        metrics = _metrics(values)
        # The requested 50% target is an overall physical sparsity target.
        # Local/global rates are retained in the report; their calibration
        # mismatch is a diagnostic of layer-type response, not hidden.
        error = abs(metrics["sparsity"]["whole"] - TARGET)
        points.append(dict(policy=policy, metrics=metrics, max_error=error, key=key))
        print(json.dumps(dict(event="calibration", method=method, candidate=ci,
                              sparsity=metrics["sparsity"], max_error=error)), flush=True)
    feasible = [p for p in points if p["max_error"] <= .02]
    selected = min(feasible or points, key=lambda p: (p["max_error"], p["metrics"].get("mean_canvas_steps", 1e9)))
    result = dict(method=method, target=50, policy=selected["policy"],
                  selected_metrics=selected["metrics"], max_error=selected["max_error"],
                  status="attained" if selected["max_error"] <= .02 else "search_incomplete",
                  calibration_ids=[r["id"] for r in calibration], candidates=points,
                  threshold_schedule="uniform_all_denoising_steps")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    evaluation, calibration = _rows(root)
    (root / "manifest.json").write_text(json.dumps({"seed": 42, "total": N_TOTAL,
        "calibration": calibration, "evaluation": evaluation}, indent=2, sort_keys=True) + "\n")
    return evaluation, calibration


def run(root: Path):
    evaluation, calibration = prepare(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # Reuse the native trajectory runner, replacing only its benchmark score
    # callback; HumanEval execution is performed in a subprocess below.
    base.aime_score = lambda row, text: _score(row, text)[0]
    cfg = json.loads(base.RULER_CFG.read_text())
    cfg.update(beta=3.0, gamma=.5, trajectory_gamma=.5, gate_trajectory_gamma=.65,
               gate_tau=2.5, rank=32, projection_seed=1729, physical_tile=[128, 64],
               canvas=256, max_steps=48)
    adapter = base._adapter(cfg); projections = base.Projections()
    thresholds = {m: _calibrate(root, adapter, cfg, calibration, projections, m) for m in SPARSE}
    if any(v["status"] != "attained" for v in thresholds.values()):
        raise RuntimeError("one or more methods missed 50% ±2% calibration")
    all_rows = {}
    for method in METHODS:
        policy = None if method == "dense" else thresholds[method]["policy"]
        values = []
        for row in evaluation:
            key = "dense" if method == "dense" else hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
            values.append(_cache(root, "evaluation", method, key, row,
                                 lambda row=row, method=method, policy=policy: _run_one(adapter, row, method, policy, cfg, projections)))
        all_rows[method] = values
        (root / f"{method}_evaluation.json").write_text(json.dumps(values, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps(dict(event="method_complete", method=method, metrics=_metrics(values))), flush=True)
    report = {"protocol": {"seed": 42, "total": N_TOTAL, "calibration": N_CAL,
                            "evaluation": len(evaluation), "target_sparsity": .5},
              "methods": {m: _metrics(v) for m, v in all_rows.items()},
              "thresholds": thresholds,
              "dense_calibration": "not applicable: native dense has no skip threshold",
              "manifest_sha256": _sha(root / "manifest.json")}
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    lines = ["# HumanEval query-sensitivity study", "", "Seed 42; 24 fixed tasks (6 calibration, 18 disjoint evaluation); pass@1.", "", "| Method | Pass@1 | Whole sparsity | Local | Global | Mean steps/canvas |", "|---|---:|---:|---:|---:|---:|"]
    for method, m in report["methods"].items():
        s = m["sparsity"]
        lines.append(f"| {method} | {m['accuracy']:.1%} | {s['whole']:.1%} | {s['local']:.1%} | {s['global']:.1%} | {m.get('mean_canvas_steps', 0):.2f} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=("prepare", "run")); ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    a = ap.parse_args()
    if a.stage == "prepare": prepare(a.root)
    else: run(a.root)


if __name__ == "__main__": main()
