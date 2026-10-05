"""HumanEval development study of C_progress_v2 confidence-progress routing.

This driver evaluates C_progress_v2 against dense, Gaussian32, and C_gate on a
30-question balanced HumanEval subset at 50% physical sparsity.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import humaneval_query_sensitivity_v2 as v2
from . import aime_temporal_sweep as base
from .experiment import atomic, fingerprint, sha
from .query_adaptive import UNIFORM_METHODS


ROOT = Path(os.environ.get(
    "HUMANEVAL_PROGRESS_V2_ROOT",
    "/home/exouser/ljy/dlm/results/humaneval_query_sensitivity_progress_v2"))
SOURCE_ROOT = Path("/home/exouser/ljy/dlm/results/humaneval_query_sensitivity_v2")
METHODS = tuple(x.strip() for x in os.environ.get(
    "HUMANEVAL_PROGRESS_V2_METHODS", "dense,gaussian32,C_gate,C_progress_v2").split(",")
                if x.strip())
SPARSE = tuple(method for method in METHODS if method != "dense")
TARGET = .50
TOLERANCE = .02
PROGRESS_LAMBDA = float(os.environ.get("HUMANEVAL_PROGRESS_V2_LAMBDA", "0.2"))
PROGRESS_SCALE = float(os.environ.get("HUMANEVAL_PROGRESS_V2_SCALE", "0.2"))
PROGRESS_GAMMA = float(os.environ.get("HUMANEVAL_PROGRESS_V2_GAMMA", "0.6"))


def _select_rows():
    source = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    calibration = list(source["calibration"])
    grouped = {}
    for row in source["evaluation"]:
        grouped.setdefault(row["stratum"], []).append(row)
    evaluation = []
    for stratum in sorted(grouped):
        evaluation.extend(sorted(grouped[stratum], key=lambda x: int(x["source_id"]))[:5])
    if len(evaluation) != 30 or len({x["stratum"] for x in evaluation}) != 6:
        raise RuntimeError("balanced 30-question selection failed")
    if {x["id"] for x in calibration} & {x["id"] for x in evaluation}:
        raise RuntimeError("calibration/evaluation overlap")
    return calibration, evaluation, source.get("audit", {})


def _configuration(calibration, evaluation):
    import torch
    import transformers

    parent = json.loads(base.RULER_CFG.read_text())
    cfg = dict(
        schema="humaneval_query_sensitivity_progress_v2",
        model=parent["model"], revision=parent["revision"],
        library=parent["library"], torch_library=parent["torch_library"],
        m_ref=parent["m_ref"], beta=3., gamma=.5,
        trajectory_gamma=.5, gate_trajectory_gamma=.65, gate_tau=2.5,
        progress_lambda=PROGRESS_LAMBDA,
        progress_scale=PROGRESS_SCALE,
        progress_gamma=PROGRESS_GAMMA,
        rank=32, projection_seed=1729, physical_tile=[128, 64],
        canvas=256, max_steps=48, seed=42,
        methods=list(METHODS), target=TARGET, tolerance=TOLERANCE,
        calibration_ids=[x["id"] for x in calibration],
        evaluation_ids=[x["id"] for x in evaluation],
        selection="five evaluation prompts per v2 function-name stratum; seed 42",
        threshold_fit="disjoint calibration, then final-cohort tile-count budget matching",
        accuracy_used_for_selection=False,
        torch=torch.__version__, transformers=transformers.__version__,
    )
    files = [Path(__file__), Path(v2.__file__), Path(base.__file__),
             Path(__file__).with_name("query_adaptive.py"),
             Path(__file__).with_name("query_sensitivity_uniform.py"),
             Path(__file__).with_name("integration.py"),
             Path(__file__).with_name("cuda.py"),
             Path("src/dllm/models/adapters/diffusion_gemma.py"),
             Path("fast_dllm_v2/scripts/eval_blasst_mixed_tasks.py"),
             Path(cfg["library"]), Path(cfg["torch_library"]),
             SOURCE_ROOT / "manifest.json"]
    cfg["source_hashes"] = {str(p.resolve()): sha(p) for p in files}
    cfg["fingerprint"] = fingerprint(cfg)
    return cfg


def _write_manifest(calibration, evaluation, audit):
    atomic(ROOT / "manifest.json", dict(
        seed=42, total=len(calibration) + len(evaluation),
        calibration=calibration, evaluation=evaluation,
        audit=dict(source="humaneval_query_sensitivity_v2", **audit),
    ))


def _initial(method):
    return {
        "gaussian32": v2._policy(-.35, -.50),
        "C_gate": v2._policy(.95, .55),
        "C_progress_v2": v2._policy(.95, .55),
    }[method]


def _run_one(adapter, row, method, policy, cfg, projections):
    return v2._run_one(adapter, row, method, policy, cfg, projections)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    calibration, evaluation, audit = _select_rows()
    manifest = ROOT / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if [x["id"] for x in old["calibration"]] != [x["id"] for x in calibration] or \
                [x["id"] for x in old["evaluation"]] != [x["id"] for x in evaluation]:
            raise RuntimeError("existing progress-v2 study manifest differs")
    else:
        _write_manifest(calibration, evaluation, audit)
    cfg = _configuration(calibration, evaluation)
    cfg_path = ROOT / "configuration.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old["fingerprint"] != cfg["fingerprint"]:
            raise RuntimeError("existing progress-v2 study configuration differs")
    else:
        atomic(cfg_path, cfg)

    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    base.aime_score = lambda row, text: 0.
    adapter = base._adapter(cfg)
    projections = base.Projections()

    disjoint = {}
    for method in SPARSE:
        disjoint[method] = v2._fit(ROOT, adapter, cfg, calibration, projections,
                                   method, _initial(method), "calibration",
                                   max_points=18)
    atomic(ROOT / "calibration_complete.json", {
        "passed": True, "thresholds": disjoint,
        "calibration_ids": [x["id"] for x in calibration],
    })

    thresholds = {}
    for method in SPARSE:
        thresholds[method] = v2._fit(
            ROOT, adapter, cfg, evaluation, projections, method,
            disjoint[method]["policy"], "budget_match", max_points=18)
    atomic(ROOT / "thresholds_frozen.json", {
        "passed": True, "accuracy_used": False, "thresholds": thresholds,
    })

    all_rows = {}
    for method in METHODS:
        policy = None if method == "dense" else thresholds[method]["policy"]
        key = "dense" if policy is None else fingerprint(policy)[:16]
        stage = "evaluation" if policy is None else "budget_match"
        values = []
        for row in evaluation:
            result = dict(v2._cache(
                ROOT, stage, method, key, row,
                lambda row=row: _run_one(adapter, row, method, policy, cfg, projections)))
            result["score"], result["humaneval_execution"] = v2._score(row, result["prediction"])
            values.append(result)
        all_rows[method] = values
        atomic(ROOT / f"{method}_evaluation.json", values)
        print(json.dumps(dict(event="method_complete", method=method,
                              metrics=v2._metrics(values))), flush=True)

    report = dict(protocol=dict(seed=42, calibration=len(calibration), evaluation=len(evaluation),
                                target=TARGET, tolerance=TOLERANCE,
                                calibration_ids=[x["id"] for x in calibration],
                                evaluation_ids=[x["id"] for x in evaluation],
                                threshold_fit="disjoint then final-cohort tile-count matching",
                                accuracy_used_for_selection=False),
                  methods={m: v2._metrics(v) for m, v in all_rows.items()},
                  thresholds=thresholds, fingerprint=cfg["fingerprint"])
    atomic(ROOT / "report.json", report)
    lines = ["# HumanEval C_progress_v2 development study", "",
             f"Seed 42; {len(evaluation)} fixed prompts; {cfg['selection']}; tile-count-only threshold fitting (transductive).", "",
             "| Method | Pass@1 | Overall | Local | Global | Mean calls/canvas | P90 calls | Syntax failures | Capped canvases |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in report["methods"].items():
        sp = metrics["sparsity"]
        lines.append(f"| {method} | {metrics['accuracy']:.1%} | {sp['whole']:.2%} | {sp['local']:.2%} | {sp['global']:.2%} | {metrics['mean_canvas_steps']:.2f} | {metrics['p90_steps']:.1f} | {metrics['execution_status_counts'].get('syntax_error', 0)} | {metrics['cap_canvases']} |")
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    atomic(ROOT / "final_complete.json", dict(
        passed=True, n=len(evaluation), methods=METHODS,
        fingerprint=cfg["fingerprint"]))


if __name__ == "__main__":
    run()