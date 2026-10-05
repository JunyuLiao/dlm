"""Collect operator diagnostics from kernel on a few questions."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from . import humaneval_query_sensitivity_v2 as v2
from . import aime_temporal_sweep as base
from .query_adaptive import UNIFORM_METHODS
from .integration import install


ROOT = Path(os.environ.get(
    "DIAGNOSTIC_ROOT",
    "/home/exouser/ljy/dlm/results/kernel_diagnostics_v1"))
SOURCE_ROOT = Path("/home/exouser/ljy/dlm/results/humaneval_query_sensitivity_v2")
METHODS = ("dense", "C_gate", "TLAR")
TARGET = .50
N_QUESTIONS = 10


def _select_rows():
    source = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    # Use first 10 questions from evaluation
    evaluation = source["evaluation"][:N_QUESTIONS]
    calibration = source["calibration"]
    if {x["id"] for x in calibration} & {x["id"] for x in evaluation}:
        raise RuntimeError("calibration/evaluation overlap")
    return calibration, evaluation, source.get("audit", {})


def _configuration(calibration, evaluation):
    import torch
    import transformers

    parent = json.loads(base.RULER_CFG.read_text())
    cfg = dict(
        schema="kernel_diagnostics_v1",
        model=parent["model"], revision=parent["revision"],
        library="/home/exouser/ljy/dlm/results/value_direction_hopper_v1/build/value_direction_fdb8df0bd94d8e69.so",
        torch_library="/home/exouser/ljy/dlm/results/value_direction_hopper_v1/build/value_direction_torch.so",
        m_ref=parent["m_ref"], beta=3., gamma=.5,
        trajectory_gamma=.5, gate_trajectory_gamma=.65, gate_tau=2.5,
        rank=32, projection_seed=1729, physical_tile=[128, 64],
        canvas=256, max_steps=48, seed=42,
        methods=list(METHODS), target=TARGET,
        calibration_ids=[x["id"] for x in calibration],
        evaluation_ids=[x["id"] for x in evaluation],
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
    cfg["source_hashes"] = {str(p.resolve()): str(p) for p in files}
    cfg["fingerprint"] = str(cfg)
    return cfg


def _initial(method):
    return {
        "C_gate": v2._policy(.95, .55),
        "TLAR": v2._policy(.95, .55),
    }[method]


def _run_one(adapter, row, method, policy, cfg, projections):
    # Use the base _run_one but we need to capture the router for diagnostics
    # We'll monkey-patch install to return the router
    original_install = base.install
    
    router_holder = {}
    
    @contextmanager
    def patched_install(adapter, library, policy, **kwargs):
        with original_install(adapter, library, policy, **kwargs) as (binding, router):
            router_holder['router'] = router
            yield binding, router
    
    base.install = patched_install
    try:
        if method == "dense":
            out = base._run_one(adapter, row, method, None, cfg, projections, diagnostics=True)
        else:
            out = base._run_one(adapter, row, method, policy, cfg, projections, diagnostics=True)
    finally:
        base.install = original_install
    
    out.update(stratum=row["stratum"], entry_point=row["entry_point"],
               generation_budget=row["generation_budget"])
    # Store router for diagnostics extraction
    out['_router'] = router_holder.get('router')
    return out


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    calibration, evaluation, audit = _select_rows()
    cfg = _configuration(calibration, evaluation)
    cfg_path = ROOT / "configuration.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old["fingerprint"] != cfg["fingerprint"]:
            raise RuntimeError("existing diagnostic study configuration differs")
    else:
        with cfg_path.open('w') as f:
            json.dump(cfg, f, indent=2)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    base.aime_score = lambda row, text: 0.
    adapter = base._adapter(cfg)
    projections = base.Projections()

    # Just run dense and C_gate for now, collect diagnostics
    for method in ("dense", "C_gate"):
        policy = None if method == "dense" else _initial(method)
        print(f"Running {method}...")
        for row in evaluation:
            result = _run_one(adapter, row, method, policy, cfg, projections)
            # Extract diagnostics from router if available
            router = result.pop('_router', None)
            if router and hasattr(router, 'diagnostics') and router.diagnostics:
                diag = router.diagnostics
                out_dir = ROOT / "diagnostics" / method / row["id"].replace("/", "_")
                out_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np
                for k, v in diag.items():
                    if hasattr(v, 'numpy'):
                        v = v.numpy()
                    np.save(out_dir / f"{k}.npy", v)
                print(f"  Saved diagnostics for {row['id']}: {diag['record_count'].item()} records")
            else:
                print(f"  No diagnostics for {row['id']}")
        print(f"  Completed {method}")


if __name__ == "__main__":
    run()