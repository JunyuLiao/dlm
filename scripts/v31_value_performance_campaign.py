"""Resumable one-shard launcher for the AIME26/RULER128K value panel.

This wrapper freezes the suite-specific budget and source identity explicitly.
It delegates one shard to the paired vLLM runner and never treats a partial
JSONL file as a completed shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


ARMS = {
    "dense_full_fix51994": ("dense", "default", None),
    "dense_piecewise": ("dense", "PIECEWISE", None),
    "allkept_fa4": ("allkept", "PIECEWISE", None),
    "current_v31_control": ("mage", "PIECEWISE", None),
    "value_v1_online_discard_mass": ("mage", "PIECEWISE", "value_v1_online_discard_mass"),
    "value_v2_online_preserve_mass": ("mage", "PIECEWISE", "value_v2_online_preserve_mass"),
    "value_v2_online_jcgate": ("mage", "PIECEWISE", "value_v2_online_preserve_mass"),
    "value_v3a_singleton_delete": ("mage", "PIECEWISE", "value_v3a_singleton_delete"),
    "value_v3b_approx_batch8": ("mage", "PIECEWISE", "value_v3b_approx_batch8"),
    "value_mass_gated_online_swap8": ("mage", "PIECEWISE", "value_mass_gated_online"),
    "value_mass_gated_online_cgate_swap16": ("mage", "PIECEWISE", "value_mass_gated_online_cgate"),
}

FIXED_VALUE_ARMS = {
    "value_mass_gated_online_swap8": (2, 8, None),
    "value_mass_gated_online_cgate_swap16": (2, 16, "cgate"),
}
SOURCE_FILES = (
    "experiments/numerical_qk_reuse/vllm_adapter.py",
    "experiments/numerical_qk_reuse/v31_value_selectors.py",
    "experiments/numerical_qk_reuse/v31_value_kernels.py",
    "scripts/v31_vllm_paired_bench.py",
    "scripts/v31_value_performance_campaign.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def query_gpu() -> tuple[str, str]:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"CUDA preflight failed: {e}") from e
    if not rows:
        raise RuntimeError("CUDA preflight found no GPU")
    if apps:
        raise RuntimeError(f"GPU is occupied: {apps}")
    return rows, apps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--suite", choices=("aime26", "ruler128k"), required=True)
    p.add_argument("--manifests", required=True, type=Path)
    p.add_argument("--cells", required=True, type=Path)
    p.add_argument("--attempt", required=True, type=Path)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--purpose", choices=("audit", "clean"), required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--sparsity-target", choices=("50", "75"), default="50")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=31)
    args = p.parse_args()
    if args.shards <= 0 or not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid shard")
    if ((args.arm.startswith("value_v") and "online" in args.arm)
            or args.arm in FIXED_VALUE_ARMS) and args.threshold is None:
        raise ValueError("online arms require a frozen threshold")
    if args.attempt.exists():
        raise ValueError(f"attempt already exists: {args.attempt}")
    root = Path(__file__).resolve().parents[1]
    cells_bytes = args.cells.read_bytes()
    cells = json.loads(cells_bytes)
    if not cells:
        raise ValueError("empty cells file")
    expected = len(cells[args.shard_index::args.shards])
    manifest_paths = [args.manifests / f"{ds}_generation_manifest.json" for ds in {c["dataset"] for c in cells}]
    if any(not path.exists() for path in manifest_paths):
        raise FileNotFoundError("missing frozen generation manifest")
    gpu_identity, _ = query_gpu()
    attempt = args.attempt.resolve()
    attempt.mkdir(parents=True)
    (attempt / "private").mkdir()
    public_path = attempt / "records.jsonl"
    private_path = attempt / "private" / f"{args.arm}.private.jsonl"
    arm, graph, selector = ARMS[args.arm]
    is_aime = args.suite == "aime26"
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("MAGE_", "VALUE_", "LOCAL_")):
            env.pop(key, None)
    if args.sparsity_target == "50":
        mage_k = "1728" if is_aime else "8192"
    else:
        # The native budgets are approximately halved to move the realized
        # GLOBAL tile sparsity from the existing ~50% regime toward ~75%.
        mage_k = "864" if is_aime else "4096"
    env.update({
        "PYTHONPATH": f"{root / 'src'}:{root}", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_VISIBLE_DEVICES": "0", "FIX_51994": "1",
        "FA4_LOCAL_FIX": "1", "FIX_51994": "1", "BLOCK": "64" if is_aime else "32",
        "CHUNK": "4096" if is_aime else "16384", "MEM": "0.80" if is_aime else "0.85",
        "MAX_MODEL_LEN": "9216" if is_aime else "136192", "SEED_BASE": str(args.seed_base),
        "MAGE_K": mage_k, "MAGE_SELECT": "fa4", "MAGE_GRAN": "qblock_max",
        "MAGE_STEP": "1", "MAGE_CARRY": "1", "MAGE_RESELECT_TRIGGER": "0.15",
        "MAGE_TRIGGER_SIGNAL": "settle", "MAGE_STICKY": "1.386", "LOGIT_STATS": "fused",
        "DP_BUILD": "chunked", "OBSERVE": "fa4", "KV_COPY": "triton", "MERGE": "triton",
        "VALUE_AUDIT": "1" if args.purpose == "audit" else "0",
        "VALUE_CLEAN_TIMING": "1" if args.purpose == "clean" else "0",
        "SHARD": f"{args.shard_index}/{args.shards}",
    })
    if selector:
        env["VALUE_SELECTOR"] = selector
    if args.arm in FIXED_VALUE_ARMS:
        pool_factor, swap_count, row_weight = FIXED_VALUE_ARMS[args.arm]
        env["VALUE_POOL_FACTOR"] = str(pool_factor)
        env["VALUE_SWAP_COUNT"] = str(swap_count)
        if row_weight is not None:
            env["MAGE_ROWW"] = row_weight
    if args.arm == "value_v2_online_jcgate":
        env["MAGE_ROWW"] = "jcgate"
    if args.threshold is not None:
        env["VALUE_THRESHOLD"] = str(args.threshold)
    config = {
        "schema": "v31_value_performance_shard_v1", "suite": args.suite, "arm": args.arm,
        "purpose": args.purpose, "shard_index": args.shard_index, "shards": args.shards,
        "planned_records": expected, "cells_sha256": hashlib.sha256(cells_bytes).hexdigest(),
        "manifests_sha256": {p.name: digest(p) for p in manifest_paths},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {name: digest(root / name) for name in SOURCE_FILES},
        "gpu_identity": gpu_identity, "host": socket.gethostname(), "threshold": args.threshold,
        "sparsity_target": args.sparsity_target,
        "runtime": {k: env[k] for k in ("BLOCK", "CHUNK", "MEM", "MAX_MODEL_LEN", "MAGE_K", "MAGE_STEP",
                                         "MAGE_CARRY", "MAGE_GRAN", "MAGE_STICKY", "VALUE_AUDIT", "VALUE_CLEAN_TIMING")},
    }
    (attempt / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    cmd = [args.python, str(root / "scripts/v31_vllm_paired_bench.py"), args.model, str(args.manifests),
           str(args.cells), str(public_path), str(private_path), arm, graph]
    with (attempt / "worker.log").open("w") as log:
        result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = [json.loads(line) for line in public_path.read_text().splitlines() if line.strip()] if public_path.exists() else []
    errors = []
    if len(rows) != expected:
        errors.append(f"records {len(rows)} != planned shard records {expected}")
    for row in rows:
        if row.get("cuda_graph_captures") != 0:
            errors.append("timed CUDA graph capture")
        if row.get("value_clean_timing") != (args.purpose == "clean"):
            errors.append("audit/clean identity mismatch")
        if row.get("local_kv_budget") is not None:
            errors.append("LOCAL routing was enabled")
    receipt = {"status": "complete" if result.returncode == 0 and not errors else "failed",
               "returncode": result.returncode, "records": len(rows), "planned": expected,
               "gpu_seconds": time.perf_counter() - start, "validation_errors": sorted(set(errors))}
    (attempt / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] == "complete":
        (attempt / "attempt_complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    raise SystemExit(0 if receipt["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
