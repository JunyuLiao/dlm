"""Finish AIME once per model and disclose target drift without recalibration.

This supplemental coordinator leaves the already-running frozen worker intact.
Run it only after the original coordinator exits. Numerical execution continues
through that exact coordinator; its old target-miss exit is now informational.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ARM = "value_v2_gaussian32_reuse_r4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def benchmark_counts(base, rematch, model):
    rows, reference = [], []
    for shard in range(1, 5):
        relative = Path("runs") / model / "aime26" / ARM / f"shard{shard:03}/attempt001"
        leaf = rematch / relative
        marker = json.loads((leaf / "final_complete.json").read_text())
        receipt = json.loads((leaf / "receipt.json").read_text())
        records = leaf / "records.jsonl"
        if receipt["records_sha256"] != sha(records):
            raise RuntimeError("records hash mismatch: " + str(leaf))
        rows.extend(map(json.loads, records.read_text().splitlines()))
        reference.extend(map(json.loads, (base / relative / "records.jsonl").read_text().splitlines()))
    keys = [(r["id"], r["seed"]) for r in rows]
    expected = {(r["id"], r["seed"]) for r in reference}
    if len(rows) != 90 or len(set(keys)) != 90 or set(keys) != expected:
        raise RuntimeError("full benchmark identity mismatch: " + model)
    eligible = sum(r["counts"]["eligible_tiles"] for r in rows)
    kept = sum(r["counts"]["kept_tiles"] for r in rows)
    if not 0 <= kept <= eligible or not eligible:
        raise RuntimeError("invalid work counters")
    sparsity = 1 - kept / eligible
    return {"requests": 90, "eligible_tiles": eligible, "kept_tiles": kept,
            "sparsity": sparsity, "target_met": .48 <= sparsity <= .52,
            "post_benchmark_recalibration": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("rematch", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.base, a.rematch, a.out = a.base.resolve(), a.rematch.resolve(), a.out.resolve()
    policy = json.loads((a.out / "protocol.json").read_text())
    original = Path(__file__).with_name("multi_model_aime_rematch.py")
    if sha(original) != policy["numerical_coordinator_sha256"]:
        raise RuntimeError("frozen numerical coordinator changed")
    for model in ("llada22", "trado"):
        if not (a.rematch / model).exists():
            if model != "trado":
                raise RuntimeError("existing LLaDA benchmark must finish first")
            gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                           "--format=csv,noheader"], text=True)
            if gpu.strip():
                raise RuntimeError("GPU must be idle before next model")
            start = time.time()
            with (a.out / "private/trado.log").open("x") as log:
                child = subprocess.Popen([sys.executable, str(original), str(a.base),
                                          str(a.rematch), "--model", model],
                                         stdout=log, stderr=subprocess.STDOUT)
                state = {"pid": child.pid, "model": model, "suite": "aime26",
                         "phase": "calibrate_then_single_final_benchmark",
                         "attempt": "completion_policy001", "started": start}
                (a.base / "private/active_worker.json").write_text(json.dumps(state) + "\n")
                code = child.wait()
            write(a.out / "trado_process_receipt.json", {**state, "returncode": code,
                  "finished": time.time(), "allocated_worker_seconds": time.time() - start})
        # A nonzero old guard is acceptable only after all four complete shards.
        result = benchmark_counts(a.base, a.rematch, model)
        write(a.out / model / "final_complete.json", {**result,
              "status": "completed_target_met" if result["target_met"] else "completed_target_drift_disclosed",
              "numerical_coordinator_sha256": sha(original),
              "policy_sha256": sha(a.out / "protocol.json")})
        print(json.dumps({"event": "benchmark_complete", "model": model, **result}), flush=True)
    write(a.out / "generation_complete.json", {"status": "complete", "rerun_requests": 180,
          "post_benchmark_recalibration": False, "policy_sha256": sha(a.out / "protocol.json")})


if __name__ == "__main__":
    main()
