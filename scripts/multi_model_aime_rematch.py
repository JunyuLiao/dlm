"""Count-only, transductive AIME calibration followed by fresh V2 generation.

Numerical execution comes exclusively from the original model-specific frozen
worker. This coordinator never imports a scorer or reads gold/completion text.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ARM = "value_v2_gaussian32_reuse_r4"
INITIAL = {"llada22": .03, "trado": .025}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def pool(rows):
    keys = [(r["id"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate request identity")
    eligible = sum(r["counts"]["eligible_tiles"] for r in rows)
    kept = sum(r["counts"]["kept_tiles"] for r in rows)
    if not 0 <= kept <= eligible or not eligible:
        raise ValueError("invalid physical tile counts")
    return {"requests": len(rows), "eligible_tiles": eligible,
            "kept_tiles": kept, "sparsity": 1 - kept / eligible}


def next_threshold(trials, low, high):
    """Use only realized work; nonmonotone trajectories remain in the record."""
    below = [t["threshold"] for t in trials if t["sparsity"] < .5]
    above = [t["threshold"] for t in trials if t["sparsity"] >= .5]
    pairs = [(l, h) for l in below for h in above if l < h]
    if pairs:
        low, high = min(pairs, key=lambda p: math.log(p[1] / p[0]))
    elif below:
        low = max(below)
    elif above:
        high = min(above)
    return math.sqrt(low * high)


def run_model(base, out, model):
    bindings = json.loads((base / "private/model_worker_bindings002.json").read_text())
    source = base / "private/execution" / bindings[model]
    sys.path[:0] = [str(base / "private/compat" / model), str(source / "src"),
                    str(source), str(source / "scripts")]
    os.environ.update(HF_MODULES_CACHE=str(base / "private/hf_modules"),
                      TRITON_CACHE_DIR=str(base / "private/triton_cache"))
    worker = importlib.import_module("multi_model_worker")
    rows = json.loads((base / "private/data" / model / "aime26.json").read_text())
    # Only the prompt, ID, prefix and seed are passed to generation.
    cells = [(r, seed) for r in rows for seed in r["seeds"]]
    cells = [c for shard in range(4) for i, c in enumerate(cells) if i % 4 == shard]
    if len(rows) != 30 or len(cells) != 90 or {s for _, s in cells} != {42, 43, 44}:
        raise ValueError("AIME cohort differs from approved 30 x 3")
    leaf = out / model
    leaf.mkdir(exist_ok=False)
    write(leaf / "binding.json", {
        "numerical_source_commit": bindings[model], "worker_sha256": sha(worker.__file__),
        "manifest_sha256": sha(base / "private/data" / model / "aime26.json"),
        "original_protocol_sha256": sha(base / "protocol.json"),
        "rematch_protocol_sha256": sha(out / "rematch_protocol.json"),
        "calibration_reads_accuracy": False, "transductive": True,
        "sampler_cap": worker.CAPS[model]["aime26"], "pid": os.getpid(),
    })
    harness = worker.ModelHarness(base, model).load()
    old = json.loads((base / "calibration" / model / "aime26/attempt001/selected_threshold.json").read_text())
    original_rows = []
    for shard in range(1, 5):
        p = base / "runs" / model / "aime26" / ARM / f"shard{shard:03}/attempt001/records.jsonl"
        original_rows.extend(map(json.loads, p.read_text().splitlines()))
    trials = [{"threshold": old["selected_threshold"], **pool(original_rows),
               "role": "historical_sparsity_only"}]
    low, high = 1e-6, old["selected_threshold"]
    threshold = INITIAL[model]
    for index in range(1, 16):
        candidate = leaf / "calibration" / f"candidate{index:03}"
        candidate.mkdir(parents=True, exist_ok=False)
        write(candidate / "config.json", {"threshold": threshold, "requests": 90,
              "cap": worker.CAPS[model]["aime26"], "accuracy_blinded": True,
              "transductive": True, "coordinator_sha256": sha(__file__)})
        harness.runtime.arm, harness.runtime.threshold = ARM, threshold
        observed = []
        for row, seed in cells:
            item = {k: row[k] for k in ("id", "prompt", "answer_prefix") if k in row}
            pub, private = worker.run_cell(harness, item, worker.CAPS[model]["aime26"], seed)
            observed.append(pub)
            worker.append(candidate / "records.jsonl", pub)
            # Calibration outputs are deliberately discarded without scoring.
            del private
            print(json.dumps({"event": "calibration_progress", "model": model,
                  "candidate": index, "completed": len(observed), "expected": 90}), flush=True)
        result = {"threshold": threshold, **pool(observed)}
        trials.append(result)
        write(candidate / "final_complete.json", result)
        print(json.dumps({"event": "calibration_complete", "model": model, **result}), flush=True)
        if .48 <= result["sparsity"] <= .52:
            write(leaf / "selected_threshold.json", {**result, "trials": trials,
                  "transductive": True, "accuracy_blinded": True})
            break
        threshold = next_threshold(trials, low, high)
    else:
        raise RuntimeError("15 count-only candidates exhausted; retain evidence")
    # A distinct, fresh generation pass is required even though the whole
    # cohort was used for tile-count calibration.
    for shard in range(1, 5):
        worker.benchmark(harness, out, "aime26", ARM, shard, 4, "attempt001", threshold)
    rerun = []
    for shard in range(1, 5):
        p = out / "runs" / model / "aime26" / ARM / f"shard{shard:03}/attempt001/records.jsonl"
        rerun.extend(map(json.loads, p.read_text().splitlines()))
    result = pool(rerun)
    if result["requests"] != 90 or not .48 <= result["sparsity"] <= .52:
        write(leaf / "target_miss.json", result)
        raise RuntimeError("fresh benchmark misses physical 48-52% target")
    write(leaf / "final_complete.json", {**result, "threshold": threshold,
          "transductive": True, "fresh_rerun": True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--model", choices=("llada22", "trado"))
    a = p.parse_args()
    base, out = a.base.resolve(), a.out.resolve()
    if a.model:
        run_model(base, out, a.model)
        return
    if not (out / "rematch_protocol.json").exists():
        raise RuntimeError("freeze the supplemental protocol before launch")
    for model in ("llada22", "trado"):
        if (out / model / "final_complete.json").exists():
            continue
        gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader"], text=True)
        if gpu.strip():
            raise RuntimeError("GPU must be idle before the next worker")
        log = out / "private" / f"{model}_attempt001.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with log.open("x") as f:
            child = subprocess.Popen([sys.executable, __file__, str(base), str(out),
                                     "--model", model], stdout=f, stderr=subprocess.STDOUT)
            worker_state = {"pid": child.pid, "model": model, "phase": "aime_rematch",
                            "suite": "aime26", "attempt": "rematch_attempt001",
                            "started": start}
            (base / "private/active_worker.json").write_text(json.dumps(worker_state) + "\n")
            rc = child.wait()
        write(out / f"{model}_process_receipt.json", {**worker_state, "returncode": rc,
              "finished": time.time(), "allocated_worker_seconds": time.time() - start})
        if rc:
            raise RuntimeError(f"{model} failed; preserve attempt")
    write(out / "generation_complete.json", {"status": "complete", "rerun_requests": 180})


if __name__ == "__main__":
    main()
