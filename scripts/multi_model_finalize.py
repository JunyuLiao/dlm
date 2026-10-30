"""Wait for full primary coverage, then qualify diagnostics, score and audit.

This process never repairs or relaunches a primary worker and never signals one.
Any failed diagnostic/scoring/audit leaves the root completion marker absent.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multi_model_worker import ARMS, CAPS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage(root):
    completed = {}
    for model, suites in CAPS.items():
        for suite in suites:
            expected = 90 if suite == "aime26" else 164 if suite == "humaneval" else 130
            for arm in ARMS:
                cells = []
                for shard in range(1, 5):
                    markers = sorted((root / "runs" / model / suite / arm / f"shard{shard:03}").glob("attempt*/final_complete.json"))
                    if not markers:
                        break
                    rows = (markers[0].parent / "records.jsonl").read_text().splitlines()
                    cells.extend((r["id"], r["seed"]) for r in map(json.loads, rows))
                if len(cells) == len(set(cells)) == expected:
                    completed[f"{model}/{suite}/{arm}"] = len(cells)
    return completed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--attempt", default="attempt001")
    a = p.parse_args()
    root = a.root.resolve()
    source = Path(__file__).resolve().parents[1]
    leaf = root / "finalization" / a.attempt
    leaf.mkdir(parents=True, exist_ok=False)
    (leaf / "config.json").write_text(json.dumps({
        "source_commit": source.name, "source_sha256": sha(Path(__file__)),
        "expected_primary_requests": 5110, "expected_panels": 40,
        "monitor_interval_seconds": 900, "gpu_hour_limit": None,
        "started": time.time(), "pid": os.getpid(),
    }, indent=2) + "\n")
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
                       HF_MODULES_CACHE=str(root / "private/hf_modules"),
                       TRITON_CACHE_DIR=str(root / "private/triton_cache"))

    def run(script, arguments, label, env=environment, model=None):
        log = root / "private/finalization" / a.attempt / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            started = time.time()
            child = subprocess.Popen([sys.executable, str(source / "scripts" / script), *arguments],
                                     env=env, stdout=handle, stderr=subprocess.STDOUT)
            if model is not None:
                active = {"pid": child.pid, "model": model, "phase": "reuse_diagnostic",
                          "attempt": a.attempt, "started": started}
                (root / "private/active_worker.json").write_text(json.dumps(active) + "\n")
            returncode = child.wait()
        with (leaf / "process_receipts.jsonl").open("a") as handle:
            handle.write(json.dumps({"label": label, "pid": child.pid, "returncode": returncode,
                                     "model": model, "started": started, "finished": time.time(),
                                     "allocated_worker_seconds": time.time() - started}) + "\n")
        if returncode:
            raise RuntimeError(f"finalization child failed: {label}/{returncode}")

    try:
        while True:
            panels = coverage(root)
            print(json.dumps({"event": "waiting_for_full_primary_coverage", "panels": len(panels),
                              "requests_in_complete_panels": sum(panels.values()),
                              "time": time.time()}), flush=True)
            if len(panels) == 40 and sum(panels.values()) == 5110:
                gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                     capture_output=True, text=True, check=True)
                if not gpu.stdout.strip():
                    break
            time.sleep(60)
        verification = "campaign_" + a.attempt
        # Audit before adding any further GPU work. Verification is repeated
        # after diagnostics so it also protects the final publication boundary.
        run("multi_model_verify.py", [str(root), "--attempt", verification + "_before"], "verify_before")
        for model in CAPS:
            gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, check=True)
            if gpu.stdout.strip():
                raise RuntimeError("unexpected GPU worker before development diagnostic")
            diagnostic = root / "reuse_diagnostic" / model / a.attempt
            env = dict(environment)
            compat = root / "private/compat" / model
            if compat.exists():
                env["PYTHONPATH"] = str(compat) + os.pathsep + env["PYTHONPATH"]
            env["MULTI_MODEL_STATS_QUERY_BLOCKS"] = "8" if model == "ultrallada" else "1"
            run("multi_model_reuse_diagnostic.py", [str(root), model, "--attempt", a.attempt],
                "reuse_" + model, env, model)
            receipt = json.loads((diagnostic / "receipt.json").read_text())
            if not receipt["measurements"] or not all(r["observer_output_equal"] and r["observer_counts_equal"]
                                                       for r in receipt["requests"]):
                raise RuntimeError("development observer qualification incomplete: " + model)
        analysis = "analysis_campaign_" + a.attempt
        run("multi_model_score.py", [str(root), "--output", analysis], "score")
        run("multi_model_verify.py", [str(root), "--attempt", verification + "_after"], "verify_after")
        result = {"status": "complete", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "primary_requests": 5110, "model_task_arm_panels": 40,
                  "analysis": analysis, "analysis_summary_sha256": sha(root / analysis / "summary.json"),
                  "paired_sha256": sha(root / analysis / "paired_vs_v31.json"),
                  "scoring_identity_sha256": sha(root / analysis / "scoring_identity.json"),
                  "verification": verification + "_after",
                  "verification_sha256": sha(root / "verification" / (verification + "_after") / "receipt.json"),
                  "protocol_sha256": sha(root / "protocol.json"),
                  "development_diagnostics": {m: sha(root / "reuse_diagnostic" / m / a.attempt / "receipt.json") for m in CAPS},
                  "clean_latency_claim": False, "gpu_hour_limit": None}
        with (leaf / "receipt.json").open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
        with (root / "final_complete.json").open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except BaseException as error:
        (leaf / "failure.json").write_text(json.dumps({"error": repr(error), "time": time.time()}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
