"""Serial, resumable orchestration with immutable attempts and no time ceiling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multi_model_worker import ARMS, CAPS


def next_attempt(parent):
    n = 1
    while (parent / f"attempt{n:03}").exists():
        n += 1
    return f"attempt{n:03}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--models", nargs="+", default=list(CAPS))
    p.add_argument("--suites", nargs="+", default=["humaneval", "aime26", "ruler128k"])
    p.add_argument("--calibration-only", action="store_true")
    p.add_argument("--wait-for-idle", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    source = Path(__file__).resolve().parents[1]
    queue_attempt = next_attempt(root / "queue")
    leaf = root / "queue" / queue_attempt
    leaf.mkdir(parents=True)
    config = {"models": a.models, "suites": a.suites, "calibration_only": a.calibration_only,
              "source_binding": source.name, "pid": os.getpid(), "started": time.time(),
              "gpu_hour_limit": None, "monitor_interval_seconds": 900}
    (leaf / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for model in a.models:
        qualification = sorted((root / "qualification" / model).glob("attempt*/receipt.json"))
        if not qualification:
            raise RuntimeError(f"model qualification absent: {model}")
        for suite in a.suites:
            if suite not in CAPS[model]:
                continue
            data = root / "private/data" / model / f"{suite}.json"
            if not data.exists():
                print(json.dumps({"event": "waiting_for_data", "model": model, "suite": suite}), flush=True)
                while not data.exists():
                    time.sleep(30)
            environment = dict(os.environ)
            paths = [str(source / "src"), str(source)]
            compat = root / "private/compat" / model
            if compat.exists():
                paths.insert(0, str(compat))
            environment.update(PYTHONPATH=os.pathsep.join(paths),
                               HF_MODULES_CACHE=str(root / "private/hf_modules"),
                               TRITON_CACHE_DIR=str(root / "private/triton_cache"))
            def run(command, log_name):
                # Driver and competing-worker check before every model child.
                while True:
                    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
                    if not gpu.stdout.strip():
                        break
                    if not a.wait_for_idle:
                        raise RuntimeError("GPU has an active worker before launch: " + gpu.stdout.strip())
                    print(json.dumps({"event": "waiting_for_current_worker_exit", "gpu_pids": gpu.stdout.strip()}), flush=True)
                    time.sleep(10)
                log = root / "private/logs" / model / suite / log_name
                log.parent.mkdir(parents=True, exist_ok=True)
                started = time.time()
                with log.open("x") as handle:
                    child = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                    active = {"pid": child.pid, "model": model, "suite": suite,
                              "phase": "calibration" if "--calibrate" in command else "benchmark",
                              "attempt": command[command.index("--attempt") + 1]}
                    (root / "private/active_worker.json").write_text(json.dumps(active) + "\n")
                    # Supervisor termination never signals its current GPU child.
                    # A replacement waits for a normal completed shard boundary.
                    import signal
                    stop = [False]
                    def supervisor_stop(signum, frame):
                        stop[0] = True
                    previous = signal.signal(signal.SIGTERM, supervisor_stop)
                    rc = child.wait()
                    signal.signal(signal.SIGTERM, previous)
                process_receipt = {"event": "worker_exit", **active, "returncode": rc,
                                   "started": started, "finished": time.time(),
                                   "allocated_worker_seconds": time.time() - started}
                with (leaf / "process_receipts.jsonl").open("a") as handle:
                    handle.write(json.dumps(process_receipt) + "\n")
                print(json.dumps(process_receipt), flush=True)
                if stop[0]:
                    (leaf / "stopped_at_worker_boundary.json").write_text(json.dumps(process_receipt, indent=2) + "\n")
                    raise SystemExit(0)
                if rc:
                    (leaf / "worker_failure.json").write_text(json.dumps(dict(active, returncode=rc), indent=2) + "\n")
                    raise RuntimeError(f"worker failed: {model}/{suite}/{log_name}")
            selected = sorted((root / "calibration" / model / suite).glob("attempt*/selected_threshold.json"))
            if selected:
                threshold = json.loads(selected[-1].read_text())
            else:
                attempt = next_attempt(root / "calibration" / model / suite)
                run([sys.executable, str(source / "scripts/multi_model_worker.py"), str(root),
                     model, suite, "--calibrate", "--attempt", attempt], f"calibration_{attempt}.log")
                threshold = json.loads((root / "calibration" / model / suite / attempt / "selected_threshold.json").read_text())
            if not threshold["in_range"]:
                raise RuntimeError(f"calibration misses 48-52%: {model}/{suite}; preserve evidence for repair")
            if a.calibration_only:
                continue
            for arm in ARMS:
                for shard in range(1, 5):
                    parent = root / "runs" / model / suite / arm / f"shard{shard:03}"
                    if list(parent.glob("attempt*/final_complete.json")):
                        continue
                    attempt = next_attempt(parent)
                    run([sys.executable, str(source / "scripts/multi_model_worker.py"), str(root),
                         model, suite, "--arm", arm, "--shard", str(shard), "--shards", "4",
                         "--threshold", str(threshold["selected_threshold"]), "--attempt", attempt],
                        f"{arm}_shard{shard:03}_{attempt}.log")
    (leaf / "final_complete.json").write_text(json.dumps({"status": "complete", "scope": config,
                                                         "finished": time.time()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
