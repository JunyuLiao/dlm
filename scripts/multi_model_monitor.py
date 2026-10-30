"""Append host GPU/process health every 15 minutes; never signal workloads."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def query(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def read_state(path):
    # Older frozen workers publish this small file without an atomic rename.
    # A reader at a worker transition must retain driver monitoring.
    for attempt in range(3):
        try:
            return json.loads(path.read_text()), None
        except (OSError, json.JSONDecodeError) as error:
            last_error = type(error).__name__
            if attempt < 2:
                time.sleep(.05)
    return None, last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    health = root / "health"
    health.mkdir(parents=True, exist_ok=True)
    while True:
        record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "monitor_pid": os.getpid(), "interval_seconds": args.interval,
                  "gpu": query(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"]),
                  "compute_processes": query(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"])}
        state = root / "private" / "active_worker.json"
        if state.exists():
            value, error = read_state(state)
            if error:
                record["worker_state_read_error"] = error
        else:
            value = None
        if value is not None:
            pid = int(value["pid"])
            record["worker"] = {k: value[k] for k in ("pid", "model", "phase", "attempt") if k in value}
            record["worker"]["liveness"] = query(["ps", "-p", str(pid), "-o", "pid=,ppid=,lstart=,etime=,pcpu=,pmem=,comm="])
            forward = root / "private/forward_heartbeats" / f"worker{pid}.json"
            if forward.exists():
                heartbeat, error = read_state(forward)
                if heartbeat is not None:
                    record["worker"]["forward_heartbeat_age_seconds"] = time.time() - heartbeat["time"]
                    record["worker"]["denoising_forwards_in_request"] = heartbeat["counts"].get("denoising_forwards")
                    record["worker"]["block_start"] = heartbeat["block_start"]
                else:
                    record["worker"]["forward_state_read_error"] = error
        record_files = list(root.glob("runs/**/records.jsonl"))
        record["record_files"] = len(record_files)
        record["complete_shards"] = len(list(root.glob("runs/**/final_complete.json")))
        record["record_lines"] = sum(sum(1 for line in p.open() if line.strip()) for p in record_files)
        if record_files:
            record["latest_progress_age_seconds"] = time.time() - max(p.stat().st_mtime for p in record_files)
        record["final_complete"] = (root / "final_complete.json").exists()
        with (health / "gpu_health.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.once or record["final_complete"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
