"""Reorder unstarted cells after the last LLaDA AIME shard finishes normally."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--supervisor-pid", type=int, required=True)
    p.add_argument("--attempt", default="priority_attempt001")
    p.add_argument("--qualify-held-counts", action="store_true")
    p.add_argument("--candidate-source", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    leaf = root / "scheduling" / a.attempt
    leaf.mkdir(parents=True, exist_ok=False)
    proc = Path("/proc") / str(a.supervisor_pid)
    start = proc.joinpath("stat").read_text().split()[21]
    coordinator = root / "private/execution/8c0d14b031636ddcc708df75fa1d57aefa8b0ba4"
    bindings = root / "private/model_worker_bindings001.json"
    candidate = a.candidate_source.resolve() if a.candidate_source else Path(__file__).resolve().parents[1]
    stages = [(["trado", "ultrallada"], ["humaneval"]),
              (["trado"], ["aime26"]),
              (["llada22"], ["ruler128k"]),
              (["ultrallada"], ["aime26", "ruler128k"])]
    config = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "supervisor_pid": a.supervisor_pid, "supervisor_start_ticks": start,
              "stages": [{"models": m, "suites": s} for m, s in stages],
              "qualify_held_counts_before_unstarted_models": a.qualify_held_counts,
              "candidate_source_commit": candidate.name if a.qualify_held_counts else None,
              "worker_sources_sha256": hashlib.sha256(bindings.read_bytes()).hexdigest(),
              "scientific_settings_changed": False, "gpu_workers_signaled": False,
              "reason": "qualify full short-benchmark adapters before longest full-horizon panels"}
    (leaf / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    while proc.exists():
        try:
            active = json.loads((root / "private/active_worker.json").read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(1)
            continue
        child_proc = Path("/proc") / str(active["pid"])
        if active.get("model") == "llada22" and active.get("suite") == "aime26" and child_proc.exists():
            try:
                argv = child_proc.joinpath("cmdline").read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            argv = [x.decode() for x in argv if x]
            is_last = ("--arm" in argv and argv[argv.index("--arm") + 1] == "value_v3a_singleton_reuse_r4"
                       and "--shard" in argv and argv[argv.index("--shard") + 1] == "4")
            if is_last:
                command = proc.joinpath("cmdline").read_bytes().decode().replace("\0", " ")
                if "multi_model_queue.py" not in command or str(coordinator) not in command:
                    raise RuntimeError("supervisor identity mismatch")
                if proc.joinpath("stat").read_text().split()[21] != start:
                    raise RuntimeError("supervisor PID identity changed")
                if int(child_proc.joinpath("stat").read_text().split()[3]) != a.supervisor_pid:
                    raise RuntimeError("last shard is not supervised by expected coordinator")
                os.kill(a.supervisor_pid, signal.SIGTERM)
                (leaf / "boundary_stop_requested.json").write_text(json.dumps({
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "supervisor_pid": a.supervisor_pid, "finishing_gpu_worker_pid": active["pid"],
                    "supervisor_start_ticks": start, "gpu_worker_signaled": False}, indent=2) + "\n")
                break
        time.sleep(10)
    while proc.exists():
        time.sleep(10)
    parent = root / "runs/llada22/aime26/value_v3a_singleton_reuse_r4"
    if any(not list((parent / f"shard{s:03}").glob("attempt*/final_complete.json")) for s in range(1, 5)):
        raise RuntimeError("last LLaDA AIME shard did not complete; preserve failure")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True)
    if gpu.stdout.strip():
        raise RuntimeError("GPU occupied at scheduling boundary")
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join((str(coordinator / "src"), str(coordinator))),
                       MULTI_MODEL_STATS_QUERY_BLOCKS="8")
    if a.qualify_held_counts:
        env = dict(environment)
        env.update(PYTHONPATH=os.pathsep.join((str(root / "private/compat/ultrallada"), str(candidate / "src"), str(candidate))),
                   HF_MODULES_CACHE=str(root / "private/hf_modules"),
                   TRITON_CACHE_DIR=str(root / "private/triton_cache"))
        log = root / "private/held_counts_qualification006.log"
        with log.open("x") as handle:
            child = subprocess.Popen([sys.executable, str(candidate / "scripts/multi_model_qualify_held_counts.py"),
                                      str(root), "--attempt", "attempt006"],
                                     env=env, stdout=handle, stderr=subprocess.STDOUT)
            (root / "private/active_worker.json").write_text(json.dumps({
                "pid": child.pid, "model": "ultrallada", "phase": "qualification",
                "attempt": "held_counts_attempt006", "started": time.time()}) + "\n")
            rc = child.wait()
        gate = root / "qualification/held_counts/attempt006/receipt.json"
        enabled = rc == 0 and gate.exists() and json.loads(gate.read_text())["status"] == "pass"
        if enabled:
            values = json.loads(bindings.read_text())
            values["ultrallada"] = candidate.name
            bindings = root / "private/model_worker_bindings002.json"
            with bindings.open("x") as handle:
                handle.write(json.dumps(values, indent=2) + "\n")
        (leaf / "held_counts_optimization.json").write_text(json.dumps({
            "status": "enabled" if enabled else "qualification_failed_original_source_retained",
            "model": "ultrallada", "candidate_source_commit": candidate.name,
            "qualification_returncode": rc,
            "qualification_sha256": hashlib.sha256(gate.read_bytes()).hexdigest() if gate.exists() else None,
            "numerical_worker_sources_sha256": hashlib.sha256(bindings.read_bytes()).hexdigest(),
            "existing_llada_and_trado_sources_changed": False}, indent=2) + "\n")
    for index, (models, suites) in enumerate(stages, 1):
        log = root / "private" / f"priority_stage{index:03}.log"
        with log.open("x") as handle:
            result = subprocess.run([sys.executable, str(coordinator / "scripts/multi_model_queue.py"),
                                     str(root), "--models", *models, "--suites", *suites,
                                     "--model-source-bindings", str(bindings)],
                                    env=environment, stdout=handle, stderr=subprocess.STDOUT)
        receipt = {"stage": index, "models": models, "suites": suites,
                   "returncode": result.returncode, "time": time.time()}
        with (leaf / "stage_receipts.jsonl").open("a") as handle:
            handle.write(json.dumps(receipt) + "\n")
        if result.returncode:
            raise RuntimeError("priority queue stage failed: " + str(index))
    (leaf / "final_complete.json").write_text(json.dumps({"status": "schedule_complete", "time": time.time()}) + "\n")


if __name__ == "__main__":
    main()
