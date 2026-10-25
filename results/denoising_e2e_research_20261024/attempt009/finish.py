"""Completion-only controller with receipt-based cross-namespace waiting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    here = Path(__file__).resolve().parent
    root = here.parent / "attempt003"
    spec = json.loads((here / "binding.json").read_text())
    protocol = json.loads((root / "e2e_protocol.json").read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != spec["controller_sha256"]:
        raise RuntimeError("controller source drift")
    if hashlib.sha256((root / "e2e_protocol.json").read_bytes()).hexdigest() != spec["parent_protocol_sha256"]:
        raise RuntimeError("parent protocol drift")
    for path, expected in spec["sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"bound source drift: {path}")
    started = time.time()
    with (here / "launch_receipt.json").open("x") as out:
        json.dump({"started_unix": started, "parent_protocol_sha256": spec["parent_protocol_sha256"],
                   "source_checkpoint": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "scope": "same25questions2seeds9arms; completion only; no expansion",
                   "previous_controller": "attempt007 interrupted; attempt008 namespace check failed and preserved"}, out, indent=2)
    targets = [arm for arm in protocol["arms"] if arm != "v31_optimized"]
    last_signature = None
    last_change = time.time()
    try:
        while True:
            state = []
            for arm in targets:
                leaf = root / "runs/panel/long" / arm
                state.append((arm, len(list(leaf.glob("audit???.json"))), len(list(leaf.glob("clean???.json"))),
                              (leaf / "final_complete.json").exists()))
                receipt = root / "launches/long" / arm / "exit_receipt.json"
                if receipt.exists() and json.loads(receipt.read_text()).get("returncode"):
                    raise RuntimeError(f"non-V31 long worker failed: {arm}")
            signature = tuple(state)
            if signature != last_signature:
                last_signature, last_change = signature, time.time()
            with (here / "progress.jsonl").open("a") as out:
                out.write(json.dumps({"unix": time.time(), "nonv31_long": [
                    {"arm": a, "audit_records": n, "clean_records": c, "complete": done}
                    for a, n, c, done in state]}) + "\n")
            if all(done for _, _, _, done in state):
                break
            if time.time() - last_change > 14400:
                raise RuntimeError("no non-V31 completion artifact for four hours")
            time.sleep(30)

        env = os.environ.copy()
        env.update(PYTHONPATH="/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm", HF_HUB_OFFLINE="1")
        python = "/home/exouser/dyh/v31_env_20261005/bin/python"
        jobs = [
            ("v31_long", here.parent / "attempt005/launch_v31.py", ["--suite", "long"], False),
            ("official_score", root / "score_v3.py", ["--root", str(root)], True),
            ("analysis", root / "analyze_v4.py", ["--root", str(root)], True),
            ("paired_trajectory", root / "paired_trajectory.py", ["--root", str(root)], True),
            ("native_bottlenecks", root / "trajectory_bottlenecks.py", ["--root", str(root)], True),
            ("support_diversity", root / "support_diversity.py", ["--root", str(root)], True),
            ("output_diagnostics", root / "output_diagnostics.py", ["--root", str(root)], True),
            ("plots", root / "plot_trajectories.py", ["--root", str(root)], True),
            ("verify", root / "verify.py", ["--root", str(root)], True),
            ("report", root / "report.py", ["--root", str(root)], True),
        ]
        commands = []
        for name, script, extra, cpu_only in jobs:
            job_env = dict(env)
            if cpu_only:
                job_env["CUDA_VISIBLE_DEVICES"] = ""
            command = [python, str(script), *extra]
            job_start = time.time()
            commands.append({"name": name, "command": command, "cpu_only": cpu_only, "started_unix": job_start})
            with (here / "private" / f"{name}.log").open("x") as log:
                result = subprocess.run(command, env=job_env, stdout=log, stderr=subprocess.STDOUT)
            receipt = {"name": name, "returncode": result.returncode, "elapsed_seconds": time.time() - job_start,
                       "command": command, "cpu_only": cpu_only}
            with (here / f"{name}_receipt.json").open("x") as out:
                json.dump(receipt, out, indent=2)
            print(json.dumps(receipt), flush=True)
            if result.returncode:
                raise RuntimeError(f"completion stage failed: {name}")
        with (here / "final_complete.json").open("x") as out:
            json.dump({"passed": True, "scope": "execution/scoring/readout completion; publication remains primary-reviewed",
                       "parent_protocol_sha256": spec["parent_protocol_sha256"], "elapsed_seconds": time.time() - started,
                       "commands": commands, "method_promoted": False}, out, indent=2)
    except BaseException as error:
        with (here / "failed_preserved.json").open("w") as out:
            json.dump({"error_class": type(error).__name__, "reason": str(error), "unix": time.time(),
                       "automatic_retry": False}, out, indent=2)
        raise


if __name__ == "__main__":
    main()
