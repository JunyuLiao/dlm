"""Wait for normal worker completion, qualify a kernel variant, resume queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--supervisor-pid", type=int, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    source = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(source / "src") + os.pathsep + str(source),
                       HF_MODULES_CACHE=str(root / "private/hf_modules"),
                       TRITON_CACHE_DIR=str(root / "private/triton_cache"))
    while Path(f"/proc/{a.supervisor_pid}").exists():
        time.sleep(10)
    while True:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            break
        time.sleep(10)
    gate = root / "qualification/kernels/attempt005/receipt.json"
    if not gate.exists():
        log = root / "private/kernels_qualification005.log"
        with log.open("x") as handle:
            subprocess.run([sys.executable, str(source / "scripts/multi_model_qualify.py"),
                            str(root), "--attempt", "attempt005"], env=environment,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    if json.loads(gate.read_text())["status"] != "pass":
        raise RuntimeError("batched statistics qualification failed; preserve evidence")
    environment["MULTI_MODEL_STATS_QUERY_BLOCKS"] = "8"
    log = root / "private/queue004.log"
    print(json.dumps({"event": "qualified_resume", "statistics_query_blocks": 8,
                      "source": source.name, "time": time.time()}), flush=True)
    with log.open("x") as handle:
        subprocess.run([sys.executable, str(source / "scripts/multi_model_queue.py"), str(root)],
                       env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
