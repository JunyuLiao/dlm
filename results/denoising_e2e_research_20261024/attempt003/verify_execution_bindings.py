"""Supplement the frozen closeout with explicit operational source bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "binding_review001"
    out.mkdir(exist_ok=False)
    assert read(root / "final_complete.json")["passed"]
    protocol_hash = sha(root / "e2e_protocol.json")
    bindings = {}
    for name, worker, launcher in [
        ("attempt005", "code/v31_worker.py", "launch_v31.py"),
        ("attempt006", "code/long_worker.py", "launch_long.py"),
    ]:
        directory = root.parent / name
        binding = read(directory / "execution_binding.json")
        assert binding["parent_protocol_sha256"] == protocol_hash
        assert binding["worker_sha256"] == sha(directory / worker)
        assert binding["launcher_sha256"] == sha(directory / launcher)
        assert all(sha(path) == expected for path, expected in binding["native_method_sources"].items())
        bindings[name] = sha(directory / "execution_binding.json")
    assert read(root.parent / "attempt006/execution_binding.json")["V31_separate_binding_sha256"] == bindings["attempt005"]
    protocol = read(root / "e2e_protocol.json")
    directory_binding = read(root / "result_directory_bindings.json")
    cells = []
    for suite in ["short", "long"]:
        for arm in protocol["arms"]:
            expected_binding = bindings["attempt005"] if arm == "v31_optimized" else bindings["attempt006"] if suite == "long" else None
            leaf = root / directory_binding["result_paths"].get(suite, {}).get(arm, f"runs/panel/{suite}/{arm}")
            launch = root / directory_binding["launch_paths"].get(suite, {}).get(arm, f"launches/{suite}/{arm}")
            marker = read(leaf / "final_complete.json")
            receipt = read(launch / "launch_receipt.json")
            if expected_binding is not None:
                assert marker["execution_binding_sha256"] == expected_binding
                assert receipt["execution_binding_sha256"] == expected_binding
            for clean_path in sorted(leaf.glob("clean[0-9][0-9][0-9].json")):
                clean = read(clean_path)
                audit = read(clean_path.with_name(clean_path.name.replace("clean", "audit")))
                if expected_binding is not None:
                    assert clean["execution_binding_sha256"] == audit["execution_binding_sha256"] == expected_binding
                assert clean["cuda_capture_events"] == clean["request_jit_events"] == 0
                cells.append(dict(suite=suite, arm=arm, id=clean["id"], seed=clean["seed"], binding=expected_binding,
                                  clean_zero_new_capture_and_jit=True))
    assert len(cells) == 450
    prior_runtime = read(root / "runtime_source_audit001.json")
    runtime_root = Path("/home/exouser/dyh/v31_env_20261005/lib/python3.12/site-packages/vllm")
    assert all(sha(runtime_root / path) == expected for path, expected in prior_runtime["files"].items())
    result = dict(passed=True, checked_unix=time.time(), protocol_sha256=protocol_hash,
                  verifier_sha256=sha(Path(__file__)), execution_bindings=bindings,
                  runtime_sources_match_running_audit=True, cells=cells, method_promoted=False,
                  scope="Operational-source and clean-timing supplement; no new generation or score rule")
    (out / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(passed=True, cells=len(cells), bindings=len(bindings), runtime_files=len(prior_runtime["files"]))))


if __name__ == "__main__":
    main()
