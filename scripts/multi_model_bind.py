"""Freeze the current committed Python execution tree in a private leaf."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    target = a.root / "private/execution" / commit
    if target.exists():
        print(target.resolve())
        return
    target.mkdir(parents=True)
    for directory in ("experiments/numerical_qk_reuse", "experiments/value_direction_hopper", "src/dllm", "scripts"):
        shutil.copytree(directory, target / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"))
    identities = {}
    for f in sorted(target.rglob("*")):
        if f.is_file():
            identities[str(f.relative_to(target))] = hashlib.sha256(f.read_bytes()).hexdigest()
    receipt = {"commit": commit, "files": identities}
    (target / "source_identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(target.resolve())


if __name__ == "__main__":
    main()
