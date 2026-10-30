"""Download immutable public model snapshots into the authorized campaign leaf."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


MODELS = {
    "llada22": "inclusionAI/LLaDA2.2-mini",
    "ultrallada": "relaxe-system-lab/UltraLLaDA",
    "trado": "Gen-Verse/TraDo-8B-Instruct",
}


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    private = root / "private"
    private.mkdir(parents=True, exist_ok=True)
    identities = {}
    for key, repo in MODELS.items():
        info = HfApi().model_info(repo)
        identity = {"repo_id": repo, "revision": info.sha,
                    "files": [x.rfilename for x in info.siblings]}
        pin = private / f"{key}_model_pin.json"
        if pin.exists():
            old = json.loads(pin.read_text())
            identity = old
        else:
            pin.write_text(json.dumps(identity, indent=2) + "\n")
        print(json.dumps({"event": "download_start", "model": key,
                          "revision": identity["revision"], "time": time.time()}), flush=True)
        target = private / "models" / key
        snapshot_download(repo, revision=identity["revision"], local_dir=target,
                          allow_patterns=["*.json", "*.py", "*.safetensors", "*.model",
                                          "*.txt", "*.tiktoken", "*.jinja", "README.md"],
                          max_workers=4)
        fingerprints = {}
        for path in sorted(target.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                fingerprints[str(path.relative_to(target))] = {
                    "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        identities[key] = {"repo_id": repo, "revision": identity["revision"],
                           "files": fingerprints}
        (root / f"{key}_model_identity.json").write_text(
            json.dumps(identities[key], indent=2) + "\n")
        print(json.dumps({"event": "download_complete", "model": key,
                          "files": len(fingerprints), "time": time.time()}), flush=True)
    (root / "models_complete.json").write_text(json.dumps(identities, indent=2) + "\n")


if __name__ == "__main__":
    main()
