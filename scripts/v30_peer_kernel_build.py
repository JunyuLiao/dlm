"""Explicit CPU build of a pinned, isolated Junyu kernel; never runs a GPU.

Inputs/output paths are supplied in a private JSON config. Original peer source
is read-only; the sole ABI guard patch is written to a new source copy. This
builder is separate from the peer's hard-coded toolchain, not a method change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PEER_COMMIT = "b890ff49488474c5d476df44019597dc045a5969"
TRT_COMMIT = "7965842954628b0b5456a2f7d59786d3dcd41647"
OLD_GUARD = "self.abi_version!=3 or debug_scores.shape"
NEW_GUARD = "self.abi_version not in (3,4) or debug_scores.shape"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_debug_abi(source):
    if source.count(OLD_GUARD) != 1 or NEW_GUARD in source:
        raise ValueError("Expected exactly the reviewed peer ABI3-only guard")
    return source.replace(OLD_GUARD, NEW_GUARD)


def contained_new_directory(path, own_root):
    path, own_root = Path(path).resolve(), Path(own_root).resolve()
    if path == own_root or not path.is_relative_to(own_root):
        raise ValueError("Output must be a new child of the explicit own root")
    if path.exists():
        raise FileExistsError("Use a new attempt directory; preserve old attempts")
    return path


def validate_config(config):
    if config["peer_commit"] != PEER_COMMIT or config["trt_commit"] != TRT_COMMIT:
        raise ValueError("Unreviewed source pin")
    # Start with the original builder's numerical defaults; faster flags require
    # a separately named and numerically qualified experiment.
    if config["flags"] != dict(fast_sfu=False, inline_roles=False, online_ratio=False):
        raise ValueError("This build freezes peer default numerical flags")
    src = Path(config["source"])
    required = {"cuda.py", "csrc/value_direction.cu", "csrc/value_direction.h", "csrc/torch_bridge.cpp"}
    if set(config["source_hashes"]) != required:
        raise ValueError("Exact reviewed source inventory required")
    for rel, expected in config["source_hashes"].items():
        if digest(src / rel) != expected:
            raise ValueError("Peer source bytes do not match frozen export")
    if "inline constexpr uint32_t ABI_VERSION=4;" not in (src / "csrc/value_direction.h").read_text():
        raise ValueError("Expected ABI4 header")
    inputs = {
        "nvcc": Path(config["cuda_home"]) / "bin/nvcc",
        "cutlass": Path(config["cutlass_include"]) / "cute/tensor.hpp",
        "cccl": Path(config["cuda_home"]) / "include/cccl/cuda/std/type_traits",
        "runtime": Path(config["cuda_home"]) / "lib64/libcudart.so",
        "trt": Path(config["trt_source"]) / "cpp/kernels/fmha_v2/src/fmha/hopper/utils_hgmma_bf16.h",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError("Missing " + name + " build input")
    return inputs


def header_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def build(config):
    inputs = validate_config(config)
    out = contained_new_directory(config["output"], config["own_root"])
    out.mkdir(parents=True, exist_ok=False)
    os.umask(0o077)
    src = Path(config["source"])
    cuda = Path(config["cuda_home"])
    copied = out / "source"
    copied.mkdir()
    for rel in config["source_hashes"]:
        dest = copied / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dest)
    guard_file = copied / "cuda.py"
    guard_file.write_text(patch_debug_abi(guard_file.read_text()))
    command = [str(inputs["nvcc"]), "-std=c++17", "-O3", "-arch=sm_90a", "--shared",
               "-Xcompiler=-fPIC", "-lineinfo", "--ptxas-options=-v",
               "-I" + config["cutlass_include"], "-I" + str(Path(config["trt_source"]) / "cpp/kernels/fmha_v2/src"),
               "-I" + str(cuda / "include/cccl"), str(copied / "csrc/value_direction.cu"),
               "-o", str(out / "value_direction.so"), "-L" + str(cuda / "lib64"),
               "-lcudart", "-lcuda", "-Xlinker=-rpath," + str(cuda / "lib64")]
    # Entire selected header trees are recorded; direct-header-only hashes miss
    # transitive include changes. Paths and compiler logs remain private.
    manifest = {name: header_manifest(root) for name, root in {
        "trt_fmha": Path(config["trt_source"]) / "cpp/kernels/fmha_v2/src",
        "cutlass": config["cutlass_include"], "cuda": cuda / "include"}.items()}
    (out / "headers.private.json").write_text(json.dumps(manifest, sort_keys=True))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MAX_JOBS="1")
    version = subprocess.run([str(inputs["nvcc"]), "--version"], env=env, text=True, capture_output=True, check=True)
    (out / "config.private.json").write_text(json.dumps(config, indent=2))
    with (out / "stdout.private.log").open("w") as stdout, (out / "stderr.private.log").open("w") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    proof = dict(command=command, compiler_version=version.stdout, config=config,
                 returncode=result.returncode, original_source_hashes=config["source_hashes"],
                 patched_cuda_sha256=digest(guard_file), header_manifest_sha256=digest(out / "headers.private.json"))
    (out / "build.private.json").write_text(json.dumps(proof, indent=2))
    receipt = dict(peer_source_commit=PEER_COMMIT, trt_headers_commit=TRT_COMMIT,
                   flags=config["flags"], returncode=result.returncode, gpu_process_started=False,
                   gpu_reserved_seconds=0, kernel_qualified=False, e2e_executed=False,
                   scope="CPU compilation only; no GPU runtime import or launch",
                   source_patch="ABI3/ABI4 diagnostic guard in isolated Python wrapper only",
                   kernel_sha256=digest(out / "value_direction.so") if result.returncode == 0 else None,
                   compiler_version=version.stdout.strip())
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(build(json.loads(args.config.read_text())))
