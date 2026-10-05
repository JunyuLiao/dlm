"""v11 preselected-support Hopper consumer: explicit build, verified load, call.

Build is explicit (never during generation). Output: a C-ABI kernel .so
(nvcc, sm_90a) plus an ATen bridge in the private namespace ``vd_support_v1``,
disjoint from Junyu's ``value_direction_hopper`` so both load in one process.
Every artifact is named by a hash of sources + compiler + options + torch, and
``build_identity.json`` records them; ``load`` re-verifies before loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

SOURCE = Path(__file__).parent / 'csrc'
FMHA = Path('/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/sources_db080045f7a5fbce')
CUTLASS = Path('/home/exouser/chw/dlm_test/cutlass/include')
NVCC = Path('/usr/local/cuda/bin/nvcc')
ABI_VERSION = 1
_LOADED = None


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(build_dir: Path, *, counters_only=False) -> dict:
    files = [SOURCE / 'support_consumer.cu', SOURCE / 'support_consumer.h', SOURCE / 'support_bridge.cpp',
             FMHA / 'fmha/hopper/utils_hgmma_bf16.h', FMHA / 'fmha/hopper/utils_warpgroup.h']
    hashes = {p.name: sha256(p) for p in files}
    nvcc_version = subprocess.run([str(NVCC), '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--shared', '-Xcompiler=-fPIC', '-lineinfo', '--ptxas-options=-v']
    key = hashlib.sha256(json.dumps([hashes, nvcc_version, flags, torch.__version__, ABI_VERSION],
                                    sort_keys=True).encode()).hexdigest()[:16]
    out = Path(build_dir) / f'support_{key}'
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'sources'
    snapshot.mkdir(exist_ok=True)
    for p in files[:3]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    kernel = out / f'support_consumer_{key}.so'
    cmd = [str(NVCC), *flags, '-I' + str(CUTLASS), '-I' + str(FMHA), '-I/usr/local/cuda/include/cccl',
           str(snapshot / 'support_consumer.cu'), '-o', str(kernel), '-L/usr/local/cuda/lib64', '-lcudart',
           '-Xlinker=-rpath,/usr/local/cuda/lib64']
    started = time.monotonic()
    if not kernel.exists():
        result = subprocess.run(cmd, capture_output=True, text=True)
        (out / 'nvcc.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'nvcc failed; see {out / "nvcc.log"}')
    nvcc_seconds = time.monotonic() - started
    # Direct g++ build with the same flags torch's cpp_extension produced for
    # Junyu's bridge (no ninja in this environment; nothing is installed).
    from torch.utils.cpp_extension import include_paths, library_paths
    import sysconfig
    bridge_dir = out / 'bridge'
    bridge_dir.mkdir(exist_ok=True)
    bridge = bridge_dir / f'vd_support_bridge_{key}.so'
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    bridge_cmd = (['c++', '-shared', '-fPIC', '-std=c++17', '-O3', f'-D_GLIBCXX_USE_CXX11_ABI={abi}',
                   '-DTORCH_API_INCLUDE_EXTENSION_H', f'-DTORCH_EXTENSION_NAME=vd_support_bridge_{key}']
                  + [f'-isystem{path}' for path in include_paths()]
                  + ['-isystem' + sysconfig.get_paths()['include'], '-I/usr/local/cuda/include', '-I' + str(snapshot),
                     str(snapshot / 'support_bridge.cpp'), str(kernel), '-L/usr/local/cuda/lib64', '-lcudart']
                  + [f'-L{path}' for path in library_paths()]
                  + ['-lc10_cuda', '-lc10', '-ltorch_cpu', '-ltorch', '-ltorch_python',
                     '-Wl,-rpath,' + str(out), '-Wl,-rpath,/usr/local/cuda/lib64', '-o', str(bridge)])
    started = time.monotonic()
    if not bridge.exists():
        result = subprocess.run(bridge_cmd, capture_output=True, text=True)
        (out / 'bridge.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'bridge build failed; see {out / "bridge.log"}')
    identity = dict(schema='v11_support_build_v1', key=key, abi_version=ABI_VERSION, sources=hashes,
                    nvcc=str(NVCC), nvcc_version=nvcc_version, flags=flags, command=cmd, torch=torch.__version__,
                    cutlass=str(CUTLASS), fmha_headers=str(FMHA), kernel=str(kernel), kernel_sha256=sha256(kernel),
                    bridge=str(bridge), bridge_sha256=sha256(bridge), namespace='vd_support_v1', bridge_command=bridge_cmd,
                    nvcc_seconds=nvcc_seconds, bridge_seconds=time.monotonic() - started)
    (out / 'build_identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    return identity


def load(identity_path) -> dict:
    """Verify hashes and torch version, then load the bridge exactly once per process."""
    global _LOADED
    identity = json.loads(Path(identity_path).read_text())
    if identity['torch'] != torch.__version__:
        raise RuntimeError('support bridge built for a different torch')
    for name in ('kernel', 'bridge'):
        if sha256(identity[name]) != identity[f'{name}_sha256']:
            raise RuntimeError(f'{name} binary differs from its build identity')
    if _LOADED is not None:
        if _LOADED['bridge'] != identity['bridge']:
            raise RuntimeError('a different vd_support_v1 bridge is already loaded in this process')
        return _LOADED
    torch.ops.load_library(identity['bridge'])
    _LOADED = identity
    return identity


def attention(q, k, v, skipped, eligible, *, scale, window=0, mask=None, layout=0, counters=False):
    """Returns (output [B,H,Q,D] view, lse [B,H,Q], invalid [B,H,Q] bool, counters)."""
    if _LOADED is None:
        raise RuntimeError('call support.load(build_identity.json) first')
    return torch.ops.vd_support_v1.attention(q, k, v, skipped, eligible, mask, int(window), float(scale),
                                             int(layout), bool(counters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.build_dir), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
