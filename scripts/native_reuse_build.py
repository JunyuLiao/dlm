"""Explicit private build of pinned Junyu CUDA and its ATen bridge.

Run only after the three pinned csrc files and two FMHA headers have been
copied into ``--inputs``. All compiler output and provenance stay in
``--build-dir``. This script never changes an installed package or peer tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, *, env=None, log):
    start = time.monotonic()
    with Path(log).open('w', encoding='utf-8') as stream:
        proc = subprocess.run(command, text=True, stdout=stream,
                              stderr=subprocess.STDOUT, env=env)
    seconds = time.monotonic() - start
    if proc.returncode:
        raise RuntimeError(f'command failed ({proc.returncode}) after {seconds:.1f}s; see {log}')
    return seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--cutlass', type=Path,
                        default=Path('/home/exouser/chw/dlm_test/cutlass/include'))
    parser.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc'))
    args = parser.parse_args()
    import torch
    from torch.utils.cpp_extension import load

    inputs = args.inputs.resolve()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    rels = [Path('value_direction.cu'), Path('value_direction.h'),
            Path('torch_bridge.cpp'),
            Path('fmha/hopper/utils_hgmma_bf16.h'),
            Path('fmha/hopper/utils_warpgroup.h')]
    for rel in rels:
        if not (inputs / rel).is_file():
            raise FileNotFoundError(inputs / rel)
    if not args.cutlass.is_dir() or not args.nvcc.is_file():
        raise FileNotFoundError('CUDA compiler or read-only CUTLASS include missing')
    source_hashes = {str(rel): sha256(inputs / rel) for rel in rels}
    key = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = build / f'sources_{key}'
    for rel in rels:
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != source_hashes[str(rel)]:
                raise RuntimeError(f'immutable build snapshot differs: {target}')
        else:
            shutil.copyfile(inputs / rel, target)
    kernel = build / f'value_direction_{key}.so'
    kernel_cmd = [str(args.nvcc), '-std=c++17', '-O3', '-arch=sm_90a',
                  '--shared', '-Xcompiler=-fPIC', '-lineinfo', '--ptxas-options=-v',
                  '-I' + str(args.cutlass.resolve()), '-I' + str(snapshot),
                  '-I/usr/local/cuda/include/cccl', str(snapshot / 'value_direction.cu'),
                  '-o', str(kernel), '-L/usr/local/cuda/lib64', '-lcudart', '-lcuda',
                  '-Xlinker=-rpath,/usr/local/cuda/lib64']
    nvcc_seconds = 0.
    if not kernel.exists():
        nvcc_seconds = run(kernel_cmd, log=build / f'nvcc_{key}.log')
    bridge_sources = {str((snapshot / rel).resolve()): source_hashes[str(rel)]
                      for rel in (Path('torch_bridge.cpp'), Path('value_direction.h'))}
    bridge_sources[str(kernel.resolve())] = sha256(kernel)
    torch_key = hashlib.sha256(json.dumps([bridge_sources, torch.__version__],
                                          sort_keys=True).encode()).hexdigest()[:16]
    bridge_dir = build / f'torch_{torch_key}'
    bridge_dir.mkdir(exist_ok=True)
    bridge_name = 'value_direction_torch_' + torch_key
    bridge = bridge_dir / (bridge_name + '.so')
    bridge_seconds = 0.
    if not bridge.exists():
        start = time.monotonic()
        prior = os.environ.get('MAX_JOBS')
        os.environ['MAX_JOBS'] = '4'
        try:
            # Mirrors torch_build.py. load() also verifies the dynamic link.
            load(name=bridge_name, sources=[str((snapshot / 'torch_bridge.cpp').resolve())],
                 extra_cflags=['-O3', '-std=c++17', '-I/usr/local/cuda/include'],
                 extra_ldflags=[str(kernel.resolve()), '-L/usr/local/cuda/lib64',
                                '-lcudart', '-lc10_cuda',
                                '-Wl,-rpath,' + str(kernel.parent.resolve()),
                                '-Wl,-rpath,/usr/local/cuda/lib64'],
                 build_directory=str(bridge_dir), is_python_module=False, verbose=True)
        finally:
            if prior is None:
                os.environ.pop('MAX_JOBS', None)
            else:
                os.environ['MAX_JOBS'] = prior
        bridge_seconds = time.monotonic() - start
    if not bridge.is_file():
        raise RuntimeError(f'ATen bridge missing after build: {bridge}')
    provenance = dict(sources=bridge_sources, torch=torch.__version__,
                      library=str(bridge.resolve()))
    (bridge_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    receipt = dict(source_hashes=source_hashes, kernel=str(kernel),
                   kernel_sha256=sha256(kernel), bridge=str(bridge),
                   bridge_sha256=sha256(bridge), bridge_provenance=provenance,
                   torch=torch.__version__, kernel_command=kernel_cmd,
                   max_jobs=4, nvcc_seconds=nvcc_seconds,
                   bridge_seconds=bridge_seconds)
    (build / 'build_identity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
