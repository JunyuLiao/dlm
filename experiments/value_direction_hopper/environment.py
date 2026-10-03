"""Read-only environment provenance capture, with no installation or tuning."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


def run(output):
    if output.exists():raise FileExistsError(output)
    packages={}
    for name in ('torch','transformers','triton','flash-attn','flashinfer-python','nvidia-cutlass-dsl','numpy'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    commands={
        'nvcc':['/usr/local/cuda/bin/nvcc','--version'],
        'nsight_systems':['nsys','--version'],
        'nsight_compute':['/usr/local/cuda/bin/ncu','--version'],
        'gpu':['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],
        'reference_commit':['git','-C','reference/TensorRT-LLM-value-aware','rev-parse','HEAD'],
        'repository_commit':['git','rev-parse','HEAD']}
    results={}
    for name,cmd in commands.items():
        try:
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
            results[name]=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as exc:results[name]=dict(command=cmd,error=repr(exc))
    import torch
    folder=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
    # Hash all header inputs, not only the four archived direct source files.
    manifest={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.rglob('*')) if p.is_file()}
    output.write_text(json.dumps(dict(packages=packages,commands=results,cutlass_headers=manifest,
        cutlass_header_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        scope='Observed environment only. No driver clock, privilege, package or system settings changed.'),indent=2)+'\n')
    print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
