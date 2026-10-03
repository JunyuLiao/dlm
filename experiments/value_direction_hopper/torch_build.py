"""Explicit build of the ATen bridge; never JIT-build inside inference."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def build(kernel):
    from torch.utils.cpp_extension import load
    source=Path(__file__).parent/'csrc/torch_bridge.cpp'
    header=source.with_name('value_direction.h')
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,header,kernel)}
    key=hashlib.sha256(json.dumps([sources,torch.__version__],sort_keys=True).encode()).hexdigest()[:16]
    directory=kernel.parent/('torch_'+key);directory.mkdir(exist_ok=True)
    name='value_direction_torch_'+key
    load(name=name,sources=[str(source.resolve())],extra_cflags=['-O3','-std=c++17','-I/usr/local/cuda/include'],
         extra_ldflags=[str(kernel.resolve()),'-L/usr/local/cuda/lib64','-lcudart','-lc10_cuda',
                        '-Wl,-rpath,'+str(kernel.parent.resolve()),'-Wl,-rpath,/usr/local/cuda/lib64'],
         build_directory=str(directory.resolve()),is_python_module=False,verbose=True)
    dest=directory/(name+'.so')
    (directory/'provenance.json').write_text(json.dumps(dict(sources=sources,torch=torch.__version__,library=str(dest.resolve())),indent=2)+'\n')
    print(dest.resolve(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernel',type=Path,required=True);build(p.parse_args().kernel)
