"""Explicit build and read-only query of archived kernel resource limits."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from .cuda import ROOT,SOURCE,BUILD

FIELDS=('registers','local_bytes','static_shared_bytes','dynamic_shared_bytes','threads',
        'cluster_size','multiprocessors','sm_max_threads','active_clusters','grid_clusters',
        'blocks_per_sm_limit','binary_version','ptx_version')


class Info(ct.Structure):
    _fields_=[(name,ct.c_int) for name in FIELDS]


def build(kernel):
    key=kernel.stem.removeprefix('value_direction_')
    proof=json.loads((kernel.parent/(key+'.json')).read_text())
    archived=kernel.parent/key/'value_direction.cu'
    source=SOURCE/'resources.cu'
    ident=hashlib.sha256(source.read_bytes()+archived.read_bytes()+json.dumps(proof['options'],sort_keys=True).encode()).hexdigest()[:16]
    dest=BUILD/f'resources_{ident}.so'
    if not dest.exists():
        cmd=list(proof['command']);cmd[cmd.index(str(SOURCE/'value_direction.cu'))]=str(source)
        cmd[cmd.index('-o')+1]=str(dest)
        cmd.insert(1,f'-DVD_ARCHIVED_SOURCE="{archived.resolve()}"')
        result=subprocess.run(cmd,text=True,capture_output=True)
        (BUILD/f'resources_{ident}.json').write_text(json.dumps(dict(kernel=str(kernel.resolve()),kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
        if result.returncode:print(result.stderr)
        result.check_returncode()
    print(dest,flush=True)
    return dest


def query(library,output):
    if output.exists():raise FileExistsError(output)
    lib=ct.CDLL(str(library.resolve()));fn=lib.value_direction_resources
    fn.argtypes=[ct.c_int,ct.c_int,ct.POINTER(Info)];fn.restype=ct.c_int
    torch.cuda.init();rows=[]
    for width,schedule in ((256,0),(256,1),(512,0),(512,1),(512,2)):
        info=Info();status=fn(width,schedule,ct.byref(info))
        if status:raise RuntimeError(f'CUDA resource query error {status}')
        row=dict(width=width,schedule=('cp','tma','split')[schedule],**{f:getattr(info,f) for f in FIELDS})
        row['max_cluster_resident_warp_fraction']=info.active_clusters*info.cluster_size*info.threads/(info.multiprocessors*info.sm_max_threads)
        row['native_grid_cluster_waves_lower_bound']=(info.grid_clusters+info.active_clusters-1)//info.active_clusters
        rows.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps(dict(library=str(library),cases=rows,scope='CUDA occupancy API launch-resource upper bounds, not measured achieved occupancy, Tensor Core utilization or stalls'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernel',type=Path);p.add_argument('--library',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.kernel:build(a.kernel)
    elif a.library and a.output:query(a.library,a.output)
    else:p.error('Supply --kernel to build or --library/--output to query')
