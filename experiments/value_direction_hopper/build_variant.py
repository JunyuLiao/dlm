"""Explicit isolated kernel variant builder; leaves frozen launch code unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .cuda import SOURCE,BUILD


def build(base,ptx_cluster=False):
    old=json.loads((base.parent/(base.stem.removeprefix('value_direction_')+'.json')).read_text())
    files=[SOURCE/'value_direction.cu',SOURCE/'value_direction.h']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    options=dict(old['options'],float_wgmma_operands=True,ptx_cluster=ptx_cluster)
    key=hashlib.sha256(json.dumps([hashes,options,old['command']],sort_keys=True).encode()).hexdigest()[:16]
    destination=BUILD/f'value_direction_{key}.so'
    if not destination.exists():
        archive=BUILD/key;archive.mkdir(exist_ok=True)
        for file in files:(archive/file.name).write_bytes(file.read_bytes())
        cmd=list(old['command'])
        if '-DVD_F32_OPERANDS=1' not in cmd:cmd.insert(1,'-DVD_F32_OPERANDS=1')
        if ptx_cluster:cmd.insert(1,'-DVD_PTX_CLUSTER=1')
        cmd[cmd.index('-o')+1]=str(destination)
        result=subprocess.run(cmd,text=True,capture_output=True)
        (BUILD/(key+'.json')).write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
        if result.returncode:print(result.stderr,flush=True)
        result.check_returncode()
    print(destination,flush=True)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--ptx-cluster',action='store_true')
    a=p.parse_args();build(a.base,a.ptx_cluster)
