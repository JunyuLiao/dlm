"""Frozen build for the unique Q32 warp-MMA online refresh experiment."""
from pathlib import Path
import hashlib,json,subprocess,os

def build(root):
 src=Path('experiments/numerical_qk_reuse/csrc/v2_online_q32_unique_sm90.cu');header=src.with_name('v2_online_sm90.h')
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=False,q32_unique=True,production=False,tma_all=False,async_vote=True,fast_math=True,pz='bf16x3')
 key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=Path(root)/key;dest.mkdir(parents=True,exist_ok=False)
 (dest/src.name).write_bytes(src.read_bytes());(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-Xlinker=-rpath,/usr/local/cuda/lib64']
 tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library,flush=True);return library
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
