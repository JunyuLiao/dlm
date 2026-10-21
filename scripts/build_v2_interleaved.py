"""Frozen interleaved-QK, ordered-routing, retained-PV cooperative build."""
from pathlib import Path
import hashlib,json,subprocess,os
import torch

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--q128',action='store_true');a=p.parse_args()
 src=Path('experiments/numerical_qk_reuse/csrc/v2_online_interleaved_sm90.cu');header=src.with_name('v2_online_sm90.h')
 if a.q128:src=src.with_name('v2_online_interleaved_q128_sm90.cu')
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=True,global_queue=True,interleaved_qk=True,queue_bytes=98304,queue_units_per_qblock=2,production=False,tma_all=True,async_vote=True,fast_math=True,pz='bf16x3',unique_qk=True,qk_lookahead_slots=2)
 if a.q128:options.update(queue_bytes=196608,queue_units_per_qblock=1,interleaved_q128=True)
 key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=a.out/key;dest.mkdir(parents=True,exist_ok=False)
 (dest/src.name).write_bytes(src.read_bytes());(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so';cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
 tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library,flush=True)
if __name__=='__main__':main()
