"""Separate frozen build of the packed unique-router/dual-consumer variant."""
from pathlib import Path
import hashlib,json,subprocess,os
import torch

def build(root,unroll=4,groups=1,kept_queue=False,register_pv=False,global_queue=False,q128_queue=False,qreg_queue=False):
 src=Path('experiments/numerical_qk_reuse/csrc')/('v2_online_unique_packed_sm90.cu' if groups==1 else 'v2_online_dual_router_packed_sm90.cu');header=src.with_name('v2_online_sm90.h')
 if kept_queue:src=src.with_name('v2_online_kept_queue_sm90.cu' if groups==2 else 'v2_online_unique_kept_queue_sm90.cu')
 if register_pv:src=src.with_name('v2_online_kept_queue_rs_sm90.cu')
 if global_queue:src=src.with_name('v2_online_global_queue_sm90.cu')
 if q128_queue:src=src.with_name('v2_online_q128_global_queue_sm90.cu')
 if qreg_queue:src=src.with_name('v2_online_qreg_queue_sm90.cu')
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=True,three=True,unique_packed=True,channel_groups=groups,kept_queue=kept_queue,register_pv=register_pv,production=False,tma_all=True,async_vote=True,fast_math=True,unroll=unroll,pz='bf16x3')
 if global_queue:options.update(global_queue=True,queue_bytes=24576)
 if q128_queue:options.update(global_queue=True,q128_queue=True,queue_bytes=49152,queue_units_per_qblock=1)
 if qreg_queue:options.update(global_queue=True,qreg_queue=True,queue_bytes=24576,queue_units_per_qblock=2)
 key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=Path(root)/key;dest.mkdir(parents=True,exist_ok=False)
 text=src.read_text();needle='#pragma unroll\n            for(int kk=0;kk<512;';assert text.count(needle)==1
 text=text.replace(needle,needle.replace('#pragma unroll',f'#pragma unroll {unroll}'))
 (dest/src.name).write_text(text);(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so'
 cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','--prec-div=false','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
 tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library,flush=True);return library
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--unroll',type=int,default=4);p.add_argument('--groups',type=int,choices=[1,2],default=1);p.add_argument('--kept-queue',action='store_true');p.add_argument('--register-pv',action='store_true');p.add_argument('--global-queue',action='store_true');p.add_argument('--q128-queue',action='store_true');p.add_argument('--qreg-queue',action='store_true');a=p.parse_args();build(a.out,a.unroll,a.groups,a.kept_queue,a.register_pv,a.global_queue,a.q128_queue,a.qreg_queue)
