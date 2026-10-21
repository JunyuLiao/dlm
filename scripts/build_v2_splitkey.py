"""Frozen packed split-key/unique-QK online refresh variant."""
from pathlib import Path
import hashlib,json,subprocess,os
import torch

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--lookahead',action='store_true');p.add_argument('--q32',action='store_true');p.add_argument('--kv128-bundle',action='store_true');p.add_argument('--channel-groups',type=int,choices=[1,2],default=1);a=p.parse_args()
 src=Path('experiments/numerical_qk_reuse/csrc/v2_online_splitkey_packed_sm90.cu');header=src.with_name('v2_online_sm90.h')
 if a.lookahead:src=src.with_name('v2_online_splitkey_lookahead_sm90.cu')
 if a.channel_groups==2:
  assert a.lookahead;src=src.with_name('v2_online_splitkey_lookahead256_sm90.cu')
 if a.q32:
  assert not a.lookahead and a.channel_groups==1;src=src.with_name('v2_online_splitkey_q32_sm90.cu')
 if a.kv128_bundle:
  assert not a.lookahead and not a.q32 and a.channel_groups==1;src=src.with_name('v2_online_kv128_bundle_sm90.cu')
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=True,split_key=2,kv128_bundle=a.kv128_bundle,channel_groups=a.channel_groups,unique_qk=a.channel_groups==1,qk_multiplicity=a.channel_groups,qk_pz_overlap=a.lookahead,production=False,tma_all=True,async_vote=True,fast_math=True,pz='bf16x3',q_tile=32 if a.q32 else 64)
 key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=a.out/key;dest.mkdir(parents=True,exist_ok=False)
 (dest/src.name).write_bytes(src.read_bytes());(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so';cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
 tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library,flush=True)
if __name__=='__main__':main()
