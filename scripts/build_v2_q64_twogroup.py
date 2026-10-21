"""Freeze the Q64 single-WG/two-channel online scheduling experiment."""
from pathlib import Path
import hashlib,json,subprocess,os
import torch

def build(root,unroll=4,shared_p=False,cluster_barrier=False):
 src=Path('experiments/numerical_qk_reuse/csrc/v2_online_q64_twogroup_sm90.cu');header=src.with_name('v2_online_sm90.h')
 if shared_p:src=src.with_name('v2_online_q64_sharedp_sm90.cu')
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=True,q64_twogroup=True,shared_p=shared_p,cluster_barrier=cluster_barrier,production=False,tma_all=True,async_vote=True,fast_math=True,unroll=unroll,pz='bf16x3')
 key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=Path(root)/key;dest.mkdir(parents=True,exist_ok=False)
 code=src.read_text();needle='#pragma unroll\n        for(int kk=0;kk<512;';assert code.count(needle)==1;code=code.replace(needle,needle.replace('#pragma unroll',f'#pragma unroll {unroll}'))
 if cluster_barrier:
  begin=code.index('        __syncthreads();\n        int ix=')
  end=code.index('        if(take){',begin)
  code=code[:begin]+'''        __syncthreads();
        if(t==0){float risk=0.f;
            #pragma unroll
            for(int w=0;w<4;++w)risk=fmaxf(risk,s.warp_risk[j&1][w]);
            s.vote[j&1]=risk;}
        cluster_sync();
        float risk=0.f;
        if(lane==0){auto* peer=cg::this_cluster().map_shared_rank(&s,1-half);risk=fmaxf(s.vote[j&1],peer->vote[j&1]);}
        risk=__shfl_sync(0xffffffff,risk,0);
        int ix=(head*((a.queries+127)/128)+qb)*tiles+j;
        float limit=base_limit;if(a.held&&a.held[ix])limit*=held_factor;
        bool take=j==0||j>=a.protect_tile||!(risk<limit);
        if(t==0&&half==0&&channel==0){a.skipped[ix]=!take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}
'''+code[end:]
 (dest/src.name).write_text(code);(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so'
 cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
 tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library,flush=True);return library
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--unroll',type=int,default=4);p.add_argument('--shared-p',action='store_true');p.add_argument('--cluster-barrier',action='store_true');a=p.parse_args();build(a.out,a.unroll,a.shared_p,a.cluster_barrier)
