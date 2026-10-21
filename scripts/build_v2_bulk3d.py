"""One 3D TMA transaction per selected Q/K/V tile."""
from pathlib import Path
import hashlib,json,subprocess,os
import torch

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pv-qk-pipeline',action='store_true');p.add_argument('--aligned-mask',action='store_true');p.add_argument('--interior-mask',action='store_true');p.add_argument('--hoisted-desc',action='store_true');p.add_argument('--unroll',type=int,choices=[1,2,4,8,16,32],default=32);a=p.parse_args()
 if a.interior_mask:a.aligned_mask=True
 src=Path('experiments/numerical_qk_reuse/csrc/v2_online_pv_qk_pipeline_sm90.cu' if a.pv_qk_pipeline else 'experiments/numerical_qk_reuse/csrc/v2_online_bulk3d_sm90.cu');header=src.with_name('v2_online_sm90.h');hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=True,fold=True,bulk3d=True,pv_qk_pipeline=a.pv_qk_pipeline,aligned_mask=a.aligned_mask,interior_mask=a.interior_mask,hoisted_desc=a.hoisted_desc,production=False,tma_all=True,async_vote=True,fast_math=True,unroll=a.unroll,pz='bf16x3');key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=a.out/key;dest.mkdir(parents=True,exist_ok=False)
 code=src.read_text();needle='#pragma unroll\n        for(int kk=0;kk<512;';assert code.count(needle)==1;code=code.replace(needle,needle.replace('#pragma unroll',f'#pragma unroll {a.unroll}'))
 if a.hoisted_desc:
  assert not a.pv_qk_pipeline
  anchor='    for(int j=0;j<tiles;++j){'
  code=code.replace(anchor,'''    auto k_tensor=make_tensor(make_smem_ptr(s.k),QL{});
    auto q0=local_tile(tq,Shape<_64,_16>{},make_coord(0,0));auto k0=local_tile(k_tensor,Shape<_64,_16>{},make_coord(0,0));
    uint64_t q_descriptor=GMMA::make_gmma_desc<GMMA::Major::K>(q0),k_descriptor=GMMA::make_gmma_desc<GMMA::Major::K>(k0);
'''+anchor)
  begin=code.index('            auto qa=local_tile(tq,');end=code.index('\n        }\n        warpgroup_commit();',begin)
  code=code[:begin]+'''            uint64_t offset=(kk/64)*512+(kk%64)/8;
            mm<64>(q_descriptor+offset,k_descriptor+offset,score);'''+code[end:]
 if a.aligned_mask:
  code=code.replace('template<bool PREFETCH_V,bool PREFETCH_K>','template<bool PREFETCH_V,bool PREFETCH_K,bool ALIGNED>')
  code=code.replace('score[reg]=(qstart+row<a.queries&&key<a.keys)?score[reg]*(a.scale*1.4426950408889634f):-INFINITY;','if constexpr(ALIGNED)score[reg]*=(a.scale*1.4426950408889634f);else score[reg]=(qstart+row<a.queries&&key<a.keys)?score[reg]*(a.scale*1.4426950408889634f):-INFINITY;')
  code=code.replace('template<bool V,bool K> cudaError_t launch','template<bool V,bool K,bool ALIGNED> cudaError_t launch').replace('auto fn=partitioned<V,K>;','auto fn=partitioned<V,K,ALIGNED>;')
  for v,k in [('false','false'),('true','false'),('true','true'),('false','true')]:
   code=code.replace(f'return launch<{v},{k}>(*p,stream,maps);',f'{{if(p->queries%128==0&&p->keys%64==0)return launch<{v},{k},true>(*p,stream,maps);return launch<{v},{k},false>(*p,stream,maps);}}')
  if a.interior_mask:
   start=code.index('            #pragma unroll\n            for(int n=0;n<8;++n)for(int c=0;c<2;++c){\n                int reg=n*4+r*2+c,key=j*64+')
   end=code.index('            local_mx[r]=quad_max(maximum);',start)
   original=code[start:end]
   full=original.replace('if constexpr(ALIGNED)score[reg]*=(a.scale*1.4426950408889634f);else score[reg]=(qstart+row<a.queries&&key<a.keys)?score[reg]*(a.scale*1.4426950408889634f):-INFINITY;','score[reg]*=(a.scale*1.4426950408889634f);')
   tail=original.replace('if constexpr(ALIGNED)score[reg]*=(a.scale*1.4426950408889634f);else ','')
   code=code[:start]+'            if constexpr(ALIGNED){if(j*64+64<=a.keys){\n'+full+'            }else{\n'+tail+'            }}else{\n'+tail+'            }\n'+code[end:]
   code=code.replace('if(p->queries%128==0&&p->keys%64==0)','if(p->queries%128==0)')
 (dest/src.name).write_text(code);(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so';cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','--prec-div=false','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64'];tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library)
if __name__=='__main__':main()
