"""One 3D TMA transaction per selected Q/K/V tile."""
from pathlib import Path
import hashlib,json,subprocess,os
import torch

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pv-qk-pipeline',action='store_true');p.add_argument('--aligned-mask',action='store_true');p.add_argument('--interior-mask',action='store_true');p.add_argument('--hoisted-desc',action='store_true');p.add_argument('--front-v',action='store_true');p.add_argument('--front-k',action='store_true');p.add_argument('--qk-asm',action='store_true');p.add_argument('--static-qk-desc',action='store_true');p.add_argument('--low-qk-desc',action='store_true');p.add_argument('--tree-reductions',action='store_true');p.add_argument('--unroll',type=int,choices=[1,2,4,8,16,32],default=32);a=p.parse_args()
 if a.static_qk_desc or a.low_qk_desc:a.qk_asm=True
 if a.interior_mask:a.aligned_mask=True
 src=Path('experiments/numerical_qk_reuse/csrc/v2_online_pv_qk_pipeline_sm90.cu' if a.pv_qk_pipeline else 'experiments/numerical_qk_reuse/csrc/v2_online_bulk3d_sm90.cu');header=src.with_name('v2_online_sm90.h');hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,header,Path(__file__)]}
 options=dict(packed=True,fold=True,bulk3d=True,pv_qk_pipeline=a.pv_qk_pipeline,aligned_mask=a.aligned_mask,interior_mask=a.interior_mask,hoisted_desc=a.hoisted_desc,front_v=a.front_v,front_k=a.front_k,qk_asm=a.qk_asm,static_qk_desc=a.static_qk_desc,low_qk_desc=a.low_qk_desc,tree_reductions=a.tree_reductions,production=False,tma_all=True,async_vote=True,fast_math=True,unroll=a.unroll,pz='bf16x3');key=hashlib.sha256(json.dumps([hashes,options],sort_keys=True).encode()).hexdigest()[:16];dest=a.out/key;dest.mkdir(parents=True,exist_ok=False)
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
 if a.front_v or a.front_k:
  assert not a.pv_qk_pipeline
  code=code.replace('int vphase,epoch,take[2];','int vphase,epoch,take[2],v_done,k_done;')
  code=code.replace('s.vphase=0;s.epoch=0;','s.vphase=0;s.epoch=0;s.v_done=0;s.k_done=0;')
  anchor='template<bool PREFETCH_V,bool PREFETCH_K>\n__global__'
  code=code.replace(anchor,'''__device__ __forceinline__ void local_publish(int* p,int value){unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p));asm volatile("st.release.cta.shared.u32 [%0], %1;"::"r"(addr),"r"(value):"memory");}
__device__ __forceinline__ int local_observe(int* p){unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p));int value;asm volatile("ld.acquire.cta.shared.u32 %0, [%1];":"=r"(value):"r"(addr):"memory");return value;}
'''+anchor)
  if a.front_v:
   code=code.replace('        if constexpr(PREFETCH_V)load_v(a,s,j,kh,maps);','')
   code=code.replace('    for(int j=0;j<tiles;++j){','''    for(int j=0;j<tiles;++j){
        if constexpr(PREFETCH_V)if(t==0){while(local_observe(&s.v_done)<j){}load_v(a,s,j,kh,maps);}
''')
   anchor='        if(j+1<tiles)wait_barrier(&s.kbar,(j+1)&1);asm volatile("bar.sync %0, 128;"::"r"(wg+1):"memory");proxy();'
   assert code.count(anchor)==1
   code=code.replace(anchor,anchor+'\n        if(t==128)local_publish(&s.v_done,j+1);')
  if a.front_k:
   code=code.replace('        if constexpr(PREFETCH_K)if(j+1<tiles){load_k(a,s,j+1,kh,maps);asm volatile("cp.async.commit_group;":::"memory");}','')
   anchor='        warpgroup_wait<0>();fence(score);'
   assert code.count(anchor)==1
   code=code.replace(anchor,anchor+'''
        if constexpr(PREFETCH_K){if(t==128)local_publish(&s.k_done,j+1);if(t==0&&j+1<tiles){while(local_observe(&s.k_done)<j+1){}load_k(a,s,j+1,kh,maps);}}
''')
 if a.qk_asm:
  assert not a.pv_qk_pipeline and not a.hoisted_desc
  registers='{'+', '.join('%'+str(i) for i in range(32))+'}'
  statements=['{','.reg .b64 ad,bd;','.reg .pred p;','mov.b64 ad,%32;','mov.b64 bd,%33;','setp.ne.b32 p,0,0;']
  for i in range(32):
   statements.append('wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 '+registers+',ad,bd,p,1,1,0,0;')
   if i==0:statements.append('setp.ne.b32 p,1,0;')
   if i!=31:
    delta=506 if i%4==3 else 2
    statements.extend([f'add.u64 ad,ad,{delta};',f'add.u64 bd,bd,{delta};'])
  statements.append('}')
  if a.low_qk_desc:
   statements=['{','.reg .b64 ad,bd;','.reg .b32 al,bl;','.reg .pred p;','cvt.u32.u64 al,%32;','cvt.u32.u64 bl,%33;','setp.ne.b32 p,0,0;']
   for i in range(32):
    statements.extend(['mov.b64 ad,{al,0x40000040};','mov.b64 bd,{bl,0x40000040};','wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 '+registers+',ad,bd,p,1,1,0,0;'])
    if i==0:statements.append('setp.ne.b32 p,1,0;')
    if i!=31:
     delta=506 if i%4==3 else 2
     statements.extend([f'add.u32 al,al,{delta};',f'add.u32 bl,bl,{delta};'])
   statements.append('}')
  if a.static_qk_desc:
   # This named build requires dynamic SMEM at offset zero. Trap on ABI drift;
   # frozen static shared layout has no other declarations or hidden buffers.
   code=code.replace('auto& s=*reinterpret_cast<Shared*>(raw);','auto& s=*reinterpret_cast<Shared*>(raw);if(__cvta_generic_to_shared(raw)!=0)asm volatile("trap;");')
   statements=['{','.reg .b64 ad,bd;','.reg .pred p,w;','setp.eq.b64 w,%32,0x4000004000010000;','@!w bra WG1;']
   for half in range(2):
    if half:statements.append('WG1:')
    statements.append('setp.ne.b32 p,0,0;')
    for i in range(32):
     offset=(i//4)*512+(i%4)*2
     statements.extend([f'mov.b64 ad,0x{0x4000004000010000+half*4096+offset:x};',f'mov.b64 bd,0x{0x4000004000012000+offset:x};','wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 '+registers+',ad,bd,p,1,1,0,0;'])
     if i==0:statements.append('setp.ne.b32 p,1,0;')
    if half==0:statements.append('bra DONE;')
   statements.extend(['DONE:','}'])
  asm='\n'.join('        '+json.dumps(x+'\n') for x in statements)
  outputs=', '.join(f'"+f"(acc[{i}])' for i in range(32))
  function='''__device__ __forceinline__ void qk_sequence(uint64_t qa,uint64_t kb,float (&acc)[32]){
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
'''+asm+'\n        : '+outputs+'\n        : "l"(qa),"l"(kb));\n#endif\n}\n'
  code=code.replace('template<bool PREFETCH_V,bool PREFETCH_K>\n__global__',function+'template<bool PREFETCH_V,bool PREFETCH_K>\n__global__')
  begin=code.index(f'        #pragma unroll {a.unroll}\n        for(int kk=0;kk<512;')
  end=code.index('        warpgroup_commit();',begin)
  code=code[:begin]+'''        auto qa=local_tile(tq,Shape<_64,_16>{},make_coord(0,0));auto kb=local_tile(tk,Shape<_64,_16>{},make_coord(0,0));
        qk_sequence(GMMA::make_gmma_desc<GMMA::Major::K>(qa),GMMA::make_gmma_desc<GMMA::Major::K>(kb),score);
'''+code[end:]
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
 if a.tree_reductions:
  code=code.replace('float maximum=-INFINITY;','float maximum=-INFINITY,m0=-INFINITY,m1=-INFINITY,m2=-INFINITY,m3=-INFINITY;')
  code=code.replace('maximum=fmaxf(maximum,score[reg]);','if constexpr(false){}else if(n%4==0)m0=fmaxf(m0,score[reg]);else if(n%4==1)m1=fmaxf(m1,score[reg]);else if(n%4==2)m2=fmaxf(m2,score[reg]);else m3=fmaxf(m3,score[reg]);')
  code=code.replace('local_mx[r]=quad_max(maximum);','maximum=fmaxf(fmaxf(m0,m1),fmaxf(m2,m3));local_mx[r]=quad_max(maximum);')
  code=code.replace('float risk=0.f;\n        #pragma unroll\n        for(int w=0;w<8;++w)risk=fmaxf(risk,s.warp_risk[j&1][w]);', 'float risk=lane<8?s.warp_risk[j&1][lane]:0.f;risk=fmaxf(risk,__shfl_xor_sync(0xffffffff,risk,1));risk=fmaxf(risk,__shfl_xor_sync(0xffffffff,risk,2));risk=fmaxf(risk,__shfl_xor_sync(0xffffffff,risk,4));risk=__shfl_sync(0xffffffff,risk,0);')
 (dest/src.name).write_text(code);(dest/header.name).write_bytes(header.read_bytes());library=dest/'v2_online_sm90.so';cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
 cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','--prec-div=false','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(dest/src.name),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64'];tmp=dest/'tmp';tmp.mkdir();env=os.environ.copy();env['TMPDIR']=str(tmp.resolve());r=subprocess.run(cmd,capture_output=True,text=True,env=env)
 (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=hashes,options=options,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr[-4000:])
 print(library)
if __name__=='__main__':main()
