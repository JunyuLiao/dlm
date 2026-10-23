"""Build exact-arithmetic paged V2 from the frozen qualified contiguous source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch

BASE = Path('results/sparse_execution_20261021/attempt002/build/200354cb919fb6b8')

def build(root, *, swizzle=False, paged=True):
    source = BASE/'v2_online_bulk3d_sm90.cu'
    header = BASE/'v2_online_sm90.h'
    code = source.read_text()
    if swizzle:
        code = code.replace('int channel=blockIdx.z;int head=blockIdx.y,qb=blockIdx.x,',
                            'int channel=blockIdx.x;int head=blockIdx.z,qb=blockIdx.y,')
        code = code.replace('int(blockIdx.z)*2', 'int(blockIdx.x)*2')
        code = code.replace('dim3((p.queries+127)/128,p.heads,4)',
                            'dim3(4,(p.queries+127)/128,p.heads)')
    if paged:
        code = code.replace('struct TmaMaps{CUtensorMap query,key,value;};', '''struct PagedLayout {
    int const* table; int pages, page_size; uint64_t token_stride, head_stride;
};
struct TmaMaps{CUtensorMap query,key,value; int const* table; int page_size;};''')
        code = code.replace('if(threadIdx.x==0)load3d(s.k,&maps.key,kh*a.keys+tile*64,&s.kbar);', '''if(threadIdx.x==0){
        int token=tile*64,physical=maps.table[token/maps.page_size]*maps.page_size+token%maps.page_size;
        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(&s.kbar)),out=static_cast<unsigned>(__cvta_generic_to_shared(s.k));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 65536;"::"r"(addr):"memory");
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
            ::"r"(out),"l"(&maps.key),"r"(0),"r"(physical),"r"(0),"r"(kh),"r"(addr):"memory");
    }''')
        channel = 'int(blockIdx.x)*2' if swizzle else 'int(blockIdx.z)*2'
        old = '''        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
            ::"r"(out),"l"(&maps.value),"r"(0),"r"(kh*a.keys+tile*64),"r"('''+channel+'''),"r"(addr):"memory");'''
        new = '''        int token=tile*64,physical=maps.table[token/maps.page_size]*maps.page_size+token%maps.page_size;
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
            ::"r"(out),"l"(&maps.value),"r"(0),"r"(physical),"r"('''+channel+'''),"r"(kh),"r"(addr):"memory");'''
        if code.count(old)!=1: raise RuntimeError('V load source drift')
        code = code.replace(old,new)
        code = code.replace('v2_online_sm90_sm90(Params const* p,cudaStream_t stream,int schedule)',
                            'v2_online_paged_sm90(Params const* p,v2_pipeline::PagedLayout const* layout,cudaStream_t stream,int schedule)')
        begin=code.index('    uint32_t box[3]=',code.index('extern "C" cudaError_t v2_online_paged_sm90'))
        end=code.index('    if(schedule<0',begin)
        code=code[:begin]+'''    if(!layout||!layout->table||layout->page_size%64)return cudaErrorInvalidValue;
    maps.table=layout->table;maps.page_size=layout->page_size;
    uint32_t qbox[3]={64,64,8},qstrides[3]={1,1,1};uint64_t qpitch[2]={1024,128};
    uint64_t qdims[3]={64,uint64_t(p->heads)*p->queries,8};
    if(cuTensorMapEncodeTiled(&maps.query,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,const_cast<void*>(p->q),qdims,qpitch,qbox,qstrides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)return cudaErrorInvalidValue;
    uint32_t box[4]={64,64,8,1},vbox[4]={64,64,2,1},strides[4]={1,1,1,1};
    uint64_t dims[4]={64,uint64_t(layout->pages)*layout->page_size,8,uint64_t(p->kv_heads)};
    uint64_t pitch[3]={layout->token_stride*2,128,layout->head_stride*2};
    auto encode=[&](CUtensorMap* map,void const* ptr,uint32_t* shape){return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,const_cast<void*>(ptr),dims,pitch,shape,strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);};
    if(encode(&maps.key,p->k,box)!=CUDA_SUCCESS||encode(&maps.value,p->v,vbox)!=CUDA_SUCCESS)return cudaErrorInvalidValue;
'''+code[end:]
        # Keep the existing ABI introspection/loader usable; contiguous launch
        # is intentionally unavailable in this separately named binary.
        code+='\nextern "C" cudaError_t v2_online_sm90_sm90(Params const*,cudaStream_t,int){return cudaErrorNotSupported;}\n'
    options=dict(json.loads((BASE/'build_receipt.json').read_text())['options'],paged=paged,cta_swizzle=swizzle,arithmetic='unchanged_200354cb')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,header,Path(__file__)]}
    key=hashlib.sha256((code+json.dumps(options,sort_keys=True)).encode()).hexdigest()[:16]
    dest=Path(root)/key;dest.mkdir(parents=True,exist_ok=False)
    target=dest/'v2_online_paged_sm90.cu';target.write_text(code);(dest/header.name).write_bytes(header.read_bytes())
    library=dest/'v2_online_sm90.so';cutlass=Path(torch.__file__).parent.parent/'flashinfer/data/cutlass/include'
    cmd=['/usr/local/cuda/bin/nvcc','--use_fast_math','--prec-div=false','-std=c++17','-O3','-arch=sm_90a','--shared','-Xcompiler=-fPIC','-lineinfo','--ptxas-options=-v','-I'+str(cutlass),'-I/usr/local/cuda/include/cccl',str(target),'-o',str(library),'-L/usr/local/cuda/lib64','-lcudart','-lcuda','-Xlinker=-rpath,/usr/local/cuda/lib64']
    temp=dest/'tmp';temp.mkdir();env=dict(os.environ,TMPDIR=str(temp.resolve()))
    result=subprocess.run(cmd,text=True,capture_output=True,env=env)
    (dest/'build_receipt.json').write_text(json.dumps(dict(command=cmd,sources=sources,options=options,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    if result.returncode:raise RuntimeError(result.stderr[-5000:])
    print(library,flush=True);return library

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--swizzle',action='store_true');p.add_argument('--contiguous',action='store_true');a=p.parse_args()
    build(a.out,swizzle=a.swizzle,paged=not a.contiguous)
