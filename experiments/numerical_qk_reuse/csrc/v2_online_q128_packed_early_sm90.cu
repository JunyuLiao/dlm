// SPDX-License-Identifier: Apache-2.0
// V2 channel-split CTAs: one WG per CTA, duplicated QK, disjoint output channels.
// Extends this study's attributed Junyu Gaussian32/Hopper and Yuhan reuse work.
// CUTLASS WGMMA layouts and NVIDIA BLASST vote buffering inform execution only.
#include "v2_online_sm90.h"
#include <cuda_bf16.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cmath>
#include <utility>
using BF=__nv_bfloat16;
using fmha::v2_online_sm90::Params;
namespace cg=cooperative_groups;
namespace v2_pipeline {
__device__ __forceinline__ void publish_epoch(int* ptr,int epoch){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;"::"r"(addr),"r"(epoch):"memory");
}
__device__ __forceinline__ int partner_epoch(int* ptr,int rank){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(ptr)),remote;int epoch;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(remote):"r"(addr),"r"(rank));
    asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];":"=r"(epoch):"r"(remote):"memory");
    return epoch;
}

using namespace cute;
struct TmaMaps{CUtensorMap query,key,value;};
using QL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_512>{}));
using PL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_64>{}));
using ZL=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<BF>{},Shape<_32,_64>{}));
using VL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_128,_64>{},Step<_2,_1>{}));
struct alignas(1024) Shared {
    BF q[128*512],k[64*512],v[64*128];
    float z[64*32];
    BF zhi[64*32],zlo[64*32];
    float warp_risk[8];
    uint64_t qbar[2],kbar,vbar;int vphase,epoch,take;
};
__device__ __forceinline__ void cluster_sync(){
    asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;":::"memory");
}
__device__ __forceinline__ void warpgroup_commit(){warpgroup_commit_batch();}
__device__ __forceinline__ void proxy(){asm volatile("fence.proxy.async.shared::cta;":::"memory");}
__device__ __forceinline__ void copy16(void* dst,void const* src,bool valid=true){
    uint32_t sh=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(sh),"l"(src),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void copy_wait(){asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");}
__device__ __forceinline__ float quad_sum(float x){x+=__shfl_xor_sync(0xffffffff,x,1,4);return x+__shfl_xor_sync(0xffffffff,x,2,4);}
__device__ __forceinline__ float quad_max(float x){x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,4));return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2,4));}
__device__ __forceinline__ float warp_max(float x){for(int n=16;n;n/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,n));return x;}
template<int N,bool TRANSPOSE,std::size_t... I>
__device__ __forceinline__ void mm(uint64_t a,uint64_t b,float (&acc)[N/2],std::index_sequence<I...>){
    using Op=std::conditional_t<N==32,SM90::GMMA::MMA_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
        std::conditional_t<N==64,SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
        std::conditional_t<N==16,SM90::GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
        SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>>>>;
    Op::fma(a,b,acc[I]...);
}
template<int N,bool TR=false>
__device__ __forceinline__ void mm(uint64_t a,uint64_t b,float (&acc)[N/2]){mm<N,TR>(a,b,acc,std::make_index_sequence<N/2>{});}
template<int N,std::size_t... I>
__device__ __forceinline__ void mmrs(uint32_t (&p)[4],uint64_t b,float (&acc)[N/2],std::index_sequence<I...>){
 using Op=std::conditional_t<N==32,SM90::GMMA::MMA_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>,SM90::GMMA::MMA_64x128x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>>;
 Op::fma(p[0],p[1],p[2],p[3],b,acc[I]...);
}
template<int N> __device__ __forceinline__ void mmrs(uint32_t (&p)[4],uint64_t b,float (&acc)[N/2]){mmrs<N>(p,b,acc,std::make_index_sequence<N/2>{});}
__device__ __forceinline__ uint32_t pair(float a,float b){auto x=__floats2bfloat162_rn(a,b);return *reinterpret_cast<uint32_t*>(&x);}
template<int N> __device__ __forceinline__ void fence(float (&a)[N]){
    #pragma unroll
    for(int i=0;i<N;++i)warpgroup_fence_operand(a[i]);
}
__device__ __forceinline__ void init_barrier(uint64_t* p){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(addr):"memory");
}
__device__ __forceinline__ void load3d(void* dest,CUtensorMap const* map,int row,uint64_t* p){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p)),out=static_cast<unsigned>(__cvta_generic_to_shared(dest));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 65536;"::"r"(addr):"memory");
    #pragma unroll
    for(int col=0;col<512;col+=64){
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            ::"r"(out+col*64*2),"l"(map),"r"(col),"r"(row),"r"(addr):"memory");
    }
}
__device__ __forceinline__ void wait_barrier(uint64_t* p,int phase){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p));int done;
    do{asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p;}"
        :"=r"(done):"r"(addr),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ void load_k(Params const& a,Shared& s,int tile,int kh,TmaMaps const& maps){
    if(threadIdx.x==0)load3d(s.k,&maps.key,kh*a.keys+tile*64,&s.kbar);
}
__device__ __forceinline__ void load_v(Params const& a,Shared& s,int tile,int kh,TmaMaps const& maps){
    if(threadIdx.x==0){
        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(&s.vbar)),out=static_cast<unsigned>(__cvta_generic_to_shared(s.v));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;"::"r"(addr):"memory");
        #pragma unroll
        for(int col=0;col<128;col+=64){
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                ::"r"(out+col*64*2),"l"(&maps.value),"r"(col+int(blockIdx.z)*128),"r"(kh*a.keys+tile*64),"r"(addr):"memory");
        }
    }
}

template<bool PREFETCH_V,bool PREFETCH_K>
__global__ void __launch_bounds__(256,1)
partitioned(__grid_constant__ Params const a,__grid_constant__ TmaMaps const maps){
    extern __shared__ __align__(1024) unsigned char raw[];
    auto& s=*reinterpret_cast<Shared*>(raw);
    int t=threadIdx.x,u=t%128,wg=t/128,lane=t%32,warp=u/32;
    int channel=blockIdx.z;int head=blockIdx.y,qb=blockIdx.x,qstart=qb*128+wg*64;
    int kh=head/(a.heads/a.kv_heads),tiles=(a.keys+63)/64;
    auto tq=make_tensor(make_smem_ptr(s.q+wg*64*512),QL{});
    if(t==0){init_barrier(&s.qbar[0]);init_barrier(&s.qbar[1]);init_barrier(&s.kbar);init_barrier(&s.vbar);s.vphase=0;s.epoch=0;}
    __syncthreads();
    if(t==0){
        asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
        load3d(s.q,&maps.query,head*a.queries+qb*128,&s.qbar[0]);load3d(s.q+64*512,&maps.query,head*a.queries+qb*128+64,&s.qbar[1]);
    }
    load_k(a,s,0,kh,maps);
    wait_barrier(&s.qbar[wg],0);wait_barrier(&s.kbar,0);__syncthreads();proxy();
    float output[64]={},state[2][8]={{0}};
    float rm[2]={-INFINITY,-INFINITY},rs[2]={0,0},om[2]={-INFINITY,-INFINITY},os[2]={0,0};
    float ref=a.reference[kh],base_limit=expf(2.f*a.log_threshold);
    for(int j=0;j<tiles;++j){
        auto tk=make_tensor(make_smem_ptr(s.k),QL{});
        float score[32]={};fence(score);warpgroup_arrive();
        #pragma unroll
        for(int kk=0;kk<512;kk+=16){
            auto qa=local_tile(tq,Shape<_64,_16>{},make_coord(0,kk/16));
            auto kb=local_tile(tk,Shape<_64,_16>{},make_coord(0,kk/16));
            mm<64>(GMMA::make_gmma_desc<GMMA::Major::K>(qa),GMMA::make_gmma_desc<GMMA::Major::K>(kb),score);
        }
        warpgroup_commit();
        auto packed=reinterpret_cast<BF const*>(a.debug_scores)+(kh*tiles+j)*4096;
        copy16(s.zhi+t*8,packed+t*8);copy16(s.zlo+t*8,packed+2048+t*8);
        warpgroup_wait<0>();fence(score);__syncthreads();
        if constexpr(PREFETCH_K)if(j+1<tiles)load_k(a,s,j+1,kh,maps);
        float mx[2],den[2],eta[2],local_mx[2];
        #pragma unroll
        for(int r=0;r<2;++r){
            int row=warp*16+lane/4+r*8;float maximum=-INFINITY;
            #pragma unroll
            for(int n=0;n<8;++n)for(int c=0;c<2;++c){
                int reg=n*4+r*2+c,key=j*64+n*8+lane%4*2+c;
                score[reg]=(qstart+row<a.queries&&key<a.keys)?score[reg]*a.scale:-INFINITY;
                maximum=fmaxf(maximum,score[reg]);
            }
            local_mx[r]=quad_max(maximum);float safe=isfinite(local_mx[r])?local_mx[r]:0.f,sum=0.f;
            #pragma unroll
            for(int n=0;n<8;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c;score[reg]=__expf(score[reg]-safe);sum+=score[reg];}
            den[r]=quad_sum(sum);
        }
        auto zh=make_tensor(make_smem_ptr(s.zhi),ZL{});auto zl=make_tensor(make_smem_ptr(s.zlo),ZL{});
        #pragma unroll
        for(int r=0;r<2;++r){
            int row=warp*16+lane/4+r*8;mx[r]=local_mx[r];float safe_m=isfinite(mx[r])?mx[r]:0.f;
            float inverse=__expf(local_mx[r]-safe_m)/fmaxf(den[r],1.e-30f);
            float next=fmaxf(rm[r],mx[r]),safe=isfinite(next)?next:0.f,mass=den[r]*__expf(mx[r]-safe);
            rs[r]=rs[r]*__expf(rm[r]-safe)+mass;rm[r]=next;eta[r]=mass/fmaxf(rs[r],1.e-30f);
            #pragma unroll
            for(int n=0;n<8;++n)for(int c=0;c<2;++c){
                int reg=n*4+r*2+c,key=n*8+lane%4*2+c;
                
            }
        }
        copy_wait();__syncthreads();proxy();
        if constexpr(PREFETCH_V)load_v(a,s,j,kh,maps);
        float mu[16]={};fence(mu);warpgroup_arrive();
        #pragma unroll
        for(int kk=0;kk<64;kk+=16){
            uint32_t hi[4],lo[4];
            #pragma unroll
            for(int n=0;n<2;++n)for(int r=0;r<2;++r){int reg=(kk/8+n)*4+r*2;float a=score[reg],b=score[reg+1],ah=float(BF(a)),bh=float(BF(b));hi[n*2+r]=pair(ah,bh);lo[n*2+r]=pair(a-ah,b-bh);}
            auto bh=local_tile(zh,Shape<_32,_16>{},make_coord(0,kk/16));auto bl=local_tile(zl,Shape<_32,_16>{},make_coord(0,kk/16));
            mmrs<32>(hi,GMMA::make_gmma_desc<GMMA::Major::MN>(bl),mu);mmrs<32>(lo,GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);mmrs<32>(hi,GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);
        }
        warpgroup_commit();warpgroup_wait<0>();fence(mu);
        #pragma unroll
        for(int r=0;r<2;++r)for(int n=0;n<4;++n)for(int c=0;c<2;++c)mu[n*4+r*2+c]/=fmaxf(den[r],1.e-30f);
        float worst=0.f;
        #pragma unroll
        for(int r=0;r<2;++r){
            float square=0.f;
            #pragma unroll
            for(int n=0;n<4;++n)for(int c=0;c<2;++c){float delta=mu[n*4+r*2+c]-state[r][n*2+c];square+=delta*delta;}
            worst=fmaxf(worst,eta[r]*eta[r]*quad_sum(square)/(ref*ref));
        }
        worst=warp_max(worst);if(lane==0)s.warp_risk[t/32]=worst;__syncthreads();
        if(t==0){float risk=0.f;
            #pragma unroll
            for(int w=0;w<8;++w)risk=fmaxf(risk,s.warp_risk[w]);
            int ix=(head*((a.queries+127)/128)+qb)*tiles+j;
            float limit=base_limit;if(a.held&&a.held[ix])limit*=expf(-2.f*a.sticky);
            s.take=j==0||j>=a.protect_tile||!(risk<limit);
            if(channel==0){a.skipped[ix]=!s.take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}}
        __syncthreads();bool take=s.take;
        if(take){
            #pragma unroll
            for(int r=0;r<2;++r){
                int row=warp*16+lane/4+r*8;float next=fmaxf(om[r],mx[r]),safe=isfinite(next)?next:0.f;
                float old=__expf(om[r]-safe),mass=den[r]*__expf(mx[r]-safe);
                os[r]=os[r]*old+mass;om[r]=next;
                if(old!=1.f){
                    #pragma unroll
                    for(int n=0;n<16;++n)for(int c=0;c<2;++c)output[n*4+r*2+c]*=old;
                }
                #pragma unroll
                for(int n=0;n<4;++n)for(int c=0;c<2;++c)state[r][n*2+c]=(1.f-eta[r])*state[r][n*2+c]+eta[r]*mu[n*4+r*2+c];
                eta[r]=__expf(mx[r]-safe);
            }
            if constexpr(!PREFETCH_V)load_v(a,s,j,kh,maps);
            copy_wait();wait_barrier(&s.vbar,s.vphase);__syncthreads();proxy();
            auto tv=make_tensor(make_smem_ptr(s.v),VL{});
            fence(output);warpgroup_arrive();
            #pragma unroll
            for(int kk=0;kk<64;kk+=16){
                uint32_t p[4];
                #pragma unroll
                for(int n=0;n<2;++n)for(int r=0;r<2;++r){int reg=(kk/8+n)*4+r*2;p[n*2+r]=pair(eta[r]*score[reg],eta[r]*score[reg+1]);}
                auto vb=local_tile(tv,Shape<_128,_16>{},make_coord(0,kk/16));mmrs<128>(p,GMMA::make_gmma_desc<GMMA::Major::MN>(vb),output);
            }
            warpgroup_commit();warpgroup_wait<0>();fence(output);
            __syncthreads();if(t==0)s.vphase^=1;
        }
        if constexpr(!PREFETCH_K)if(j+1<tiles)load_k(a,s,j+1,kh,maps);
        copy_wait();if(j+1<tiles)wait_barrier(&s.kbar,(j+1)&1);__syncthreads();proxy();
    }
    #pragma unroll
    for(int n=0;n<16;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){
        int row=qstart+warp*16+lane/4+r*8,col=channel*128+n*8+lane%4*2+c;
        if(row<a.queries)static_cast<BF*>(a.output)[(head*a.queries+row)*512+col]=BF(output[n*4+r*2+c]/fmaxf(os[r],1.e-30f));
    }
    if(a.trace&&channel==0){
        #pragma unroll
        for(int r=0;r<2;++r){int row=qstart+warp*16+lane/4+r*8;if(row<a.queries){
            if(lane%4==0)a.log_normalizer[head*a.queries+row]=rm[r]+logf(fmaxf(rs[r],1.e-30f));
            #pragma unroll
            for(int n=0;n<4;++n)for(int c=0;c<2;++c)a.projected_state[(head*a.queries+row)*32+n*8+lane%4*2+c]=state[r][n*2+c];
        }}
    }
    __syncthreads();
}
template<bool V,bool K> cudaError_t launch(Params const& p,cudaStream_t stream,TmaMaps const& maps){
    auto fn=partitioned<V,K>;auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));if(err!=cudaSuccess)return err;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((p.queries+127)/128,p.heads,4);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=sizeof(Shared);cfg.stream=stream;
    return cudaLaunchKernelEx(&cfg,fn,p,maps);
}
}
extern "C" uint32_t v2_online_sm90_abi_version(){return fmha::v2_online_sm90::ABI_VERSION;}
extern "C" uint32_t v2_online_sm90_params_size(){return sizeof(Params);}
extern "C" cudaError_t v2_online_sm90_sm90(Params const* p,cudaStream_t stream,int schedule){
    using namespace v2_pipeline;
    if(!p||!p->debug_scores||p->batch!=1||p->width!=512||p->mode!=1||p->mask_kind!=0||p->heads%p->kv_heads||p->keys<1||p->queries<1)return cudaErrorInvalidValue;
    TmaMaps maps{};
    uint32_t box[2]={64,64},strides[2]={1,1};uint64_t pitch[1]={1024};
    uint64_t qdims[2]={512,uint64_t(p->heads)*p->queries},kdims[2]={512,uint64_t(p->kv_heads)*p->keys};
    auto encode=[&](CUtensorMap* map,void const* ptr,uint64_t* dims){return cuTensorMapEncodeTiled(map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,const_cast<void*>(ptr),dims,pitch,box,strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);};
    if(encode(&maps.query,p->q,qdims)!=CUDA_SUCCESS||encode(&maps.key,p->k,kdims)!=CUDA_SUCCESS||encode(&maps.value,p->v,kdims)!=CUDA_SUCCESS)return cudaErrorInvalidValue;
    if(schedule!=0&&schedule!=3)return cudaErrorInvalidValue;
    if(schedule==0)return launch<false,false>(*p,stream,maps);
    if(schedule==1)return launch<true,false>(*p,stream,maps);
    if(schedule==2)return launch<true,true>(*p,stream,maps);
    if(schedule==3)return launch<false,true>(*p,stream,maps);
    return cudaErrorInvalidValue;
}
