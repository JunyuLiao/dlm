// SPDX-License-Identifier: Apache-2.0
// V2 ordered online attention: fused local-softmax combination and next-K overlap.
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
using namespace cute;
struct TmaMaps{CUtensorMap query,key,value;};
using QL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_512>{}));
using PL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_64>{}));
using ZL=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<BF>{},Shape<_32,_64>{}));
using VL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_256,_64>{},Step<_2,_1>{}));
struct alignas(1024) Shared {
    BF q[64*512],k[64*512],v[64*512],p[64*64];
    float z[64*32];
    BF zhi[64*32],zlo[64*32],pl_buffer[64*64];
    float mx[2][64],den[2][64],delta[2][64],warp_risk[4],vote[2];
    uint64_t qbar,kbar,vbar;int vphase;
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
    using Op=std::conditional_t<N==32,SM90::GMMA::MMA_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
        std::conditional_t<N==64,SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>,
        std::conditional_t<N==16,SM90::GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
        SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>>>>;
    Op::fma(a,b,acc[I]...);
}
template<int N,bool TR=false>
__device__ __forceinline__ void mm(uint64_t a,uint64_t b,float (&acc)[N/2]){mm<N,TR>(a,b,acc,std::make_index_sequence<N/2>{});}
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
    if(threadIdx.x==0)load3d(s.v,&maps.value,kh*a.keys+tile*64,&s.vbar);
}

template<bool PREFETCH_V,bool PREFETCH_K>
__global__ void __cluster_dims__(2,1,1) __launch_bounds__(256,1)
partitioned(__grid_constant__ Params const a,__grid_constant__ TmaMaps const maps){
    extern __shared__ __align__(1024) unsigned char raw[];
    auto& s=*reinterpret_cast<Shared*>(raw);
    int t=threadIdx.x,u=t%128,wg=t/128,lane=t%32,warp=u/32;
    int head=blockIdx.y,qb=blockIdx.x/2,half=blockIdx.x%2,qstart=blockIdx.x*64;
    int kh=head/(a.heads/a.kv_heads),tiles=(a.keys+63)/64;
    auto tq=make_tensor(make_smem_ptr(s.q),QL{});
    if(t==0){init_barrier(&s.qbar);init_barrier(&s.kbar);init_barrier(&s.vbar);s.vphase=0;}
    __syncthreads();
    if(t==0){
        asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
        load3d(s.q,&maps.query,head*a.queries+qstart,&s.qbar);
    }
    load_k(a,s,0,kh,maps);
    wait_barrier(&s.qbar,0);wait_barrier(&s.kbar,0);__syncthreads();proxy();
    float output[128]={},state[2][4]={{0}};
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
        // Z is current FP32 Gaussian32; one cooperative copy for both WGs.
        for(int x=t*4;x<64*32;x+=256*4){
            int key=j*64+x/32;
            auto src=a.z+(kh*a.keys+(key<a.keys?key:0))*32+x%32;
            copy16(s.z+x,src,key<a.keys);
        }
        warpgroup_wait<0>();fence(score);
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
            sum=quad_sum(sum);if(lane%4==0){s.mx[wg][row]=local_mx[r];s.den[wg][row]=sum;}
        }
        copy_wait();
        auto ph=make_tensor(make_smem_ptr(s.p),PL{});
        auto pl=make_tensor(make_smem_ptr(s.pl_buffer),PL{});
        // Convert FP32 Z to two BF16 terms in place, retaining values in registers
        // across the first barrier so the overlapped allocation cannot race.
        float zv[8];
        #pragma unroll
        for(int i=0;i<8;++i)zv[i]=s.z[t+i*256];
        auto zh=make_tensor(make_smem_ptr(s.zhi),ZL{});
        auto zl=make_tensor(make_smem_ptr(s.zlo),ZL{});
        #pragma unroll
        for(int i=0;i<8;++i){int key=(t+i*256)/32,dim=(t+i*256)%32;float hi=float(BF(zv[i]));zh(dim,key)=BF(hi);zl(dim,key)=BF(zv[i]-hi);}
        #pragma unroll
        for(int r=0;r<2;++r){
            int row=warp*16+lane/4+r*8;mx[r]=local_mx[r];float safe_m=isfinite(mx[r])?mx[r]:0.f;
            den[r]=s.den[wg][row];
            float inverse=__expf(local_mx[r]-safe_m)/fmaxf(den[r],1.e-30f);
            float next=fmaxf(rm[r],mx[r]),safe=isfinite(next)?next:0.f,mass=den[r]*__expf(mx[r]-safe);
            rs[r]=rs[r]*__expf(rm[r]-safe)+mass;rm[r]=next;eta[r]=mass/fmaxf(rs[r],1.e-30f);
            #pragma unroll
            for(int n=0;n<8;++n)for(int c=0;c<2;++c){
                int reg=n*4+r*2+c,key=n*8+lane%4*2+c;score[reg]*=inverse;
                if(n/4==wg){float hi=float(BF(score[reg]));ph(row,key)=BF(hi);pl(row,key)=BF(score[reg]-hi);}
            }
        }
        __syncthreads();proxy();
        if constexpr(PREFETCH_K)if(j+1<tiles){load_k(a,s,j+1,kh,maps);asm volatile("cp.async.commit_group;":::"memory");}
        if constexpr(PREFETCH_V)load_v(a,s,j,kh,maps);
        float mu[8]={};fence(mu);warpgroup_arrive();
        #pragma unroll
        for(int kk=0;kk<64;kk+=16){
            auto ah=local_tile(ph,Shape<_64,_16>{},make_coord(0,kk/16));
            auto al=local_tile(pl,Shape<_64,_16>{},make_coord(0,kk/16));
            auto bh=local_tile(zh,Shape<_16,_16>{},make_coord(wg,kk/16));
            auto bl=local_tile(zl,Shape<_16,_16>{},make_coord(wg,kk/16));
            mm<16,true>(GMMA::make_gmma_desc<GMMA::Major::K>(ah),GMMA::make_gmma_desc<GMMA::Major::MN>(bl),mu);
            mm<16,true>(GMMA::make_gmma_desc<GMMA::Major::K>(al),GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);
            mm<16,true>(GMMA::make_gmma_desc<GMMA::Major::K>(ah),GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);
        }
        warpgroup_commit();warpgroup_wait<0>();fence(mu);
        #pragma unroll
        for(int r=0;r<2;++r){
            int row=warp*16+lane/4+r*8;float square=0.f;
            #pragma unroll
            for(int n=0;n<2;++n)for(int c=0;c<2;++c){float delta=mu[n*4+r*2+c]-state[r][n*2+c];square+=delta*delta;}
            square=quad_sum(square);if(lane%4==0)s.delta[wg][row]=eta[r]*eta[r]*square/(ref*ref);
        }
        __syncthreads();
        float worst=0.f;
        if(wg==0){
            #pragma unroll
            for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;worst=fmaxf(worst,s.delta[0][row]+s.delta[1][row]);}
            worst=warp_max(worst);if(lane==0)s.warp_risk[warp]=worst;
        }
        __syncthreads();
        if(t==0)s.vote[j&1]=fmaxf(fmaxf(s.warp_risk[0],s.warp_risk[1]),fmaxf(s.warp_risk[2],s.warp_risk[3]));
        cluster_sync();
        auto cluster=cg::this_cluster();auto* first=cluster.map_shared_rank(&s,0);auto* second=cluster.map_shared_rank(&s,1);
        float risk=fmaxf(first->vote[j&1],second->vote[j&1]);
        int ix=(head*((a.queries+127)/128)+qb)*tiles+j;
        float limit=base_limit;if(a.held&&a.held[ix])limit*=expf(-2.f*a.sticky);
        bool take=j==0||j>=a.protect_tile||!(risk<limit);
        if(t==0&&half==0){a.skipped[ix]=!take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}
        if(take){
            auto tp=make_tensor(make_smem_ptr(s.p),PL{});
            #pragma unroll
            for(int r=0;r<2;++r){
                int row=warp*16+lane/4+r*8;float next=fmaxf(om[r],mx[r]),safe=isfinite(next)?next:0.f;
                float old=__expf(om[r]-safe),mass=den[r]*__expf(mx[r]-safe);
                os[r]=os[r]*old+mass;om[r]=next;
                if(old!=1.f){
                    #pragma unroll
                    for(int n=0;n<32;++n)for(int c=0;c<2;++c)output[n*4+r*2+c]*=old;
                }
                #pragma unroll
                for(int n=0;n<4;++n)*reinterpret_cast<__nv_bfloat162*>(&tp(row,wg*32+n*8+lane%4*2))=
                    __floats2bfloat162_rn(mass*score[(n+wg*4)*4+r*2],mass*score[(n+wg*4)*4+r*2+1]);
                #pragma unroll
                for(int n=0;n<2;++n)for(int c=0;c<2;++c)state[r][n*2+c]=(1.f-eta[r])*state[r][n*2+c]+eta[r]*mu[n*4+r*2+c];
            }
            if constexpr(!PREFETCH_V)load_v(a,s,j,kh,maps);
            copy_wait();wait_barrier(&s.vbar,s.vphase);__syncthreads();proxy();
            auto tv=make_tensor(make_smem_ptr(s.v+wg*64*256),VL{});
            fence(output);warpgroup_arrive();
            #pragma unroll
            for(int kk=0;kk<64;kk+=16){
                auto pa=local_tile(tp,Shape<_64,_16>{},make_coord(0,kk/16));
                auto vb=local_tile(tv,Shape<_256,_16>{},make_coord(0,kk/16));
                mm<256,true>(GMMA::make_gmma_desc<GMMA::Major::K>(pa),GMMA::make_gmma_desc<GMMA::Major::MN>(vb),output);
            }
            warpgroup_commit();warpgroup_wait<0>();fence(output);
            __syncthreads();if(t==0)s.vphase^=1;
        }
        if constexpr(!PREFETCH_K)if(j+1<tiles)load_k(a,s,j+1,kh,maps);
        copy_wait();if(j+1<tiles)wait_barrier(&s.kbar,(j+1)&1);__syncthreads();proxy();
    }
    #pragma unroll
    for(int n=0;n<32;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){
        int row=qstart+warp*16+lane/4+r*8,col=wg*256+n*8+lane%4*2+c;
        if(row<a.queries)static_cast<BF*>(a.output)[(head*a.queries+row)*512+col]=BF(output[n*4+r*2+c]/fmaxf(os[r],1.e-30f));
    }
    if(a.trace){
        #pragma unroll
        for(int r=0;r<2;++r){int row=qstart+warp*16+lane/4+r*8;if(row<a.queries){
            if(wg==0&&lane%4==0)a.log_normalizer[head*a.queries+row]=rm[r]+logf(fmaxf(rs[r],1.e-30f));
            #pragma unroll
            for(int n=0;n<2;++n)for(int c=0;c<2;++c)a.projected_state[(head*a.queries+row)*32+wg*16+n*8+lane%4*2+c]=state[r][n*2+c];
        }}
    }
    cluster_sync();
}
template<bool V,bool K> cudaError_t launch(Params const& p,cudaStream_t stream,TmaMaps const& maps){
    auto fn=partitioned<V,K>;auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));if(err!=cudaSuccess)return err;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((p.queries+127)/128*2,p.heads,1);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=sizeof(Shared);cfg.stream=stream;
    return cudaLaunchKernelEx(&cfg,fn,p,maps);
}
}
extern "C" uint32_t v2_online_sm90_abi_version(){return fmha::v2_online_sm90::ABI_VERSION;}
extern "C" uint32_t v2_online_sm90_params_size(){return sizeof(Params);}
extern "C" cudaError_t v2_online_sm90_sm90(Params const* p,cudaStream_t stream,int schedule){
    using namespace v2_pipeline;
    if(!p||p->batch!=1||p->width!=512||p->mode!=1||p->mask_kind!=0||p->heads%p->kv_heads||p->keys<1||p->queries<1)return cudaErrorInvalidValue;
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
