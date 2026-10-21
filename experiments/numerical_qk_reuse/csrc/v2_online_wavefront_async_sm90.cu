// SPDX-License-Identifier: Apache-2.0
// V2 online wavefront: four KV tiles parallel QK/PZ, original ordered routing, conditional full PV.
// Extends this study's attributed Junyu Gaussian32/Hopper and Yuhan reuse work.
// CUTLASS WGMMA layouts and NVIDIA BLASST vote buffering inform execution only.
#include "v2_online_sm90.h"
#include <cuda_bf16.h>
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
__device__ __forceinline__ int remote_epoch(int* ptr,int rank){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(ptr)),remote;int epoch;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(remote):"r"(addr),"r"(rank));
    asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];":"=r"(epoch):"r"(remote):"memory");
    return epoch;
}

using namespace cute;
using QL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_512>{}));
using PL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_64>{}));
using ZL=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<BF>{},Shape<_32,_64>{}));
using VL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_256,_64>{}));
struct alignas(1024) Shared {
    BF q[64*512],k[64*512],v[64*512],p[64*64];
    float z[64*32];
    BF zhi[64*32],zlo[64*32],pl_buffer[64*64];
    float mx[2][64],den[2][64],delta[2][64],warp_risk[4],vote[2];
    int route_epoch,vote_epoch,take;
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
        std::conditional_t<N==16,SM90::GMMA::MMA_64x16x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>,
        SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>>>;
    Op::fma(a,b,acc[I]...);
}
template<int N,bool TR=false>
__device__ __forceinline__ void mm(uint64_t a,uint64_t b,float (&acc)[N/2]){mm<N,TR>(a,b,acc,std::make_index_sequence<N/2>{});}
template<int N> __device__ __forceinline__ void fence(float (&a)[N]){
    #pragma unroll
    for(int i=0;i<N;++i)warpgroup_fence_operand(a[i]);
}
__device__ __forceinline__ void load_k(Params const& a,Shared& s,int tile,int kh){
    auto tk=make_tensor(make_smem_ptr(s.k),QL{});
    for(int x=threadIdx.x*8;x<64*512;x+=256*8){
        int row=x/512,col=x%512,key=tile*64+row;
        auto src=static_cast<BF const*>(a.k)+(kh*a.keys+(key<a.keys?key:0))*512+col;
        copy16(&tk(row,col),src,key<a.keys);
    }
}
__device__ __forceinline__ void load_v(Params const& a,Shared& s,int tile,int kh){
    int t=threadIdx.x,u=t%128,wg=t/128;
    auto tv=make_tensor(make_smem_ptr(s.v+wg*64*256),VL{});
    for(int x=u*8;x<64*256;x+=128*8){
        int key=tile*64+x/256,col=x%256;
        auto src=static_cast<BF const*>(a.v)+(kh*a.keys+(key<a.keys?key:0))*512+wg*256+col;
        copy16(&tv(col,x/256),src,key<a.keys);
    }
}

template<bool PREFETCH_V,bool PREFETCH_K>
__global__ void __cluster_dims__(8,1,1) __launch_bounds__(256,1)
partitioned(__grid_constant__ Params const a){
    extern __shared__ __align__(1024) unsigned char raw[];
    auto& s=*reinterpret_cast<Shared*>(raw);
    int t=threadIdx.x,u=t%128,wg=t/128,lane=t%32,warp=u/32;
    int head=blockIdx.y,qb=blockIdx.x/8,rank=blockIdx.x%8,half=rank%2,position=rank/2,qstart=qb*128+half*64;
    int kh=head/(a.heads/a.kv_heads),tiles=(a.keys+63)/64;
    auto tq=make_tensor(make_smem_ptr(s.q),QL{});
    for(int x=t*8;x<64*512;x+=256*8){
        int row=x/512,col=x%512;
        auto src=static_cast<BF const*>(a.q)+(head*a.queries+(qstart+row<a.queries?qstart+row:0))*512+col;
        copy16(&tq(row,col),src,qstart+row<a.queries);
    }
    load_k(a,s,position,kh);copy_wait();__syncthreads();proxy();
    if(t==0){s.route_epoch=0;s.vote_epoch=0;}
    if(t<64){s.mx[1][t]=-INFINITY;s.den[1][t]=0.f;}
    for(int x=t;x<64*32;x+=256)s.z[x]=0.f;
    __syncthreads();cluster_sync();
    float output[128]={},state[2][4]={{0}};
    float rm[2]={-INFINITY,-INFINITY},rs[2]={0,0},om[2]={-INFINITY,-INFINITY},os[2]={0,0};
    float ref=a.reference[kh],base_limit=expf(2.f*a.log_threshold);
    for(int base=0;base<tiles;base+=4){
        int j=base+position;
        if(position==0){
            if(t==0)while(remote_epoch(&s.route_epoch,6+half)<base){}
            __syncthreads();
            auto* previous=cg::this_cluster().map_shared_rank(&s,6+half);
            #pragma unroll
            for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;rm[r]=previous->mx[1][row];rs[r]=previous->den[1][row];
                #pragma unroll
                for(int n=0;n<2;++n)for(int c=0;c<2;++c)state[r][n*2+c]=previous->z[row*32+wg*16+n*8+lane%4*2+c];
            }
        }
        cluster_sync();
        auto tk=make_tensor(make_smem_ptr(s.k),QL{});
        float score[16]={};fence(score);warpgroup_arrive();
        #pragma unroll
        for(int kk=0;kk<512;kk+=16){
            auto qa=local_tile(tq,Shape<_64,_16>{},make_coord(0,kk/16));
            auto kb=local_tile(tk,Shape<_32,_16>{},make_coord(wg,kk/16));
            mm<32>(GMMA::make_gmma_desc<GMMA::Major::K>(qa),GMMA::make_gmma_desc<GMMA::Major::K>(kb),score);
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
            for(int n=0;n<4;++n)for(int c=0;c<2;++c){
                int reg=n*4+r*2+c,key=j*64+wg*32+n*8+lane%4*2+c;
                score[reg]=(qstart+row<a.queries&&key<a.keys)?score[reg]*a.scale:-INFINITY;
                maximum=fmaxf(maximum,score[reg]);
            }
            local_mx[r]=quad_max(maximum);float safe=isfinite(local_mx[r])?local_mx[r]:0.f,sum=0.f;
            #pragma unroll
            for(int n=0;n<4;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c;score[reg]=__expf(score[reg]-safe);sum+=score[reg];}
            sum=quad_sum(sum);if(lane%4==0){s.mx[wg][row]=local_mx[r];s.den[wg][row]=sum;}
        }
        copy_wait();__syncthreads();
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
            int row=warp*16+lane/4+r*8;mx[r]=fmaxf(s.mx[0][row],s.mx[1][row]);float safe_m=isfinite(mx[r])?mx[r]:0.f;
            den[r]=s.den[0][row]*__expf(s.mx[0][row]-safe_m)+s.den[1][row]*__expf(s.mx[1][row]-safe_m);
            float inverse=__expf(local_mx[r]-safe_m)/fmaxf(den[r],1.e-30f);
            #pragma unroll
            for(int n=0;n<4;++n)for(int c=0;c<2;++c){
                int reg=n*4+r*2+c,key=wg*32+n*8+lane%4*2+c;score[reg]*=inverse;
                float hi=float(BF(score[reg]));ph(row,key)=BF(hi);pl(row,key)=BF(score[reg]-hi);
            }
        }
        __syncthreads();proxy();

        if constexpr(PREFETCH_V)load_v(a,s,j,kh);
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
        auto cluster=cg::this_cluster();
        int previous_rank=((position+3)%4)*2+half;
        if(position>0){
            if(t==0)while(remote_epoch(&s.route_epoch,previous_rank)<j){}
            __syncthreads();
            auto* previous=cluster.map_shared_rank(&s,previous_rank);
            #pragma unroll
            for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;
                rm[r]=previous->mx[1][row];rs[r]=previous->den[1][row];
                #pragma unroll
                for(int n=0;n<2;++n)for(int c=0;c<2;++c)state[r][n*2+c]=previous->z[row*32+wg*16+n*8+lane%4*2+c];
            }
        }
        #pragma unroll
        for(int r=0;r<2;++r){
            int row=warp*16+lane/4+r*8;
            float next=fmaxf(rm[r],mx[r]),safe=isfinite(next)?next:0.f,mass=den[r]*__expf(mx[r]-safe);
            rs[r]=rs[r]*__expf(rm[r]-safe)+mass;rm[r]=next;eta[r]=mass/fmaxf(rs[r],1.e-30f);
            float square=0.f;
            #pragma unroll
            for(int n=0;n<2;++n)for(int c=0;c<2;++c){float delta=mu[n*4+r*2+c]-state[r][n*2+c];square+=delta*delta;}
            square=quad_sum(square);if(lane%4==0)s.delta[wg][row]=eta[r]*eta[r]*square/(ref*ref);
        }
        __syncthreads();
        if(wg==0){
            float worst=0.f;
            #pragma unroll
            for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;worst=fmaxf(worst,s.delta[0][row]+s.delta[1][row]);}
            worst=warp_max(worst);if(lane==0)s.warp_risk[warp]=worst;
        }
        __syncthreads();
        if(t==0){
            s.vote[j&1]=fmaxf(fmaxf(s.warp_risk[0],s.warp_risk[1]),fmaxf(s.warp_risk[2],s.warp_risk[3]));
            publish_epoch(&s.vote_epoch,j+1);
            while(remote_epoch(&s.vote_epoch,position*2+1-half)<j+1){}
            auto* partner=cluster.map_shared_rank(&s,position*2+1-half);
            float risk=fmaxf(s.vote[j&1],partner->vote[j&1]);
            int ix=(head*((a.queries+127)/128)+qb)*tiles+j;float limit=base_limit;
            if(a.held&&j<tiles&&a.held[ix])limit*=expf(-2.f*a.sticky);
            s.take=j<tiles&&(j==0||j>=a.protect_tile||!(risk<limit));
            if(half==0&&j<tiles){a.skipped[ix]=!s.take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}
        }
        __syncthreads();bool take=s.take;
        #pragma unroll
        for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;
            if(wg==0&&lane%4==0){s.mx[1][row]=rm[r];s.den[1][row]=rs[r];}
            #pragma unroll
            for(int n=0;n<2;++n)for(int c=0;c<2;++c){
                if(take)state[r][n*2+c]=(1.f-eta[r])*state[r][n*2+c]+eta[r]*mu[n*4+r*2+c];
                s.z[row*32+wg*16+n*8+lane%4*2+c]=state[r][n*2+c];
            }
        }
        __syncthreads();if(t==0)publish_epoch(&s.route_epoch,j+1);
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
                    __floats2bfloat162_rn(mass*score[n*4+r*2],mass*score[n*4+r*2+1]);

            }
            if constexpr(!PREFETCH_V)load_v(a,s,j,kh);
            copy_wait();__syncthreads();proxy();
            auto tv=make_tensor(make_smem_ptr(s.v+wg*64*256),VL{});
            fence(output);warpgroup_arrive();
            #pragma unroll
            for(int kk=0;kk<64;kk+=16){
                auto pa=local_tile(tp,Shape<_64,_16>{},make_coord(0,kk/16));
                auto vb=local_tile(tv,Shape<_256,_16>{},make_coord(0,kk/16));
                mm<256,true>(GMMA::make_gmma_desc<GMMA::Major::K>(pa),GMMA::make_gmma_desc<GMMA::Major::MN>(vb),output);
            }
            warpgroup_commit();warpgroup_wait<0>();fence(output);
        }
        if constexpr(!PREFETCH_K)if(base+4<tiles)load_k(a,s,j+4,kh);
        copy_wait();__syncthreads();proxy();
    }
    #pragma unroll
    for(int n=0;n<32;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){
        int row=qstart+warp*16+lane/4+r*8,col=wg*256+n*8+lane%4*2+c;
        if(row<a.queries)static_cast<float*>(a.output)[((position*a.heads+head)*a.queries+row)*512+col]=output[n*4+r*2+c];
    }
    #pragma unroll
    for(int r=0;r<2;++r){int row=qstart+warp*16+lane/4+r*8;if(row<a.queries&&wg==0&&lane%4==0){
        int ix=(position*a.heads+head)*a.queries+row;a.debug_scores[ix*2]=om[r];a.debug_scores[ix*2+1]=os[r];
    }}
    if(a.trace&&position==3){
        #pragma unroll
        for(int r=0;r<2;++r){int row=qstart+warp*16+lane/4+r*8;if(row<a.queries){
            if(wg==0&&lane%4==0)a.log_normalizer[head*a.queries+row]=rm[r]+logf(fmaxf(rs[r],1.e-30f));
            #pragma unroll
            for(int n=0;n<2;++n)for(int c=0;c<2;++c)a.projected_state[(head*a.queries+row)*32+wg*16+n*8+lane%4*2+c]=state[r][n*2+c];
        }}
    }
    cluster_sync();
}
template<bool V,bool K> cudaError_t launch(Params const& p,cudaStream_t stream){
    auto fn=partitioned<V,K>;auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));if(err!=cudaSuccess)return err;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((p.queries+127)/128*8,p.heads,1);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=sizeof(Shared);cfg.stream=stream;
    return cudaLaunchKernelEx(&cfg,fn,p);
}
}
extern "C" uint32_t v2_online_sm90_abi_version(){return fmha::v2_online_sm90::ABI_VERSION;}
extern "C" uint32_t v2_online_sm90_params_size(){return sizeof(Params);}
extern "C" cudaError_t v2_online_wavefront_launch(Params const* p,cudaStream_t stream,int schedule){
    using namespace v2_pipeline;
    if(!p||p->batch!=1||p->width!=512||p->mode!=1||p->mask_kind!=0||p->heads%p->kv_heads||p->keys<1||p->queries<1)return cudaErrorInvalidValue;
    if(schedule!=0)return cudaErrorInvalidValue;
    if(schedule==0)return launch<false,false>(*p,stream);
    if(schedule==1)return launch<true,false>(*p,stream);
    if(schedule==2)return launch<true,true>(*p,stream);
    if(schedule==3)return launch<false,true>(*p,stream);
    return cudaErrorInvalidValue;
}

namespace v2_pipeline {
__global__ void combine(float const* partial,float const* stats,BF* out,int n,int h){
    int row=blockIdx.x,head=blockIdx.y,col=threadIdx.x;float mx=-INFINITY;
    #pragma unroll
    for(int part=0;part<4;++part)mx=fmaxf(mx,stats[((part*h+head)*n+row)*2]);
    float safe=isfinite(mx)?mx:0.f,den=0.f,num=0.f;
    #pragma unroll
    for(int part=0;part<4;++part){int ix=(part*h+head)*n+row;float factor=__expf(stats[ix*2]-safe);den+=stats[ix*2+1]*factor;num+=partial[ix*512+col]*factor;}
    out[(head*n+row)*512+col]=BF(num/fmaxf(den,1.e-30f));
}
}
extern "C" cudaError_t v2_online_wavefront_combine(float const* partial,float const* stats,void* out,int n,int h,cudaStream_t stream){
    v2_pipeline::combine<<<dim3(n,h),512,0,stream>>>(partial,stats,static_cast<BF*>(out),n,h);return cudaGetLastError();
}
