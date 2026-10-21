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
__device__ __forceinline__ float fast_exp2(float x){float y;asm volatile("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x));return y;}

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
using VL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_256,_64>{},Step<_2,_1>{}));
struct alignas(1024) Shared {
    union {BF q[128*512];BF v[64*512];};
    BF k[64*512],p[128*64],zhi[64*32],zlo[64*32];
    float warp_risk[8],old[128],denout[128];
    uint64_t qbar[2],kbar,vbar;int ready,take,ack[2],finished;
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
        SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>>>>;
    Op::fma(a,b,acc[I]...);
}
template<int N,bool TR=false>
__device__ __forceinline__ void mm(uint64_t a,uint64_t b,float (&acc)[N/2]){mm<N,TR>(a,b,acc,std::make_index_sequence<N/2>{});}
template<int N,std::size_t... I>
__device__ __forceinline__ void mmrs(uint32_t (&p)[4],uint64_t b,float (&acc)[N/2],std::index_sequence<I...>){
 using Op=std::conditional_t<N==32,SM90::GMMA::MMA_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>,SM90::GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>>;
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
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 65536;"::"r"(addr):"memory");
        #pragma unroll
        for(int col=0;col<512;col+=64){
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                ::"r"(out+col*64*2),"l"(&maps.value),"r"(col),"r"(kh*a.keys+tile*64),"r"(addr):"memory");
        }
    }
}


__device__ __forceinline__ void put_epoch(int* p,int x){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;"::"r"(a),"r"(x):"memory");}
__device__ __forceinline__ int get_epoch(int* p){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));int x;asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];":"=r"(x):"r"(a):"memory");return x;}
template<bool PREFETCH_V,bool PREFETCH_K>
__global__ void __cluster_dims__(4,1,1) __launch_bounds__(256,1)
partitioned(__grid_constant__ Params const a,__grid_constant__ TmaMaps const maps){
    extern __shared__ __align__(1024) unsigned char raw[];auto& s=*reinterpret_cast<Shared*>(raw);
    int t=threadIdx.x,u=t%128,wg=t/128,lane=t%32,warp=u/32,rank=blockIdx.x%4,qb=blockIdx.x/4,head=blockIdx.y;
    int kh=head/(a.heads/a.kv_heads),tiles=(a.keys+63)/64;auto cluster=cg::this_cluster();auto* router=cluster.map_shared_rank(&s,0);
    if(t==0){init_barrier(&s.qbar[0]);init_barrier(&s.qbar[1]);init_barrier(&s.kbar);init_barrier(&s.vbar);s.ready=0;s.ack[0]=s.ack[1]=0;s.finished=0;}
    __syncthreads();cluster_sync();
    if(rank==0){
        int qstart=qb*128+wg*64;
        if(t==0){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");load3d(s.q,&maps.query,head*a.queries+qb*128,&s.qbar[0]);load3d(s.q+64*512,&maps.query,head*a.queries+qb*128+64,&s.qbar[1]);}
        load_k(a,s,0,kh,maps);wait_barrier(&s.qbar[wg],0);wait_barrier(&s.kbar,0);__syncthreads();proxy();
        auto tq=make_tensor(make_smem_ptr(s.q+wg*64*512),QL{});float state[2][8]={{0}},rm[2]={-INFINITY,-INFINITY},rs[2]={0,0},om[2]={-INFINITY,-INFINITY},os[2]={0,0};
        float ref=a.reference[kh],limit0=expf(2.f*a.log_threshold);
        for(int j=0;j<tiles;++j){auto tk=make_tensor(make_smem_ptr(s.k),QL{});float score[32]={};fence(score);warpgroup_arrive();
            #pragma unroll
            for(int kk=0;kk<512;kk+=16){auto qa=local_tile(tq,Shape<_64,_16>{},make_coord(0,kk/16));auto kb=local_tile(tk,Shape<_64,_16>{},make_coord(0,kk/16));mm<64>(GMMA::make_gmma_desc<GMMA::Major::K>(qa),GMMA::make_gmma_desc<GMMA::Major::K>(kb),score);}
            warpgroup_commit();auto packed=reinterpret_cast<BF const*>(a.debug_scores)+(kh*tiles+j)*4096;copy16(s.zhi+t*8,packed+t*8);copy16(s.zlo+t*8,packed+2048+t*8);
            warpgroup_wait<0>();fence(score);float mx[2],den[2],eta[2];
            #pragma unroll
            for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;float maximum=-INFINITY;
                #pragma unroll
                for(int n=0;n<8;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c,key=j*64+n*8+lane%4*2+c;score[reg]=(qstart+row<a.queries&&key<a.keys)?score[reg]*(a.scale*1.4426950408889634f):-INFINITY;maximum=fmaxf(maximum,score[reg]);}
                mx[r]=quad_max(maximum);float safe=isfinite(mx[r])?mx[r]:0.f,sum=0.f;
                #pragma unroll
                for(int n=0;n<8;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c;score[reg]=fast_exp2(score[reg]-safe);sum+=score[reg];}
                den[r]=quad_sum(sum);float next=fmaxf(rm[r],mx[r]),sn=isfinite(next)?next:0.f,mass=den[r]*fast_exp2(mx[r]-sn);rs[r]=rs[r]*fast_exp2(rm[r]-sn)+mass;rm[r]=next;eta[r]=mass/fmaxf(rs[r],1.e-30f);}
            copy_wait();__syncthreads();proxy();if(j+1<tiles)load_k(a,s,j+1,kh,maps);
            auto zh=make_tensor(make_smem_ptr(s.zhi),ZL{}),zl=make_tensor(make_smem_ptr(s.zlo),ZL{});float mu[16]={};fence(mu);warpgroup_arrive();
            #pragma unroll
            for(int kk=0;kk<64;kk+=16){uint32_t hi[4],lo[4];
                #pragma unroll
                for(int n=0;n<2;++n)for(int r=0;r<2;++r){int reg=(kk/8+n)*4+r*2;float x=score[reg],y=score[reg+1],xh=float(BF(x)),yh=float(BF(y));hi[n*2+r]=pair(xh,yh);lo[n*2+r]=pair(x-xh,y-yh);}
                auto bh=local_tile(zh,Shape<_32,_16>{},make_coord(0,kk/16)),bl=local_tile(zl,Shape<_32,_16>{},make_coord(0,kk/16));mmrs<32>(hi,GMMA::make_gmma_desc<GMMA::Major::MN>(bl),mu);mmrs<32>(lo,GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);mmrs<32>(hi,GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);}
            warpgroup_commit();warpgroup_wait<0>();fence(mu);float worst=0.f;
            #pragma unroll
            for(int r=0;r<2;++r){float square=0.f;
                #pragma unroll
                for(int n=0;n<4;++n)for(int c=0;c<2;++c){float d=mu[n*4+r*2+c]-den[r]*state[r][n*2+c];square+=d*d;}
                worst=fmaxf(worst,eta[r]*eta[r]*quad_sum(square)/(den[r]*den[r]*ref*ref));}
            worst=warp_max(worst);if(lane==0)s.warp_risk[t/32]=worst;__syncthreads();
            if(t==0){if(j)while(get_epoch(&s.ack[0])<j||get_epoch(&s.ack[1])<j){}float risk=0.f;
                #pragma unroll
                for(int w=0;w<8;++w)risk=fmaxf(risk,s.warp_risk[w]);int ix=(head*((a.queries+127)/128)+qb)*tiles+j;float limit=limit0;if(a.held&&a.held[ix])limit*=expf(-2.f*a.sticky);s.take=j==0||j>=a.protect_tile||!(risk<limit);a.skipped[ix]=!s.take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}
            __syncthreads();bool take=s.take;
            if(take){
                #pragma unroll
                for(int r=0;r<2;++r){int row=wg*64+warp*16+lane/4+r*8;float next=fmaxf(om[r],mx[r]),safe=isfinite(next)?next:0.f,old=fast_exp2(om[r]-safe),beta=fast_exp2(mx[r]-safe);os[r]=os[r]*old+den[r]*beta;om[r]=next;if(lane%4==0)s.old[row]=old;
                    #pragma unroll
                    for(int n=0;n<8;++n){int key=n*8+lane%4*2;*reinterpret_cast<__nv_bfloat162*>(&s.p[row*64+key])=__floats2bfloat162_rn(beta*score[n*4+r*2],beta*score[n*4+r*2+1]);}
                    #pragma unroll
                    for(int n=0;n<4;++n)for(int c=0;c<2;++c)state[r][n*2+c]=(1.f-eta[r])*state[r][n*2+c]+(eta[r]/fmaxf(den[r],1.e-30f))*mu[n*4+r*2+c];}}
            __syncthreads();if(t==0)put_epoch(&s.ready,j+1);
            if(j+1<tiles)wait_barrier(&s.kbar,(j+1)&1);__syncthreads();proxy();
        }
        #pragma unroll
        for(int r=0;r<2;++r){int row=qstart+warp*16+lane/4+r*8;if(lane%4==0)s.denout[row-qb*128]=os[r];if(a.trace&&row<a.queries){if(lane%4==0)a.log_normalizer[head*a.queries+row]=rm[r]*0.6931471805599453f+logf(fmaxf(rs[r],1.e-30f));
            #pragma unroll
            for(int n=0;n<4;++n)for(int c=0;c<2;++c)a.projected_state[(head*a.queries+row)*32+n*8+lane%4*2+c]=state[r][n*2+c];}}
        __syncthreads();if(t==0)put_epoch(&s.finished,1);
    }else if(rank<3){
        int half=rank-1,qstart=qb*128+half*64,phase=0;float output[128]={};auto tp=make_tensor(make_smem_ptr(s.k),PL{});
        for(int j=0;j<tiles;++j){if(t==0)while(get_epoch(&router->ready)<j+1){}__syncthreads();bool take=router->take;
            if(take){load_v(a,s,j,kh,maps);
                for(int x=t*8;x<64*64;x+=256*8){int row=x/64,key=x%64;*reinterpret_cast<uint4*>(&tp(row,key))=*reinterpret_cast<uint4 const*>(&router->p[(half*64+row)*64+key]);}
            }
            // Copy old row factors before acknowledging the router's slot.
            float old[2];
            #pragma unroll
            for(int r=0;r<2;++r){int row=half*64+warp*16+lane/4+r*8;old[r]=take?router->old[row]:1.f;}
            __syncthreads();proxy();if(t==0)put_epoch(&router->ack[half],j+1);
            if(take){
                #pragma unroll
                for(int r=0;r<2;++r)if(old[r]!=1.f){
                    #pragma unroll
                    for(int n=0;n<32;++n)for(int c=0;c<2;++c)output[n*4+r*2+c]*=old[r];}
                wait_barrier(&s.vbar,phase);__syncthreads();proxy();auto tv=make_tensor(make_smem_ptr(s.v+wg*64*256),VL{});fence(output);warpgroup_arrive();
                #pragma unroll
                for(int kk=0;kk<64;kk+=16){auto pa=local_tile(tp,Shape<_64,_16>{},make_coord(0,kk/16));auto vb=local_tile(tv,Shape<_256,_16>{},make_coord(0,kk/16));mm<256,true>(GMMA::make_gmma_desc<GMMA::Major::K>(pa),GMMA::make_gmma_desc<GMMA::Major::MN>(vb),output);}
                warpgroup_commit();warpgroup_wait<0>();fence(output);phase^=1;__syncthreads();}
        }
        if(t==0)while(get_epoch(&router->finished)<1){}__syncthreads();
        #pragma unroll
        for(int n=0;n<32;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){int row=qstart+warp*16+lane/4+r*8,col=wg*256+n*8+lane%4*2+c;if(row<a.queries)static_cast<BF*>(a.output)[(head*a.queries+row)*512+col]=BF(output[n*4+r*2+c]/fmaxf(router->denout[row-qb*128],1.e-30f));}
    }
    cluster_sync();
}
template<bool V,bool K> cudaError_t launch(Params const& p,cudaStream_t stream,TmaMaps const& maps){
    auto fn=partitioned<V,K>;auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));if(err!=cudaSuccess)return err;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((p.queries+127)/128*4,p.heads,1);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=sizeof(Shared);cfg.stream=stream;
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
    if(schedule<0||schedule>3)return cudaErrorInvalidValue;
    if(schedule==0)return launch<false,false>(*p,stream,maps);
    if(schedule==1)return launch<true,false>(*p,stream,maps);
    if(schedule==2)return launch<true,true>(*p,stream,maps);
    if(schedule==3)return launch<false,true>(*p,stream,maps);
    return cudaErrorInvalidValue;
}
