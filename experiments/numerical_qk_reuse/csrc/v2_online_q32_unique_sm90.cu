// SPDX-License-Identifier: Apache-2.0
// Unique QK/PZ per Q32 CTA, warp MMA, full D512 output, Q128 cluster vote.
// Junyu Gaussian32 admission and Yuhan mask-reuse lineage are unchanged.
// Experimental scheduling variant: no dense PV or second discovery traversal.
#include "v2_online_sm90.h"
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cmath>
using BF=__nv_bfloat16;
using fmha::v2_online_sm90::Params;
namespace w=nvcuda::wmma;
namespace cg=cooperative_groups;
namespace {
constexpr int DS=528,PS=72,ZS=40;
struct alignas(128) Shared {
 BF q[32*DS],k[64*DS],v[64*DS],ph[32*PS],pl[32*PS],zh[64*ZS],zl[64*ZS];
 float scores[32*64],mu[32*32],state[32*32];
 float rm[32],rs[32],om[32],os[32],mx[32],den[32],eta[32],old[32];
 float warp_risk[4],vote[2];int epoch,take;
};
__device__ __forceinline__ void copy16(void* dst,void const* src,bool valid){
 unsigned sh=static_cast<unsigned>(__cvta_generic_to_shared(dst));
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(sh),"l"(src),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void wait_copy(){asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");}
__device__ __forceinline__ float qsum(float x){x+=__shfl_xor_sync(0xffffffff,x,1,4);return x+__shfl_xor_sync(0xffffffff,x,2,4);}
__device__ __forceinline__ float qmax(float x){x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,4));return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2,4));}
__device__ __forceinline__ float wmax(float x){for(int n=16;n;n/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,n));return x;}
__device__ __forceinline__ float ex(float x){float y;asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ void publish(int* p,int e){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p));asm volatile("st.release.cluster.shared::cluster.u32 [%0],%1;"::"r"(a),"r"(e):"memory");}
__device__ __forceinline__ int observe(int* p,int rank){unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(p)),r;int e;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(a),"r"(rank));asm volatile("ld.acquire.cluster.shared::cluster.u32 %0,[%1];":"=r"(e):"r"(r):"memory");return e;}
using Acc=w::fragment<w::accumulator,16,16,16,float>;
using A=w::fragment<w::matrix_a,16,16,16,BF,w::row_major>;
using BK=w::fragment<w::matrix_b,16,16,16,BF,w::col_major>;
using BV=w::fragment<w::matrix_b,16,16,16,BF,w::row_major>;
template<bool EARLY_K>
__global__ void __cluster_dims__(4,1,1) __launch_bounds__(192,1)
online(__grid_constant__ Params const a){
 extern __shared__ __align__(128) unsigned char raw[];auto& s=*reinterpret_cast<Shared*>(raw);
 int t=threadIdx.x,warp=t/32,lane=t%32,head=blockIdx.y,rank=blockIdx.x%4,qb=blockIdx.x/4,qs=blockIdx.x*32;
 int kh=head/(a.heads/a.kv_heads),tiles=(a.keys+63)/64;
 auto loadk=[&](int j){for(int x=t;x<64*64;x+=192){int row=x/64,col=x%64*8;copy16(s.k+row*DS+col,static_cast<BF const*>(a.k)+(kh*a.keys+j*64+row)*512+col,j*64+row<a.keys);}};
 for(int x=t;x<32*64;x+=192){int row=x/64,col=x%64*8;copy16(s.q+row*DS+col,static_cast<BF const*>(a.q)+(head*a.queries+qs+row)*512+col,qs+row<a.queries);}
 loadk(0);wait_copy();
 for(int x=t;x<32*32;x+=192)s.state[x]=0.f;
 if(t<32){s.rm[t]=s.om[t]=-INFINITY;s.rs[t]=s.os[t]=0.f;}
 if(t==0)s.epoch=0;
 __syncthreads();cg::this_cluster().sync();
 Acc output[16];
 if(warp>=2){
  #pragma unroll
  for(int i=0;i<16;++i)w::fill_fragment(output[i],0.f);
 }
 for(int j=0;j<tiles;++j){
  if(warp<2){
   Acc score[4];
   #pragma unroll
   for(int n=0;n<4;++n)w::fill_fragment(score[n],0.f);
   #pragma unroll 4
   for(int kk=0;kk<512;kk+=16){A q;w::load_matrix_sync(q,s.q+warp*16*DS+kk,DS);
    #pragma unroll
    for(int n=0;n<4;++n){BK k;w::load_matrix_sync(k,s.k+n*16*DS+kk,DS);w::mma_sync(score[n],q,k,score[n]);}}
   #pragma unroll
   for(int n=0;n<4;++n)w::store_matrix_sync(s.scores+warp*16*64+n*16,score[n],64,w::mem_row_major);
  }
  __syncthreads();
  if constexpr(EARLY_K)if(j+1<tiles)loadk(j+1);
  for(int x=t;x<64*32;x+=192){int row=x/32,col=x%32;float z=j*64+row<a.keys?a.z[(kh*a.keys+j*64+row)*32+col]:0.f;BF hi=BF(z);s.zh[row*ZS+col]=hi;s.zl[row*ZS+col]=BF(z-float(hi));}
  if(t<128){int row=t/4,c=t%4;float mx=-INFINITY;
   #pragma unroll
   for(int n=0;n<16;++n){int k=c+n*4;float x=(qs+row<a.queries&&j*64+k<a.keys)?s.scores[row*64+k]*(a.scale*1.4426950408889634f):-INFINITY;s.scores[row*64+k]=x;mx=fmaxf(mx,x);}
   mx=qmax(mx);float safe=isfinite(mx)?mx:0.f,sum=0.f;
   #pragma unroll
   for(int n=0;n<16;++n){int k=c+n*4;float p=ex(s.scores[row*64+k]-safe);sum+=p;BF hi=BF(p);s.ph[row*PS+k]=hi;s.pl[row*PS+k]=BF(p-float(hi));}
   sum=qsum(sum);if(c==0){float nm=fmaxf(s.rm[row],mx),safe_nm=isfinite(nm)?nm:0.f,mass=sum*ex(mx-safe_nm);s.rs[row]=s.rs[row]*ex(s.rm[row]-safe_nm)+mass;s.rm[row]=nm;s.eta[row]=mass/fmaxf(s.rs[row],1.e-30f);s.mx[row]=mx;s.den[row]=sum;}
  }
  __syncthreads();
  if(warp<2){Acc mu[2];
   #pragma unroll
   for(int n=0;n<2;++n)w::fill_fragment(mu[n],0.f);
   #pragma unroll
   for(int kk=0;kk<64;kk+=16){A ph,pl;w::load_matrix_sync(ph,s.ph+warp*16*PS+kk,PS);w::load_matrix_sync(pl,s.pl+warp*16*PS+kk,PS);
    #pragma unroll
    for(int n=0;n<2;++n){BV hi,lo;w::load_matrix_sync(hi,s.zh+kk*ZS+n*16,ZS);w::load_matrix_sync(lo,s.zl+kk*ZS+n*16,ZS);w::mma_sync(mu[n],ph,lo,mu[n]);w::mma_sync(mu[n],pl,hi,mu[n]);w::mma_sync(mu[n],ph,hi,mu[n]);}}
   #pragma unroll
   for(int n=0;n<2;++n)w::store_matrix_sync(s.mu+warp*16*32+n*16,mu[n],32,w::mem_row_major);
  }
  __syncthreads();
  if(t<128){int row=t/4,c=t%4;float square=0.f;
   #pragma unroll
   for(int n=0;n<8;++n){int x=row*32+n*4+c;float delta=s.mu[x]-s.den[row]*s.state[x];square+=delta*delta;}
   float ratio=s.eta[row]/fmaxf(s.den[row],1.e-30f),ref=a.reference[kh];float worst=wmax(qsum(square)*ratio*ratio/(ref*ref));if(lane==0)s.warp_risk[warp]=worst;
  }
  __syncthreads();
  if(t==0){int slot=j&1;s.vote[slot]=fmaxf(fmaxf(s.warp_risk[0],s.warp_risk[1]),fmaxf(s.warp_risk[2],s.warp_risk[3]));publish(&s.epoch,j+1);float risk=0.f;
   #pragma unroll
   for(int r=0;r<4;++r){while(observe(&s.epoch,r)<j+1){}auto* peer=cg::this_cluster().map_shared_rank(&s,r);risk=fmaxf(risk,peer->vote[slot]);}
   int ix=(head*((a.queries+127)/128)+qb)*tiles+j;float limit=expf(2.f*a.log_threshold);if(a.held&&a.held[ix])limit*=expf(-2.f*a.sticky);s.take=j==0||j>=a.protect_tile||!(risk<limit);if(rank==0){a.skipped[ix]=!s.take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}
  }
  __syncthreads();
  if(s.take){
   if(t<128){int row=t/4,c=t%4;float next=fmaxf(s.om[row],s.mx[row]),safe=isfinite(next)?next:0.f,old=ex(s.om[row]-safe),beta=ex(s.mx[row]-safe),eta=s.eta[row];
    if(c==0){s.om[row]=next;s.os[row]=s.os[row]*old+s.den[row]*beta;s.old[row]=old;}
    #pragma unroll
    for(int n=0;n<16;++n){int k=n*4+c;s.ph[row*PS+k]=BF(ex(s.scores[row*64+k]-(isfinite(s.mx[row])?s.mx[row]:0.f))*beta);}
    #pragma unroll
    for(int n=0;n<8;++n){int x=row*32+n*4+c;s.state[x]=(1.f-eta)*s.state[x]+(eta/fmaxf(s.den[row],1.e-30f))*s.mu[x];}
   }
   for(int x=t;x<64*64;x+=192){int row=x/64,col=x%64*8;copy16(s.v+row*DS+col,static_cast<BF const*>(a.v)+(kh*a.keys+j*64+row)*512+col,j*64+row<a.keys);}
   wait_copy();__syncthreads();
   if(warp>=2){int qr=(warp-2)/2,dc=(warp-2)%2*256;
    #pragma unroll
    for(int n=0;n<16;++n){
     #pragma unroll
     for(int x=0;x<output[n].num_elements;++x){int row=qr*16+lane/4+((x/2)%2)*8;output[n].x[x]*=s.old[row];}}
    #pragma unroll
    for(int kk=0;kk<64;kk+=16){A p;w::load_matrix_sync(p,s.ph+qr*16*PS+kk,PS);
     #pragma unroll
     for(int n=0;n<16;++n){BV v;w::load_matrix_sync(v,s.v+kk*DS+dc+n*16,DS);w::mma_sync(output[n],p,v,output[n]);}}
   }
  }
  if constexpr(!EARLY_K)if(j+1<tiles)loadk(j+1);
  wait_copy();__syncthreads();
 }
 if(warp>=2){int qr=(warp-2)/2,dc=(warp-2)%2*256;
  #pragma unroll
  for(int n=0;n<16;++n){
   #pragma unroll
   for(int x=0;x<output[n].num_elements;++x){int row=qr*16+lane/4+((x/2)%2)*8,col=dc+n*16+lane%4*2+(x/4)*8+x%2;if(qs+row<a.queries)static_cast<BF*>(a.output)[(head*a.queries+qs+row)*512+col]=BF(output[n].x[x]/fmaxf(s.os[row],1.e-30f));}}
 }
 if(a.trace){for(int x=t;x<32*32;x+=192){int row=x/32;if(qs+row<a.queries)a.projected_state[(head*a.queries+qs)*32+x]=s.state[x];}if(t<32&&qs+t<a.queries)a.log_normalizer[head*a.queries+qs+t]=s.rm[t]*0.6931471805599453f+logf(fmaxf(s.rs[t],1.e-30f));}
 __syncthreads();cg::this_cluster().sync();
}
template<bool K> cudaError_t launch(Params const& p,cudaStream_t stream){auto fn=online<K>;auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));if(e!=cudaSuccess)return e;cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((p.queries+127)/128*4,p.heads,1);cfg.blockDim=dim3(192);cfg.dynamicSmemBytes=sizeof(Shared);cfg.stream=stream;return cudaLaunchKernelEx(&cfg,fn,p);}
}
extern "C" uint32_t v2_online_sm90_abi_version(){return fmha::v2_online_sm90::ABI_VERSION;}
extern "C" uint32_t v2_online_sm90_params_size(){return sizeof(Params);}
extern "C" cudaError_t v2_online_sm90_sm90(Params const* p,cudaStream_t stream,int schedule){if(!p||p->batch!=1||p->width!=512||p->mode!=1||p->mask_kind!=0||p->heads%p->kv_heads||p->keys<1||p->queries<1)return cudaErrorInvalidValue;if(schedule==0)return launch<false>(*p,stream);if(schedule==2||schedule==3)return launch<true>(*p,stream);return cudaErrorInvalidValue;}
