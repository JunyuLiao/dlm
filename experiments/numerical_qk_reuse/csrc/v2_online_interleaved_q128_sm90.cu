// SPDX-License-Identifier: Apache-2.0
// Interleaved Q128 producers and local Q128 routing, two WGs per role CTA.
// Separate optimization variant of the preserved three-WG bounded queue design.
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
using QL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_128>{}));
using PL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,_64>{}));
using ZL=decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<BF>{},Shape<_32,_64>{}));
using VL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_128,_64>{},Step<_2,_1>{}));
struct alignas(128) ScoreQueue {float scores[2][128*64];int ready[2][32],done[2][32];};
struct alignas(128) Queue {
 BF p[2][128*64];float old[2][128],denout[128],vote[2];
 int ready[2][32],done[2][128],epoch[32],finished[32],tile[2];
};
static_assert(2*sizeof(ScoreQueue)+sizeof(Queue)<=196608);
struct alignas(128) Shared {
 union{struct{BF q[128*128],k[64*128];};struct{BF v[64*128],p[128*64];};struct{BF zhi[64*32],zlo[64*32];};};
 float warp_risk[8];int take,next;uint64_t qbar[2],kbar;
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
 using Op=SM90::GMMA::MMA_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>;
 Op::fma(p[0],p[1],p[2],p[3],b,acc[I]...);
}
template<int N> __device__ __forceinline__ void mmrs(uint32_t (&p)[4],uint64_t b,float (&acc)[N/2]){mmrs<N>(p,b,acc,std::make_index_sequence<N/2>{});}
__device__ __forceinline__ uint32_t pair(float a,float b){auto x=__floats2bfloat162_rn(a,b);return *reinterpret_cast<uint32_t*>(&x);}
__device__ __forceinline__ float fast_exp2(float x){float y;asm volatile("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x));return y;}
template<int N> __device__ __forceinline__ void fence(float (&a)[N]){
    #pragma unroll
    for(int i=0;i<N;++i)warpgroup_fence_operand(a[i]);
}
__device__ __forceinline__ void init_barrier(uint64_t* p){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(addr):"memory");
}
__device__ __forceinline__ void load3d(void* dest,CUtensorMap const* map,int row,int part,uint64_t* p){
 unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p)),out=static_cast<unsigned>(__cvta_generic_to_shared(dest));
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;"::"r"(addr):"memory");
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"::"r"(out),"l"(map),"r"(0),"r"(row),"r"(part*2),"r"(addr):"memory");
}
__device__ __forceinline__ void wait_barrier(uint64_t* p,int phase){
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(p));int done;
    do{asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p;}"
        :"=r"(done):"r"(addr),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ void wg_sync(int id){asm volatile("bar.sync %0, 128;"::"r"(id):"memory");}
__device__ __forceinline__ void publish(int* p,int value){asm volatile("st.release.gpu.global.u32 [%0], %1;"::"l"(p),"r"(value):"memory");}
__device__ __forceinline__ int observe(int* p){int value;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(p):"memory");return value;}

__device__ __forceinline__ void wait_epoch(int* p,int expected){while(observe(p)<expected)__nanosleep(64);}
__global__ void __launch_bounds__(256,2)
interleaved(__grid_constant__ Params const a,__grid_constant__ TmaMaps const maps){
 extern __shared__ __align__(128) unsigned char raw[];auto& s=*reinterpret_cast<Shared*>(raw);
 int t=threadIdx.x,u=t%128,wg=t/128,lane=t%32,warp=u/32,units=a.heads*((a.queries+127)/128);
 int role=int(blockIdx.x)/units,unit=int(blockIdx.x)%units,q128=unit%((a.queries+127)/128),head=unit/((a.queries+127)/128);
 int qb=q128,half=wg,qs=q128*128+wg*64,kh=head/(a.heads/a.kv_heads),tiles=(a.keys+63)/64;
 auto* sq=reinterpret_cast<ScoreQueue*>(const_cast<void*>(a.mask));
 auto* queues=reinterpret_cast<Queue*>(sq+units*2);auto& queue=queues[unit];
 if(t==0){init_barrier(&s.qbar[0]);init_barrier(&s.qbar[1]);init_barrier(&s.kbar);
  if(role<2){auto& q=sq[unit*2+role];q.ready[0][0]=q.ready[1][0]=q.done[0][0]=q.done[1][0]=0;}
  if(role==2){queue.epoch[0]=queue.finished[0]=queue.ready[0][0]=queue.ready[1][0]=0;
   for(int slot=0;slot<2;++slot)for(int c=0;c<4;++c)queue.done[slot][c*32]=0;}}
 __syncthreads();cg::this_grid().sync();
 if(role<2){
  auto& q=sq[unit*2+role];auto tq=make_tensor(make_smem_ptr(s.q+wg*64*128),QL{}),tk=make_tensor(make_smem_ptr(s.k),QL{});
  for(int j=role;j<tiles;j+=2){int sequence=j/2,slot=sequence&1;
   if(t==0&&sequence>=2)wait_epoch(&q.done[slot][0],sequence-1);__syncthreads();
   float score[32]={};
   #pragma unroll 1
   for(int part=0;part<4;++part){
    if(t==0){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");load3d(s.q,&maps.query,head*a.queries+qb*128,part,&s.qbar[0]);load3d(s.q+64*128,&maps.query,head*a.queries+qb*128+64,part,&s.qbar[1]);load3d(s.k,&maps.key,kh*a.keys+j*64,part,&s.kbar);}
    wait_barrier(&s.qbar[wg],part&1);wait_barrier(&s.kbar,part&1);__syncthreads();proxy();fence(score);warpgroup_arrive();
    #pragma unroll
    for(int kk=0;kk<128;kk+=16){auto qa=local_tile(tq,Shape<_64,_16>{},make_coord(0,kk/16));auto kb=local_tile(tk,Shape<_64,_16>{},make_coord(0,kk/16));mm<64>(GMMA::make_gmma_desc<GMMA::Major::K>(qa),GMMA::make_gmma_desc<GMMA::Major::K>(kb),score);}
    warpgroup_commit();warpgroup_wait<0>();fence(score);__syncthreads();
   }
   #pragma unroll
   for(int n=0;n<8;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){int row=wg*64+warp*16+lane/4+r*8,key=n*8+lane%4*2+c;__stcg(&q.scores[slot][row*64+key],score[n*4+r*2+c]);}
   __syncthreads();if(t==0)publish(&q.ready[slot][0],sequence+1);
  }
 }else if(role==2){
  float state[2][8]={{0}},rm[2]={-INFINITY,-INFINITY},rs[2]={0,0},om[2]={-INFINITY,-INFINITY},os[2]={0,0};
  float ref=a.reference[kh],limit0=expf(2.f*a.log_threshold),held_factor=expf(-2.f*a.sticky),inv_ref=1.f/(ref*ref);
  int sequence=0;
  for(int j=0;j<tiles;++j){
   int source=j&1,qi=j/2,sslot=qi&1;auto& q=sq[unit*2+source];
   if(t==0)wait_epoch(&q.ready[sslot][0],qi+1);__syncthreads();float score[32];
   #pragma unroll
   for(int n=0;n<8;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){int row=wg*64+warp*16+lane/4+r*8,key=n*8+lane%4*2+c;score[n*4+r*2+c]=__ldcg(&q.scores[sslot][row*64+key]);}
   __syncthreads();if(t==0)publish(&q.done[sslot][0],qi+1);
   auto packed=reinterpret_cast<BF const*>(a.debug_scores)+(kh*tiles+j)*4096;
   copy16(s.zhi+t*8,packed+t*8);copy16(s.zlo+t*8,packed+2048+t*8);
   float mx[2],den[2],eta[2];
   #pragma unroll
   for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;float maximum=-INFINITY;
    #pragma unroll
    for(int n=0;n<8;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c,key=j*64+n*8+lane%4*2+c;score[reg]=(qs+row<a.queries&&key<a.keys)?score[reg]*(a.scale*1.4426950408889634f):-INFINITY;maximum=fmaxf(maximum,score[reg]);}
    mx[r]=quad_max(maximum);float safe=isfinite(mx[r])?mx[r]:0.f,sum=0.f;
    #pragma unroll
    for(int n=0;n<8;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c;score[reg]=fast_exp2(score[reg]-safe);sum+=score[reg];}
    den[r]=quad_sum(sum);float next=fmaxf(rm[r],mx[r]),sn=isfinite(next)?next:0.f,mass=den[r]*fast_exp2(mx[r]-sn);rs[r]=rs[r]*fast_exp2(rm[r]-sn)+mass;rm[r]=next;eta[r]=mass/fmaxf(rs[r],1.e-30f);
   }
   copy_wait();__syncthreads();proxy();auto zh=make_tensor(make_smem_ptr(s.zhi),ZL{}),zl=make_tensor(make_smem_ptr(s.zlo),ZL{});float mu[16]={};fence(mu);warpgroup_arrive();
   #pragma unroll
   for(int kk=0;kk<64;kk+=16){uint32_t hi[4],lo[4];
    #pragma unroll
    for(int n=0;n<2;++n)for(int r=0;r<2;++r){int reg=(kk/8+n)*4+r*2;float x=score[reg],y=score[reg+1],xh=float(BF(x)),yh=float(BF(y));hi[n*2+r]=pair(xh,yh);lo[n*2+r]=pair(x-xh,y-yh);}
    auto bh=local_tile(zh,Shape<_32,_16>{},make_coord(0,kk/16));auto bl=local_tile(zl,Shape<_32,_16>{},make_coord(0,kk/16));mmrs<32>(hi,GMMA::make_gmma_desc<GMMA::Major::MN>(bl),mu);mmrs<32>(lo,GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);mmrs<32>(hi,GMMA::make_gmma_desc<GMMA::Major::MN>(bh),mu);}
   warpgroup_commit();warpgroup_wait<0>();fence(mu);float worst=0.f;
   #pragma unroll
   for(int r=0;r<2;++r){float square=0.f;
    #pragma unroll
    for(int n=0;n<4;++n)for(int c=0;c<2;++c){float delta=mu[n*4+r*2+c]-den[r]*state[r][n*2+c];square+=delta*delta;}
    float ratio=eta[r]/fmaxf(den[r],1.e-30f);worst=fmaxf(worst,ratio*ratio*quad_sum(square)*inv_ref);}
   worst=warp_max(worst);if(lane==0)s.warp_risk[t/32]=worst;__syncthreads();int slot=sequence&1;
   if(t==0){float risk=0.f;
    #pragma unroll
    for(int w=0;w<8;++w)risk=fmaxf(risk,s.warp_risk[w]);
    int ix=(head*((a.queries+127)/128)+qb)*tiles+j;float limit=limit0;if(a.held&&a.held[ix])limit*=held_factor;s.take=j==0||j>=a.protect_tile||!(risk<limit);a.skipped[ix]=!s.take;a.eligible[ix]=1;if(a.trace)a.risks[ix]=sqrtf(risk);}

   __syncthreads();if(s.take){
    if(t==0&&sequence>=2)for(int c=0;c<4;++c)wait_epoch(&queue.done[slot][c*32],sequence-1);__syncthreads();
    #pragma unroll
    for(int r=0;r<2;++r){int row=wg*64+warp*16+lane/4+r*8;float next=fmaxf(om[r],mx[r]),safe=isfinite(next)?next:0.f,old=fast_exp2(om[r]-safe),beta=fast_exp2(mx[r]-safe);os[r]=os[r]*old+den[r]*beta;om[r]=next;if(lane%4==0)__stcg(&queue.old[slot][row],old);
     #pragma unroll
     for(int n=0;n<8;++n){auto bits=__floats2bfloat162_rn(beta*score[n*4+r*2],beta*score[n*4+r*2+1]);__stcg(reinterpret_cast<uint32_t*>(&queue.p[slot][row*64+n*8+lane%4*2]),*reinterpret_cast<uint32_t*>(&bits));}
     #pragma unroll
     for(int n=0;n<4;++n)for(int c=0;c<2;++c)state[r][n*2+c]=(1.f-eta[r])*state[r][n*2+c]+(eta[r]/fmaxf(den[r],1.e-30f))*mu[n*4+r*2+c];}
    __syncthreads();if(t==0){queue.tile[slot]=j;publish(&queue.ready[slot][0],sequence+1);}sequence++;
   }
  }
  #pragma unroll
  for(int r=0;r<2;++r){int row=qs+warp*16+lane/4+r*8;if(lane%4==0)__stcg(&queue.denout[row-qb*128],os[r]);if(a.trace&&row<a.queries){if(lane%4==0)a.log_normalizer[head*a.queries+row]=rm[r]*0.6931471805599453f+logf(fmaxf(rs[r],1.e-30f));
   #pragma unroll
   for(int n=0;n<4;++n)for(int c=0;c<2;++c)a.projected_state[(head*a.queries+row)*32+n*8+lane%4*2+c]=state[r][n*2+c];}}
  __syncthreads();if(t==0)publish(&queue.finished[0],1);
 }else{
  int channel=role-3,phase=0;float output[64]={};auto tp=make_tensor(make_smem_ptr(s.p+wg*64*64),PL{});auto tv=make_tensor(make_smem_ptr(s.v),VL{});
  for(int sequence=0;;++sequence){int slot=sequence&1;
   if(t==0){while(observe(&queue.ready[slot][0])<sequence+1&&!observe(&queue.finished[0]))__nanosleep(64);s.next=observe(&queue.ready[slot][0])>=sequence+1;}__syncthreads();if(!s.next)break;
   int j=queue.tile[slot];if(t==0){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");load3d(s.v,&maps.value,kh*a.keys+j*64,channel,&s.kbar);}
   for(int x=u*2;x<64*64;x+=128*2){int row=x/64,key=x%64;*reinterpret_cast<uint32_t*>(&tp(row,key))=__ldcg(reinterpret_cast<uint32_t*>(&queue.p[slot][wg*64*64+x]));}
   #pragma unroll
   for(int r=0;r<2;++r){int row=warp*16+lane/4+r*8;float old=__ldcg(&queue.old[slot][wg*64+row]);if(old!=1.f){
    #pragma unroll
    for(int n=0;n<16;++n)for(int c=0;c<2;++c)output[n*4+r*2+c]*=old;}}
   wait_barrier(&s.kbar,phase);__syncthreads();proxy();fence(output);warpgroup_arrive();
   #pragma unroll
   for(int kk=0;kk<64;kk+=16){auto pa=local_tile(tp,Shape<_64,_16>{},make_coord(0,kk/16));auto vb=local_tile(tv,Shape<_128,_16>{},make_coord(0,kk/16));mm<128>(GMMA::make_gmma_desc<GMMA::Major::K>(pa),GMMA::make_gmma_desc<GMMA::Major::MN>(vb),output);}
   warpgroup_commit();warpgroup_wait<0>();fence(output);phase^=1;__syncthreads();if(t==0)publish(&queue.done[slot][channel*32],sequence+1);
  }
  #pragma unroll
  for(int n=0;n<16;++n)for(int r=0;r<2;++r)for(int c=0;c<2;++c){int row=qs+warp*16+lane/4+r*8,col=channel*128+n*8+lane%4*2+c;if(row<a.queries)static_cast<BF*>(a.output)[(head*a.queries+row)*512+col]=BF(output[n*4+r*2+c]/fmaxf(__ldcg(&queue.denout[row-qb*128]),1.e-30f));}
 }
}
cudaError_t launch(Params const& p,cudaStream_t stream,TmaMaps const& maps){
 auto fn=interleaved;auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));if(e!=cudaSuccess)return e;
 int units=p.heads*((p.queries+127)/128),sm=0,per=0;cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per,fn,256,sizeof(Shared));if(units*7>sm*per)return cudaErrorInvalidConfiguration;
 Params args=p;TmaMaps m=maps;void* argv[]={&args,&m};return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(fn),dim3(units*7),dim3(256),argv,sizeof(Shared),stream);
}
}
extern "C" uint32_t v2_online_sm90_abi_version(){return fmha::v2_online_sm90::ABI_VERSION;}
extern "C" uint32_t v2_online_sm90_params_size(){return sizeof(Params);}
extern "C" cudaError_t v2_online_sm90_sm90(Params const* p,cudaStream_t stream,int schedule){
 using namespace v2_pipeline;if(!p||!p->mask||!p->debug_scores||p->batch!=1||p->width!=512||p->mode!=1||p->mask_kind!=0||p->heads%p->kv_heads||p->keys<1||p->queries<1||(schedule!=0&&schedule!=3))return cudaErrorInvalidValue;
 TmaMaps maps{};uint32_t box[3]={64,64,2},strides[3]={1,1,1};uint64_t pitch[2]={1024,128},qdims[3]={64,uint64_t(p->heads)*p->queries,8},kdims[3]={64,uint64_t(p->kv_heads)*p->keys,8};
 auto encode=[&](CUtensorMap* map,void const* ptr,uint64_t* dims){return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,const_cast<void*>(ptr),dims,pitch,box,strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);};
 if(encode(&maps.query,p->q,qdims)!=CUDA_SUCCESS||encode(&maps.key,p->k,kdims)!=CUDA_SUCCESS||encode(&maps.value,p->v,kdims)!=CUDA_SUCCESS)return cudaErrorInvalidValue;return launch(*p,stream,maps);
}

