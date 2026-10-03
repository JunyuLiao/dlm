// SPDX-License-Identifier: Apache-2.0
// Experimental clustered FMHA specialization. Uses TensorRT-LLM's Apache-2.0
// Hopper WGMMA instruction wrappers, CUTLASS layouts, and the XQA division of
// QK/router and PV roles. No changes to the frozen routing algorithm.
#include "value_direction.h"
#include <cuda_bf16.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <fmha/hopper/utils_hgmma_bf16.h>
#include <fmha/hopper/utils_warpgroup.h>
#include <cmath>
#include <mma.h>
#include <climits>
#include <utility>

namespace fmha::value_direction {
using BF = __nv_bfloat16;
using namespace cute;
namespace cg = cooperative_groups;
struct TmaMaps {CUtensorMap query,key;};

__device__ __forceinline__ void cluster_rendezvous() {
#if defined(VD_PTX_CLUSTER) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
    // Same release/acquire cluster barrier, not the relaxed-arrival variant.
    // All lanes of each warp participate uniformly, including PV on skips.
    asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;":::"memory");
#else
    cg::this_cluster().sync();
#endif
}

__global__ void pack_mask_kernel(uint8_t const* src,uint32_t* dst,int64_t rows,int keys) {
    int64_t word=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    int64_t words_per_row=int64_t((keys+63)/64)*2;
    int64_t row=word/words_per_row;
    int key=(word%words_per_row)*32+threadIdx.x%32;
    bool valid=row<rows&&key<keys&&src[row*keys+key]!=0;
    uint32_t packed=__ballot_sync(0xffffffff,valid);
    if(threadIdx.x%32==0&&row<rows)dst[word]=packed;
}

// Explicit, separately benchmarked SFU variant. The default preserves libdevice
// arithmetic; fast_math is never enabled implicitly by a compiler-wide flag.
#ifdef VD_FAST_SFU
#define vd_exp __expf
#define vd_log __logf
#else
#define vd_exp expf
#define vd_log logf
#endif

#ifdef VD_INLINE_ROLES
#define VD_ROLE __forceinline__
#else
#define VD_ROLE __noinline__
#endif

template<int D> struct alignas(128) Shared {
    BF q[64*D];
    BF k[64*D];
    BF v[64*D];
    BF p[2][64*64];
    float z[2][64*32];
    float scale[2][64];
    float warp_risk[4];
    int warp_eligible[4];
    float worst;
    float limit;
    int eligible;
    uint64_t q_barrier,k_barrier;
};

__device__ inline float row_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1, 4);
    return x + __shfl_xor_sync(0xffffffff, x, 2, 4);
}
__device__ inline float row_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1, 4));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2, 4));
}
__device__ inline float warp_max(float x) {
    for (int n=16; n; n/=2) x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,n));
    return x;
}
__device__ inline void named_sync(int id) {
    asm volatile("bar.sync %0, 128;"::"r"(id):"memory");
}
__device__ inline void proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
}
__device__ __forceinline__ void copy16(void* dst,void const* src,bool valid=true) {
    uint32_t shared=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(shared),"l"(src),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void copy_wait() {
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
}
__device__ inline float logadd(float a, float b) {
    float m=fmaxf(a,b);
    return isfinite(m) ? m+vd_log(vd_exp(a-m)+vd_exp(b-m)) : m;
}
__device__ inline float online_mass(float previous,float candidate,float& alpha,float& oldscale) {
#ifdef VD_ONLINE_RATIO
    // Same log-space retained-state equations, with a stable ratio. Unlike a
    // max/scaled-sum state, this does not change the stored normalizer convention.
    if(!isfinite(previous)){alpha=isfinite(candidate)?1.f:0.f;oldscale=0.f;return candidate;}
    if(!isfinite(candidate)){alpha=0.f;oldscale=1.f;return previous;}
    float difference=candidate-previous,small=vd_exp(-fabsf(difference));
    float large=__fdividef(1.f,1.f+small);
    alpha=difference>=0.f?large:small*large;
    oldscale=difference>=0.f?small*large:large;
    return fmaxf(previous,candidate)+vd_log(1.f+small);
#else
    float combined=logadd(previous,candidate),safe=isfinite(combined)?combined:0.f;
    alpha=isfinite(candidate)?vd_exp(candidate-safe):0.f;
    oldscale=vd_exp(previous-safe);
    return combined;
#endif
}
__device__ inline uint64_t timer() {uint64_t n;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(n));return n;}
__device__ inline void barrier_init(uint64_t* p) {
    uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(addr):"memory");
}
__device__ inline void barrier_expect(uint64_t* p,int bytes) {
    uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(addr),"r"(bytes):"memory");
}
__device__ inline void barrier_wait(uint64_t* p,int phase) {
    uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(p));int done;
    do {asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p;}"
                     :"=r"(done):"r"(addr),"r"(phase):"memory");}while(!done);
}
__device__ inline void tma_load(void* dst,CUtensorMap const* map,int x,int y,uint64_t* barrier) {
    uint32_t target=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t done=static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(target),"l"(map),"r"(x),"r"(y),"r"(done):"memory");
}
__device__ inline void measured(Params const& a,int tile,int field,uint64_t start) {
    if(a.timings) {
        int64_t i=(((int64_t(blockIdx.z)*a.heads+blockIdx.y)*gridDim.x+blockIdx.x)*((a.keys+63)/64)+tile)*TIMING_FIELDS+field;
        a.timings[i]=timer()-start;
    }
}

template<int D> using KL = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},Shape<_64,Int<D>>{}));
using PL = KL<64>;
// V is represented as (output-channel, key) for GMMA's B operand.
using VL = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{},Shape<_256,_64>{}));

// Isolated compiler-scheduling experiment. The original TRT wrapper exposes
// FP32 accumulators through uint32 register constraints; CUTLASS uses float
// constraints and operand fences. Both issue the same BF16 WGMMA instruction.
template<int N,bool TB,std::size_t... I>
__device__ __forceinline__ void float_mma(uint64_t a,uint64_t b,float (&acc)[N/2],std::index_sequence<I...>) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using Op=std::conditional_t<N==64,SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K,TB?GMMA::Major::MN:GMMA::Major::K>,
        SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K,TB?GMMA::Major::MN:GMMA::Major::K>>;
    Op::fma(a,b,acc[I]...);
#endif
}
template<int N,bool TB>
__device__ __forceinline__ void mma(uint64_t a,uint64_t b,float (&acc)[N/2]) {
#ifdef VD_F32_OPERANDS
    float_mma<N,TB>(a,b,acc,std::make_index_sequence<N/2>{});
#else
    Hgmma_bf16<N,false,TB>::mma(a,b,reinterpret_cast<uint32_t(&)[N/2]>(acc));
#endif
}
template<int N>
__device__ __forceinline__ void fence_accumulator(float (&acc)[N]) {
#ifdef VD_F32_OPERANDS
    #pragma unroll
    for(int i=0;i<N;++i)cute::warpgroup_fence_operand(acc[i]);
#endif
}

template<int D,bool TMA=false>
__device__ void qk(Params const& a, Shared<D>& s, int batch, int head, int qb, int tile, float (&score)[32],TmaMaps const* maps=nullptr) {
    uint64_t start= a.timings ? timer():0;
    int t=threadIdx.x, kh=head/(a.heads/a.kv_heads);
    auto tq=make_tensor(make_smem_ptr(s.q),KL<D>{});
    auto tk=make_tensor(make_smem_ptr(s.k),KL<D>{});
    if constexpr(TMA) {
        if(t==0) {
            barrier_expect(&s.k_barrier,64*D*2);
            #pragma unroll
            for(int col=0;col<D;col+=64)tma_load(s.k+col*64,&maps->key,col,(batch*a.kv_heads+kh)*a.keys+tile*64,&s.k_barrier);
        }
    } else for (int x=t*8;x<64*D;x+=128*8) {
        int row=x/D, col=x%D, pos=tile*64+row;
        auto* src=static_cast<BF const*>(a.k)+((batch*a.kv_heads+kh)*a.keys+(pos<a.keys?pos:0))*D+col;
        copy16(&tk(row,col),src,pos<a.keys);
    }
    if(a.mode==1) for (int x=t*4;x<64*32;x+=128*4) {
        int pos=tile*64+x/32;
        auto* src=a.z+((batch*a.kv_heads+kh)*a.keys+(pos<a.keys?pos:0))*32+x%32;
        copy16(s.z[0]+x,src,pos<a.keys && a.mode==1);
    }
    copy_wait();named_sync(1); proxy_fence();
    if constexpr(TMA) {barrier_wait(&s.k_barrier,tile&1);named_sync(1);}
    #pragma unroll
    for (int i=0;i<32;++i) score[i]=0.f;
    fence_accumulator(score);
    warpgroup_arrive();
    #pragma unroll
    for (int kk=0;kk<D;kk+=16) {
        auto qa=local_tile(tq,Shape<_64,_16>{},make_coord(0,kk/16));
        auto kb=local_tile(tk,Shape<_64,_16>{},make_coord(0,kk/16));
        mma<64,false>(GMMA::make_gmma_desc<GMMA::Major::K>(qa),GMMA::make_gmma_desc<GMMA::Major::K>(kb),score);
    }
    warpgroup_commit(); warpgroup_wait<0>();
    fence_accumulator(score);
    if(threadIdx.x==0)measured(a,tile,0,start);
}

struct Decision {float worst;bool eligible,skip;};
template<int D,bool SPLIT=false>
__device__ __forceinline__ Decision decide(Params const& a,Shared<D>& s);

template<int D,bool OVERLAP,bool TMA=false>
__device__ VD_ROLE void blasst_router(Params const& a,Shared<D>& s,TmaMaps const* maps=nullptr) {
    int t=threadIdx.x,lane=t%32,warp=t/32,batch=blockIdx.z,head=blockIdx.y;
    int qb=blockIdx.x/2,half=blockIdx.x%2,qstart=blockIdx.x*64;
    int tiles=(a.keys+63)/64;
    float score[32],previous[2]={-INFINITY,-INFINITY},seen[2]={-INFINITY,-INFINITY};
    qk<D,TMA>(a,s,batch,head,qb,0,score,maps);
    for(int j=0;j<tiles;++j) {
        int slot=j&1;float maximum[2];bool active[2];float worst=-INFINITY;int any=0;
        #pragma unroll
        for(int r=0;r<2;++r) {
            int row=warp*16+lane/4+r*8,qi=qstart+row;
            maximum[r]=-INFINITY;uint64_t packed=0;
            if(a.mask_kind==3&&qi<a.queries) {
                int mh=a.mask_heads==1?0:head;
                packed=static_cast<uint64_t const*>(a.mask)[((int64_t(batch)*a.mask_heads+mh)*a.queries+qi)*tiles+j];
            }
            #pragma unroll
            for(int n=0;n<8;++n)for(int c=0;c<2;++c) {
                int reg=n*4+r*2+c,key=j*64+n*8+lane%4*2+c;
                bool valid=qi<a.queries&&key<a.keys;float bias=0.f;
                if(valid&&a.mask_kind==3)valid=(packed>>(key-j*64))&1;
                else if(valid&&a.mask_kind) {
                    int mh=a.mask_heads==1?0:head;
                    int64_t offset=((int64_t(batch)*a.mask_heads+mh)*a.queries+qi)*a.keys+key;
                    if(a.mask_kind==1)valid=static_cast<uint8_t const*>(a.mask)[offset];
                    else {bias=float(static_cast<BF const*>(a.mask)[offset]);valid=isfinite(bias)&&bias>-1.e4f;}
                }
                float x=float(BF(float(BF(score[reg]))*a.scale));
                if(a.mask_kind==2)x=float(BF(x+bias));
                score[reg]=valid?x:-INFINITY;maximum[r]=fmaxf(maximum[r],score[reg]);
                if(a.debug_scores&&qi<a.queries&&key<a.keys)a.debug_scores[((int64_t(batch)*a.heads+head)*a.queries+qi)*a.keys+key]=score[reg];
            }
            maximum[r]=row_max(maximum[r]);active[r]=isfinite(maximum[r]);any|=active[r];
            float risk=active[r]?(isfinite(previous[r])?maximum[r]-seen[r]:INFINITY):-INFINITY;
            worst=fmaxf(worst,risk);seen[r]=fmaxf(seen[r],maximum[r]);
        }
        worst=warp_max(worst);int vote=__any_sync(0xffffffff,any);
        if(lane==0){s.warp_risk[warp]=worst;s.warp_eligible[warp]=vote;}
        named_sync(1);
        if(t==0){s.worst=-INFINITY;s.eligible=0;for(int w=0;w<4;++w){s.worst=fmaxf(s.worst,s.warp_risk[w]);s.eligible|=s.warp_eligible[w];}}
        auto decision=decide(a,s);
        if(t==0&&half==0){
            int64_t index=((int64_t(batch)*a.heads+head)*((a.queries+127)/128)+qb)*tiles+j;
            a.skipped[index]=decision.skip;a.eligible[index]=decision.eligible;if(a.trace)a.risks[index]=decision.worst;
        }
        if(decision.eligible&&!decision.skip){
            auto tp=make_tensor(make_smem_ptr(s.p[slot]),PL{});
            #pragma unroll
            for(int r=0;r<2;++r){
                int row=warp*16+lane/4+r*8;float ell=0.f;
                #pragma unroll
                for(int n=0;n<8;++n)for(int c=0;c<2;++c){int reg=n*4+r*2+c;score[reg]=vd_exp(score[reg]-(active[r]?maximum[r]:0.f));ell+=score[reg];}
                ell=row_sum(ell);float inv=1.f/fmaxf(ell,1.e-30f);
                float bz=active[r]?maximum[r]+vd_log(fmaxf(ell,1.e-30f)):-INFINITY;
                float combined=logadd(previous[r],bz),safe=isfinite(combined)?combined:0.f;
                float alpha=active[r]?vd_exp(bz-safe):0.f;
                if(lane%4==0)s.scale[slot][row]=vd_exp(previous[r]-safe);
                #pragma unroll
                for(int n=0;n<8;++n)*reinterpret_cast<__nv_bfloat162*>(&tp(row,n*8+lane%4*2))=
                    __floats2bfloat162_rn(alpha*(score[n*4+r*2]*inv),alpha*(score[n*4+r*2+1]*inv));
                previous[r]=combined;
            }
            // PV consumers may prefetch V only after the decision; this barrier
            // publishes P before PV. A skipped block executes neither softmax
            // nor P staging. QK(next) still overlaps retained PV(previous).
            named_sync(1);proxy_fence();cluster_rendezvous();
        }
        if constexpr(!OVERLAP)__syncthreads();
        if(j+1<tiles)qk<D,TMA>(a,s,batch,head,qb,j+1,score,maps);
    }
    #pragma unroll
    for(int r=0;r<2;++r){
        int row=qstart+warp*16+lane/4+r*8;
        if(row<a.queries){
            if(lane%4==0)a.log_normalizer[(batch*a.heads+head)*a.queries+row]=previous[r];
            if(a.trace)for(int d=0;d<8;++d)a.projected_state[((batch*a.heads+head)*a.queries+row)*32+lane%4*8+d]=0.f;
        }
    }
}

template<int D,bool SPLIT>
__device__ __forceinline__ Decision decide(Params const& a,Shared<D>& s) {
    auto cluster=cg::this_cluster();
    cluster_rendezvous();
    auto* first=cluster.map_shared_rank(&s,0);
    auto* second=cluster.map_shared_rank(&s,1);
    float worst=fmaxf(first->worst,second->worst);
    bool eligible=first->eligible||second->eligible;
    float limit=first->limit;
    cluster_rendezvous();
    return {worst,eligible,eligible && worst<limit};
}

template<int D>
__device__ __forceinline__ void projected_pv(Shared<D>& s) {
    namespace w=nvcuda::wmma;
    int t=threadIdx.x,warp=t/32;
    float* ph=reinterpret_cast<float*>(s.k);
    float* pl=ph+64*64;
    for(int i=t;i<64*64;i+=128) {
        float x=ph[i],hi=w::__float_to_tf32(x);
        ph[i]=hi;pl[i]=w::__float_to_tf32(x-hi);
    }
    for(int i=t;i<64*32;i+=128) {
        float x=s.z[0][i],hi=w::__float_to_tf32(x);
        s.z[0][i]=hi;s.z[1][i]=w::__float_to_tf32(x-hi);
    }
    named_sync(1);
    w::fragment<w::accumulator,16,16,8,float> acc[2];
    w::fill_fragment(acc[0],0.f);w::fill_fragment(acc[1],0.f);
    #pragma unroll
    for(int kk=0;kk<64;kk+=8) {
        w::fragment<w::matrix_a,16,16,8,w::precision::tf32,w::row_major> ah,al;
        w::load_matrix_sync(ah,ph+warp*16*64+kk,64);
        w::load_matrix_sync(al,pl+warp*16*64+kk,64);
        #pragma unroll
        for(int n=0;n<2;++n) {
            w::fragment<w::matrix_b,16,16,8,w::precision::tf32,w::row_major> bh,bl;
            w::load_matrix_sync(bh,s.z[0]+kk*32+n*16,32);
            w::load_matrix_sync(bl,s.z[1]+kk*32+n*16,32);
            // Accumulate smaller cross terms before the dominant product.
            w::mma_sync(acc[n],ah,bl,acc[n]);
            w::mma_sync(acc[n],al,bh,acc[n]);
            w::mma_sync(acc[n],ah,bh,acc[n]);
        }
    }
    // Z is no longer needed by any warp before its allocation is reused for mu.
    named_sync(1);
    w::store_matrix_sync(s.z[1]+warp*16*32,acc[0],32,w::mem_row_major);
    w::store_matrix_sync(s.z[1]+warp*16*32+16,acc[1],32,w::mem_row_major);
    named_sync(1);
}

// Register-fed TF32x3 avoids the full P spill/reload and WMMA staging barriers.
// The A fragment is a four-lane shuffle of the existing QK accumulator layout.
// Each warp owns 16 query rows throughout; no full-dimensional PV is used here.
template<int D,bool SHAREDP=false>
__device__ __forceinline__ void projected_registers(Shared<D>& s,float const (&p)[32],float (&mu)[2][8]) {
    using Op=SM80_16x8x8_F32TF32TF32F32_TN;
    int lane=threadIdx.x%32,l=lane%4;
    float acc[4][4]={{0}};
    constexpr int UNROLL=SHAREDP?1:8;
    #pragma unroll UNROLL
    for(int kk=0;kk<8;++kk) {
        uint32_t ah[4],al[4];
        #pragma unroll
        for(int t=0;t<2;++t) for(int r=0;r<2;++r) {
            float x;
            if constexpr(SHAREDP) {
                int row=(threadIdx.x/32)*16+lane/4+r*8;
                x=reinterpret_cast<float*>(s.k)[row*72+kk*8+l+4*t];
            } else {
                int src=(l+4*t)/2;
                float even=__shfl_sync(0xffffffff,p[kk*4+r*2],src,4);
                float odd=__shfl_sync(0xffffffff,p[kk*4+r*2+1],src,4);
                x=(l&1)?odd:even;
            }
            float hi=nvcuda::wmma::__float_to_tf32(x);
            ah[t*2+r]=__float_as_uint(hi);al[t*2+r]=__float_as_uint(nvcuda::wmma::__float_to_tf32(x-hi));
        }
        #pragma unroll
        for(int n=0;n<4;++n) {
            uint32_t bh[2],bl[2];
            #pragma unroll
            for(int t=0;t<2;++t) {
                float x=s.z[0][(kk*8+l+4*t)*32+n*8+lane/4];
                float hi=nvcuda::wmma::__float_to_tf32(x);
                bh[t]=__float_as_uint(hi);bl[t]=__float_as_uint(nvcuda::wmma::__float_to_tf32(x-hi));
            }
            Op::fma(acc[n][0],acc[n][1],acc[n][2],acc[n][3],ah[0],ah[1],ah[2],ah[3],bl[0],bl[1],acc[n][0],acc[n][1],acc[n][2],acc[n][3]);
            Op::fma(acc[n][0],acc[n][1],acc[n][2],acc[n][3],al[0],al[1],al[2],al[3],bh[0],bh[1],acc[n][0],acc[n][1],acc[n][2],acc[n][3]);
            Op::fma(acc[n][0],acc[n][1],acc[n][2],acc[n][3],ah[0],ah[1],ah[2],ah[3],bh[0],bh[1],acc[n][0],acc[n][1],acc[n][2],acc[n][3]);
        }
    }
    #pragma unroll
    for(int n=0;n<4;++n) for(int r=0;r<2;++r) for(int c=0;c<2;++c) mu[r][n*2+c]=acc[n][r*2+c];
}

template<int D,bool OVERLAP,bool FIXED=false,bool SPLIT=false,bool TMA=false,bool SHAREDP=false>
__device__ VD_ROLE void router(Params const& a,Shared<D>& s,TmaMaps const* maps=nullptr) {
    int t=threadIdx.x, lane=t%32, warp=t/32;
    int batch=blockIdx.z, head=blockIdx.y, qb=blockIdx.x/(SPLIT?4:2), half=blockIdx.x%2;
    int qstart=qb*128+half*64, kh=head/(a.heads/a.kv_heads);
    float score[32];
    float projected[2][8]={{0}}, previous[2]={-INFINITY,-INFINITY}, running[2]={-INFINITY,-INFINITY};
    float reference=a.reference[batch*a.kv_heads+kh];
    int const mode=FIXED?1:a.mode, precision=FIXED?(SHAREDP?3:2):a.projected_precision;
    int const mask_kind=FIXED?3:a.mask_kind;
    qk<D,TMA>(a,s,batch,head,qb,0,score,maps);
    int tiles=(a.keys+63)/64;
    for(int j=0;j<tiles;++j) {
        uint64_t started=a.timings?timer():0;
        int slot=j&1;
        float mu[2][8]={{0}}, alpha[2], oldscale[2], combined[2];
        bool active[2];float maximum[2];
        {
            float worst=-INFINITY; int any=0;
            #pragma unroll
            for(int r=0;r<2;++r) {
                int row=warp*16+lane/4+r*8, qi=qstart+row;
                maximum[r]=-INFINITY;
                uint64_t packed=0;
                if(mask_kind==3 && qi<a.queries) {
                    int mh=a.mask_heads==1?0:head;
                    packed=static_cast<uint64_t const*>(a.mask)[((int64_t(batch)*a.mask_heads+mh)*a.queries+qi)*tiles+j];
                }
                #pragma unroll
                for(int n=0;n<8;++n) for(int c=0;c<2;++c) {
                    int reg=n*4+r*2+c, key=j*64+n*8+(lane%4)*2+c;
                    bool valid=qi<a.queries && key<a.keys;
                    float bias=0.f;
                    if(valid && mask_kind==3) valid=(packed>>(key-j*64))&1;
                    else if(valid && mask_kind) {
                        int mh=a.mask_heads==1 ? 0:head;
                        int64_t off=((int64_t(batch)*a.mask_heads+mh)*a.queries+qi)*a.keys+key;
                        if(mask_kind==1) valid=static_cast<uint8_t const*>(a.mask)[off]!=0;
                        else {bias=float(static_cast<BF const*>(a.mask)[off]); valid=isfinite(bias)&&bias>-1.e4f;}
                    }
                    float x=float(BF(float(BF(score[reg]))*a.scale));
                    if(mask_kind==2) x=float(BF(x+bias));
                    score[reg]=valid ? x:-INFINITY;
                    if(a.debug_scores && qi<a.queries && key<a.keys)
                        a.debug_scores[((int64_t(batch)*a.heads+head)*a.queries+qi)*a.keys+key]=score[reg];
                    maximum[r]=fmaxf(maximum[r],score[reg]);
                }
                maximum[r]=row_max(maximum[r]);
                active[r]=isfinite(maximum[r]); any|=active[r];
                float ell=0.f;
                #pragma unroll
                for(int n=0;n<8;++n) for(int c=0;c<2;++c) {
                    int reg=n*4+r*2+c;
                    score[reg]=vd_exp(score[reg]-(active[r]?maximum[r]:0.f)); ell+=score[reg];
                }
                ell=row_sum(ell);
                float inverse=1.f/fmaxf(ell,1.e-30f);
                #pragma unroll
                for(int n=0;n<8;++n) for(int c=0;c<2;++c) score[n*4+r*2+c]*=inverse;
                {
                    float bz=active[r] ? maximum[r]+vd_log(fmaxf(ell,1.e-30f)):-INFINITY;
                    combined[r]=online_mass(previous[r],bz,alpha[r],oldscale[r]);
                }
                // QK has completed: reuse its K staging allocation for FP32
                // probabilities. This avoids dynamic-index local-memory loads
                // from the GMMA accumulator array during the PZ loop.
                float* scratch=reinterpret_cast<float*>(s.k);
                #pragma unroll
                for(int n=0;n<8;++n) for(int c=0;c<2;++c)
                    if(precision!=2) scratch[row*(precision==3?72:64)+n*8+(lane%4)*2+c]=score[n*4+r*2+c];
                if(lane%4==0) s.scale[slot][row]=oldscale[r];
                auto tp=make_tensor(make_smem_ptr(s.p[slot]),PL{});
                #pragma unroll
                for(int n=0;n<8;++n)
                    *reinterpret_cast<__nv_bfloat162*>(&tp(row,n*8+(lane%4)*2))=
                        __floats2bfloat162_rn(alpha[r]*score[n*4+r*2],alpha[r]*score[n*4+r*2+1]);
            }
            named_sync(1);
            if(t==0)measured(a,j,1,started);
            started=a.timings?timer():0;
            if(mode==1 && precision==1) projected_pv(s);
            if(mode==1 && precision==2) projected_registers<D,false>(s,score,mu);
            if(mode==1 && precision==3) projected_registers<D,true>(s,score,mu);
            if(t==0)measured(a,j,2,started);
            started=a.timings?timer():0;
            #pragma unroll
            for(int r=0;r<2;++r) {
                int row=warp*16+lane/4+r*8;
                if(mode==1) {
                    if(precision==1) {
                        #pragma unroll
                        for(int d=0;d<8;++d) mu[r][d]=s.z[1][row*32+(lane%4)*8+d];
                    } else if(precision<2) {
                        #pragma unroll 1
                        for(int key=0;key<64;++key) {
                            float w=reinterpret_cast<float*>(s.k)[row*64+key];
                            #pragma unroll
                            for(int d=0;d<8;++d) mu[r][d]+=w*s.z[0][key*32+(lane%4)*8+d];
                        }
                    }
                }
                float square=0.f;
                #pragma unroll
                for(int d=0;d<8;++d) {float delta=alpha[r]*(mu[r][d]-projected[r][d]);square+=delta*delta;}
                float risk;
                if constexpr(FIXED) risk=row_sum(square)/(fmaxf(reference,1.e-12f)*fmaxf(reference,1.e-12f));
                else risk=mode==1 ? vd_log(sqrtf(row_sum(square))/fmaxf(reference,1.e-12f)):
                    mode==2 ? maximum[r]-running[r]:INFINITY;
                // The physical decision is max over ROW-WEIGHTED scores, not
                // a product of two tile maxima. The fixed specialization uses
                // squared risks/thresholds; its weight must therefore square.
                if(a.sensitivity && mode==1 && active[r]) {
                    float weight=a.sensitivity[batch*a.queries+qstart+row];
                    if constexpr(FIXED) risk*=weight*weight;
                    else risk+=vd_log(weight);
                }
                // Existing experiment convention: compare against maxima of
                // PRIOR seen blocks, then update seen even if this tile drops.
                running[r]=fmaxf(running[r],maximum[r]);
                risk=active[r] ? (isfinite(previous[r])?risk:INFINITY):-INFINITY;
                worst=fmaxf(worst,risk);
            }
            worst=warp_max(worst); int vote=__any_sync(0xffffffff,any);
            if(lane==0){s.warp_risk[warp]=worst;s.warp_eligible[warp]=vote;}
            named_sync(1);
            if(t==0) {
                s.worst=-INFINITY;s.eligible=0;
                for(int w=0;w<4;++w){s.worst=fmaxf(s.worst,s.warp_risk[w]);s.eligible|=s.warp_eligible[w];}
            }
        }
        if(t==0)measured(a,j,5,started);
        started=a.timings?timer():0;
        auto decision=decide<D,SPLIT>(a,s);
        if(t==0)measured(a,j,3,started);
        bool skip=decision.skip,eligible=decision.eligible;
        if(t==0 && half==0) {
            int64_t dest=((int64_t(batch)*a.heads+head)*((a.queries+127)/128)+qb)*tiles+j;
            a.skipped[dest]=skip;a.eligible[dest]=eligible;
            if(a.trace) a.risks[dest]=FIXED?(decision.eligible?0.5f*logf(decision.worst):-INFINITY):decision.worst;
        }
        if(eligible && !skip) {
            #pragma unroll
            for(int r=0;r<2;++r){
                #pragma unroll
                for(int d=0;d<8;++d) projected[r][d]=oldscale[r]*projected[r][d]+alpha[r]*mu[r][d];
                previous[r]=combined[r];
            }
        }
        if constexpr(!OVERLAP) __syncthreads();
        if(j+1<tiles) qk<D,TMA>(a,s,batch,head,qb,j+1,score,maps);
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        int row=qstart+warp*16+lane/4+r*8;
        if(row<a.queries) {
            if(lane%4==0) a.log_normalizer[(batch*a.heads+head)*a.queries+row]=previous[r];
            if(a.trace) {
                #pragma unroll
                for(int d=0;d<8;++d) {
                    int dim=precision>=2?(d/2)*8+(lane%4)*2+d%2:(lane%4)*8+d;
                    a.projected_state[((batch*a.heads+head)*a.queries+row)*32+dim]=projected[r][d];
                }
            }
        }
    }
}

template<int D,bool OVERLAP,bool SPLIT=false>
__device__ VD_ROLE void consumer(Params const& a,Shared<D>& s) {
    int t=threadIdx.x,wg=t/128+(SPLIT?1:0),lane=t%32,warp=(t%128)/32;
    int batch=blockIdx.z,head=blockIdx.y,qstart=(SPLIT?(blockIdx.x/4*2+blockIdx.x%2):blockIdx.x)*64,kh=head/(a.heads/a.kv_heads);
    float acc[128]={};
    for(int j=0;j<(a.keys+63)/64;++j) {
        int slot=j&1;
        auto decision=decide<D,SPLIT>(a,s);
        uint64_t started=a.timings?timer():0;
        if(!decision.skip && decision.eligible) {
            if(a.mode==2)cluster_rendezvous();
            if constexpr(SPLIT) {
                auto* source=cg::this_cluster().map_shared_rank(&s,blockIdx.x%2);
                for(int x=t;x<64*64/4;x+=blockDim.x)
                    reinterpret_cast<uint2*>(s.p[slot])[x]=reinterpret_cast<uint2 const*>(source->p[slot])[x];
                if(t<64)s.scale[slot][t]=source->scale[slot][t];
                __syncthreads();
            }
            int part=wg-1, u=t%128;
            auto tv=make_tensor(make_smem_ptr(s.v+part*64*256),VL{});
            for(int x=u*8;x<64*256;x+=128*8) {
                int key=x/256, col=x%256, pos=j*64+key;
                auto* src=static_cast<BF const*>(a.v)+((batch*a.kv_heads+kh)*a.keys+(pos<a.keys?pos:0))*D+part*256+col;
                copy16(&tv(col,key),src,pos<a.keys);
            }
            #pragma unroll
            for(int n=0;n<32;++n) for(int r=0;r<2;++r) for(int c=0;c<2;++c)
                acc[n*4+r*2+c]*=s.scale[slot][warp*16+lane/4+r*8];
            copy_wait();named_sync(wg+1);proxy_fence();fence_accumulator(acc);warpgroup_arrive();
            auto tp=make_tensor(make_smem_ptr(s.p[slot]),PL{});
            #pragma unroll
            for(int kk=0;kk<64;kk+=16) {
                auto pa=local_tile(tp,Shape<_64,_16>{},make_coord(0,kk/16));
                auto vb=local_tile(tv,Shape<_256,_16>{},make_coord(0,kk/16));
                mma<256,true>(GMMA::make_gmma_desc<GMMA::Major::K>(pa),GMMA::make_gmma_desc<GMMA::Major::MN>(vb),acc);
            }
            warpgroup_commit();warpgroup_wait<0>();
            fence_accumulator(acc);
        }
        if(t==(SPLIT?0:128))measured(a,j,4,started);
        if constexpr(!OVERLAP) __syncthreads();
    }
    {
        int part=wg-1;
        #pragma unroll
        for(int n=0;n<32;++n) for(int r=0;r<2;++r) for(int c=0;c<2;++c) {
            int row=qstart+warp*16+lane/4+r*8, col=part*256+n*8+(lane%4)*2+c;
            if(row<a.queries) static_cast<BF*>(a.output)[((batch*a.heads+head)*a.queries+row)*D+col]=BF(acc[n*4+r*2+c]);
        }
    }
}

// D512 alternative: router and PV live on separate CTAs in one four-CTA
// physical-tile cluster. This gives each role its own register file instead of
// spilling the router under the 384-thread per-CTA launch bound. Two query-half
// routers own votes; two PV CTAs consume their double-buffered P through DSM.
// QK/PZ(next) can progress independently of retained PV(previous).
template<bool TMA=false,bool SHAREDP=false>
__global__ void __cluster_dims__(4,1,1) __launch_bounds__(256,1) split_kernel(__grid_constant__ Params const a,__grid_constant__ TmaMaps const maps) {
    extern __shared__ __align__(1024) unsigned char storage[];
    auto& s=*reinterpret_cast<Shared<512>*>(storage);
    int rank=blockIdx.x%4,t=threadIdx.x,batch=blockIdx.z,head=blockIdx.y;
    int qstart=(blockIdx.x/4*2+rank%2)*64;
    if(t==0)s.limit=expf(2.f*a.log_threshold);
    if(rank<2) {
        auto tq=make_tensor(make_smem_ptr(s.q),KL<512>{});
        if constexpr(TMA) {
            if(t==0){barrier_init(&s.q_barrier);barrier_init(&s.k_barrier);}
            __syncthreads();
            if(t==0){
                asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
                barrier_expect(&s.q_barrier,64*512*2);
                #pragma unroll
                for(int col=0;col<512;col+=64)tma_load(s.q+col*64,&maps.query,col,(batch*a.heads+head)*a.queries+qstart,&s.q_barrier);
            }
            barrier_wait(&s.q_barrier,0);
        } else for(int x=t*8;x<64*512;x+=blockDim.x*8) {
            int row=x/512,col=x%512;
            auto* src=static_cast<BF const*>(a.q)+((batch*a.heads+head)*a.queries+(qstart+row<a.queries?qstart+row:0))*512+col;
            copy16(&tq(row,col),src,qstart+row<a.queries);
        }
        copy_wait();__syncthreads();proxy_fence();
        if(t<128)router<512,true,true,true,TMA,SHAREDP>(a,s,&maps);
        else for(int j=0;j<(a.keys+63)/64;++j)decide<512,true>(a,s);
    } else consumer<512,true,true>(a,s);
    cluster_rendezvous();
}

template<bool TMA=false,bool SHAREDP=false>
cudaError_t launch_split(Params a,cudaStream_t stream,TmaMaps maps={}) {
    auto fn=split_kernel<TMA,SHAREDP>;
    auto error=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<512>));
    if(error!=cudaSuccess)return error;
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3((a.queries+127)/128*4,a.heads,a.batch);
    cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=sizeof(Shared<512>);cfg.stream=stream;
    return cudaLaunchKernelEx(&cfg,fn,a,maps);
}

template<int D,bool OVERLAP,bool FIXED=false,bool TMA=false,bool SHAREDP=false>
__global__ void __cluster_dims__(2,1,1) __launch_bounds__(128*(1+D/256),1) kernel(__grid_constant__ Params const a,__grid_constant__ TmaMaps const maps) {
    extern __shared__ __align__(1024) unsigned char storage[];
    auto& s=*reinterpret_cast<Shared<D>*>(storage);
    int t=threadIdx.x,batch=blockIdx.z,head=blockIdx.y,qstart=blockIdx.x*64;
    if(t==0)s.limit=FIXED?expf(2.f*a.log_threshold):a.log_threshold;
    auto tq=make_tensor(make_smem_ptr(s.q),KL<D>{});
    if constexpr(TMA) {
        if(t==0) {barrier_init(&s.q_barrier);barrier_init(&s.k_barrier);}
        __syncthreads();
        if(t==0) {
            asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
            barrier_expect(&s.q_barrier,64*D*2);
            #pragma unroll
            for(int col=0;col<D;col+=64)tma_load(s.q+col*64,&maps.query,col,(batch*a.heads+head)*a.queries+qstart,&s.q_barrier);
        }
        barrier_wait(&s.q_barrier,0);
    } else for(int x=t*8;x<64*D;x+=blockDim.x*8) {
        int row=x/D,col=x%D;
        auto* src=static_cast<BF const*>(a.q)+((batch*a.heads+head)*a.queries+(qstart+row<a.queries?qstart+row:0))*D+col;
        copy16(&tq(row,col),src,qstart+row<a.queries);
    }
    copy_wait();__syncthreads();proxy_fence();
    if(t<128) {
        if constexpr(!FIXED){
            if(a.mode==2)blasst_router<D,OVERLAP,TMA>(a,s,&maps);
            else router<D,OVERLAP,FIXED,false,TMA,SHAREDP>(a,s,&maps);
        } else router<D,OVERLAP,FIXED,false,TMA,SHAREDP>(a,s,&maps);
    } else {
        consumer<D,OVERLAP>(a,s);
    }
    // Keep distributed shared memory alive until the partner completes.
    cluster_rendezvous();
}

template<int D,bool O,bool FIXED=false,bool TMA=false,bool SHAREDP=false> cudaError_t launch(Params a,cudaStream_t stream,TmaMaps maps={}) {
    auto fn=kernel<D,O,FIXED,TMA,SHAREDP>;
    auto error=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared<D>));
    if(error!=cudaSuccess)return error;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim=dim3(((a.queries+127)/128)*2,a.heads,a.batch);
    cfg.blockDim=dim3(128*(1+D/256));cfg.dynamicSmemBytes=sizeof(Shared<D>);cfg.stream=stream;
    return cudaLaunchKernelEx(&cfg,fn,a,maps);
}
} // namespace fmha::value_direction

extern "C" uint32_t value_direction_abi_version() {return fmha::value_direction::ABI_VERSION;}
extern "C" uint32_t value_direction_params_size() {return sizeof(fmha::value_direction::Params);}

namespace {
// Validate both exported schedules identically. Kernel addressing currently uses
// signed 32-bit Q/K/V offsets; reject oversized tensors rather than overflowing.
bool valid_params(fmha::value_direction::Params const* p,int schedule) {
    if(!p || !p->q || !p->k || !p->v || !p->z || !p->reference || !p->output || !p->skipped || !p->eligible || !p->log_normalizer ||
       p->batch<1 || p->heads<1 || p->kv_heads<1 || p->heads%p->kv_heads || p->queries<1 || p->keys<1 ||
       (p->width!=256 && p->width!=512) || p->batch>65535 || p->heads>65535 ||
       p->mask_kind<0 || p->mask_kind>3 || (p->mask_kind && !p->mask) ||
       (p->mask_heads!=1 && p->mask_heads!=p->heads) || p->mode<0 || p->mode>2 ||
       p->trace<0 || p->trace>1 || (p->trace && (!p->projected_state||!p->risks)) ||
       p->projected_precision<0 || p->projected_precision>3 || schedule<0 || schedule>2 ||
       !std::isfinite(p->scale) || std::isnan(p->log_threshold))return false;
    if(int64_t(p->batch)*p->heads*p->queries*p->width>INT_MAX ||
       int64_t(p->batch)*p->kv_heads*p->keys*p->width>INT_MAX)return false;
    for(auto ptr:{p->q,p->k,p->v,static_cast<void const*>(p->z),static_cast<void const*>(p->output)})
        if(reinterpret_cast<uintptr_t>(ptr)%16)return false;
    if(p->mask_kind==3 && reinterpret_cast<uintptr_t>(p->mask)%8)return false;
    if(schedule==2 && (p->width!=512 || p->mode!=1 || p->mask_kind!=3 || p->projected_precision<2))return false;
    return true;
}
}

extern "C" cudaError_t value_direction_pack_mask(uint8_t const* src,uint64_t* dst,int batch,int heads,int queries,int keys,cudaStream_t stream) {
    if(!src||!dst||batch<1||heads<1||queries<1||keys<1)return cudaErrorInvalidValue;
    int64_t rows=int64_t(batch)*heads*queries,words=rows*((keys+63)/64)*2;
    if((words+7)/8>2147483647)return cudaErrorInvalidValue;
    fmha::value_direction::pack_mask_kernel<<<(words+7)/8,256,0,stream>>>(src,reinterpret_cast<uint32_t*>(dst),rows,keys);
    return cudaGetLastError();
}

extern "C" cudaError_t value_direction_sm90_tma(fmha::value_direction::Params const* p,cudaStream_t stream,int overlap) {
    using namespace fmha::value_direction;
    if(!valid_params(p,overlap)||p->mask_kind!=3||p->projected_precision<2||!overlap)return cudaErrorInvalidValue;
    TmaMaps maps{};uint32_t box[2]={64,64},stride[2]={1,1};uint64_t pitch[1]={uint64_t(p->width)*2};
    uint64_t qdims[2]={uint64_t(p->width),uint64_t(p->batch)*p->heads*p->queries};
    uint64_t kdims[2]={uint64_t(p->width),uint64_t(p->batch)*p->kv_heads*p->keys};
    auto encode=[&](CUtensorMap* map,void const* ptr,uint64_t* dims) {
        return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,const_cast<void*>(ptr),dims,pitch,box,stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    if(encode(&maps.query,p->q,qdims)!=CUDA_SUCCESS||encode(&maps.key,p->k,kdims)!=CUDA_SUCCESS)return cudaErrorInvalidValue;
    if(p->mode!=1) {
        return p->width==256?launch<256,true,false,true>(*p,stream,maps):launch<512,true,false,true>(*p,stream,maps);
    }
    if(p->projected_precision==3) {
        if(overlap==2&&p->width==512)return launch_split<true,true>(*p,stream,maps);
        return p->width==256?launch<256,true,true,true,true>(*p,stream,maps):launch<512,true,true,true,true>(*p,stream,maps);
    }
    if(overlap==2&&p->width==512)return launch_split<true>(*p,stream,maps);
    return p->width==256?launch<256,true,true,true>(*p,stream,maps):launch<512,true,true,true>(*p,stream,maps);
}

extern "C" cudaError_t value_direction_sm90(fmha::value_direction::Params const* p,cudaStream_t stream,int overlap) {
    using namespace fmha::value_direction;
    if(!valid_params(p,overlap))return cudaErrorInvalidValue;
    if(p->mode==1 && p->projected_precision==3 && p->mask_kind==3) {
        if(overlap==2&&p->width==512)return launch_split<false,true>(*p,stream);
        if(p->width==256)return overlap?launch<256,true,true,false,true>(*p,stream):launch<256,false,true,false,true>(*p,stream);
        if(p->width==512)return overlap?launch<512,true,true,false,true>(*p,stream):launch<512,false,true,false,true>(*p,stream);
    }
    if(p->mode==1 && p->projected_precision==2 && p->mask_kind==3) {
        if(overlap==2 && p->width==512)return launch_split(*p,stream);
        if(p->width==256)return overlap?launch<256,true,true>(*p,stream):launch<256,false,true>(*p,stream);
        if(p->width==512)return overlap?launch<512,true,true>(*p,stream):launch<512,false,true>(*p,stream);
    }
    if(p->width==256)return overlap?launch<256,true>(*p,stream):launch<256,false>(*p,stream);
    if(p->width==512)return overlap?launch<512,true>(*p,stream):launch<512,false>(*p,stream);
    return cudaErrorInvalidValue;
}
