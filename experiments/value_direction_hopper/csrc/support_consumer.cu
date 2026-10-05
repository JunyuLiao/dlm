// SPDX-License-Identifier: Apache-2.0
// v11 preselected-support Hopper consumer. Building blocks (WGMMA wrappers,
// CUTLASS SW128 layouts, BF16 score/scale transform, per-tile softmax with
// log-space retained-mass accumulation, alpha-scaled P, oldscale rescaling of
// the PV accumulator) are taken from value_direction.cu's retained path. What
// is different, and why it is a separate kernel:
//   * The support is an INPUT read before any work for a tile. Both roles test
//     the same read-only (eligible && !skipped) entry and skip the tile before
//     K cp.async, QK WGMMA, V cp.async or PV WGMMA. There is no prefetch of a
//     tile whose support entry has not been read.
//   * No decide(): the decision already exists, so the two 64-row halves of a
//     Q128 tile need no cluster exchange. Each CTA is independent.
//   * Producer/consumer hand-off is one all-thread named barrier per KEPT tile;
//     both roles iterate the identical kept sequence, so arrival counts match
//     and the P/scale double-buffer slot is (kept ordinal & 1).
//   * No projected-V / sketch loads or arithmetic.
#include "support_consumer.h"
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <fmha/hopper/utils_hgmma_bf16.h>
#include <fmha/hopper/utils_warpgroup.h>
#include <cmath>

namespace fmha::support_consumer {
using BF = __nv_bfloat16;
using namespace cute;

template<int D> struct alignas(128) Shared {
    BF q[64 * D];
    BF k[64 * D];
    BF v[64 * D];
    BF p[2][64 * 64];
    float scale[2][64];
    int invalid[64];
};

__device__ inline float row_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1, 4);
    return x + __shfl_xor_sync(0xffffffff, x, 2, 4);
}
__device__ inline float row_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1, 4));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2, 4));
}
__device__ inline int row_or(int x) {
    x |= __shfl_xor_sync(0xffffffff, x, 1, 4);
    return x | __shfl_xor_sync(0xffffffff, x, 2, 4);
}
__device__ inline void named_sync(int id, int threads) {
    // bar.sync == barrier.sync.aligned: every participant executes THIS instruction.
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ inline void cross_role_sync(int id, int threads) {
    // Producer and consumer reach the same barrier from DIFFERENT instructions,
    // which .aligned forbids (compute-sanitizer synccheck flagged it). The
    // non-aligned barrier.sync is the PTX form defined for that case.
    asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ inline void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void copy16(void* dst, void const* src, bool valid = true) {
    uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(shared), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void copy_wait() { asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory"); }
__device__ inline float logadd(float a, float b) {
    float m = fmaxf(a, b);
    return isfinite(m) ? m + logf(expf(a - m) + expf(b - m)) : m;
}

template<int D> using KL = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{}, Shape<_64, Int<D>>{}));
using PL = KL<64>;
using VL = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<BF>{}, Shape<_256, _64>{}));

template<int N, bool TB>
__device__ __forceinline__ void mma(uint64_t a, uint64_t b, float (&acc)[N / 2]) {
    Hgmma_bf16<N, false, TB>::mma(a, b, reinterpret_cast<uint32_t(&)[N / 2]>(acc));
}

// Barrier ids: 1 producer warpgroup, 2/3 consumer warpgroups, 4 hand-off, 5 epilogue.
constexpr int HANDOFF = 4, EPILOGUE = 5;

__device__ __forceinline__ bool kept(Params const& a, int batch, int head, int qb, int j, int tiles) {
    int64_t index = ((int64_t(batch) * a.heads + head) * ((a.queries + 127) / 128) + qb) * tiles + j;
    return __ldg(a.eligible + index) != 0 && __ldg(a.skipped + index) == 0;
}

template<int D>
__device__ __noinline__ void producer(Params const& a, Shared<D>& s) {
    constexpr int THREADS = 128 * (1 + D / 256);
    int t = threadIdx.x, lane = t % 32, warp = t / 32;
    int batch = blockIdx.z, head = blockIdx.y, qb = blockIdx.x / 2, qstart = blockIdx.x * 64;
    int kh = head / (a.heads / a.kv_heads), tiles = (a.keys + 63) / 64;
    auto tq = make_tensor(make_smem_ptr(s.q), KL<D>{});
    auto tk = make_tensor(make_smem_ptr(s.k), KL<D>{});
    BF const* kbase = static_cast<BF const*>(a.k) + int64_t(batch) * a.k_stride[0] + int64_t(kh) * a.k_stride[1];
    float score[32], previous[2] = {-INFINITY, -INFINITY};
    int bad[2] = {0, 0}, ordinal = 0;
    for (int j = 0; j < tiles; ++j) {
        if (!kept(a, batch, head, qb, j, tiles)) continue;   // before ANY work for tile j
        int slot = ordinal & 1;
        for (int x = t * 8; x < 64 * D; x += 128 * 8) {
            int row = x / D, col = x % D, pos = j * 64 + row;
            copy16(&tk(row, col), kbase + int64_t(pos < a.keys ? pos : 0) * a.k_stride[2] + col, pos < a.keys);
        }
        copy_wait(); named_sync(1, 128); proxy_fence();
        #pragma unroll
        for (int i = 0; i < 32; ++i) score[i] = 0.f;
        warpgroup_arrive();
        #pragma unroll
        for (int kk = 0; kk < D; kk += 16) {
            auto qa = local_tile(tq, Shape<_64, _16>{}, make_coord(0, kk / 16));
            auto kb = local_tile(tk, Shape<_64, _16>{}, make_coord(0, kk / 16));
            mma<64, false>(GMMA::make_gmma_desc<GMMA::Major::K>(qa), GMMA::make_gmma_desc<GMMA::Major::K>(kb), score);
        }
        warpgroup_commit(); warpgroup_wait<0>();
        auto tp = make_tensor(make_smem_ptr(s.p[slot]), PL{});
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            int row = warp * 16 + lane / 4 + r * 8, qi = qstart + row;
            float maximum = -INFINITY;
            uint64_t packed = 0;
            if (a.mask && qi < a.queries) {
                int mh = a.mask_heads == 1 ? 0 : head;
                packed = __ldg(a.mask + ((int64_t(batch) * a.mask_heads + mh) * a.queries + qi) * tiles + j);
            }
            #pragma unroll
            for (int n = 0; n < 8; ++n) for (int c = 0; c < 2; ++c) {
                int reg = n * 4 + r * 2 + c, key = j * 64 + n * 8 + (lane % 4) * 2 + c;
                bool valid = qi < a.queries && key < a.keys;
                if (valid && a.mask) valid = (packed >> (key - j * 64)) & 1;
                else if (valid && a.window > 0) valid = key >= qi + (a.keys - a.queries) - a.window + 1;
                // Same BF16 transform as fresh T and observe_scores: BF16 QK, BF16 scaled.
                float x = float(BF(float(BF(score[reg])) * a.scale));
                bad[r] |= valid && (x != x || x == INFINITY);
                score[reg] = (valid && x > -INFINITY && x < INFINITY) ? x : -INFINITY;
                maximum = fmaxf(maximum, score[reg]);
            }
            maximum = row_max(maximum);
            bool active = isfinite(maximum);
            float ell = 0.f;
            #pragma unroll
            for (int n = 0; n < 8; ++n) for (int c = 0; c < 2; ++c) {
                int reg = n * 4 + r * 2 + c;
                score[reg] = expf(score[reg] - (active ? maximum : 0.f)); ell += score[reg];
            }
            ell = row_sum(ell);
            float inverse = 1.f / fmaxf(ell, 1.e-30f);
            float bz = active ? maximum + logf(fmaxf(ell, 1.e-30f)) : -INFINITY;
            float combined = logadd(previous[r], bz), safe = isfinite(combined) ? combined : 0.f;
            float alpha = active ? expf(bz - safe) : 0.f;
            if (lane % 4 == 0) s.scale[slot][row] = expf(previous[r] - safe);
            #pragma unroll
            for (int n = 0; n < 8; ++n)
                *reinterpret_cast<__nv_bfloat162*>(&tp(row, n * 8 + (lane % 4) * 2)) =
                    __floats2bfloat162_rn(alpha * (score[n * 4 + r * 2] * inverse), alpha * (score[n * 4 + r * 2 + 1] * inverse));
            previous[r] = combined;
        }
        proxy_fence();                   // P/scale written by this thread -> async-proxy PV reads
        cross_role_sync(HANDOFF, THREADS);    // publish P(slot) to the PV warpgroups
        ++ordinal;
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        int row = warp * 16 + lane / 4 + r * 8, qi = qstart + row;
        int invalid = row_or(bad[r]);
        if (lane % 4 == 0) {
            s.invalid[row] = invalid;
            if (qi < a.queries) {
                int64_t i = (int64_t(batch) * a.heads + head) * a.queries + qi;
                a.log_normalizer[i] = invalid ? -INFINITY : previous[r];
                a.invalid[i] = invalid;
            }
        }
    }
    if (a.counters && t == 0) {
        int32_t* c = a.counters + ((int64_t(batch) * a.heads + head) * gridDim.x + blockIdx.x) * COUNTER_FIELDS;
        c[0] = ordinal; c[1] = ordinal;  // K tiles loaded, QK WGMMA tiles issued (producer)
    }
    cross_role_sync(EPILOGUE, THREADS);
}

template<int D>
__device__ __noinline__ void consumer(Params const& a, Shared<D>& s) {
    constexpr int THREADS = 128 * (1 + D / 256);
    int t = threadIdx.x, wg = t / 128, part = wg - 1, u = t % 128, lane = t % 32, warp = u / 32;
    int batch = blockIdx.z, head = blockIdx.y, qb = blockIdx.x / 2, qstart = blockIdx.x * 64;
    int kh = head / (a.heads / a.kv_heads), tiles = (a.keys + 63) / 64;
    BF const* vbase = static_cast<BF const*>(a.v) + int64_t(batch) * a.v_stride[0] + int64_t(kh) * a.v_stride[1] + part * 256;
    auto tv = make_tensor(make_smem_ptr(s.v + part * 64 * 256), VL{});
    auto tp0 = make_tensor(make_smem_ptr(s.p[0]), PL{});
    auto tp1 = make_tensor(make_smem_ptr(s.p[1]), PL{});
    float acc[128] = {};
    int ordinal = 0;
    for (int j = 0; j < tiles; ++j) {
        if (!kept(a, batch, head, qb, j, tiles)) continue;   // same predicate, same order
        int slot = ordinal & 1;
        cross_role_sync(HANDOFF, THREADS);
        for (int x = u * 8; x < 64 * 256; x += 128 * 8) {
            int key = x / 256, col = x % 256, pos = j * 64 + key;
            copy16(&tv(col, key), vbase + int64_t(pos < a.keys ? pos : 0) * a.v_stride[2] + col, pos < a.keys);
        }
        #pragma unroll
        for (int n = 0; n < 32; ++n) for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c)
            acc[n * 4 + r * 2 + c] *= s.scale[slot][warp * 16 + lane / 4 + r * 8];
        copy_wait(); named_sync(wg + 1, 128); proxy_fence();
        warpgroup_arrive();
        #pragma unroll
        for (int kk = 0; kk < 64; kk += 16) {
            auto pa = local_tile(slot ? tp1 : tp0, Shape<_64, _16>{}, make_coord(0, kk / 16));
            auto vb = local_tile(tv, Shape<_256, _16>{}, make_coord(0, kk / 16));
            mma<256, true>(GMMA::make_gmma_desc<GMMA::Major::K>(pa), GMMA::make_gmma_desc<GMMA::Major::MN>(vb), acc);
        }
        warpgroup_commit(); warpgroup_wait<0>();
        ++ordinal;
    }
    if (a.counters && u == 0) {
        int32_t* c = a.counters + ((int64_t(batch) * a.heads + head) * gridDim.x + blockIdx.x) * COUNTER_FIELDS;
        if (part == 0) { c[2] = ordinal; c[3] = ordinal; }  // V tiles loaded, PV WGMMA tiles (first PV warpgroup)
    }
    cross_role_sync(EPILOGUE, THREADS);       // producer's invalid flags are now visible
    #pragma unroll
    for (int n = 0; n < 32; ++n) for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c) {
        int local = warp * 16 + lane / 4 + r * 8, row = qstart + local, col = part * 256 + n * 8 + (lane % 4) * 2 + c;
        if (row < a.queries)
            static_cast<BF*>(a.output)[int64_t(batch) * a.o_stride[0] + int64_t(head) * a.o_stride[1] + int64_t(row) * a.o_stride[2] + col] =
                BF(s.invalid[local] ? 0.f : acc[n * 4 + r * 2 + c]);
    }
}

template<int D>
__global__ void __launch_bounds__(128 * (1 + D / 256), 1) support_kernel(__grid_constant__ Params const a) {
    extern __shared__ __align__(1024) unsigned char storage[];
    auto& s = *reinterpret_cast<Shared<D>*>(storage);
    int t = threadIdx.x, batch = blockIdx.z, head = blockIdx.y, qstart = blockIdx.x * 64;
    auto tq = make_tensor(make_smem_ptr(s.q), KL<D>{});
    BF const* qbase = static_cast<BF const*>(a.q) + int64_t(batch) * a.q_stride[0] + int64_t(head) * a.q_stride[1];
    for (int x = t * 8; x < 64 * D; x += blockDim.x * 8) {
        int row = x / D, col = x % D, qi = qstart + row;
        copy16(&tq(row, col), qbase + int64_t(qi < a.queries ? qi : 0) * a.q_stride[2] + col, qi < a.queries);
    }
    copy_wait(); proxy_fence(); __syncthreads(); proxy_fence();
    if (t < 128) producer<D>(a, s); else consumer<D>(a, s);
}

template<int D> cudaError_t launch(Params const& a, cudaStream_t stream) {
    auto fn = support_kernel<D>;
    auto error = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared<D>));
    if (error != cudaSuccess) return error;
    dim3 grid(((a.queries + 127) / 128) * 2, a.heads, a.batch);
    fn<<<grid, 128 * (1 + D / 256), sizeof(Shared<D>), stream>>>(a);
    return cudaGetLastError();
}
}  // namespace fmha::support_consumer

extern "C" uint32_t support_consumer_abi_version() { return fmha::support_consumer::ABI_VERSION; }
extern "C" uint32_t support_consumer_params_size() { return sizeof(fmha::support_consumer::Params); }

extern "C" cudaError_t support_consumer_sm90(fmha::support_consumer::Params const* p, cudaStream_t stream) {
    using namespace fmha::support_consumer;
    if (!p || !p->q || !p->k || !p->v || !p->skipped || !p->eligible || !p->output || !p->log_normalizer || !p->invalid ||
        p->batch < 1 || p->heads < 1 || p->kv_heads < 1 || p->heads % p->kv_heads || p->queries < 1 || p->keys < 1 ||
        (p->width != 256 && p->width != 512) || p->batch > 65535 || p->heads > 65535 || p->window < 0 ||
        (p->mask && p->mask_heads != 1 && p->mask_heads != p->heads) || !std::isfinite(p->scale))
        return cudaErrorInvalidValue;
    for (auto ptr : {p->q, p->k, p->v, static_cast<void const*>(p->output)})
        if (reinterpret_cast<uintptr_t>(ptr) % 16) return cudaErrorInvalidValue;
    for (auto const* stride : {p->q_stride, p->k_stride, p->v_stride})
        for (int i = 0; i < 3; ++i) if (stride[i] % 8) return cudaErrorInvalidValue;   // 16-byte cp.async rows
    return p->width == 256 ? launch<256>(*p, stream) : launch<512>(*p, stream);
}
