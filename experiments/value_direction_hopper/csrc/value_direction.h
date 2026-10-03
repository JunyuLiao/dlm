// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace fmha::value_direction {
inline constexpr uint32_t ABI_VERSION=4;
inline constexpr int TIMING_FIELDS=6;
// Versioned, allocation-free ABI. All pointers refer to the caller's device and
// stream. B,H,N,D contiguous; native KV heads; 128Q x 64KV physical decisions.
struct Params {
    void const* q;
    void const* k;
    void const* v;
    float const* z;
    float const* reference;
    void const* mask;
    void* output;
    uint8_t* skipped;
    uint8_t* eligible;
    float* log_normalizer;
    float* projected_state;
    float* risks;
    int batch, heads, kv_heads, queries, keys, width;
    int mask_kind, mask_heads, mode, trace;
    float scale, log_threshold;
    uint64_t* timings;
    int projected_precision; // 0: IEEE; 1: staged TF32x3; 2: register TF32x3; 3: shared-P TF32x3 control
    float* debug_scores; // optional diagnostic-only full logits, never needed for routing
    float const* sensitivity; // optional B,Q FP32 row weights; null preserves original routing
};
}  // namespace fmha::value_direction

extern "C" cudaError_t value_direction_sm90(
    fmha::value_direction::Params const*, cudaStream_t, int overlap);
extern "C" uint32_t value_direction_abi_version();
extern "C" uint32_t value_direction_params_size();
extern "C" cudaError_t value_direction_sm90_tma(
    fmha::value_direction::Params const*,cudaStream_t,int overlap);
extern "C" cudaError_t value_direction_pack_mask(
    uint8_t const*,uint64_t*,int batch,int mask_heads,int queries,int keys,cudaStream_t);
