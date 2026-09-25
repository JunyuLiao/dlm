// SPDX-License-Identifier: Apache-2.0
// v11 PRESELECTED-SUPPORT current-QK/PV consumer (separate ABI; Junyu's
// value_direction ABI v4 and its fresh-T routing kernel are untouched).
//
// Contract: O = softmax(current transformed QK restricted to KEPT and LEGAL
// entries) @ current V, per query head, on the existing Q128 x KV64 decision
// grid. The support map (skipped, eligible) is READ-ONLY input; a tile is kept
// iff eligible && !skipped. A dropped tile issues no K load, no QK WGMMA, no
// V load and no PV WGMMA. No projected-V / sketch work exists in this kernel.
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace fmha::support_consumer {
inline constexpr uint32_t ABI_VERSION = 1;
inline constexpr int COUNTER_FIELDS = 4;  // k_tiles_loaded, qk_tiles, v_tiles_loaded, pv_tiles (per CTA)
struct Params {
    void const* q;                 // BF16 [B,H,Q,D] with element strides below; last dim contiguous
    void const* k;                 // BF16 [B,HK,K,D]
    void const* v;                 // BF16 [B,HK,K,D]
    int64_t q_stride[3];           // batch, head, position (elements)
    int64_t k_stride[3];
    int64_t v_stride[3];
    uint8_t const* skipped;        // [B,H,ceil(Q/128),ceil(K/64)] contiguous, read-only
    uint8_t const* eligible;       // same geometry, read-only
    uint64_t const* mask;          // optional packed legality [B,mask_heads,Q,ceil(K/64)]; null -> window rule
    void* output;                  // BF16, element strides below; last dim contiguous
    int64_t o_stride[3];           // batch, head, position
    float* log_normalizer;         // [B,H,Q] FP32 (retained log mass; -inf if none or invalid)
    uint8_t* invalid;              // [B,H,Q] rows with NaN/+inf legal retained scores (output zeroed)
    int32_t* counters;             // optional [B,H,gridX,COUNTER_FIELDS] diagnostic work counters
    int batch, heads, kv_heads, queries, keys, width, window, mask_heads;
    float scale;
};
}  // namespace fmha::support_consumer

extern "C" uint32_t support_consumer_abi_version();
extern "C" uint32_t support_consumer_params_size();
extern "C" cudaError_t support_consumer_sm90(fmha::support_consumer::Params const*, cudaStream_t);
