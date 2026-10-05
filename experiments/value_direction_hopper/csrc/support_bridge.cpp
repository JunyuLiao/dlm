// SPDX-License-Identifier: Apache-2.0
// ATen operator for the v11 preselected-support consumer. Disjoint namespace
// (vd_support_v1) from Junyu's value_direction_hopper bridge, so both can be
// loaded in one process. No host reads, synchronization or compilation.
#include "support_consumer.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {
using at::Tensor;

void check_qkv(Tensor const& x, char const* name, Tensor const& q) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.scalar_type() == at::kBFloat16 && x.device() == q.device(),
                name, " must be a CUDA BF16 [B,heads,len,D] tensor on q's device");
    TORCH_CHECK(x.stride(3) == 1 && x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 && x.stride(2) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                name, " needs a contiguous head dimension, 16-byte aligned rows and base");
}

// layout 0: output [B,H,Q,D] contiguous. layout 1: output [B,Q,H,D] contiguous
// (the model's post-attention layout), returned as a [B,H,Q,D] strided view.
std::vector<Tensor> attention(Tensor const& q, Tensor const& k, Tensor const& v, Tensor const& skipped,
                              Tensor const& eligible, c10::optional<Tensor> const& mask, int64_t window,
                              double scale, int64_t layout, bool counters) {
    using fmha::support_consumer::Params;
    TORCH_CHECK(support_consumer_abi_version() == fmha::support_consumer::ABI_VERSION &&
                support_consumer_params_size() == sizeof(Params), "support consumer ABI mismatch");
    check_qkv(q, "q", q); check_qkv(k, "k", q); check_qkv(v, "v", q);
    auto b = q.size(0), h = q.size(1), nq = q.size(2), d = q.size(3), hk = k.size(1), nk = k.size(2);
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == b && k.size(3) == d && h % hk == 0 && (d == 256 || d == 512) &&
                nq > 0 && nk > 0, "unsupported dimensions/GQA");
    auto qb = (nq + 127) / 128, kt = (nk + 63) / 64;
    for (auto const* m : {&skipped, &eligible})
        TORCH_CHECK(m->is_cuda() && m->device() == q.device() && m->scalar_type() == at::kBool && m->is_contiguous() &&
                    m->sizes() == at::IntArrayRef({b, h, qb, kt}), "support maps must be contiguous CUDA bool [B,H,Qtiles,Ktiles]");
    int64_t mask_heads = 1;
    if (mask.has_value()) {
        auto const& m = *mask;
        TORCH_CHECK(m.is_cuda() && m.device() == q.device() && m.scalar_type() == at::kUInt64 && m.is_contiguous() &&
                    m.dim() == 4 && m.size(0) == b && (m.size(1) == 1 || m.size(1) == h) && m.size(2) == nq && m.size(3) == kt,
                    "packed mask must be contiguous CUDA uint64 [B,1|H,Q,Ktiles]");
        mask_heads = m.size(1);
    }
    TORCH_CHECK(window >= 0 && (layout == 0 || layout == 1), "invalid window/layout");
    c10::cuda::CUDAGuard guard(q.device());
    auto opts = q.options();
    Tensor out = layout == 0 ? at::empty({b, h, nq, d}, opts) : at::empty({b, nq, h, d}, opts).transpose(1, 2);
    auto lse = at::empty({b, h, nq}, opts.dtype(at::kFloat));
    auto invalid = at::empty({b, h, nq}, opts.dtype(at::kBool));
    auto count = counters ? at::zeros({b, h, qb * 2, fmha::support_consumer::COUNTER_FIELDS}, opts.dtype(at::kInt))
                          : at::empty({1}, opts.dtype(at::kInt));
    Params p{};
    p.q = q.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr();
    for (int i = 0; i < 3; ++i) { p.q_stride[i] = q.stride(i); p.k_stride[i] = k.stride(i); p.v_stride[i] = v.stride(i); p.o_stride[i] = out.stride(i); }
    p.skipped = reinterpret_cast<uint8_t const*>(skipped.data_ptr());
    p.eligible = reinterpret_cast<uint8_t const*>(eligible.data_ptr());
    p.mask = mask.has_value() ? reinterpret_cast<uint64_t const*>(mask->data_ptr()) : nullptr;
    p.output = out.data_ptr(); p.log_normalizer = lse.data_ptr<float>();
    p.invalid = reinterpret_cast<uint8_t*>(invalid.data_ptr());
    p.counters = counters ? count.data_ptr<int32_t>() : nullptr;
    p.batch = b; p.heads = h; p.kv_heads = hk; p.queries = nq; p.keys = nk; p.width = d;
    p.window = window; p.mask_heads = mask_heads; p.scale = scale;
    TORCH_CHECK(out.stride(3) == 1, "output head dimension must be contiguous");
    C10_CUDA_CHECK(support_consumer_sm90(&p, c10::cuda::getCurrentCUDAStream(q.get_device()).stream()));
    return {out, lse, invalid, count};
}
}  // namespace

TORCH_LIBRARY(vd_support_v1, m) {
    m.def("attention(Tensor q, Tensor k, Tensor v, Tensor skipped, Tensor eligible, Tensor? mask, int window, "
          "float scale, int layout, bool counters) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(vd_support_v1, CUDA, m) { m.impl("attention", attention); }
