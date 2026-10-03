// SPDX-License-Identifier: Apache-2.0
// Inference-only ATen operator. No pybind tensor unpacking, host votes, implicit
// synchronization, compilation or cache ownership inside the attention call.
#include "value_direction.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {
using at::Tensor;
std::vector<Tensor> attention(Tensor const& q,Tensor const& k,Tensor const& v,Tensor const& z,Tensor const& ref,
    Tensor const& mask,int64_t mask_kind,double scale,double log_threshold,int64_t mode,int64_t precision,bool trace,bool tma,int64_t schedule,
    c10::optional<Tensor> const& sensitivity) {
    using fmha::value_direction::Params;
    TORCH_CHECK(value_direction_abi_version()==fmha::value_direction::ABI_VERSION && value_direction_params_size()==sizeof(Params),"Kernel ABI mismatch");
    TORCH_CHECK(q.is_cuda() && q.dim()==4 && k.dim()==4 && v.sizes()==k.sizes(),"Expected CUDA B,H,N,D Q/K/V");
    auto b=q.size(0),h=q.size(1),nq=q.size(2),d=q.size(3),hk=k.size(1),nk=k.size(2);
    TORCH_CHECK(b>0 && h>0 && hk>0 && nq>0 && nk>0 && h%hk==0 && k.size(0)==b && k.size(3)==d && (d==256||d==512),"Unsupported dimensions/GQA");
    TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16,"BF16 QKV required");
    TORCH_CHECK(z.sizes()==at::IntArrayRef({b,hk,nk,32}) && z.scalar_type()==at::kFloat && ref.sizes()==at::IntArrayRef({b,hk}) && ref.scalar_type()==at::kFloat,"FP32 native-head Gaussian32/ref required");
    for(auto const* x:{&q,&k,&v,&z,&ref})TORCH_CHECK(x->device()==q.device() && x->is_contiguous(),"Contiguous tensors on one CUDA device required");
    if(sensitivity.has_value())TORCH_CHECK(sensitivity->sizes()==at::IntArrayRef({b,nq}) &&
        sensitivity->scalar_type()==at::kFloat && sensitivity->device()==q.device() &&
        sensitivity->is_contiguous() && mode==1,"FP32 B,Q sensitivity requires value routing");
    int mh=1;
    TORCH_CHECK(mask_kind>=0 && mask_kind<=3,"Invalid mask kind");
    if(mask_kind) {
        TORCH_CHECK(mask.dim()==4 && mask.device()==q.device() && mask.is_contiguous() && mask.size(0)==b &&
            (mask.size(1)==1||mask.size(1)==h) && mask.size(2)==nq && mask.size(3)==(mask_kind==3?(nk+63)/64:nk),"Invalid structural mask");
        TORCH_CHECK(mask.scalar_type()==(mask_kind==1?at::kBool:mask_kind==2?at::kBFloat16:at::kUInt64),"Invalid mask dtype");
        mh=mask.size(1);
    }
    c10::cuda::CUDAGuard guard(q.device());
    auto out=at::empty_like(q),skipped=at::empty({b,h,(nq+127)/128,(nk+63)/64},q.options().dtype(at::kBool));
    auto eligible=at::empty_like(skipped),lse=at::empty({b,h,nq},q.options().dtype(at::kFloat));
    auto state=trace?at::empty({b,h,nq,32},q.options().dtype(at::kFloat)):at::empty({1},q.options().dtype(at::kFloat));
    auto risk=trace?at::empty(skipped.sizes(),q.options().dtype(at::kFloat)):at::empty({1},q.options().dtype(at::kFloat));
    Params p{};
    p.q=q.data_ptr();p.k=k.data_ptr();p.v=v.data_ptr();p.z=z.data_ptr<float>();p.reference=ref.data_ptr<float>();p.mask=mask_kind?mask.data_ptr():nullptr;
    p.output=out.data_ptr();p.skipped=reinterpret_cast<uint8_t*>(skipped.data_ptr());p.eligible=reinterpret_cast<uint8_t*>(eligible.data_ptr());
    p.log_normalizer=lse.data_ptr<float>();p.projected_state=state.data_ptr<float>();p.risks=risk.data_ptr<float>();
    p.batch=b;p.heads=h;p.kv_heads=hk;p.queries=nq;p.keys=nk;p.width=d;p.mask_kind=mask_kind;p.mask_heads=mh;
    p.mode=mode;p.trace=trace;p.scale=scale;p.log_threshold=log_threshold;p.projected_precision=precision;
    p.sensitivity=sensitivity.has_value()?sensitivity->data_ptr<float>():nullptr;
    auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    C10_CUDA_CHECK(tma?value_direction_sm90_tma(&p,stream,schedule):value_direction_sm90(&p,stream,schedule));
    return {out,skipped,eligible,lse,state,risk};
}
}

TORCH_LIBRARY(value_direction_hopper,m) {
    m.def("attention(Tensor q, Tensor k, Tensor v, Tensor z, Tensor reference, Tensor mask, int mask_kind, float scale, float log_threshold, int mode, int precision, bool trace, bool tma, int schedule, Tensor? sensitivity=None) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(value_direction_hopper,CUDA,m) {m.impl("attention",attention);}
