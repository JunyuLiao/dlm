"""CPU-only exact-equality fast path; native GPU replay is unchanged."""

def compare_fast(a,b,mask=None):
    from scripts.v31_refresh_operator_replay import digest
    if mask is not None:a,b=a[mask],b[mask]
    if a.shape==b.shape and a.dtype==b.dtype and digest(a)==digest(b):
        return dict(exact=True,unequal_fraction=0.,max_abs=0.,rms=0.,relative_l2=0.,
                    nonfinite_pattern_equal=True,elements=a.numel())
    return _original_compare(a,b)


def outputs_fast(tensors):
    import torch
    result={k:v.detach().cpu().clone() for k,v in tensors.items() if k in ('out','out_partial','lse_partial','lse','aux0')}
    if 'out_partial' in result:
        finite=torch.isfinite(result['lse_partial'])
        mask=finite.transpose(-1,-2).unsqueeze(-1).expand_as(result['out_partial'])
        result['out_partial_active']=result['out_partial'].reshape(-1) if bool(finite.all()) else result['out_partial'][mask]
        result['partial_active_mask']=mask
        del result['out_partial']
    return result


_original_compare=None


def install(module):
    global _original_compare
    _original_compare=module.compare
    module.compare=compare_fast
    module.outputs=outputs_fast
