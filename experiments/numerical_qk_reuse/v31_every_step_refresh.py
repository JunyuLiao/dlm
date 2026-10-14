"""Fresh V31 mass selection followed by same-call native sparse FA4 output.

The full-support dense observer's output is discarded. There is no model
forward added, cached mass substitution, held-map bonus, or output reuse.
Previous maps below serve audit counters only and never influence selection.
"""
import torch


def forward(adapter, layer, q, buffers, scale, prefix, n):
    from . import v27_fa4
    adapter._mage_units_w = adapter._mage_held = adapter._k_override = None
    begin = end = None
    if adapter.value_audit and q.is_cuda:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
    _, mask = adapter._select_initial_refresh(layer, q, buffers, scale, prefix, n)
    pt, units = prefix // 64, q.shape[1] * ((n + 127) // 128)
    expected = min(adapter.mage_k // 64, pt)
    valid = (mask[..., :pt].sum(-1) == expected).all() & mask[..., pt:].all()
    if q.is_cuda:
        torch._assert_async(valid, 'every-step selector budget or tail protection failed')
    elif not bool(valid):
        raise AssertionError('every-step selector budget or tail protection failed')
    if end is not None:
        end.record()
        adapter._every_step_events.append((begin, end))
    adapter.calls['mage_selections'] += 1
    adapter.calls['every_step_dense_observations'] = adapter.calls.get('every_step_dense_observations', 0) + 1
    rectangles = units * ((prefix + n + 63) // 64)
    for key in ('every_step_observer_qk_rectangles', 'every_step_observer_pv_rectangles'):
        adapter.calls[key] = adapter.calls.get(key, 0) + rectangles
    adapter.calls['every_step_tail_qk_rectangles'] = adapter.calls.get('every_step_tail_qk_rectangles', 0) + units * (((prefix+n+63)//64)-pt)
    if adapter.value_audit:
        if adapter._every_step_counts is None:
            adapter._every_step_counts = q.new_zeros(3, dtype=torch.float64)
        prior = adapter._every_step_previous.get(layer)
        if prior is not None and prior.shape == mask.shape:
            new = mask[..., :pt]
            old = prior[..., :pt]
            additions = (new & ~old).sum()
            iou = (new & old).sum(-1) / (new | old).sum(-1).clamp_min(1)
            adapter._every_step_counts += torch.stack((additions, iou.sum(), iou.new_tensor(units))).double()
        adapter._every_step_previous[layer] = mask.detach()
    lists = v27_fa4.block_sparse_tensors(mask)
    return v27_fa4.sparse_lists(q, buffers['k'], buffers['v'], lists, scale)


def receipt(adapter):
    if not adapter.mage_always_refresh:
        return {}
    if adapter._every_step_events:
        torch.cuda.synchronize()
    swaps, iou_sum, units = (adapter._every_step_counts.tolist()
                            if adapter._every_step_counts is not None else (0., 0., 0.))
    return dict(mage_always_refresh=True, every_step_budget_errors=0,
        every_step_output='same_call_native_sparse_fa4', every_step_selection='current_full_support_v31_qblock_max',
        every_step_cross_step_state='audit_only; no held bonus, carry, cached scores or output',
        every_step_swaps=int(swaps), every_step_iou_previous=iou_sum / units if units else None,
        every_step_selector_audit_ms=sum(a.elapsed_time(b) for a, b in adapter._every_step_events))
