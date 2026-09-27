"""Untimed, independent Torch diagnostics for finer retained-support geometry.

This module does not change the production selector.  Every proposed geometry
reruns its sequential retained-state recurrence on the same cached scores;
the G0 closures instead share one row-wise K32 logical decision.  Returned
values contain scalars and hashes only.  Inputs and temporary support tensors
must remain private to the caller.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def _exact_matmul():
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _digest(value):
    array = value.detach().to(device='cpu', dtype=torch.uint8).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate(scores, projected, reference, sensitivity, q, k, v, legal, kind, threshold):
    if kind not in ('GLOBAL', 'LOCAL'):
        raise ValueError('kind must be GLOBAL or LOCAL')
    if not math.isfinite(float(threshold)) and float(threshold) != -math.inf:
        raise ValueError('threshold must be finite or -infinity')
    if scores.ndim != 4 or scores.dtype != torch.float32:
        raise ValueError('scores must be FP32 [B,H,Q,K]')
    b, h, nq, nk = scores.shape
    if not (b and h and nq and nk):
        raise ValueError('empty dimensions are unsupported')
    if projected.ndim != 4 or projected.shape[0] != b or projected.shape[2:] != (nk, 32):
        raise ValueError('projected must be [B,HK,K,32]')
    hk = projected.shape[1]
    if not hk or h % hk:
        raise ValueError('GQA requires H divisible by HK')
    if projected.dtype != torch.float32 or reference.shape != (b, hk) or reference.dtype != torch.float32:
        raise ValueError('projected/reference must be FP32 with matching GQA shape')
    if not bool(torch.isfinite(projected).all()):
        raise ValueError('projected must be finite')
    if not bool(torch.isfinite(reference).all()) or bool((reference <= 0).any()):
        raise ValueError('reference must be finite and positive')
    if sensitivity is None:
        sensitivity = torch.ones((b, nq), device=scores.device, dtype=torch.float32)
    if sensitivity.shape != (b, nq) or sensitivity.dtype != torch.float32:
        raise ValueError('sensitivity must be FP32 [B,Q]')
    if bool((sensitivity <= 0).any()) or not bool(torch.isfinite(sensitivity).all()):
        raise ValueError('sensitivity must be positive and finite')
    if not (q.ndim == k.ndim == v.ndim == 4 and q.shape[:3] == (b, h, nq)
            and k.shape[:3] == v.shape[:3] == (b, hk, nk)
            and q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]):
        raise ValueError('Q/K/V or GQA dimensions do not match scores')
    if any(x.device != scores.device for x in (projected, reference, sensitivity, q, k, v)):
        raise ValueError('all inputs must share a device')
    if legal is None:
        from dllm.attention.blasst.core import _attention_validity
        legal = _attention_validity(None, q, k, is_causal=False, sliding_window=None)
    if legal.dtype != torch.bool or legal.device != scores.device:
        raise ValueError('legal must be a bool mask on the input device')
    try:
        legal = torch.broadcast_to(legal, scores.shape)
    except RuntimeError as exc:
        raise ValueError('legal must broadcast to [B,H,Q,K]') from exc
    heads = 8 if kind == 'GLOBAL' else 2
    if h % heads or (h//hk) % heads:
        raise ValueError(f'{kind} {heads}-head geometry must stay within each KV head group')
    return sensitivity, legal


def _group_any(value, heads, queries, keys):
    """Physical closure of a [B,H,Q,K32] logical bitmap."""
    b, h, nq, kt = value.shape
    if h % heads or keys % 32:
        raise ValueError('invalid head/key grouping')
    kgroup = keys // 32
    padded = F.pad(value, (0, (-kt) % kgroup, 0, (-nq) % queries))
    qtiles = padded.shape[2] // queries
    ktiles = padded.shape[3] // kgroup
    grouped = padded.reshape(b, h//heads, heads, qtiles, queries, ktiles, kgroup)
    bits = grouped.any(dim=(2, 4, 6))
    expanded = bits.repeat_interleave(heads, 1).repeat_interleave(queries, 2)
    expanded = expanded.repeat_interleave(kgroup, 3)
    return bits, expanded[:, :, :nq, :kt]


def _group_rows(risk, active, heads, queries):
    b, h, nq = risk.shape
    if h % heads:
        raise ValueError('head group does not divide H')
    missing = (-nq) % queries
    rr = F.pad(risk, (0, missing), value=-math.inf)
    aa = F.pad(active, (0, missing), value=False)
    qtiles = rr.shape[-1] // queries
    rr = rr.reshape(b, h//heads, heads, qtiles, queries)
    aa = aa.reshape(b, h//heads, heads, qtiles, queries)
    return rr.amax(dim=(2, 4)), aa.any(dim=(2, 4))


def _select(scores, projected, reference, sensitivity, *, heads, queries, keys, threshold):
    """Rerun the Junyu online retained-state rule; no geometry reuses another's state."""
    b, h, nq, nk = scores.shape
    hk = projected.shape[1]
    if h % heads:
        raise ValueError('head geometry does not divide H')
    kv_for_head = torch.arange(h, device=scores.device) // (h//hk)
    previous = torch.full((b, h, nq), -math.inf, dtype=torch.float32, device=scores.device)
    state = torch.zeros((b, h, nq, 32), dtype=torch.float32, device=scores.device)
    ref = reference[:, kv_for_head, None].clamp_min(1e-12)
    log_sens = sensitivity.log()[:, None, :]
    retained, eligible, risk_maxima = [], [], []
    begin = time.perf_counter()
    for lo in range(0, nk, keys):
        hi = min(lo+keys, nk)
        block = scores[..., lo:hi]
        finite = torch.isfinite(block)
        bad = torch.isnan(block) | torch.isposinf(block)
        active = finite.any(-1)
        bad_row = bad.any(-1)
        clean = torch.where(finite, block, -math.inf)
        maximum = clean.amax(-1)
        safe_maximum = torch.where(active, maximum, 0.)
        weights = torch.exp(clean-safe_maximum[..., None])
        ell = weights.sum(-1)
        weights = weights / ell.clamp_min(1e-30)[..., None]
        block_z = torch.where(active, maximum + ell.clamp_min(1e-30).log(), -math.inf)
        sketch = projected[:, kv_for_head, lo:hi, :]
        mu = torch.matmul(weights, sketch)
        combined = torch.logaddexp(previous, block_z)
        safe = torch.where(torch.isfinite(combined), combined, 0.)
        alpha = torch.where(active, torch.exp(block_z-safe), 0.)
        delta = alpha[..., None] * (mu-state)
        norm = torch.linalg.vector_norm(delta, dim=-1)
        risk = (norm/ref).log() + log_sens
        risk = torch.where(active, torch.where(torch.isfinite(previous), risk, math.inf), -math.inf)
        risk = torch.where(bad_row, math.inf, risk)
        worst, any_active = _group_rows(risk, active | bad_row, heads, queries)
        keep_group = any_active & ~(worst < threshold)
        keep = keep_group.repeat_interleave(heads, 1).repeat_interleave(queries, 2)[:, :, :nq]
        update = keep & (active | bad_row)
        old = torch.exp(previous-safe)
        next_state = old[..., None]*state + alpha[..., None]*mu
        state = torch.where(update[..., None], next_state, state)
        previous = torch.where(update, combined, previous)
        retained.append(keep)
        eligible.append(any_active)
        risk_maxima.append(worst)
    if scores.is_cuda:
        torch.cuda.synchronize(scores.device)
    seconds = time.perf_counter()-begin
    bits = torch.stack(retained, dim=-1)
    risks = torch.stack(risk_maxima, -1)
    near_boundary, minimum_risk_distance = 0, None
    if math.isfinite(threshold):
        finite_risks = risks[torch.isfinite(risks)]
        if finite_risks.numel():
            distances = (finite_risks-threshold).abs()
            minimum_risk_distance = float(distances.amin())
            near_boundary = int((distances <= 1e-3).sum())
    return dict(bits=bits, eligible=torch.stack(eligible, -1),
                risk=risks,
                minimum_finite_risk_distance_to_threshold=minimum_risk_distance,
                risk_groups_within_1e_3_of_threshold=near_boundary,
                torch_reference_selection_seconds=seconds)


def _expand_tiles(bits, keys, nk):
    return bits.repeat_interleave(keys, -1)[..., :nk]


def _group_tile_bits(bits, heads, queries):
    b, h, nq, kt = bits.shape
    padded = F.pad(bits, (0, 0, 0, (-nq) % queries))
    return padded.reshape(b, h//heads, heads, padded.shape[2]//queries,
                          queries, kt).any(dim=(2, 4))


def _historical_mass_budget(scores, value_aware_bits, heads, queries, keys):
    """Mass-ranked, argmax-mandatory reference at a matched tile count.

    This is not old G75: the target count, observation and mandatory rule are
    different.  Normalization and argmax use historical cached scores only.
    """
    b, h, nq, nk = scores.shape
    invalid = torch.isnan(scores) | torch.isposinf(scores)
    if bool(invalid.any()):
        return None, dict(status='not_run', reason='invalid historical score prevents mass ranking',
                          invalid_score_elements=int(invalid.sum()))
    clean = torch.where(torch.isfinite(scores), scores, -math.inf)
    live = torch.isfinite(clean).any(-1)
    probability = clean.softmax(-1)
    probability = torch.where(torch.isfinite(probability), probability, 0.)
    kt = math.ceil(nk/keys)
    masses = F.pad(probability, (0, kt*keys-nk)).reshape(b, h, nq, kt, keys).sum(-1)
    # Mandatory support follows highest historical tile mass, not highest
    # individual token score. argmax resolves ties to the lowest tile index.
    first_argmax = masses.argmax(-1)
    active_tiles = F.pad(torch.isfinite(clean), (0, kt*keys-nk)).reshape(
        b, h, nq, kt, keys).any(-1)
    group_active = _group_tile_bits(active_tiles, heads, queries)
    qpad = (-nq) % queries
    masses = F.pad(masses, (0, 0, 0, qpad))
    group_mass = masses.reshape(b, h//heads, heads, (nq+qpad)//queries,
                                queries, kt).amax(dim=(2, 4))
    mandatory_row = torch.zeros((b, h, nq, kt), dtype=torch.bool, device=scores.device)
    mandatory_row.scatter_(-1, first_argmax[..., None], live[..., None])
    mandatory = _group_tile_bits(mandatory_row, heads, queries)
    target = _group_tile_bits(value_aware_bits, heads, queries).sum(-1)
    mandatory_count = mandatory.sum(-1)
    remaining = (target-mandatory_count).clamp_min(0)
    ranking = torch.argsort(group_mass, dim=-1, descending=True, stable=True)
    available = (~mandatory & group_active).gather(-1, ranking)
    ranked_fill = available & (available.cumsum(-1) <= remaining[..., None])
    chosen = mandatory.clone()
    chosen.scatter_(-1, ranking, ranked_fill | mandatory.gather(-1, ranking))
    expanded = chosen.repeat_interleave(heads, 1).repeat_interleave(queries, 2)[:, :, :nq]
    return expanded, dict(status='compared',
                          matched_value_aware_target_tiles=int(target.sum()),
                          selected_tiles=int(chosen.sum()),
                          eligible_historical_tiles=int(group_active.sum()),
                          mandatory_tiles=int(mandatory_count.sum()),
                          groups_mandatory_exceeded_budget=int((mandatory_count > target).sum()),
                          mandatory_rule='union of live-row lowest-coordinate historical tile-mass argmax',
                          tie_rule='stable lowest tile coordinate',
                          normalization='per historical row over finite scores',
                          interpretation='matched-count diagnostic; old G75 used a different budget and observation, while inherited value-aware mandatory is first support rather than argmax')


def _metrics(observed, reference):
    diff = (observed-reference).float()
    flat = diff.abs().flatten().cpu()
    den = float(torch.linalg.vector_norm(reference.float()))
    err = float(torch.linalg.vector_norm(diff))
    return dict(reference_l2=den, absolute_error_l2=err, relative_l2=err/(den or 1.),
                max_abs=float(flat.max()), p99_abs=float(torch.kthvalue(flat, max(1, math.ceil(flat.numel()*.99))).values),
                finite=bool(torch.isfinite(observed).all()))


def _scalar_distribution(value):
    flat = value.float().flatten().cpu()
    if not flat.numel():
        return dict(min=None, mean=None, p99=None, max=None)
    return dict(min=float(flat.min()), mean=float(flat.mean()),
                p99=float(torch.kthvalue(flat, max(1, math.ceil(flat.numel()*.99))).values),
                max=float(flat.max()))


def _row_output_error(observed, reference):
    difference = torch.linalg.vector_norm((observed-reference).float(), dim=-1)
    denominator = torch.linalg.vector_norm(reference.float(), dim=-1)
    return _scalar_distribution(difference/denominator.clamp_min(1e-12))


def _attention_inputs(q, k, v, scale):
    h, hk = q.shape[1], k.shape[1]
    keys = k.float().repeat_interleave(h//hk, dim=1)
    values = v.float().repeat_interleave(h//hk, dim=1)
    with _exact_matmul():
        score = torch.matmul(q.float(), keys.transpose(-1, -2))*float(scale)
    return score, values


def _attention(score, values, legal, support):
    selected = score.masked_fill(~(legal & support), -math.inf)
    p = selected.softmax(-1)
    p = torch.where(torch.isfinite(p), p, 0.)
    with _exact_matmul():
        out = p @ values
    return p, out


def _coarse_parity(scores, projected, reference, sensitivity, threshold, coarse):
    if not scores.is_cuda or scores.shape[-2] != 256 or projected.shape[-1] != 32:
        return dict(status='not_run', reason='requires actual CUDA Q256 selector geometry')
    from .cached_executor import route_only
    args = (scores.contiguous(), projected.contiguous(), reference.contiguous())
    routed = route_only(*args, sensitivity=sensitivity.contiguous(), log_threshold=threshold,
                        variant='generic')
    # A noneligible tile is neither skipped nor executed in the production bitmap.
    produced_keep = routed.eligible & ~routed.skipped
    torch_keep = coarse['bits'][:, :, ::128, :]
    mismatch = produced_keep ^ torch_keep
    eligible_mismatch = routed.eligible ^ coarse['eligible']
    count = int(mismatch.sum())
    return dict(status='match' if not count and not int(eligible_mismatch.sum()) else 'mismatch',
                kept_mismatch_tiles=count, eligible_mismatch_tiles=int(eligible_mismatch.sum()),
                native_keep_sha256=_digest(produced_keep), torch_keep_sha256=_digest(torch_keep),
                near_threshold_groups=coarse['risk_groups_within_1e_3_of_threshold'],
                minimum_finite_risk_distance=coarse['minimum_finite_risk_distance_to_threshold'])


@torch.inference_mode()
def compare_geometries(scores, projected, reference, sensitivity, q, k, v, *,
                       scale, threshold, kind, legal=None):
    """Compare inherited-threshold geometry on one frozen real input.

    The coarse route parity is exact. A mismatch is reported explicitly with
    its risk-boundary context and never reclassified as a passing tolerance.
    ``legal=None`` asks the native bidirectional validity helper for its mask;
    callers with another native mask must pass it explicitly.
    """
    sensitivity, legal = _validate(scores, projected, reference, sensitivity,
                                   q, k, v, legal, kind, threshold)
    # Explicit current legality is authoritative. Preserve NaN/+inf on legal
    # positions so the selector's bad-tile protection can still see them.
    masked_illegal_score_positions = int((~legal).sum())
    scores = scores.masked_fill(~legal, -math.inf).contiguous()
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('scale must be positive and finite')
    layouts = [('one_head_Q128_K64', 1, 128, 64),
               ('one_head_Q16_K64', 1, 16, 64)]
    if kind == 'GLOBAL':
        layouts.append(('GLOBAL_8head_Q2_K32', 8, 2, 32))
    else:
        layouts.append(('LOCAL_2head_Q8_K32', 2, 8, 32))
    selected = {}
    for name, hg, qg, kg in layouts:
        selected[name] = _select(scores, projected, reference, sensitivity,
                                 heads=hg, queries=qg, keys=kg, threshold=float(threshold))
    logical = _select(scores, projected, reference, sensitivity,
                      heads=1, queries=1, keys=32, threshold=float(threshold))
    g0_layouts = [('one_head_Q128_K64', 1, 128, 64),
                  ('one_head_Q16_K64', 1, 16, 64),
                  ('one_head_Q16_K32', 1, 16, 32)]
    g0_layouts.append(('GLOBAL_8head_Q2_K32', 8, 2, 32) if kind == 'GLOBAL'
                      else ('LOCAL_2head_Q8_K32', 2, 8, 32))
    current_score, current_values = _attention_inputs(q, k, v, scale)
    full, full_out = _attention(current_score, current_values, legal,
                                torch.ones_like(legal))
    denominator = int(legal.sum())
    results = {}
    mass_reference = {}
    for name, _, _, kg in layouts:
        item = selected[name]
        support = _expand_tiles(item['bits'], kg, scores.shape[-1]) & legal
        _, out = _attention(current_score, current_values, legal, support)
        removed = full.masked_fill(support, 0.).sum(-1)
        results[name] = dict(support_sha256=_digest(support),
                             kept_legal_pairs=int(support.sum()),
                             eligible_legal_pairs_denominator=denominator,
                             kept_legal_pair_fraction=float(support.sum())/denominator if denominator else None,
                             legal_pairs_per_row=_scalar_distribution(support.sum(-1)),
                             removed_mass_per_row=_scalar_distribution(removed),
                             output_vs_full_fp32=_metrics(out, full_out),
                             output_relative_l2_per_row=_row_output_error(out, full_out),
                             torch_reference_selection_seconds=item['torch_reference_selection_seconds'],
                             risk_groups_within_1e_3_of_threshold=item['risk_groups_within_1e_3_of_threshold'],
                             minimum_finite_risk_distance_to_threshold=item['minimum_finite_risk_distance_to_threshold'])
        _, hg, qg, _ = next(layout for layout in layouts if layout[0] == name)
        mass_bits, mass_meta = _historical_mass_budget(scores, item['bits'], hg, qg, kg)
        if mass_bits is not None:
            mass_support = _expand_tiles(mass_bits, kg, scores.shape[-1]) & legal
            _, mass_out = _attention(current_score, current_values, legal, mass_support)
            mass_removed = full.masked_fill(mass_support, 0.).sum(-1)
            mass_meta.update(support_sha256=_digest(mass_support),
                             kept_legal_pairs=int(mass_support.sum()),
                             legal_pairs_added_vs_value_aware=int((mass_support & ~support).sum()),
                             legal_pairs_removed_vs_value_aware=int((support & ~mass_support).sum()),
                             legal_pair_symmetric_difference=int((support ^ mass_support).sum()),
                             removed_mass_per_row=_scalar_distribution(mass_removed),
                             output_vs_full_fp32=_metrics(mass_out, full_out),
                             output_relative_l2_per_row=_row_output_error(mass_out, full_out))
        mass_reference[name] = mass_meta
    logical_support = _expand_tiles(logical['bits'], 32, scores.shape[-1]) & legal
    g0 = {}
    for name, hg, qg, kg in g0_layouts:
        _, closed32 = _group_any(logical['bits'], hg, qg, kg)
        support = _expand_tiles(closed32, 32, scores.shape[-1]) & legal
        _, out = _attention(current_score, current_values, legal, support)
        removed = full.masked_fill(support, 0.).sum(-1)
        g0[name] = dict(support_sha256=_digest(support),
                        kept_legal_pairs=int(support.sum()),
                        legal_extra_pairs_over_logical=int((support & ~logical_support).sum()),
                        legal_pairs_per_row=_scalar_distribution(support.sum(-1)),
                        removed_mass_per_row=_scalar_distribution(removed),
                        output_vs_full_fp32=_metrics(out, full_out),
                        output_relative_l2_per_row=_row_output_error(out, full_out))
    parity = _coarse_parity(scores, projected, reference, sensitivity, float(threshold),
                            selected['one_head_Q128_K64'])
    return dict(schema='v21b_geometry_diagnostic_001', kind=kind, threshold=float(threshold),
                shape=dict(batch=scores.shape[0], query_heads=scores.shape[1],
                           kv_heads=projected.shape[1], queries=scores.shape[2],
                           keys=scores.shape[3], head_dim=q.shape[-1]),
                denominator_legal_pairs=denominator,
                masked_illegal_score_positions=masked_illegal_score_positions,
                storage_bytes=dict(cached_scores=scores.numel()*scores.element_size(),
                                   projected_v=projected.numel()*projected.element_size(),
                                   per_row_retained_state=scores.shape[0]*scores.shape[1]*scores.shape[2]*(1+32)*4,
                                   bitmap_by_geometry={name: scores.shape[0]*(scores.shape[1]//hg)*
                                   math.ceil(scores.shape[2]/qg)*math.ceil(scores.shape[3]/kg)
                                   for name, hg, qg, kg in layouts}),
                full_fp32_reference_l2=float(torch.linalg.vector_norm(full_out)),
                coarse_route_only_parity=parity,
                independent_sequential=results,
                historical_mass_argmax_matched_tile_budget=mass_reference,
                g0_common_logical=dict(rowwise_K32_support_sha256=_digest(logical_support),
                                       rowwise_kept_legal_pairs=int(logical_support.sum()),
                                       torch_reference_selection_seconds=logical['torch_reference_selection_seconds'],
                                       closures=g0),
                projected_v_setup_seconds=None,
                timing_note='Torch reference selection and full FP32 attention are diagnostic, not production cost; projected-V setup is external and must be reported separately. Output error uses full FP32 QK, softmax, and PV and does not claim actual BF16-P production error.')


@torch.inference_mode()
def matched_q16_screen(scores, projected, reference, sensitivity, q, k, v, *,
                       scale, threshold, kind, legal=None):
    """Fixed five-point GLOBAL Q16 work/error screen on one frozen input.

    Cross-host selection is intentionally external: the caller combines the
    two predeclared LongBench call-3 receipts once, without quality answers.
    Every candidate reruns the Q16 retained-state recurrence; none is a mask
    split from the coarse selector.  This is diagnostic FP32 math, not a
    production-kernel timing or BF16-P error claim.
    """
    if kind != 'GLOBAL':
        raise ValueError('matched Q16 screen is frozen to GLOBAL geometry')
    sensitivity, legal = _validate(scores, projected, reference, sensitivity,
                                   q, k, v, legal, kind, threshold)
    if not math.isfinite(float(threshold)) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('matched screen needs finite inherited threshold and positive scale')
    masked_illegal = int((~legal).sum())
    scores = scores.masked_fill(~legal, -math.inf).contiguous()
    b, h, nq, nk = scores.shape
    kt = math.ceil(nk/64)
    coarse = _select(scores, projected, reference, sensitivity,
                     heads=1, queries=128, keys=64, threshold=float(threshold))
    current_score, current_values = _attention_inputs(q, k, v, scale)
    full, full_out = _attention(current_score, current_values, legal,
                                torch.ones_like(legal))

    def receipt(item, bitmap_bytes):
        support = _expand_tiles(item['bits'], 64, nk) & legal
        _, out = _attention(current_score, current_values, legal, support)
        removed = full.masked_fill(support, 0.).sum(-1)
        return dict(support_sha256=_digest(support),
                    retained_legal_pairs=int(support.sum()),
                    bitmap_bytes=bitmap_bytes,
                    removed_mass_per_row=_scalar_distribution(removed),
                    output_vs_full_fp32=_metrics(out, full_out),
                    output_relative_l2_per_row=_row_output_error(out, full_out),
                    torch_reference_selection_seconds=item['torch_reference_selection_seconds'],
                    risk_groups_within_1e_3_of_threshold=item['risk_groups_within_1e_3_of_threshold'],
                    minimum_finite_risk_distance_to_threshold=item['minimum_finite_risk_distance_to_threshold'])

    baseline = receipt(coarse, b*h*math.ceil(nq/128)*kt)
    offsets = (0., -.25, -.5, -1., -2.)
    candidates = []
    previous_pairs = None
    nonmonotonic = []
    for offset in offsets:
        value = float(threshold) + offset
        selected = _select(scores, projected, reference, sensitivity,
                           heads=1, queries=16, keys=64, threshold=value)
        measured = receipt(selected, b*h*math.ceil(nq/16)*kt)
        pairs = measured['retained_legal_pairs']
        if previous_pairs is not None and pairs < previous_pairs:
            nonmonotonic.append(dict(previous_offset=candidates[-1]['offset'],
                                     current_offset=offset,
                                     previous_pairs=previous_pairs, current_pairs=pairs))
        candidates.append(dict(offset=offset, threshold=value,
                               retained_pair_ratio_to_coarse=(pairs/baseline['retained_legal_pairs']
                                                              if baseline['retained_legal_pairs'] else None),
                               error_ratio_to_coarse=(measured['output_vs_full_fp32']['relative_l2']/
                                                      baseline['output_vs_full_fp32']['relative_l2']
                                                      if baseline['output_vs_full_fp32']['relative_l2'] else None),
                               **measured))
        previous_pairs = pairs
    return dict(schema='v21b_matched_q16_screen_001', kind='GLOBAL',
                inherited_threshold=float(threshold), offsets=list(offsets),
                shape=dict(batch=b, query_heads=h, kv_heads=projected.shape[1],
                           queries=nq, keys=nk, head_dim=q.shape[-1]),
                masked_illegal_score_positions=masked_illegal,
                denominator_legal_pairs=int(legal.sum()),
                full_fp32_reference_l2=float(torch.linalg.vector_norm(full_out)),
                coarse_baseline=baseline, q16_candidates=candidates,
                nonmonotonic_retained_work_samples=nonmonotonic,
                selection_contract=('Cross-host root combines exactly the two frozen LongBench call-3 '
                                    'GLOBAL states. Choose one fixed threshold by aggregate legal-work '
                                    'distance, report mismatch; separately report best retained work with '
                                    'aggregate error <=1.05x coarse and each state <=1.15x coarse. '
                                    'No answer or quality labels are used.'),
                timing_note=('Torch selection and full FP32 attention are diagnostic only; '
                             'projected-V setup and production BF16-P error are not measured here.'))
