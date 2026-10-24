"""Protocol-pure helpers for native pulse/hold diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


SCHEDULES = {
    'AA': ('A', 'A'),
    'SA': ('S', 'A'),
    'AS': ('A', 'S'),
    'SS': ('S', 'S'),
}
SCORE_NAMES = ('current_error', 'response', 'decision', 'sham')


def first_canvas_anchors(terminal_calls: int) -> tuple[int, ...]:
    """Completed denoise calls at reference-relative boundaries; no fallback."""
    if terminal_calls < 1:
        raise ValueError('terminal_calls must be positive')
    candidates = (max(2, terminal_calls // 2), max(2, terminal_calls - 2))
    return tuple(sorted({x for x in candidates if x < terminal_calls}))


def native_argmax_tie_summary(scaled_logits: torch.Tensor,
                              native_winners: torch.Tensor) -> dict:
    """Compare torch's native first-index argmax to sampled native state."""
    flat = scaled_logits.reshape(-1, scaled_logits.shape[-1])
    winners = flat.argmax(dim=-1)
    top = flat.topk(2, dim=-1).values
    tied = top[:, 0] == top[:, 1]
    native = native_winners.reshape(-1).to(device=winners.device)
    mismatch = winners != native
    return dict(positions=flat.shape[0], tie_positions=int(tied.sum().item()),
        native_winner_mismatches=int(mismatch.sum().item()),
        native_winner_mismatch_ties=int((mismatch & tied).sum().item()),
        native_winner_equal=bool(not mismatch.any().item()))


def acceptance_reconstructed_on_device(scaled_logits: torch.Tensor,
                                       entropy_bound: float) -> tuple[torch.Tensor, dict]:
    """Same-device reconstruction; explicitly not a captured native mask."""
    logits = scaled_logits.reshape(-1, scaled_logits.shape[-1]).float()
    logp = logits.log_softmax(dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(-1)
    ordered, indices = entropy.sort()
    cumulative = ordered.cumsum(0)
    prior_max = ordered.cummax(0).values
    slack = entropy_bound - (cumulative - prior_max)
    sorted_keep = slack >= 0
    accepted = torch.zeros_like(sorted_keep)
    accepted.scatter_(0, indices, sorted_keep)
    threshold_distance = slack.abs()
    return accepted.reshape(scaled_logits.shape[:-1]), dict(
        measurement='reconstructed_on_device_from_native_returned_scaled_logits',
        exact_native_acceptance_capture=False,
        threshold_tie_positions=int((slack == 0).sum().item()),
        threshold_near_positions=int((threshold_distance <= 1e-7).sum().item()),
        minimum_threshold_distance=float(threshold_distance.min().item()),
        acceptance_count=int(accepted.sum().item()), positions=int(accepted.numel()))


def reference_winner_margin(reference_scaled: torch.Tensor,
                            candidate_scaled: torch.Tensor) -> dict:
    """Signed candidate margin for the fixed reference winner at every position."""
    ref = reference_scaled.reshape(-1, reference_scaled.shape[-1]).float()
    cur = candidate_scaled.reshape(-1, candidate_scaled.shape[-1]).float()
    if ref.shape != cur.shape:
        raise ValueError('reference and candidate logits must align')
    ref_winner = ref.argmax(dim=-1)
    winner_score = cur.gather(1, ref_winner[:, None]).squeeze(1)
    alternatives = cur.clone()
    alternatives.scatter_(1, ref_winner[:, None], -torch.inf)
    margin = winner_score - alternatives.max(dim=-1).values
    ref_top = ref.topk(2, dim=-1).values
    ref_margin = ref_top[:, 0] - ref_top[:, 1]
    erosion = ref_margin - margin
    return dict(reference_winner_boundary_crossings=int((margin < 0).sum().item()),
        reference_winner_ties=int((margin == 0).sum().item()),
        signed_margin_change_per_position=(margin-ref_margin).tolist(),
        margin_mean=float(margin.mean().item()),
        margin_abs_change_mean=float((margin-ref_margin).abs().mean().item()),
        margin_erosion_worst=float(erosion.max().item()),
        margin_erosion_p01=float(torch.quantile(erosion, .01).item()),
        margin_erosion_p10=float(torch.quantile(erosion, .10).item()),
        candidate_margin_p01=float(torch.quantile(margin, .01).item()),
        candidate_margin_p10=float(torch.quantile(margin, .10).item()),
        reference_margin_p01=float(torch.quantile(ref_margin, .01).item()),
        reference_margin_p10=float(torch.quantile(ref_margin, .10).item()))


def centered_distribution_metrics(reference_scaled: torch.Tensor,
                                  candidate_scaled: torch.Tensor) -> dict:
    """Compare distributions and row-centered scores, separating common shifts."""
    ref = reference_scaled.reshape(-1, reference_scaled.shape[-1]).float()
    cur = candidate_scaled.reshape(-1, candidate_scaled.shape[-1]).float()
    if ref.shape != cur.shape:
        raise ValueError('reference and candidate logits must align')
    delta = cur-ref
    centered = delta-delta.mean(dim=-1, keepdim=True)
    lp = ref.log_softmax(-1)
    lq = cur.log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    midpoint = .5*(p+q)
    lm = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    kl_pq = (p*(lp-lq)).sum(-1)
    kl_qp = (q*(lq-lp)).sum(-1)
    js = .5*((p*(lp-lm)).sum(-1)+(q*(lq-lm)).sum(-1))
    return dict(raw_logit_l2=float(delta.norm().item()),
        centered_logit_l2=float(centered.norm().item()),
        centered_logit_l2_mean_row=float(centered.norm(dim=-1).mean().item()),
        kl_reference_to_candidate_mean=float(kl_pq.mean().item()),
        kl_candidate_to_reference_mean=float(kl_qp.mean().item()),
        js_mean=float(js.mean().item()))


def pulse_hold_contrasts(outcomes: Mapping[str, Mapping[str, float]],
                         endpoint: str) -> dict:
    """Scalar factorial contrasts; caller supplies matching aligned endpoints."""
    missing = set(SCHEDULES)-set(outcomes)
    if missing:
        return dict(available=False, missing=sorted(missing))
    aa, sa, ass, ss = [float(outcomes[name][endpoint]) for name in ('AA','SA','AS','SS')]
    return dict(available=True, endpoint=endpoint,
        propagated_pulse=sa-aa,
        direct_second_pulse=ass-aa,
        additional_held_application=ss-sa,
        interaction=ss-sa-ass+aa)


def build_short_single_deletion_bank(unit_blocks: Sequence[int], max_candidates: int,
                                     seed: int) -> list[dict]:
    """Enumerate exact one-rectangle deletion supports; hash-stratify if bounded."""
    if len(unit_blocks) != 4 or any(x < 1 for x in unit_blocks):
        raise ValueError('four routing units with at least one eligible block required')
    options = [(u, b) for u, count in enumerate(unit_blocks) for b in range(count)]
    if len(options) <= max_candidates:
        selected = options
        policy = 'complete_lexicographic_enumeration'
    else:
        # Stable integer hash rank; every unit receives a place before remaining slots.
        rank = lambda pair: ((pair[0]+1)*0x9E3779B1 ^ (pair[1]+1)*0x85EBCA77 ^ seed) & 0xffffffff
        selected = []
        for unit in range(4):
            candidates = [x for x in options if x[0] == unit]
            selected.append(min(candidates, key=rank))
        selected.extend(sorted((x for x in options if x not in selected), key=rank)[:max_candidates-4])
        policy = 'unit_stratified_hash_rank'
    rows = []
    for ordinal, (unit, block) in enumerate(selected):
        kept = [[True]*count for count in unit_blocks]
        kept[unit][block] = False
        rows.append(dict(candidate_index=ordinal, deleted_unit=unit, deleted_block=block,
            support=kept, physical_deletions=1, construction=policy))
    return rows


def factorial_schedule(candidate: str, schedule: str) -> tuple[str, ...]:
    if schedule not in SCHEDULES:
        raise ValueError(schedule)
    if schedule == 'AA':
        return SCHEDULES[schedule]
    if not candidate:
        raise ValueError('candidate support identity is required')
    return tuple(candidate if arm == 'S' else 'A' for arm in SCHEDULES[schedule])


def select_finalists(candidates: Sequence[Mapping], decision_first_pulse: Mapping[str, Mapping],
                     minimum_feasible: int = 4) -> dict:
    """Freeze support hashes by score using one common feasible candidate set."""
    feasible = [x for x in candidates if x['feasible']]
    if len(feasible) < minimum_feasible:
        return dict(identifying=False, feasible_count=len(feasible), abstain=True,
            selections={name: None for name in ('current_error','response','decision','mass','sham')})
    by_hash = {x['support_hash']: x for x in feasible}
    choose = {}
    choose['current_error'] = min(feasible, key=lambda x: (x['current_error'], x['support_hash']))
    choose['response'] = min(feasible, key=lambda x: (
        float('inf') if x['response_score'] is None else x['response_score'], x['support_hash']))
    choose['mass'] = min(feasible, key=lambda x: (x['whole_deleted_mass'], x['support_hash']))
    decision_rows = [(by_hash[h], score) for h, score in decision_first_pulse.items() if h in by_hash]
    if decision_rows:
        choose['decision'] = min(decision_rows, key=lambda item: (
            tuple(item[1]['score_tuple']), item[0]['support_hash']))[0]
    else:
        choose['decision'] = None
    # Deterministic support-hash sham from the identical feasible set.
    choose['sham'] = min(feasible, key=lambda x: x['support_hash'])
    return dict(identifying=True, feasible_count=len(feasible), abstain=False,
        selections={name: None if row is None else row['support_hash']
                    for name,row in choose.items()},
        distinct_selected_hashes=sorted({row['support_hash'] for row in choose.values() if row is not None}),
        score_firewall='All hashes selected before second-call, stopping, or task outcomes are read')


def next_branch_action(schedule: str, intervention_call_index: int,
                       first_canvas_committed: bool) -> str:
    """Return the native first-canvas factorial action; commit is absorbing."""
    if schedule not in SCHEDULES:
        raise ValueError(schedule)
    if first_canvas_committed:
        return 'NATIVE_DENSE'
    pair = SCHEDULES[schedule]
    if intervention_call_index < 0:
        raise ValueError('call index must be nonnegative')
    if intervention_call_index < 2:
        return pair[intervention_call_index]
    return 'A'


@dataclass
class WorkEstimate:
    name: str
    model_forwards: int
    allocated_seconds: int
    notes: str


def budget_check(estimates: Sequence[WorkEstimate], max_forwards: int,
                 max_seconds: int) -> dict:
    forwards = sum(x.model_forwards for x in estimates)
    seconds = sum(x.allocated_seconds for x in estimates)
    return dict(forwards=forwards, seconds=seconds,
        within_forwards=forwards <= max_forwards,
        within_seconds=seconds <= max_seconds,
        components=[x.__dict__ for x in estimates])
