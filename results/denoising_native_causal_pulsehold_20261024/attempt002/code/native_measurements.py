"""Sampler-state capture and branch contrasts; no RNG use or state writes."""
from __future__ import annotations

import hashlib

import torch

from pulsehold import (acceptance_reconstructed_on_device,
                       centered_distribution_metrics, native_argmax_tie_summary,
                       reference_winner_margin)


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def capture_native_sample(native_step, *args, capture: bool = True, pre_commit=None, **kwargs):
    """Call the original compiled helper once, then snapshot its actual state."""
    before_commit = bool(args[8][args[1]].item()) if pre_commit is None else bool(pre_commit)
    result = native_step(*args, **kwargs)
    if not capture:
        return result, None
    slot = int(args[1].reshape(-1)[0].item())
    decode_index = int(args[2].reshape(-1)[0].item())
    if args[1].numel() != 1:
        raise ValueError('single-request observer required')
    scaled = result.detach()
    row = dict(is_commit=before_commit, scaled_logits=scaled,
        native_winners=args[6][slot].detach().clone(),
        native_history=args[13][slot].detach().clone(),
        native_history_len=int(args[14][slot].item()),
        native_confidence=bool(args[9][slot].item()),
        native_commit_pending=bool(args[8][slot].item()),
        native_step=int(args[7][slot].item()),
        sampled_canvas=args[15][decode_index].detach().clone(),
        canvas_after=args[5][slot].detach().clone(),
        self_condition_after=args[10][slot].detach().clone(),
        confidence_threshold=float(kwargs['confidence_threshold']),
        entropy_bound=float(kwargs['entropy_bound']))
    if not before_commit:
        row['winner_measurement'] = native_argmax_tie_summary(
            scaled, row['native_winners'])
        accepted_mask, acceptance = acceptance_reconstructed_on_device(
            scaled, row['entropy_bound'])
        row['acceptance_measurement'] = acceptance
        row['acceptance_reconstructed_mask'] = accepted_mask.detach().cpu().clone()
        row['acceptance_label'] = 'reconstructed_on_device_not_native_buffer'
    row['state_hashes'] = dict(canvas=_digest(row['canvas_after']),
        winners=_digest(row['native_winners']), history=_digest(row['native_history']),
        self_condition=_digest(row['self_condition_after']))
    return result, row


def _state_delta(reference: dict, candidate: dict) -> dict:
    out = dict(is_commit=candidate['is_commit'],
        actual_native_winner_mismatch=int((reference['native_winners'] != candidate['native_winners']).sum()),
        native_history_mismatch=int((reference['native_history'] != candidate['native_history']).sum().item()),
        native_history_len_changed=reference['native_history_len'] != candidate['native_history_len'],
        native_confidence_changed=reference['native_confidence'] != candidate['native_confidence'],
        native_commit_pending_changed=reference['native_commit_pending'] != candidate['native_commit_pending'],
        native_step_delta=candidate['native_step']-reference['native_step'],
        sampled_canvas_mismatch=int((reference['sampled_canvas'] != candidate['sampled_canvas']).sum().item()),
        canvas_mismatch=int((reference['canvas_after'] != candidate['canvas_after']).sum().item()),
        self_condition_l2=float((reference['self_condition_after'].float()-candidate['self_condition_after'].float()).norm().item()))
    if not candidate['is_commit']:
        # Count accepted differences only at the explicitly reconstructed level.
        out['acceptance_reconstruction'] = 'device_reconstructed_only'
        out['acceptance_reconstructed_mismatches'] = int((reference['acceptance_reconstructed_mask'] != candidate['acceptance_reconstructed_mask']).sum())
        out['native_winner_diagnostic'] = dict(
            reference=reference['winner_measurement'], candidate=candidate['winner_measurement'])
        out['logits'] = centered_distribution_metrics(reference['scaled_logits'], candidate['scaled_logits'])
        out['reference_winner_margin'] = reference_winner_margin(
            reference['scaled_logits'], candidate['scaled_logits'])
        out['acceptance_measurement'] = dict(
            reference=reference['acceptance_measurement'],
            candidate=candidate['acceptance_measurement'])
    return out


def align_branch(reference_events: list[dict], candidate_events: list[dict]) -> dict:
    n = min(len(reference_events), len(candidate_events))
    rows = [_state_delta(reference_events[i], candidate_events[i]) for i in range(n)]
    return dict(aligned_calls=n, reference_events=len(reference_events),
        candidate_events=len(candidate_events), events=rows,
        second_call_available=n >= 2 and not candidate_events[0]['native_commit_pending'])


def decision_score(reference_first: dict, sparse_first: dict,
                   stable_anchor_mask: torch.Tensor) -> dict:
    """Current first-pulse score; no second-call or stopping outcome is read."""
    ref_logits = reference_first['scaled_logits'].reshape(-1, reference_first['scaled_logits'].shape[-1]).float()
    cur_logits = sparse_first['scaled_logits'].reshape(-1, sparse_first['scaled_logits'].shape[-1]).float()
    ref_winner = reference_first['native_winners'].reshape(-1).to(ref_logits.device)
    cur_winner = sparse_first['native_winners'].reshape(-1).to(ref_logits.device)
    stable = stable_anchor_mask.reshape(-1).to(ref_logits.device, dtype=torch.bool)
    # argmax_canvas is the native helper's actual clean winner state.
    flips = (ref_winner != cur_winner) & stable
    ref_score = ref_logits.gather(1, ref_winner[:, None]).squeeze(1)
    others = cur_logits.clone()
    others.scatter_(1, ref_winner[:, None], -torch.inf)
    candidate_margin = cur_logits.gather(1, ref_winner[:, None]).squeeze(1)-others.max(-1).values
    native_history = sparse_first['native_history']
    stable_set_empty = not bool(stable.any().item())
    logp = sparse_first['scaled_logits'].float().log_softmax(-1)
    token_entropy = -(logp.exp()*logp).sum(-1)
    # Stable-set wins are first; lower canvas entropy then lower worst margin erosion.
    ref_top = ref_logits.topk(2, dim=-1).values
    reference_margin = ref_top[:, 0]-ref_top[:, 1]
    erosion = reference_margin-candidate_margin
    return dict(score='lexicographic_current_first_pulse_only',
        stable_anchor_positions=int(stable.sum().item()), stable_set_empty=stable_set_empty,
        reference_winner_flips_stable=int(flips.sum().item()),
        mean_canvas_entropy=float(token_entropy.mean().item()),
        worst_reference_margin_erosion=float(erosion.max().item()),
        tie_positions=int((ref_top[:, 0] == ref_top[:, 1]).sum().item()),
        history_digest=_digest(native_history),
        score_tuple=[int(flips.sum().item()), float(token_entropy.mean().item()),
                     float(erosion.max().item())],
        score_firewall='No second-call, stop, task-score, or final-output fields used')
