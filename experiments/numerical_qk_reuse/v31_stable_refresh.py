"""Selector-only conservative refresh, independently derived for the stable study.

Raw statistics use natural-log units and [query-head * qblock, tile, row].
The dense anchor is immutable. All output attention remains with native FA4.
No value ranking, query-head sharing, or implicit dense rescue is implemented.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class StableConfig:
    mode: str = 'stable'
    hysteresis: float = math.log(1.25)
    persistence: int = 2
    event_cap: int = 2
    cumulative_cap: int = 8
    removal_tolerance: float = 0.01
    updates_enabled: bool = True
    probes: int = 0

    def __post_init__(self):
        if self.mode not in ('raw', 'stable', 'bounded_diagnostic'):
            raise ValueError('unknown stable-refresh policy')
        if (not math.isfinite(self.hysteresis) or self.hysteresis < 0
                or self.persistence < 1 or self.event_cap < 0 or self.cumulative_cap < 0
                or not 0 <= self.removal_tolerance <= 1 or not 0 <= self.probes <= 4):
            raise ValueError('invalid stability parameter')


def assert_budget(selected, k, mandatory=None):
    ok = (selected.sum(-1) == k).all()
    if mandatory is not None:
        ok = ok & (~mandatory | selected).all()
    if selected.is_cuda:
        torch._assert_async(ok, 'stable refresh budget/protection violation')
    elif not bool(ok):
        raise AssertionError('stable refresh budget/protection violation')


def rows_to_tiles(head_lse, n=None):
    """Explicit transpose: [H, rows, PT] -> [H*QB, PT, 128]."""
    h, rows, pt = head_lse.shape
    n = rows if n is None else n
    if rows != n:
        raise ValueError('valid-row geometry mismatch')
    qb = (n + 127) // 128
    padded = torch.nn.functional.pad(head_lse, (0, 0, 0, qb * 128 - n), value=-math.inf)
    return padded.reshape(h, qb, 128, pt).permute(0, 1, 3, 2).reshape(h * qb, pt, 128).contiguous()


def mass_scores(z):
    if z.shape[1] == 0:
        return z.new_empty(z.shape[:2])
    denom = z.logsumexp(1, keepdim=True)
    norm = torch.where(torch.isfinite(denom), z - denom, -math.inf)
    return norm.amax(-1)


def stable_topk(scores, k, mandatory=None):
    """Lower physical tile ID wins equal-score ties on CPU and CUDA."""
    scores = scores if mandatory is None else scores.masked_fill(mandatory, math.inf)
    out = torch.zeros_like(scores, dtype=torch.bool)
    out.scatter_(1, scores.argsort(dim=-1, descending=True, stable=True)[:, :k], True)
    assert_budget(out, k, mandatory)
    return out


def corrected_bounded(scores, current, k, m, cap=8, mandatory=None):
    """Limited diagnostic; no empty-frontier writes or invalid-map repairs."""
    assert_budget(current, k, mandatory)
    if k == 0 or k == scores.shape[1]:
        return current.clone()
    mandatory = torch.zeros_like(current) if mandatory is None else mandatory
    frontier = stable_topk(scores, min(m, scores.shape[1]))
    held = current.clone()
    available = frontier & ~current
    for _ in range(cap):
        incoming = scores.masked_fill(~available, -math.inf)
        ci = incoming.argmax(-1)
        weakest = scores.masked_fill(~held | mandatory, math.inf).argmin(-1)
        gain = scores.gather(1, ci[:, None]) - scores.gather(1, weakest[:, None])
        accept = available.any(-1) & (held & ~mandatory).any(-1) & (gain[:, 0] > 0)
        proposal = held.clone()
        proposal.scatter_(1, weakest[:, None], False)
        proposal.scatter_(1, ci[:, None], True)
        held = torch.where(accept[:, None], proposal, held)
        available.scatter_(1, ci[:, None], False)
        assert_budget(held, k, mandatory)
    return held


def support_drift(fresh, anchor, common):
    """Common-support absolute, shift-residual and normalized TV changes."""
    active = common[..., None] & torch.isfinite(anchor) & torch.isfinite(fresh)
    delta = torch.where(active, fresh - anchor, 0.)
    count = active.sum(1).clamp_min(1)
    shift = delta.sum(1) / count
    residual = torch.where(active, delta - shift[:, None], 0.)
    a = anchor.masked_fill(~active, -math.inf)
    b = fresh.masked_fill(~active, -math.inf)
    pa, pb = torch.softmax(a, 1), torch.softmax(b, 1)
    tv = torch.nan_to_num((pa - pb).abs()).sum(1) * 0.5
    return torch.stack((delta.abs().amax(), shift.abs().amax(), residual.abs().amax(), tv.amax()))


class StableState:
    def __init__(self, anchor, selected, config=StableConfig(), mandatory=None):
        self.anchor = anchor.clone()
        self.selected = selected.clone()
        self.initial = selected.clone()
        self.config = config
        self.k = min(selected.shape[1], int(selected.sum(-1)[0])) if selected.shape[0] else 0
        self.mandatory = torch.zeros_like(selected) if mandatory is None else mandatory.clone()
        assert_budget(selected, self.k, self.mandatory)
        self.streak = torch.zeros_like(selected, dtype=torch.int32)
        self.age = torch.zeros_like(selected, dtype=torch.int32)
        self.last_observed = torch.zeros_like(selected, dtype=torch.int32)
        self.cumulative = torch.zeros(selected.shape[0], device=selected.device, dtype=torch.int32)
        self.removed_before = torch.zeros_like(selected)
        self.previous_additions = torch.zeros_like(selected)
        self.events = 0
        # Aggregate tensors stay on device until the request receipt.
        self.counts = torch.zeros(11, device=anchor.device, dtype=torch.float64)
        self.max_event_swaps = torch.zeros((), device=anchor.device, dtype=torch.int32)
        self.drift_max = torch.zeros(4, device=anchor.device, dtype=torch.float32)

    def observe(self, fresh, tail_lse, probe_mask=None):
        old = self.selected
        if fresh.shape != self.anchor.shape or tail_lse.shape != (old.shape[0], self.anchor.shape[-1]):
            raise ValueError('observation geometry mismatch')
        self.events += 1
        self.last_observed.masked_fill_(old, self.events)
        active = old[..., None] & torch.isfinite(self.anchor)
        invalid = ((active & ~torch.isfinite(fresh)).any() | torch.isnan(fresh).any() | torch.isposinf(fresh).any()
                   | torch.isnan(tail_lse).any() | torch.isposinf(tail_lse).any())
        if fresh.is_cuda:
            torch._assert_async(~invalid, 'numerical-invalid observation; no silent dense rescue')
        elif bool(invalid):
            raise FloatingPointError('numerical-invalid observation; no silent dense rescue')
        support = old
        if probe_mask is not None:
            if not self.config.probes:
                raise ValueError('probing must be separately configured')
            count = (probe_mask & ~old).sum(-1)
            ok = (count <= self.config.probes).all()
            if fresh.is_cuda: torch._assert_async(ok, 'outside probe budget exceeded')
            elif not bool(ok): raise AssertionError('outside probe budget exceeded')
            support = old | probe_mask
        hybrid = torch.where(support[..., None], fresh, self.anchor)
        score = mass_scores(hybrid)
        cfg = self.config
        proposed = stable_topk(score + cfg.hysteresis * old if cfg.mode == 'stable' else score,
                               self.k, self.mandatory)
        favorable = proposed & ~old
        self.streak = torch.where(favorable, (self.streak + 1).clamp_max(100000), 0)
        self.drift_max = torch.maximum(self.drift_max, support_drift(fresh, self.anchor, old)) if self.anchor.numel() else self.drift_max
        selected = old.clone()
        rejected_guard = torch.zeros(old.shape[0], device=old.device, dtype=torch.bool)
        if cfg.updates_enabled and cfg.mode == 'raw':
            selected = proposed
        elif cfg.updates_enabled and cfg.mode == 'bounded_diagnostic':
            selected = corrected_bounded(score, old, self.k, min(self.k+32, score.shape[1]), 8, self.mandatory)
        elif cfg.updates_enabled and self.k and self.k < score.shape[1]:
            eligible = favorable & (self.streak >= cfg.persistence)
            removable = old & ~proposed & ~self.mandatory
            # Denominator is the full current sparse support, including all mandatory tail keys.
            norm = torch.logaddexp(fresh.masked_fill(~old[..., None], -math.inf).logsumexp(1), tail_lse)
            removed_mass = torch.zeros_like(norm)
            for _ in range(cfg.event_cap):
                ci = score.masked_fill(~eligible, -math.inf).argmax(-1)
                wi = score.masked_fill(~removable, math.inf).argmin(-1)
                gain = score.gather(1, ci[:, None]) - score.gather(1, wi[:, None])
                mass = (fresh.gather(1, wi[:, None, None].expand(-1, 1, fresh.shape[-1]))[:, 0] - norm).exp()
                mass = torch.nan_to_num(mass)
                guard = (removed_mass + mass).amax(-1) <= cfg.removal_tolerance
                candidate_ok = eligible.any(-1) & removable.any(-1) & (gain[:, 0] > cfg.hysteresis) & (self.cumulative < cfg.cumulative_cap)
                rejected_guard |= candidate_ok & ~guard
                accept = candidate_ok & guard
                proposal = selected.clone()
                proposal.scatter_(1, wi[:, None], False)
                proposal.scatter_(1, ci[:, None], True)
                selected = torch.where(accept[:, None], proposal, selected)
                removed_mass += torch.where(accept[:, None], mass, 0.)
                self.cumulative += accept.to(torch.int32)
                eligible.scatter_(1, ci[:, None], False)
                removable.scatter_(1, wi[:, None], False)
                assert_budget(selected, self.k, self.mandatory)
        additions, removals = selected & ~old, old & ~selected
        swaps = additions.sum(-1)
        self.max_event_swaps = torch.maximum(self.max_event_swaps,swaps.max().to(torch.int32))
        union = (selected | old).sum(-1).clamp_min(1)
        iou = (selected & old).sum(-1) / union
        anchor_iou = (selected & self.initial).sum(-1) / (selected | self.initial).sum(-1).clamp_min(1)
        reentry = (additions & self.removed_before).sum()
        reversals = (removals & self.previous_additions).sum()
        self.counts += torch.stack((swaps.sum(), (swaps>0).sum(), (swaps==0).sum(), rejected_guard.sum(),
                                    iou.sum(), anchor_iou.sum(), reentry, reversals,
                                    (favorable.any(-1)&(swaps==0)).sum(),
                                    (favorable&(self.streak<cfg.persistence)).any(-1).sum(),
                                    (favorable.any(-1)&(self.cumulative>=cfg.cumulative_cap)).sum())).to(torch.float64)
        self.removed_before |= removals
        self.previous_additions = additions
        self.age = torch.where(selected, 0, (self.age + 1).clamp_max(100000))
        self.selected = selected
        return selected, dict(swaps=swaps, rejected_guard=rejected_guard, proposed=proposed, score=score,
                              iou_previous=iou, iou_anchor=anchor_iou)
