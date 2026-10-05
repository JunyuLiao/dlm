"""Torch oracle for historical transformed-score routing and current-V output.

The score tensor already includes the native QK rounding, scale and optional
additive bias. ``-inf`` denotes an illegal key. No Q or K is accepted here.
This oracle deliberately allows small Q, K and D for exact unit tests.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class Output:
    output: torch.Tensor
    skipped: torch.Tensor
    eligible: torch.Tensor
    log_normalizer: torch.Tensor
    projected_state: torch.Tensor
    risk: torch.Tensor
    invalid_scores: torch.Tensor


def attention(scores, v, z=None, reference=None, *, sensitivity=None,
              log_threshold=-math.inf, skipped=None, q_tile=128, k_tile=64):
    """Route whole physical Q/K tiles, then normalize on retained legal keys.

    ``skipped`` is a held M3 bitmap. When supplied, no projected-V selection is
    performed; ``z`` and ``reference`` may be omitted. Invalid (+inf/NaN)
    scores force their tile to be retained and flag their row for fallback.
    The returned output for a flagged row is zero and must not be consumed.
    """
    if scores.ndim != 4 or scores.dtype != torch.float32:
        raise ValueError('scores must be FP32 [B,H,Q,K]')
    if v.ndim != 4 or v.shape[0] != scores.shape[0] or v.shape[2] != scores.shape[3]:
        raise ValueError('v must be [B,KVH,K,D] with matching B,K')
    b, h, nq, nk = scores.shape
    hk, d = v.shape[1], v.shape[-1]
    if not (nq and nk and hk and h % hk == 0 and q_tile > 0 and k_tile > 0):
        raise ValueError('invalid dimensions or GQA')
    if v.device != scores.device or math.isnan(log_threshold):
        raise ValueError('device mismatch or NaN threshold')
    qb = (nq + q_tile - 1) // q_tile
    kt = (nk + k_tile - 1) // k_tile
    if sensitivity is None:
        sensitivity = scores.new_ones((b, nq))
    if sensitivity.shape != (b, nq) or sensitivity.dtype != torch.float32 or sensitivity.device != scores.device:
        raise ValueError('sensitivity must be FP32 [B,Q]')
    if bool((~torch.isfinite(sensitivity) | (sensitivity <= 0)).any()):
        raise ValueError('sensitivity must be finite positive')
    if skipped is not None:
        if skipped.shape != (b, h, qb, kt) or skipped.dtype != torch.bool or skipped.device != scores.device:
            raise ValueError('skipped must be bool [B,H,ceil(Q/128),ceil(K/64)]')
    else:
        if z is None or reference is None or z.shape[:3] != (b, hk, nk) or reference.shape != (b, hk):
            raise ValueError('M1 requires projected current V and reference')
        if z.dtype != torch.float32 or reference.dtype != torch.float32 or z.device != scores.device or reference.device != scores.device:
            raise ValueError('z and reference must be device-matched FP32')
    r = 0 if z is None else z.shape[-1]
    out = torch.zeros((b, h, nq, d), dtype=v.dtype, device=v.device)
    sk = torch.zeros((b, h, qb, kt), dtype=torch.bool, device=scores.device)
    el = torch.zeros_like(sk)
    lse = torch.full((b, h, nq), -math.inf, dtype=torch.float32, device=scores.device)
    state = torch.zeros((b, h, nq, r), dtype=torch.float32, device=scores.device)
    risks = torch.full((b, h, qb, kt), -math.inf, dtype=torch.float32, device=scores.device)
    bad = torch.isnan(scores) | torch.isposinf(scores)
    for bi in range(b):
        for hi in range(h):
            vi = v[bi, hi // (h // hk)].float()
            zi = None if z is None else z[bi, hi // (h // hk)]
            ref = None if reference is None else max(float(reference[bi, hi // (h // hk)]), 1e-12)
            for qi in range(qb):
                lo, end = qi*q_tile, min(nq, (qi+1)*q_tile)
                previous = scores.new_full((end-lo,), -math.inf)
                projected = scores.new_zeros((end-lo, r))
                selected = torch.zeros((end-lo, nk), dtype=torch.bool, device=scores.device)
                for ki in range(kt):
                    ks, ke = ki*k_tile, min(nk, (ki+1)*k_tile)
                    tile = scores[bi, hi, lo:end, ks:ke]
                    legal = torch.isfinite(tile)
                    active = legal.any(-1)
                    invalid = bad[bi, hi, lo:end, ks:ke].any(-1)
                    eligible = bool((active | invalid).any())
                    el[bi, hi, qi, ki] = eligible
                    if skipped is not None:
                        drop = bool(skipped[bi, hi, qi, ki]) and not bool(invalid.any()) and eligible
                    else:
                        if eligible:
                            maximum = tile.masked_fill(~legal, -math.inf).amax(-1)
                            block_z = torch.logsumexp(tile.masked_fill(~legal, -math.inf), -1)
                            combined = torch.logaddexp(previous, block_z)
                            alpha = torch.where(active, torch.exp(block_z - combined), torch.zeros_like(block_z))
                            weights = torch.where(legal, torch.exp(tile - maximum[:, None]), 0.)
                            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)
                            mu = weights @ zi[ks:ke]
                            delta = alpha[:, None] * (mu - projected)
                            magnitude = torch.linalg.vector_norm(delta, dim=-1)
                            risk = torch.log(magnitude / ref) + torch.log(sensitivity[bi, lo:end])
                            risk = torch.where(active, torch.where(torch.isfinite(previous), risk, math.inf), -math.inf)
                            risk = torch.where(invalid, math.inf, risk)
                            worst = float(risk.amax())
                            risks[bi, hi, qi, ki] = worst
                            drop = worst < log_threshold
                        else:
                            drop = False
                    sk[bi, hi, qi, ki] = drop
                    if eligible and not drop:
                        selected[:, ks:ke] = legal
                        if skipped is None:
                            old = torch.where(torch.isfinite(previous), torch.exp(previous-combined), 0.)
                            projected = old[:, None]*projected + alpha[:, None]*mu
                            previous = combined
                if skipped is None:
                    state[bi, hi, lo:end] = projected
                all_scores = scores[bi, hi, lo:end].masked_fill(~selected, -math.inf)
                normalizer = torch.logsumexp(all_scores, -1)
                finite = torch.isfinite(normalizer)
                probs = torch.where(selected & finite[:, None], torch.exp(all_scores - normalizer[:, None]), 0.)
                value = probs @ vi
                invalid_rows = bad[bi, hi, lo:end].any(-1)
                out[bi, hi, lo:end] = torch.where(invalid_rows[:, None], 0., value).to(v.dtype)
                lse[bi, hi, lo:end] = torch.where(invalid_rows, -math.inf, normalizer)
    return Output(out, sk, el, lse, state, risks, bad.any(-1))
