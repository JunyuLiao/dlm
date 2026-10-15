"""Matched-snapshot diagnostics for the fixed-budget V2 follow-up.

This script is intentionally an operator diagnostic.  It evaluates the
control-sized mass support, online V2 risk, and full-information value
omission risk on the same saved Q/K/V snapshots.  It never runs the sampler
and writes only sanitized aggregates.  Full-information scores use the true
full-dimensional V output; online scores use the sequential preserve-mass
state.  Every fixed-budget map owns exactly K prefix tiles per qblock unless
the intersection arm has fewer tiles by definition.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.diffusion_gemma_jl_output_aware.projections import Projections


def tile_stats(q, k, v, sketch, scale, j):
    h, nq, _ = q.shape
    hk = k.shape[0]
    groups = h // hk
    kh = torch.arange(h, device=q.device) // groups
    kt = k.index_select(0, kh)[:, j * 64:(j + 1) * 64]
    vt = v.index_select(0, kh)[:, j * 64:(j + 1) * 64]
    zt = sketch.index_select(0, kh)[:, j * 64:(j + 1) * 64]
    score = torch.einsum('hqd,hkd->hqk', q.float(), kt.float()) * float(scale)
    lm = torch.logsumexp(score, -1)
    weights = torch.softmax(score, -1)
    full_mu = torch.einsum('hqk,hkd->hqd', weights, vt.float())
    proj_mu = torch.einsum('hqk,hkr->hqr', weights, zt.float())
    return lm, full_mu, proj_mu


def online_map(log_mass, mus, nu, threshold):
    """V2 preserve-mass map and its per-qblock online risks."""
    h, q, pt = log_mass.shape
    qb = (q + 127) // 128
    pad = qb * 128 - q
    lm = torch.nn.functional.pad(log_mass, (0, 0, 0, pad), value=-float('inf'))
    nu = torch.nn.functional.pad(nu, (0, pad), value=1.)
    output = torch.zeros((h, qb * 128, mus[0].shape[-1]), device=log_mass.device)
    previous = torch.full((h, qb * 128), -float('inf'), device=log_mass.device)
    support = torch.zeros_like(previous, dtype=torch.bool)
    keep = torch.zeros((h, qb, pt), device=log_mass.device, dtype=torch.bool)
    risk = torch.full_like(keep, -float('inf'), dtype=torch.float32)
    log_thr = math.log(float(threshold))
    for j in range(pt):
        z = lm[:, :, j]
        combined = torch.logaddexp(previous, z)
        eta = torch.where(torch.isfinite(z), torch.exp(z - combined), torch.zeros_like(z))
        delta = mus[j] - output
        rho = eta * delta.norm(dim=-1) / nu.clamp_min(1e-12)
        r = torch.where(torch.isfinite(z), rho.log(), torch.full_like(rho, -float('inf')))
        r = r.view(h, qb, 128)
        active = torch.isfinite(z).view(h, qb, 128)
        rr = r.amax(-1)
        first = (active & ~support.view(h, qb, 128)).any(-1)
        take = active.any(-1) & (first | (rr >= log_thr))
        keep[:, :, j] = take
        risk[:, :, j] = rr
        row_take = take.repeat_interleave(128, 1)
        output = torch.where(row_take[..., None], (1 - eta[..., None]) * output + eta[..., None] * mus[j], output)
        previous = combined
        support |= active.reshape(h, qb * 128) & row_take
    return keep, risk


def mass_map(log_mass, k):
    h, q, pt = log_mass.shape
    qb = (q + 127) // 128
    z = torch.nn.functional.pad(log_mass, (0, 0, 0, qb * 128 - q), value=-float('inf'))
    share = z - torch.logsumexp(z, -1, keepdim=True)
    score = share.view(h, qb, 128, pt).amax(2)
    take = min(int(k), pt)
    out = torch.zeros((h, qb, pt), device=log_mass.device, dtype=torch.bool)
    out.scatter_(-1, score.topk(take, -1).indices, True)
    return out


def full_scores(log_mass, full_mus, full_out, nu):
    h, q, pt = log_mass.shape
    qb = (q + 127) // 128
    log_z = torch.logsumexp(log_mass, -1)
    out = torch.zeros((h, q, pt), device=log_mass.device)
    for j in range(pt):
        alpha = torch.exp(log_mass[:, :, j] - log_z)
        g = alpha[..., None] * (full_mus[j] - full_out)
        remain = (1 - alpha).clamp_min(1e-12)
        e = g.norm(dim=-1) / (remain * nu.clamp_min(1e-12))
        out[:, :, j] = e
    return out.view(h, qb, 128, pt).amax(2)


def qblock_scores(row_scores, q):
    h, _, pt = row_scores.shape
    qb = (q + 127) // 128
    padded = torch.nn.functional.pad(row_scores, (0, 0, 0, qb * 128 - q), value=-float('inf'))
    return padded.view(h, qb, 128, pt).amax(2)


def swap_map(base, candidate, score, swaps):
    out = base.clone()
    h, qb, pt = out.shape
    for hi in range(h):
        for bi in range(qb):
            add = candidate[hi, bi] & ~out[hi, bi]
            rem = out[hi, bi]
            if not bool(add.any()) or not bool(rem.any()) or swaps <= 0:
                continue
            add_order = torch.argsort(score[hi, bi].masked_fill(~add, -float('inf')), descending=True, stable=True)
            rem_order = torch.argsort(score[hi, bi].masked_fill(~rem, float('inf')), descending=False, stable=True)
            for ai, ri in zip(add_order[:swaps].tolist(), rem_order[:swaps].tolist()):
                if not torch.isfinite(score[hi, bi, ai]) or not torch.isfinite(score[hi, bi, ri]):
                    continue
                if float(score[hi, bi, ai]) <= float(score[hi, bi, ri]):
                    break
                out[hi, bi, ri] = False
                out[hi, bi, ai] = True
    return out


def evaluate_map(log_mass, full_mus, full_out, nu, maps, control):
    h, q, pt = log_mass.shape
    log_z = torch.logsumexp(log_mass, -1)
    names = list(maps)
    nums = {name: torch.zeros_like(full_out) for name in names}
    dens = {name: torch.zeros((h, q), device=log_mass.device) for name in names}
    for j in range(pt):
        alpha = torch.exp(log_mass[:, :, j] - log_z)
        for name in names:
            mask = maps[name][:, :, j].repeat_interleave(128, 1)[:, :q]
            nums[name] += alpha[..., None] * full_mus[j] * mask[..., None]
            dens[name] += alpha * mask
    result = {}
    cset = control
    for name in names:
        out = nums[name] / dens[name].clamp_min(1e-12)[..., None]
        err = (out - full_out).norm(dim=-1) / nu.clamp_min(1e-12)
        inter = (maps[name] & cset).sum((-1, -2)).float()
        union = (maps[name] | cset).sum((-1, -2)).float().clamp_min(1)
        result[name] = dict(keep_tiles=float(maps[name].sum().item() / (h * maps[name].shape[1])),
                            retained_mass=float(dens[name].mean().item()),
                            error_mean=float(err.mean().item()), error_max=float(err.max().item()),
                            iou_control=float((inter / union).mean().item()))
    return result


@torch.inference_mode()
def one(path, threshold=.027, budget=128, pool_factor=2, swaps=16):
    raw = torch.load(path, map_location='cpu')
    device = torch.device('cuda')
    q = raw['q'][0].to(device).float()
    k = raw['k'][0].to(device).float()
    v = raw['v'][0].to(device).float()
    scale, prefix = float(raw['scale']), int(raw['prefix'])
    h, nq, d = q.shape
    hk = k.shape[0]
    pt = prefix // 64
    nu = v.square().sum(-1).mean(-1).sqrt()[torch.arange(h, device=device) // (h // hk)][:, None]
    proj = Projections().get(int(raw['layer']), hk, d, 'gaussian', 32, 1729, device)
    sketch = torch.einsum('hkd,hdr->hkr', v, proj)
    log_mass = torch.empty((h, nq, pt), device=device)
    full_out = torch.zeros((h, nq, d), device=device)
    for j in range(pt):
        lm, fm, _ = tile_stats(q, k, v, sketch, scale, j)
        log_mass[:, :, j] = lm
    log_z = torch.logsumexp(log_mass, -1)
    full_mus = []
    proj_mus = []
    for j in range(pt):
        lm, fm, pm = tile_stats(q, k, v, sketch, scale, j)
        full_mus.append(fm)
        proj_mus.append(pm)
        full_out += torch.exp(lm - log_z)[..., None] * fm
    control = mass_map(log_mass, budget)
    full_sc = full_scores(log_mass, full_mus, full_out, nu)
    # The full-information score is the true final-output omission score.  The
    # online scores use the same sequential V2 state, once with projected V and
    # once with full V.
    online_p, online_p_r = online_map(log_mass, proj_mus, nu, threshold)
    online_f, online_f_r = online_map(log_mass, full_mus, nu, threshold)
    online_p_s = online_p_r
    online_f_s = online_f_r
    mass_candidate = torch.zeros_like(control)
    mass_qscore = qblock_scores(log_mass, nq)
    pool = min(pt, budget * pool_factor)
    mass_candidate.scatter_(-1, mass_qscore.topk(pool, -1).indices, True)
    maps = {
        'control_mass': control,
        'v2_online': online_p,
        'mass_gated_online_swap8': swap_map(control, mass_candidate, online_p_s, 8),
        'mass_gated_online_swap16': swap_map(control, mass_candidate, online_p_s, swaps),
        'mass_gated_full_swap16': swap_map(control, mass_candidate, full_sc, swaps),
        'intersection_online': control & online_p,
        'intersection_full': control & online_p,
    }
    # The phase arm is mass-only on a fresh selection and value-refined after
    # reuse.  Snapshot age is the closest available proxy for that lifecycle.
    maps['mass_gated_online_phase'] = swap_map(control, mass_candidate, online_p_s, 0 if int(raw.get('age', 0)) == 0 else swaps)
    # C_gate uses a fixed quota; this diagnostic uses an attention-diffuseness
    # proxy because sampler logits are not stored in private snapshots.
    row_mass = torch.softmax(log_mass, -1).amax(-1)
    cweight = (1 + 3 * (1 - row_mass)).clamp(1, 4)
    cweight = torch.nn.functional.pad(cweight, (0, ((nq + 127)//128) * 128 - nq), value=1.)
    cp = online_p_r * cweight.view(h, (nq + 127)//128, 128).amax(2)[..., None]
    maps['mass_gated_online_cgate'] = swap_map(control, mass_candidate, cp, swaps)
    # Re-rank the intersection by the true final-output score; this is the
    # full-information intersection arm.
    maps['intersection_full'] = torch.zeros_like(control)
    for hi in range(h):
        for bi in range(control.shape[1]):
            allowed = control[hi, bi] & online_p[hi, bi]
            idx = torch.argsort(full_sc[hi, bi].masked_fill(~allowed, -float('inf')), descending=True, stable=True)
            maps['intersection_full'][hi, bi, idx[:int(allowed.sum())]] = True
    metrics = evaluate_map(log_mass, full_mus, full_out, nu, maps, control)
    # Add direct support diagnostics for the intersection and exact budgets.
    for name, mask in maps.items():
        metrics[name]['budget_min'] = int(mask.sum(-1).min().item())
        metrics[name]['budget_max'] = int(mask.sum(-1).max().item())
    return dict(file=Path(path).name, age=int(raw.get('age', 0)), phase=raw.get('phase'),
                prefix_tiles=pt, query_rows=nq, threshold=threshold, budget=budget,
                pool_factor=pool_factor, swaps=swaps, metrics=metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.027)
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--pool-factor', type=int, default=2)
    p.add_argument('--swaps', type=int, default=16)
    args = p.parse_args()
    rows = [one(x, args.threshold, args.budget, args.pool_factor, args.swaps) for x in args.snapshots]
    names = sorted(rows[0]['metrics'])
    aggregate = {name: {key: sum(row['metrics'][name][key] for row in rows) / len(rows)
                         for key in ('keep_tiles','retained_mass','error_mean','error_max','iou_control')}
                 for name in names}
    out = dict(schema='v31_fixed_budget_value_diagnostic_v1', status='complete',
               threshold=args.threshold, budget=args.budget, pool_factor=args.pool_factor,
               swaps=args.swaps, snapshots=rows, aggregate=aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
