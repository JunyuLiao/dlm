"""Offline full-dimensional-V diagnostic for the online V2 selector.

This is deliberately an operator diagnostic: it runs on saved private Q/K/V
snapshots, does not invoke the model, and writes only aggregate metrics.  The
online state is streamed tile by tile, so it never materializes a
Q-by-K-by-512 tensor or a full per-tile value summary.
"""
import argparse
import json
from pathlib import Path

import torch

from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
from experiments.numerical_qk_reuse.v31_value_selectors import SELECTORS, select


def tile_mu(q, k, v, scale, j):
    # Native GQA: each query head uses its KV head's tile.
    groups = q.shape[0] // k.shape[0]
    kh = torch.arange(q.shape[0], device=q.device)//groups
    kt = k.index_select(0, kh)[:, j*64:(j+1)*64]
    vt = v.index_select(0, kh)[:, j*64:(j+1)*64]
    score = torch.einsum('hqd,hkd->hqk', q.float(), kt.float()) * scale
    lm = torch.logsumexp(score, -1)
    w = torch.softmax(score, -1)
    mu = torch.einsum('hqk,hkd->hqd', w, vt.float())
    return lm, mu


@torch.inference_mode()
def one(path, threshold=0.027):
    raw = torch.load(path, map_location='cpu')
    device = torch.device('cuda')
    q = raw['q'][0].to(device)
    k = raw['k'][0].to(device)
    v = raw['v'][0].to(device)
    scale, prefix = float(raw['scale']), int(raw['prefix'])
    h, n, d = q.shape
    hk, nk, _ = k.shape
    qb, pt = (n + 127)//128, prefix//64
    qpad = qb*128
    nu = (v.float().square().sum(-1).mean(-1).sqrt()).contiguous()
    log_mass = torch.full((h, qpad, pt), -float('inf'), device=device)
    full_keep = torch.zeros((h, qb, pt), dtype=torch.bool, device=device)
    previous = torch.full((h, qpad), -float('inf'), device=device)
    output = torch.zeros((h, qpad, d), device=device)
    support = torch.zeros((h, qpad), dtype=torch.bool, device=device)
    log_threshold = torch.log(torch.tensor(threshold, device=device))
    q_full = torch.nn.functional.pad(q, (0, 0, 0, qpad-n))
    for j in range(pt):
        lm, mu = tile_mu(q_full, k, v, scale, j)
        log_mass[:, :, j] = lm
        combined = torch.logaddexp(previous, lm)
        eta = torch.exp(lm-combined)
        delta = mu-output
        rho = eta * delta.norm(dim=-1) / nu[torch.arange(h, device=device)//(h//hk)][:, None].clamp_min(1e-12)
        risk = torch.log(rho).view(h, qb, 128).amax(-1)
        active = torch.isfinite(lm).view(h, qb, 128).any(-1)
        first = (~support.view(h, qb, 128)).any(-1)
        take = active & (first | (risk >= log_threshold))
        full_keep[:, :, j] = take
        take_rows = take.repeat_interleave(128, 1)
        output = torch.where(take_rows[..., None],
                             (1-eta[..., None])*output+eta[..., None]*mu, output)
        previous = combined
        support |= torch.isfinite(lm)
    # The current V31 control is the exact-mass qblock_max top-128 map.
    share = log_mass[:, :n].view(h, qb, 128, pt)
    share = share - torch.logsumexp(share, -1, keepdim=True)
    score = share.amax(2)
    control_keep = torch.zeros((h, qb, pt), dtype=torch.bool, device=device)
    control_keep.scatter_(-1, score.topk(min(128, pt), -1).indices, True)
    # Re-evaluate full, control-masked, and full-V-masked outputs in one pass.
    log_z = torch.logsumexp(log_mass[:, :n], -1)
    full_num = torch.zeros((h, n, d), device=device)
    control_num = torch.zeros_like(full_num)
    fullv_num = torch.zeros_like(full_num)
    full_den = torch.zeros((h, n), device=device)
    control_den = torch.zeros_like(full_den)
    fullv_den = torch.zeros_like(full_den)
    for j in range(pt):
        lm, mu = tile_mu(q, k, v, scale, j)
        alpha = torch.exp(lm-log_z)
        ck = control_keep[:, :, j].repeat_interleave(128, 1)[:, :n]
        fk = full_keep[:, :, j].repeat_interleave(128, 1)[:, :n]
        full_num += alpha[..., None]*mu
        control_num += alpha[..., None]*mu*ck[..., None]
        fullv_num += alpha[..., None]*mu*fk[..., None]
        full_den += alpha
        control_den += alpha*ck
        fullv_den += alpha*fk
    full_out = full_num / full_den.clamp_min(1e-12)[..., None]
    control_out = control_num / control_den.clamp_min(1e-12)[..., None]
    fullv_out = fullv_num / fullv_den.clamp_min(1e-12)[..., None]
    row_nu = nu[torch.arange(h, device=device)//(h//hk)][:, None]
    control_err = (control_out-full_out).norm(dim=-1)/row_nu.clamp_min(1e-12)
    fullv_err = (fullv_out-full_out).norm(dim=-1)/row_nu.clamp_min(1e-12)
    inter = (control_keep & full_keep).sum(-1).float()
    union = (control_keep | full_keep).sum(-1).float().clamp_min(1)
    # Projected V2 with the same Q/K mass and the repository's rank-32 map.
    matrix = Projections().get(int(raw['layer']), hk, d, 'gaussian', 32, 1729, device)
    sketch = torch.einsum('hkd,hdr->hkr', v.float(), matrix).view(1, hk, nk, 32)
    stats = statistics_cuda(q[None], k[None], sketch, nu, scale, pt)
    stats.log_mass[:, :pt].copy_(log_mass.view(h, qb, 128, pt).permute(0, 1, 3, 2).reshape(h*qb, pt, 128))
    projected, _ = select(stats, SELECTORS[1], budget=128, threshold=threshold,
                           mandatory=torch.zeros((h*qb, pt), dtype=torch.bool, device=device))
    projected = projected[0].bool()
    p = projected[:, :, :pt].reshape(h, qb, pt)
    pi = (p & full_keep).sum(-1).float()
    pu = (p | full_keep).sum(-1).float().clamp_min(1)
    result = dict(file=Path(path).name, age=int(raw['age']), phase=raw['phase'], layer=int(raw['layer']),
                  prefix_tiles=pt,
                  full_v_keep_tiles=float(full_keep.sum().item()/(h*qb)),
                  control_keep_tiles=float(control_keep.sum().item()/(h*qb)),
                  projected_keep_tiles=float(p.sum().item()/(h*qb)),
                  full_v_retained_mass=float(fullv_den.mean().item()),
                  control_error_mean=float(control_err.mean().item()),
                  control_error_max=float(control_err.max().item()),
                  full_v_error_mean=float(fullv_err.mean().item()),
                  full_v_error_max=float(fullv_err.max().item()),
                  full_v_vs_control_iou=float((inter/union).mean().item()),
                  projected_vs_full_v_iou=float((pi/pu).mean().item()))
    del raw, q, k, v, stats, matrix, sketch
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.027)
    a = p.parse_args()
    rows = [one(x, a.threshold) for x in a.snapshots]
    keys = ('control_error_mean','full_v_error_mean','full_v_vs_control_iou','projected_vs_full_v_iou')
    aggregate = {k: sum(r[k] for r in rows)/len(rows) for k in keys}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='offline_full_dim_v2_diagnostic_v1', threshold=a.threshold,
                                        snapshots=rows, aggregate=aggregate, status='complete'), indent=2)+'\n')
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
