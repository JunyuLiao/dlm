"""Full-dimensional diagnostics on private frozen Q/K/V snapshots.

This separate process performs attention evaluations, never model forwards.
Its outputs cannot become selector inputs or affect generation trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.numerical_qk_reuse.v31_value_projection import refresh
from experiments.numerical_qk_reuse import v27_fa4
from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda
from experiments.numerical_qk_reuse.v31_value_selectors import (
    SELECTORS, full_support, mandatory_map, masked_output, select)


def error(output, reference, nu):
    # Outputs are native [1,N,H,D]. nu is a current KV-head reference norm.
    h = output.shape[2]
    scale = nu.repeat_interleave(h//nu.numel())
    e = (output.float()-reference.float()).norm(dim=-1)/scale[None, None].clamp_min(1e-12)
    return dict(max_row=float(e.max()), mean_row=float(e.mean()),
                rms_row=float(e.square().mean().sqrt()))


def overlap(a, b):
    intersection, union = (a & b).sum(), (a | b).sum()
    return float(intersection/union.clamp_min(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots', required=True)
    p.add_argument('--thresholds', required=True, help='frozen calibration JSON with thresholds keyed by arm')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('new diagnostic output required')
    thresholds = json.loads(Path(args.thresholds).read_text())['thresholds']
    if set(thresholds) != set(SELECTORS[:2]):
        raise ValueError('both independently calibrated thresholds required')
    bank = Projections()
    reports = []
    start = time.perf_counter()
    for source in sorted(Path(args.snapshots).glob('*.pt')):
        snap = torch.load(source, map_location='cpu', weights_only=True)
        q, k, v, held = (snap[key].cuda() for key in ('q', 'k', 'v', 'kept'))
        hk, nk, d = v.shape[1:]
        matrix = bank.get(snap['layer'], hk, d, 'gaussian', 32, 1729, q.device)
        sketch = torch.empty((1,hk,nk,32), device='cuda')
        norms = torch.empty((1,hk,nk), device='cuda')
        valid = torch.ones((1,hk,nk), dtype=torch.bool, device='cuda')
        nu = refresh(v, matrix, sketch, norms, valid, 0)[0]
        stats = statistics_cuda(q, k, sketch, nu, snap['scale'], snap['prefix']//64)
        observation=torch.empty((q.shape[1],stats.blocks,stats.prefix_tiles,128),
                                device=q.device,dtype=torch.float32)
        dense=observe_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                            snap['scale'],observation)
        stats.log_mass[:,:stats.prefix_tiles].view_as(observation).copy_(observation)
        mandatory = mandatory_map(stats)
        alpha, c, full_sketch, _ = full_support(stats)
        reused = v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(held),snap['scale'])
        if not torch.isfinite(dense).all() or not torch.isfinite(reused).all():
            raise ValueError('non-finite diagnostic output')
        pt = stats.prefix_tiles
        flat_held = held.reshape(-1,held.shape[-1])
        used_sketch = masked_output(alpha,c,flat_held)
        used_mass = (alpha*flat_held[...,None]).sum(1)
        record = dict(snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            phase=snap['phase'], distance_from_refresh=snap['age'], layer=snap['layer'],
            prefix_tokens=snap['prefix'], generating_selector=snap['selector'],
            reused_full_dimensional_error=error(reused,dense,nu),
            reused_retained_mass=dict(min_row=float(used_mass.min()),mean_row=float(used_mass.mean())),
            reused_sketch_max_error=float(((used_sketch-full_sketch).norm(dim=-1)/stats.nu).max()),
            fresh={})
        masks = {}
        sticky = 0. if snap['age']==0 else 1.386
        fresh_held = None if snap['age']==0 else flat_held
        # Independent mass-only formula from the recorded source note.
        log_share = stats.log_mass[:,:pt]-torch.logsumexp(stats.log_mass[:,:pt],1,keepdim=True)
        mass_score = log_share.amax(-1)+sticky*flat_held[:,:pt]
        control = mandatory.clone()
        control[:,:pt].scatter_(1,mass_score.topk(min(128,pt),dim=1).indices,True)
        masks['current_v31_control'] = control.reshape_as(held)
        for name in SELECTORS:
            masks[name], _ = select(stats,name,budget=128,threshold=thresholds.get(name),
                                    mandatory=mandatory,held=fresh_held,sticky=sticky)
        exact = masks[SELECTORS[3]]
        for name, mask in masks.items():
            fresh = v27_fa4.sparse_lists(q,k,v,v27_fa4.block_sparse_tensors(mask),snap['scale'])
            flat = mask.reshape_as(flat_held)
            mass = (alpha*flat[...,None]).sum(1)
            projected = masked_output(alpha,c,flat)
            record['fresh'][name] = dict(full_dimensional_error=error(fresh,dense,nu),
                fresh_vs_reused_error=error(fresh,reused,nu),mask_iou_with_reused=overlap(mask,held),
                mask_iou_with_exact=overlap(mask,exact),
                retained_mass=dict(min_row=float(mass.min()),mean_row=float(mass.mean())),
                sketch_max_error=float(((projected-full_sketch).norm(dim=-1)/stats.nu).max()),
                kept_prefix_tiles=int(mask[...,:pt].sum()))
        reports.append(record)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(dict(status='partial',diagnostics=reports),indent=2)+'\n')
        del q,k,v,held,stats,alpha,c,sketch,norms,masks,dense,reused,observation
    if not reports:
        raise ValueError('no snapshots')
    out = dict(status='complete',kind='sampled offline diagnostic; not accuracy or clean timing',
        operator='unchanged contiguous FA4 masked-attention operator; model alias2/paged merge rounding not included',
        scope='sampled layer 5, first canvas, first two long audit requests per shard; not all layers or calls',
        fresh_rule='initial age0 uses no sticky; held ages simulate refresh with inherited sticky',
        mass_source='native FA4 prefix observation, same as production selectors',
        thresholds_sha256=hashlib.sha256(Path(args.thresholds).read_bytes()).hexdigest(),
        projection_manifest=bank.manifest,gpu_seconds=time.perf_counter()-start,diagnostics=reports)
    destination.write_text(json.dumps(out,indent=2)+'\n')
    destination.with_suffix('.complete.json').write_text(json.dumps(dict(status='complete',snapshots=len(reports)))+'\n')


if __name__ == '__main__':
    main()
