"""Bounded, identical-state mechanism probes; never changes native attention.

Online row masks here intentionally remain legacy counterfactual probes.
Their whole-tile closures represent the corrected execution semantics.
One declared head and 64-query group per prompt, every decoder layer/call.
All query rows of that head are used only for query-union probes. These are
diagnostics on dense trajectories, not deployment policies or speed estimates.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dllm.attention.blasst.core import _prepare_attention_scores, _attention_type, _expand_valid_mask
from dllm.evaluation.ruler.io import read_jsonl, write_json
from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation
from experiments.diffusion_attention_threshold_modeling.proxy import _region_proxies
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.runner import _study_rows, _runner_manifest
from .runner import BASE, ROOT, MODEL, REVISION, RULER

BETAS = {25: -.6744897501960817, 50: 0., 75: .6744897501960817, 90: 1.2815515655446004}
LAMBDAS = (.001, .1, .5, .9, .99, .9995, .9999, 1.)


def online_mask(maxima, valid, lam=1., order="forward"):
    """Return retained row/tile positions and previous-max margins."""
    count = maxima.shape[-1]
    indices = torch.arange(count, device=maxima.device)
    if order == "reverse":
        indices = indices.flip(0)
    elif order == "random":
        indices = torch.randperm(count, generator=torch.Generator().manual_seed(42)).to(maxima.device)
    elif order == "finalmax":
        margins = maxima - maxima.max(-1, keepdim=True).values
        return valid & (margins >= math.log(lam)), margins
    elif order != "forward":
        raise ValueError(order)
    ordered = maxima[..., indices]
    previous = F.pad(ordered.cummax(-1).values[..., :-1], (1, 0), value=-torch.inf)
    margins = (ordered - previous)[..., indices.argsort()]
    return valid & (margins >= math.log(lam)), margins


def gaussian_mask(proxy, eligible, beta):
    selected = proxy[eligible]
    degenerate = len(selected) < 2 or float(selected.std(unbiased=False)) < 1.e-6
    if degenerate:
        return eligible.clone(), True, False
    z = (proxy - selected.mean()) / selected.std(unbiased=False).clamp_min(1.e-6)
    keep = eligible & (z >= beta)
    fallback = not bool(keep.any())
    if fallback:
        keep[proxy.masked_fill(~eligible, -torch.inf).argmax()] = True
    return keep, False, fallback


def group_counts(keep, valid, size=64, order=None):
    """Count eligible and retained physical tiles for a fixed row-level mask."""
    if order is not None:
        keep, valid = keep[order], valid[order]
    padding = (-keep.shape[0]) % size
    keep = F.pad(keep, (0, 0, 0, padding), value=False)
    valid = F.pad(valid, (0, 0, 0, padding), value=False)
    retained = keep.reshape(-1, size, keep.shape[-1]).any(1)
    eligible = valid.reshape(-1, size, valid.shape[-1]).any(1)
    return {"eligible_tiles": int(eligible.sum()), "skipped_tiles": int((eligible & ~retained).sum())}


def mask_metrics(keep, counts, mass, contributions):
    valid = counts > 0
    eligible = valid.any(0)
    physical = keep.any(0)
    retained_mass = (mass * keep).sum(-1)
    active = valid.any(-1)
    dense = contributions.sum(1)
    routed = (contributions * keep[..., None]).sum(1) / retained_mass[:, None].clamp_min(1.e-30)
    error = (routed - dense).square().sum(-1)
    norm = dense.square().sum(-1)
    return dict(eligible_tiles=int(eligible.sum()), skipped_tiles=int((eligible & ~physical).sum()),
                valid_elements=int(counts.sum()), skipped_elements=int((counts * ~keep).sum()),
                retained_mass_sum=float(retained_mass[active].sum()), valid_rows=int(active.sum()),
                output_squared_error=float(error[active].sum()), dense_squared_norm=float(norm[active].sum()),
                empty_rows=int((active & ~keep.any(-1)).sum()))


def select_rows():
    rows = []
    for split in ("calibration", "final"):
        study, source = _study_rows(BASE / "manifest.json", split)
        per_task = defaultdict(int)
        for row in source:
            if row['task'] != 'vt' and per_task[row['task']] < 2:
                rows.append({**row, "diagnostic_split": split})
                per_task[row['task']] += 1
    assert len(rows) == 16
    assert len({r['sample_id'] for r in rows}) == len({r['prompt_hash'] for r in rows}) == 16
    return study, rows


class Observer:
    def __init__(self, directory, rows):
        self.directory = directory
        self.indices = {r['sample_id']: i for i, r in enumerate(rows)}
        self.split = {r['sample_id']: r['diagnostic_split'] for r in rows}
        self.example = None
        self.dirty = False
        self.records = []
        self.arrays = {}
        self.calls = defaultdict(int)
        self.previous = {}
        self.previous_layer = {}
        self.frequency = {}
        policy=json.loads((BASE/'calibration/blasst_policy.json').read_text())
        self.deployed_lambdas={typ:{int(round(r['target_sparsity']*100)):r['lambda'] for r in policy[typ]} for typ in ('local','global')}

    def flush(self):
        if self.example is None or not self.dirty:
            return
        out = self.directory / "shards" / self.example
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'snapshots.npz', **self.arrays)
        write_json(out / 'diagnostics.json', self.records)
        self.dirty = False

    @torch.no_grad()
    def __call__(self, module, query, key, value, attention_mask=None, **kwargs):
        runtime = module._blasst_2d_runtime
        runtime.attention_output_guard = True
        metadata = dict(runtime.metadata)
        example = metadata['example_id']
        if example != self.example:
            self.flush()
            self.example = example
            self.records, self.arrays = [], {}
            self.calls, self.previous, self.previous_layer, self.frequency = defaultdict(int), {}, {}, {}
        layer = int(module.layer_idx)
        call = self.calls[layer]
        self.calls[layer] += 1
        head = self.indices[example] % query.shape[1]
        start = (self.indices[example] % max(1, math.ceil(query.shape[-2] / 64))) * 64
        end = min(start + 64, query.shape[-2])
        expanded_key, expanded_value, scores, valid = _prepare_attention_scores(
            module, query, key, value, attention_mask, scaling=kwargs.get('scaling'),
            is_causal=kwargs.get('is_causal'), sliding_window=kwargs.get('sliding_window'))
        valid = _expand_valid_mask(valid, scores)
        # Same native BF16 QK path as the deployed eager sparse reference.
        selected_scores = scores[0, head, start:end].float()
        selected_valid = valid[0, head, start:end]
        length = scores.shape[-1]
        pad = (-length) % 64
        count = (length + pad) // 64
        blocks = F.pad(selected_scores, (0, pad), value=-torch.inf).reshape(end-start, count, 64)
        counts = F.pad(selected_valid, (0, pad), value=False).reshape(end-start, count, 64).sum(-1)
        maxima = blocks.max(-1).values
        valid_tiles = counts > 0
        probabilities = selected_scores.softmax(-1)
        probabilities = torch.where(selected_valid.any(-1, keepdim=True), probabilities, 0.)
        pblocks = F.pad(probabilities, (0, pad)).reshape(end-start, count, 64)
        mass = pblocks.sum(-1)
        values = F.pad(expanded_value[0, head].float(), (0, 0, 0, pad)).reshape(count, 64, -1)
        contributions = torch.einsum('qtk,tkd->qtd', pblocks, values)
        scale = float(kwargs.get('scaling') or query.shape[-1] ** -.5)
        proxy, eligible, _ = _region_proxies(
            query[:, head:head+1], expanded_key[:, head:head+1], selected_valid[None,None],
            q_start=start, q_end=end, kv_start=0, kv_end=length, kv_block_size=64, scaling=scale)
        proxy, eligible = proxy[0,0], eligible[0,0]
        precise = query[0,head,start:end].float() @ expanded_key[0,head].float().T * scale
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            expanded_mask = attention_mask.expand_as(scores)
            precise += expanded_mask[0,head,start:end,:length]
        precise.masked_fill_(~selected_valid, -torch.inf)
        precise_max = F.pad(precise,(0,pad),value=-torch.inf).reshape(end-start,count,64).max(-1).values
        typ = _attention_type(module, kwargs.get('sliding_window'))
        record = dict(example_id=example, split=self.split[example], layer=layer, call=call, head=head,
                      query_start=start, query_length=query.shape[-2], kv_length=length,
                      attention_type=typ, metadata=metadata, policies={}, reuse={}, grouping={})
        masks = {}
        for order in ('forward','reverse','random','finalmax'):
            for lam in LAMBDAS:
                name = f'{order}_{lam:g}'
                masks[name], margins = online_mask(maxima, valid_tiles, lam, order)
                if lam == 1.:
                    physical_margin = margins.masked_fill(~valid_tiles,-torch.inf).max(0).values
                    record[name+'_records'] = dict(strict=int((eligible & (physical_margin>0)).sum()),
                                                   tie_only=int((eligible & (physical_margin==0)).sum()))
        for lam in (.9995, 1.):
            masks[f'fp32_{lam:g}'], _ = online_mask(precise_max, valid_tiles, lam)
            original = masks[f'forward_{lam:g}']
            masks[f'closure_{lam:g}'] = original.any(0)[None,:] & valid_tiles
        for target,lam in self.deployed_lambdas[typ].items():
            masks[f'deployed_s{target}'],_=online_mask(maxima,valid_tiles,lam)
        for target,beta in BETAS.items():
            keep,degenerate,fallback = gaussian_mask(proxy,eligible,beta)
            masks[f'sol_{target}'] = keep[None,:] & valid_tiles
            record[f'sol_{target}_degenerate'] = int(degenerate)
            record[f'sol_{target}_fallback'] = int(fallback)
        k = int(masks['sol_50'].any(0).sum())
        _, margins = online_mask(maxima,valid_tiles)
        for name,rank in (('matched_blasst_rank',margins.masked_fill(~valid_tiles,-torch.inf).max(0).values),
                          ('matched_mass_oracle',mass.sum(0))):
            keep = torch.zeros_like(eligible)
            keep[rank.masked_fill(~eligible,-torch.inf).topk(k).indices] = True
            masks[name] = keep[None,:] & valid_tiles
        for name,keep in masks.items():
            record['policies'][name] = mask_metrics(keep,counts,mass,contributions)
        # Prefix/canvas mass counts split by tokens; a boundary-straddling tile
        # is reported separately, avoiding double counting physical tiles.
        prefix = max(0,length-query.shape[-2])
        region = torch.arange(count,device=scores.device)*64
        regions = {'prefix': region+64<=prefix, 'canvas':region>=prefix,
                   'boundary':(region<prefix)&(region+64>prefix)}
        for name in ('forward_1','reverse_1','sol_50'):
            keep=masks[name]
            record['policies'][name]['regions'] = {
                r: dict(eligible_tiles=int((eligible & sel).sum()), skipped_tiles=int((eligible & sel & ~keep.any(0)).sum()))
                for r,sel in regions.items()}
        full_scores=scores[0,head].float()
        full_valid=valid[0,head]
        full_max=F.pad(full_scores,(0,pad),value=-torch.inf).reshape(query.shape[-2],count,64).max(-1).values
        full_eligible=F.pad(full_valid,(0,pad),value=False).reshape(query.shape[-2],count,64).any(-1)
        for lam in (.9995,1.):
            full_keep,_=online_mask(full_max,full_eligible,lam)
            groups={str(size):group_counts(full_keep,full_eligible,size) for size in (1,8,16,32,64,128,256)}
            packed=np.packbits(full_keep.cpu().numpy(),axis=-1)
            lex=np.lexsort(packed[:,::-1].T)
            groups['oracle_lex64']=group_counts(full_keep,full_eligible,64,torch.as_tensor(lex,device=scores.device))
            perm=torch.randperm(full_keep.shape[0],generator=torch.Generator().manual_seed(42)).to(scores.device)
            groups['random64']=group_counts(full_keep,full_eligible,64,perm)
            record['grouping'][str(lam)]=groups
        window=(layer,length,start,end)
        for name in ('forward_1','sol_50'):
            now=masks[name].any(0)
            for kind,cache,cachekey in (
                ('step',self.previous,(window,name)),
                ('layer',self.previous_layer,(typ,call,length,start,end,name))):
                if cachekey in cache:
                    old,old_valid=cache[cachekey]
                    common=eligible & old_valid
                    union=(now|old)&common
                    intersection=(now&old)&common
                    reused=old[None,:] & valid_tiles
                    record['reuse'][kind+'_'+name]=dict(intersection=int(intersection.sum()), union=int(union.sum()),
                        eligible=int(common.sum()), previous_retained=int((old&common).sum()), current_retained=int((now&common).sum()),
                        reused_mass_sum=float((mass*reused).sum()), fresh_mass_sum=float((mass*now[None,:]).sum()), rows=int(valid_tiles.any(-1).sum()))
                cache[cachekey]=(now.clone(),eligible.clone())
            fkey=(window,name)
            frequency,n=self.frequency.get(fkey,(torch.zeros_like(proxy),0))
            if n:
                top=torch.zeros_like(eligible)
                top[frequency.masked_fill(~eligible,-torch.inf).topk(int(now.sum())).indices]=True
                record['reuse']['history_'+name]=dict(prior_calls=n, matched_retained_tiles=int(top.sum()),
                    reused_mass_sum=float((mass*top[None,:]).sum()), fresh_mass_sum=float((mass*now[None,:]).sum()), rows=int(valid_tiles.any(-1).sum()))
            self.frequency[fkey]=(frequency+now,n+1)
        keyname=f'l{layer}_c{call}'
        for name,tensor in dict(maxima=maxima,precise_maxima=precise_max,counts=counts,mass=mass,proxy=proxy).items():
            self.arrays[keyname+'_'+name]=tensor.cpu().numpy()
        record['snapshot_key']=keyname
        self.records.append(record)
        self.dirty = True


def collect(root: Path=ROOT):
    torch.backends.cuda.matmul.allow_tf32=False
    study,rows=select_rows()
    out=root/'same_state'
    out.mkdir(parents=True,exist_ok=True)
    manifest=_runner_manifest(study,rows,out/'runner_manifest.json',adapter='diffusion_gemma')
    write_json(out/'protocol.json',dict(head='sample_index modulo 16', query_tile='sample_index modulo 4',
        trajectories='native dense', policy_tuning=False, output_errors='FP32 probability/value diagnostic, not BF16 kernel error',
        lambdas=LAMBDAS,betas=BETAS,samples=[r['sample_id'] for r in rows]))
    observer=Observer(out,rows)
    config=RulerRunConfig(model_adapter='diffusion_gemma',model_path=MODEL,revision=REVISION,
        manifest_path=str(manifest),ruler_root=RULER,output_dir=str(out),num_samples=len(rows),
        context_length=16384,attention_backend='eager-dense',temperature=0.,precision='bfloat16',progress_every=1)
    try:
        summary=run_evaluation(config,attention_observer=observer,before_prediction_commit=observer.flush)
    finally:
        observer.flush()
    original={r['sample_id']:r for r in read_jsonl(BASE/'dense/predictions.jsonl')}
    final=[r for r in read_jsonl(out/'predictions.jsonl') if r['sample_id'] in original]
    parity={r['sample_id']:r['completion_tokens']==original[r['sample_id']]['completion_tokens'] for r in final}
    write_json(out/'native_dense_parity.json',parity)
    if not all(parity.values()) or len(parity)!=8:
        raise RuntimeError('native observer trajectory parity failed')
    return summary
