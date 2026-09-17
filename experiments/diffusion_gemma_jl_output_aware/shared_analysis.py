"""Uniform shared-QKV diagnostics for all candidates at their frozen thresholds."""
import json
import math
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.diffusion_gemma_value_aware.operators import Config as LegacyConfig, value_summaries, streaming_mask, block_state
from experiments.diffusion_gemma_value_aware_gpu.kernels import route as legacy_route, GUARD as LEGACY_GUARD
from experiments.diffusion_gemma_value_aware.protocol import sha, frozen_write
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .config import Config, PROJECTED, BASELINES, TARGETS
from .projections import Projections
from .screen import sketches_for, legacy_state
from .diagnostics import TYPE_RULES
from . import kernels, reference
from .trace_kernels import trace


def type_flags(data):
    support = data['supported'].bool()
    return {key:val & support for key, val in dict(low_attention=data['alpha'] < .01,
        redundant=(data['distance'] < .1) & (data['alpha'] >= .1),
        distinctive=(data['distance'] > .5) & (data['centered'] >= .05),
        internally_cancelling=data['kappa'] < .25).items()}


def compare_traces(projected, full, skip, threshold, prefix):
    supported = full['supported'].bool()
    active = full['active'].bool()
    eligible = active.any(-2)
    chosen = skip[..., None, :]
    pr, fr = projected['risk'], full['centered']
    cutoff = math.exp(threshold)
    dangerous = (pr < cutoff) & (fr >= cutoff) & supported
    true_types, sk_types = type_flags(full), type_flags(projected)
    out = dict(valid_row_blocks=int(supported.sum()), dangerous_underestimates=int(dangerous.sum()),
        severe_underestimates=int(((projected['centered'] < .5*fr) & (fr >= .05) & supported).sum()),
        eligible_tiles=int(eligible.sum()), skipped_tiles=int((skip & eligible).sum()), types={}, regions={})
    votes = (pr < cutoff) & supported
    out['mixed_query_tiles'] = int((votes.any(-2) & (~votes & supported).any(-2)).sum())
    full_votes = (fr < cutoff) & supported
    full_skip = eligible & (full_votes | ~active).all(-2)
    out['matched_full_threshold_disagreement'] = int(((full_skip != skip) & eligible).sum())
    for name, flag in true_types.items():
        any_type = flag.any(-2)
        out['types'][name] = dict(row_blocks=int(flag.sum()), skipped_row_blocks=int((flag & chosen).sum()),
            any_query_tiles=int(any_type.sum()), mixed_query_tiles=int((any_type & (~flag & supported).any(-2)).sum()),
            skipped_tiles=int((any_type & skip).sum()),
            projected_classification_disagreements=int(((flag != sk_types[name]) & supported).sum()),
            full_update_sq_sum=float(fr.square().masked_fill(~flag, 0.).sum()),
            skipped_full_update_sq_sum=float(fr.square().masked_fill(~(flag & chosen), 0.).sum()))
    starts = torch.arange(skip.shape[-1], device=skip.device)*64
    for name, region in zip(('prefix', 'canvas', 'boundary'),
            (starts+64 <= prefix, starts >= prefix, (starts < prefix) & (starts+64 > prefix))):
        out['regions'][name] = dict(eligible_tiles=int((eligible & region).sum()),
            skipped_tiles=int((skip & region).sum()), valid_row_blocks=int((supported & region).sum()),
            dangerous_underestimates=int((dangerous & region).sum()),
            full_update_sq_sum=float(fr.square().masked_fill(~(supported & region), 0.).sum()),
            skipped_full_update_sq_sum=float(fr.square().masked_fill(~(supported & region & chosen), 0.).sum()))
    # Fixed lattice within the supported population, never selected by error.
    pp, ff = pr[supported][::16].cpu().numpy(), fr[supported][::16].cpu().numpy()
    cp = projected['centered'][supported][::16].cpu().numpy()
    positive = ff > 1e-12
    ratios = cp[positive]/ff[positive]
    out.update(norm_ratio_quantiles=np.quantile(ratios, [.01,.1,.5,.9,.99]).tolist() if len(ratios) else [],
        sampled_projected_risk=pp.tolist(), sampled_full_risk=ff.tolist(),
        pearson_risk=float(np.corrcoef(pp, ff)[0,1]) if len(pp)>1 and pp.std() and ff.std() else None,
        spearman_risk=float(spearmanr(pp, ff).statistic) if len(pp)>1 and pp.std() and ff.std() else None,
        zero_true_update_rows=int(((fr <= 1e-12) & supported).sum()), type_rules=TYPE_RULES,
        contributions='Squared immediate centered updates; not additive final-output-error attribution')
    return out


def baseline_mask(state, values, kv_valid, name, config, policy, native_state=None):
    logt = policy['log_scale']-math.log(int(kv_valid.sum())) if 'log_scale' in policy else policy['log_threshold']
    if policy.get('cap_one'):
        logt = min(0., logt)
    if policy.get('unattainable') and policy.get('cap_one'):
        logt = 0.
    c = LegacyConfig(**config, log_threshold=logt)
    meta = value_summaries(values, kv_valid)
    if name == 'unweighted_centered':
        return streaming_mask(native_state if native_state is not None else legacy_state(state), meta, c)
    if name in ('mass', 'risk') and native_state is not None:
        state = native_state
    q = state['logz'].shape[-2]
    stats = [torch.nn.functional.pad(state[key].transpose(-1,-2).reshape(1,1,-1,q),
                (0,128-q), value=0 if key=='count' else -torch.inf).contiguous()
             for key in ('b','logz','count','b')]
    selected, margin = legacy_route(stats, c, meta, logt)
    if name in ('mass', 'risk') and bool((margin < LEGACY_GUARD).any()):
        return streaming_mask(state, meta, c)
    return selected[0,0].reshape(1,1,-1)


def selected_sources(sources):
    selected = {}
    for src in sorted(sources, key=lambda s:(s['id'],s['layer'],s['step'])):
        if src['step'] != 0 and src['layer'] not in (0,5):
            continue
        key = (src['id'].split('/')[0], src['layer'], src['step'])
        selected.setdefault(key, src)
    return selected


def analyze(root, sources, contract):
    """One source per layer/benchmark at step0, plus selected later steps.

    All final candidates are kept. Shared-state diagnostics do not select
    projection matrices, methods or final thresholds.
    """
    selected = selected_sources(sources)
    bank = Projections()
    completed = []
    for key, source in selected.items():
        path = Path(source['path'])
        if sha(path.read_bytes()) != source['sha256']:
            raise ValueError('Shared diagnostic source changed')
        data = torch.load(path, weights_only=True, map_location='cuda')
        s, valid, v = data['scores'], data['valid'], data['values']
        full = kernels.block_statistics(s, valid, v.float())
        native_state = block_state(s, valid, v)
        has = valid.any(-1)
        probs = torch.softmax(torch.where(has[...,None], s.float().masked_fill(~valid,-torch.inf), 0.), -1).masked_fill(~valid,0.)
        dense_out = probs@v.float()
        for name, cfg in {**BASELINES, **PROJECTED}.items():
            for target in TARGETS:
                policy_path = root/'policies'/key[0]/f'{name}_s{int(target*100)}.json'
                if not policy_path.exists():
                    continue  # Failed policy remains an explicit workflow failure.
                policy = json.loads(policy_path.read_text())['policy'][source['attention_type']]
                ident = dict(fingerprint=contract['fingerprint'], source=source,
                    policy_path=str(policy_path), policy_sha256=sha(policy_path.read_bytes()), name=name, target=target,
                    diagnostic_sources={str(p):sha(p.read_bytes()) for p in (Path(__file__),Path(__file__).with_name('trace_kernels.py'))})
                dest = root/'shared_diagnostics'/f'{path.stem}_{name}_s{int(target*100)}.json'
                if dest.exists():
                    old = json.loads(dest.read_text())
                    if old['identity'] != ident:
                        raise ValueError('Shared diagnostic identity changed')
                    completed.append(dict(path=str(dest),sha256=sha(dest.read_bytes()))); continue
                if name in PROJECTED:
                    c = Config(**cfg, log_threshold=policy['log_threshold'])
                    sk, norm, ref = sketches_for(data, c, bank)
                    state = full if c.family == 'identity' else kernels.block_statistics(s,valid,sk,norm)
                    masks, margin, _ = kernels.route(state,ref,c)
                    skip = masks[0,:,0].unsqueeze(0)
                    # Match the deployed numerical boundary guard exactly.
                    if float(margin.min()) < kernels.GUARD:
                        trusted = reference.block_statistics(s, valid, sk, norm)
                        skip = reference.route(trusted, ref, c)
                    projected = trace(state,ref,skip,c)
                else:
                    ref = value_summaries(v,data['kv_valid'])['ref']
                    skip = baseline_mask(full,v,data['kv_valid'],name,cfg,policy,native_state)
                    c = Config(family='identity',log_threshold=-1000.)
                    projected = trace(full,ref,skip,c)
                truth = trace(full,ref,skip,Config(family='identity'))
                result = compare_traces(projected,truth,skip,c.log_threshold,source['prefix'])
                if name in BASELINES:
                    for field in ('dangerous_underestimates','severe_underestimates','matched_full_threshold_disagreement','pearson_risk','spearman_risk'):
                        result[field] = None
                    result['risk_comparison_scope'] = 'Not applicable: baseline threshold units differ; full-dimensional type/keep/error accounting only'
                positions = skip.repeat_interleave(64,-1)[...,:s.shape[-1]][...,None,:]
                kept = probs.masked_fill(positions,0.)
                mass = kept.sum(-1)
                sparse = (kept/mass.clamp_min(1e-30)[...,None])@v.float()
                result.update(error_sq=float((sparse-dense_out).square().sum()), dense_sq=float(dense_out.square().sum()),
                    retained_mass_sum=float(mass.sum()),valid_query_rows=int(has.sum()),
                    support_convention='Each full-dimensional replay follows this candidate actual retained support on shared QKV')
                _write(dest,dict(identity=ident,metrics=result))
                completed.append(dict(path=str(dest),sha256=sha(dest.read_bytes())))
        print('shared diagnostics',key,flush=True)
    frozen_write(root/'shared_diagnostics_index.json',completed)
    return completed
