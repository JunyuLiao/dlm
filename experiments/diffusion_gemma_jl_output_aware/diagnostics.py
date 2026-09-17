"""Explicitly sampled shared-state diagnostics, never inputs to deployed routing."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.diffusion_gemma_value_aware.protocol import sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from . import reference
from .config import Config

TYPE_RULES = dict(low_attention='online alpha <0.01',
    redundant='centered mean distance/reference <0.1 and online alpha >=0.1',
    distinctive='centered mean distance/reference >0.5 and centered update/reference >=0.05',
    internally_cancelling='norm(weighted mean)/(weighted mean token norm+1e-12) <0.25')


def types(trace, mean_norm, ref, width=None):
    mu, previous = trace['mu'], trace['previous']
    if width is not None:
        mu, previous = mu[..., :width], previous[..., :width]
    distance = (mu-previous).norm(dim=-1)/ref[..., None]
    kappa = mu.norm(dim=-1)/(mean_norm+1e-12)
    alpha = trace['alpha']
    valid = trace['active'] & trace['support']
    return {k: v & valid for k, v in dict(low_attention=alpha < .01,
        redundant=(distance < .1) & (alpha >= .1),
        distinctive=(distance > .5) & (alpha*distance >= .05),
        internally_cancelling=kappa < .25).items()}


def paired_risk_diagnostics(scores, valid, values, sketches, ref, config, skip=None, prefix=None, full_state=None):
    """Same QKV AND retained support for projected versus full-dimensional risk.

    Full PV is confined to this diagnostic. The candidate decisions are formed
    before full-dimensional information is computed, then replayed as a fixed
    support sequence for both measurements. This measures adaptive-prefix
    underestimation, not an independent full-control trajectory comparison.
    """
    primary_width = config.rank if config.method == 'cancellation_guard' else sketches.shape[-1]
    small = reference.block_statistics(scores, valid, sketches,
        sketches[..., :primary_width].norm(dim=-1))
    if skip is None:
        skip = reference.route(small, ref, config)
    _, projected = reference.route(small, ref, config, True, forced_skip=skip)
    full = full_state if full_state is not None else reference.block_statistics(scores, valid, values.float())
    _, exact = reference.route(full, ref, Config(family='identity'), True, forced_skip=skip)
    output = dict(valid_row_blocks=0, dangerous_underestimates=0, severe_underestimates=0,
        valid_tiles=0, matched_full_threshold_disagreement=0, mixed_query_tiles=0,
        types={k:dict(row_blocks=0, skipped_row_blocks=0, any_query_tiles=0,
            mixed_query_tiles=0, skipped_tiles=0, projected_classification_disagreements=0,
            full_update_sq_sum=0., skipped_full_update_sq_sum=0.) for k in TYPE_RULES},
        ratios=[], projected_risks=[], full_risks=[], zero_true_update_rows=0,
        by_region={region:dict(eligible_tiles=0, skipped_tiles=0, valid_row_blocks=0,
            dangerous_underestimates=0, full_update_sq_sum=0., skipped_full_update_sq_sum=0.)
            for region in ('prefix', 'boundary', 'canvas')})
    prefix = scores.shape[-1] if prefix is None else prefix
    for j, (p, f) in enumerate(zip(projected, exact)):
        active = f['active'] & f['support']
        pr = p['log_risk'].exp()
        fr = f['delta'].norm(dim=-1)/ref[..., None]
        tau = np.exp(config.log_threshold)
        output['valid_row_blocks'] += int(active.sum())
        output['dangerous_underestimates'] += int(((pr < tau) & (fr >= tau) & active).sum())
        output['severe_underestimates'] += int(((pr < .5*fr) & (fr >= .05) & active).sum())
        eligible = full['eligible'][..., j]
        full_skip = eligible & (f['log_risk'].amax(-1) < config.log_threshold)
        output['valid_tiles'] += int(eligible.sum())
        region = 'prefix' if (j+1)*64 <= prefix else 'canvas' if j*64 >= prefix else 'boundary'
        rd = output['by_region'][region]
        rd['eligible_tiles'] += int(eligible.sum())
        rd['skipped_tiles'] += int((skip[..., j] & eligible).sum())
        rd['valid_row_blocks'] += int(active.sum())
        rd['dangerous_underestimates'] += int(((pr < tau) & (fr >= tau) & active).sum())
        rd['full_update_sq_sum'] += float(fr.square().masked_fill(~active, 0.).sum())
        rd['skipped_full_update_sq_sum'] += float(fr.square().masked_fill(~(active & skip[..., j, None]), 0.).sum())
        output['matched_full_threshold_disagreement'] += int(((full_skip != skip[..., j]) & eligible).sum())
        row_votes = (pr < tau) & active
        mixed = row_votes.any(-1) & (~row_votes & active).any(-1)
        output['mixed_query_tiles'] += int(mixed.sum())
        projected_types = types(p, small['mean_norm'][..., j], ref, primary_width)
        true_types = types(f, full['mean_norm'][..., j], ref)
        for kind, flag in true_types.items():
            d = output['types'][kind]
            any_type = flag.any(-1)
            d['row_blocks'] += int(flag.sum())
            d['skipped_row_blocks'] += int((flag & skip[..., j, None]).sum())
            d['any_query_tiles'] += int(any_type.sum())
            d['mixed_query_tiles'] += int((any_type & (~flag & active).any(-1)).sum())
            d['skipped_tiles'] += int((any_type & skip[..., j]).sum())
            d['projected_classification_disagreements'] += int(((projected_types[kind] != flag) & active).sum())
            d['full_update_sq_sum'] += float(fr.square().masked_fill(~flag, 0.).sum())
            d['skipped_full_update_sq_sum'] += float(fr.square().masked_fill(~(flag & skip[..., j, None]), 0.).sum())
        # Deterministic per-tile lattice, not tail- or score-selected examples.
        pp, ff = pr[active].flatten()[::16], fr[active].flatten()[::16]
        output['projected_risks'].extend(pp.cpu().tolist())
        output['full_risks'].extend(ff.cpu().tolist())
        centered = p['delta'][..., :primary_width].norm(dim=-1)/ref[..., None]
        cp = centered[active].flatten()[::16]
        output['ratios'].extend((cp[ff > 1e-12]/ff[ff > 1e-12]).cpu().tolist())
        output['zero_true_update_rows'] += int(((fr <= 1e-12) & active).sum())
    for field in ('ratios', 'projected_risks', 'full_risks'):
        values_ = np.array(output[field], dtype=float)
        output[field+'_quantiles'] = np.quantile(values_, [.01, .1, .5, .9, .99]).tolist() if len(values_) else []
    if len(output['projected_risks']) > 1:
        a, b = np.array(output['projected_risks']), np.array(output['full_risks'])
        output['pearson_risk'] = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
        # Stable ranks, ties averaged: scipy is already in the environment.
        from scipy.stats import spearmanr
        output['spearman_risk'] = float(spearmanr(a, b).statistic) if a.std() and b.std() else None
    positions = skip.repeat_interleave(64, -1)[..., :scores.shape[-1]][..., None, :]
    valid = valid.expand_as(scores)
    has = valid.any(-1)
    dense_p = scores.float().masked_fill(~valid, -torch.inf)
    dense_p = torch.softmax(torch.where(has[..., None], dense_p, 0.), -1).masked_fill(~valid, 0.)
    retained = dense_p.masked_fill(positions, 0.)
    mass = retained.sum(-1)
    dense_o = dense_p@values.float()
    sparse_o = (retained/mass.clamp_min(1e-30)[..., None])@values.float()
    output.update(error_sq=float((sparse_o-dense_o).square().sum()), dense_sq=float(dense_o.square().sum()),
        retained_mass_sum=float(mass.sum()), valid_query_rows=int(has.sum()),
        eligible_tiles=int(full['eligible'].sum()), skipped_tiles=int(skip.sum()),
        support_convention='Full-dimensional risk replay uses the exact candidate retained-mask history',
        distortion_convention='Projected/full centered-update norm ratio, excluding zero full updates; policy-risk agreement separately uses the actual candidate score',
        error_contributions='Sums of squared immediate centered updates, not additive final-output error',
        type_rules=TYPE_RULES)
    return output


class SnapshotObserver:
    """Bounded, prespecified sample from an otherwise unchanged dense trajectory.

    One full128-query tile and one query head per sampled call, all KV positions.
    All layers at step0; steps4/12/24 additionally on the first two calibration
    examples. Head=(layer+example_index+step)%16; tile alternates with layer.
    Native rounded scores, original V and validity are sufficient to replay
    attention selection/output; never store an all-sequence square matrix.
    """
    def __init__(self, root, row, index):
        self.root, self.row, self.index = root, row, index
        self.seen, self.sources = set(), []

    def capture(self, layer, step, kind, prefix, scores, valid, values, native_heads):
        if step != 0 and not (self.index < 2 and step in (4, 12, 24)):
            return
        if (layer, step) in self.seen:
            return
        self.seen.add((layer, step))
        head = (layer+self.index+step)%scores.shape[1]
        qb = (layer+self.index)%((scores.shape[-2]+127)//128)
        start = qb*128
        name = sha(f'{self.row["id"]}/{layer}/{step}/{head}/{qb}')
        path = self.root/'shared_states'/self.row['benchmark']/(name+'.pt')
        identity = dict(id=self.row['id'], split=self.row['split'], prompt_hash=self.row['prompt_hash'],
            layer=layer, step=step, head=head, native_kv_head=head//(scores.shape[1]//native_heads),
            native_head_count=native_heads, query_tile=qb, attention_type=kind, prefix=prefix)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(identity=identity, scores=scores[:, head:head+1, start:start+128].cpu().contiguous(),
                valid=valid.expand_as(scores)[:, head:head+1, start:start+128].cpu().contiguous(),
                kv_valid=valid.expand_as(scores)[:, head:head+1].any(-2).cpu().contiguous(),
                values=values[:, head:head+1].cpu().contiguous())
            temp = path.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(path)
        self.sources.append(dict(path=str(path), sha256=sha(path.read_bytes()), **identity))
