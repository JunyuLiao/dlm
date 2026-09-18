"""Counterfactual projection distortion with identical full-dimensional support.

This is diagnostic-only: projected token values are reduced by block softmax,
then the trace follows the full-control mask. No final examples or outcomes
are read. Three fixed seeds are all retained, not selected by performance.
"""
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch
from experiments import diffusion_gemma_ruler8k_gaussian_sweep as study
from experiments.diffusion_gemma_jl_output_aware import kernels, reference
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_jl_output_aware.screen import sketches_for
from experiments.diffusion_gemma_jl_output_aware.trace_kernels import trace
from experiments.diffusion_gemma_jl_output_aware.shared_analysis import compare_traces

base = study.base


def common_support_metrics(projected, full, full_skip, log_threshold, prefix):
    out = compare_traces(projected, full, full_skip, log_threshold, prefix)
    active, supported = full['active'].bool(), full['supported'].bool()
    eligible = active.any(-2)
    # Only counterfactual votes; these do NOT replace the support used in trace.
    votes = (projected['centered'] < math.exp(log_threshold)) & supported
    skip = eligible & (votes | ~active).all(-2)
    positive = supported & (full['centered'] > 1e-12)
    ratios = projected['centered'][positive] / full['centered'][positive]
    true_positive = supported & (full['centered'] >= math.exp(log_threshold))
    out.update(counterfactual_skipped_tiles=int(skip.sum()),
        counterfactual_disagreement_tiles=int(((skip != full_skip) & eligible).sum()),
        counterfactual_unsafe_skip_tiles=int((skip & ~full_skip & eligible).sum()),
        full_above_threshold_rows=int(true_positive.sum()), positive_true_rows=int(positive.sum()),
        norm_ratio_sum=float(ratios.double().sum()), norm_ratio_sq_sum=float(ratios.double().square().sum()),
        norm_relative_error_sq_sum=float((ratios.double()-1.).square().sum()),
        factor_two_underestimate_rows=int((ratios < .5).sum()),
        factor_two_overestimate_rows=int((ratios > 2.).sum()),
        support_convention='Full-dimensional control retained support, identical for every rank and seed. Counterfactual votes never alter this support.',
        threshold_convention='Frozen full-dimensional threshold, not this rank calibrated threshold; isolates norm distortion in common units.')
    return out


def collect(root, sources, contract):
    selected = study.selected_sources(sources)
    bank, completed = Projections(), []
    code_sources = {str(p):base.sha(p.read_bytes()) for p in (Path(__file__), Path(study.__file__))}
    for key, source in selected.items():
        base.evidence.check_sources({source['path']:source['sha256']})
        data = torch.load(source['path'], weights_only=True, map_location='cuda')
        s, valid, v = data['scores'], data['valid'], data['values']
        full = kernels.block_statistics(s, valid, v.float())
        _, _, ref = sketches_for(data, Config(family='identity'), bank)
        supports = {}
        for target in base.TARGETS:
            p = root/'policies'/base.BENCHMARK/f'full_centered_s{int(100*target)}.json'
            policy = base.read(p)['policy'][source['attention_type']]
            c = Config(family='identity', log_threshold=policy['log_threshold'])
            masks, margin, _ = kernels.route(full, ref, c)
            skip = masks[0,:,0].unsqueeze(0)
            if float(margin.min()) < kernels.GUARD:
                skip = reference.route(reference.block_statistics(s, valid, v.float()), ref, c)
            supports[target] = (p, c, skip, trace(full, ref, skip, c))
        for rank in study.DIAGNOSTIC_RANKS:
            for seed in study.SEEDS:
                c = Config(family='gaussian', rank=rank, projection_seed=seed)
                sk, norm, _ = sketches_for(data, c, bank)
                st = kernels.block_statistics(s, valid, sk, norm)
                for target, (p, full_config, skip, truth) in supports.items():
                    ident = dict(fingerprint=contract['fingerprint'], source=source, task=key[1],
                        rank=rank, seed=seed, target=target, family='gaussian',
                        policy_path=str(p), policy_sha256=base.sha(p.read_bytes()), diagnostic_sources=code_sources)
                    dest = root/'common_support_diagnostics'/f'{Path(source["path"]).stem}_r{rank}_seed{seed}_s{int(target*100)}.json'
                    if dest.exists():
                        if base.read(dest)['identity'] != ident: raise ValueError('Common-support diagnostic identity changed')
                    else:
                        result = common_support_metrics(trace(st, ref, skip, c), truth, skip,
                            full_config.log_threshold, source['prefix'])
                        base._write(dest, dict(identity=ident, metrics=result))
                    completed.append(dict(path=str(dest), sha256=base.sha(dest.read_bytes())))
        print('common support diagnostics', key, flush=True)
    base.frozen_write(root/'common_support_projection_matrices.json', bank.manifest)
    base.frozen_write(root/'common_support_index.json', completed)
    return completed


def summarize(root, contract):
    """Recompute count-weighted diagnostics and verify all390×6×3×2 cells."""
    path = root/'common_support_index.json'
    if not path.exists(): return [], {}, [dict(error='Missing common-support diagnostic index')]
    sources = {str(path):base.sha(path.read_bytes())}; groups = defaultdict(list)
    expected = {(s['path'],r,seed,t) for s in study.selected_sources(base.read(root/'shared_state_index.json')).values()
                for r in study.DIAGNOSTIC_RANKS for seed in study.SEEDS for t in base.TARGETS}
    actual = set(); index = base.read(path)
    for item in index:
        p = Path(item['path']); base.evidence.check_sources({str(p):item['sha256']})
        data = base.read(p); d, m = data['identity'], data['metrics']
        source = d['source']; cell = (source['path'],d['rank'],d['seed'],d['target'])
        if d['fingerprint'] != contract['fingerprint'] or cell in actual: raise ValueError('Common-support identity/duplicate changed')
        actual.add(cell)
        srcs = {str(p):item['sha256'], source['path']:source['sha256'], d['policy_path']:d['policy_sha256'], **d['diagnostic_sources']}
        base.evidence.check_sources(srcs); sources.update(srcs)
        for axis, val in (('overall','all'), ('attention_type',source['attention_type']), ('task',d['task']),
                          ('layer',str(source['layer'])), ('head',str(source['head'])), ('step',str(source['step']))):
            groups[d['rank'],d['seed'],d['target'],axis,val].append(m)
    matrices_path = root/'common_support_projection_matrices.json'
    matrices = base.read(matrices_path); bank = Projections()
    for key, records in matrices.items():
        first = records[0]
        bank.get(first['layer'], len(records), first['value_width'], first['family'], first['rank'], first['seed'], 'cpu')
        if bank.manifest.get(key) != records: raise ValueError('Diagnostic projection matrix hash changed')
    sources[str(matrices_path)] = base.sha(matrices_path.read_bytes())
    output = []
    counts = ('eligible_tiles','skipped_tiles','valid_row_blocks','dangerous_underestimates','severe_underestimates',
        'counterfactual_skipped_tiles','counterfactual_disagreement_tiles','counterfactual_unsafe_skip_tiles',
        'full_above_threshold_rows','positive_true_rows','factor_two_underestimate_rows','factor_two_overestimate_rows')
    for (rank,seed,target,axis,val), rows in sorted(groups.items()):
        sums = {k:sum(r[k] for r in rows) for k in counts}
        n = max(sums['positive_true_rows'],1)
        corr = [r['spearman_risk'] for r in rows if r['spearman_risk'] is not None]
        types = {t:{k:sum(r['types'][t][k] for r in rows) for k in rows[0]['types'][t]} for t in rows[0]['types']}
        regions = {t:{k:sum(r['regions'][t][k] for r in rows) for k in rows[0]['regions'][t]} for t in rows[0]['regions']}
        output.append(dict(rank=rank, seed=seed, target=target, axis=axis, index=val, states=len(rows), **sums,
            mean_norm_ratio=sum(r['norm_ratio_sum'] for r in rows)/n,
            rms_relative_norm_error=math.sqrt(sum(r['norm_relative_error_sq_sum'] for r in rows)/n),
            factor_two_underestimate_rate=sums['factor_two_underestimate_rows']/n,
            dangerous_underestimate_rate=sums['dangerous_underestimates']/max(sums['full_above_threshold_rows'],1),
            tile_disagreement_rate=sums['counterfactual_disagreement_tiles']/max(sums['eligible_tiles'],1),
            median_state_spearman=float(np.median(corr)) if corr else None,
            norm_ratio_quantiles_by_state=[r['norm_ratio_quantiles'] for r in rows], types=types, regions=regions))
    violations = [] if actual == expected and len(index) == len(expected) else [dict(error='Common-support coverage differs', expected=len(expected), completed=len(actual))]
    return output, sources, violations
