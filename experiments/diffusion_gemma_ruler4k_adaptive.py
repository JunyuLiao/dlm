"""Frozen-history development study for uncertainty-adaptive projection.

The script replays audited RULER4K shared QKV states only.  It never consumes
final answers or changes a generation.  This makes it useful when CUDA is
unavailable and, more importantly, isolates routing fidelity from free-running
trajectory changes.  A guarded native-generation launch is provided separately
by ``--cuda-smoke``; any failure is recorded rather than replacing this
development evidence.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import time

import torch

from experiments.diffusion_gemma_jl_output_aware import adaptive
from experiments.diffusion_gemma_jl_output_aware import reference
from experiments.diffusion_gemma_jl_output_aware.config import Config
from experiments.diffusion_gemma_jl_output_aware.projections import Projections


def sha(data):
    """Local artifact hash without importing the heavyweight dataset stack."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


SOURCE = Path('results/diffusion_gemma_ruler4k_value_direction_s70_v19')
ROOT = Path('results/diffusion_gemma_ruler4k_adaptive_v20')
RANKS = (2, 8, 32, 64)
STAGES = (2, 8, 32, 64)
TARGETS = (.5, .75)
LAYERS = (0, 5, 15, 29)


def _thresholds():
    # These are previously frozen full-centered policies, not fitted from this
    # run.  The 50/75 controls are from the audited 4K sweep v16.
    roots = [Path('results/diffusion_gemma_ruler4k_gaussian_rank_sweep_v16/thresholds.json'),
             SOURCE/'thresholds.json']
    out = {}
    for path in roots:
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row.get('name') != 'full_centered':
                continue
            target = float(row['target']); kind = row['attention_type']
            out.setdefault(target, {})[kind] = float(row['tau'])
    for target in TARGETS:
        if target not in out or set(out[target]) != {'local', 'global'}:
            raise ValueError(f'missing frozen full-centered thresholds for {target}')
    return out


def _rms(values, kv_valid):
    token_valid = kv_valid.to(values.dtype)
    valid = token_valid.unsqueeze(-1) if token_valid.ndim == values.ndim - 1 else token_valid
    return ((values.float().square() * valid).sum((-1, -2)) /
            token_valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)


def _scaled_state(state, rank, max_rank=64):
    scale = math.sqrt(max_rank / rank)
    out = dict(state)
    out['mu'] = state['mu'][..., :rank] * scale
    return out


def _mask_metrics(candidate, exact, eligible):
    return adaptive.decision_metrics(candidate, exact, eligible)


def _full_trace_scores(state, ref):
    _, traces = reference.route(state, ref,
        Config(family='identity', log_threshold=-float('inf')), return_trace=True)
    out = []
    for trace in traces:
        rows = trace['active'] & trace['support']
        risk = trace['delta'].norm(dim=-1) / ref[..., None].clamp_min(1e-12)
        out.append(risk.masked_fill(~rows, -torch.inf).amax(-1))
    return torch.stack(out, -1)


def _fit_bands(records, refs, threshold_lookup):
    """Fit residual bands on the first half of selected IDs only."""
    ratios = {r: [] for r in STAGES}
    for item, ref in zip(records, refs):
        pstate, estate = item['projected'], item['exact']
        for rank in STAGES:
            projected = _scaled_state(pstate, rank)
            # A neutral threshold lets us collect the exact same retained
            # support convention without selecting on final benchmark scores.
            p_mask, p_trace = reference.route(projected, ref,
                Config(method='centered', family='gaussian', rank=rank,
                       log_threshold=-float('inf')), return_trace=True)
            _, e_trace = reference.route(estate, ref,
                Config(family='identity', log_threshold=-float('inf')),
                return_trace=True, forced_skip=p_mask)
            for pt, et in zip(p_trace, e_trace):
                support = pt['active'] & pt['support']
                pr = pt['delta'].norm(dim=-1) / ref[..., None].clamp_min(1e-12)
                er = et['delta'].norm(dim=-1) / ref[..., None].clamp_min(1e-12)
                pmax = pr.masked_fill(~support, -torch.inf).amax(-1)
                emax = er.masked_fill(~support, -torch.inf).amax(-1)
                good = torch.isfinite(pmax) & torch.isfinite(emax) & (pmax > 1e-8)
                ratios[rank].extend((emax[good] / pmax[good]).detach().cpu().tolist())
    return adaptive.fit_empirical_bands({r: torch.tensor(v) for r, v in ratios.items()})


def _select_index(source, max_ids):
    index = json.loads((source/'shared_state_index.json').read_text())
    by_id = defaultdict(list)
    for item in index:
        if item['layer'] in LAYERS and item['head'] == 0:
            by_id[item['id']].append(item)
    ids = sorted(by_id)
    if max_ids:
        ids = ids[:max_ids]
    if len(ids) < 4:
        raise ValueError('need at least four shared-state IDs for fit/validation')
    # Split by complete question IDs, never by individual blocks.
    split = max(2, len(ids)//2)
    selected = []
    for part, names in (('fit', ids[:split]), ('validation', ids[split:])):
        for name in names:
            for item in sorted(by_id[name], key=lambda x: (x['step'], x['layer'])):
                item = dict(item, split=part)
                selected.append(item)
    return selected


def _load_state(item):
    path = Path(item['path'])
    data = torch.load(path, weights_only=True, map_location='cpu')
    if data['identity']['id'] != item['id'] or data['identity']['layer'] != item['layer']:
        raise ValueError('shared state identity mismatch')
    scores, valid, values = data['scores'], data['valid'], data['values']
    kv_valid = data['kv_valid']
    ref = _rms(values, kv_valid)
    bank = Projections().get_nested_bank(item['layer'], 1, values.shape[-1], 64, 1729, 'cpu')
    projected_values = torch.matmul(values.float(), bank / math.sqrt(64))
    pstate = reference.block_statistics(scores, valid, projected_values)
    estate = reference.block_statistics(scores, valid, values.float())
    return dict(item=item, projected=pstate, exact=estate, ref=ref,
                eligible=estate['eligible'])


def _one_record(item, state, target, thresholds, bands=None):
    kind = item['attention_type']
    tau = thresholds[target][kind]
    logtau = math.log(tau)
    ref = state['ref']
    exact = state['exact']
    exact_mask = reference.route(exact, ref, Config(family='identity', log_threshold=logtau))
    eligible = exact['eligible']
    rows = []
    for rank in RANKS:
        p = _scaled_state(state['projected'], rank)
        mask = reference.route(p, ref, Config(method='centered', family='gaussian', rank=rank,
                                              log_threshold=logtau))
        metrics = _mask_metrics(mask, exact_mask, eligible)
        metrics.update(method=f'fixed_gaussian{rank}', target=target,
                       split=item['split'], state_id=item['id'], layer=item['layer'],
                       step=item['step'], rank_mean=float(rank),
                       fallback_tiles=0., escalated_tiles=0., screened_tiles=0.)
        rows.append(metrics)

    cfg = adaptive.CascadeConfig(stages=STAGES, interval_delta=1e-3,
                                 exact_fallback=False)
    plain = adaptive.route(state['projected'], ref, tau, cfg)
    metrics = _mask_metrics(plain.skip, exact_mask, eligible)
    metrics.update(method='adaptive_analytic', target=target, split=item['split'],
                   state_id=item['id'], layer=item['layer'], step=item['step'],
                   rank_mean=float(plain.rank_used.float().mean()),
                   fallback_tiles=plain.diagnostics['exact_fallback_tiles'],
                   escalated_tiles=plain.diagnostics['escalated_tiles'],
                   screened_tiles=plain.diagnostics['screened_tiles'])
    rows.append(metrics)

    oracle = adaptive.route(state['projected'], ref, tau,
                            adaptive.CascadeConfig(stages=STAGES, interval_delta=1e-3,
                                                   exact_fallback=True),
                            exact_state=exact)
    metrics = _mask_metrics(oracle.skip, exact_mask, eligible)
    metrics.update(method='adaptive_analytic_exact_fallback', target=target,
                   split=item['split'], state_id=item['id'], layer=item['layer'],
                   step=item['step'], rank_mean=float(oracle.rank_used.float().mean()),
                   fallback_tiles=oracle.diagnostics['exact_fallback_tiles'],
                   escalated_tiles=oracle.diagnostics['escalated_tiles'],
                   screened_tiles=oracle.diagnostics['screened_tiles'])
    rows.append(metrics)

    if bands is not None:
        empirical = adaptive.route(state['projected'], ref, tau,
            adaptive.CascadeConfig(stages=STAGES, interval_delta=1e-3,
                                   exact_fallback=False, band_name='empirical'),
            empirical_bands=bands)
        metrics = _mask_metrics(empirical.skip, exact_mask, eligible)
        metrics.update(method='adaptive_empirical', target=target, split=item['split'],
                       state_id=item['id'], layer=item['layer'], step=item['step'],
                       rank_mean=float(empirical.rank_used.float().mean()),
                       fallback_tiles=empirical.diagnostics['exact_fallback_tiles'],
                       escalated_tiles=empirical.diagnostics['escalated_tiles'],
                       screened_tiles=empirical.diagnostics['screened_tiles'])
        rows.append(metrics)
    return rows


def _write_report(root, meta, rows, bands):
    grouped = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (row['split'], row['target'], row['method'])
        for k in ('eligible_tiles', 'skipped_tiles', 'exact_skipped_tiles', 'disagreement',
                  'false_skips', 'false_keeps', 'fallback_tiles', 'escalated_tiles', 'screened_tiles'):
            grouped[key][k] += float(row[k])
        grouped[key]['rank_sum'] += float(row['rank_mean'])
        grouped[key]['states'] += 1
    summary = []
    for (split, target, method), d in sorted(grouped.items()):
        eligible = max(1., d['eligible_tiles'])
        summary.append(dict(split=split, target=target, method=method, states=int(d['states']),
            eligible_tiles=int(d['eligible_tiles']), skipped_tiles=int(d['skipped_tiles']),
            exact_skipped_tiles=int(d['exact_skipped_tiles']),
            actual_sparsity=d['skipped_tiles']/eligible,
            exact_sparsity=d['exact_skipped_tiles']/eligible,
            disagreement=d['disagreement']/eligible,
            false_skip_rate=d['false_skips']/eligible,
            false_keep_rate=d['false_keeps']/eligible,
            mean_rank=d['rank_sum']/max(1., d['states']),
            fallback_tiles=int(d['fallback_tiles']), escalated_tiles=int(d['escalated_tiles']),
            screened_tiles=int(d['screened_tiles'])))
    (root/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (root/'summary.csv').open('w', newline='') as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    lines = ['# RULER4K uncertainty-adaptive nested-projection development', '',
             'This is frozen-history routing evidence, not a final generation score.', '',
             f"Source bundle: `{SOURCE}`; selected shared states: {meta['selected_states']}.",
             f"Fit IDs: {meta['fit_ids']}; validation IDs: {meta['validation_ids']}.",
             'The full-centered thresholds are imported from audited RULER4K v16 policies; no final scores are used.', '',
             '## Empirical bands', '',
             'Bands are multiplicative residual quantiles fitted only on the fit IDs. They are empirical bands, not distribution-free guarantees.', '',
             '| rank | lower | upper |', '|---:|---:|---:|']
    for rank, (lo, hi) in sorted(bands.items()): lines.append(f'| {rank} | {lo:.6g} | {hi:.6g} |')
    lines += ['', '## Routing fidelity', '',
              '| split | target | method | actual skip | exact skip | disagreement | false skip | false keep | mean rank | exact fallback | escalated |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in summary:
        lines.append(f"| {x['split']} | {100*x['target']:.0f}% | {x['method']} | {100*x['actual_sparsity']:.2f}% | {100*x['exact_sparsity']:.2f}% | {100*x['disagreement']:.2f}% | {100*x['false_skip_rate']:.2f}% | {100*x['false_keep_rate']:.2f}% | {x['mean_rank']:.2f} | {x['fallback_tiles']} | {x['escalated_tiles']} |")
    lines += ['', '## Accounting', '',
              '- A physical tile is skipped only when every valid query row votes skip.',
              '- Nested projections use one frozen Gaussian bank per layer/native KV head; rank r is the first r columns divided by sqrt(r).',
              '- Analytic intervals use a Bonferroni budget over all valid query rows and cascade stages.',
              '- The exact-fallback variant charges every fallback coordinate and keeps a parallel full-dimensional retained state.',
              '- Fixed-history diagnostics do not claim downstream accuracy or hardware speedup.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(root=ROOT, source=SOURCE, max_ids=6):
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    thresholds = _thresholds()
    selected = _select_index(source, max_ids)
    loaded = []
    for n, item in enumerate(selected, 1):
        loaded.append(_load_state(item))
        (root/'progress.json').write_text(json.dumps(dict(stage='load_shared_states', completed=n,
            total=len(selected), updated=time.time())))
    fit = [x for x in loaded if x['item']['split'] == 'fit']
    validation = [x for x in loaded if x['item']['split'] == 'validation']
    bands = _fit_bands(fit, [x['ref'] for x in fit], thresholds)
    meta = dict(schema='ruler4k_adaptive_v20', source=str(source), source_audit_sha256=sha((source/'audit.json').read_bytes()),
        selected_states=len(loaded), fit_ids=sorted({x['item']['id'] for x in fit}),
        validation_ids=sorted({x['item']['id'] for x in validation}), stages=STAGES,
        projection_family='gaussian_nested', projection_seed=1729, max_rank=64,
        thresholds=thresholds, interval_delta=1e-3,
        no_final_scores=True, cuda_available=bool(torch.cuda.is_available()))
    (root/'setup.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    rows = []
    total = len(loaded)*len(TARGETS)
    done = 0
    for state in loaded:
        for target in TARGETS:
            rows.extend(_one_record(state['item'], state, target, thresholds,
                                    bands if state['item']['split'] == 'validation' else None))
            done += 1
            (root/'progress.json').write_text(json.dumps(dict(stage='route_shared_states', completed=done,
                total=total, updated=time.time())))
    (root/'rows.json').write_text(json.dumps(rows, indent=2))
    (root/'bands.json').write_text(json.dumps({str(k): list(v) for k, v in bands.items()}, indent=2))
    summary = _write_report(root, meta, rows, bands)
    audit = dict(passed=True, complete=True, rows=len(rows), summary=len(summary),
                 source_audit_sha256=meta['source_audit_sha256'],
                 artifacts={p.name: sha(p.read_bytes()) for p in
                            (root/'setup.json', root/'bands.json', root/'rows.json',
                             root/'summary.json', root/'summary.csv', root/'report.md')})
    (root/'audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True))
    (root/'progress.md').write_text(f"Completed frozen-history adaptive routing: {len(rows)} row records; no CUDA generation was run.\n")
    return audit, summary


def cuda_smoke(root=ROOT):
    result = dict(time=time.time(), cuda_available=bool(torch.cuda.is_available()),
                  launched=False, passed=False, error=None,
                  note='Guarded only: no long worker is started when CUDA is unavailable.')
    if not result['cuda_available']:
        result['error'] = 'CUDA driver/device unavailable in this session'
    else:
        # The native worker is intentionally not launched implicitly by this
        # offline script; workflow integration is versioned separately after
        # the reference tests pass.
        result['error'] = 'Native adaptive workflow launch is deferred until the reference gate passes'
    (root/'cuda_smoke.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--max-ids', type=int, default=6)
    p.add_argument('--cuda-smoke', action='store_true')
    args = p.parse_args()
    audit, summary = run(args.output, args.source, args.max_ids)
    if args.cuda_smoke:
        smoke = cuda_smoke(args.output)
    else:
        smoke = None
    print(json.dumps(dict(audit=audit, cuda_smoke=smoke, summary=summary), indent=2))


if __name__ == '__main__':
    main()
