"""Existing scalar-CDF/inverse-length search, with audited short calibration runs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.diffusion_gemma_value_aware.calibration import quantile_threshold
from experiments.diffusion_gemma_value_aware.rank_calibration import next_policy as rank_next
from experiments.diffusion_gemma_value_aware.policy_search import next_policy, select_point
from experiments.diffusion_gemma_value_aware.repair_original import scale_upper
from experiments.diffusion_gemma_value_aware_followup.policies import measurements, explicit_boundaries
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint
from experiments.diffusion_gemma_value_aware.run import shard_path
from .protocol import PARENT, LATEST
from .config import BASELINES, PROJECTED
from .runner import cached, load_output
from .screen import distributions

CALIBRATION_BUDGET = 512


def rows_for(setup, benchmark):
    rows = [dict(r, generation_budget=CALIBRATION_BUDGET,
                 original_generation_budget=r['generation_budget'])
            for r in setup['calibration'] if r['benchmark'] == benchmark]
    if len(rows) != 6 or any(r['split'] != 'calibration' for r in rows):
        raise ValueError('Exactly six existing calibration questions per benchmark are required')
    return rows


def audit_policy(root, policy, setup, contract):
    if policy['fingerprint'] != contract['fingerprint'] or policy['heldout_used']:
        raise ValueError('Policy execution/selection provenance changed')
    if policy.get('imported_aime'):
        check_sources(policy['sources'])
        previous = json.loads(Path(policy['imported_aime']).read_text())
        if previous['config'] != policy['config'] or previous['thresholds']['aime26'] != policy['policy']:
            raise ValueError('Imported AIME baseline operator or thresholds changed')
        return
    rows = rows_for(setup, policy['benchmark'])
    if set(policy['calibration_ids']) != {r['id'] for r in rows}:
        raise ValueError('Threshold fitting sample contamination')
    check_sources(policy['distribution_sources'])
    for point in policy['trace']:
        if set(point['sources']) != {r['id'] for r in rows}:
            raise ValueError('Incomplete calibration point')
        outputs = []
        for row in rows:
            source = point['sources'][row['id']]
            path = Path(source['path'])
            if sha(path.read_bytes()) != source['sha256']:
                raise ValueError('Raw calibration shard changed')
            out = load_output(path)
            check_result(out, row, contract['fingerprint'], policy['config'], point.get('source_policy', point['policy']))
            outputs.append(out)
        _, achieved = measurements(outputs, {'method': 'centered', **policy['config']})
        if any(abs(achieved[k]-point['achieved'][k]) > 1e-10 for k in achieved):
            raise ValueError('Calibration measurement disagrees with raw physical counts')
    selected = next(p for p in policy['trace'] if p['iteration'] == policy['selected_round'])
    if policy['policy'] != selected['policy'] or policy['measured'] != selected['achieved']:
        raise ValueError('Threshold did not come from the selected verified joint point')


def import_aime_baseline(root, name, target, config, setup, contract):
    if name not in BASELINES or name == 'unweighted_centered':
        return None
    source = PARENT/'final_configs'/f'{name}_s{int(100*target)}.json'
    old = json.loads(source.read_text())
    if old['config'] != config:
        raise ValueError('Historical AIME baseline config mismatch')
    check_sources(old['sources'])
    parent_policy_path = Path(old['policy_sources']['aime26']['path'])
    parent_policy = json.loads(parent_policy_path.read_text())
    if 'parent_policy' in parent_policy:
        parent_policy = parent_policy['parent_policy']
    result = dict(fingerprint=contract['fingerprint'], benchmark='aime26', name=name,
        config=config, target=target, policy=old['thresholds']['aime26'], heldout_used=False,
        measured=parent_policy.get('measured'), cap_one_unattainable=parent_policy.get('cap_one_unattainable'),
        imported_aime=str(source), sources={str(source):sha(source.read_bytes()), **old['sources']},
        calibration_ids=[r['id'] for r in setup['calibration'] if r['benchmark'] == 'aime26'],
        calibration_budget='Historical full2048-token calibration, unchanged operator; not refitted on final data',
        rule='Reuse already-verified AIME baseline policy and compatible final outputs')
    audit_policy(root, result, setup, contract)
    frozen_write(root/'policies/aime26'/f'{name}_s{int(100*target)}.json', result)
    return result


def fit_one(adapter, root, setup, contract, benchmark, name, config, target):
    dest = root/'policies'/benchmark/f'{name}_s{int(100*target)}.json'
    if dest.exists():
        policy = json.loads(dest.read_text()); audit_policy(root, policy, setup, contract); return policy
    if benchmark == 'aime26':
        imported = import_aime_baseline(root, name, target, config, setup, contract)
        if imported:
            return imported
    rows = rows_for(setup, benchmark)
    upper = scale_upper(rows)
    trace_path = root/'calibration_traces'/benchmark/f'{name}_s{int(100*target)}.json'
    observed = json.loads(trace_path.read_text()) if trace_path.exists() else []
    values, sources = (None, {}) if name.startswith('blasst_') else distributions(root, name, benchmark)
    if name.startswith('blasst_'):
        initial_source = LATEST/'final_configs'/f'{name}_s{int(100*target)}.json'
        sources[str(initial_source)] = sha(initial_source.read_bytes())
    if values is not None:
        # Actual zero risks remain -inf in raw distributions and routing. A
        # finite log-space floor makes the existing quantile/search serializer
        # usable; it does not change any routing score or infer a guarantee.
        values = {k:np.maximum(v, -1e30) for k, v in values.items()}
    if observed:
        policy = deepcopy(observed[-1]['policy'])
    elif name.startswith('blasst_'):
        source = LATEST/'final_configs'/f'{name}_s{int(100*target)}.json'
        previous = json.loads(source.read_text())
        policy = deepcopy(previous['thresholds']['longbench_v2'])
        # Historical saturation is only a warm start. This calibration must
        # independently observe lambda1 before declaring it unattainable.
        for entry in policy.values():
            entry.pop('unattainable', None)
            entry.pop('lambda_at_one', None)
        sources[str(source)] = sha(source.read_bytes())
    else:
        policy = {k:dict(log_threshold=quantile_threshold([v], target)['log_threshold']) for k, v in values.items()}
    for iteration in range(len(observed), 3):
        if observed:
            best, fixed = select_point(observed, target, name == 'blasst_original', upper)
            if all(fixed[k] or abs(best['achieved'][k]-target) <= .02 for k in ('local', 'global')):
                break
            policy = rank_next(observed, target, values) if values is not None else next_policy(observed, target, name == 'blasst_original', upper)
            if any(p['policy'] == policy for p in observed):
                break
        label = f'{name}_s{int(100*target)}/{benchmark}/{_fingerprint(policy)[:16]}'
        outputs, raw_sources = [], {}
        for row in rows:
            out = cached(adapter, root, row, 'calibration', label, name, config, policy, contract)
            path = shard_path(root, 'calibration', label, row['id'])
            outputs.append(out)
            raw_sources[row['id']] = dict(path=str(path), sha256=sha(path.read_bytes()))
        metrics, achieved = measurements(outputs, {'method': 'centered', **config})
        point = dict(iteration=iteration, policy=deepcopy(policy), achieved=achieved, metrics=metrics,
            sources=raw_sources, source_condition=label, source_ids=[r['id'] for r in rows])
        observed.append(point); _write(trace_path, observed)
        print('calibrated', benchmark, name, target, achieved, flush=True)
    if not observed:
        raise ValueError('No verified threshold point')
    best, fixed = select_point(observed, target, name == 'blasst_original', upper)
    best = explicit_boundaries(best, fixed, upper)
    observed = [best if p['iteration'] == best['iteration'] else p for p in observed]
    result = dict(fingerprint=contract['fingerprint'], benchmark=benchmark, name=name, config=config,
        target=target, policy=best['policy'], measured=best['achieved'], trace=observed,
        selected_round=best['iteration'], calibration_ids=[r['id'] for r in rows],
        calibration_budget=CALIBRATION_BUDGET, heldout_used=False,
        cap_one_unattainable=fixed if name == 'blasst_original' else None,
        within_two_points=all(abs(v-target) <= .02 for v in best['achieved'].values()),
        distribution_sources=sources,
        rule='Existing independent local/global empirical-CDF rank refinement or BLASST inverse-L search; at most3 verified joint points; no final retuning')
    audit_policy(root, result, setup, contract)
    frozen_write(dest, result)
    return result
