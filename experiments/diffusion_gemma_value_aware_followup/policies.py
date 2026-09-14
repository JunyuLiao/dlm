"""Calibration-only policy IO over the existing fit/search/physical counters.

Old AIME evidence is reusable only for an identical operator and the original
six prompts. LongBench-v1 policies never stand in for LongBench-v2 policies.
Every observed search point carries raw, hash-checked calibration provenance.
"""
import json
from pathlib import Path
import numpy as np

from experiments.diffusion_gemma_value_aware.calibration import aggregate
from experiments.diffusion_gemma_value_aware.policy_audit import point_for, verify_measurement
from experiments.diffusion_gemma_value_aware.policy_search import select_point, is_one
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware.refinements import CONFIGS as REFINEMENTS
from .analysis import screen_source
from .engine import check_result
from .execution import provenance
from .controls import CONFIG as CONTROL
from .protocol import PREVIOUS, TARGETS

BENCHMARKS = ('longbench_v2', 'aime26')
METHODS = ('blasst_original', 'blasst_aggressive', 'value', 'mass', 'mass_value',
    'risk', 'aligned', 'centered', 'compensate', 'zero_pv', 'mass_exact', 'no_value_control')
KINDS = ('local', 'global')


def candidate_configs(selection):
    configs = dict(blasst_original=dict(method='blasst'),
        blasst_aggressive=dict(method='blasst'), mass=dict(method='mass'),
        aligned=dict(method='aligned'), centered=dict(method='centered'),
        compensate=dict(method='compensate'), zero_pv=dict(method='zero_pv'),
        mass_exact=dict(REFINEMENTS['mass_exact']), no_value_control=dict(CONTROL))
    for name in ('value', 'mass_value', 'risk'):
        pool = selection['selected'][name]['pooling']
        if pool not in ('max', 'mean', 'rms', 'p95', 'vector_mean'):
            raise ValueError('unknown selected pooling')
        configs[name] = dict(method=name, pooling=pool)
    return configs


def calibration_rows(setup, benchmark):
    if benchmark not in BENCHMARKS:
        raise ValueError('unknown benchmark; v1 cannot substitute for v2')
    rows = [r for r in setup['calibration'] if r['benchmark'] == benchmark]
    if len(rows) != 6 or len({r['id'] for r in rows}) != 6:
        raise ValueError('expected six unique calibration examples')
    if any(r['split'] != 'calibration' for r in rows):
        raise ValueError('final/development examples cannot calibrate thresholds')
    return rows


def probe_key(name, config):
    if name.startswith('blasst_'):
        return 'blasst'
    return f'{name}_{config["pooling"]}' if name in ('value', 'mass_value', 'risk') else name


def inputs(root, execution):
    audit = json.loads((root/'initial_audit.json').read_text())
    analysis = json.loads((root/'screen_analysis_audit.json').read_text())
    proposals = json.loads((root/'threshold_proposals.json').read_text())
    selection = json.loads((root/'pooling_selection.json').read_text())
    for data in (audit, analysis, proposals, selection):
        if data['fingerprint'] != execution['fingerprint']:
            raise ValueError('calibration input execution mismatch')
    if not audit['complete'] or not analysis['complete']:
        raise ValueError('complete dense baseline and shared screen required')
    if analysis['heldout_used'] or selection['heldout_used']:
        raise ValueError('selection contaminated by final samples')
    if selection['source_sha256'] != sha((root/'screen_summary.json').read_bytes()):
        raise ValueError('pooling selection summary changed')
    for data in (proposals, selection):
        for path, expected in data['code_sources'].items():
            if sha(Path(path).read_bytes()) != expected:
                raise ValueError('calibration proposal source code changed')
    for source in proposals['calibration_sources'].values():
        if source['split'] != 'calibration':
            raise ValueError('proposal includes non-calibration sources')
        for key, digest_key in (('path', 'sha256'), ('arrays_path', 'arrays_sha256')):
            if sha(Path(source[key]).read_bytes()) != source[digest_key]:
                raise ValueError('calibration proposal raw source changed')
    return proposals, candidate_configs(selection)


def starting_policy(proposals, benchmark, name, config, target):
    if benchmark not in BENCHMARKS or target not in TARGETS:
        raise ValueError('unknown benchmark/target')
    by_kind = proposals['policies'][benchmark][probe_key(name, config)]
    if name.startswith('blasst_'):
        return {kind: dict(log_scale=by_kind[kind]['targets'][str(target)]['log_scale'],
            cap_one=name == 'blasst_original',
            source='existing exponential fit plus physical dense-margin correction; lambda=exp(log_scale)/valid_KV_length')
            for kind in KINDS}
    return {kind: dict(log_threshold=by_kind[kind][str(target)]['log_threshold'],
        dense_unattainable=by_kind[kind][str(target)]['unattainable'],
        source='independent calibration-only physical-risk quantile; scalar shared across layers/heads/steps')
        for kind in KINDS}


def distributions(root, rows, name, config, execution):
    if any(r['split'] != 'calibration' for r in rows):
        raise ValueError('only calibration risk distributions may select thresholds')
    if name.startswith('blasst_'):
        raise ValueError('BLASST keeps its existing inverse-length fit/search')
    imports = json.loads((root/'imported_sources.json').read_text())
    parts = {k: [] for k in KINDS}; sources = {}
    for row in rows:
        data, path, source = screen_source(root, row, execution, imports)
        if name != 'mass_exact':
            from .supplement import source as supplemental_source
            _, path, source = supplemental_source(root, row, execution, data)
        sources[row['id']] = source
        with np.load(path) as arrays:
            for kind in KINDS:
                parts[kind].append(arrays[f'{probe_key(name, config)}__{kind}'].astype(np.float64))
    values = {kind: np.sort(np.concatenate(parts[kind])) for kind in KINDS}
    if any(np.isnan(v).any() or not np.isfinite(v).any() for v in values.values()):
        raise ValueError('invalid calibration risk distribution')
    return values, dict(sources=sources, source_ids=[r['id'] for r in rows],
        rule='existing empirical-rank scalar-risk refinement; calibration only', heldout_used=False)


def measurements(outputs, config):
    records = [r for out in outputs for r in out['records'] if r['probe'] == 'execution']
    metrics = {kind: aggregate([r for r in records if kind == 'overall' or r['attention_type'] == kind])
        for kind in ('overall', *KINDS)}
    field = 'pv_omission' if config['method'] in ('compensate', 'zero_pv') else 'physical_sparsity'
    return metrics, {kind: metrics[kind][field] for kind in KINDS}


def audit_point(point, rows, config, execution):
    if any(r['split'] != 'calibration' for r in rows):
        raise ValueError('verification requires calibration rows')
    if set(point['source_ids']) != {r['id'] for r in rows}:
        raise ValueError('calibration point sample mismatch')
    if set(point['sources']) != set(point['source_ids']):
        raise ValueError('incomplete calibration raw sources')
    raw_policy = point.get('source_policy', point['policy'])
    if raw_policy != point['policy']:
        from experiments.diffusion_gemma_value_aware.repair_original import scale_upper
        upper=scale_upper(rows)
        if config['method']!='blasst' or not point.get('exact_lambda1_reexpression'):
            raise ValueError('unverified calibration threshold change')
        for kind in KINDS:
            if raw_policy[kind] == point['policy'][kind]:
                continue
            # This exception is only a representation change of an actually
            # saturated lambda1 run. It never combines independently fitted
            # types or changes any calibrated mask. The upper length includes
            # the prompt, full budget, and two native canvases.
            if not (is_one(raw_policy[kind],upper) and constant_one(point['policy'][kind])):
                raise ValueError('lambda1 reexpression lacks exact boundary evidence')
    outputs = []
    for row in rows:
        source = point['sources'][row['id']]
        if source['stage'] not in ('calibration', 'original_boundary_repair'):
            raise ValueError('cannot verify a policy using development/final outputs')
        fp = source['fingerprint']
        if fp != execution['fingerprint']:
            if (row['benchmark'] != 'aime26' or fp != execution['previous_fingerprint']
                    or not point.get('imported_policy')):
                raise ValueError('incompatible imported calibration source')
        path = Path(source['path']); raw = path.read_bytes()
        if sha(raw) != source['sha256']:
            raise ValueError('calibration raw source changed')
        out = json.loads(raw)
        check_result(out, row, fp, config, raw_policy)
        for key, expected in provenance(config).items():
            if out.get(key) != expected:
                raise ValueError('calibration raw operator source changed')
        outputs.append(out)
    metrics, achieved = measurements(outputs, config)
    if any(abs(achieved[k] - point['achieved'][k]) > 1e-8 for k in KINDS):
        raise ValueError('calibration sparsity disagrees with raw physical counts')
    return metrics


def constant_one(entry):
    return bool(entry.get('cap_one') and (entry.get('unattainable')
        or ('log_scale' not in entry and entry.get('log_threshold') == 0.)))


def explicit_boundaries(point, fixed, upper):
    """Use exact1 at every final length, not merely at calibration lengths."""
    from copy import deepcopy
    result=deepcopy(point)
    for kind, required in fixed.items():
        if not required or constant_one(result['policy'][kind]):
            continue
        if not is_one(result['policy'][kind],upper):
            raise ValueError('cannot assert an unmeasured lambda1 boundary')
        result.setdefault('source_policy',deepcopy(point['policy']))
        result['policy'][kind]=dict(log_threshold=0.,cap_one=True,unattainable=True,
            source='exact lambda1 fallback for this attention type only')
        result['exact_lambda1_reexpression']=dict(calibration_log_length_upper=upper,
            reason='same saturated calibration operator; explicit constant1 guarantees fallback for longer final inputs')
    return result


def import_aime_policy(root, rows, name, config, target, execution):
    """Import a verified point, not an old benchmark label or an accuracy claim."""
    if any(r['benchmark'] != 'aime26' for r in rows):
        return None
    old_path = PREVIOUS/'verified_policies'/'aime26'/f'{name}_s{round(target*100)}.json'
    if not old_path.exists():
        return None
    policy = json.loads(old_path.read_text())
    if policy['config'] != config:
        return None
    if (policy['name'], policy['benchmark'], policy['target'], policy['fingerprint']) != (
            name, 'aime26', target, execution['previous_fingerprint']):
        raise ValueError('old AIME policy identity mismatch')
    previous = json.loads((PREVIOUS/'setup.json').read_text())
    old_rows = {r['id']: r for r in previous['calibration'] if r['benchmark'] == 'aime26'}
    if set(old_rows) != {r['id'] for r in rows}:
        raise ValueError('old AIME calibration set changed')
    for row in rows:
        if any(row[k] != old_rows[row['id']][k] for k in
                ('prompt', 'prompt_hash', 'prompt_tokens', 'seed', 'generation_budget')):
            raise ValueError('old AIME calibration prompt/settings changed')
    selected = point_for(policy)
    stage = selected.get('verification_stage', policy.get('verification_stage', 'calibration'))
    if stage not in ('calibration', 'original_boundary_repair'):
        raise ValueError('old AIME policy uses non-calibration evidence')
    from experiments.diffusion_gemma_value_aware.run import shard_path
    outputs = []; sources = {}
    for row in rows:
        path = shard_path(PREVIOUS, stage, selected['source_condition'], row['id'])
        raw = path.read_bytes(); out = json.loads(raw)
        check_result(out, row, execution['previous_fingerprint'], config, out['thresholds'])
        outputs.append(out)
        sources[row['id']] = dict(path=str(path), sha256=sha(raw), stage=stage,
            fingerprint=execution['previous_fingerprint'])
    verify_measurement(policy, outputs, rows)
    # A historical lambda=1 repair can use an equivalent clearer representation.
    # Keep the actually executed representation so future audits need no exception.
    executed = outputs[0]['thresholds']
    if any(out['thresholds'] != executed for out in outputs):
        raise ValueError('one calibration group used multiple policies')
    metrics, achieved = measurements(outputs, config)
    point = dict(iteration=0, policy=executed, achieved=achieved, metrics=metrics,
        error=max(abs(v-target) for v in achieved.values()), sources=sources,
        source_ids=[r['id'] for r in rows], source_condition=selected['source_condition'],
        imported_policy=dict(path=str(old_path), sha256=sha(old_path.read_bytes()),
            original_selected_round=selected['iteration'], original_policy=policy['policy']))
    audit_point(point, rows, config, execution)
    return point


def publish(root, benchmark, name, config, target, observations, rows, execution, upper, search_source):
    for point in observations:
        audit_point(point, rows, config, execution)
    best, fixed = select_point(observations, target, name == 'blasst_original', upper)
    best=explicit_boundaries(best,fixed,upper)
    observations=[best if p['iteration']==best['iteration'] else p for p in observations]
    audit_point(best,rows,config,execution)
    policy = dict(fingerprint=execution['fingerprint'], benchmark=benchmark, name=name,
        config=config, target=target, policy=best['policy'], measured=best['achieved'],
        error=best['error'], within_two_points=best['error'] <= .02,
        trace=observations, selected_round=best['iteration'],
        cap_one_unattainable=fixed if name == 'blasst_original' else None,
        unattained_in_tested_range={k: v < target-.02 for k, v in best['achieved'].items()},
        calibration_ids=[r['id'] for r in rows], heldout_used=False,
        search_version='followup_empirical_rank_v1', search_provenance=search_source,
        metric='pv_omission' if config['method'] in ('compensate', 'zero_pv') else 'physical_sparsity',
        **provenance(config))
    frozen_write(root/'verified_policies'/benchmark/f'{name}_s{round(target*100)}.json', policy)
    return policy


def audit_policy(policy, rows, config, execution):
    if policy['fingerprint'] != execution['fingerprint'] or policy['config'] != config:
        raise ValueError('policy execution/config mismatch')
    if policy['heldout_used'] or set(policy['calibration_ids']) != {r['id'] for r in rows}:
        raise ValueError('policy calibration sample contamination')
    for key, expected in provenance(config).items():
        if policy.get(key) != expected:
            raise ValueError('policy operator source changed')
    point = point_for(policy)
    metrics = audit_point(point, rows, config, execution)
    if policy['measured'] != point['achieved']:
        raise ValueError('policy measurement changed')
    if abs(policy['error']-max(abs(v-policy['target']) for v in point['achieved'].values()))>1e-8:
        raise ValueError('policy calibration error changed')
    if policy['name'] == 'blasst_original':
        from experiments.diffusion_gemma_value_aware.policy_search import is_one
        from experiments.diffusion_gemma_value_aware.repair_original import scale_upper
        if not all(e.get('cap_one') for e in policy['policy'].values()):
            raise ValueError('original BLASST must cap lambda at one')
        for kind, fixed in (policy.get('cap_one_unattainable') or {}).items():
            if fixed and not constant_one(policy['policy'][kind]):
                raise ValueError('unattainable original BLASST must use lambda=1')
    return metrics
