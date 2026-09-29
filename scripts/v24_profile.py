"""v24 direct cost of the six frozen bootstrap6 arms on common native states.

Complete ``model.forward(...).logits`` and whole native denoising step, timed on
captured native sequences from canvas start (N16 reaches B0, BO, D4, D7, A9,
D12 when the canvas naturally gets that far). Reuses v20's replay, warmup,
native brackets, rotation and no-JIT guards unchanged; this module only builds
the arm set from a freshly bound v21 profile config (same builders as the
bootstrap6 panel binder) and adds the two bootstrap phases to per-call counter
deltas. A fixed-state replay is a timing diagnostic, not a scored method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts import v20_profile as base

SCOPE = 'GLOBAL_ONLY_NATIVE_LOCAL'
PLUGIN = 'experiments.numerical_qk_reuse.v21:install'
ARMS = ('D_native', 'T_scope', 'M3_R3_A8_incumbent', 'M1_native_bootstrap2_observe1',
        'M3_native_bootstrap2_observe1', 'B_native_bootstrap2_observe1')
METHODS = {'M3_R3_A8_incumbent': ('M3_R3_A8_current_output', None),
           'M1_native_bootstrap2_observe1': ('M1_R1_A8_current_output', 'native_bootstrap2_observe1'),
           'M3_native_bootstrap2_observe1': ('M3_R3_A8_current_output', 'native_bootstrap2_observe1'),
           'B_native_bootstrap2_observe1': ('B_A8_matched', 'native_bootstrap2_observe1')}
ARMS_V25 = ('D_native', 'M3_native_bootstrap2_observe1', 'M3_boot_aligned16',
            'B_native_bootstrap2_observe1', 'B_boot_aligned16')
METHODS.update({'M3_boot_aligned16': ('M3_R3_A8_current_output', 'native_bootstrap2_observe1'),
                'B_boot_aligned16': ('B_A8_matched', 'native_bootstrap2_observe1')})
STORAGE = {'M3_boot_aligned16': 'aligned16', 'B_boot_aligned16': 'aligned16'}
# v27 direct cost: named A/R/B/M2/threshold points on the same bootstrap mainline.
# name -> (parent arm, v21 extra kwargs). All use grouped-Q, FP32 current scores,
# model-major output and aligned16_odd storage like the v26 seven-arm panel.
BOOT = 'native_bootstrap2_observe1'
V27 = {'M1_R1_A8': ('M1_R1_A8_current_output', {}),
       'M2_pool_R1_A8': ('M1_R1_A8_current_output', dict(mu_mode='pooled')),
       'M2c_pool_R1_A8': ('M1_R1_A8_current_output', dict(mu_mode='pooled_compact')),
       'M3_R3_A8': ('M3_R3_A8_current_output', {}),
       'M3_R6_A8': ('M3_R3_A8_current_output', dict(decision_interval=6)),
       'M3_R3_A16': ('M3_R3_A8_current_output', dict(score_period=16)),
       'M3_R6_A16': ('M3_R3_A8_current_output', dict(score_period=16, decision_interval=6)),
       'M3_R3_A64': ('M3_R3_A8_current_output', dict(score_period=64)),
       'B_A8': ('B_A8_matched', dict(hold_only=True)),
       'B_A16': ('B_A8_matched', dict(hold_only=True, score_period=16)),
       'B_A64': ('B_A8_matched', dict(hold_only=True, score_period=64))}
V27['M3_R3_A64_G8k'] = ('M3_R3_A8_current_output', dict(score_period=64, min_route_keys=8192))
V27['M3_R6_A64'] = ('M3_R3_A8_current_output', dict(score_period=64, decision_interval=6))
ARMS_V27 = ('D_native', 'T_scope') + tuple(V27)
SHIFTS = ('minus_ln2', None, 'plus_ln2', 'plus_2ln2', 'plus_3ln2', 'plus_4ln2')
for _shift in SHIFTS:
    _label = 'P0' if _shift is None else _shift
    V27['M3_R3_A8_' + _label] = ('M3_R3_A8_current_output',
                                 {} if _shift is None else dict(threshold_shift=_shift))
ARMS_V27THR = ('D_native',) + tuple('M3_R3_A8_' + ('P0' if s is None else s) for s in SHIFTS)
# v27 attribution: the same-consumer all-kept dense control separates kernel efficiency
# from sparsity; the length-gated variant shows the short-context fallback.
ARMS_V27ATTR = ('D_native', 'D_matched', 'M3_R3_A64', 'M3_R6_A64', 'B_A64', 'M3_R3_A64_G8k')
# v27 all-layer scope (GLOBAL + LOCAL routed with their own frozen P0 thresholds).
ALL_SCOPE = 'ALL_NATIVE_LEGAL'
for _name in ('M1_R1_A8', 'M2c_pool_R1_A8', 'M3_R3_A64', 'M3_R6_A64', 'B_A64'):
    V27[_name + '_ALL'] = (V27[_name][0], dict(V27[_name][1], scope=ALL_SCOPE))
V27['M3_R6_A64_mid3'] = ('M3_R3_A8_current_output', dict(score_period=64, decision_interval=6, route_layers='mid3'))
V27['M3_R6_A64_pairs'] = ('M3_R3_A8_current_output', dict(score_period=64, decision_interval=6, share_layers='pairs'))
V27['M3_R6_A64_one'] = ('M3_R3_A8_current_output', dict(score_period=64, decision_interval=6, share_layers='one'))
V27['M3_R3_A64_pairs'] = ('M3_R3_A8_current_output', dict(score_period=64, share_layers='pairs'))
V27['M3_R3_A64_one'] = ('M3_R3_A8_current_output', dict(score_period=64, share_layers='one'))
V27['M1_R1_A8_one'] = ('M1_R1_A8_current_output', dict(share_layers='one'))
V27['B_A64_mid3'] = ('B_A8_matched', dict(hold_only=True, score_period=64, route_layers='mid3'))
ARMS_V27LAYERS = ('D_native', 'D_matched', 'M3_R6_A64', 'M3_R6_A64_mid3', 'M3_R6_A64_pairs', 'M3_R6_A64_one',
                  'M3_R3_A64', 'M3_R3_A64_pairs', 'M3_R3_A64_one', 'M1_R1_A8', 'M1_R1_A8_one', 'B_A64', 'B_A64_mid3')
ARMS_V27ALL = ('D_native', 'D_matched', 'D_matched_ALL', 'M1_R1_A8_ALL', 'M2c_pool_R1_A8_ALL',
               'M3_R3_A64_ALL', 'M3_R6_A64_ALL', 'B_A64_ALL', 'M3_R6_A64', 'B_A64')
ARM_SETS = {'v24': ARMS, 'v25': ARMS_V25, 'v27': ARMS_V27, 'v27thr': ARMS_V27THR, 'v27attr': ARMS_V27ATTR,
            'v27all': ARMS_V27ALL, 'v27layers': ARMS_V27LAYERS}
PHASE_KEYS = ('bootstrap_dense_calls', 'bootstrap_observation_calls')
WRAPPER_DROP = ('fingerprint', 'condition', 'plugin', 'v20_arm', 'v20_scope', 'decision_interval',
                'score_refresh_period', 'output_mode', 'control')


def build_arms(v21_profile_config, arm_set='v24'):
    """Six arms with the panel's selectors, output contract and grouped-Q producer."""
    from experiments.numerical_qk_reuse import v21
    from scripts import v20_bind as old
    arms = {arm['name']: arm for arm in v21_profile_config['arms']}
    native, combined = arms['D_native'], arms['combined']
    if native['condition'] != 'native_dense':
        raise ValueError('bound v21 profile config lacks untouched native arm')
    method_base = {k: v for k, v in combined['config']['parent_config'].items() if k not in WRAPPER_DROP}
    result = [native]
    names = ARM_SETS[arm_set]
    control_base = {k: v for k, v in native['config'].items() if k not in WRAPPER_DROP}
    t_config = old.control_config(dict(control_base, control='T_scope'), 'v20_fresh_T', SCOPE)
    for dense, dense_scope in (('D_matched', SCOPE), ('D_matched_ALL', 'ALL_NATIVE_LEGAL')):
        if dense not in names:
            continue
        matched = v21.effective_control_config(dict(control_base, consumer='triton'), dense_scope,
                                               output_score_precision='fp32_scores_bf16_pv',
                                               output_layout='model_major')
        result.append(dict(name=dense, plugin=PLUGIN, condition=matched['condition'], config=matched))
    if 'T_scope' in names:
        result.append(dict(name='T_scope', plugin=t_config['plugin'], condition=t_config['condition'],
                           config=t_config))
    for name in [n for n in names if n in METHODS]:
        parent_arm, bootstrap = METHODS[name]
        selector = 'legacy_recompute' if parent_arm == 'B_A8_matched' else 'prefix_block_summary'
        config = v21.effective_config(dict(method_base, selector=selector, selector_layers='all'),
                                      parent_arm, SCOPE,
                                      output_score_precision='fp32_scores_bf16_pv',
                                      output_layout='model_major', bootstrap_policy=bootstrap,
                                      observation_producer='grouped_q',
                                      route_storage=STORAGE.get(name, 'logical'))
        result.append(dict(name=name, plugin=PLUGIN, condition=config['condition'], config=config))
    for name in [n for n in names if n in V27]:
        parent_arm, extra = V27[name]
        extra = dict(extra)
        scope = extra.pop('scope', SCOPE)
        selector = 'legacy_recompute' if parent_arm == 'B_A8_matched' else 'prefix_block_summary'
        config = v21.effective_config(dict(method_base, selector=selector, selector_layers='all'),
                                      parent_arm, scope,
                                      output_score_precision='fp32_scores_bf16_pv',
                                      output_layout='model_major', bootstrap_policy=BOOT,
                                      observation_producer='grouped_q',
                                      route_storage='aligned16_odd', **extra)
        result.append(dict(name=name, plugin=PLUGIN, condition=config['condition'], config=config))
    return result


def derive_config(v21_profile_config, arm_set='v24', targets=None, sequence_length=None):
    config = {k: v for k, v in v21_profile_config.items() if k != 'arms'}
    if arm_set.startswith('v27'):
        if not targets or sequence_length not in (24, 32):
            raise ValueError('v27 direct cost needs predeclared targets and N24/N32')
        config.update(schema='v27_direct_cost_v1', arm_set=arm_set,
                      arms=build_arms(v21_profile_config, arm_set), targets=targets,
                      boundaries=(['model_forward', 'denoising_step'] if arm_set == 'v27'
                                  else ['model_forward']),
                      sequence_lengths=[sequence_length], reps=3, blocks=3, warmup=1,
                      # Shared-support followers are not modelled by the counter twin.
                      counter_twins=arm_set != 'v27layers', operator_probe=False,
                      prepared_support_floor=arm_set != 'v27layers',
                      derived_from_v21_profile_config_sha256=hashlib.sha256(
                          json.dumps(v21_profile_config, sort_keys=True).encode()).hexdigest())
        return config
    config.update(schema='v24_bootstrap_direct_cost_v1', arm_set=arm_set,
                  arms=build_arms(v21_profile_config, arm_set),
                  boundaries=['model_forward', 'denoising_step'], sequence_lengths=[16],
                  reps=3, blocks=3, warmup=1, counter_twins=False, operator_probe=False,
                  prepared_support_floor=False,
                  derived_from_v21_profile_config_sha256=hashlib.sha256(
                      json.dumps(v21_profile_config, sort_keys=True).encode()).hexdigest())
    return config


def validate_config(config):
    base.validate_config(config)
    arm_set = config.get('arm_set', 'v24')
    expected = ARM_SETS[arm_set]
    if [arm['name'] for arm in config['arms']] != list(expected):
        raise ValueError('arm inventory differs from the declared arm set')
    lengths = [24] if arm_set.startswith('v27') and config.get('sequence_lengths') == [24] else \
        [32] if arm_set.startswith('v27') else [16]
    if config.get('scope') != SCOPE or config.get('sequence_lengths') != lengths:
        raise ValueError('v24/v27 are GLOBAL-only and replay from canvas start')
    from experiments.numerical_qk_reuse import v21
    for arm in [a for a in config['arms'] if a['name'] in V27]:
        v21.validate_effective(arm['config'], arm['condition'])
        parent_arm, extra = V27[arm['name']]
        extra = {k: v for k, v in extra.items() if k != 'scope'}
        c = arm['config']
        if (c['parent_config'].get('v20_arm') != parent_arm or c.get('bootstrap_policy') != BOOT or
                c.get('route_storage') != 'aligned16_odd' or c.get('observation_producer') != 'grouped_q' or
                any(c.get(k) != v for k, v in extra.items())):
            raise ValueError(f'v27 arm contract drift: {arm["name"]}')
    for arm in [a for a in config['arms'] if a['name'] in METHODS]:
        v21.validate_effective(arm['config'], arm['condition'])
        want = METHODS[arm['name']][1]
        if arm['config'].get('route_storage', 'logical') != STORAGE.get(arm['name'], 'logical'):
            raise ValueError(f'route storage drift: {arm["name"]}')
        if (arm['config'].get('bootstrap_policy') != want or
                arm['config'].get('observation_producer') != 'grouped_q' or
                arm['config'].get('output_score_precision') != 'fp32_scores_bf16_pv' or
                arm['config'].get('output_layout') != 'model_major'):
            raise ValueError(f'v24 arm contract drift: {arm["name"]}')
    return config


def _flatten(config):
    flattened = dict(config)
    flattened['arms'] = [dict(arm, config=arm['config'].get('parent_config', arm['config']))
                         for arm in config['arms']]
    return flattened


def classify(delta):
    """One decoder call's GLOBAL phase from counter deltas (5 GLOBAL layer-calls)."""
    if not delta:
        return 'native'
    if delta.get('bootstrap_dense_calls'):
        return 'B0'
    if delta.get('bootstrap_observation_calls'):
        return 'BO'
    if delta.get('score_refresh_calls'):
        return 'A'
    if delta.get('decision_refresh_calls'):
        return 'D'
    if delta.get('held_decision_calls'):
        return 'H'
    if delta.get('calls') or delta.get('attention_calls'):
        return 'routed_other'
    return 'native'


def phase_table(report):
    """Per target/boundary/arm median event ms by phase, from accepted per-call rows."""
    import statistics
    table = {}
    for key, target in report['targets'].items():
        for boundary, lengths in target.get('boundaries', {}).items():
            for label, entry in lengths.items():
                for arm, data in entry['arms'].items():
                    summary = data['summary']
                    phases = [classify(d) for d in summary.get('phase_deltas', [])]
                    medians = summary.get('per_call_event_median_ms') or []
                    by = {}
                    for phase, ms in zip(phases, medians):
                        by.setdefault(phase, []).append(ms)
                    table.setdefault(key, {}).setdefault(boundary, {})[arm] = dict(
                        sequence=''.join({'native': 'N', 'B0': '0', 'BO': 'O', 'A': 'A', 'D': 'D',
                                          'H': 'H', 'routed_other': '?'}[p] for p in phases),
                        per_phase_median_ms={p: statistics.median(v) for p, v in by.items()},
                        per_phase_calls={p: len(v) for p, v in by.items()},
                        sequence_event_median_ms=data['direct_epoch']['event_median_ms'])
    return table


def profile(config, checkpoint=None):
    validate_config(config)
    original_snapshot = base.counter_snapshot
    original_preflight = base.preflight_identity

    def snapshot(runtime):
        values = original_snapshot(runtime)
        if runtime is not None:
            counter = runtime.get('counters')
            raw = counter() if callable(counter) else {}
            for key in PHASE_KEYS:
                if type(raw.get(key)) is int:
                    values[key] = raw[key]
        return values

    def preflight(_config, paths):
        from experiments.numerical_qk_reuse import v21
        for arm in [a for a in _config['arms'] if a['name'] in METHODS or a['name'] in V27]:
            v21.validate_effective(arm['config'], arm['condition'])
        return original_preflight(_flatten(_config), paths)

    with patch.object(base, 'counter_snapshot', snapshot), \
         patch.object(base, 'preflight_identity', preflight):
        report = base.profile(config, checkpoint=checkpoint)
    report['schema'] = config.get('schema', 'v24_bootstrap_direct_cost_v1')
    report['phase_table'] = phase_table(report)
    report['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    report['note'] = ('Fixed native-state replay timing diagnostic; phases from counter deltas; '
                      'no physical counter twin; not a scored or fixed-step method')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v21-profile-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm-set', choices=tuple(ARM_SETS), default='v24')
    parser.add_argument('--targets', type=Path, help='v27: predeclared targets JSON list')
    parser.add_argument('--sequence-length', type=int, help='v27: 24 or 32')
    args = parser.parse_args(argv)
    out = args.out
    derived = out.with_name(out.name + '.config.json')
    partial = out.with_name(out.name + '.partial.json')
    failure = out.with_name(out.name + '.failure.json')
    occupied = [p for p in (out, derived, partial, failure) if p.exists()]
    if occupied:
        raise FileExistsError(f'v24 output exists: {occupied}')
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = derive_config(json.loads(args.v21_profile_config.read_bytes()), args.arm_set,
                               targets=json.loads(args.targets.read_bytes()) if args.targets else None,
                               sequence_length=args.sequence_length)
        validate_config(config)
        with derived.open('x', encoding='utf-8') as stream:
            json.dump(config, stream, indent=1)
        report = profile(config, checkpoint=lambda r: base.atomic_json(partial, r))
        base.atomic_json(out, report)
        partial.unlink(missing_ok=True)
    except BaseException as exc:
        base.atomic_json(failure, dict(schema='v24_profile_failure_v1', error_type=type(exc).__name__,
                                       error=str(exc)[:2000], pid=os.getpid(),
                                       partial_path=str(partial) if partial.exists() else None))
        raise


if __name__ == '__main__':
    main()
