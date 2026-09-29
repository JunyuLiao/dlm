"""v21 diagnostic output precision and exact model-major storage on v20.

The nested v20 config is passed intact to its installer.  In particular its
cache, selector, clock, and control binding remain the v20 contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from . import v20
from .cached_executor import OUTPUT_LAYOUTS, OUTPUT_PRECISIONS


PLUGIN = 'experiments.numerical_qk_reuse.v21:install'
CONTROL_PLUGIN = 'experiments.numerical_qk_reuse.v20_controls:install'
CONTROL_CONDITION = 'v20_dense_consumer'
SOURCES = ('v21.py', 'generic_kernels.py', 'cached_executor.py', 'integration.py',
           'v20_controls.py')
REQUEST_ENVELOPE = ('phase', 'diagnostic', 'timing_events', 'thinking', 'max_new_tokens')
# v23 optional method fields. Absent keys mean the original behaviour, so
# every previously bound v21 config keeps its fingerprint.
BOOTSTRAP_POLICIES = ('native_bootstrap2_observe1',)
OBSERVATION_PRODUCERS = ('repeat_interleave', 'grouped_q')
ROUTE_STORAGES = ('logical', 'aligned16', 'aligned16_odd')
MU_MODES = ('exact', 'pooled', 'pooled_compact')
SCORE_PERIODS = (8, 16, 64)
# v27 optional clock/threshold overrides on top of a named parent arm. The
# parent arm stays as namespaced provenance; the effective values are bound on
# the runtime and reported through one authoritative effective_method record.
DECISION_INTERVALS = (6,)
_LN2 = 0.6931471805599453
# Named log-threshold shifts in ln2 units; larger shifts allow more deletion.
# The realized sparsity is measured, never inferred from the shift.
THRESHOLD_SHIFTS = {'minus_ln2': -_LN2, 'plus_ln2': _LN2, 'plus_2ln2': 2 * _LN2,
                    'plus_3ln2': 3 * _LN2, 'plus_4ln2': 4 * _LN2}


def _fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _check_modes(precision, layout, parent):
    if precision not in OUTPUT_PRECISIONS or layout not in OUTPUT_LAYOUTS:
        raise ValueError('unknown v21 output precision or layout')
    if precision != OUTPUT_PRECISIONS[0] or layout != OUTPUT_LAYOUTS[0]:
        if parent.get('consumer', 'triton') != 'triton' or parent.get('kernel_variant') != 'generic':
            raise ValueError('v21 nondefault output modes require generic Triton consumer')


def effective_config(base: dict, arm: str, scope: str, *,
                     output_score_precision='legacy_bf16_scores', output_layout='head_major',
                     bootstrap_policy=None, observation_producer='repeat_interleave',
                     route_storage='logical', mu_mode='exact', score_period=8,
                     decision_interval=None, hold_only=False, threshold_shift=None) -> dict:
    """Build a wrapper identity while preserving the parent v20 identity."""
    if route_storage != 'logical' and (route_storage not in ROUTE_STORAGES
                                       or output_score_precision != 'fp32_scores_bf16_pv'):
        raise ValueError('v25 route storage requires a known variant and FP32 current-output scores')
    prepared = dict(base)
    if output_score_precision not in OUTPUT_PRECISIONS or output_layout not in OUTPUT_LAYOUTS:
        raise ValueError('unknown v21 output precision or layout')
    if output_score_precision != OUTPUT_PRECISIONS[0] or output_layout != OUTPUT_LAYOUTS[0]:
        if prepared.get('consumer', 'triton') != 'triton' or prepared.get('kernel_variant', 'generic') != 'generic':
            raise ValueError('v21 nondefault output modes require generic Triton consumer')
        prepared['consumer'] = 'triton'
        prepared['kernel_variant'] = 'generic'
    parent = v20.effective_config(prepared, arm, scope)
    _check_modes(output_score_precision, output_layout, parent)
    extra = {}
    if bootstrap_policy is not None:
        if bootstrap_policy not in BOOTSTRAP_POLICIES:
            raise ValueError('unknown v23 bootstrap policy')
        extra['bootstrap_policy'] = bootstrap_policy
    if observation_producer != 'repeat_interleave':
        if observation_producer not in OBSERVATION_PRODUCERS:
            raise ValueError('unknown v23 observation producer')
        extra['observation_producer'] = observation_producer
    if route_storage != 'logical':
        if route_storage not in ROUTE_STORAGES or output_score_precision != 'fp32_scores_bf16_pv':
            raise ValueError('v25 route storage requires a known variant and FP32 current-output scores')
        extra['route_storage'] = route_storage
    if mu_mode != 'exact':
        if mu_mode not in MU_MODES or bootstrap_policy is None:
            raise ValueError('v26 pooled mu requires the bootstrap mainline')
        extra['mu_mode'] = mu_mode
    if score_period != 8:
        if score_period not in SCORE_PERIODS or bootstrap_policy is None:
            raise ValueError('v26 score period must be 8 or 16 on the bootstrap mainline')
        extra['score_period'] = score_period
    if decision_interval is not None:
        if (decision_interval not in DECISION_INTERVALS or bootstrap_policy is None
                or arm != 'M3_R3_A8_current_output'):
            raise ValueError('v27 decision interval override requires the bootstrap M3 parent')
        extra['decision_interval'] = decision_interval
    if hold_only:
        if hold_only is not True or bootstrap_policy is None or arm != 'B_A8_matched':
            raise ValueError('v27 hold_only requires the bootstrap matched-B parent')
        extra['hold_only'] = True
    if threshold_shift is not None:
        if threshold_shift not in THRESHOLD_SHIFTS or bootstrap_policy is None:
            raise ValueError('v27 threshold shift must be a named shift on the bootstrap mainline')
        extra['threshold_shift'] = threshold_shift
    return _wrap(parent, 'v20_method', output_score_precision, output_layout, extra)


def effective_control_config(base: dict, scope: str, *,
                             output_score_precision='legacy_bf16_scores', output_layout='head_major') -> dict:
    """Freeze the D_matched same-consumer control under the v20 control installer."""
    if scope not in v20.SCOPES or base.get('diagnostic') is not False:
        raise ValueError('D_matched requires explicit v20 native scope and diagnostic=False')
    if output_score_precision not in OUTPUT_PRECISIONS or output_layout not in OUTPUT_LAYOUTS:
        raise ValueError('unknown v21 output precision or layout')
    consumer = base.get('consumer', 'triton')
    if consumer not in ('triton', 'hopper') or (consumer == 'hopper' and not base.get('support_build')):
        raise ValueError('D_matched requires a qualified consumer identity')
    parent = dict(base)
    parent.update(condition=CONTROL_CONDITION, plugin=CONTROL_PLUGIN,
                  control='D_matched', v20_scope=scope, diagnostic=False,
                  consumer=consumer, kernel_variant='generic')
    root = Path(__file__).resolve().parent
    parent['source_hashes'] = dict(base.get('source_hashes', {}))
    parent['source_hashes'][str(root / 'v20_controls.py')] = (
        hashlib.sha256((root / 'v20_controls.py').read_bytes()).hexdigest())
    parent.pop('fingerprint', None)
    parent['fingerprint'] = _fingerprint(parent)
    _check_modes(output_score_precision, output_layout, parent)
    return _wrap(parent, 'v20_control', output_score_precision, output_layout)


def _wrap(parent, parent_kind, output_score_precision, output_layout, extra=None):
    root = Path(__file__).resolve().parent
    source_hashes = {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in SOURCES}
    result = dict(plugin=PLUGIN, parent_kind=parent_kind,
                  parent_config=parent, condition=parent['condition'],
                  output_score_precision=output_score_precision, output_layout=output_layout,
                  output_precision_status=('diagnostic' if output_score_precision == 'fp32_scores_bf16_pv'
                                           else 'legacy'), source_hashes=source_hashes)
    result.update({name: parent[name] for name in REQUEST_ENVELOPE if name in parent})
    result.update(extra or {})
    result['fingerprint'] = _fingerprint(result)
    return result


def _validate_control(parent, condition):
    if condition != CONTROL_CONDITION or parent.get('condition') != condition:
        raise ValueError('v21 D_matched control condition drift')
    if (parent.get('plugin') != CONTROL_PLUGIN or parent.get('control') != 'D_matched' or
            parent.get('v20_scope') not in v20.SCOPES or parent.get('diagnostic') is not False or
            parent.get('kernel_variant') != 'generic' or parent.get('consumer') not in ('triton', 'hopper')):
        raise ValueError('v21 D_matched parent control identity drift')
    if parent['consumer'] == 'hopper' and not parent.get('support_build'):
        raise ValueError('v21 D_matched Hopper identity missing')
    original = dict(parent)
    fingerprint = original.pop('fingerprint', None)
    if fingerprint is None or fingerprint != _fingerprint(original):
        raise ValueError('v21 parent control fingerprint drift')
    source = Path(__file__).resolve().with_name('v20_controls.py')
    if parent.get('source_hashes', {}).get(str(source)) != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('v21 parent control source byte identity drift')
    return parent['v20_scope'], None


def validate_effective(config: dict, condition: str):
    if config.get('plugin') != PLUGIN or config.get('condition') != condition:
        raise ValueError('v21 plugin or condition differs from effective config')
    parent = config.get('parent_config')
    if not isinstance(parent, dict):
        raise ValueError('v21 requires intact parent v20 config')
    for name in REQUEST_ENVELOPE:
        if (name in config) != (name in parent) or (name in parent and config[name] != parent[name]):
            raise ValueError(f'v21 request envelope {name} differs from parent')
    parent_kind = config.get('parent_kind')
    if parent_kind == 'v20_method':
        scope, interval = v20.validate_effective(parent, condition)
    elif parent_kind == 'v20_control':
        scope, interval = _validate_control(parent, condition)
    else:
        raise ValueError('v21 parent kind must name a qualified method or D_matched control')
    precision, layout = config.get('output_score_precision'), config.get('output_layout')
    _check_modes(precision, layout, parent)
    if 'bootstrap_policy' in config and (config['bootstrap_policy'] not in BOOTSTRAP_POLICIES
                                         or parent_kind != 'v20_method'):
        raise ValueError('v23 bootstrap policy identity drift')
    if 'observation_producer' in config and (config['observation_producer'] not in OBSERVATION_PRODUCERS[1:]
                                             or parent_kind != 'v20_method'):
        raise ValueError('v23 observation producer identity drift')
    if 'route_storage' in config and (config['route_storage'] not in ROUTE_STORAGES[1:]
                                      or parent_kind != 'v20_method'
                                      or precision != 'fp32_scores_bf16_pv'):
        raise ValueError('v25 route storage identity drift')
    if 'mu_mode' in config and (config['mu_mode'] not in MU_MODES[1:] or 'bootstrap_policy' not in config):
        raise ValueError('v26 mu mode identity drift')
    if 'score_period' in config and (config['score_period'] not in SCORE_PERIODS[1:]
                                     or 'bootstrap_policy' not in config):
        raise ValueError('v26 score period identity drift')
    arm = parent.get('v20_arm')
    if 'decision_interval' in config and (config['decision_interval'] not in DECISION_INTERVALS
                                          or 'bootstrap_policy' not in config
                                          or arm != 'M3_R3_A8_current_output'):
        raise ValueError('v27 decision interval identity drift')
    if 'hold_only' in config and (config['hold_only'] is not True or 'bootstrap_policy' not in config
                                  or arm != 'B_A8_matched'):
        raise ValueError('v27 hold_only identity drift')
    if 'threshold_shift' in config and (config['threshold_shift'] not in THRESHOLD_SHIFTS
                                        or 'bootstrap_policy' not in config):
        raise ValueError('v27 threshold shift identity drift')
    status = 'diagnostic' if precision == 'fp32_scores_bf16_pv' else 'legacy'
    if config.get('output_precision_status') != status:
        raise ValueError('v21 output precision status drift')
    original = dict(config)
    fingerprint = original.pop('fingerprint', None)
    if fingerprint is None or fingerprint != _fingerprint(original):
        raise ValueError('v21 effective config fingerprint drift')
    root = Path(__file__).resolve().parent
    for name in SOURCES:
        source = root / name
        if config.get('source_hashes', {}).get(str(source)) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError('v21 source byte identity drift')
    return scope, interval


@contextmanager
def install(adapter, config: dict, condition: str):
    """Bind the qualified v20 control then set owner output attributes."""
    scope, interval = validate_effective(config, condition)
    if config['parent_kind'] == 'v20_method':
        parent_install = v20.install
    else:
        from .v20_controls import install as parent_install
    with parent_install(adapter, config['parent_config'], condition) as runtime:
        router = runtime['router']
        owner = router if config['parent_kind'] == 'v20_method' else router.owner
        if owner.output_mode != v20.PREQK_MODE:
            raise ValueError('v21 requires parent current-output mode')
        owner.output_score_precision = config['output_score_precision']
        owner.output_layout = config['output_layout']
        if 'observation_producer' in config:
            owner.observation_producer = config['observation_producer']
        if 'route_storage' in config:
            owner.route_storage = config['route_storage']
        if 'mu_mode' in config:
            owner.mu_mode = config['mu_mode']
        if 'score_period' in config:
            if owner.cache.entries or owner.calls:
                raise RuntimeError('score period must be bound before any routed call')
            owner.cache.score_period = config['score_period']
        if 'decision_interval' in config:
            if owner.cache.entries or owner.calls:
                raise RuntimeError('decision interval must be bound before any routed call')
            owner.cache.decision_interval = config['decision_interval']
        if config.get('hold_only'):
            if owner.cache.entries or owner.calls:
                raise RuntimeError('hold_only must be bound before any routed call')
            owner.cache.hold_only = True
        if 'threshold_shift' in config:
            if owner.cache.entries or owner.calls:
                raise RuntimeError('threshold shift must be bound before any routed call')
            shift = THRESHOLD_SHIFTS[config['threshold_shift']]
            owner.thresholds = {kind: dict(value, log_threshold=float(value['log_threshold']) + shift)
                                for kind, value in owner.thresholds.items()}
        if 'bootstrap_policy' in config:
            if owner.cache.entries or owner.calls:
                raise RuntimeError('bootstrap must be bound before any routed call')
            owner.bootstrap = config['bootstrap_policy']
            owner.cache.origin = 1
        owner.output_precision_extra_qk_elements_upper_bound = 0
        parent_counters = runtime['counters']

        def effective_method():
            # Read from the BOUND runtime, never from the parent arm name.
            cache = getattr(owner, 'cache', None)
            if cache is None:
                return None
            return dict(score_period=cache.score_period,
                        decision_interval=None if cache.hold_only else cache.decision_interval,
                        hold_only=bool(cache.hold_only), score_clock_origin=cache.origin,
                        mu_mode=getattr(owner, 'mu_mode', 'exact'),
                        route_storage=getattr(owner, 'route_storage', 'logical'),
                        scope=scope, bootstrap_policy=getattr(owner, 'bootstrap', None),
                        threshold_shift=config.get('threshold_shift'),
                        log_thresholds={k: float(v['log_threshold']) for k, v in owner.thresholds.items()},
                        output_score_precision=owner.output_score_precision,
                        output_layout=owner.output_layout,
                        observation_producer=getattr(owner, 'observation_producer', 'repeat_interleave'),
                        selector=getattr(owner, 'selector', None),
                        parent_v20_arm=config['parent_config'].get('v20_arm'))

        def counters():
            values = parent_counters()
            effective = effective_method() if config['parent_kind'] == 'v20_method' else None
            clock = {}
            if effective is not None and 'score_refresh_period' in values:
                # v20 writes its constant parent A8 here; keep it namespaced.
                clock = dict(parent_v20_score_refresh_period=values['score_refresh_period'],
                             parent_v20_decision_interval=values.get('decision_interval'),
                             score_refresh_period=effective['score_period'],
                             decision_interval=effective['decision_interval'])
            return dict(values, **clock, effective_method=effective,
                        v21_scope=scope, v21_decision_interval=interval,
                        v23_bootstrap_policy=config.get('bootstrap_policy'),
                        v23_observation_producer=config.get('observation_producer', 'repeat_interleave'),
                        v25_route_storage=config.get('route_storage', 'logical'),
                        v26_mu_mode=config.get('mu_mode', 'exact'),
                        v26_score_period=config.get('score_period', 8),
                        v27_decision_interval=config.get('decision_interval'),
                        v27_hold_only=bool(config.get('hold_only', False)),
                        v27_threshold_shift=config.get('threshold_shift'),
                        output_score_precision=owner.output_score_precision,
                        output_layout=owner.output_layout,
                        output_precision_status=config['output_precision_status'],
                        output_precision_extra_qk_elements_upper_bound=
                            owner.output_precision_extra_qk_elements_upper_bound,
                        output_precision_extra_qk_counter_scope=
                            'conservative full QK dispatch geometry; retained physical work counted separately')

        yield {**runtime, 'counters': counters}
