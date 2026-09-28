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
                     bootstrap_policy=None, observation_producer='repeat_interleave') -> dict:
    """Build a wrapper identity while preserving the parent v20 identity."""
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
        if 'bootstrap_policy' in config:
            if owner.cache.entries or owner.calls:
                raise RuntimeError('bootstrap must be bound before any routed call')
            owner.bootstrap = config['bootstrap_policy']
            owner.cache.origin = 1
        owner.output_precision_extra_qk_elements_upper_bound = 0
        parent_counters = runtime['counters']

        def counters():
            values = parent_counters()
            return dict(values, v21_scope=scope, v21_decision_interval=interval,
                        v23_bootstrap_policy=config.get('bootstrap_policy'),
                        v23_observation_producer=config.get('observation_producer', 'repeat_interleave'),
                        output_score_precision=owner.output_score_precision,
                        output_layout=owner.output_layout,
                        output_precision_status=config['output_precision_status'],
                        output_precision_extra_qk_elements_upper_bound=
                            owner.output_precision_extra_qk_elements_upper_bound,
                        output_precision_extra_qk_counter_scope=
                            'conservative full QK dispatch geometry; retained physical work counted separately')

        yield {**runtime, 'counters': counters}
