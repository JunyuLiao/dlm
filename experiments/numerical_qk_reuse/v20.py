"""Explicit v20 M1/M3/anchor-held methods on the existing numerical router.

The score cache owns both clocks. This module only freezes their requested
values and checks that the effective bound runtime actually used them.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


PLUGIN = 'experiments.numerical_qk_reuse.v20:install'
PREQK_MODE = 'historical_route_preqk_current_output'
ALL_NATIVE_LEGAL = 'ALL_NATIVE_LEGAL'
GLOBAL_ONLY_NATIVE_LOCAL = 'GLOBAL_ONLY_NATIVE_LOCAL'
SCOPES = (ALL_NATIVE_LEGAL, GLOBAL_ONLY_NATIVE_LOCAL)
SCORE_PERIOD = 8
MAX_CACHE_BYTES = 4 * 1024**3
MAX_SUMMARY_BYTES = 1024**3
ARM_INTERVALS = {
    'M1_R1_A8_current_output': 1,
    'M3_R2_A8_current_output': 2,
    'M3_R3_A8_current_output': 3,
    'B_A8_matched': 8,
}
GLOBAL_CONDITIONS = {
    'M1_R1_A8_current_output': 'global_M1',
    'M3_R2_A8_current_output': 'global_M3',
    'M3_R3_A8_current_output': 'v20_global_M3_R3',
    'B_A8_matched': 'global_B8',
}


def _fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def effective_config(base: dict, arm: str, scope: str) -> dict:
    """Return a host-local, fingerprinted production config from a frozen base.

    The caller supplies the policy, binary/consumer identity, task budget and
    model identity. No threshold, mask, native stopping or request budget is
    inferred here.
    """
    if arm not in ARM_INTERVALS or scope not in SCOPES:
        raise ValueError('explicit v20 arm and scope required')
    if base.get('diagnostic') is not False:
        raise ValueError('v20 production fast_t requires diagnostic=False')
    if any(key in base and (type(base[key]) is not int or base[key] <= 0)
           for key in ('max_cache_bytes', 'max_summary_bytes')):
        raise ValueError('memory caps must be positive integer bytes')
    if not isinstance(base.get('policy'), dict) or set(base['policy']) != {'local', 'global'}:
        raise ValueError('frozen local/global risk policy required')
    if base.get('consumer') == 'hopper' and not base.get('support_build'):
        raise ValueError('Hopper consumer requires frozen support_build')
    selector = base.get('selector', 'legacy_recompute')
    selector_layers = base.get('selector_layers', 'all')
    if selector not in ('legacy_recompute', 'prefix_block_summary') or selector_layers not in ('local', 'all'):
        raise ValueError('unsupported v20 selector or selector layers')
    if arm == 'B_A8_matched' and selector != 'legacy_recompute':
        raise ValueError('matched anchor-held B cannot build intermediate summaries')
    interval = ARM_INTERVALS[arm]
    condition = ('M1' if interval == 1 else 'M3') if scope == ALL_NATIVE_LEGAL else GLOBAL_CONDITIONS[arm]
    result = dict(base)
    result.update(v20_arm=arm, v20_scope=scope, condition=condition, plugin=PLUGIN,
                  score_refresh_period=SCORE_PERIOD, decision_interval=interval,
                  output_mode=PREQK_MODE, fast_t=True, diagnostic=False,
                  max_cache_bytes=MAX_CACHE_BYTES, max_summary_bytes=MAX_SUMMARY_BYTES,
                  selector=selector, selector_layers=selector_layers,
                  telemetry='minimal',
                  support='native_mask')
    # The source paths remain host-local. The hash VALUES become part of the
    # later path-independent mathematical/execution identity.
    source_hashes = dict(base.get('source_hashes', {}))
    root = Path(__file__).resolve().parent
    for source in (root / 'v20.py', root / 'cache.py', root / 'integration.py',
                   root / 'global_scope.py'):
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['source_hashes'] = source_hashes
    result.pop('fingerprint', None)
    result['fingerprint'] = _fingerprint(result)
    return result


def validate_effective(config: dict, condition: str) -> tuple[str, int]:
    arm, scope = config.get('v20_arm'), config.get('v20_scope')
    if arm not in ARM_INTERVALS or scope not in SCOPES:
        raise ValueError('named v20 arm/scope required')
    interval = ARM_INTERVALS[arm]
    required_condition = ('M1' if interval == 1 else 'M3') if scope == ALL_NATIVE_LEGAL else GLOBAL_CONDITIONS[arm]
    expected = dict(condition=required_condition, plugin=PLUGIN,
                    score_refresh_period=SCORE_PERIOD, decision_interval=interval,
                    output_mode=PREQK_MODE, fast_t=True, diagnostic=False,
                    max_cache_bytes=MAX_CACHE_BYTES, max_summary_bytes=MAX_SUMMARY_BYTES,
                    telemetry='minimal',
                    support='native_mask')
    if condition != required_condition or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('v20 effective phase/mode/scope differs from frozen config')
    if (config.get('selector') not in ('legacy_recompute', 'prefix_block_summary') or
            config.get('selector_layers') not in ('local', 'all') or
            (arm == 'B_A8_matched' and config['selector'] != 'legacy_recompute')):
        raise ValueError('v20 selector differs from qualified method')
    if (type(config.get('max_cache_bytes')) is not int or
            type(config.get('max_summary_bytes')) is not int):
        raise ValueError('memory caps must be integer bytes')
    original = dict(config)
    fingerprint = original.pop('fingerprint', None)
    if fingerprint is None or fingerprint != _fingerprint(original):
        raise ValueError('v20 effective config fingerprint drift')
    source_hashes = config.get('source_hashes', {})
    root = Path(__file__).resolve().parent
    for source in (root / 'v20.py', root / 'cache.py', root / 'integration.py',
                   root / 'global_scope.py'):
        if source_hashes.get(str(source)) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError('v20 source byte identity drift')
    return scope, interval


@contextmanager
def install(adapter, config: dict, condition: str):
    """Existing numerical Attention/ScoreCache execution, with v20 guards."""
    scope, interval = validate_effective(config, condition)
    if scope == ALL_NATIVE_LEGAL:
        from .integration import install as existing_install
    else:
        from .global_scope import install as existing_install
    with existing_install(adapter, config, condition) as runtime:
        router, state = runtime['router'], runtime['state']
        if (router.cache.score_period != SCORE_PERIOD or
                router.cache.decision_interval != interval or
                router.cache.max_bytes != MAX_CACHE_BYTES or
                router.max_summary_bytes != MAX_SUMMARY_BYTES or
                router.selector != config['selector'] or
                router.selector_layers != config['selector_layers'] or
                router.output_mode != PREQK_MODE or
                state.fast_t is not True):
            raise ValueError('bound v20 runtime differs from explicit A/R/output/T config')
        if router.support != 'native_mask':
            raise ValueError('v20 routing is outside native legal support')
        original_counters = runtime['counters']
        def counters():
            measured = original_counters()
            transient = measured.get('score_peak_transient_bytes')
            summary = measured.get('summary_peak_bytes')
            return dict(measured, v20_arm=config['v20_arm'], v20_scope=scope,
                        score_refresh_period=SCORE_PERIOD, decision_interval=interval,
                        output_mode=router.output_mode, fast_t=state.fast_t,
                        retained_score_cap_bytes=MAX_CACHE_BYTES,
                        summary_cap_bytes=MAX_SUMMARY_BYTES,
                        selector=router.selector, selector_layers=router.selector_layers,
                        history_summary_transient_upper_bound_bytes=(transient + summary
                            if type(transient) is int and type(summary) is int else None))
        yield {**runtime, 'counters': counters}
