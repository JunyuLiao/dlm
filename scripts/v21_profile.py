"""Direct v21 four-mode profile on frozen native states; no generation score read.

Uses v20's capture, StepSnapshot replay, native brackets, warmup, accepted CUDA
event timing, phase counters, and durable target checkpoints.  This module only
validates v21 wrapper identities and adds a separate untimed copy/allocation
trace.  Run in a dedicated process, away from accepted natural-request timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts import v20_profile as base


PLUGIN = 'experiments.numerical_qk_reuse.v21:install'
MODES = {
    'legacy': ('legacy_bf16_scores', 'head_major'),
    'layout_only': ('legacy_bf16_scores', 'model_major'),
    'numeric_only': ('fp32_scores_bf16_pv', 'head_major'),
    'combined': ('fp32_scores_bf16_pv', 'model_major'),
}


def validate_config(config):
    """Check frozen target and exact variant grid before model/GPU load."""
    base.validate_config(config)
    if config.get('counter_twins') is not True:
        raise ValueError('v21 needs a separate untimed physical counter pass')
    if config.get('scope') != 'GLOBAL_ONLY_NATIVE_LOCAL':
        raise ValueError('v21 profile requires frozen GLOBAL_ONLY_NATIVE_LOCAL scope')
    if set(config.get('boundaries', ['model_forward', 'denoising_step'])) != {'model_forward', 'denoising_step'}:
        raise ValueError('v21 requires both complete-forward and native-step boundaries')
    arms = config['arms']
    if len(arms) != 5 or arms[0]['name'] != 'D_native':
        raise ValueError('v21 requires native plus four exact output modes')
    if set(a['name'] for a in arms[1:]) != set(MODES):
        raise ValueError('v21 output arm names must be legacy/layout_only/numeric_only/combined')
    common = None
    for arm in arms[1:]:
        wrapper = arm['config']
        if arm.get('plugin') != PLUGIN or wrapper.get('plugin') != PLUGIN:
            raise ValueError('v21 wrapper installer required')
        if wrapper.get('parent_kind') != 'v20_method' or not isinstance(wrapper.get('parent_config'), dict):
            raise ValueError('v21 M3 method parent config required')
        parent = wrapper['parent_config']
        if wrapper.get('condition') != arm['condition'] or parent.get('condition') != arm['condition']:
            raise ValueError('v21 wrapper/parent condition drift')
        if parent.get('v20_arm') != 'M3_R3_A8_current_output' or parent.get('v20_scope') != config['scope']:
            raise ValueError('v21 frozen M3_R3_A8 GLOBAL scope required')
        if parent.get('decision_interval') != 3 or parent.get('score_refresh_period') != 8:
            raise ValueError('v21 frozen R3/A8 clocks required')
        if (wrapper.get('output_score_precision'), wrapper.get('output_layout')) != MODES[arm['name']]:
            raise ValueError('v21 output mode/name mismatch')
        # Identity of the inherited control must be the same across four arms.
        identity = {k: v for k, v in parent.items() if k not in ('fingerprint',)}
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError('v21 variants have different v20 parent controls')
    return config


def _flatten_for_preflight(config):
    """v20's inherited byte/model checker expects a flat arm config."""
    flattened = dict(config)
    flattened['arms'] = [dict(arm, config=(arm['config']['parent_config'] if i else arm['config']))
                         for i, arm in enumerate(config['arms'])]
    return flattened


def preflight_identity(config, paths):
    """Verify wrapper hashes/fingerprints, then reuse v20's strict flat checks."""
    validate_config(config)
    from experiments.numerical_qk_reuse import v21
    for arm in config['arms'][1:]:
        v21.validate_effective(arm['config'], arm['condition'])
    return base.preflight_identity(_flatten_for_preflight(config), paths)


def copy_allocation_trace(model, sequence, runtime, boundary, accepted_rows):
    """Separate profiler pass; never contributes to accepted timing samples."""
    import torch

    expected = [(r['input_digest'], r['output_digest']) for r in accepted_rows]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, profile_memory=True) as trace:
        observed = base.replay_untimed(model, sequence, runtime, boundary)
    torch.cuda.synchronize()
    if observed != expected:
        raise AssertionError('copy/allocation trace replay input/output drift')
    operations = {}
    positive_cuda_bytes = 0
    for event in trace.events():
        name = str(getattr(event, 'name', '<unknown>'))
        cuda_bytes = int(getattr(event, 'cuda_memory_usage', 0) or 0)
        positive_cuda_bytes += max(0, cuda_bytes)
        if any(token in name.lower() for token in ('copy', 'clone', 'contiguous', 'empty', 'memcpy')):
            operations[name] = operations.get(name, 0) + 1
    return dict(status='qualified', boundary=boundary, sequence_calls=len(sequence),
                definition='separate replay under torch.profiler; operation counts include profiler overhead and do not measure copy bytes precisely',
                operation_counts=dict(sorted(operations.items())),
                positive_cuda_allocation_event_bytes=positive_cuda_bytes,
                memory_allocated_before_bytes=before,
                memory_allocated_after_bytes=torch.cuda.memory_allocated(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                input_output_digests_match=True)


def profile(config, checkpoint=None):
    validate_config(config)
    original_preflight = base.preflight_identity
    original_counter = base.counter_replay

    def inherited_preflight(_config, paths):
        # This call is before create_adapter(...).load() in base.profile.
        from experiments.numerical_qk_reuse import v21
        for arm in _config['arms'][1:]:
            v21.validate_effective(arm['config'], arm['condition'])
        return original_preflight(_flatten_for_preflight(_config), paths)

    def counted(model, sequence, runtime, boundary, accepted_rows, **kwargs):
        physical, support = original_counter(model, sequence, runtime, boundary,
                                             accepted_rows, **kwargs)
        physical['copy_allocation_trace'] = copy_allocation_trace(
            model, sequence, runtime, boundary, accepted_rows)
        return physical, support

    # One isolated process runs one profile; temporary hooks reuse v20's exact
    # replay machinery without changing its accepted intervals or source.
    with patch.object(base, 'preflight_identity', inherited_preflight), \
         patch.object(base, 'counter_replay', counted):
        report = base.profile(config, checkpoint=checkpoint)
    report['schema'] = 'v21_direct_output_modes_v1'
    report['v21_scope'] = config['scope']
    report['v21_modes'] = MODES
    report['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['diagnostic_note'] = ('copy/allocation and physical counters are separate replay passes after accepted timing; '
                                 'no profiler hooks or device-to-host tensor export in accepted intervals')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + '.partial.json')
    failure = out.with_name(out.name + '.failure.json')
    lock = out.with_name(out.name + '.lock')
    occupied = [p for p in (out, partial, failure, lock, out.with_name(out.name + '.writing')) if p.exists()]
    if occupied:
        raise FileExistsError(f'profile output/receipt already exists: {occupied}')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    latest = []
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        def checkpoint(report):
            latest[:] = list(report['targets'])
            base.atomic_json(partial, report)
        report = profile(config, checkpoint=checkpoint)
        base.atomic_json(out, report)
        partial.unlink(missing_ok=True)
    except BaseException as exc:
        base.atomic_json(failure, dict(schema='v21_profile_failure_v1',
                                       error_type=type(exc).__name__, completed_targets=latest,
                                       partial_path=str(partial) if partial.exists() else None,
                                       config_sha256=(hashlib.sha256(config_bytes).hexdigest()
                                                      if 'config_bytes' in locals() else None)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
