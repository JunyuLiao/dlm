"""CPU-only, redacted numerical and phase gate for selected v20 profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.v20_report import METHOD_ARMS, canonical_arm, _phase
from scripts.v20_operator_probe import within_v11_envelope


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_receipt(profile):
    result = {}
    for path, value in profile.get('source_sha256', {}).items():
        name = Path(path).name
        if name in result and result[name] != value:
            raise ValueError(f'ambiguous source identity: {name}')
        result[name] = value
    return result


def selected_sequence(target, boundary_name):
    boundary = target.get('boundaries', {}).get(boundary_name, {})
    if not boundary:
        return None, None
    label = max(boundary, key=lambda x: int(x.removeprefix('N')))
    return label, boundary[label]


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def evaluate_arm(name, arm, seq, *, canvas, scope):
    method = canonical_arm(arm)
    record = seq.get('arms', {}).get(name)
    diag = seq.get('diagnostic_replays', {}).get(name, {})
    reasons = []
    if record is None:
        return dict(status='missing', reasons=['arm timing absent'])
    blocks = record.get('blocks', [])
    if not blocks or any(b.get('triton_misses') != 0 or
                         b.get('triton_specializations_before') != b.get('triton_specializations_after') or
                         b.get('triton_disk_entries_before') != b.get('triton_disk_entries_after')
                         for b in blocks):
        reasons.append('accepted block missing or JIT/cache change')
    summary = record.get('summary') or {}
    phase_counts = {p: 0 for p in ('A', 'D', 'H')}
    for delta in summary.get('phase_deltas', []):
        phase = _phase(delta, arm)
        if phase in phase_counts:
            phase_counts[phase] += 1
    bad = []
    probe_rows = []
    if method in METHOD_ARMS or method == 'G75L30_nativeQ128':
        if diag.get('status') != 'qualified':
            reasons.append('diagnostic counter twin not qualified')
        probe = diag.get('operator_probe', {})
        probe_rows = probe.get('rows', []) if probe.get('status') == 'qualified' else []
        required_layers = [5] if scope == 'GLOBAL_ONLY_NATIVE_LOCAL' else [0, 5]
        phases_to_probe = ('A', 'H') if method == 'G75L30_nativeQ128' else ('A', 'D', 'H')
        required = {(layer, phase) for layer in required_layers for phase in phases_to_probe
                    if phase_counts[phase]}
        observed = {(r.get('layer'), r.get('phase')) for r in probe_rows
                    if r.get('canvas') == canvas and r.get('v11_envelope_pass') is True}
        missing = sorted(required - observed)
        if missing:
            reasons.append('operator phase/layer samples missing')
        if not probe_rows:
            reasons.append('operator probe absent')
        bad = [r for r in probe_rows if r.get('v11_envelope_pass') is not True or
               any(not _finite_nonnegative(r.get(k)) for k in
                   ('new_max_abs_error', 'triton_max_abs_error', 'relative_l2_error',
                    'actual_output_max_abs_error', 'actual_output_relative_l2_error')) or
               not (within_v11_envelope(r['new_max_abs_error'], r['triton_max_abs_error'],
                                        r['relative_l2_error']) and
                    within_v11_envelope(r['actual_output_max_abs_error'], r['triton_max_abs_error'],
                                        r['actual_output_relative_l2_error']))]
        if bad:
            reasons.append('operator probe failure or invalid error')
        samples = [dict(layer=r['layer'], phase=r['phase'], decoder_call=r.get('decoder_call'),
                        current_max_abs_error=r['actual_output_max_abs_error'],
                        current_relative_l2_error=r['actual_output_relative_l2_error'],
                        prepared_max_abs_error=r['new_max_abs_error'],
                        prepared_relative_l2_error=r['relative_l2_error'],
                        triton_max_abs_error=r['triton_max_abs_error'],
                        envelope_pass=r['v11_envelope_pass']) for r in probe_rows if r not in bad]
    else:
        missing, samples = [], []
    peak = max((b.get('peak_allocated_bytes', 0) for b in blocks), default=None)
    if peak is not None and not _finite_nonnegative(peak):
        reasons.append('invalid peak allocation')
    epoch = record.get('direct_epoch', {})
    event_ms = epoch.get('event_median_ms')
    if not _finite_nonnegative(event_ms):
        reasons.append('direct complete-call epoch timing absent')
    maxima = {key: max((r[key] for r in probe_rows if r not in bad), default=None)
              for key in ('new_max_abs_error', 'relative_l2_error',
                          'actual_output_max_abs_error', 'actual_output_relative_l2_error')}
    return dict(status='failed' if reasons else 'qualified', canonical_arm=method,
                scope=scope, reasons=reasons, reached_phases=phase_counts,
                missing_operator_samples=[dict(layer=l, phase=p) for l, p in missing],
                operator_samples=samples, operator_error_maxima=maxima,
                diagnostic_twin=diag.get('status', 'missing'),
                prepared_support_floor=diag.get('prepared_support_floor', {}).get('status', 'N/A'),
                direct_epoch_event_median_ms=event_ms, peak_allocated_bytes=peak,
                accepted_blocks=len(blocks), no_new_jit=bool(blocks) and not any(
                    'JIT' in x for x in reasons))


def summarize(paths):
    hosts, seen = [], set()
    for path in paths:
        raw = Path(path).read_bytes()
        profile = json.loads(raw)
        if profile.get('schema') != 'v20_direct_full_forward_v1':
            raise ValueError(f'invalid profile schema: {path}')
        identity = profile.get('runtime_identity', {})
        host_key = (identity.get('hostname'), identity.get('gpu_uuid'))
        if host_key in seen:
            raise ValueError(f'duplicate host/GPU profile: {host_key}')
        seen.add(host_key)
        arms = {arm['name']: arm for arm in profile.get('arms', [])}
        targets, processed = [], set()
        for target in profile.get('targets', {}).values():
            key = (target.get('dataset'), target.get('id'), target.get('canvas'))
            redacted = dict(dataset=target.get('dataset'), id_sha256=digest(str(target.get('id')).encode()),
                            canvas=target.get('canvas'), requested_call=target.get('requested_call'),
                            resolution=target.get('resolution'))
            if target.get('resolution', {}).get('missing'):
                redacted.update(status='missing', reason='native target call or canvas absent')
                targets.append(redacted)
                continue
            if key in processed:
                redacted['shared_canvas_profile'] = True
                targets.append(redacted)
                continue
            processed.add(key)
            canvas = int(target['canvas'])
            native_names = [name for name, arm in arms.items() if canonical_arm(arm) == 'D_native']
            boundary_rows = {}
            for boundary_name in ('model_forward', 'denoising_step'):
                label, seq = selected_sequence(target, boundary_name)
                if seq is None:
                    boundary_rows[boundary_name] = dict(status='missing', reason='boundary sequence absent')
                    continue
                drifts = seq.get('native_bracket_drift', [])
                boundary_row = dict(status='qualified', sequence=label,
                    reached_calls=seq.get('reached_calls'), requested_calls=seq.get('requested_calls'),
                    native_bracket_drift=drifts,
                    native_bracket_relative_drift_max=max((abs(d['close_median_ms'] /
                        d['open_median_ms'] - 1) for d in drifts
                        if d.get('open_median_ms', 0) > 0), default=None),
                    arms={name: evaluate_arm(name, arm, seq, canvas=canvas,
                        scope=arm.get('config', {}).get('v20_scope'))
                          for name, arm in arms.items()})
                if len(native_names) == 1:
                    native_ms = boundary_row['arms'][native_names[0]].get('direct_epoch_event_median_ms')
                    for arm_row in boundary_row['arms'].values():
                        cost = arm_row.get('direct_epoch_event_median_ms')
                        arm_row['arm_over_native_event_ratio'] = (cost / native_ms if
                            _finite_nonnegative(native_ms) and native_ms > 0 and
                            _finite_nonnegative(cost) else None)
                if any(x['status'] == 'failed' for x in boundary_row['arms'].values()):
                    boundary_row['status'] = 'failed'
                elif any(x['status'] == 'missing' for x in boundary_row['arms'].values()):
                    boundary_row['status'] = 'missing'
                boundary_rows[boundary_name] = boundary_row
            redacted['boundaries'] = boundary_rows
            redacted['status'] = ('failed' if any(b['status'] == 'failed' for b in boundary_rows.values())
                                  else 'missing' if any(b['status'] == 'missing' for b in boundary_rows.values())
                                  else 'qualified')
            targets.append(redacted)
        hosts.append(dict(profile_sha256=digest(raw), source_sha256=source_receipt(profile),
                          hostname=identity.get('hostname'), gpu_uuid=identity.get('gpu_uuid'),
                          model_revision=profile.get('revision'), operator_probe=profile.get('operator_probe', False),
                          counter_twins=profile.get('counter_twins', False),
                          prepared_support_floor=profile.get('prepared_support_floor', False),
                          targets=targets))
    checks = [t['status'] for h in hosts for t in h['targets'] if not t.get('shared_canvas_profile')]
    status = ('failed' if 'failed' in checks else 'missing' if 'missing' in checks or not checks
              else 'qualified')
    return dict(schema='v20_qualification_gate_v1', status=status, hosts=hosts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(f"{report['status']}: {len(report['hosts'])} host profile(s)")
    return {'qualified': 0, 'failed': 1, 'missing': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
