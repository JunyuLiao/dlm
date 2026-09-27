"""CPU summary of bounded v21 numerical diagnostics; makes no promotion decision.

Usage: python -m scripts.v21_summarize_diagnostic --input host1.json host2.json
       --out-dir results/v21_diagnostic_summary

Only scalar diagnostics, hashes, statuses, and safe stage-error locations are
read into the output. Missing and failed states remain explicit. Operator
precision ratios are diagnostic comparisons, not deployment gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics


SCHEMA = 'v21_numerical_diagnostic_v1'
FAILURE_SCHEMA = 'v21_numerical_diagnostic_failure_v1'
SUMMARY_SCHEMA = 'v21_numerical_diagnostic_summary_v1'
ARM_NAMES = ('D_native', 'D_matched_legacy', 'D_matched_new', 'M3_R3_legacy',
             'M3_R3_new', 'M3_R3_layout', 'M3_R3_combined')
PAIR_NAMES = (('D_matched_legacy', 'D_matched_new'),
              ('M3_R3_legacy', 'M3_R3_new'))
PROTOCOL = (Path(__file__).resolve().parents[1] / 'results' /
            'fan_m1_m3_multidataset_20260927' / 'frozen_protocol.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_states(protocol):
    rows = []
    for dataset in ('aime26', 'longbench_v2', 'ruler4k'):
        ids = protocol['ids'][dataset][:2]
        if len(ids) != 2:
            raise ValueError(f'protocol lacks two {dataset} IDs')
        for number, id_ in enumerate(ids):
            canvas = 1 if number == 1 and dataset != 'ruler4k' else 0
            calls = (0, 3, 6) if canvas == 1 else (0, 1, 3)
            for call in calls:
                rows.append(dict(dataset=dataset, id=id_, canvas=canvas, call_index=call))
    assert len(rows) == 18
    return rows


def state_key(dataset, id_, canvas, call):
    return f'{dataset}|{id_}|canvas{canvas}|call{call}'


def _metric(row, name):
    value = row.get(name) if isinstance(row, dict) else None
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _path(row, *keys):
    for key in keys:
        if not isinstance(row, dict):
            return None
        row = row.get(key)
    return row


def _scalar_metrics(row):
    if not isinstance(row, dict):
        return None
    keys = ('elements', 'finite_comparison', 'nonfinite_native', 'nonfinite_other',
            'native_norm_denominator', 'reference_norm_denominator', 'relative_l2',
            'max_abs', 'p99_abs', 'p99_method', 'p99_sample_stride',
            'p99_sample_elements', 'top1_mismatch_positions')
    return {key: row[key] for key in keys if key in row}


def _operator(record):
    if not isinstance(record, dict):
        return None
    metrics = ('native_vs_full_fp32', 'legacy_vs_full_fp32',
               'fp32_bf16pv_vs_full_fp32', 'legacy_vs_native',
               'fp32_bf16pv_vs_native')
    return dict(qkv=record.get('qkv'),
                metrics={name: _scalar_metrics(record.get(name)) for name in metrics},
                invalid_legacy=record.get('invalid_legacy'),
                invalid_new=record.get('invalid_new'))


def _support_operator(record):
    if not isinstance(record, dict):
        return None
    metrics = ('legacy_vs_full_fp32', 'fp32_bf16pv_vs_full_fp32', 'old_vs_new')
    return dict(support_digest=record.get('support_digest'),
                eligible_tiles=record.get('eligible_tiles'),
                skipped_tiles=record.get('skipped_tiles'),
                metrics={name: _scalar_metrics(record.get(name)) for name in metrics},
                invalid_legacy=record.get('invalid_legacy'),
                invalid_new=record.get('invalid_new'))


def _signal(record):
    if not isinstance(record, dict):
        return None
    keys = ('processor_type', 'stopper_type', 'confidence_threshold',
            'processed_entropy', 'confident', 'stable', 'native_criterion_stop',
            'prior_finished', 'effective_stop', 'top_digest')
    return {key: record[key] for key in keys if key in record}


def _arm_state(arm, call):
    if not isinstance(arm, dict):
        return dict(status='absent')
    if arm.get('status') != 'ok':
        return dict(status=arm.get('status', 'unknown'), error_type=arm.get('error_type'))
    target = arm.get('targets', {}).get(call, {})
    return dict(status='ok', output_digest=arm.get('output_digests', {}).get(call),
                input_digest=arm.get('input_digests', {}).get(call),
                output_score_precision=arm.get('output_score_precision'),
                output_layout=arm.get('output_layout'),
                output_precision_extra_qk_elements_upper_bound=
                    arm.get('output_precision_extra_qk_elements_upper_bound'),
                support_digest=arm.get('support_digests', {}).get(call),
                support_phase=arm.get('support_phases', {}).get(call),
                layer5_output_storage=arm.get('layer5_output_storage', {}).get(call),
                logits_vs_native=_scalar_metrics(target.get('logits_vs_native')),
                same_history_signal=_signal(target.get('same_history_signal')),
                step_vs_native=target.get('step_vs_native'),
                full_step=arm.get('step', {}).get(call) if isinstance(arm.get('step'), dict) else None,
                same_support_operator=_support_operator(target.get('same_support_operator')))


def _state(group, call, provenance):
    target = group.get('targets', {}).get(call, {})
    layout = {}
    for pair, rows in (group.get('layout_exactness') or {}).items():
        if isinstance(rows, dict) and call in rows:
            layout[pair] = rows[call]
    m3 = (group.get('m3_support_identity') or {}).get(call)
    return dict(dataset=group['dataset'], id=group['id'], canvas=int(group['canvas']),
                call_index=int(call), status=target.get('status', 'unreported'),
                resolution=target.get('resolution'),
                host=provenance['hostname'], gpu_uuid=provenance['gpu_uuid'],
                input_sha256=provenance['input_sha256'],
                native_output_digest=group.get('native_output_digests', {}).get(call),
                native_input_digest=group.get('native_input_digests', {}).get(call),
                native_signal=_signal(target.get('native_signal')),
                native_full_step=(group.get('native_step') or {}).get(call),
                operator=_operator(target.get('operator')),
                same_support_operator=_support_operator(_path(group, 'arms', 'M3_R3_legacy',
                                                               'targets', call, 'same_support_operator')),
                arms={name: _arm_state(group.get('arms', {}).get(name), call)
                      for name in ARM_NAMES[1:]},
                layout_exactness=layout,
                m3_support_identity=m3)


def _input_provenance(path, report, digest):
    runtime = report.get('runtime_identity', {})
    return dict(path=str(Path(path).resolve()), input_sha256=digest,
                schema=report.get('schema'), hostname=runtime.get('hostname'),
                gpu_uuid=runtime.get('gpu_uuid'), model=report.get('model'),
                revision=report.get('revision'), seed=report.get('seed'),
                manifest_sha256=_path(report, 'preflight', 'manifest_sha256'),
                protocol_sha256=report.get('protocol_sha256'),
                source_sha256=report.get('source_sha256'),
                native_kernel_trace=report.get('native_kernel_trace'),
                trace_status=report.get('trace_status'),
                arm_fingerprints={a.get('name'): a.get('fingerprint')
                                  for a in report.get('arms', []) if isinstance(a, dict)})


def _pair(old, new):
    if old is None or new is None:
        return None
    ratio = None if old == 0 else new / old
    return dict(old=old, new=new, new_over_old=ratio,
                direction=('improved' if new < old else 'worse' if new > old else 'equal'),
                zero_old_denominator=old == 0)


def _ratio_summary(pairs):
    valid = [p for p in pairs if p is not None]
    ratios = [p['new_over_old'] for p in valid if p['new_over_old'] is not None]
    counts = Counter(p['direction'] for p in valid)
    return dict(compared=len(valid), not_comparable=len(pairs)-len(valid),
                improved=counts['improved'], worse=counts['worse'], equal=counts['equal'],
                zero_old_denominator=sum(p['zero_old_denominator'] for p in valid),
                ratio_median=(statistics.median(ratios) if ratios else None),
                ratio_min=(min(ratios) if ratios else None),
                ratio_max=(max(ratios) if ratios else None))


def aggregate(states):
    numeric, support = [], []
    logits = {f'{old}|{new}': [] for old, new in PAIR_NAMES}
    top1 = {key: Counter() for key in logits}
    signal = {key: Counter() for key in logits}
    step = {key: Counter() for key in logits}
    layout = Counter()
    m3_support = Counter()
    statuses = Counter(state['status'] for state in states)
    for state in states:
        metrics = _path(state, 'operator', 'metrics') or {}
        numeric.append(_pair(_metric(metrics.get('legacy_vs_full_fp32'), 'relative_l2'),
                             _metric(metrics.get('fp32_bf16pv_vs_full_fp32'), 'relative_l2')))
        same = _path(state, 'same_support_operator', 'metrics') or {}
        support.append(_pair(_metric(same.get('legacy_vs_full_fp32'), 'relative_l2'),
                             _metric(same.get('fp32_bf16pv_vs_full_fp32'), 'relative_l2')))
        for old, new in PAIR_NAMES:
            name = f'{old}|{new}'
            o, n = state['arms'].get(old, {}), state['arms'].get(new, {})
            om, nm = o.get('logits_vs_native'), n.get('logits_vs_native')
            logits[name].append(_pair(_metric(om, 'relative_l2'), _metric(nm, 'relative_l2')))
            old_top, new_top = _metric(om, 'top1_mismatch_positions'), _metric(nm, 'top1_mismatch_positions')
            if old_top is not None and new_top is not None:
                top1[name]['improved' if new_top < old_top else 'worse' if new_top > old_top else 'equal'] += 1
            osig, nsig = o.get('same_history_signal'), n.get('same_history_signal')
            if isinstance(osig, dict) and isinstance(nsig, dict):
                for field in ('effective_stop', 'confident', 'stable'):
                    if field in osig and field in nsig:
                        signal[name][field + ('_changed' if osig[field] != nsig[field] else '_equal')] += 1
            for label, row in (('old', o), ('new', n)):
                changes = row.get('step_vs_native')
                if isinstance(changes, dict):
                    for field, changed in changes.items():
                        step[name][f'{label}_{field}_{"true" if changed else "false"}'] += 1
        for pair, record in state.get('layout_exactness', {}).items():
            if isinstance(record, dict):
                layout[pair + ('_exact' if record.get('exact_output_elements') and
                              record.get('identical_support') else '_mismatch')] += 1
        m3 = state.get('m3_support_identity')
        if isinstance(m3, dict):
            m3_support['identical' if m3.get('identical_bitmaps') else 'different'] += 1
    return dict(states=len(states), statuses=dict(statuses),
                operator_full_fp32_relative_l2=_ratio_summary(numeric),
                same_frozen_M3_support_relative_l2=_ratio_summary(support),
                logits_relative_l2_vs_native={key: _ratio_summary(rows) for key, rows in logits.items()},
                logits_top1_mismatch_direction={key: dict(counts) for key, counts in top1.items()},
                same_history_signal_pair_counts={key: dict(counts) for key, counts in signal.items()},
                full_step_vs_native_counts={key: dict(counts) for key, counts in step.items()},
                layout_parity_counts=dict(layout),
                M3_bitmap_identity_descriptive=dict(m3_support))


def summarize(paths):
    protocol = json.loads(PROTOCOL.read_text())
    expected = expected_states(protocol)
    expected_keys = {state_key(**dict(dataset=x['dataset'], id_=x['id'], canvas=x['canvas'],
                                     call=x['call_index'])) for x in expected}
    seen, provenance, failures, stage_errors = {}, [], [], []
    if not paths:
        raise ValueError('at least one diagnostic input required')
    for path in paths:
        digest = sha256(path)
        report = json.loads(Path(path).read_text())
        schema = report.get('schema')
        if schema == FAILURE_SCHEMA:
            failures.append(dict(path=str(Path(path).resolve()), input_sha256=digest,
                                 error_type=report.get('error_type'),
                                 config_sha256=report.get('config_sha256'),
                                 partial_path=report.get('partial_path')))
            continue
        if schema != SCHEMA:
            raise ValueError(f'unknown diagnostic schema in {path}: {schema}')
        if [a.get('name') for a in report.get('arms', [])] != list(ARM_NAMES):
            raise ValueError(f'arm list drift in {path}')
        if report.get('seed') != 101 or report.get('protocol_sha256') != sha256(PROTOCOL):
            raise ValueError(f'frozen seed/protocol drift in {path}')
        source = _input_provenance(path, report, digest)
        provenance.append(source)
        for error in report.get('stage_errors', []):
            stage_errors.append(dict(input_sha256=digest, host=source['hostname'],
                                     group=error.get('group'), stage=error.get('stage'),
                                     error_type=error.get('error_type'),
                                     code_location=error.get('code_location')))
        for group in report.get('groups', {}).values():
            if not all(key in group for key in ('dataset', 'id', 'canvas')):
                raise ValueError(f'group identity incomplete in {path}')
            for call in group.get('targets', {}):
                state = _state(group, call, source)
                key = state_key(state['dataset'], state['id'], state['canvas'], state['call_index'])
                if key not in expected_keys:
                    raise ValueError(f'unfrozen target {key} in {path}')
                if key in seen:
                    raise ValueError(f'duplicate target across inputs: {key}')
                seen[key] = state
    states = []
    for target in expected:
        key = state_key(target['dataset'], target['id'], target['canvas'], target['call_index'])
        states.append(seen.get(key, dict(**target, status='not_reported', host=None,
                                         gpu_uuid=None, arms={})))
    datasets = {name: aggregate([state for state in states if state['dataset'] == name])
                for name in ('aime26', 'longbench_v2', 'ruler4k')}
    hosts = {name: aggregate([state for state in states if state.get('host') == name])
             for name in sorted({state.get('host') for state in states if state.get('host')})}
    return dict(schema=SUMMARY_SCHEMA, diagnostic_only=True, promotion_decision=None,
                protocol_path=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
                summarizer_path=str(Path(__file__).resolve()),
                summarizer_sha256=sha256(__file__),
                expected_states=18, reported_states=len(seen),
                unreported_states=18-len(seen),
                input_provenance=provenance, failure_receipts=failures,
                stage_errors=stage_errors, states=states,
                aggregate=dict(overall=aggregate(states), per_dataset=datasets, per_host=hosts),
                ratio_definition='new relative-L2 / old relative-L2; reference denominator is '
                                 'native logits or independent full-FP32 attention as labeled; '
                                 'zero old error has no numeric ratio',
                p99_definition='source diagnostic deterministic stride sample; max and L2 use full tensors')


def markdown(summary):
    lines = ['# v21 numerical diagnostic summary', '',
             'Diagnostic only. No precision promotion decision is made here.', '',
             f"Reported states: {summary['reported_states']}/18; unreported: {summary['unreported_states']}; "
             f"stage errors: {len(summary['stage_errors'])}; failure receipts: {len(summary['failure_receipts'])}.", '',
             '| Dataset | ID | Canvas | Call | Host | Status | Operator new/old | '
             'Frozen M3 new/old | D logits new/old | M3 logits new/old |',
             '|---|---|---:|---:|---|---|---:|---:|---:|---:|']
    for state in summary['states']:
        operator = _path(state, 'operator', 'metrics') or {}
        op = _pair(_metric(operator.get('legacy_vs_full_fp32'), 'relative_l2'),
                   _metric(operator.get('fp32_bf16pv_vs_full_fp32'), 'relative_l2'))
        support = _path(state, 'same_support_operator', 'metrics') or {}
        same = _pair(_metric(support.get('legacy_vs_full_fp32'), 'relative_l2'),
                     _metric(support.get('fp32_bf16pv_vs_full_fp32'), 'relative_l2'))
        ratios = []
        for old, new in PAIR_NAMES:
            pair = _pair(_metric(_path(state, 'arms', old, 'logits_vs_native'), 'relative_l2'),
                         _metric(_path(state, 'arms', new, 'logits_vs_native'), 'relative_l2'))
            ratios.append(pair)
        def fmt(pair):
            return 'N/A' if pair is None or pair['new_over_old'] is None else f"{pair['new_over_old']:.4g}"
        lines.append('| ' + ' | '.join([state['dataset'], state['id'], str(state['canvas']),
                                       str(state['call_index']), str(state.get('host') or 'N/A'),
                                       state['status'], fmt(op), fmt(same),
                                       fmt(ratios[0]), fmt(ratios[1])]) + ' |')
    lines += ['', 'Aggregate directions (new relative-L2 versus old):', '']
    for label, item in (('Operator full FP32', summary['aggregate']['overall']['operator_full_fp32_relative_l2']),
                        ('Frozen M3 support', summary['aggregate']['overall']['same_frozen_M3_support_relative_l2'])):
        lines.append(f"- {label}: {item['improved']} improved, {item['worse']} worse, "
                     f"{item['equal']} equal, {item['not_comparable']} unavailable; "
                     f"median ratio {item['ratio_median'] if item['ratio_median'] is not None else 'N/A'}.")
    lines += ['', 'Per-state metrics, failures, layout parity, and source identities are in '
              '`numeric_diagnostic_summary.json`.', '']
    return '\n'.join(lines)


def state_csv(summary):
    stream = io.StringIO()
    fields = ['dataset', 'id', 'canvas', 'call_index', 'host', 'status',
              'operator_native_rel_l2', 'operator_legacy_rel_l2', 'operator_new_rel_l2',
              'operator_old_max_abs', 'operator_new_max_abs',
              'operator_old_p99_abs', 'operator_new_p99_abs',
              'same_support_old_rel_l2', 'same_support_new_rel_l2',
              'D_old_logits_rel_l2', 'D_new_logits_rel_l2',
              'M3_old_logits_rel_l2', 'M3_new_logits_rel_l2',
              'D_old_top1_mismatch', 'D_new_top1_mismatch',
              'M3_old_top1_mismatch', 'M3_new_top1_mismatch',
              'D_old_stop', 'D_new_stop', 'M3_old_stop', 'M3_new_stop',
              'D_old_step_stop_changed', 'D_new_step_stop_changed',
              'M3_old_step_stop_changed', 'M3_new_step_stop_changed',
              'M3_bitmap_identical_descriptive', 'layout_legacy_exact', 'layout_combined_exact']
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for state in summary['states']:
        op = _path(state, 'operator', 'metrics') or {}
        old = op.get('legacy_vs_full_fp32')
        new = op.get('fp32_bf16pv_vs_full_fp32')
        same = _path(state, 'same_support_operator', 'metrics') or {}
        def arm_metric(arm, name):
            return _metric(_path(state, 'arms', arm, 'logits_vs_native'), name)
        layouts = state.get('layout_exactness', {})
        def exact(pair):
            row = layouts.get(pair)
            return (None if not isinstance(row, dict) else
                    bool(row.get('exact_output_elements') and row.get('identical_support')))
        writer.writerow(dict(dataset=state['dataset'], id=state['id'], canvas=state['canvas'],
                             call_index=state['call_index'], host=state.get('host'), status=state['status'],
                             operator_native_rel_l2=_metric(op.get('native_vs_full_fp32'), 'relative_l2'),
                             operator_legacy_rel_l2=_metric(old, 'relative_l2'),
                             operator_new_rel_l2=_metric(new, 'relative_l2'),
                             operator_old_max_abs=_metric(old, 'max_abs'),
                             operator_new_max_abs=_metric(new, 'max_abs'),
                             operator_old_p99_abs=_metric(old, 'p99_abs'),
                             operator_new_p99_abs=_metric(new, 'p99_abs'),
                             same_support_old_rel_l2=_metric(same.get('legacy_vs_full_fp32'), 'relative_l2'),
                             same_support_new_rel_l2=_metric(same.get('fp32_bf16pv_vs_full_fp32'), 'relative_l2'),
                             D_old_logits_rel_l2=arm_metric('D_matched_legacy', 'relative_l2'),
                             D_new_logits_rel_l2=arm_metric('D_matched_new', 'relative_l2'),
                             M3_old_logits_rel_l2=arm_metric('M3_R3_legacy', 'relative_l2'),
                             M3_new_logits_rel_l2=arm_metric('M3_R3_new', 'relative_l2'),
                             D_old_top1_mismatch=arm_metric('D_matched_legacy', 'top1_mismatch_positions'),
                             D_new_top1_mismatch=arm_metric('D_matched_new', 'top1_mismatch_positions'),
                             M3_old_top1_mismatch=arm_metric('M3_R3_legacy', 'top1_mismatch_positions'),
                             M3_new_top1_mismatch=arm_metric('M3_R3_new', 'top1_mismatch_positions'),
                             D_old_stop=_path(state, 'arms', 'D_matched_legacy', 'same_history_signal', 'effective_stop'),
                             D_new_stop=_path(state, 'arms', 'D_matched_new', 'same_history_signal', 'effective_stop'),
                             M3_old_stop=_path(state, 'arms', 'M3_R3_legacy', 'same_history_signal', 'effective_stop'),
                             M3_new_stop=_path(state, 'arms', 'M3_R3_new', 'same_history_signal', 'effective_stop'),
                             D_old_step_stop_changed=_path(state, 'arms', 'D_matched_legacy', 'step_vs_native', 'return_stop_changed'),
                             D_new_step_stop_changed=_path(state, 'arms', 'D_matched_new', 'step_vs_native', 'return_stop_changed'),
                             M3_old_step_stop_changed=_path(state, 'arms', 'M3_R3_legacy', 'step_vs_native', 'return_stop_changed'),
                             M3_new_step_stop_changed=_path(state, 'arms', 'M3_R3_new', 'step_vs_native', 'return_stop_changed'),
                             M3_bitmap_identical_descriptive=_path(state, 'm3_support_identity', 'identical_bitmaps'),
                             layout_legacy_exact=exact('M3_R3_legacy|M3_R3_layout'),
                             layout_combined_exact=exact('M3_R3_new|M3_R3_combined')))
    return stream.getvalue()


def atomic_write(path, content):
    path = Path(path)
    temp = path.with_name(path.name + '.writing')
    with temp.open('x', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.input)
    atomic_write(args.out_dir / 'numeric_diagnostic_summary.json',
                 json.dumps(summary, indent=2, sort_keys=True) + '\n')
    atomic_write(args.out_dir / 'numeric_diagnostic_summary.md', markdown(summary))
    atomic_write(args.out_dir / 'numeric_diagnostic_states.csv', state_csv(summary))
    print(json.dumps(dict(schema=SUMMARY_SCHEMA, out_dir=str(args.out_dir),
                          reported_states=summary['reported_states'],
                          stage_errors=len(summary['stage_errors']))))


if __name__ == '__main__':
    main()
