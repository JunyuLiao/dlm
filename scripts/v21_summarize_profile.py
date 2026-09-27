"""CPU summary of v21 direct full-forward and denoising-step profiles.

Usage: python -m scripts.v21_summarize_profile --input host1.json host2.json
       --out-dir results/profile_cost_001

One measured canvas/boundary/N4-or-N16 cell is counted once even when the v20
report aliases it under several requested call targets. This summarizes
accepted direct timing and separate untimed probes; it makes no promotion or
deployment decision.
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


SCHEMA = 'v21_direct_output_modes_v1'
FAILURE_SCHEMA = 'v21_profile_failure_v1'
OUT_SCHEMA = 'v21_profile_cost_summary_v1'
ARMS = ('D_native', 'legacy', 'layout_only', 'numeric_only', 'combined')
BOUNDARIES = ('model_forward', 'denoising_step')
LENGTHS = ('N4', 'N16')
PROTOCOL = (Path(__file__).resolve().parents[1] / 'results' /
            'fan_m1_m3_multidataset_20260927' / 'frozen_protocol.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_targets(protocol):
    result = []
    for dataset in ('aime26', 'longbench_v2', 'ruler4k'):
        for index, id_ in enumerate(protocol['ids'][dataset][:2]):
            canvas = 1 if index == 1 and dataset != 'ruler4k' else 0
            calls = (0, 3, 6) if canvas == 1 else (0, 1, 3)
            result.extend(dict(dataset=dataset, id=id_, canvas=canvas, call_index=call)
                          for call in calls)
    if len(result) != 18:
        raise ValueError('frozen v20 protocol does not define eighteen targets')
    return result


def target_key(row):
    return f"{row['dataset']}|{row['id']}|canvas{row['canvas']}|call{row['call_index']}"


def _metric(row, key):
    value = row.get(key) if isinstance(row, dict) else None
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _ratio(numerator, denominator):
    return (numerator / denominator if numerator is not None and denominator not in (None, 0)
            else None)


def _physical(probe, *, precision):
    if not isinstance(probe, dict):
        return dict(status='absent')
    physical = probe.get('physical')
    if probe.get('status') != 'qualified' or not isinstance(physical, dict):
        return dict(status=probe.get('status', 'unknown'), reason=probe.get('reason'),
                    copy_allocation_trace=probe.get('copy_allocation_trace'))
    rows = physical.get('rows', [])
    totals = Counter()
    phases = Counter()
    for row in rows:
        phase = row.get('phase')
        phases[phase] += 1
        whole = row.get('by_segment', {}).get('whole', {})
        for key in ('eligible_pairs', 'executed_qk_pairs', 'skipped_qk_pairs',
                    'executed_pv_pairs', 'skipped_pv_pairs',
                    'executed_qk_multiply_accumulates', 'executed_pv_multiply_accumulates'):
            totals[key] += int(whole.get(key, 0))
        if precision == 'fp32_scores_bf16_pv' and phase == 'A':
            # The counter twin charges historical full QK at A. v21 numeric
            # output also recomputes QK on retained tiles; its legal-pair
            # count equals this row's retained PV pairs.
            totals['precision_anchor_extra_qk_pairs'] += int(whole.get('executed_pv_pairs', 0))
            totals['precision_anchor_extra_qk_multiply_accumulates'] += int(
                whole.get('executed_pv_multiply_accumulates', 0))
    totals['corrected_executed_qk_pairs'] = (totals['executed_qk_pairs'] +
                                            totals['precision_anchor_extra_qk_pairs'])
    totals['corrected_executed_qk_multiply_accumulates'] = (
        totals['executed_qk_multiply_accumulates'] +
        totals['precision_anchor_extra_qk_multiply_accumulates'])
    eligible = totals['eligible_pairs']
    return dict(status='qualified', calls=physical.get('calls'),
                phase_counts=dict(phases), legal_pair_counts=dict(totals),
                eligible_pairs_denominator=eligible,
                executed_qk_over_eligible=_ratio(totals['corrected_executed_qk_pairs'], eligible),
                executed_pv_over_eligible=_ratio(totals['executed_pv_pairs'], eligible),
                counter_scope='actual legal q-k pairs in routed layers; numeric A includes '
                              'a separate retained-support QK pass; not Q/K/V projection work',
                copy_allocation_trace=probe.get('copy_allocation_trace'))


def _digests(arm):
    summary = arm.get('summary') or {}
    values = summary.get('input_output_digests')
    return values if isinstance(values, list) else None


def _digest_parity(left, right):
    a, b = _digests(left), _digests(right)
    if a is None or b is None:
        return dict(status='unavailable')
    if len(a) != len(b):
        return dict(status='length_mismatch', left_calls=len(a), right_calls=len(b))
    return dict(status='compared', calls=len(a),
                input_exact=[x[0] == y[0] for x, y in zip(a, b)],
                output_exact=[x[1] == y[1] for x, y in zip(a, b)],
                all_input_exact=all(x[0] == y[0] for x, y in zip(a, b)),
                all_output_exact=all(x[1] == y[1] for x, y in zip(a, b)))


def _physical_count_parity(left, right):
    a, b = left.get('legal_pair_counts'), right.get('legal_pair_counts')
    if not isinstance(a, dict) or not isinstance(b, dict):
        return dict(status='unavailable')
    fields = ('eligible_pairs', 'executed_qk_pairs', 'skipped_qk_pairs',
              'executed_pv_pairs', 'skipped_pv_pairs')
    return dict(status='count_only', all_equal=all(a.get(field) == b.get(field) for field in fields),
                fields={field: a.get(field) == b.get(field) for field in fields},
                exact_bitmap_available=False)


def _arm(arm, probe, reached, *, precision):
    summary = arm.get('summary') or {}
    epoch = arm.get('direct_epoch') or {}
    blocks = arm.get('blocks') or []
    event = _metric(summary, 'event_sum_median_ms')
    wall = _metric(summary, 'wall_sum_median_ms')
    epoch_event = _metric(epoch, 'event_median_ms')
    epoch_wall = _metric(epoch, 'wall_median_ms')
    peaks = [_metric(block, 'peak_allocated_bytes') for block in blocks]
    peaks = [value for value in peaks if value is not None]
    return dict(status='measured' if event is not None else 'missing',
                direct_event_sum_median_ms=event, direct_wall_sum_median_ms=wall,
                direct_event_per_reached_call_ms=_ratio(event, reached),
                direct_wall_per_reached_call_ms=_ratio(wall, reached),
                direct_epoch_event_median_ms=epoch_event,
                direct_epoch_wall_median_ms=epoch_wall,
                accepted_peak_allocated_bytes_median=(statistics.median(peaks) if peaks else None),
                accepted_peak_allocated_bytes_max=(max(peaks) if peaks else None),
                accepted_blocks=len(blocks),
                sequence_calls=summary.get('sequence_calls'),
                input_output_digests=_digests(arm),
                physical=_physical(probe, precision=precision))


def _bracket(rows):
    if not isinstance(rows, list):
        return dict(status='missing')
    observations = []
    for row in rows:
        opening, closing = _metric(row, 'open_median_ms'), _metric(row, 'close_median_ms')
        observations.append(dict(block=row.get('block'), open_median_ms=opening,
                                 close_median_ms=closing,
                                 close_over_open=_ratio(closing, opening),
                                 absolute_change_ms=(closing-opening if None not in (opening, closing) else None)))
    ratios = [x['close_over_open'] for x in observations if x['close_over_open'] is not None]
    return dict(status='reported', observations=observations,
                close_over_open_min=(min(ratios) if ratios else None),
                close_over_open_max=(max(ratios) if ratios else None),
                close_over_open_median=(statistics.median(ratios) if ratios else None))


def _cell(boundary, length, measured):
    reached = measured.get('reached_calls')
    requested = measured.get('requested_calls')
    if type(reached) is not int or reached <= 0 or requested not in (4, 16):
        raise ValueError('profile cell lacks reached/requested sequence length')
    modes = {
        'D_native': 'legacy_bf16_scores',
        'legacy': 'legacy_bf16_scores',
        'layout_only': 'legacy_bf16_scores',
        'numeric_only': 'fp32_scores_bf16_pv',
        'combined': 'fp32_scores_bf16_pv',
    }
    arms = {name: _arm(measured.get('arms', {}).get(name, {}),
                       measured.get('diagnostic_replays', {}).get(name), reached,
                       precision=modes[name]) for name in ARMS}
    native, legacy = arms['D_native'], arms['legacy']
    comparisons = {}
    for name in ARMS[1:]:
        arm = arms[name]
        comparisons[name] = dict(
            direct_event_vs_native=_ratio(arm['direct_event_sum_median_ms'],
                                          native['direct_event_sum_median_ms']),
            direct_wall_vs_native=_ratio(arm['direct_wall_sum_median_ms'],
                                         native['direct_wall_sum_median_ms']),
            direct_epoch_event_vs_native=_ratio(arm['direct_epoch_event_median_ms'],
                                                native['direct_epoch_event_median_ms']),
            direct_epoch_wall_vs_native=_ratio(arm['direct_epoch_wall_median_ms'],
                                               native['direct_epoch_wall_median_ms']),
            direct_event_vs_legacy=_ratio(arm['direct_event_sum_median_ms'],
                                          legacy['direct_event_sum_median_ms']),
            direct_epoch_event_vs_legacy=_ratio(arm['direct_epoch_event_median_ms'],
                                                legacy['direct_epoch_event_median_ms']))
    return dict(boundary=boundary, sequence_label=length, requested_calls=requested,
                reached_calls=reached, call_count_is_actual=True,
                timing_boundary=('model.forward decoder/logits; state.begin excluded, sampler excluded'
                                 if boundary == 'model_forward' else
                                 'native _denoising_step with observe/controller, sampler and stop'),
                native_bracket_drift=_bracket(measured.get('native_bracket_drift')),
                arms=arms, comparisons=comparisons,
                layout_only_vs_legacy_digest=_digest_parity(measured.get('arms', {}).get('legacy', {}),
                                                             measured.get('arms', {}).get('layout_only', {})),
                combined_vs_numeric_digest=_digest_parity(measured.get('arms', {}).get('numeric_only', {}),
                                                           measured.get('arms', {}).get('combined', {})),
                layout_only_vs_legacy_physical_count_parity=_physical_count_parity(
                    arms['legacy']['physical'], arms['layout_only']['physical']),
                combined_vs_numeric_physical_count_parity=_physical_count_parity(
                    arms['numeric_only']['physical'], arms['combined']['physical']))


def _provenance(path, report):
    return dict(input_path=str(Path(path).resolve()), input_sha256=sha256(path),
                schema=report.get('schema'), model=report.get('model'),
                revision=report.get('revision'), manifest_sha256=report.get('manifest_sha256'),
                runtime_identity=report.get('runtime_identity'),
                source_sha256=report.get('source_sha256'),
                scope=report.get('v21_scope'), modes=report.get('v21_modes'),
                arms=[dict(name=a.get('name'), condition=a.get('condition'),
                           plugin=a.get('plugin'), fingerprint=(a.get('config') or {}).get('fingerprint'))
                      for a in report.get('arms', [])])


def summarize(paths):
    if not paths:
        raise ValueError('at least one profile input required')
    frozen = frozen_targets(json.loads(PROTOCOL.read_text()))
    allowed = {target_key(t) for t in frozen}
    provenance, failures, targets, canvases = [], [], {}, {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get('schema') == FAILURE_SCHEMA:
            failures.append(dict(path=str(Path(path).resolve()), sha256=sha256(path),
                                 error_type=report.get('error_type'),
                                 completed_targets=report.get('completed_targets'),
                                 partial_path=report.get('partial_path')))
            continue
        if report.get('schema') != SCHEMA:
            raise ValueError(f'unknown profile schema in {path}: {report.get("schema")}')
        names = [a.get('name') for a in report.get('arms', [])]
        if len(names) != 5 or names[0] != 'D_native' or set(names) != set(ARMS):
            raise ValueError(f'five-arm profile grid drift in {path}')
        if set(report.get('selected_boundaries', [])) != set(BOUNDARIES):
            raise ValueError(f'boundary grid drift in {path}')
        if report.get('v21_scope') != 'GLOBAL_ONLY_NATIVE_LOCAL':
            raise ValueError(f'v21 routed scope drift in {path}')
        source = _provenance(path, report)
        provenance.append(source)
        runtime = report.get('runtime_identity', {})
        host, gpu = runtime.get('hostname'), runtime.get('gpu_uuid')
        if not host or not gpu:
            raise ValueError(f'missing host/GPU identity in {path}')
        for key, target in report.get('targets', {}).items():
            identity = dict(dataset=target['dataset'], id=target['id'],
                            canvas=target['canvas'], call_index=target['requested_call'])
            canonical = target_key(identity)
            if canonical != key or canonical not in allowed:
                raise ValueError(f'unfrozen or mislabeled profile target in {path}: {key}')
            if canonical in targets:
                raise ValueError(f'duplicate target across profile inputs: {canonical}')
            resolution = target.get('resolution') or {}
            item = dict(**identity, host=host, gpu_uuid=gpu,
                        status=('missing' if resolution.get('missing') else 'reached'),
                        resolution=resolution, input_sha256=source['input_sha256'])
            targets[canonical] = item
            canvas_key = (identity['dataset'], identity['id'], identity['canvas'])
            group = canvases.setdefault(canvas_key, dict(dataset=identity['dataset'], id=identity['id'],
                                                         canvas=identity['canvas'], host=host,
                                                         gpu_uuid=gpu, input_sha256=source['input_sha256'],
                                                         requested_targets=[], aliases=[], boundaries=None))
            if group['host'] != host or group['input_sha256'] != source['input_sha256']:
                raise ValueError('same canvas split across profile sources')
            group['requested_targets'].append(item)
            boundary_rows = target.get('boundaries') or {}
            if boundary_rows:
                digest = hashlib.sha256(json.dumps(boundary_rows, sort_keys=True).encode()).hexdigest()
                if group['boundaries'] is None:
                    group['boundaries'] = boundary_rows
                    group['boundary_alias_sha256'] = digest
                    group['canonical_target'] = canonical
                elif digest != group['boundary_alias_sha256']:
                    raise ValueError(f'shared canvas boundary alias differs: {canonical}')
                group['aliases'].append(canonical)
    measured = []
    for group in canvases.values():
        boundary_rows = group.pop('boundaries')
        if boundary_rows is None:
            group['measurement_status'] = 'missing_all_requested_states'
            continue
        group['measurement_status'] = 'measured'
        cells = {}
        for boundary in BOUNDARIES:
            for length in LENGTHS:
                value = boundary_rows.get(boundary, {}).get(length)
                key = f'{boundary}|{length}'
                cells[key] = (_cell(boundary, length, value) if value is not None else
                              dict(boundary=boundary, sequence_label=length,
                                   status='not_measured', reason='not reached or omitted by source'))
                if value is not None:
                    measured.append(dict(dataset=group['dataset'], id=group['id'],
                                         canvas=group['canvas'], host=group['host'],
                                         gpu_uuid=group['gpu_uuid'], **cells[key]))
        group['cells'] = cells
    state_rows = [targets.get(target_key(t), dict(**t, status='not_reported', host=None,
                                                  gpu_uuid=None, resolution=None)) for t in frozen]
    # These are cell counts, never duplicated target aliases or pooled timing.
    aggregate = {}
    for label, subset in [('overall', measured)] + [
            (f'dataset:{d}', [cell for cell in measured if cell['dataset'] == d])
            for d in ('aime26', 'longbench_v2', 'ruler4k')] + [
            (f'host:{h}', [cell for cell in measured if cell['host'] == h])
            for h in sorted({cell['host'] for cell in measured})]:
        by_boundary_length = {}
        for boundary in BOUNDARIES:
            for length in LENGTHS:
                rows = [cell for cell in subset if cell['boundary'] == boundary and cell['sequence_label'] == length]
                modes = {}
                for arm in ARMS:
                    direct = [cell['arms'][arm]['direct_event_sum_median_ms'] for cell in rows]
                    direct = [x for x in direct if x is not None]
                    epoch = [cell['arms'][arm]['direct_epoch_event_median_ms'] for cell in rows]
                    epoch = [x for x in epoch if x is not None]
                    vs_native = ([cell['comparisons'][arm]['direct_event_vs_native'] for cell in rows]
                                 if arm != 'D_native' else [])
                    vs_native = [x for x in vs_native if x is not None]
                    modes[arm] = dict(cells=len(rows), direct_event_sum_median_of_cells_ms=(statistics.median(direct) if direct else None),
                                      direct_epoch_event_median_of_cells_ms=(statistics.median(epoch) if epoch else None),
                                      direct_event_vs_native_median_of_cells=(statistics.median(vs_native) if vs_native else None))
                by_boundary_length[f'{boundary}|{length}'] = dict(cells=len(rows), arms=modes)
        aggregate[label] = by_boundary_length
    return dict(schema=OUT_SCHEMA, promotion_decision=None, diagnostic_only=True,
                protocol_path=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
                summarizer_path=str(Path(__file__).resolve()), summarizer_sha256=sha256(__file__),
                input_provenance=provenance, failure_receipts=failures,
                requested_states=state_rows,
                requested_state_statuses=dict(Counter(row['status'] for row in state_rows)),
                unique_canvases=list(canvases.values()), unique_measured_cells=measured,
                unique_measured_cell_count=len(measured), aggregate=aggregate,
                units=dict(event='CUDA event elapsed stream span with host launch gaps; not GPU-active sum',
                           wall='host wall span',
                           per_call='direct sum median / actual reached calls',
                           direct_epoch='one outer sequence span including fixture restoration; separate from direct sum',
                           QK_PV='native-legal pair-weighted counters from an untimed separate replay',
                           copy='separate profiler operation counts; not precise copy bytes'),
                caution='N4/N16 are requested lengths; reached_calls can be shorter. '
                        'Model forward excludes state.begin; denoising step includes controller, sampler and stop. '
                        'Cross-layout physical equality is count parity; exact bitmap unavailable in this profile.')


def csv_text(summary):
    stream = io.StringIO()
    fields = ['dataset', 'id', 'canvas', 'host', 'boundary', 'sequence_label',
              'requested_calls', 'reached_calls', 'arm', 'direct_event_sum_median_ms',
              'direct_event_per_reached_call_ms', 'direct_epoch_event_median_ms',
              'direct_wall_sum_median_ms', 'direct_epoch_wall_median_ms',
              'direct_event_vs_native', 'direct_event_vs_legacy',
              'direct_epoch_event_vs_native', 'accepted_peak_allocated_bytes_median',
              'eligible_pairs', 'executed_qk_pairs_corrected', 'executed_pv_pairs',
              'precision_anchor_extra_qk_pairs', 'qk_over_eligible', 'pv_over_eligible',
              'copy_trace_status', 'layout_output_exact', 'layout_support_count_equal']
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for cell in summary['unique_measured_cells']:
        for name in ARMS:
            arm = cell['arms'][name]
            physical = arm['physical']
            counts = physical.get('legal_pair_counts') or {}
            comparison = cell['comparisons'].get(name) or {}
            pair = (cell['layout_only_vs_legacy_digest'] if name == 'layout_only' else
                    cell['combined_vs_numeric_digest'] if name == 'combined' else None)
            support = (cell['layout_only_vs_legacy_physical_count_parity'] if name == 'layout_only' else
                       cell['combined_vs_numeric_physical_count_parity'] if name == 'combined' else None)
            writer.writerow(dict(dataset=cell['dataset'], id=cell['id'], canvas=cell['canvas'],
                                 host=cell['host'], boundary=cell['boundary'],
                                 sequence_label=cell['sequence_label'],
                                 requested_calls=cell['requested_calls'], reached_calls=cell['reached_calls'],
                                 arm=name, direct_event_sum_median_ms=arm['direct_event_sum_median_ms'],
                                 direct_event_per_reached_call_ms=arm['direct_event_per_reached_call_ms'],
                                 direct_epoch_event_median_ms=arm['direct_epoch_event_median_ms'],
                                 direct_wall_sum_median_ms=arm['direct_wall_sum_median_ms'],
                                 direct_epoch_wall_median_ms=arm['direct_epoch_wall_median_ms'],
                                 direct_event_vs_native=comparison.get('direct_event_vs_native'),
                                 direct_event_vs_legacy=comparison.get('direct_event_vs_legacy'),
                                 direct_epoch_event_vs_native=comparison.get('direct_epoch_event_vs_native'),
                                 accepted_peak_allocated_bytes_median=arm['accepted_peak_allocated_bytes_median'],
                                 eligible_pairs=counts.get('eligible_pairs'),
                                 executed_qk_pairs_corrected=counts.get('corrected_executed_qk_pairs'),
                                 executed_pv_pairs=counts.get('executed_pv_pairs'),
                                 precision_anchor_extra_qk_pairs=counts.get('precision_anchor_extra_qk_pairs'),
                                 qk_over_eligible=physical.get('executed_qk_over_eligible'),
                                 pv_over_eligible=physical.get('executed_pv_over_eligible'),
                                 copy_trace_status=(physical.get('copy_allocation_trace') or {}).get('status'),
                                 layout_output_exact=(pair or {}).get('all_output_exact'),
                                 layout_support_count_equal=(support or {}).get('all_equal')))
    return stream.getvalue()


def markdown(summary):
    lines = ['# v21 direct profile cost summary', '',
             'CPU summary of direct complete-call timing. No promotion decision.', '',
             f"Requested states: {summary['requested_state_statuses']}; unique measured cells: "
             f"{summary['unique_measured_cell_count']}. Each canvas/boundary/length appears once.", '',
             '| Dataset | ID | Canvas | Host | Boundary | Length | Reached | '
             'Native ms | Legacy/native | Layout/legacy | Numeric/legacy | Combined/legacy |',
             '|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4g}'
    for cell in summary['unique_measured_cells']:
        a, c = cell['arms'], cell['comparisons']
        lines.append('| ' + ' | '.join([
            cell['dataset'], cell['id'], str(cell['canvas']), cell['host'], cell['boundary'],
            cell['sequence_label'], str(cell['reached_calls']),
            fmt(a['D_native']['direct_event_sum_median_ms']),
            fmt(c['legacy']['direct_event_vs_native']),
            fmt(c['layout_only']['direct_event_vs_legacy']),
            fmt(c['numeric_only']['direct_event_vs_legacy']),
            fmt(c['combined']['direct_event_vs_legacy'])]) + ' |')
    lines += ['', 'Direct epoch spans, wall spans, bracket drift, physical pair denominators, '
              'copy traces, accepted peak memory, digest parity, every requested state, and source '
              'identities are in `profile_cost_001.json`. The CSV has one row per unique cell and arm.', '']
    return '\n'.join(lines)


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
    atomic_write(args.out_dir / 'profile_cost_001.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    atomic_write(args.out_dir / 'profile_cost_001.csv', csv_text(summary))
    atomic_write(args.out_dir / 'profile_cost_001.md', markdown(summary))
    print(json.dumps(dict(schema=OUT_SCHEMA, unique_measured_cells=summary['unique_measured_cell_count'],
                          requested_state_statuses=summary['requested_state_statuses'])))


if __name__ == '__main__':
    main()
