"""Sanitized evidence supplement; reads completed panels and never runs a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v31_value_longbench_report_no_exact import ARMS
import v31_value_longbench_report as report
from v31_value_longbench_report import cell_key, stage


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite diagnostic values')
    return dict(n=len(values), mean=float(values.mean()),
                median=float(np.median(values)), min=float(values.min()),
                max=float(values.max()))


def agreement(reference, other):
    reference = {cell_key(r): r for r in reference}
    other = {cell_key(r): r for r in other}
    if reference.keys() != other.keys():
        raise ValueError('unmatched trajectory comparison')
    return {field: sum(reference[k][field] != other[k][field] for k in reference)
            for field in ('output_hash', 'output_tokens', 'denoise_forwards',
                          'canvases', 'finish_reason')}


def main():
    p = argparse.ArgumentParser()
    for name in ('report', 'audit', 'clean', 'diagnostics', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    report.ARMS = ARMS
    root = Path(__file__).resolve().parents[1]
    study = root / 'results/v31_value_selectors_20261008'
    destination = Path(args.output)
    if destination.exists() or not destination.resolve().is_relative_to(study):
        raise ValueError('new canonical output directory required')
    marker = read(Path(args.report) / 'panel_complete.json')
    summary = read(Path(args.report) / 'summary.json')
    if (marker['status'] != 'complete' or marker['arms'] != len(ARMS)
            or marker['cells_per_arm'] != 503
            or digest(Path(args.report) / 'summary.json') != marker['summary_sha256']):
        raise ValueError('complete immutable panel required')
    passes = dict(audit=stage(study, args.audit, 'audit'),
                  clean=stage(study, args.clean, 'clean'))
    control = passes['clean']['current_v31_control']['config']
    identities = ('source_sha256', 'model_revision', 'manifests_sha256',
                  'model_source_inventory_sha256', 'host_fingerprint', 'gpu_identity')
    lineage, metrics, trajectories = [], {}, {}
    for purpose, arms in passes.items():
        dense = arms['dense_full_fix51994']['records']
        trajectories[purpose] = {
            arm: agreement(dense, item['records']) for arm, item in arms.items()}
        for arm, item in arms.items():
            for relative in item['source_attempts']:
                attempt = study / relative
                config, receipt = read(attempt / 'config.json'), read(attempt / 'receipt.json')
                if any(config[k] != control[k] for k in identities):
                    raise ValueError('shard-level source, host, GPU or input mismatch')
                if (receipt['status'] != 'complete' or receipt['returncode'] != 0
                        or receipt['validation_errors'] or receipt['timed_cuda_captures']):
                    raise ValueError('invalid completed shard receipt')
                lineage.append(dict(purpose=purpose, arm=arm, attempt=relative,
                    source_commit=config['source_commit'], cells=config['cell_count'],
                    config_sha256=digest(attempt / 'config.json'),
                    receipt_sha256=digest(attempt / 'receipt.json'),
                    gpu_seconds=receipt['gpu_seconds']))
    for arm, item in passes['audit'].items():
        records = item['records']
        adapters = [r['receipts']['adapter'] for r in records]
        total = lambda key: sum(a.get(key, 0) or 0 for a in adapters)
        n = sum(r['denoise_forwards'] for r in records)
        selection_calls = total('value_selection_calls')
        timings = summary['arms'][arm]['audit']
        metrics[arm] = dict(
            global_calls=total('global_calls'), local_calls=total('native_local_calls_audited'),
            global_eligible_tiles=total('global_eligible_tiles'),
            global_kept_tiles=total('global_kept_tiles'),
            local_eligible_rectangles=total('native_local_eligible_rectangles'),
            local_skipped_tiles=total('native_local_skipped_tiles'),
            overall_eligible_rectangles=total('overall_eligible_rectangles'),
            overall_skipped_rectangles=total('overall_skipped_rectangles'),
            selector_calls=selection_calls,
            initial_maps=total('mage_selections'), refresh_maps=total('mage_reselections'),
            held_global_calls=total('mage_reused_calls'),
            carried_global_calls=total('mage_carried_calls'),
            trigger_checks=total('trigger_checks'), triggers=total('triggers'),
            dense_fallback_calls=total('dense_fallback_calls'),
            pool_fallback_calls=total('mage_pool_fallbacks'),
            batch_support_fallbacks='not separately instrumented in frozen production source',
            invalid_rows='per-request last-selector observation, not a sum over all selections',
            invalid_rows_last_observation_sum=total('value_invalid_rows'),
            c_gate='disabled by the frozen control configuration',
            kernel_paths={key: sorted({a[key] for a in adapters if key in a})
                for key in ('value_statistics_kernel', 'value_selection_kernel',
                            'value_output_path', 'value_prefix_mass_source')},
            projection_manifests=[json.loads(x) for x in sorted({
                json.dumps(a['value_projection_manifest'], sort_keys=True)
                for a in adapters if 'value_projection_manifest' in a})],
            timing_scope='instrumented audit events; these are not clean request latencies',
            discovery_ms_per_denoise_call=(timings['discovery_ms'] / n
                if timings['discovery_ms'] is not None else None),
            selection_ms_per_denoise_call=(timings['selection_ms'] / n
                if timings['selection_ms'] is not None else None),
            selection_ms_per_selector_call=(timings['selection_ms'] / selection_calls
                if selection_calls and timings['selection_ms'] is not None else None))
    index_path = Path(args.diagnostics) / 'diagnostic_index.json'
    index = read(index_path)
    snapshots, diagnostic_sources = [], []
    for relative in index['completed']:
        complete = study / relative
        path = complete.with_name('summary.json')
        diagnostic = read(path)
        if diagnostic['status'] != 'complete' or diagnostic.get('excluded_fresh_selector') != 'value_v3b_greedy_exact':
            raise ValueError('diagnostic identity mismatch')
        snapshots.extend(diagnostic['diagnostics'])
        diagnostic_sources.append(dict(path=str(path.relative_to(study)),
            sha256=digest(path), snapshots=len(diagnostic['diagnostics'])))
    if not snapshots:
        raise ValueError('no completed sampled diagnostics')
    reused, fresh, quality = {}, {}, {}
    for row in snapshots:
        group = f"{row['generating_selector']}|age{row['distance_from_refresh']:02d}"
        reused.setdefault(group, []).append(row)
        for arm, values in row['fresh'].items():
            fresh.setdefault(f"{arm}|age{row['distance_from_refresh']:02d}", []).append(values)
            base = row['fresh']['current_v31_control']['full_dimensional_error']['max_row']
            quality.setdefault(f"{arm}|age{row['distance_from_refresh']:02d}", []).append(
                values['full_dimensional_error']['max_row'] - base)
    diagnostic_summary = dict(
        status=index['status'], failures=index['failures'],
        scope='sampled layer5, first canvas, first two long requests per audit shard, ages0/1/7 when available',
        operator='contiguous FA4 masked operator; excludes model alias2/paged merge rounding',
        aggregation='equal-weight snapshot summaries, not all-row or all-request population estimates',
        fresh_comparison='each fresh selector evaluated on the same frozen Q/K/V snapshot; no model forward',
        snapshot_count=len(snapshots), sources=diagnostic_sources,
        reused={g: dict(max_row_error=describe([r['reused_full_dimensional_error']['max_row'] for r in rows]),
                        mean_row_error=describe([r['reused_full_dimensional_error']['mean_row'] for r in rows]),
                        retained_mass_mean=describe([r['reused_retained_mass']['mean_row'] for r in rows]))
                for g, rows in reused.items()},
        fresh={g: dict(max_row_error=describe([r['full_dimensional_error']['max_row'] for r in rows]),
                       retained_mass_mean=describe([r['retained_mass']['mean_row'] for r in rows]),
                       mask_iou_with_reused=describe([r['mask_iou_with_reused'] for r in rows]),
                       mask_iou_with_mass_control=describe([r['mask_iou_with_mass_control'] for r in rows]),
                       max_error_delta_from_mass_control=describe(quality[g]))
               for g, rows in fresh.items()})
    result = dict(status='complete evidence supplement to eight-arm LongBench panel',
        report_summary_sha256=marker['summary_sha256'],
        generation_source_sha256=control['source_sha256'],
        source_commits=sorted({r['source_commit'] for r in lineage}),
        source_policy='Git metadata differs across validated predecessor shards; all17 generation-source hashes are identical',
        lineage=lineage, metrics=metrics, trajectory_mismatches_vs_native_full=trajectories,
        sampled_offline_diagnostics=diagnostic_summary,
        limitations=['RULER and HumanEval frozen inputs remain unavailable',
                     'exact greedy completed shard01 is retained separately as partial diagnostic',
                     'thresholds are coarse development choices, not matched-sparsity target fits',
                     'all-kept and native dense use distinct consumers; agreement is reported',
                     'no whole-study completion or noninferiority claim'])
    destination.mkdir(parents=True)
    output = destination / 'evidence.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    (destination / 'supplement_complete.json').write_text(json.dumps(dict(
        status='complete', evidence_sha256=digest(output),
        panel_summary_sha256=marker['summary_sha256']), indent=2) + '\n')


if __name__ == '__main__':
    main()
