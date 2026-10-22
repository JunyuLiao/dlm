"""Qualify, calibrate and run the frozen optimized V2/V31 native panel."""
from pathlib import Path
import argparse
import json
import math
import time

from scripts.optimized_refresh_campaign import (
    ARMS, ROOT, SUITES, read, read_lines, run_job, score_suite,
    sha, utc, verify, write,
)
from scripts.optimized_refresh_report import campaign_report


def held_work(leaf):
    totals = dict(calls=0, eligible_tiles=0, kept_tiles=0)
    for row in read_lines(leaf / 'records.jsonl'):
        adapter = row.get('receipts', {}).get('adapter', {})
        phase = adapter.get('value_phase_tiles', {}).get('held')
        if phase is None:
            continue
        for key in totals:
            totals[key] += int(phase.get(key, 0))
    if totals['calls'] == 0 or totals['eligible_tiles'] == 0:
        raise RuntimeError(f'no audited held GLOBAL tiles in calibration: {leaf}')
    totals['sparsity'] = 1.0 - totals['kept_tiles'] / totals['eligible_tiles']
    return totals


def qualify(study):
    if (study / 'qualification/gates.json').exists():
        gate = read(study / 'qualification/gates.json')
        if gate.get('status') == 'pass' and gate.get('protocol_sha256') == sha(study / 'protocol.json'):
            return gate
        raise RuntimeError('qualification gate exists but does not match this protocol')
    verify(study)
    rows = []
    for suite in SUITES:
        cells = study / f'private/schedules/{suite}/smoke.json'
        if not cells.exists():
            cells = study / f'private/schedules/{suite}/calibration.json'
        for arm in ARMS:
            tau = 0.05 if arm.startswith('v2') else None
            audit = run_job(study, suite, arm, 'audit', cells, 'qualification', tau)
            clean = run_job(study, suite, arm, 'clean', cells, 'qualification', tau)
            audit_rows = read_lines(audit / 'records.jsonl')
            clean_rows = read_lines(clean / 'records.jsonl')
            if [(r['output_hash'], r['denoise_forwards']) for r in audit_rows] != [(r['output_hash'], r['denoise_forwards']) for r in clean_rows]:
                raise RuntimeError(f'audit/clean trajectory mismatch: {suite}/{arm}')
            for row in audit_rows + clean_rows:
                adapter = ((row.get('receipts') or {}).get('adapter') or {})
                if row['denoise_forwards'] <= 0 or (arm.startswith(('v2', 'v31')) and adapter.get('optimized_refresh_extra_model_forwards') != 0):
                    raise RuntimeError(f'extra/missing model forwards: {suite}/{arm}')
                if row['value_audit'] and adapter.get('native_local_skipped_tiles') != 0:
                    raise RuntimeError(f'LOCAL skip in qualification: {suite}/{arm}')
                if row['cuda_graph_captures'] != 0:
                    raise RuntimeError(f'capture in qualification timing: {suite}/{arm}')
                if arm.startswith(('v2', 'v31')):
                    if not adapter.get('resident_sparse_execution'):
                        raise RuntimeError(f'resident held consumer not installed: {suite}/{arm}')
                    if adapter.get('resident_sparse_held_calls', 0) <= 0:
                        raise RuntimeError(f'no held calls observed: {suite}/{arm}')
                    if adapter.get('resident_sparse_held_kv_copy_calls') != 0 or adapter.get('resident_sparse_held_selector_calls') != 0 or adapter.get('resident_sparse_held_list_builds') != 0:
                        raise RuntimeError(f'held path performed avoidable setup: {suite}/{arm}')
                    if row['value_clean_timing'] and (adapter.get('resident_sparse_request_triton_compiles') != 0 or bool(adapter.get('resident_sparse_request_jit_events'))):
                        raise RuntimeError(f'JIT event during clean timing: {suite}/{arm}')
                    if arm.startswith('v2'):
                        expected = read(study / 'protocol.json')['kernel_sha256']
                        options = adapter.get('optimized_refresh_build_options') or {}
                        if adapter.get('optimized_refresh_library_sha256') != expected or not options.get('interior_mask') or not options.get('bulk3d'):
                            raise RuntimeError(f'wrong V2 refresh library/build: {suite}/{arm}')
                        if adapter.get('optimized_refresh_projection_pack') != 'fused_value_pack':
                            raise RuntimeError(f'wrong V2 projection/packing wrapper: {suite}/{arm}')
            rows.append(dict(suite=suite, arm=arm, threshold=tau,
                             audit= str(audit.relative_to(study)), clean=str(clean.relative_to(study)),
                             cells=len(audit_rows), trajectory_match=True))
    gate = dict(status='pass', utc=utc(), protocol_sha256=sha(study / 'protocol.json'),
                checks=['synthetic kernel parity and changing-input graph qualification',
                        'native audit/clean trajectory agreement', 'zero extra forwards',
                        'zero LOCAL skips', 'resident held path has no selector/list/KV-copy work',
                        'zero clean captures and request JIT events', 'exact optimized V2 library/build receipt'],
                rows=rows)
    write(study / 'qualification/gates.json', gate)
    return gate


def calibrate_held(study, suite):
    dest = study / f'calibration_held/{suite}/selected_threshold.json'
    if dest.exists():
        result = read(dest)
        if not result.get('in_target_range'):
            raise RuntimeError('existing held calibration is outside target')
        return result
    cells = study / f'private/schedules/{suite}/calibration.json'
    candidates = read(study / 'protocol.json')['calibration']['candidate_thresholds']
    measured = {}
    def measure(tau):
        if str(tau) in measured:
            return
        leaf = run_job(study, suite, 'v2_online_optimized', 'audit', cells,
                       'held_calibration_full_request', float(tau))
        measured[str(tau)] = held_work(leaf)
    def in_range():
        return any(.40 <= row['sparsity'] <= .50 for row in measured.values())
    for tau in candidates:
        measure(tau)
        if in_range():
            break
    if not in_range():
        for _ in range(read(study / 'protocol.json')['calibration']['refinements']):
            below = [float(k) for k, v in measured.items() if v['sparsity'] < .45]
            above = [float(k) for k, v in measured.items() if v['sparsity'] >= .45]
            brackets = [(a, b) for a in below for b in above if a < b]
            if not brackets:
                break
            lo, hi = min(brackets, key=lambda pair: pair[1] - pair[0])
            measure(math.sqrt(lo * hi) if lo > 0 else hi / 2)
            if in_range():
                break
    winner = min(measured, key=lambda key: (0 if 0.40 <= measured[key]['sparsity'] <= 0.50 else 1,
                                           abs(measured[key]['sparsity'] - 0.45), float(key)))
    selected = measured[winner]
    result = dict(status='held_physical_tile_calibration', suite=suite,
                  threshold=float(winner), held_sparsity=selected['sparsity'],
                  held_eligible_tiles=selected['eligible_tiles'],
                  held_kept_tiles=selected['kept_tiles'], held_calls=selected['calls'],
                  calibration_cells=len(read(cells)), in_target_range=0.40 <= selected['sparsity'] <= 0.50,
                  target_sparsity=[0.40, 0.50], candidates=measured,
                  protocol_sha256=sha(study / 'protocol.json'), accuracy_used=False)
    write(dest, result)
    if not result['in_target_range']:
        raise RuntimeError(f'held sparsity calibration outside 40-50% for {suite}; preserve candidates and freeze a refined calibration protocol')
    return result


def run(study):
    if (study / 'PAUSED_USER_REQUEST.json').exists():
        raise RuntimeError('paused attempts are immutable; create a new attempt')
    verify(study)
    gate = qualify(study)
    if gate['status'] != 'pass':
        raise RuntimeError('native qualification gate failed')
    thresholds = {suite: calibrate_held(study, suite)['threshold'] for suite in SUITES}
    write(study / 'calibration_held/complete.json', dict(status='complete', thresholds=thresholds,
          protocol_sha256=sha(study / 'protocol.json'), utc=utc()))
    for suite in SUITES:
        for cells in sorted((study / f'private/schedules/{suite}').glob('shard???.json')):
            for arm in ARMS:
                tau = thresholds[suite] if arm.startswith('v2') else None
                for pass_name in ['audit', 'clean']:
                    run_job(study, suite, arm, pass_name, cells, 'runs', tau)
        score_suite(study, suite)
        write(study / 'state.json', dict(status='suite_complete', suite=suite,
              updated_utc=utc(), protocol_sha256=sha(study / 'protocol.json')))
    write(study / 'generation_complete.json', dict(status='complete', utc=utc(),
          protocol_sha256=sha(study / 'protocol.json')))
    campaign_report(study)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['qualify', 'calibrate', 'run', 'verify'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--suite', choices=list(SUITES))
    args = parser.parse_args()
    study=args.study.resolve()
    try:
        if args.action == 'verify':
            print(verify(study)['source_commit'])
        elif args.action == 'qualify':
            print(json.dumps(qualify(study), indent=2))
        elif args.action == 'calibrate':
            if args.suite is None:
                parser.error('--suite is required for calibrate')
            print(json.dumps(calibrate_held(study, args.suite), indent=2))
        else:
            run(study)
    except Exception as error:
        previous=read(study/'state.json') if (study/'state.json').exists() else {}
        write(study/'state.json',dict(previous,status='failed',error=str(error),updated_utc=utc()))
        raise


if __name__ == '__main__':
    main()
