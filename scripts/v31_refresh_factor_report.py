"""Sanitized report for completed factor isolation, with no new target scoring."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.v31_refresh_factor_analysis import factorial, arm_name, sha


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    study = args.study.resolve()
    if (study / 'final_complete.json').exists():
        raise RuntimeError('completed factor study is immutable')
    op_path = study / 'operator/attempt001/summary.json'
    op = json.loads(op_path.read_text())
    marker = json.loads((op_path.parent / 'final_complete.json').read_text())
    if sha(op_path) != marker['summary_sha256'] or sha(op_path.parent / 'records.jsonl') != marker['records_sha256']:
        raise RuntimeError('operator completion identity mismatch')
    identity_path = study / 'specs/replay_identity.json'
    identity = json.loads(identity_path.read_text())
    if sha(identity_path) != op['identity_sha256'] or sha(study / 'specs/protocol.json') != identity['protocol_sha256']:
        raise RuntimeError('frozen protocol/replay identity mismatch')
    root = Path(__file__).resolve().parents[1]
    for rel, digest in identity['source_sha256'].items():
        if sha(root / rel) != digest:
            raise RuntimeError('frozen selector/replay source changed')
    if any(op[k] for k in ('cpu_map_mismatches', 'budget_errors', 'swap_cap_errors', 'cuda_graph_captures')):
        raise RuntimeError('operator gate failed')
    native = json.loads((study / 'reports/native_factor_contrasts.json').read_text())
    data = op['configurations']
    records = [json.loads(line) for line in (op_path.parent / 'records.jsonl').read_text().splitlines()]
    ds_metrics, epoch_histograms = {}, {}
    for arm, config in data.items():
        rr = [r for r in records if r['arm'] == arm]
        if len(rr) != op['frames_per_arm']:
            raise RuntimeError('unmatched operator frames')
        ds_metrics[arm] = {}
        for ds in ('longbench_v2_0shot_think', 'aime26'):
            subset = [r for r in rr if r['dataset'] == ds and r['common_reuse']]
            ds_metrics[arm][ds] = dict(frames=len(subset),
                row_error_frame_mean=sum(r['row_error_mean'] for r in subset) / len(subset),
                row_error_max=max(r['row_error_max'] for r in subset),
                retained_mass_frame_mean=sum(r['true_retained_mass_mean'] for r in subset) / len(subset))
        grouped = defaultdict(list)
        for r in rr:
            grouped[r['trajectory'], r['canvas'], r['layer']].append(r)
        epochs = []
        for group in grouped.values():
            count = 0
            for r in group:
                if r['phase'] in ('initial', 'progress'):
                    if count:
                        epochs.append(count)
                    count = 0
                count += int(r['observation'])
            if count:
                epochs.append(count)
        epoch_histograms[arm] = dict(Counter(epochs))
    contrasts = []
    def add(factor, candidate, reference, held):
        a, b = data[candidate], data[reference]
        contrasts.append(dict(factor=factor, candidate=candidate, reference=reference, held_fixed=held,
            common_reuse_error_ratio=a['common_reuse_row_error']['mean'] / b['common_reuse_row_error']['mean'],
            common_reuse_mass_delta=a['common_reuse_retained_mass_mean'] - b['common_reuse_retained_mass_mean'],
            map_change_p95_delta=a['next_call_map_change']['p95'] - b['next_call_map_change']['p95'],
            swaps_delta=a['swaps'] - b['swaps'], swaps_ratio=a['swaps'] / b['swaps'] if b['swaps'] else None))
    for progress in (True, False):
        for interval in (1, 2, 4, 8):
            add('stability', arm_name('stable', progress, interval), arm_name('raw', progress, interval),
                dict(progress_dense=progress, interval=interval))
    for mode in ('raw', 'stable'):
        for interval in (1, 2, 4, 8):
            add('progress', arm_name(mode, True, interval), arm_name(mode, False, interval),
                dict(mode=mode, interval=interval))
        for progress in (True, False):
            for i, reference_interval in enumerate((1, 2, 4, 8)):
                for interval in (1, 2, 4, 8)[i + 1:]:
                    add('interval', arm_name(mode, progress, interval), arm_name(mode, progress, reference_interval),
                        dict(mode=mode, progress_dense=progress))
    add('progress_without_cheap_refresh', 'v31_progress_dense_control', 'v31_no_progress_dense_hold', dict(interval=0))
    write(study / 'reports/operator_factor_contrasts.json', dict(status='complete', contrasts=contrasts,
        workload_metrics=ds_metrics, nonempty_anchor_epoch_observation_histograms=epoch_histograms,
        scope=op['progress_scope'], metric_scope=op['error_scopes']))
    lines = ['# V31 refresh factor isolation', '',
        '**Completed: all 16 operator settings, two held references, and paired analysis of existing native results. No new 503-problem generation has started.**', '',
        'The 64-problem LongBench panel is exploratory evidence for selecting further tests. It has only 8 of the 16 factorial cells, so progress with cheap refresh has no matched native comparison. Operator replay fills the factorial on identical frozen Q/K/V; it does not supply missing native accuracy or denoising results.', '',
        '## Operator diagnostics', '',
        f"620 real snapshots / 670 validated files, all five GLOBAL layers in both workloads, {op['cpu_reference_events']:,} real cheap events checked against an independent NumPy oracle. Fresh QK error <= {op['max_fresh_statistic_error']:.3g}; hybrid score error <= {op['max_cpu_score_error']:.3g}. Map mismatches, fixed-budget errors, stable cap errors and actual CUDA graph captures are zero. The 26 new tests and 38 CUDA/FA4 tests passed.", '',
        'Each arm has the same 570 post-anchor frames; 50 pre-anchor bootstrap/carry frames are excluded equally. The table compares the same 480 non-selection frames for every arm. Errors are full-dimensional row-output L2 versus dense attention. The output-change p95 includes zero-change and scheduled dense calls over the common 570-frame denominator.', '',
        '| Progress | Interval | Raw / stable row error | Raw / stable retained mass | Raw / stable swaps | Raw / stable change p95 |',
        '|---|---:|---:|---:|---:|---:|']
    for progress in (True, False):
        for interval in (1, 2, 4, 8):
            a, b = data[arm_name('raw', progress, interval)], data[arm_name('stable', progress, interval)]
            lines.append(f"| {'On' if progress else 'Off'} | {interval} | {a['common_reuse_row_error']['mean']:.4f} / {b['common_reuse_row_error']['mean']:.4f} | {a['common_reuse_retained_mass_mean']:.6f} / {b['common_reuse_retained_mass_mean']:.6f} | {a['swaps']:,} / {b['swaps']:,} | {a['next_call_map_change']['p95']:.5f} / {b['next_call_map_change']['p95']:.5f} |")
    progress_ratios = [x['common_reuse_error_ratio'] for x in contrasts if x['factor'] == 'progress']
    lines += ['',
        f"Progress dense lowers mean error on common sparse calls by {100 * (1 - max(progress_ratios)):.2f}-{100 * (1 - min(progress_ratios)):.2f}% at matched mode and interval. This is conditional operator evidence; its native effect with cheap refresh remains unmeasured.", '',
        'More frequent raw refresh improves the operator mass/error monotonically in this replay. Stable also improves with frequency, but keeps less mass and has slightly higher error than raw at every matched setting. Stable substantially suppresses churn and next-call perturbation; that property does not establish better native accuracy.', '',
        'Stable + progress + interval 8 makes **zero swaps** and exactly matches the progress hold map/error. Each of its 15 nonempty anchor periods has only one cheap event, so the two-event persistence condition cannot be reached. Its zero p95 is an inactive update mechanism on these snapshots. At other settings, a zero p95 can coexist with large rare changes; see the maxima in the operator receipt.', '',
        '## Native accuracy and trajectory: stability alone', '',
        '| Progress | Interval | Raw / stable correct out of 64 | Stable minus raw | Stable N change | Stable N/C change | Stable output length change |',
        '|---|---:|---:|---:|---:|---:|---:|']
    grid = {r['arm']: r for r in native['factorial']}
    for c in native['contrasts']:
        if c['factor'] != 'stability' or c['workload'] != 'longbench_v2_0shot_think':
            continue
        a = next(r for r in grid[c['reference']]['native_results'] if r['workload'] == c['workload'])
        b = next(r for r in grid[c['candidate']]['native_results'] if r['workload'] == c['workload'])
        lines.append(f"| {'On' if c['held_fixed']['progress_dense'] else 'Off'} | {c['held_fixed']['interval']} | {a['correct']} / {b['correct']} | {c['correct_delta']:+d} | {100*(c['N_ratio']-1):+.2f}% | {100*(c['NC_ratio']-1):+.2f}% | {100*(c['T_ratio']-1):+.2f}% |")
    lines += ['',
        'Stability is an interval-dependent bundle effect, not a uniform win. At interval 8 without progress it rescues eight cells relative to raw, but only gains one cell over progress V31 and two over matching hold, while increasing N by 22.29% and 11.85% respectively. At interval 1, lower N accompanies shorter outputs. The paired intervals and all existing one-factor contrasts are in `native_factor_contrasts.json`; they are exploratory and unadjusted for multiple comparisons.', '',
        '**Interval alone, progress off:** raw has counts 28/32/27 and mean N 304.38/249.17/324.14 at 1/4/8. Stable has counts 32/32/35 and N 218.64/273.38/293.83. Interval 2 is missing. The operator trend toward more frequent updates does not predict this native accuracy ordering.', '',
        '**Progress alone, hold controls:** progress-on versus progress-off gives 34 versus 33 correct, N -8.54%, N/C -10.34%, and output length +6.69%. Accuracy has a broad paired 95% interval of [-10.94, +14.06] pp. This isolates progress only at zero cheap updates. All no-progress cheap arms share the AIME hold correctness vector; AIME is mostly all-kept at K=128.', '',
        '## Candidate selection for 503 problems', '',
        'A compact factor-preserving design starts with the four raw/stable × progress-on/off arms at interval 2. Add stable no-progress intervals 1/4/8 to test the full frequency sweep. This gives **seven method arms**, two matched held references and three native-consumer controls; a full execution would be 12 × 503 = 6,036 cells. It identifies stability and progress at interval 2 and frequency within stable no-progress. It does not estimate every interaction across all 16 settings.', '',
        'Stable no-progress s1 is the lower-N candidate; s8 is the higher observed accuracy candidate with trajectory inflation. Raw progress s2 is the near-control-N comparator. Include the missing no-progress s2 pair to isolate progress. Raw no-progress s4 is an optional frequency challenger if a raw interval sweep is selected. Do not drop a counterpart solely because it scored poorly on the noisy subset; keep the comparison needed to answer each factor question.', '',
        'Use fresh source/runtime-matched 503 controls. The old 268/503 V31 result is historical, not the matched reference for this follow-up. The user will choose arms after this report; `503_arm_proposal.json` is a proposal and no native job has been launched.', '',
        '## Selector overhead and clean timing', '',
        f"The replay used one H100, with {op['replay_wall_seconds']:.1f} seconds of replay wall time including CPU reference checks. Peak replay allocation was {op['peak_cuda_allocated_bytes']/2**30:.3f} GiB for the multi-arm diagnostic. Per-arm QK/ranking CUDA event totals and selector-state bytes are in the operator summary. Replay wall time and simultaneous-arm peak allocation are not native selector latency or memory. Production uses duplicate retained/tail QK, unchanged FA4 output, no selector PV/probes, and GPU ranking with existing host decisions.", '',
        '**Clean timing remains paused.** Native N and output-length effects are trajectory metrics, not speedup. Existing native receipts establish dense LOCAL routing and zero graph captures; this offline replay changes no native output code. Stable uses zero hysteresis, persistence 2 cheap events, cap 2/event and 8/anchor epoch, and removal guard 0.025. Persistence therefore requires at least 2/4/8/16 reuse steps at intervals 1/2/4/8. Internal stabilizer components are not separately ablated.', '',
        'Sources: native publication `a136b40d`; factor replay freeze `be9e23dc`; operator receipt and source identity stored with this study. Source study artifacts remain immutable.', '']
    report = study / 'reports/final_report.md'
    report.write_text('\n'.join(lines))
    files = [study / 'reports' / name for name in ('native_factor_contrasts.json', 'native_analysis.md', 'operator_factor_contrasts.json', 'aime_specificity_check.json', '503_arm_proposal.json', 'final_report.md')]
    files += [study / 'specs' / name for name in ('protocol.json', 'unit_test_receipt.json', 'gpu_test_receipt.json', 'native_source_reuse_receipt.json', 'runtime_identity.json', 'replay_identity.json')]
    files += [op_path, op_path.parent / 'final_complete.json']
    write(study / 'final_complete.json', dict(status='complete', scope='Existing native analysis and complete operator factorial only',
        operator_factorial_arms=16, operator_held_references=2, native_factorial_settings_available=8,
        operator_input_files=670, operator_snapshots=620, cpu_reference_events=op['cpu_reference_events'],
        native_generation_launched=False, clean_timing='paused', report_source_sha256=sha(__file__),
        artifact_sha256={str(path.relative_to(study)): sha(path) for path in files}))
    print(json.dumps(dict(status='complete', operator_contrasts=len(contrasts), operator_arms=18, native_generation_launched=False)))


if __name__ == '__main__':
    main()
