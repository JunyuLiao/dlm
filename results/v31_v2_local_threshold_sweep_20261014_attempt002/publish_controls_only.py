"""Verify and summarize the completed, user-authorized controls-only scope."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXECUTION = ROOT / 'private/execution'
sys.path.insert(0, str(EXECUTION / 'scripts'))
spec = importlib.util.spec_from_file_location('frozen_campaign', EXECUTION / 'scripts/v31_v2_local_threshold_sweep.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
import v31_official as official


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


assert read(ROOT / 'PAUSED_USER_REQUEST.json')['status'] == 'paused_user_request'
assert read(ROOT / 'controls_complete.json')['status'] == 'complete'
assert not (ROOT / 'final_complete.json').exists(), 'Whole campaign must remain incomplete'
assert not list((ROOT / 'runs/longbench_v2').glob('v2_tau_*/shard*/attempt*/config.json'))
assert not (ROOT / 'selection/refinement_thresholds.json').exists()
protocol = read(ROOT / 'protocol.json')
campaign.verify(ROOT, protocol)
protocol_sha = hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest()
summaries = [read(p) for p in sorted((ROOT / 'reports').glob('*/*/summary.json'))]
expected = {
    (suite, arm)
    for suite in ['aime26', 'math500', 'longbench_v2']
    for arm in campaign.REFERENCES + ([] if suite == 'longbench_v2' else [campaign.label(t) for t in campaign.GRID])
}
assert {(s['suite'], s['arm']) for s in summaries} == expected
total = 0
for s in summaries:
    destination = ROOT / 'reports' / s['suite'] / s['arm']
    marker = read(destination / 'final_complete.json')
    receipt = read(destination / 'receipt.json')
    assert marker['records'] == marker['scored'] == s['items'] == campaign.SUITES[s['suite']]['count']
    assert not receipt['validation_errors'] and receipt['protocol_sha256'] == protocol_sha
    rows = []
    for relative in s['attempts']:
        attempt = ROOT / relative
        cfg = read(attempt / 'config.json')
        ar = read(attempt / 'attempt_complete.json')
        planned = read(ROOT / 'private/schedules' / s['suite'] / (attempt.parent.name + '.json'))
        shard_rows = campaign.jsonl(attempt / 'records.jsonl')
        assert cfg['protocol_sha256'] == ar['protocol_sha256'] == protocol_sha
        assert cfg['source_commit'] == ar['source_commit'] == protocol['source_commit']
        assert ar['status'] == 'complete' and not ar['validation_errors']
        assert not campaign.validate_records(shard_rows, s['suite'], s['arm'], s['threshold'], planned)
        rows.extend(shard_rows)
    assert campaign.aggregate(rows) == s['work'], 'Independent receipt aggregation differs'
    scores = read(ROOT / 'scoring' / s['suite'] / s['arm'] / 'official.official.json')[s['arm']]
    assert set(scores) == {campaign.key(r) for r in rows}
    assert sum(scores.values()) == s['correct']
    coverage = s['official_scorer']['coverage'][s['arm']]
    assert coverage['scored'] == s['items'] and not coverage['missing'] and not coverage['dropped']
    if s['arm'] != 'current_v31_control':
        cb = read(ROOT / 'scoring' / s['suite'] / 'current_v31_control/official.binding.json')
        vb = read(ROOT / 'scoring' / s['suite'] / s['arm'] / 'official.binding.json')
        official.check_comparable({**cb, **vb}, ['current_v31_control', s['arm']], set(scores))
    total += s['items']

# Reconstruct derived tables if the coordinator was stopped during publication.
campaign.publish(ROOT)
curve = read(ROOT / 'curve.json')
curve['status'] = 'paused_user_request'
curve['completed_scope'] = 'LongBench controls and initial AIME26/MATH500 grid'
write(ROOT / 'curve.json', curve)

with (ROOT / 'phase_accounting.csv').open('w', newline='') as f:
    fields = ['suite', 'arm', 'threshold', 'phase', 'calls', 'eligible_tiles', 'kept_tiles', 'skipped_tiles', 'kept_fraction', 'sparsity']
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for s in summaries:
        for phase, p in s['work']['phases'].items():
            w.writerow(dict(suite=s['suite'], arm=s['arm'], threshold=s['threshold'], phase=phase, **p))

with (ROOT / 'paired_comparisons.csv').open('w', newline='') as f:
    fields = ['suite', 'arm', 'delta_pp', 'ci95_low_pp', 'ci95_high_pp', 'problem_clusters', 'mcnemar_exact_p', 'control_only_correct', 'candidate_only_correct', 'output_hash_mismatches', 'output_token_count_mismatches', 'denoising_call_mismatches']
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for s in summaries:
        if 'paired_vs_v31' in s:
            p = s['paired_vs_v31']; low, high = p['problem_cluster_bootstrap_ci95_pp']
            row = dict(suite=s['suite'], arm=s['arm'], **p, ci95_low_pp=low, ci95_high_pp=high)
            w.writerow({k: row[k] for k in fields})

cost = []
for s in summaries:
    a = s['work']
    cost.append(dict(suite=s['suite'], arm=s['arm'],
        physical_fa4_consumer_global_tiles=a['global_kept_tiles'],
        value_selector_discovery_qk_rectangles=a['selector_discovery_qk_rectangles'],
        value_selector_candidate_evaluations=a['selector_candidate_evaluations'],
        value_selector_discovery_ms=a['selector_discovery_ms'],
        value_selector_selection_ms=a['selector_selection_ms'],
        native_denoising_forwards=a['N'], visible_canvases=a['C'], output_tokens=a['T'],
        audit_wall_s=a['audit_wall_s'], audit_decode_s=a['audit_decode_s'],
        worker_gpu_reservation_s=sum(read(ROOT / rel / 'receipt.json')['worker_reservation_seconds'] for rel in s['attempts']),
        timing_status='instrumented audit; clean timing paused'))
write(ROOT / 'cost_separation.json', cost)

lines = ['# Completed controls and short-workload threshold results', '',
    'The authorized LongBench controls are complete and verified. The full threshold campaign is paused by user request.',
    'No full LongBench V2 threshold run or finer threshold run was launched. No shared threshold can yet be selected.',
    'Accuracy tuning on these evaluated panels is exploratory. Clean timing remains paused.', '']

for suite, title in [('longbench_v2', 'LongBench-v2: 503 problems, seed 1'),
                     ('aime26', 'AIME26: 30 problems x seeds 42/43/44'),
                     ('math500', 'MATH500: fixed sampled 100 problems, seed 1')]:
    lines += ['', '## ' + title, '', '| Method | Correct | Accuracy | Calls/problem | Median calls | Canvases/problem | Calls/canvas | Tokens/problem | GLOBAL sparsity | Overall sparsity | Kept GLOBAL tiles | Work/V31 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    ordered = campaign.REFERENCES + ([] if suite == 'longbench_v2' else [campaign.label(t) for t in campaign.GRID])
    for arm in ordered:
        s = next(s for s in summaries if s['suite'] == suite and s['arm'] == arm)
        a = s['work']
        name = 'V2 tau=' + str(s['threshold']) if s['threshold'] is not None else s['arm']
        lines.append(f"| {name} | {s['correct']}/{s['items']} | {s['accuracy']:.2%} | {a['mean_N']:.2f} | {a['median_N']:.1f} | {a['mean_C']:.2f} | {a['N_per_C']:.2f} | {a['mean_T']:.1f} | {a['global_sparsity']:.2%} | {a['overall_sparsity']:.2%} | {a['global_kept_tiles']:,} | {s.get('global_kept_ratio_vs_v31', 1):.4f} |")

lines += ['', '## LongBench difficulty and length strata', '',
          '| Method | Easy | Hard | Short | Medium | Long |',
          '|---|---:|---:|---:|---:|---:|']
for arm in campaign.REFERENCES:
    s = next(s for s in summaries if s['suite'] == 'longbench_v2' and s['arm'] == arm)
    d = next(iter(s['official_scorer']['arms'][arm].values()))
    r = d['runs']['seed1_repeat0']
    lines.append('| ' + arm + ' | ' + ' | '.join(f"{r[k]:.1f}%" for k in ['Easy', 'Hard', 'Short', 'Medium', 'Long']) + ' |')

lines += ['', '## AIME26 per-seed accuracy', '', '| Method | Seed 42 | Seed 43 | Seed 44 | Total |', '|---|---:|---:|---:|---:|']
for s in summaries:
    if s['suite'] != 'aime26': continue
    arm_score = s['official_scorer']['arms'][s['arm']]
    seeds = arm_score['accuracy_by_panel_seed']
    lines.append(f"| {s['arm']} | {round(seeds['42']*.3)}/30 | {round(seeds['43']*.3)}/30 | {round(seeds['44']*.3)}/30 | {s['correct']}/90 |")

lines += ['', '## Parsing and coverage', '',
          '| Workload | Method | Scored | Capped | Unparsed | No final response | Input-truncated items |',
          '|---|---|---:|---:|---:|---:|---:|']
for s in summaries:
    d = s['official_scorer']['arms'][s['arm']]
    if s['suite'] == 'longbench_v2': d = next(iter(d.values()))
    lines.append(f"| {s['suite']} | {s['arm']} | {s['items']} | {d.get('capped', 0)} | {d.get('unparsed', 0)} | {d.get('no_final_response', 0)} | {d.get('truncated_items', 'n/a')} |")

lines += ['', '## Paired accuracy versus V31', '',
          '| Workload | Method | Delta (pp) | Problem-bootstrap 95% interval (pp) | V31-only / candidate-only correct | Exploratory McNemar p |',
          '|---|---|---:|---|---:|---:|']
for s in summaries:
    if 'paired_vs_v31' not in s: continue
    p = s['paired_vs_v31']; low, high = p['problem_cluster_bootstrap_ci95_pp']
    lines.append(f"| {s['suite']} | {s['arm']} | {p['delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {p['control_only_correct']} / {p['candidate_only_correct']} | {p['mcnemar_exact_p']:.4f} |")

lines += ['', 'Interpretation: AIME uses 30 problem clusters, keeping its three seeds together; the cell-level McNemar value is exploratory.',
          'The V2 0.021 AIME accuracy difference is +7.78 pp, with a paired problem-bootstrap interval spanning zero.',
          'V2 0.021 retains more physical GLOBAL work than V31 on the short workloads; fewer denoising calls alone do not establish speedup.',
          'GLOBAL work counts audited Q128 x KV64 rectangles over native denoising calls. LOCAL is dense; prefill and commit work are separate.',
          'Canvas C is the reported visible output canvas count; receipt canvas IDs are ranked separately for progress-axis work.',
          'LongBench truncated counts describe frozen input truncation, while capped counts describe the generation output limit.',
          'MATH500 uses bound math-verify symbolic equivalence with the official final-response parser; the fixed subset effective sampling seed is 20261111.',
          '', 'Pending, resumable: the five initial LongBench V2 thresholds and two finer thresholds on all three workloads.',
          'Existing frozen protocols, completed attempts, and private generations are retained. Resume only after user instruction.', '']
(ROOT / 'summary.md').write_text('\n'.join(lines))
verification = dict(status='complete', scope='completed controls and initial short-workload grid only',
    reports=len(summaries), scored_cells=total, source_files_verified=len(protocol['execution_fingerprints']),
    input_files_verified=len(protocol['input_fingerprints']), source_commit=protocol['source_commit'],
    protocol_sha256=protocol_sha, invalid_rows=0, missing_scorer_cells=0,
    unexpected_request_cuda_captures=0, local_skipped_tiles=0,
    native_attention_and_sampler_unchanged=True, longbench_v2_threshold_attempts=0,
    finer_threshold_attempts=0, whole_campaign_complete=False, clean_timing='paused')
write(ROOT / 'receipts/controls_only_verification.json', verification)
print(json.dumps(verification))
