"""Export both completed, independently verified HumanEval budget panels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from scripts.v31_humaneval_no_overlap_queue import read, sha, write, require_complete

LABELS = {
    'dense_full_fix51994': 'Dense FULL', 'dense_piecewise': 'Dense PIECEWISE',
    'allkept_fa4': 'All-kept FA4', 'control_before': 'V31',
    'value_v2_online_preserve_mass': 'V2', 'value_v3a_singleton_delete': 'V3a',
    'full_swap8': 'Full-value swaps', 'residual_direction_guard': 'Direction-guarded residual',
}
MODES = ('thinking_on', 'thinking_off')
COPY = ('summary.md', 'summary.json', 'per_item_metrics.jsonl', 'seed43_descriptive.json',
        'independent_verification.json', 'independent_rescoring.json', 'paper_table.tex', 'paper_methods.txt')


def load(root, budget):
    study = root / f'results/humaneval_value_observe_no_overlap_k{budget}_20261019/attempt001'
    digest = sha(study / 'protocol.json')
    require_complete(study, digest)
    checked, rescored = [read(study / name) for name in ('independent_verification.json', 'independent_rescoring.json')]
    assert checked['status'] == 'passed' and checked['protocol_sha256'] == digest
    assert rescored['mismatches'] == rescored['qualification_mismatches'] == 0
    assert checked['validated_records'] == rescored['records'] == 5904
    proof = read(study / 'receipts/full164_audit_clean_equivalence.json')['comparisons']
    assert len(proof) == 22 and all(x['passed'] for x in proof)
    return study, read(study / 'protocol.json'), read(study / 'summary.json')


def publish(root, output):
    panels = {k: load(root, k) for k in (256, 128)}
    queue = root / 'results/humaneval_no_overlap_queue_20261020/attempt001'
    assert read(queue / 'final_complete.json')['status'] == 'complete'
    config = read(queue / 'protocol.json')
    assert read(queue / 'final_complete.json')['queue_protocol_sha256'] == sha(queue / 'protocol.json')
    assert config['owns_coordinators'] and config['monitor_interval_s'] == 900
    assert panels[256][1]['source_commit'] == panels[128][1]['source_commit']
    assert read(panels[256][0] / 'receipts/source_fingerprints.json') == read(panels[128][0] / 'receipts/source_fingerprints.json')
    assert panels[128][1]['shared_references']['protocol_sha256'] == sha(panels[256][0] / 'protocol.json')
    if output.exists():
        raise RuntimeError('new immutable publication attempt required')
    output.mkdir(parents=True)
    rows, contrasts, bindings = [], {}, {}
    compact = ['All164 HumanEval problems, seed42. Accuracy is correct/164; N is denoising forwards excluding commit calls.',
               'W is clean mean request time including prefill and all selector/correction work.',
               'Sparsity is measured in the trajectory-matched audit pass; LOCAL is dense.', '']
    detailed = list(compact)
    paper = ['\\begin{tabular}{rllrrrrrr}', '\\toprule',
             'K & Thinking & Method & Correct & $N$ & $N/C$ & $W$ (ms) & GLOBAL (\\%) & Overall (\\%) \\\\', '\\midrule']
    for budget, (study, protocol, summary) in panels.items():
        folder = output / f'k{budget}'
        folder.mkdir()
        for name in COPY:
            shutil.copyfile(study / 'analysis/attempt001' / name, folder / name)
        for name in ('full164_audit_clean_equivalence.json', 'independent_prefix_replication.json'):
            shutil.copyfile(study / 'receipts' / name, folder / name)
        bindings[str(budget)] = dict(study=str(study.relative_to(root)), protocol_sha256=sha(study / 'protocol.json'),
            final_complete_sha256=sha(study / 'final_complete.json'),
            analysis_complete_sha256=sha(study / 'analysis/attempt001/analysis_complete.json'),
            source_fingerprints_sha256=sha(study / 'receipts/source_fingerprints.json'),
            independent_rescoring_sha256=sha(study / 'independent_rescoring.json'),
            shared_references=protocol['shared_references'], native=protocol['native'],
            runtime_packages=protocol['runtime']['packages'], model_inventory_sha256=protocol['model_inventory_sha256'],
            humaneval_source_sha256=protocol['humaneval_source_sha256'],
            observing_fa4_policy_sha256=protocol['observing_policy_source_sha256'], compiler_policy=protocol['compiler_policy'])
        contrasts[str(budget)] = {}
        for mode in MODES:
            group = summary['modes'][mode]
            reference = group['methods']['control_before']
            dense = group['methods']['dense_full_fix51994']
            contrasts[str(budget)][mode] = group['paired_vs_v31']
            compact += [f'**K{budget}, {mode.replace("_", " ")}**', '',
                '| Method | Accuracy | N/problem | N/C | W ms | GLOBAL / overall sparsity |',
                '|---|---:|---:|---:|---:|---:|']
            detailed += [f'**K{budget}, {mode.replace("_", " ")}**', '',
                '| Method | Correct/164 | N | C | N/C | Output tokens | S ms | W ms | S/N ms | GLOBAL % | Overall % |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for label, item in group['methods'].items():
                assert item['questions'] == 164 and item['local_sparsity'] == 0
                assert item['clean_audit_hash_differences'] == item['clean_audit_N_differences'] == item['clean_audit_interval_differences'] == item['clean_audit_score_differences'] == 0
                record = dict(budget=budget, mode=mode, method=label, technical_bookend=label == 'control_after',
                    correct=item['correct'], accuracy=item['accuracy'], N=item['N_mean'], C=item['C_mean'], N_per_C=item['N_per_C'],
                    output_tokens=item['T_mean'], S_s=item['S_mean_s'], W_s=item['W_mean_s'], S_per_N_ms=item['amortized_S_per_N_ms'],
                    global_sparsity=item['global_sparsity'], overall_sparsity=item['overall_sparsity'],
                    speedup_vs_v31=reference['W_mean_s']/item['W_mean_s'], speedup_vs_dense_full=dense['W_mean_s']/item['W_mean_s'],
                    N_p95=item['N_p95'], N_max=item['N_max'], generation_caps=item['generation_caps'])
                rows.append(record)
            for label, name in LABELS.items():
                m = group['methods'][label]
                compact.append(f"| {name} | {100*m['accuracy']:.2f}% ({m['correct']}/164) | {m['N_mean']:.2f} | {m['N_per_C']:.2f} | {1000*m['W_mean_s']:.1f} | {100*m['global_sparsity']:.2f}% / {100*m['overall_sparsity']:.2f}% |")
                detailed.append(f"| {name} | {m['correct']} | {m['N_mean']:.3f} | {m['C_mean']:.3f} | {m['N_per_C']:.3f} | {m['T_mean']:.2f} | {1000*m['S_mean_s']:.3f} | {1000*m['W_mean_s']:.3f} | {m['amortized_S_per_N_ms']:.3f} | {100*m['global_sparsity']:.3f} | {100*m['overall_sparsity']:.3f} |")
                paper.append(f"{budget} & {mode.removeprefix('thinking_')} & {name} & {m['correct']}/164 & {m['N_mean']:.2f} & {m['N_per_C']:.2f} & {1000*m['W_mean_s']:.1f} & {100*m['global_sparsity']:.2f} & {100*m['overall_sparsity']:.2f} \\\\")
            threshold = group['threshold']
            note = f"V2 tau={threshold['threshold']:.12g}; first-canvas matching: {threshold['matching']}."
            compact += ['', note, '']
            detailed += ['', note, '']
    assert len(rows) == 36
    with (output / 'methods.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output / 'paired_vs_v31.json', contrasts, True)
    ledger = read(root / 'results/humaneval_reproducibility_20261019/attempt001/gpu_budget.json')
    reservations = [r for r in ledger['reservations'] if r['phase'] == 'evaluation']
    assert all('elapsed_s' in r for r in reservations)
    snapshots = [json.loads(line) for line in (queue / 'progress.jsonl').read_text().splitlines()]
    assert all(s['gpu_status']['returncode'] == s['gpu_processes']['returncode'] == 0 for s in snapshots)
    resources = dict(evaluation_charge_s=sum(r['elapsed_s'] for r in reservations),
        conservative_interruption_charge_s=sum(r['elapsed_s'] for r in reservations if r.get('elapsed_s_is_conservative_upper_bound')),
        shared_evaluation_cap_s=ledger['caps_s']['evaluation'], monitoring_snapshots=len(snapshots),
        monitor_interval_s=900, queue_protocol_sha256=sha(queue / 'protocol.json'),
        reservation_ledger_sha256=sha(root / 'results/humaneval_reproducibility_20261019/attempt001/gpu_budget.json'))
    write(output / 'resources.json', resources, True)
    write(output / 'config.json', dict(source_commit=panels[256][1]['source_commit'], panels=bindings,
        analysis_source_sha256=sha(__file__), seed=42, questions=164, shared_dense_controls=True,
        method_provenance='V31 inherits Yuhan cross-step reuse; value-aware selectors/refinements are separately named study variants'), True)
    limits = ['**Reproducibility:** all22 full audit/clean comparisons per budget pass; independent engine prefixes and official rescores pass.',
        '**Work limitation:** K128 thinking-disabled V2 misses the first-canvas0.25pp target: +0.53116pp sparsity and0.93680 absolute kept-tile ratio. The predeclared nearest-work rule retained its threshold. Whole-trajectory work is not matched in any panel by first-canvas calibration alone.',
        'Dense FULL/PIECEWISE/all-kept references are shared between budgets and are not additional independent trials.',
        'V2/V3a use Gaussian32 on all five GLOBAL layers; full-value swaps/residual affect only layer29. All LOCAL attention stays dense.',
        'Overall sparsity counts denoising GLOBAL plus native-window LOCAL Q128/KV64 rectangles, excluding prefill/commits and additional selector/PV work. W includes the complete method cost; S/N is amortized.',
        'Main accuracy uses seed42. Seed43 covers six examined questions and remains descriptive. Question-bootstrap intervals condition on the observed engines and do not establish seed robustness.']
    (output / 'summary.md').write_text('\n'.join(compact + limits) + '\n')
    (output / 'detailed_results.md').write_text('\n'.join(detailed + limits) + '\n')
    (output / 'paper_table.tex').write_text('\n'.join(paper + ['\\bottomrule', '\\end{tabular}']) + '\n')
    (output / 'paper_methods.tex').write_text('''We evaluate all 164 HumanEval tasks with seed 42 and thinking enabled and disabled. Selectable prefix budgets are four or two KV64 blocks per query-head/Q128 tile; mandatory canvas/boundary tiles and LOCAL attention remain dense. V2 uses separately work-calibrated online thresholds. Native sampler formulas, stopping and output limits remain fixed. We use explicit deterministic Inductor compilation for the model and unchanged native helpers, retain native FULL/PIECEWISE graphs, and disable intra-workgroup overlap only for mass-observing FA4 calls. Correction and selector costs remain included in clean request latency. Audit/clean outputs and native denoising intervals agree for every task/method, and independent engine prefixes and official rescoring pass. Dense controls are shared across budgets. First-canvas work calibration is transductive and does not match whole-trajectory work; K128 non-thinking V2 retains a disclosed closest threshold outside the specified sparsity tolerance. Question-bootstrap uncertainty conditions on the observed engines; the six examined seed-43 tasks are descriptive.
''')
    write(output / 'final_complete.json', dict(status='complete', seed=42, questions_per_method=164,
        budgets=[256,128], modes=list(MODES), analysis_source_sha256=sha(__file__),
        files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}), True)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    publish(Path.cwd(), args.output.absolute())
