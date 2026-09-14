"""Evidence-backed synthesis after the contracted final report is complete.

This module never generates, calibrates, chooses candidates, or changes a
contract. Comparisons at unmatched budgets carry no accuracy-gain claim.
"""
import argparse
import json
from pathlib import Path

from experiments.diffusion_gemma_value_aware.report import csv_write, table
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .engine import contract
from .evidence import check_sources
from .final import PHASES, load_contract
from .policies import METHODS
from .protocol import ROOT, sha

PAIRS = (('value', 'no_value_control'), ('mass_value', 'mass'), ('risk', 'mass'),
    ('aligned', 'no_value_control'), ('centered', 'no_value_control'),
    ('mass_exact', 'mass'), ('compensate', 'zero_pv'))


def primary_view(row):
    return row['split'] == ('heldout24' if row['benchmark'] == 'aime26' else 'full')


def family(row):
    name, marker, target = row['condition'].rpartition('_s')
    return name if marker and target.isdigit() else row['condition']


def selected_samples(raw, benchmark, split, condition):
    rows = [r for r in raw if r['benchmark'] == benchmark and r['condition'] == condition
        and (split != 'heldout24' or not r['calibration'])]
    if len(rows) != len({r['id'] for r in rows}):
        raise ValueError('duplicate paired sample')
    return {r['id']: r['accuracy'] for r in rows}


def matched_controls(rows, raw, tolerance=.03):
    result = []
    for candidate in rows:
        if not primary_view(candidate):
            continue
        for name, reference in PAIRS:
            if family(candidate) != name:
                continue
            references = [r for r in rows if r['benchmark'] == candidate['benchmark']
                and r['split'] == candidate['split'] and family(r) == reference]
            if not references:
                raise ValueError('required matched control missing from complete report')
            metric = 'pv_omission' if name == 'compensate' else 'physical_sparsity'
            key = f'overall_{metric}'
            ref = min(references, key=lambda r: (abs(r[key]-candidate[key]), r['condition']))
            gap = abs(candidate[key]-ref[key])
            type_gaps = {k: abs(candidate[f'{k}_{metric}']-ref[f'{k}_{metric}']) for k in ('local', 'global')}
            matched = gap <= tolerance
            row = dict(benchmark=candidate['benchmark'], split=candidate['split'],
                candidate=candidate['condition'], reference=ref['condition'], budget_metric=metric,
                candidate_actual=candidate[key], reference_actual=ref[key], budget_gap=gap,
                matched_overall=matched, matched_both_attention_types=all(v <= tolerance for v in type_gaps.values()),
                local_budget_gap=type_gaps['local'], global_budget_gap=type_gaps['global'],
                score_delta=None, paired_ci95=None, mass_delta=None, output_error_delta=None,
                interpretation='No matched-budget effect claim' if not matched else
                    'Observed paired point comparison; not causal proof or independent-seed confirmation')
            if matched:
                a = selected_samples(raw, candidate['benchmark'], candidate['split'], candidate['condition'])
                b = selected_samples(raw, ref['benchmark'], ref['split'], ref['condition'])
                if not a or set(a) != set(b):
                    raise ValueError('control comparison sample identities differ')
                row.update(score_delta=candidate['accuracy']-ref['accuracy'],
                    paired_ci95=paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)]),
                    mass_delta=candidate['overall_mass']-ref['overall_mass'],
                    output_error_delta=candidate['overall_relative_error']-ref['overall_relative_error'])
            result.append(row)
    return result


def calibration_transfer(rows, frozen):
    result = []
    for row in rows:
        if row['split'] not in ('full', 'heldout24') or family(row) not in METHODS:
            continue
        condition = frozen['conditions'][row['condition']]
        source = condition['policy_sources'][row['benchmark']]
        path = Path(source['path'])
        raw = path.read_bytes()
        if sha(raw) != source['sha256']:
            raise ValueError('selected calibration policy changed')
        policy = json.loads(raw)
        if policy['policy'] != condition['thresholds'][row['benchmark']] or policy['heldout_used']:
            raise ValueError('transfer analysis cannot substitute a different policy')
        metric = condition['target_metric']
        for kind in ('local', 'global'):
            calibrated = policy['measured'][kind]
            actual = row[f'{kind}_{metric}']
            threshold = policy['policy'][kind]
            result.append(dict(benchmark=row['benchmark'], split=row['split'], condition=row['condition'],
                attention_type=kind, target=row['target'], target_metric=metric,
                calibrated=calibrated, final_actual=actual, calibration_error=calibrated-row['target'],
                final_error=actual-row['target'], calibration_to_final_shift=actual-calibrated,
                calibration_miss_over_two_points=abs(calibrated-row['target'])>.02,
                final_miss_over_two_points=abs(actual-row['target'])>.02,
                cap_one_unattainable=bool(threshold.get('cap_one') and threshold.get('unattainable')),
                threshold=threshold,
                interpretation='Same frozen policy on a different sample/length/trajectory distribution; NOT isolated causal length effect'))
    return result


def layer_behavior(rows):
    return [dict(benchmark=r['benchmark'], split=r['split'], condition=r['condition'],
        overall=r['overall_physical_sparsity'], global_s=r['global_physical_sparsity'], local_s=r['local_physical_sparsity'],
        global_minus_local=r['global_physical_sparsity']-r['local_physical_sparsity'],
        global_eligible=r['global_eligible'], local_eligible=r['local_eligible'],
        global_mass=r['global_denominator_mass'], local_mass=r['local_denominator_mass'])
        for r in rows if r['split'] in ('full', 'heldout24')]


def point_score_preserving(rows):
    """Point-score description only, never an equivalence/safety guarantee."""
    result = []
    for benchmark in ('aime26', 'longbench_v2'):
        selected = [r for r in rows if r['benchmark'] == benchmark and primary_view(r)]
        dense = next(r for r in selected if r['method'] == 'dense')
        for name in METHODS:
            if name in ('compensate', 'zero_pv'):
                continue
            points = [r for r in selected if family(r) == name and r['accuracy'] >= dense['accuracy']]
            best = max(points, key=lambda r: r['overall_physical_sparsity']) if points else None
            result.append(dict(benchmark=benchmark, method=name, condition=best['condition'] if best else None,
                dense_accuracy=dense['accuracy'], accuracy=best['accuracy'] if best else None,
                actual=best['overall_physical_sparsity'] if best else None,
                statement='Highest observed sparsity with point score at least dense; not statistical equivalence'
                    if best else 'No evaluated point score reaches dense; not proof that none exists'))
    return result


def audited_inputs(root, phase):
    execution = contract(root)
    frozen = load_contract(root, phase, execution)
    folder = root/'reports'/phase
    audit_path = folder/'audit.json'
    audit = json.loads(audit_path.read_text())
    if (not audit['complete'] or not audit['raw_complete'] or audit['missing'] or audit['violations']
            or audit['completed'] != frozen['expected_shards'] or audit['expected'] != frozen['expected_shards']
            or audit['contract_sha256'] != sha((root/'final_contracts'/f'{phase}.json').read_bytes())
            or audit['fingerprint'] != execution['fingerprint']):
        raise ValueError('complete audited final report required before synthesis')
    required = {'summary.json', 'per_sample.json', 'matched_sparsity.json', 'correlations.json',
        'main_ablation.csv', 'report.md'}
    if (not required <= set(audit['artifacts'])
            or audit['per_condition'] != {name: 60 for name in frozen['conditions']}):
        raise ValueError('complete per-condition and derived-artifact evidence required')
    report_path = Path(__file__).with_name('report.py')
    if audit['report_source_sha256'] != sha(report_path.read_bytes()):
        raise ValueError('regenerate the canonical report after report source changes')
    check_sources(audit['sources'])
    sources = dict(audit['sources'])
    sources[str(audit_path)] = sha(audit_path.read_bytes())
    sources[str(report_path)] = audit['report_source_sha256']
    for name, expected in audit['artifacts'].items():
        sources[str(folder/name)] = expected
    check_sources(sources)
    return frozen, sources, {name: json.loads((folder/f'{name}.json').read_text())
        for name in ('summary', 'per_sample', 'matched_sparsity', 'correlations')}


def write_report(dest, phase, data, controls, transfer, behavior, preserving):
    rows = data['summary']
    text = '# Value-aware BLASST — empirical synthesis\n\n'
    text += ('Full contracted comparison. ' if phase == 'full' else
        '**Interim50% comparison: remaining target curves are still required.** ')
    text += 'All statements below refer to the audited small AIME26/LongBench v2 experiment, not general benchmark performance.\n\n'
    text += '## Main comparison\n\n'
    main = [dict(benchmark=r['benchmark'], split=r['split'], condition=r['condition'],
        n=r['count'], target=r['target'], threshold=threshold_text(r), accuracy=r['accuracy'],
        dense_delta=r['delta'], physical=r['overall_physical_sparsity'], global_s=r['global_physical_sparsity'],
        local_s=r['local_physical_sparsity'], mass=r['overall_denominator_mass'],
        exact_pv_mass=r['overall_mass'], error=r['overall_relative_error'], agreement=r['token_agreement'])
        for r in rows if r['split'] == 'full' and (family(r) in METHODS or r['method'] == 'dense')]
    csv_write(dest/'main_comparison.csv', main)
    text += 'All metrics in this document are fractions/ratios. The compact table shows50% targets; CSVs and the canonical report contain all frozen settings and all30 AIME results, including full30 and previously exposed24 views.\n\n'
    text += table([r for r in main if r['target'] in (0., .5)],
        ['benchmark', 'condition', 'n', 'accuracy', 'physical', 'global_s', 'local_s', 'mass', 'error', 'agreement'])+'\n\n'
    text += '## What extra V information adds\n\n'
    text += 'The following controls separate value information from online-state and threshold conventions. Comparisons choose the nearest observed actual-budget control within3 percentage points; no interpolation or target-based matching. Local/global budget gaps are exported separately, since matching the whole model can hide a different allocation.\n\n'
    text += table(controls, ['benchmark', 'candidate', 'reference', 'budget_metric', 'budget_gap',
        'matched_overall', 'matched_both_attention_types', 'score_delta', 'paired_ci95', 'output_error_delta'])+'\n\n'
    text += 'Positive paired point estimates are not universal improvements. Intervals use the existing prompt bootstrap; dependent configurations, multiple comparisons and a single paired seed limit inference. Compensation targets full-PV replacement, not physical deletion.\n\n'
    text += '## Accuracy versus measured sparsity\n\n'
    text += table(preserving, ['benchmark', 'method', 'condition', 'dense_accuracy', 'accuracy', 'actual'])+'\n\n'
    text += 'These are point-score descriptions, **not accuracy-equivalence tests or guarantees of safe sparsity**. See the canonical nearest-BLASST comparisons and paired intervals for actual-budget comparisons against original/aggressive BLASST.\n\n'
    text += '## Calibration transfer and global/local behavior\n\n'
    compact = [r for r in transfer if r['split'] == 'full' and r['target'] == .5]
    text += table(compact, ['benchmark', 'condition', 'attention_type', 'calibrated', 'final_actual',
        'calibration_to_final_shift', 'cap_one_unattainable'])+'\n\n'
    text += 'All targets/types, selected thresholds and two-point misses are in calibration_transfer.csv. Original BLASST uses its established inverse-valid-length rule, independently capped at1; verified unattainable types deploy constant1. New risks retain their frozen scalar thresholds. Changes here mix sample, sequence-length and sparse-trajectory effects; they do not isolate length causally. No final result changes calibration.\n\n'
    text += 'Physical ratios divide summed skipped tiles by summed eligible tiles. Layer_behavior.csv gives global/local counts and masses; canonical routing_marginals.csv and per_layer_head_step.csv.gz give the finer breakdown without averaging ratios.\n\n'
    text += '## Diagnostic explanation and information costs\n\n'
    text += 'The independent tuning reports [mechanism checks](../../diagnostics/report.md) and [distribution checks](../../distribution_diagnostics/report.md) quantify the model’s nearly constant scalar value-norm signal, vector cancellation, loose mass estimates, worst-row inflation and non-Gaussian Sol proxies. They use shared dense states and cannot replace the downstream tables above. Exact mass/contribution rankings use extra information and work, not task-optimal or deployable oracles.\n\n'
    text += 'The compatibility table distinguishes when QK, block softmax, full PV, means/norms and past outputs are required. [Canonical execution accounting](../../reports/'+phase+'/execution_work.csv) separates whole-block deletion, softmax omission, full-PV omission and compensation. All paths in this reference compute diagnostic work. No custom kernel, latency, throughput or speedup is measured.\n\n'
    text += '## Mass, agreement and accuracy\n\n'
    text += table(data['correlations'], ['benchmark', 'scope', 'metric', 'n', 'pearson_with_accuracy'])+'\n\n'
    text += 'These correlations are descriptive across dependent settings, not causal effects. Denominator mass is always1 for compensation/zero-PV, so their exact-PV mass and output error matter instead. Token agreement compares every generated position, including after divergence and missing/extra positions.\n\n'
    text += '## Limits and next decision\n\n'
    text += 'AIME26 full30 includes calibration6; the other24 were exposed in previous work and are exploratory. V2 is multiple choice:10 single-document QA,10 multi-document QA and10 code-repository understanding, with a32k cap that truncates14 questions. It is not full-context v2 or generative summarization/code completion. Native temperature0 selects the seeded0.4–0.8 schedule, not greedy decoding.\n\n'
    text += 'Do not select a winner solely from target attainment or attention error. Review matched-budget task scores, local/global allocation and uncertainty together. A promising mechanism still needs additional paired-seed confirmation where practical. Any subsequent design informed by these final results must be labeled exploratory. The final human-readable conclusion and next bounded revision must reflect these actual tables, including negative results; this generated synthesis is not a completion assertion.\n'
    (dest/'report.md').write_text(text)


def build(root, phase):
    frozen, sources, data = audited_inputs(root, phase)
    rows = data['summary']
    controls = matched_controls(rows, data['per_sample'])
    transfer = calibration_transfer(rows, frozen)
    behavior = layer_behavior(rows)
    preserving = point_score_preserving(rows)
    dest = root/'synthesis'/phase
    dest.mkdir(parents=True, exist_ok=True)
    datasets = dict(matched_controls=controls, calibration_transfer=transfer,
        layer_behavior=behavior, point_score_preserving=preserving)
    if any(not values for values in datasets.values()):
        raise ValueError('required synthesis evidence is missing')
    for name, values in datasets.items():
        _write(dest/f'{name}.json', values)
        csv_write(dest/f'{name}.csv', values)
    write_report(dest, phase, data, controls, transfer, behavior, preserving)
    names = [f'{n}.{ext}' for n in datasets for ext in ('json', 'csv')] + ['main_comparison.csv', 'report.md']
    _write(dest/'audit.json', dict(complete=True, phase=phase, source_scope='completed final report, raw source checks and frozen calibration provenance',
        fingerprint=frozen['fingerprint'], sources=sources, code_sha256=sha(Path(__file__).read_bytes()),
        artifacts={name: sha((dest/name).read_bytes()) for name in names},
        goal_completion_claim=False))
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=PHASES, default='full')
    args = parser.parse_args()
    print(json.dumps({k: len(v) for k, v in build(args.output, args.phase).items()}))


if __name__ == '__main__':
    main()
