"""Raw-derived, paired comparisons among the three frozen directional methods.

This is an additive analysis, not a threshold search or GPU execution path.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci

ROOT = Path('results/diffusion_gemma_jl_directional50_v6')
PAIRS = (
    ('jl_sign_r32_s50', 'jl_gaussian_r32_s50'),
    ('cancel_guard_gaussian_r32_s50', 'jl_gaussian_r32_s50'),
    ('cancel_guard_gaussian_r32_s50', 'jl_sign_r32_s50'),
)
LABELS = {
    'dense': 'Dense',
    'blasst_original_s50': 'Original BLASST',
    'blasst_aggressive_s50': 'Aggressive BLASST',
    'mass_s50': 'Mass-only',
    'full_centered_s50': 'Full-dimensional centered',
    'jl_gaussian_r32_s50': 'Gaussian32 centered',
    'jl_sign_r32_s50': 'Random-sign32 centered',
    'cancel_guard_gaussian_r32_s50': 'Gaussian32 + cancellation guard',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(raw):
    result = []
    for benchmark, count in (('aime26', 30), ('longbench_v2', 50)):
        for candidate, reference in PAIRS:
            groups = []
            for label in (candidate, reference):
                rows = [r for r in raw if r['benchmark'] == benchmark and r['condition'] == label]
                group = {r['id']: r for r in rows}
                if len(rows) != count or len(group) != count:
                    raise ValueError('Require every question exactly once, including calibration members')
                groups.append(group)
            a, b = groups
            if set(a) != set(b):
                raise ValueError('Directional results are not paired on identical IDs')
            delta = [a[i]['accuracy']-b[i]['accuracy'] for i in sorted(a)]
            lo, hi = paired_bootstrap_ci(delta)
            entry = dict(benchmark=benchmark, candidate=candidate, reference=reference, count=count,
                candidate_correct=sum(r['accuracy'] for r in a.values()),
                reference_correct=sum(r['accuracy'] for r in b.values()),
                accuracy_delta_pp=100*sum(delta)/count, paired_ci95_pp=[100*lo, 100*hi])
            gaps, in_band = [], [True, True]
            for kind in ('overall', 'global', 'local'):
                values = []
                for index, group in enumerate(groups):
                    eligible = sum(r['aggregates'][kind]['eligible'] for r in group.values())
                    skipped = sum(r['aggregates'][kind]['skipped'] for r in group.values())
                    if not 0 <= skipped <= eligible or eligible <= 0:
                        raise ValueError('Invalid physical tile counts')
                    value = skipped/eligible; values.append(value)
                    in_band[index] &= .48-1e-12 <= value <= .52+1e-12
                entry[f'candidate_{kind}_sparsity_pct'] = 100*values[0]
                entry[f'reference_{kind}_sparsity_pct'] = 100*values[1]
                gap = values[0]-values[1]; gaps.append(gap)
                entry[f'{kind}_sparsity_gap_pp'] = 100*gap
            entry.update(within2pp_all_types=all(abs(g) <= .02+1e-12 for g in gaps),
                within3pp_all_types=all(abs(g) <= .03+1e-12 for g in gaps),
                candidate_within48_52_all_types=in_band[0], reference_within48_52_all_types=in_band[1],
                interpretation='Fixed-policy/seed paired prompt bootstrap,20000 draws; full30/50 denominators include calibration. Exploratory, no multiplicity correction or threshold refitting; actual-count sparsity, no interpolation.')
            result.append(entry)
    return result


def plot_rows(summary, benchmark):
    rows = [r for r in summary if r['benchmark'] == benchmark and r['split'] == 'full']
    if len(rows) != len(LABELS) or {r['condition'] for r in rows} != set(LABELS):
        raise ValueError('Require all eight complete conditions for each plot')
    for row in rows:
        expected = 0 if row['condition'] == 'dense' else .5
        if row['target'] != expected:
            raise ValueError('50%-only figure cannot include another target')
    return sorted(rows, key=lambda r: list(LABELS).index(r['condition']))


def plots(root, summary):
    # Additive presentation correction: inherited canonical figures use an old
    # hard-coded "50/75%" title. Preserve those hash-audited originals unchanged.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = root/'directional_figures'; directory.mkdir(exist_ok=True)
    artifacts = []
    for benchmark in ('aime26', 'longbench_v2'):
        rows = plot_rows(summary, benchmark)
        title = 'AIME26 (30 questions)' if benchmark == 'aime26' else 'LongBench v2 (50 questions)'
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        metrics = [('accuracy', 'Accuracy (%)'), ('overall_mass', 'Retained dense mass (%)'),
                   ('token_agreement', 'Token agreement (%)')]
        for index, row in enumerate(rows):
            for axis, (metric, label) in zip(axes, metrics):
                axis.scatter(100*row['overall_physical_sparsity'], 100*row[metric],
                    color=f'C{index}', marker='x' if index == 0 else 'o', s=58,
                    label=LABELS[row['condition']])
                axis.set(xlabel='Measured physical tile sparsity (%)', ylabel=label, xlim=(-2, 58))
                axis.grid(alpha=.2)
        fig.suptitle(f'{title}: 50% target only; measured sparsity; no hardware-speedup claim')
        axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, .5), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .95))
        path = directory/f'{benchmark}_tradeoffs_s50.png'; fig.savefig(path, dpi=140); plt.close(fig)
        artifacts.append(path)
        sparse = rows[1:]
        fig, axis = plt.subplots(figsize=(11, 6))
        for index, kind in enumerate(('overall', 'global', 'local')):
            axis.bar([i+(index-1)*.25 for i in range(len(sparse))],
                [100*r[f'{kind}_physical_sparsity'] for r in sparse], width=.25, label=kind)
        axis.axhspan(48, 52, color='green', alpha=.12, label='Requested 48–52% band')
        axis.axhline(50, color='black', linestyle='--', linewidth=1)
        axis.set_xticks(range(len(sparse)), [LABELS[r['condition']] for r in sparse], rotation=25, ha='right')
        axis.set(ylabel='Measured physical tile sparsity (%)', ylim=(0, 60),
                 title=f'{title}: target 50%, actual overall/global/local')
        axis.legend(fontsize=8); axis.grid(axis='y', alpha=.2); fig.tight_layout()
        path = directory/f'{benchmark}_target_actual_s50.png'; fig.savefig(path, dpi=140); plt.close(fig)
        artifacts.append(path)
    return artifacts


def run(root=ROOT):
    audit = json.loads((root/'audit.json').read_text())
    proof = json.loads((root/'regeneration_verification.json').read_text())
    if not (audit['complete'] and audit['completed'] == audit['expected'] == 640
            and not audit['missing'] and not audit['violations'] and proof['passed']
            and proof['completed'] == 640 and proof['inference_performed'] is False
            and proof['audit_sha256'] == digest(root/'audit.json')):
        raise ValueError('Require the independently verified complete640-output report')
    for name, expected in audit['artifacts'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Base report artifact changed: {name}')
    rows = analyze(json.loads((root/'per_sample.json').read_text()))
    summary = json.loads((root/'summary.json').read_text())
    output = root/'directional_pairwise.json'
    output.write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    csv_path = root/'directional_pairwise.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = '# Paired directional comparisons\n\n'
    text += 'All30 AIME26 and all50 LongBench v2 questions are included, including calibration members. Positive accuracy deltas favor the candidate. CIs condition on fixed policies/seeds; they are exploratory and not multiplicity-corrected. These previously examined samples are not fresh held-out confirmation.\n\n'
    text += '## Main empirical interpretation\n\n'
    text += 'Directional routing has not established a downstream accuracy–sparsity advantage here. Gaussian32 matches the full-dimensional reference on AIME, but trails it on LongBench. Random-sign32 is slightly lower in accuracy than Gaussian32 on both benchmarks; the direct paired intervals below do not establish a projection-family winner. The cancellation guard is lower in accuracy than either plain projection on both benchmarks, despite using two32-dimensional sketches. Its direct paired intervals also include zero.\n\n'
    text += 'AIME is encouraging relative to aggressive BLASST, but the accuracy gains remain inconclusive. On LongBench, aggressive BLASST has the highest observed accuracy at nearly the same overall physical sparsity as the projected methods. The full-dimensional reference exceeds mass-only by only one question on each benchmark; those differences remain inconclusive. Original BLASST operates at substantially lower physical sparsity, especially in local layers, and is not a matched-budget baseline.\n\n'
    text += 'Lower local operator error does not reliably predict better downstream accuracy: on the sampled shared LongBench states, all three projected methods reduce error relative to aggressive BLASST, yet their final accuracy is lower. This diagnostic uses identical QKV but method-specific masks and slightly different sparsities; it does not establish a causal explanation. The superseded uncentered control was not run, so this iteration alone does not isolate the benefit of centering.\n\n'
    text += '| Method | AIME correct/30 | AIME sparsity overall/global/local (%) | LongBench correct/50 | LongBench sparsity overall/global/local (%) |\n'
    text += '| --- | --- | --- | --- | --- |\n'
    groups = {b: {r['condition']: r for r in plot_rows(summary, b)} for b in ('aime26', 'longbench_v2')}
    for condition, label in LABELS.items():
        a, b = groups['aime26'][condition], groups['longbench_v2'][condition]
        sparsity = lambda r: '/'.join(f"{100*r[k+'_physical_sparsity']:.2f}" for k in ('overall', 'global', 'local'))
        text += f"| {label} | {a['correct']:g}/30 | {sparsity(a)} | {b['correct']:g}/50 | {sparsity(b)} |\n"
    text += '\nAll six follow-on calibration policies satisfied48–52% overall/global/local on their calibration questions. Final AIME transfer stays within that band; all three projected LongBench runs miss it overall and globally, while local layers stay within it. The requested final50±2% condition is therefore NOT met on LongBench. Thresholds are left frozen; these are calibration-transfer failures, not relabeled50% achievements.\n\n'
    text += 'The shared diagnostics cover32 AIME and34 LongBench saved states, with one selected query head and128-query tile per state, from the earlier short-output calibration trajectories. Type labels overlap and are query-dependent. For Gaussian32 on LongBench, low-attention queries occur in3334/3745 tiles, distinctive queries in672, redundant queries in58, and internally cancelling queries in47. Of those,1623,216,34,and29 tiles respectively are skipped.641/672 distinctive tiles mix query types. Aggressive BLASST skips234/718 distinctive-containing and24/47 cancelling-containing tiles on the same QKV, but its retained history differs. These are descriptive comparisons, not mutually exclusive block classes or matched-history causal effects.\n\n'
    text += 'The cancellation guard reduces skipped cancelling-containing LongBench tiles from29 to27 in this sample, without improving final accuracy. Random-sign32 has fewer threshold-crossing risk underestimates than Gaussian32 here, but that diagnostic advantage also does not produce an accuracy win. No sampled severe underestimation (projected risk below half the full risk when full risk≥0.05) was observed; limited early-state coverage is not a safety guarantee. No end-to-end multi-seed or rank8/16 conclusion is claimed.\n\n'
    text += '## Direct paired comparisons\n\n'
    text += '| Benchmark | Candidate − reference | Correct | Accuracy delta (pp),95% CI | Sparsity gap overall/global/local (pp) | Matched within2pp,all types |\n'
    text += '| --- | --- | --- | --- | --- | --- |\n'
    for r in rows:
        lo, hi = r['paired_ci95_pp']
        gaps = '/'.join(f"{r[k+'_sparsity_gap_pp']:+.2f}" for k in ('overall', 'global', 'local'))
        text += f"| {r['benchmark']} | {r['candidate']} − {r['reference']} | {r['candidate_correct']:g}/{r['count']} vs {r['reference_correct']:g}/{r['count']} | {r['accuracy_delta_pp']:+.2f},[{lo:+.2f},{hi:+.2f}] | {gaps} | {r['within2pp_all_types']} |\n"
    text += '\nPhysical sparsity is recomputed from summed eligible/skipped tile counts, not averaged sample percentages. JSON/CSV also flag whether each final condition actually lies in48–52% for all three attention groupings. Unmatched budgets do not establish an accuracy–sparsity win. See the main report for baseline comparisons, mass, operator error, token agreement and block-type diagnostics. No inference or threshold selection is performed here.\n'
    plot_artifacts = plots(root, summary)
    text += '\n## Corrected figure labels\n\nThe inherited canonical tradeoff figures have a stale “50/75%” title; their plotted data contain only the50% sweep and dense. The following additive figures correct that title without modifying any audited canonical artifact.\n\n'
    for path in plot_artifacts:
        text += f'![{path.stem}]({path.relative_to(root)})\n\n'
    report = root/'directional_pairwise.md'; report.write_text(text)
    sources = {str(p): digest(p) for p in (Path(__file__), Path('tests/test_jl_directional50_analysis.py'),
        root/'audit.json', root/'regeneration_verification.json', root/'per_sample.json', root/'summary.json')}
    result = dict(passed=True, inference_performed=False, threshold_selection_performed=False,
        comparisons=len(rows), sources=sources,
        artifacts={str(p.relative_to(root)): digest(p) for p in (output, csv_path, report, *plot_artifacts)})
    (root/'directional_pairwise_audit.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args(); print(json.dumps(run(args.output)))
