"""Descriptive, paired interpretation after the complete raw-shard audit only."""
from collections import defaultdict

from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_value_aware.report import table
from experiments.diffusion_gemma_value_aware.report_metrics import correlation


def analyze(rows, raw):
    """No selection or inference: preserve the frozen policies and official scores."""
    index = {(r['benchmark'], r['split'], r['condition']): r for r in rows}
    samples = defaultdict(dict)
    for r in raw:
        splits = ['full']
        if r['benchmark'] == 'aime26':
            splits.append('calibration6' if r['calibration'] else 'noncalibration24')
        else:
            splits.append('previous30' if r['previous_manifest_member'] else 'new20')
        for split in splits:
            samples[r['benchmark'], split, r['condition']][r['id']] = r['accuracy']
    controls, associations, unattainable = [], [], []
    for row in rows:
        if row['name'] in ('value', 'risk', 'aligned'):
            key = row['benchmark'], row['split'], f"mass_s{int(100*row['target'])}"
            ref = index[key]
            a = samples[row['benchmark'], row['split'], row['condition']]
            b = samples[key]
            if set(a) != set(b):
                raise ValueError('Mass-control comparison requires identical sample IDs')
            ci = paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)])
            gaps = {k: 100*(row[f'{k}_physical_sparsity']-ref[f'{k}_physical_sparsity'])
                    for k in ('overall', 'global', 'local')}
            controls.append(dict(benchmark=row['benchmark'], split=row['split'],
                candidate=row['condition'], reference=ref['condition'],
                accuracy_delta_pp=100*(row['accuracy']-ref['accuracy']),
                paired_ci95_pp=[100*x for x in ci],
                **{f'{k}_sparsity_gap_pp':v for k,v in gaps.items()},
                within3pp_overall=abs(gaps['overall']) <= 3.,
                within3pp_all_types=all(abs(v) <= 3. for v in gaps.values())))
        if row['split'] == 'full':
            for kind, policy in (row['thresholds'] or {}).items():
                if policy.get('unattainable') and policy.get('cap_one'):
                    unattainable.append(dict(benchmark=row['benchmark'], condition=row['condition'],
                        attention_type=kind, target=100*row['target'], threshold=1.,
                        final_actual=100*row[f'{kind}_physical_sparsity']))
    for benchmark, split in sorted({(r['benchmark'], r['split']) for r in rows}):
        selected = [r for r in rows if r['benchmark']==benchmark and r['split']==split and r['name']!='dense']
        for metric in ('overall_mass', 'token_agreement', 'overall_relative_error'):
            associations.append(dict(benchmark=benchmark, split=split, metric=metric, n=len(selected),
                pearson_with_accuracy=correlation([r[metric] for r in selected], [r['accuracy'] for r in selected]),
                interpretation='Descriptive dependent configurations, including any exact aliases; no causal or significance claim'))
    return dict(mass_controls=controls, correlations=associations, unattainable=unattainable)


def narrative(rows, findings):
    text = '## Empirical reading\n\n'
    for benchmark in ('aime26', 'longbench_v2'):
        selected = [r for r in rows if r['benchmark']==benchmark and r['split']=='full']
        dense = next(r for r in selected if r['name']=='dense')
        text += f"{benchmark}: dense scored {dense['correct']:g}/{dense['count']} ({100*dense['accuracy']:.1f}%). "
        if benchmark == 'longbench_v2':
            text += (f"Dense had {dense['unparsed_answers']}/{dense['count']} unparseable answers and "
                     f"{dense['length_terminated']}/{dense['count']} length-limited generations. "
                     'Scores therefore include answer-completion behavior under the fixed128-token budget. ')
        else:
            held = next(r for r in rows if r['benchmark']==benchmark and r['split']=='noncalibration24' and r['name']=='dense')
            text += f"The noncalibration dense score was {held['correct']:g}/{held['count']}; full30 includes fitting examples. "
        text += '\n\n'
        changes = []
        for method in ('blasst_original','blasst_aggressive','value','mass','risk','aligned'):
            a = next(r for r in selected if r['name']==method and r['target']==.5)
            b = next(r for r in selected if r['name']==method and r['target']==.75)
            changes.append(dict(method=method,
                actual_s50=f"{100*a['overall_physical_sparsity']:.1f}%",
                actual_s75=f"{100*b['overall_physical_sparsity']:.1f}%",
                score_s50=f"{a['correct']:g}/{a['count']}", score_s75=f"{b['correct']:g}/{b['count']}",
                accuracy_change_pp=100*(b['accuracy']-a['accuracy']),
                mass_change_pp=100*(b['overall_mass']-a['overall_mass']),
                agreement_change_pp=100*(b['token_agreement']-a['token_agreement'])))
        text += table(changes, ('method','actual_s50','actual_s75','score_s50','score_s75','accuracy_change_pp','mass_change_pp','agreement_change_pp'))+'\n\n'
    text += '### Value-aware candidates versus mass-only control\n\n'
    text += ('These are paired point comparisons at the same target. A positive score delta alone does not establish a value-vector benefit: inspect actual tile-budget differences, including both attention types, and the prompt-bootstrap interval. All confidence intervals below are percentage points.\n\n')
    display = []
    for row in findings['mass_controls']:
        if row['split'] != ('noncalibration24' if row['benchmark']=='aime26' else 'full'):
            continue
        display.append(dict(row, paired_ci95_pp='['+', '.join(f'{v:.1f}' for v in row['paired_ci95_pp'])+']'))
    text += table(display, ('benchmark','split','candidate','reference','accuracy_delta_pp','paired_ci95_pp','overall_sparsity_gap_pp','global_sparsity_gap_pp','local_sparsity_gap_pp','within3pp_all_types'))+'\n\n'
    text += '### Calibration saturation and metric associations\n\n'
    if findings['unattainable']:
        text += 'The following targets were declared unattainable by the existing calibration and deployed exact lambda1. Final measured values are shown; the requested targets are not claimed as achieved.\n\n'
        text += table(findings['unattainable'], ('benchmark','condition','attention_type','target','threshold','final_actual'))+'\n\n'
    else:
        text += 'No policy carries an unattainable-at-lambda1 flag. Measured target errors are still reported.\n\n'
    text += 'Across the twelve sparse configurations per benchmark (dependent points, including any exact aliases), these correlations are descriptive only; dense is excluded to avoid its trivial perfect mass/agreement anchor.\n\n'
    text += table([r for r in findings['correlations'] if r['split']==('noncalibration24' if r['benchmark']=='aime26' else 'full')],
                  ('benchmark','split','metric','n','pearson_with_accuracy'))+'\n\n'
    return text
