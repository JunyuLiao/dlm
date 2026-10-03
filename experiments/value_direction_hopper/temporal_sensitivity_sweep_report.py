"""Audit and regenerate beta/gamma sweep tables and figures from final shards."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from . import query_adaptive_guardrail as core
from .query_adaptive_guardrail_report import _bootstrap
from . import temporal_sensitivity_sweep as sweep
from .experiment import atomic, fingerprint, sha, shard_path


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load(root, base, manifests, specification, beta, gamma, target):
    name = sweep.label(beta, gamma, target)
    branch = sweep.condition_root(root, beta, gamma, target)
    frozen_path = branch/'configs/threshold.json'
    if not frozen_path.exists():
        return None, None, [f'{name}: no frozen threshold']
    frozen = json.loads(frozen_path.read_text())
    cfg = sweep.config_for(base, specification, beta, gamma, target)
    problems = []
    if (frozen.get('config_fingerprint') != cfg['fingerprint'] or
        frozen.get('specification_fingerprint') != specification['fingerprint'] or
        frozen.get('calibration_ids') != [r['id'] for r in manifests['calibration']] or
        frozen.get('validation_ids') != [r['id'] for r in manifests['validation']]):
        problems.append(f'{name}: threshold provenance mismatch')
    rows = []
    for prompt in manifests['final']:
        path = shard_path(branch/'final', f'T_s{target}', prompt)
        if not path.exists():
            problems.append(f'{name}: missing {prompt["id"]}')
            continue
        row = json.loads(path.read_text())
        identity = fingerprint([cfg['fingerprint'], 'final', f'T_s{target}',
            cfg['methods']['T'], frozen['policy'], cfg['m_ref'], prompt['id'],
            prompt['prompt_hash'], prompt['seed']])
        if (row.get('identity') != identity or row.get('status') != 'complete' or
            (row['id'], row['prompt_hash'], row['seed']) !=
            (prompt['id'], prompt['prompt_hash'], prompt['seed']) or
            row['steps'] != len(row['step_records'])):
            problems.append(f'{name}: invalid identity/step record {prompt["id"]}')
            continue
        counts = row['counts']
        for field in ('eligible', 'skipped'):
            if (counts['whole'][field] != counts['global'][field] +
                counts['local'][field] or
                counts['whole'][field] != sum(
                    step['counts']['whole'][field] for step in row['step_records'])):
                problems.append(f'{name}: physical counter mismatch {prompt["id"]}')
        if row.get('routing_path') and sha(row['routing_path']) != row.get('routing_sha256'):
            problems.append(f'{name}: corrupt routing shard {prompt["id"]}')
        rows.append(row)
    return frozen, rows, problems


def _plot(root, summaries, dense):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination = root/'plots'
    destination.mkdir(exist_ok=True)
    lookup = {(row['beta'], row['gamma'], row['target']): row
              for row in summaries}
    for target in core.TARGETS:
        for field, title, filename, fmt in (
            ('mean_calls', 'Mean denoising calls/canvas',
             f'calls_beta_gamma_s{target}.png', '{:.2f}'),
            ('accuracy', 'RULER4K score (%)',
             f'accuracy_beta_gamma_s{target}.png', '{:.1%}')):
            values = np.full((len(sweep.BETAS), len(sweep.GAMMAS)), np.nan)
            for i, beta in enumerate(sweep.BETAS):
                for j, gamma in enumerate(sweep.GAMMAS):
                    item = lookup.get((beta, 0. if beta == 0 else gamma, target))
                    if item:
                        values[i, j] = item[field] * (100 if field == 'accuracy' else 1)
            fig, axis = plt.subplots(figsize=(8.8, 5.4))
            image = axis.imshow(values, aspect='auto', cmap='viridis_r'
                if field == 'mean_calls' else 'viridis')
            axis.set(xticks=range(len(sweep.GAMMAS)),
                     xticklabels=[str(x) for x in sweep.GAMMAS],
                     yticks=range(len(sweep.BETAS)),
                     yticklabels=[str(int(x)) for x in sweep.BETAS],
                     xlabel='EMA memory γ', ylabel='Sensitivity strength β',
                     title=f'{target}% target: {title}')
            for i, beta in enumerate(sweep.BETAS):
                for j, gamma in enumerate(sweep.GAMMAS):
                    item = lookup.get((beta, 0. if beta == 0 else gamma, target))
                    if not item:
                        continue
                    measured = f"{item['actual_overall']:.1%}"
                    outcome = f"{item[field]:.2f}" if field == 'mean_calls' else f"{item[field]:.1%}"
                    marker = '†' if item['calibration_status'] != 'attained' else ''
                    axis.text(j, i, f'{outcome}{marker}\n{measured} actual',
                              ha='center', va='center', fontsize=8,
                              color='white' if values[i, j] > np.nanmedian(values)
                              else 'black')
            fig.colorbar(image, ax=axis, label=title)
            fig.text(.5, .01,
                'β=0: γ is inoperative; one measured point is repeated for display.\n'
                '† missed calibration/validation guardrails. Second line: actual physical sparsity.',
                ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .08, 1, 1))
            fig.savefig(destination/filename, dpi=160)
            plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, 5))
        for gamma in sweep.GAMMAS:
            values = [lookup.get((beta, gamma, target)) for beta in sweep.BETAS[1:]]
            axis.plot(sweep.BETAS[1:],
                [row['mean_calls'] if row else np.nan for row in values],
                marker='o', label=f'γ={gamma}')
            for beta, row in zip(sweep.BETAS[1:], values):
                if row:
                    axis.annotate(f"{row['actual_overall']:.1%}",
                        (beta, row['mean_calls']), xytext=(3, 4),
                        textcoords='offset points', fontsize=7)
        zero = lookup.get((0., 0., target))
        if zero:
            axis.axhline(zero['mean_calls'], color='#999999', linestyle=':',
                         label=f'β=0 ({zero["actual_overall"]:.1%} actual)')
        axis.axhline(dense.get('mean_calls', dense.get('mean_steps')),
                     color='black', linestyle='--',
                     label='matched-kernel dense')
        axis.set(xlabel='Sensitivity strength β',
                 ylabel='Mean denoising calls/canvas',
                 title=f'{target}% target: calls versus β; labels are actual sparsity')
        axis.grid(alpha=.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(destination/f'calls_vs_beta_s{target}.png', dpi=160)
        plt.close(fig)


def report(root):
    root = Path(root)
    base, manifests, specification = sweep.prepare(root)
    smoke_path = root/'configs/smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    problems = []
    if smoke.get('fingerprint') != specification['fingerprint'] or not smoke.get('passed'):
        problems.append('Smoke missing or invalid')
    summaries, per_example, per_task = [], [], []
    groups = {}
    for beta, gamma, target in sweep.combinations():
        name = sweep.label(beta, gamma, target)
        frozen, rows, issues = _load(root, base, manifests, specification,
                                     beta, gamma, target)
        problems.extend(issues)
        if not frozen or len(rows) != len(manifests['final']):
            continue
        groups[name] = rows
        measured = core.profile(rows)
        policy = frozen['policy']
        validated = next((item for item in frozen['validation']
                          if item['policy'] == policy), None)
        summaries.append(dict(condition=name, beta=beta, gamma=gamma,
            target=target, calibration_status=frozen['status'],
            calibration_points=frozen['tested_points'],
            actual_overall=measured['overall']['whole'],
            actual_global=measured['overall']['global'],
            actual_local=measured['overall']['local'],
            calibration_actual_overall=frozen['calibration']['metrics']['overall']['whole'],
            validation_actual_overall=validated['metrics']['overall']['whole']
                if validated else None,
            accuracy=measured['accuracy'], total_calls=measured['total_steps'],
            mean_calls=measured['mean_steps'],
            median_calls=measured['median_steps'], p90_calls=measured['p90_steps'],
            p95_calls=measured['p95_steps'], cap_count=measured['cap_count'],
            eligible_tiles=measured['counts']['whole']['eligible'],
            skipped_tiles=measured['counts']['whole']['skipped'],
            executed_tiles=measured['executed_tiles'],
            call1_sparsity=measured['phase_sparsity']['step1']['whole'],
            call2_sparsity=measured['phase_sparsity']['step2']['whole'],
            log_tau_early_local=policy['early']['local']['log_threshold'],
            log_tau_early_global=policy['early']['global']['log_threshold'],
            log_tau_late_local=policy['late']['local']['log_threshold'],
            log_tau_late_global=policy['late']['global']['log_threshold']))
        for row in rows:
            per_example.append(dict(condition=name, beta=beta, gamma=gamma,
                target=target, id=row['id'], task=row['task'],
                prompt_hash=row['prompt_hash'], seed=row['seed'],
                score=row['score'], calls=row['steps'],
                eligible=row['counts']['whole']['eligible'],
                skipped=row['counts']['whole']['skipped'],
                global_eligible=row['counts']['global']['eligible'],
                global_skipped=row['counts']['global']['skipped'],
                local_eligible=row['counts']['local']['eligible'],
                local_skipped=row['counts']['local']['skipped']))
        task_groups = defaultdict(list)
        for row in rows:
            task_groups[row['task']].append(row)
        for task in sorted(task_groups):
            subset = task_groups[task]
            per_task.append(dict(condition=name, target=target, task=task,
                n=len(subset), score=float(np.mean([r['score'] for r in subset])),
                mean_calls=float(np.mean([r['steps'] for r in subset]))))
    for target in core.TARGETS:
        anchor = groups.get(sweep.label(3., .5, target))
        if not anchor:
            continue
        by_id = {row['id']: row for row in anchor}
        for beta, gamma in sweep.GRID:
            name = sweep.label(beta, gamma, target)
            if name not in groups or name == sweep.label(3., .5, target):
                continue
            for row in groups[name]:
                original = by_id[row['id']]
                if min(2, len(row['step_records'])) != min(2, len(original['step_records'])):
                    problems.append(f'{name}: early call count differs from anchor {row["id"]}')
                    continue
                for index in range(min(2, len(row['step_records']))):
                    a, b = row['step_records'][index], original['step_records'][index]
                    if (a['counts'] != b['counts'] or a['accepted'] != b['accepted'] or
                        abs(a['processed_entropy_mean'] - b['processed_entropy_mean']) > 1.e-7):
                        problems.append(f'{name}: shared early execution mismatch {row["id"]}')
    baseline_rows = []
    for prompt in manifests['final']:
        path = shard_path(core.PARENT/'final', 'kernel_dense', prompt)
        baseline_rows.append(json.loads(path.read_text()))
    dense = core.profile(baseline_rows)
    index = {row['condition']: row for row in summaries}
    comparisons = []
    for target in core.TARGETS:
        anchor_name = sweep.label(3., .5, target)
        if anchor_name not in groups:
            continue
        anchor = index[anchor_name]
        for beta, gamma in sweep.GRID:
            name = sweep.label(beta, gamma, target)
            if name not in groups or name == anchor_name:
                continue
            item = index[name]
            differences = {kind: item[f'actual_{kind}'] - anchor[f'actual_{kind}']
                           for kind in ('overall', 'global', 'local')}
            score_ci = _bootstrap(groups[name], groups[anchor_name], 'score')
            calls_ci = _bootstrap(groups[name], groups[anchor_name], 'steps')
            comparisons.append(dict(condition=name, target=target,
                beta=beta, gamma=gamma,
                matched_actual_ogl=all(abs(x) <= .02 for x in differences.values()),
                delta_overall_sparsity=differences['overall'],
                delta_global_sparsity=differences['global'],
                delta_local_sparsity=differences['local'],
                delta_score=item['accuracy'] - anchor['accuracy'],
                score_ci_low=score_ci[0], score_ci_high=score_ci[1],
                delta_mean_calls=item['mean_calls'] - anchor['mean_calls'],
                calls_ci_low=calls_ci[0], calls_ci_high=calls_ci[1]))
    _write_csv(root/'summary.csv', summaries)
    _write_csv(root/'per_example.csv', per_example)
    _write_csv(root/'per_task.csv', per_task)
    _write_csv(root/'paired_comparisons.csv', comparisons)
    atomic(root/'summary.json', dict(summary=summaries, comparisons=comparisons,
        matched_dense=dict(accuracy=dense['accuracy'],
                           mean_calls=dense['mean_steps'])))
    if summaries:
        _plot(root, summaries, dense)
    expected = {sweep.label(*combo) for combo in sweep.combinations()}
    missing = sorted(expected - {s['condition'] for s in summaries})
    audit = dict(complete=not problems and not missing,
        source_sha256=sha(Path(__file__)),
        specification_fingerprint=specification['fingerprint'],
        conditions=len(summaries), expected_conditions=len(expected),
        final_prompts=len(manifests['final']), missing=missing,
        problems=problems, early_phase_parity=not any('early' in x for x in problems),
        calibration_gate_misses=[s['condition'] for s in summaries
            if s['calibration_status'] != 'attained'])
    atomic(root/'audit.json', audit)
    lines = ['# Temporal query-sensitivity β×γ sweep','',
        f"Audit: {'complete' if audit['complete'] else 'incomplete'}; "
        f"{len(summaries)}/{len(expected)} conditions on the same 130 prompts.",'',
        'The predeclared [plan](/home/exouser/ljy/dlm/experiments/value_direction_hopper/TEMPORAL_SENSITIVITY_SWEEP_PLAN.md) defines the '
        'grid and guardrails. The router/kernel and native decoder are unchanged. '
        'All methods share the previous target-specific early local/global thresholds '
        'and unit query weights on calls 1–2. Calls 3+ use '
        '`s_i=1+β u_i`, `u_i←γ u_i+(1−γ)1[top1_i changed]` from previous '
        'predictions. β=0 is calibrated once because γ is then inoperative.','',
        'Each combination was separately calibrated on 26 disjoint prompts by '
        'complete generation, targeting the previous T anchor’s measured '
        'overall/global/local tile sparsity. Calibration required ±2 percentage '
        'points in all three rates, the same first-two-call ceilings and call-count '
        'guardrails as the prior study. Up to four calibration-ranked candidates '
        'were checked on a separate 26-prompt validation set with ±3-point '
        'tolerance and a ≤3-point score drop versus matched dense. Policy JSON '
        'and every attempted point are preserved under `conditions/`. A status '
        'of `unattainable_under_guardrails` means the closest measured point is '
        'reported, not that it achieved a safe target. The final 130 prompts '
        'were not used for threshold selection and have been examined in prior '
        'work; these are exploratory results.','',
        f"Matched-kernel dense: {dense['accuracy']:.1%} score, "
        f"{dense['mean_steps']:.2f} mean calls/canvas. Accuracy is the existing "
        'RULER4K score on complete assembled outputs. A physical tile is skipped '
        'only when all valid query rows vote to skip; whole/global/local rates '
        'pool skipped and eligible counts across all executed calls.','',
        '| β | γ | Target | Gate | Actual O/G/L | RULER score | Mean / p90 calls | 48-cap | Executed tiles | Late logτ L/G |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for item in summaries:
        lines.append(f"| {item['beta']:g} | {item['gamma']:g} | {item['target']}% | "
            f"{item['calibration_status']} | "
            f"{item['actual_overall']:.1%}/{item['actual_global']:.1%}/"
            f"{item['actual_local']:.1%} | {item['accuracy']:.1%} | "
            f"{item['mean_calls']:.2f}/{item['p90_calls']:.1f} | "
            f"{item['cap_count']}/130 | {item['executed_tiles']:,} | "
            f"{item['log_tau_late_local']:.3f}/{item['log_tau_late_global']:.3f} |")
    lines += ['', '## Paired differences from β=3, γ=0.5', '',
        'On the same prompts, task-stratified paired bootstrap 95% intervals '
        'describe exploratory score/call differences. Interpret a parameter '
        'effect as a matched-sparsity comparison only if whole/global/local '
        'actual rates are each within 2 percentage points. Intervals crossing '
        'zero do not prove equivalence, and these repeated development prompts '
        'do not provide fresh confirmation.','',
        '| β | γ | Target | Matched actual O/G/L? | Δ score [95% CI] | Δ mean calls [95% CI] |',
        '|---:|---:|---:|---|---:|---:|']
    for item in comparisons:
        lines.append(f"| {item['beta']:g} | {item['gamma']:g} | "
            f"{item['target']}% | {'yes' if item['matched_actual_ogl'] else 'no'} | "
            f"{item['delta_score']:+.1%} "
            f"[{item['score_ci_low']:+.1%}, {item['score_ci_high']:+.1%}] | "
            f"{item['delta_mean_calls']:+.2f} "
            f"[{item['calls_ci_low']:+.2f}, {item['calls_ci_high']:+.2f}] |")
    lines += ['', '## Parameter effect at matched target budgets', '',
        'Each heatmap cell gives the observed mean calls (or score), with '
        'actual overall physical sparsity underneath. A dagger marks a policy '
        'that missed calibration/validation guardrails. The β=0 row repeats '
        'one measured point for display, not three separate γ experiments.','']
    for target in core.TARGETS:
        lines += [f'### {target}% target', '',
            f'![Mean calls by beta and gamma](plots/calls_beta_gamma_s{target}.png)',
            f'![Accuracy by beta and gamma](plots/accuracy_beta_gamma_s{target}.png)',
            f'![Calls versus beta](plots/calls_vs_beta_s{target}.png)', '']
    baseline50 = index.get(sweep.label(3., .5, 50))
    baseline70 = index.get(sweep.label(3., .5, 70))
    zero70 = index.get(sweep.label(0., 0., 70))
    high_memory = index.get(sweep.label(6., .9, 70))
    high_memory_ci = next((item for item in comparisons
        if item['condition'] == sweep.label(6., .9, 70)), None)
    if baseline50 and baseline70 and zero70 and high_memory and high_memory_ci:
        fifty = [item for item in summaries if item['target'] == 50]
        lines += ['## Interpretation', '',
            f"At the 50% target, all {len(fifty)} distinct grid points passed "
            f"development guardrails. Their final mean calls span "
            f"{min(x['mean_calls'] for x in fifty):.2f}–"
            f"{max(x['mean_calls'] for x in fifty):.2f}; β=3,γ=0.5 gives "
            f"{baseline50['mean_calls']:.2f}. The spread is small relative "
            'to the much larger failure region at 70%, and one point has a '
            'local final sparsity mismatch despite calibration passing.','',
            f"At 70%, β=0 yields {zero70['mean_calls']:.2f} calls and "
            f"{zero70['accuracy']:.1%} score at {zero70['actual_overall']:.1%} "
            f"actual sparsity, but misses the trajectory guardrails. The "
            f"previous β=3,γ=0.5 anchor yields {baseline70['mean_calls']:.2f} "
            f"calls and {baseline70['accuracy']:.1%} score at "
            f"{baseline70['actual_overall']:.1%} actual sparsity. Thus a "
            'nonzero temporal weighting is associated with much shorter '
            'trajectories at this operating point, while '
            'increasing β or γ does not give a monotone improvement.','',
            f"The shortest **attained** 70% policy in this grid is β=6,γ=0.9: "
            f"{high_memory['mean_calls']:.2f} calls and "
            f"{high_memory['accuracy']:.1%} score at "
            f"{high_memory['actual_overall']:.1%} overall / "
            f"{high_memory['actual_global']:.1%} global / "
            f"{high_memory['actual_local']:.1%} local sparsity. Against "
            f"β=3,γ=0.5 it saves {-high_memory_ci['delta_mean_calls']:.2f} "
            f"calls [paired 95% interval for the difference "
            f"{high_memory_ci['calls_ci_low']:+.2f}, "
            f"{high_memory_ci['calls_ci_high']:+.2f}], but skips "
            f"{baseline70['actual_overall']-high_memory['actual_overall']:.1%} "
            'fewer tiles overall and less in the local layers. Its score '
            f"difference is {high_memory_ci['delta_score']:+.1%} "
            f"[{high_memory_ci['score_ci_low']:+.1%}, "
            f"{high_memory_ci['score_ci_high']:+.1%}]. This is not proof that "
            'β=6,γ=0.9 is better at exactly equal physical sparsity or '
            'on fresh prompts. Even this setting remains above the matched '
            f"dense trajectory of {dense['mean_steps']:.2f} calls.",'',
            'A justified next test would freeze these two policies and '
            'compare them on fresh prompts with a tighter pre-calibrated '
            'whole/global/local match. Do not retune the present 130-prompt '
            'results after observing this heatmap.']
    lines += ['', 'The early-phase execution audit compares call-1/2 physical counts, '
        'accepted positions and entropy against β=3,γ=0.5 on every matched '
        'prompt. This isolates later sensitivity/threshold allocation, but '
        'different complete trajectories still change later-state difficulty. '
        'No latency or speedup conclusion follows from this sweep.','',
        'Per-example calls and physical counts are in `per_example.csv`; '
        'per-task scores are in `per_task.csv`. The 130-prompt cohort is '
        'exploratory; any apparent winner requires a fresh confirmation set '
        'with all thresholds frozen.']
    (root/'report.md').write_text('\n'.join(lines) + '\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=sweep.ROOT)
    args = parser.parse_args()
    print(json.dumps(report(args.root), indent=2))
