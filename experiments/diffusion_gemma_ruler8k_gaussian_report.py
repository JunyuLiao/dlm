"""Rank-sweep interpretation and raw-only audit on top of the existing reporter."""
from collections import defaultdict
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments import diffusion_gemma_ruler8k_gaussian_sweep as study
from experiments import diffusion_gemma_ruler8k_projection_diagnostics as diagnostics

base, engine = study.base, study.reporting


def sensitivity(raw):
    groups = defaultdict(list)
    for r in raw:
        if r['task'] != 'vt': groups[r['condition']].append(r)
    return [dict(condition=k, count=len(v), accuracy=float(np.mean([r['accuracy'] for r in v])),
        delta=float(np.mean([r['accuracy']-r['dense_accuracy'] for r in v])), paired_ci95=engine.bootstrap(v),
        convention='Post-hoc sensitivity excluding VT; primary score retains all13 tasks.') for k,v in groups.items()]


def write_report(root, setup, rows, tasks, policies, thresholds, comparisons, audit, operators):
    pct = lambda x: f'{100*x:.2f}'
    lines = ['# RULER8K Gaussian projection-rank sweep', '',
        f'Audited generations: {audit["completed"]}/{study.EXPECTED}; complete: {audit["complete"]}. '
        'This extends the completed v14 experiment without modifying its results or kernels.', '',
        '## Controlled setup', '',
        '130 matched final questions,10 each across all13 RULER tasks. Same disjoint26 calibration (2/task) and13 development (1/task) questions. '
        'All1170 v14 final outputs are reused;1300 new final outputs cover Gaussian1/4/8/16/32 at50% and75%. '
        'The cached pool and final examples have been examined previously; this is not fresh held-out confirmation.', '',
        f'Pinned DiffusionGemma revision `{setup["revision"]}`; BF16;128-query×64-KV physical tiles; prefix+canvas skippable; '
        'native structural masks/GQA; local window1024. Original V is used for retained attention, renormalized without correction. '
        'FP32 projection/routing, TF32 disabled; fixed Gaussian matrices per layer/native KV head with seed1729, N(0,1/r) entries. '
        'Matrices follow the existing rank-dependent seed derivation and are not nested across ranks.', '',
        'Same prompts, generation seed42, official task budgets,256-token canvas,48-step limit, thinking=False. '
        'Requested temperature0 remains the native0.8→0.4 schedule sentinel, not greedy decoding. '
        '8K is the official generator total budget before chat-template overhead. Official RULER scoring is used, including fractional multi-answer credit.', '',
        '| Task | Final / calibration | Output-token budget |', '|---|---:|---:|']
    for task in setup['tasks']:
        r = next(r for r in setup['final'] if r['task'] == task)
        lines.append(f'|{task}|10 / 2|{r["generation_budget"]}|')
    lines += ['', '## Accuracy at measured sparsity', '',
        'Sparsity is sum(skipped eligible physical tiles)/sum(eligible physical tiles), not a mean of sample percentages. '
        'Mass is dense attention probability on retained positions on each candidate’s own QKV states, averaged over valid query rows. '
        'Token agreement counts equal token IDs at every position, including after divergence; missing/extra positions disagree. '
        'Operator error is reported separately on identical cached dense QKV states.', '',
        '|Condition|Score %|Δ dense pp|95% paired CI pp|Actual %|Global %|Local %|Mass %|Token agreement %|',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    bylabel = {r['condition']:r for r in rows}
    labels = ['dense'] + [f'{n}_s{int(t*100)}' for t in base.TARGETS for n in ['blasst','mass','full_centered',*[f'jl_gaussian_r{r}' for r in study.DIAGNOSTIC_RANKS]]]
    for label in labels:
        if label not in bylabel: continue
        r = bylabel[label]; ci = ', '.join(pct(x) for x in r['paired_ci95'])
        lines.append(f'|{label}|{pct(r["accuracy"])}|{pct(r["delta"])}|[{ci}]|{pct(r["overall_physical_sparsity"])}|{pct(r["global_physical_sparsity"])}|{pct(r["local_physical_sparsity"])}|{pct(r["overall_mass"])}|{pct(r["token_agreement"])}|')
    lines += ['', 'Paired uncertainty:10,000 task-stratified prompt bootstrap draws, seed42. These are unadjusted descriptive intervals across many rank comparisons, not multiplicity-controlled discovery tests.', '',
        '![Accuracy, mass and agreement versus actual sparsity](figures/tradeoffs.png)', '',
        '![Target and achieved physical sparsity](figures/target_actual.png)', '',
        '## Thresholds and calibration', '',
        'Each new rank is calibrated on26 calibration questions with full official output budgets. Existing dense-state empirical-CDF proposals and sparse-trajectory refinement are reused, with at most16 joint points. '
        'Frozen thresholds must be within2 percentage points of target overall and independently for local/global. Final outcomes never select thresholds. '
        'All prior full/Gaussian2/mass/BLASST policies are reused unchanged.', '',
        'BLASST keeps the established inverse-valid-length rule: effective log(lambda)=log_scale−log(valid KV length). '
        'Both thresholds are capped at1 for50%. The measured joint lambda_local=lambda_global=1 calibration ceiling was60.18%; only the75% request therefore permits above-one thresholds. '
        'The50% BLASST point redistributes the local/global budget under that cap and is not layer-budget matched to the other methods.', '',
        '|Condition|Kind|τ / log scale|Calibration actual %|Calibration rounds|', '|---|---|---:|---:|---:|']
    for p in policies:
        for kind in base.KINDS:
            entry = p['policy'][kind]
            value = f'log_scale={entry["log_scale"]:.7g}' if 'log_scale' in entry else f'τ={math.exp(entry["log_threshold"]):.7g}'
            lines.append(f'|{p["name"]}_s{int(p["target"]*100)}|{kind}|{value}|{pct(p["measured"][kind])}|{len(p["trace"])}|')
    lines += ['', 'Actual per-call BLASST lambda ranges and complete thresholds are in `thresholds.csv`; matrix hashes/seeds are in `projection_matrices.json`.', '',
        '## Per-task scores (%)', '', '|Task|'+'|'.join(labels)+'|', '|---|'+'|'.join(['---:']*len(labels))+'|']
    taskmap = {(r['condition'],r['task']):r for r in tasks}
    for task in setup['tasks']:
        lines.append('|'+task+'|'+'|'.join(pct(taskmap[label,task]['accuracy']) if (label,task) in taskmap else 'missing' for label in labels)+'|')
    lines += ['', '## Shared-state operator error', '',
        '390 snapshots: one prespecified calibration question per task, all30 layers, step0; one sampled head/query tile per layer. '
        'This expands v14’s single-CWE-prompt diagnostic coverage. No later-step or sparse-trajectory state coverage is claimed. '
        'Full-dimensional relative error is sqrt(sum squared sparse−dense output error / sum squared dense output norm), with every method evaluated on the same QKV snapshots.', '',
        '|Method|Target %|Shared-state physical %|Relative output error|Retained mass %|', '|---|---:|---:|---:|---:|']
    for r in operators:
        if r['axis'] == 'overall': lines.append(f'|{r["name"]}|{pct(r["target"])}|{pct(r["physical_sparsity"])}|{r["relative_output_error"]:.5f}|{pct(r["retained_mass"])}|')
    lines += ['', '## Important limitations', '',
        'All10 dense VT answers hit the official30-token budget before emitting variable names, producing0 VT credit. '
        'Some sparse outputs fit answers inside that budget. Dense-relative gains can therefore include formatting/budget effects, not better reasoning. '
        'Budgets are deliberately unchanged for this controlled comparison. `non_vt_sensitivity.csv` is a disclosed post-hoc analysis, not a replacement benchmark.', '',
        'Projection rank is the experimental change, but independently calibrated thresholds and adaptive retained histories also differ. '
        'Higher accuracy as distortion falls would support projection information loss as an explanation, not prove it is the sole cause. '
        'One generation seed is tested; the three projection seeds in common-support diagnostics are not generation replications.', '',
        'No routing/operator/kernel optimization was introduced. QK, block softmax and projected PV are still computed; physical deletion does not avoid all tile work. '
        'Projection/storage/refresh accounting is saved separately. No hardware speedup or FlashAttention comparison is claimed.', '',
        f'Missing outputs/configurations: {len(audit["missing"])}; audit violations: {len(audit["violations"])}. '
        'Failures, if any, remain in `failures.jsonl`; independently runnable configurations continue.', '']
    (root/'report.md').write_text('\n'.join(lines))


def rank_plot(root, rows, diagnostic_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for target, color in ((.5,'tab:blue'),(.75,'tab:orange')):
        values = {int(r['name'].split('_r')[-1]):r for r in rows if r['name'].startswith('jl_gaussian_r') and r['target']==target}
        ranks = sorted(values)
        axes[0].plot(ranks,[100*values[r]['accuracy'] for r in ranks],'o-',color=color,label=f'{100*target:g}% target')
        full = next((r for r in rows if r['name']=='full_centered' and r['target']==target),None)
        if full: axes[0].axhline(100*full['accuracy'],color=color,linestyle='--',alpha=.6)
        for seed in study.SEEDS:
            ds = sorted((r for r in diagnostic_rows if r['axis']=='overall' and r['target']==target and r['seed']==seed),key=lambda r:r['rank'])
            for ax,metric in zip(axes[1:],('rms_relative_norm_error','tile_disagreement_rate')):
                ax.plot([r['rank'] for r in ds],[r[metric] for r in ds],'o-',color=color,alpha=1 if seed==1729 else .3)
    for ax,title in zip(axes,('RULER score (%) / dashed full control','Shared-support RMS relative norm error','Common-threshold tile disagreement')):
        ax.set(xlabel='Gaussian rank',ylabel=title,xscale='log'); ax.set_xticks(study.DIAGNOSTIC_RANKS,[str(r) for r in study.DIAGNOSTIC_RANKS]); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); path=root/'figures'/'rank_distortion.png';fig.savefig(path,dpi=140);plt.close(fig)
    return path


def regenerate(root):
    contract = study.execution(root)
    ds, sources, violations = diagnostics.summarize(root, contract)
    with patch.object(engine, 'write_report', write_report): audit = engine.regenerate(root)
    audit['violations'].extend(violations); audit['complete'] = audit['complete'] and not violations
    audit['sources'].update(sources)
    rows, raw = base.read(root/'summary.json'), base.read(root/'per_sample.json')
    engine.export(root, 'common_support', ds)
    engine.export(root, 'non_vt_sensitivity', sensitivity(raw))
    figure = rank_plot(root, rows, ds)
    lines = ['', '## Projection information-loss test', '',
        'On all390 shared snapshots, replay the frozen full-dimensional control support identically for Gaussian1/2/4/8/16/32. '
        'Project token values before block PV. Compare centered norm estimates against full-dimensional updates with identical retained history and threshold units. '
        'Seeds1729,2718,31415 are all reported; no direction/seed is selected using outcomes. Counterfactual projected votes do not update the common support.', '',
        '|Rank|Projection seed|Target %|Mean norm ratio|RMS relative norm error|>2× underestimate %|Threshold-dangerous %|Tile disagreement %|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ds:
        if r['axis']=='overall':
            lines.append(f'|{r["rank"]}|{r["seed"]}|{100*r["target"]:.0f}|{r["mean_norm_ratio"]:.4f}|{r["rms_relative_norm_error"]:.4f}|{100*r["factor_two_underestimate_rate"]:.2f}|{100*r["dangerous_underestimate_rate"]:.2f}|{100*r["tile_disagreement_rate"]:.2f}|')
    lines += ['', 'Norm metrics pool valid supported row/block pairs with nonzero true updates. Dangerous rate divides projected-below/full-above-threshold rows by full-above-threshold rows. '
        'Tile disagreement is counterfactual projected vs full decisions at the same full threshold/support, divided by eligible physical tiles; it is not the deployed, separately calibrated mask disagreement. '
        'Full type/cancellation and prefix/canvas/boundary breakdowns are retained in raw diagnostics and `common_support.json`.', '',
        '![Rank, accuracy and distortion](figures/rank_distortion.png)', '',
        '### Paired comparison with the full-dimensional control', '',
        '|Gaussian rank|Target %|Δ full score pp|95% paired CI pp|Actual sparsity gap pp (overall/global/local)|', '|---:|---:|---:|---|---|']
    groups = defaultdict(list)
    for r in raw: groups[r['condition']].append(r)
    for target in base.TARGETS:
        full_label = f'full_centered_s{int(target*100)}'; ref={r['id']:r for r in groups[full_label]}
        full=next((r for r in rows if r['condition']==full_label),None)
        for rank in study.DIAGNOSTIC_RANKS:
            label=f'jl_gaussian_r{rank}_s{int(target*100)}'; group=[r for r in groups[label] if r['id'] in ref]
            current=next((r for r in rows if r['condition']==label),None)
            if not group or current is None or full is None: continue
            ci=engine.bootstrap(group,ref); delta=np.mean([r['accuracy']-ref[r['id']]['accuracy'] for r in group])
            gaps=[100*(current[f'{k}_physical_sparsity']-full[f'{k}_physical_sparsity']) for k in ('overall','global','local')]
            lines.append(f'|{rank}|{100*target:.0f}|{100*delta:.2f}|[{100*ci[0]:.2f}, {100*ci[1]:.2f}]|'+', '.join(f'{x:+.2f}' for x in gaps)+'|')
    lines += ['', f'Common-support diagnostic audit violations: {len(violations)}. Overall complete audit: {audit["complete"]}.', '']
    with (root/'report.md').open('a') as stream: stream.write('\n'.join(lines))
    for p in [root/'report.md',figure,*[root/(n+ext) for n in ('common_support','non_vt_sensitivity') for ext in ('.csv','.json')]]:
        audit['artifacts'][str(p.relative_to(root))]=base.sha(p.read_bytes())
    base._write(root/'audit.json',audit)
    return audit
