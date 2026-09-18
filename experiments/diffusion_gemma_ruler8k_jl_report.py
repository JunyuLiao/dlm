"""Raw-shard-only RULER comparison and independent final audit."""
from collections import Counter, defaultdict
import csv
import gzip
import io
import itertools
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments import diffusion_gemma_ruler8k_jl as study
from experiments.diffusion_gemma_value_aware.report import summary as summarize
from experiments.diffusion_gemma_value_aware.report_metrics import accumulate_marginals, SUM_FIELDS, aggregate
from experiments.diffusion_gemma_jl_output_aware.report import diagnostic_summary
from experiments.diffusion_gemma_jl_output_aware import report as common
from experiments.diffusion_gemma_jl_output_aware.projections import Projections


def export(root, name, rows):
    study._write(root/(name+'.json'), rows)
    keys = list(dict.fromkeys(k for r in rows for k in r)); buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    (root/(name+'.csv')).write_text(buf.getvalue())


def bootstrap(group, other=None):
    """Paired, task-stratified prompt bootstrap; preserve equal task weights."""
    by_task = defaultdict(list)
    for row in group:
        ref = row['dense_accuracy'] if other is None else other[row['id']]['accuracy']
        by_task[row['task']].append(row['accuracy']-ref)
    rng = np.random.default_rng(42); draws = np.zeros(10000)
    for values in by_task.values():
        values = np.array(values); draws += values[rng.integers(0, len(values), (10000, len(values)))].mean(-1)/len(by_task)
    return np.quantile(draws, [.025, .975]).tolist()


def grouped_summary(raw, conditions, predictions):
    rows, tasks, groups = [], [], {}
    for label in study.CONDITIONS:
        group = [r for r in raw if r['condition'] == label]
        if not group: continue
        c = conditions[label]; groups[label] = group
        per_task, _ = study.official.score_predictions(predictions[label], study.RULER)
        out = dict(condition=label, name=c['name'], target=c['target'], **summarize(group))
        # The official metric rounds after averaging within task, not per row.
        out.update(accuracy=float(np.mean(list(per_task.values()))), per_task_official_scores=per_task,
            paired_ci95=bootstrap(group), paired_ci_convention='10000 paired task-stratified prompt bootstrap draws, seed42; individual official scores rounded to1e-4')
        dense_group = [r for r in raw if r['condition'] == 'dense' and r['id'] in {x['id'] for x in group}]
        dense_predictions = [p for p in predictions['dense'] if p['id'] in {x['id'] for x in group}]
        dense_tasks, _ = study.official.score_predictions(dense_predictions, study.RULER)
        out['dense_accuracy'] = float(np.mean(list(dense_tasks.values())))
        out['delta'] = out['accuracy']-out['dense_accuracy']
        out['relative_accuracy'] = out['accuracy']/out['dense_accuracy'] if out['dense_accuracy'] else None
        rows.append(out)
        for task in study.official.PAPER_TASKS:
            subset = [r for r in group if r['task'] == task]
            if not subset: continue
            t = dict(condition=label, task=task, target=c['target'], **summarize(subset))
            t.update(accuracy=per_task[task], dense_accuracy=dense_tasks[task], delta=per_task[task]-dense_tasks[task], paired_ci95=bootstrap(subset))
            tasks.append(t)
    return rows, tasks, groups


def audit_lambda(out, policy):
    records = out.get('effective_lambda', [])
    if len(records) != out['finite_calls']: raise ValueError('Missing per-call BLASST lambda observations')
    if {r['layer'] for r in records} != set(range(30)): raise ValueError('Incomplete lambda layer coverage')
    for r in records:
        k = r['attention_type']; p = policy[k]
        if k != ('global' if r['layer'] % 6 == 5 else 'local'): raise ValueError('Wrong lambda attention type')
        logt = p['log_scale']-math.log(r['valid_kv_length']) if 'log_scale' in p else p['log_threshold']
        if p.get('cap_one'): logt = min(0., logt)
        if p.get('unattainable') and p.get('cap_one'): logt = 0.
        if logt != r['log_lambda'] or math.exp(logt) != r['lambda_value']: raise ValueError('Effective lambda differs from policy')
        if p.get('cap_one') and r['lambda_value'] > 1: raise ValueError('Cap-one violated')
    return records


def plots(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = root/'figures'; folder.mkdir(exist_ok=True); names = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in ['dense', *study.CONFIGS]:
        group = sorted((r for r in rows if r['name'] == name), key=lambda r:r['target'])
        for ax, metric, title in zip(axes, ('accuracy','overall_mass','token_agreement'), ('Official RULER score (%)','Retained dense mass (%)','Token agreement (%)')):
            ax.plot([100*r['overall_physical_sparsity'] for r in group], [100*r[metric] for r in group], 'o-', label=name)
            ax.set(xlabel='Measured physical tile sparsity (%)', ylabel=title); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7); fig.tight_layout()
    path = folder/'tradeoffs.png'; fig.savefig(path, dpi=140); plt.close(fig); names.append(str(path.relative_to(root)))
    group = [r for r in rows if r['name'] != 'dense']; fig, ax = plt.subplots(figsize=(11, 4))
    for j, kind in enumerate(('overall','global','local')):
        ax.bar(np.arange(len(group))+(j-1)*.25, [100*r[f'{kind}_physical_sparsity'] for r in group], width=.25, label=kind)
    ax.scatter(range(len(group)), [100*r['target'] for r in group], marker='_', c='black', label='target', zorder=4)
    ax.set_xticks(range(len(group)), [r['condition'] for r in group], rotation=35, ha='right')
    ax.set(ylabel='Count-weighted physical sparsity (%)', ylim=(0, 100)); ax.legend(); fig.tight_layout()
    path = folder/'target_actual.png'; fig.savefig(path, dpi=140); plt.close(fig); names.append(str(path.relative_to(root)))
    return names


def write_report(root, setup, rows, tasks, policies, thresholds, comparisons, audit, diagnostics):
    pct = lambda x:f'{100*x:.2f}'
    lines = ['# DiffusionGemma RULER8K: BLASST, mass-only and directional routing', '',
        f'Audited final generations: {audit["completed"]}/{study.EXPECTED}. Complete: {audit["complete"]}.', '',
        '## Setup', '',
        '130 evaluation questions:10 per task across the13 tasks in [SparseD, Table4](https://arxiv.org/pdf/2509.24014). '
        'Disjoint calibration26 (2/task), development13 (1/task). Cached650-example pool; deterministic SHA256 selection, split/generation seed42. '
        'The pool has been used previously: this is not fresh held-out confirmation. No final scores or sparsity select thresholds.', '',
        'Pinned DiffusionGemma revision '+setup['revision']+'; BF16,128-query×64-KV tiles, prefix+canvas skippable, local window1024. '
        'Native256-token canvas and at most48 denoising steps; thinking disabled. Requested temperature0 is the existing sentinel for the native0.8→0.4 schedule, NOT greedy. '
        'Official per-task output budgets are retained;8K denotes the RULER generator total budget, not exactly8192 post-chat input tokens. '
        'Dense is computed once per sample and shared across comparisons. RULER uses its pinned official scorer, not NeMo reasoning grading.', '',
        '| Task | Final / calibration | Output budget | Final prompt tokens (min–max) |',
        '|---|---:|---:|---:|']
    for task in study.official.PAPER_TASKS:
        rs = [r for r in setup['final'] if r['task'] == task]
        lines.append(f'|{task}|10 /2|{rs[0]["generation_budget"]}|{min(r["prompt_token_count"] for r in rs)}–{max(r["prompt_token_count"] for r in rs)}|')
    lines += ['', '## Operators and calibration', '',
        'All numerical kernels/routing rules are reused unchanged. Conservative physical deletion requires every valid query row in a128×64 tile to vote skip. '
        'First-support tiles and threshold ties are retained; structural masks and GQA remain native; retained attention is ordinarily renormalized without compensation.', '',
        'BLASST compares each block maximum with the running maximum over all previously seen blocks. '
        'The existing calibration fits log(λL)=log(α)+γs, applies its calibration-only monotonic correction, then verifies complete sparse trajectories. '
        'Effective λ=exp(log_scale)/actual valid KV length L, separately local/global. '
        'Both λ are capped at1 unless the joint constant-λ=1 calibration run misses the requested whole-model target. '
        'If a single layer type has a lower ceiling but the whole-model target is attainable, type-specific calibration goals redistribute the budget without unlocking λ>1.', '',
        'Mass-only is the previous max-logit×valid-token-count mass-bound method, not exact block mass. '
        'Full-dimensional and Gaussian2 centered use attention-weighted means, the retained running output, and valid-KV RMS normalization. '
        'Gaussian2 uses frozen per-layer/native-KV-head matrices, seed1729, FP32 projection/routing; original V forms model outputs. '
        'Dense calibration-state CDF proposals and the existing independent local/global rank/scalar refinement are verified within2pp, up to16 joint points. '
        'All policies freeze before sparse final evaluation.', '',
        '| Method / target | Local threshold | Global threshold | Calibration whole/global/local (%) | λ>1 allowed |',
        '|---|---|---|---|---|']
    for p in policies:
        display = lambda k: ('log_scale=' if 'log_scale' in p['policy'][k] else 'log τ=')+f'{p["policy"][k].get("log_scale",p["policy"][k].get("log_threshold")):.7g}'
        lines.append(f'|{p["name"]} /{int(p["target"]*100)}%|{display("local")}|{display("global")}|'+
            '/'.join(pct(p['measured'][k]) for k in ('overall','global','local'))+f'|{p["aggressive_allowed"]}|')
    if (root/'blasst_cap_one.json').exists():
        ceiling = study.read(root/'blasst_cap_one.json')
        lines += ['', 'Joint λlocal=λglobal=1 calibration physical sparsity (whole/global/local): '+
            '/'.join(pct(ceiling['achieved'][k])+'%' for k in ('overall','global','local'))+'.']
    for t in thresholds:
        if t['name'] == 'blasst': lines.append(f'- {t["condition"]} {t["attention_type"]}: observed λ {t["lambda_min"]:.6g}–{t["lambda_max"]:.6g}; {t["calls_above_one"]}/{t["calls"]} calls above1.')
    lines += ['', '## Per-task official scores (%)', '', '| Task |'+'|'.join(r['condition'] for r in rows)+'|', '|---|'+'---:|'*len(rows)]
    for task in study.official.PAPER_TASKS:
        values = {t['condition']: t['accuracy'] for t in tasks if t['task'] == task}
        lines.append('|'+task+'|'+'|'.join(pct(values[r['condition']]) if r['condition'] in values else 'missing' for r in rows)+'|')
    lines += ['', '## Equal-task macro and physical sparsity', '',
        '| Condition | n | Score | Δ dense (pp) | Paired95%CI (pp) | Whole | Global | Local | Mass | Agreement |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["condition"]}|{r["count"]}|{pct(r["accuracy"])}|{pct(r["delta"])}|'+
            ' to '.join(pct(x) for x in r['paired_ci95'])+'|'+'|'.join(pct(r[k]) for k in ('overall_physical_sparsity','global_physical_sparsity','local_physical_sparsity','overall_mass','token_agreement'))+'|')
    lines += ['', 'Sparsity is Σ skipped eligible physical tiles / Σ eligible physical tiles across samples/layers/heads/steps, never an unweighted average. '
        'RULER scores may give fractional credit for multiple answers. Official per-task scores are computed jointly before equal-task macro averaging. '
        'CIs use a paired task-stratified prompt bootstrap; small differences may remain inconclusive.', '',
        'Dense attention mass and operator error use dense attention on each sparse call’s own QKV state; they do not reuse incompatible dense-generation states after trajectories diverge. '
        'A separate shared-QKV diagnostic uses one calibration source per layer at step0 and sampled later steps on layers0/5. '
        'Operator error aggregates sqrt(Σ||Os−Od||²/Σ||Od||²). Token agreement compares equal positions even after first divergence; missing/extra positions disagree.', '',
        '## Actual-sparsity comparisons', '',
        '| Candidate | Reference | Score Δ (pp) | Paired95%CI | Whole sparsity gap (pp) | All types within3pp |',
        '|---|---|---:|---|---:|---|']
    for c in comparisons:
        lines.append(f'|{c["candidate"]}|{c["reference"]}|{pct(c["delta"])}|'+ ' to '.join(pct(x) for x in c['paired_ci95'])+f'|{pct(c["sparsity_gap"])}|{c["matched_all_types"]}|')
    lines += ['', 'Do not infer an algorithmic win from matching target labels alone; compare achieved whole/global/local budgets and paired uncertainty. '
        'Full-dimensional routing is an expensive reference, not a deployable winner. QK, block softmax, projections, diagnostic PV and original-V attention still incur work; '
        'reported physical deletion is not a measured hardware speedup. No comparison with FlashAttention latency is made.', '',
        '## Failures and audit', '', f'Missing entries: {len(audit["missing"])}. Violations: {len(audit["violations"])}. '
        'Full details are preserved in audit.json and failures.jsonl, including unsuccessful calibration attempts. '
        'A completed claim requires1170 hash-checked outputs, complete shared diagnostics, matching settings, valid threshold provenance and independent raw-only report regeneration.', '',
        '![Accuracy, mass and agreement versus measured sparsity](figures/tradeoffs.png)', '',
        '![Target versus achieved sparsity](figures/target_actual.png)', '']
    (root/'report.md').write_text('\n'.join(lines))


def regenerate(root):
    setup, contract = study.prepare(root), study.execution(root)
    study.evidence.check_sources(contract['sources'])
    # Recheck raw smoke generations with adapter=None: inference is prohibited.
    proof = study.smoke(None, root, setup, contract)
    conditions, policies, missing, violations = {}, [], [], []
    sources = dict(contract['sources']); sources.update(proof['sources'])
    for label in study.CONDITIONS:
        path = root/'final_configs'/f'{label}.json'
        if not path.exists(): missing.append(dict(condition=label, error='No frozen configuration')); continue
        c = study.read(path); study.evidence.check_sources(c['sources']); sources.update(c['sources'])
        name = 'dense' if label == 'dense' else label.rsplit('_s', 1)[0]
        expected = {} if name == 'dense' else study.CONFIGS[name]
        if c['config'] != expected or c['name'] != name or c['fingerprint'] != contract['fingerprint']: raise ValueError('Final configuration changed')
        if name != 'dense':
            p = study.read(root/'policies'/study.BENCHMARK/f'{label}.json'); study.audit_policy(root, p, setup, contract)
            if c['target'] != p['target'] or c['thresholds'] != {study.BENCHMARK:p['policy']}: raise ValueError('Final policy differs')
            policies.append(p)
        elif c['target'] != 0. or c['thresholds'] != {study.BENCHMARK:None}: raise ValueError('Dense configuration changed')
        conditions[label] = c; sources[str(path)] = study.sha(path.read_bytes())
    raw, predictions, lambdas, marginals, matrices, work = [], defaultdict(list), defaultdict(list), {}, {}, []
    bank = Projections()
    bucket_file = (root/'per_layer_head_step.csv.gz').open('wb')
    buckets = io.TextIOWrapper(gzip.GzipFile(fileobj=bucket_file, mode='wb', mtime=0), encoding='utf-8', newline='')
    bucket_writer = None
    for row in setup['final']:
        dense_path = study.shard_path(root, 'dense', 'dense', row['id'])
        if not dense_path.exists(): missing.append(dict(id=row['id'], condition='dense')); continue
        dense = study.runner.cached(None, root, row, 'dense', 'dense', 'dense', {}, None, contract)
        for label, c in conditions.items():
            stage = 'dense' if label == 'dense' else 'final'; path = study.shard_path(root, stage, label, row['id'])
            if not path.exists(): missing.append(dict(id=row['id'], condition=label)); continue
            try:
                out = study.runner.cached(None, root, row, stage, label, c['name'], c['config'], c['thresholds'][study.BENCHMARK], contract)
                with patch.object(common, 'PROJECTED', study.PROJECTED): common.audit_matrices(out, c['name'], c['config'], bank, matrices)
                pair = study.evidence.pair(row, out, dense)
                if c['name'] == 'blasst': lambdas[label].extend(audit_lambda(out, c['thresholds'][study.BENCHMARK]))
                raw.append(dict(condition=label, prediction=out['prediction'], completion_tokens=out['completion_tokens'],
                    prompt_token_count=row['prompt_token_count'], **pair)); predictions[label].append(dict(row, prediction=out['prediction']))
                sources[str(path)] = study.sha(path.read_bytes()); sources.update({out['records_source']['path']: out['records_source']['sha256']})
                records = [r for r in out['records'] if r['probe'] == 'execution']; accumulate_marginals(marginals, records, study.BENCHMARK, label)
                for r in records:
                    record = dict(id=row['id'], condition=label, **r)
                    if bucket_writer is None:
                        bucket_writer = csv.DictWriter(buckets, fieldnames=list(record), extrasaction='ignore'); bucket_writer.writeheader()
                    bucket_writer.writerow(record)
                for key, values in out.get('projection_matrices', {}).items():
                    if key in matrices and matrices[key] != values: raise ValueError('Projection matrix changed across generations')
                    matrices[key] = values
                if out.get('work_accounting'): work.append(dict(id=row['id'], condition=label, **out['work_accounting']))
            except Exception as exc: violations.append(dict(id=row['id'], condition=label, error=repr(exc)))
    buckets.close(); bucket_file.close()
    rows, tasks, groups = grouped_summary(raw, conditions, predictions)
    comparisons = []
    for a, b in itertools.combinations(rows, 2):
        if a['target'] != b['target']: continue
        ga, gb = groups[a['condition']], {r['id']:r for r in groups[b['condition']]}
        paired = [r for r in ga if r['id'] in gb]
        gaps = {k: abs(a[f'{k}_physical_sparsity']-b[f'{k}_physical_sparsity']) for k in ('overall','global','local')}
        comparisons.append(dict(candidate=a['condition'], reference=b['condition'], count=len(paired),
            delta=float(np.mean([r['accuracy']-gb[r['id']]['accuracy'] for r in paired])), paired_ci95=bootstrap(paired, gb),
            sparsity_gap=gaps['overall'], sparsity_gaps=gaps, matched_all_types=all(x <= .03 for x in gaps.values())))
    thresholds = []
    for p in policies:
        label = f'{p["name"]}_s{int(p["target"]*100)}'
        for kind in study.KINDS:
            row = dict(condition=label, name=p['name'], target=p['target'], attention_type=kind,
                **p['policy'][kind], calibration_goal=p['goals'][kind], aggressive_allowed=p['aggressive_allowed'])
            if p['name'] == 'blasst':
                calls = [r for r in lambdas[label] if r['attention_type'] == kind]
                row.update(calls=len(calls), calls_above_one=sum(r['lambda_value'] > 1 for r in calls),
                    lambda_min=min((r['lambda_value'] for r in calls), default=0.), lambda_max=max((r['lambda_value'] for r in calls), default=0.))
            else: row['tau'] = math.exp(row['log_threshold'])
            thresholds.append(row)
    diagnostic_rows, diagnostic_sources = diagnostic_summary(root); sources.update(diagnostic_sources)
    if (root/'shared_state_index.json').exists():
        states = study.read(root/'shared_state_index.json'); expected_diagnostics = len(study.shared_analysis.selected_sources(states))*len(policies)
        if any(s['split'] != 'calibration' or s['id'] not in {r['id'] for r in setup['calibration']} for s in states):
            raise ValueError('Shared diagnostic states contaminated by final data')
        index = study.read(root/'shared_diagnostics_index.json') if (root/'shared_diagnostics_index.json').exists() else []
        n = len(index)
        expected_ids = {(s['path'], p['name'], p['target']) for s in study.shared_analysis.selected_sources(states).values() for p in policies}
        actual_ids = set()
        for entry in index:
            d = study.read(Path(entry['path']))['identity']
            if d['fingerprint'] != contract['fingerprint']: raise ValueError('Diagnostic fingerprint changed')
            actual_ids.add((d['source']['path'], d['name'], d['target']))
        if actual_ids != expected_ids: violations.append(dict(error='Shared diagnostic method/source coverage differs'))
        if not n or n != expected_diagnostics: violations.append(dict(error='Incomplete shared diagnostics', expected=expected_diagnostics, completed=n))
    else: violations.append(dict(error='Missing shared states'))
    # Every final condition must have the complete balanced cohort, not only1170 arbitrary entries.
    for label in study.CONDITIONS:
        if Counter(r['task'] for r in groups.get(label, [])) != dict.fromkeys(study.official.PAPER_TASKS, 10):
            missing.append(dict(condition=label, error='Incomplete balanced130-question cohort'))
    export(root, 'per_sample', raw); export(root, 'summary', rows); export(root, 'per_task', tasks)
    export(root, 'comparisons', comparisons); export(root, 'thresholds', thresholds); export(root, 'shared_operator', diagnostic_rows)
    export(root, 'work_accounting', work); study._write(root/'projection_matrices.json', matrices)
    marginal_rows = [dict(benchmark=b, condition=c, axis=axis, index=i, attention_type=k, **aggregate([dict(zip(SUM_FIELDS,v))]))
        for (b,c,axis,i,k), v in sorted(marginals.items())]
    export(root, 'routing_marginals', marginal_rows)
    figures = plots(root, rows)
    audit = dict(complete=len(raw) == study.EXPECTED and not missing and not violations, completed=len(raw), expected=study.EXPECTED,
        missing=missing, violations=violations, fingerprint=contract['fingerprint'], sources=sources, inference_performed=False)
    write_report(root, setup, rows, tasks, policies, thresholds, comparisons, audit, diagnostic_rows)
    files = [root/(n+ext) for n in ('per_sample','summary','per_task','comparisons','thresholds','shared_operator','work_accounting','routing_marginals') for ext in ('.json','.csv')]
    files += [root/'projection_matrices.json', root/'per_layer_head_step.csv.gz', root/'report.md', *[root/p for p in figures]]
    audit['artifacts'] = {str(p.relative_to(root)):study.sha(p.read_bytes()) for p in files}
    study._write(root/'audit.json', audit); return audit
