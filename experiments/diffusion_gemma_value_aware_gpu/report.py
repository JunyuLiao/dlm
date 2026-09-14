"""Raw-only audited reports, reusing benchmark scorers and metric aggregators."""
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
from experiments.diffusion_gemma_value_aware.report import summary, csv_write, table
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate, accumulate_marginals, SUM_FIELDS
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_value_aware_followup.evidence import pair
from experiments.diffusion_gemma_value_aware.run import shard_path
from .protocol import ROOT, prepare, execution, sha
from .runner import cached


def summarize(group):
    out = summary(group)
    # Official v2 accuracy is micro; report equal-domain macro separately since
    # the new17/17/16 domain quotas are not exactly equal.
    out['equal_task_macro'] = out['accuracy']
    tasks = defaultdict(list)
    for row in group:
        tasks[row['task']].append(row['dense_accuracy'])
    out['dense_equal_task_macro'] = float(np.mean([np.mean(v) for v in tasks.values()]))
    out['accuracy'] = float(np.mean([r['accuracy'] for r in group]))
    out['relative_accuracy'] = out['accuracy']/out['dense_accuracy'] if out['dense_accuracy'] else None
    out.update(correct=sum(r['accuracy'] for r in group), unparsed_answers=sum(r['unparsed_answer'] for r in group))
    return out


def plots(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = []
    for benchmark in ('aime26', 'longbench_v2'):
        selected = [r for r in rows if r['benchmark'] == benchmark and r['split'] == 'full']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for method in sorted({r['name'] for r in selected}):
            group = sorted((r for r in selected if r['name'] == method), key=lambda r:r['target'])
            for ax, metric, label in zip(axes, ('accuracy', 'overall_mass', 'token_agreement'),
                                         ('Accuracy (%)', 'Retained dense mass (%)', 'Token agreement (%)')):
                ax.plot([100*r['overall_physical_sparsity'] for r in group], [100*r[metric] for r in group],
                        marker='o', label=method, linewidth=1)
                ax.set(xlabel='Measured physical tile sparsity (%)', ylabel=label, xlim=(-2, 102))
                ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.suptitle(f'{benchmark}: fixed samples; 50/75% targets; no kernel-speedup claims')
        fig.tight_layout()
        folder = root/'figures'; folder.mkdir(exist_ok=True)
        path = folder/f'{benchmark}_tradeoffs.png'; fig.savefig(path, dpi=150); plt.close(fig)
        names.append(str(path.relative_to(root)))
        group = [r for r in selected if r['name'] != 'dense']
        fig, ax = plt.subplots(figsize=(12, 5))
        for j, kind in enumerate(('overall', 'global', 'local')):
            ax.bar(np.arange(len(group))+(j-1)*.25, [100*r[f'{kind}_physical_sparsity'] for r in group],
                   width=.25, label=kind)
        ax.set_xticks(range(len(group)), [r['condition'] for r in group], rotation=65, ha='right')
        ax.scatter(range(len(group)), [100*r['target'] for r in group], marker='_', color='black', label='target', zorder=5)
        ax.set(ylabel='Count-weighted physical sparsity (%)', ylim=(0, 100), title=benchmark)
        ax.legend(); fig.tight_layout()
        path = folder/f'{benchmark}_target_actual.png'; fig.savefig(path, dpi=150); plt.close(fig)
        names.append(str(path.relative_to(root)))
    return names


def regenerate(root=ROOT):
    setup, contract = prepare(root), execution(root)
    conditions, violations, missing, sources, checked_sources = {}, [], [], {}, {}
    for name in setup['conditions']:
        path = root/'final_configs'/f'{name}.json'
        if not path.exists():
            missing.append(dict(condition=name, error='Policy/validation contract not frozen'))
            continue
        c = json.loads(path.read_text())
        if c['fingerprint'] != contract['fingerprint']:
            raise ValueError('Final condition kernel fingerprint mismatch')
        for source, digest in c['sources'].items():
            if source in checked_sources:
                if checked_sources[source] != digest:
                    raise ValueError('Inconsistent frozen source hashes')
                continue
            if sha(Path(source).read_bytes()) != digest:
                raise ValueError(f'Frozen source changed: {source}')
            checked_sources[source] = digest
        sources[str(path)] = sha(path.read_bytes())
        conditions[name] = c
    raw, dimensions = [], {}
    temp = root/'per_layer_head_step.csv.gz.tmp'
    with temp.open('wb') as binary, gzip.GzipFile(filename='', fileobj=binary, mode='wb', compresslevel=3, mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8', newline='') as stream:
            writer = None
            for row in setup['final']:
                try:
                    dense = cached(None, root, row, 'dense', 'dense', {}, None, contract)
                except Exception as error:
                    missing.append(dict(id=row['id'], condition='dense', error=str(error)))
                    continue
                for name, c in conditions.items():
                    try:
                        out = dense if name == 'dense' else cached(None, root, row, 'final', name, c['config'], c['thresholds'][row['benchmark']], contract)
                        stage = 'dense' if name == 'dense' else 'final'
                        path = shard_path(root, stage, name, row['id'])
                        sources[str(path)] = sha(path.read_bytes())
                        for key in ('imported_dense', 'equivalent_source'):
                            if key in out:
                                source = out[key]
                                if sha(Path(source['path']).read_bytes()) != source['sha256']:
                                    raise ValueError('Reused source changed')
                                sources[source['path']] = source['sha256']
                        raw.append(dict(condition=name, previous_manifest_member=row['previous_manifest_member'],
                            prediction=out['prediction'], completion_tokens=out['completion_tokens'],
                            backend=out['backend'], guard_fallback_calls=out.get('guard_fallback_calls', 0),
                            source=dict(path=str(path), sha256=sources[str(path)]), **pair(row, out, dense)))
                        records = out['records']
                        accumulate_marginals(dimensions, records, row['benchmark'], name)
                        for record in records:
                            item = dict(id=row['id'], benchmark=row['benchmark'], condition=name, **record)
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(item)); writer.writeheader()
                            writer.writerow(item)
                    except FileNotFoundError as error:
                        missing.append(dict(condition=name, id=row['id'], error=str(error)))
                    except Exception as error:
                        violations.append(dict(condition=name, id=row['id'], error=str(error)))
    complete = len(raw) == 1040 and not missing and not violations and len(conditions) == 13
    sources.update(checked_sources)
    audit = dict(complete=False, raw_complete=complete, completed=len(raw), expected=1040,
                 missing=missing, violations=violations, sources=sources, fingerprint=contract['fingerprint'])
    _write(root/'audit.json', audit)
    if not complete:
        return audit
    temp.replace(root/'per_layer_head_step.csv.gz')
    groups, taskgroups = defaultdict(list), defaultdict(list)
    for r in raw:
        groups[r['benchmark'], 'full', r['condition']].append(r)
        if r['benchmark'] == 'aime26':
            groups[r['benchmark'], 'calibration6' if r['calibration'] else 'noncalibration24', r['condition']].append(r)
        else:
            groups[r['benchmark'], 'previous30' if r['previous_manifest_member'] else 'new20', r['condition']].append(r)
        taskgroups[r['benchmark'], r['task'], r['condition']].append(r)
    rows = []
    for (benchmark, split, label), group in sorted(groups.items()):
        c = conditions[label]
        rows.append(dict(benchmark=benchmark, split=split, condition=label,
            name=label.rsplit('_s', 1)[0] if label != 'dense' else 'dense',
            method=c['config'].get('method', 'dense'), pooling=c['config'].get('pooling'),
            target=c['target'], thresholds=c['thresholds'].get(benchmark), **summarize(group)))
    tasks = [dict(benchmark=b, task=t, condition=n, **summarize(g)) for (b,t,n),g in sorted(taskgroups.items())]
    compared = []
    for row in rows:
        if row['name'] in ('dense', 'blasst_original', 'blasst_aggressive'):
            continue
        candidates = [r for r in rows if r['benchmark'] == row['benchmark'] and r['split'] == row['split']
                      and r['name'] in ('blasst_original', 'blasst_aggressive')]
        for refname in ('blasst_original', 'blasst_aggressive'):
            ref = min((r for r in candidates if r['name'] == refname),
                      key=lambda r:abs(r['overall_physical_sparsity']-row['overall_physical_sparsity']))
            a = {r['id']:r['accuracy'] for r in groups[row['benchmark'], row['split'], row['condition']]}
            b = {r['id']:r['accuracy'] for r in groups[ref['benchmark'], ref['split'], ref['condition']]}
            if set(a) != set(b):
                raise ValueError('Unpaired comparison')
            compared.append(dict(benchmark=row['benchmark'], split=row['split'], condition=row['condition'],
                reference=ref['condition'], accuracy_delta_pp=100*(row['accuracy']-ref['accuracy']),
                paired_ci95=paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)]),
                **{f'{k}_sparsity_gap_pp':100*(row[f'{k}_physical_sparsity']-ref[f'{k}_physical_sparsity']) for k in ('overall','global','local')},
                within3pp_actual_budget=abs(row['overall_physical_sparsity']-ref['overall_physical_sparsity']) <= .03))
    _write(root/'per_sample.json', raw); _write(root/'summary.json', rows)
    csv_write(root/'summary.csv', rows); csv_write(root/'per_task.csv', tasks)
    _write(root/'comparisons.json', compared); csv_write(root/'comparisons.csv', compared)
    csv_write(root/'routing_marginals.csv', [dict(benchmark=b, condition=n, axis=a, index=i, attention_type=k,
        **aggregate([dict(zip(SUM_FIELDS, values))])) for (b,n,a,i,k),values in sorted(dimensions.items())])
    display = [dict(benchmark=r['benchmark'], split=r['split'], condition=r['condition'],
        target=100*r['target'], threshold=threshold_text(r), score=f"{r['correct']:g}/{r['count']}",
        accuracy=100*r['accuracy'], dense_delta_pp=100*r['delta'], macro=100*r['equal_task_macro'],
        whole=100*r['overall_physical_sparsity'], global_s=100*r['global_physical_sparsity'],
        local_s=100*r['local_physical_sparsity'], mass=100*r['overall_mass'],
        error=r['overall_relative_error'], agreement=100*r['token_agreement'],
        unparsed=r['unparsed_answers'], length_limited=r['length_terminated']) for r in rows]
    csv_write(root/'main_results.csv', display)
    from .analysis import analyze, narrative
    findings = analyze(rows, raw)
    _write(root/'interpretation.json', findings)
    csv_write(root/'mass_control_comparisons.csv', findings['mass_controls'])
    csv_write(root/'descriptive_correlations.csv', findings['correlations'])
    text = '# Value-aware attention: accelerated engineering rerun,50/75% targets\n\n'
    text += 'Complete:1040/1040 results,13 conditions on identical30 AIME26 +50 LongBench v2 questions. No FlashAttention comparison or system-speedup claim.\n\n'
    text += narrative(rows, findings)
    text += '## Setup and calibration\n\n'
    text += 'BF16 DiffusionGemma revision '+setup['revision']+';128×64 tiles; prefix+canvas skippable; native GQA, scaling and structural masks. Seed42, native canvas256/up to48 denoising steps, thinkingFalse. Requested temperature0 is the native0.4–0.8 schedule sentinel, not greedy. Budgets2048 AIME/128 v2 are unchanged.\n\n'
    text += 'V2:17 Single-Document QA,17 Multi-Document QA,16 Code Repository Understanding;25easy/25hard. Prior30 retained plus20 metadata-selected additions, disjoint from calibration/development. '+str(sum(r['truncated'] for r in setup['final'] if r['benchmark']=='longbench_v2'))+'/50 prompts use the original32k head/tail cap. These are multiple-choice domains, not generative summarization/code completion. AIME full30 includes the fixed6 calibration problems; noncalibration24 is shown separately. Prior exposure makes this exploratory.\n\n'
    text += 'The50% policies are reused unchanged. The75% policies reuse verified compatible AIME policies where available; otherwise the existing calibration-only inverse-valid-length BLASST fit/search or scalar-risk empirical-CDF refinement, with at most3 new verification rounds, is used on the same6 calibration examples per benchmark. Original BLASST caps lambda at1 and uses exact constant1 on verified unattainable types; aggressive BLASST allows>1 with the preceding-seen-maximum rule. New risks use retained pre-block state. Value pooling=vector_mean; output-risk pooling=mean. All policy files/traces and source hashes are preserved. No final scores choose thresholds.\n\n'
    text += '## Measurement and engineering\n\n'
    text += 'Whole/global/local physical sparsity is summed skipped eligible tiles divided by summed eligible tiles over decoder denoising attention, not an average of sample percentages; dense prefill is excluded. Retained mass and relative error compare exact and approximate attention on the same Q/K/V of each run. Error=sqrt(sum error²/sum dense-output²). Positional token agreement includes every position after divergence; extra/missing tokens disagree, EOS included. V2 main accuracy is the official micro score; equal-domain macro is separate.\n\n'
    text += 'Custom Triton kernels fuse block statistics and sequential all-row routing. Native QK/logit rounding, whole-query FP32-softmax-to-BF16 and BF16 PV remain unchanged. The original FP32 diagnostic operators/reduction order are reused exactly; an experimental whole-PV regrouping failed long-context checks and was discarded. Mass/risk kernels reuse the original diagnostic block log-sums; a2e-5 reference guard protects near-threshold decisions. Exact max-based ties remain strict-less-than. Counts accumulate on GPU until generation ends. CUDA synthetic and real-model mask/output/metric proofs are saved under validation/. All sparse results use this backend; identical accelerated operators may share an explicitly audited alias. Dense native caches are reused only with matching prompts/settings and parity validation.\n\n'
    text += 'Engineering references: [Triton external libdevice functions](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html), [Triton fused-reduction example](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html). These are implementation references, not speedup evidence.\n\n'
    cols = ('condition','target','threshold','score','accuracy','dense_delta_pp','whole','global_s','local_s','mass','error','agreement','unparsed','length_limited')
    for benchmark, split in (('aime26','full'),('aime26','noncalibration24'),('longbench_v2','full')):
        text += f'## {benchmark}: {split}\n\n'+table([r for r in display if r['benchmark']==benchmark and r['split']==split], cols)+'\n\n'
    text += '## V2 per-domain results before aggregate\n\n'
    text += table([dict(task=r['task'],condition=r['condition'],score=f"{r['correct']:g}/{r['count']}", accuracy=100*r['accuracy'],unparsed=r['unparsed_answers']) for r in tasks if r['benchmark']=='longbench_v2'], ('task','condition','score','accuracy','unparsed'))+'\n\n'
    text += 'Percent columns are percentages; error is a ratio. Summary files additionally include sample/layer distributions, previous30/new20 v2 strata, full-sequence match, macro accuracy and paired bootstrap intervals. comparisons.csv chooses the nearest actually measured BLASST point for each reference separately and flags unmatched budgets; a point-score difference is not statistical equivalence. Output completion/truncation limitations remain visible rather than changing the scorer or budget after evaluation.\n'
    (root/'report.md').write_text(text)
    names = ['per_sample.json','summary.json','summary.csv','per_task.csv','comparisons.json','comparisons.csv',
             'routing_marginals.csv','main_results.csv','report.md','per_layer_head_step.csv.gz',
             'interpretation.json','mass_control_comparisons.csv','descriptive_correlations.csv']+plots(root, rows)
    audit.update(complete=True, artifacts={n:sha((root/n).read_bytes()) for n in names})
    _write(root/'audit.json', audit)
    return audit


if __name__ == '__main__':
    result = regenerate()
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts')}))
