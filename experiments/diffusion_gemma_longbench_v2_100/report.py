"""CPU-only raw-shard audit, actual NeMo rescoring and paper-facing summaries."""
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_value_aware.report import csv_write,table
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate,accumulate_marginals,SUM_FIELDS,correlation
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware_followup import evidence
from experiments.diffusion_gemma_value_aware_gpu.report import summarize
from .protocol import ROOT,prepare,execution,sha
from .runner import cached
from .calibration import condition
from . import nemo


def pairs(row,outputs):
    """Batch NeMo scoring; reuse existing positional agreement and metric formulas."""
    labels=list(outputs)
    grades=nemo.evaluate([dict(index=row['source_id'],condition=n,generation=outputs[n]['prediction'],expected_answer=row['expected']) for n in labels])
    scores={g['generation']:g for g in grades}
    def score(unused,prediction):return float(scores[prediction]['symbolic_correct'])
    def extract(prediction):
        p=scores[prediction]['predicted_answer'];return p if p in tuple('ABCD') else None
    # evidence.pair strips whitespace only for extraction. Keep both keys so
    # exact original NeMo parsing, not a second bespoke parser, is reused.
    for g in grades:scores.setdefault(g['generation'].strip(),g)
    result=[]
    with patch.object(evidence,'score',score),patch.object(evidence,'official_extractor',lambda:extract):
        for label in labels:result.append(dict(condition=label,**evidence.pair(row,outputs[label],outputs['dense'])))
    return result,grades


def comparisons(rows,groups):
    result=[]
    for row in rows:
        if row['name'] not in ('mass','risk'):continue
        for method in ('blasst_original','blasst_aggressive','mass' if row['name']=='risk' else 'risk'):
            candidates=[r for r in rows if r['name']==method]
            ref=min(candidates,key=lambda r:abs(r['overall_physical_sparsity']-row['overall_physical_sparsity']))
            a={r['id']:r['accuracy'] for r in groups[row['condition']]}
            b={r['id']:r['accuracy'] for r in groups[ref['condition']]}
            if set(a)!=set(b):raise ValueError('Unpaired comparison')
            gaps={k:row[f'{k}_physical_sparsity']-ref[f'{k}_physical_sparsity'] for k in ('overall','global','local')}
            result.append(dict(condition=row['condition'],reference=ref['condition'],accuracy_delta=row['accuracy']-ref['accuracy'],
                paired_ci95=paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)]),
                **{f'{k}_sparsity_gap':v for k,v in gaps.items()},within3pp_overall=abs(gaps['overall'])<=.03,
                within3pp_all_types=all(abs(v)<=.03 for v in gaps.values()),
                interpretation='Nearest observed actual-sparsity point; no interpolation, no multiple-comparison correction'))
    return result


def plots(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for name in ('dense','blasst_original','blasst_aggressive','mass','risk'):
        g=sorted((r for r in rows if r['name']==name),key=lambda r:r['target'])
        for ax,key,label in zip(axes,('accuracy','overall_mass','token_agreement'),('Accuracy (%)','Retained dense mass (%)','Positional token agreement (%)')):
            ax.plot([100*r['overall_physical_sparsity'] for r in g],[100*r[key] for r in g],marker='o',label=name)
            ax.set(xlabel='Measured physical tile sparsity (%)',ylabel=label,xlim=(-2,100));ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout()
    fig.savefig(folder/'tradeoffs.png',dpi=150);plt.close(fig)
    sparse=[r for r in rows if r['name']!='dense']
    fig,ax=plt.subplots(figsize=(11,5))
    for j,k in enumerate(('overall','global','local')):
        ax.bar(np.arange(len(sparse))+(j-1)*.25,[100*r[f'{k}_physical_sparsity'] for r in sparse],width=.25,label=k)
    ax.scatter(np.arange(len(sparse)),[100*r['target'] for r in sparse],color='black',marker='_',label='target',zorder=4)
    ax.set_xticks(range(len(sparse)),[r['condition'] for r in sparse],rotation=45,ha='right')
    ax.set(ylabel='Count-weighted physical tile sparsity (%)',ylim=(0,100));ax.legend();fig.tight_layout()
    fig.savefig(folder/'target_actual.png',dpi=150);plt.close(fig)
    return ['figures/tradeoffs.png','figures/target_actual.png']


def findings(setup,rows,compared):
    """Conservative empirical interpretation, entirely derived from final shards."""
    dense=next(r for r in rows if r['name']=='dense')
    lines=[f"Dense accuracy is {dense['correct']:g}/100 ({100*dense['accuracy']:.1f}%), with {dense['unparsed_answers']} unparsed and {dense['length_terminated']} length-limited outputs."]
    if dense['accuracy']<=.25:
        lines.append('Dense is at or below the25% four-choice chance level on this subset. These data cannot support a strong claim of preserved useful long-context reasoning accuracy merely through dense-relative comparisons.')
    for target in (.5,.75):
        group=[r for r in rows if r['name'] in ('mass','risk') and r['target']==target]
        values=[f"{r['name']}: {100*r['overall_physical_sparsity']:.1f}% actual sparsity (global {100*r['global_physical_sparsity']:.1f}%, local {100*r['local_physical_sparsity']:.1f}%), {r['correct']:g}/100 correct, {100*r['overall_mass']:.1f}% retained mass, {100*r['token_agreement']:.1f}% positional agreement" for r in group]
        lines.append(f"At the{int(target*100)}% target, "+'; '.join(values)+'.')
    for c in compared:
        if not c['condition'].startswith('risk_') and not c['reference'].startswith('blasst_aggressive_'):continue
        low,high=c['paired_ci95']
        detail=f"{c['condition']} versus {c['reference']}: {100*c['accuracy_delta']:+.1f}pp accuracy, paired95% CI [{100*low:+.1f}, {100*high:+.1f}]pp."
        if not c['within3pp_overall']:
            detail+=' No comparable observed overall-sparsity point exists within3pp; this is not an equal-budget win/loss.'
        elif not c['within3pp_all_types']:
            detail+=' Overall sparsity matches within3pp, but local/global allocation does not; allocation remains a confound.'
        elif low<=0<=high:
            detail+=' The interval includes zero; this sample does not resolve an accuracy advantage.'
        elif low>0:
            detail+=' The exploratory interval favors the candidate at a closely matched actual budget; it is not multiple-comparison corrected or a generalization guarantee.'
        else:
            detail+=' The exploratory interval favors the reference at a closely matched actual budget.'
        lines.append(detail)
    sparse=[r for r in rows if r['name']!='dense']
    for metric in ('overall_mass','token_agreement'):
        value=correlation([r[metric] for r in sparse],[r['accuracy'] for r in sparse])
        lines.append(f"Across the8 dependent sparse configurations, Pearson accuracy correlation with {metric} is "+('undefined (no variation)' if value is None else f'{value:.3f}')+'. This is descriptive, not causal evidence.')
    lines.append(f"Paper scope: one model, one generation seed,100 fresh questions, only two targets, and {sum(r['truncated'] for r in setup['final'])}/100 context-truncated inputs. NeMo prompt/scoring standardization fixes the prior short-output protocol issue only to the extent shown by the observed completion statistics; it does not remove context truncation or reproduce the paper\'s autoregressive serving protocol. No hardware-speedup claim is supported or intended.")
    return '\n\n'.join(lines)+'\n'


def narrative(root,setup,rows,tasks,compared,policies,figures):
    display=[dict(condition=r['condition'],target_pct=100*r['target'],threshold=threshold_text(r),
        actual_pct=100*r['overall_physical_sparsity'],global_pct=100*r['global_physical_sparsity'],local_pct=100*r['local_physical_sparsity'],
        retained_mass_pct=100*r['overall_mass'],agreement_pct=100*r['token_agreement'],correct=f"{r['correct']:g}/{r['count']}",
        dense_delta_pp=100*r['delta'],CI95_pp=[round(100*v,2) for v in r['paired_ci95']],
        unparsed=r['unparsed_answers'],length_limited=r['length_terminated']) for r in rows]
    dense=next(r for r in rows if r['name']=='dense')
    quotas=[dict(domain=k,population=setup['selection']['domain_population'][k],sample=v) for k,v in sorted(setup['selection']['domain_quotas'].items())]
    limits=[dict(condition=p['name']+f"_s{int(p['target']*100)}",local=p['measured']['local'],global_=p['measured']['global'],
        within_two_points=p['within_two_points'],lambda1_unattainable=p.get('cap_one_unattainable')) for p in policies]
    text=f'''# LongBench v2: mass-only and output-risk,100 fresh problems

## Empirical conclusions

{findings(setup,rows,compared)}

## Setup and scope

All900 final outputs use exactly the same100 cached prompts, seeds and generation settings. The pinned503-problem population is sampled with largest-remainder quotas across all six domains and within-domain subtasks, then SHA256 ordering (selection seed20260914). Final examples are disjoint from12 calibration,6 development and all previous study IDs. No answer or correctness influenced selection. This is a new held-out sample, not a direct score comparison to the previous50-question subset.

{table(quotas,['domain','population','sample'])}

The frozen manifest records difficulty, length band, source ID, prompt hash and actual token IDs. Context budget: **{setup['input_budget']:,} tokens**; generation budget: **{setup['output_budget']:,} tokens**, not128. Context-only tokenizer-aware head/tail truncation preserves question/options/instructions; **{sum(r['truncated'] for r in setup['final'])}/100 inputs are truncated**. Thus these are context-limited results, not a full-context leaderboard claim. Capacity and completion choices were made using disjoint development examples, never final accuracy. See `pilot/decision.json` and `dataset_audit.json`.

Literal model-reserved control-token spellings in repository/benchmark text are HTML-escaped before native chat formatting, consistently across all conditions; {sum(bool(r.get('literal_special_token_escapes')) for r in setup['final'])}/100 source prompts require this text-only protection. No sampled problem is discarded. Raw dataset and escaped-prompt hashes and replacement counts are retained. Preparation failures are preserved in `prepare_pre_escape_failure.log` and `prepare_shared_tokenizer_failure.log`. Parallel preparation uses isolated processor/tokenizer instances; completed earlier caches require identical renderer syntax and exact isolated token replay before reuse. The original6 pilot prompts require zero replacements and remain byte-identical; `pilot/reuse_audit.json` pins the original code snapshot and proves safe reuse without repeating their GPU inference.

DiffusionGemma revision `{setup['revision']}`, BF16, batch1, seed42, native256-token canvas and at most48 denoising iterations. `thinking=False` is the inherited native chat setting; the NeMo prompt explicitly requests step-by-step reasoning. Temperature0 in this adapter selects the native0.4–0.8 sampling schedule: **this is not greedy decoding**. All conditions use that same schedule. Dense prefill is unchanged.

The actual pinned [NeMo-Skills LongBench v2 prompt and MCQ evaluator](https://github.com/NVIDIA-NeMo/Skills/tree/{nemo.REVISION}/nemo_skills/dataset/longbench-v2) are used with their default boxed-answer/relaxed extraction. `nemo_predictions.jsonl` contains the framework's actual `predicted_answer` and `symbolic_correct` fields. Generation is our native DiffusionGemma backend, not NeMo's autoregressive serving backend or an exact reproduction of the [BLASST paper's decoding protocol](https://arxiv.org/html/2512.12087v1). No answer cleanup, judge-model scoring or post-hoc alternative parser is used.

## Algorithms and thresholds

The previously validated GPU kernels and algorithms are unchanged. Physical tiles are **128 query ×64 KV tokens**. Prefix and canvas are both skippable; mixed boundary tiles are counted separately. A physical tile is deleted only when all its valid query rows vote to skip it; strict thresholds retain ties and initial valid support is retained.

Original and aggressive BLASST both compare a tile's token-level maximum with the running maximum over all preceding tiles. Both use this repository's existing all-valid-query-row physical-tile gate, not a row-granular deployed-kernel reproduction. Original caps λ at1; aggressive permits λ>1 and is a separately labelled heuristic, not valid-range original BLASST. Existing inverse-length calibration is reused: λ=exp(log_scale)/L, with L the attention call's valid KV length and separate local/global scalars. An actually verified unattainable original-BLASST attention type uses constant λ=1 at every final length.

Mass-only uses the existing upper-bound block mass fraction `a=exp(b)*n/(Z_previous+exp(b)*n)`; output-risk uses `a*(1+mean_token_value_norm/reference_value_norm)`. Output-risk here is the existing **mean-norm bound**, not a newly introduced exact output-error oracle. Both compare the maximum valid-row risk with a scalar τ, update normalization state from retained tiles, and share one calibrated τ per attention type. The reference norm/pooling definitions are unchanged in the pinned operators.

Old scalars are warm-start proposals only. All measured calibration points come from the new12 calibration problems. The existing inverse-L BLASST search and empirical-risk-CDF refinement use at most3 new verified points per method/target. No final-set retuning occurs. Missed targets are reported at their actual sparsity; only measured constant1 boundaries justify original-BLASST unattainability claims.

{table(limits,['condition','local','global_','within_two_points','lambda1_unattainable'])}

## Main results

Accuracy is official-style per-problem MCQ correctness (micro average); equal-domain macro and per-subtask scores are also exported. Dense: **{dense['correct']:g}/100**, {dense['unparsed_answers']} unparsed, {dense['length_terminated']} length-limited outputs. Dense-relative deltas use the same prompts and a paired prompt bootstrap95% interval (20,000 draws, seed42).

{table(display,['condition','target_pct','threshold','actual_pct','global_pct','local_pct','retained_mass_pct','agreement_pct','correct','dense_delta_pp','CI95_pp','unparsed','length_limited'])}

## Per-domain accuracy before aggregation

{table([dict(task=r['task'],condition=r['condition'],correct=f"{r['correct']:g}/{r['count']}",accuracy_pct=100*r['accuracy']) for r in tasks],['task','condition','correct','accuracy_pct'])}

## Comparable actual budgets

{table(compared,['condition','reference','overall_sparsity_gap','global_sparsity_gap','local_sparsity_gap','accuracy_delta','paired_ci95','within3pp_overall','within3pp_all_types'])}

Gaps and confidence intervals in this table are fractions. Comparisons choose the nearest observed BLASST point by actual overall sparsity, not nominal target; local/global gaps remain visible. Points outside3 percentage points overall are not presented as equal-budget wins. Confidence intervals are exploratory and not corrected for multiple comparisons.100 prompts and two targets per method do not establish a general scaling law or causality.

## Metric interpretation and correctness

Sparsity is `sum(skipped eligible physical tiles)/sum(eligible physical tiles)`, separately overall/global/local, across all samples, layers, heads, denoising steps and growing sequence lengths. It is never an unweighted average of per-call percentages. Overall refers to decoder denoising attention; dense prefill is outside the routing denominator. All30 layers,16 query heads per layer and both attention types are audited; layers5,11,17,23,29 are global.

Retained mass is computed from exact dense-softmax probabilities at each **corresponding sparse execution state**, before masking, and averaged using valid-query-row counts. It is not an alignment of dense-run and sparse-run attention after their token trajectories diverge. The dense generated sequence is cached once and reused for accuracy/positional agreement. Positions continue to be compared after the first divergence; missing/extra positions disagree; denominator is the longer output length. Sequence exact match and dense-relative output error are separately exported.

`routing_marginals.csv` and `per_layer_head_step.csv.gz` contain raw-count-derived layer/head/step diagnostics, including prefix/canvas/boundary counts, dense probability mass and output error. `summary.json` includes sample p10/p50/p90 distributions; correlations are descriptive, not evidence of a causal mechanism.

An18-case new long-context CUDA smoke checks native dense token parity, all requested50/75 masks against the reference, four independent mass/risk generated-sequence replays, finite outputs, physical counts and unchanged decoding. The frozen parent GPU validation is also source-checked. Final reports are regenerated from completed immutable shards only; see `audit.json`.

No latency, throughput, hardware speedup or FlashAttention comparison is measured or claimed. Kernels are solely an experiment-runtime optimization.

## Figures and reproduction

'''
    for figure in figures:text+=f'![{Path(figure).stem}]({figure})\n\n'
    text+='Run `CUDA_VISIBLE_DEVICES=\'\' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_longbench_v2_100.workflow report` from the repository root. Calibration attempts and failures remain in the bundle; completed shards are never silently replaced.\n'
    (root/'report.md').write_text(text)


def regenerate(root=ROOT):
    setup=prepare(root);contract=execution(root)
    conditions={};sources={};missing=[];violations=[]
    for label in setup['conditions']:
        path=root/'final_configs'/f'{label}.json'
        if not path.exists():missing.append(dict(condition=label,error='No frozen condition'));continue
        c=json.loads(path.read_text())
        if c['fingerprint']!=contract['fingerprint']:raise ValueError('Final fingerprint mismatch')
        sources=evidence.merge_sources(sources,c['sources'],{str(path):sha(path.read_bytes())})
        if label!='dense':
            name=label.rsplit('_s',1)[0]
            audited=condition(root,setup,contract,name,c['target'])
            if any(c[k]!=audited[k] for k in audited):raise ValueError('Frozen policy differs from audited calibration')
        conditions[label]=c
    evidence.check_sources(sources)
    raw=[];graded=[];dimensions={};temp=root/'per_layer_head_step.csv.gz.tmp'
    with temp.open('wb') as binary,gzip.GzipFile(filename='',fileobj=binary,mode='wb',mtime=0,compresslevel=3) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
            writer=None
            for row in setup['final']:
                outputs={}
                for label,c in conditions.items():
                    try:
                        stage='dense' if label=='dense' else 'final'
                        out=cached(None,root,row,stage,label,c['config'],c['thresholds'].get('longbench_v2'),contract)
                        if out.get('equivalent_source'):
                            src=out['equivalent_source'];evidence.check_sources({src['path']:src['sha256']})
                        path=shard_path(root,stage,label,row['id']);sources[str(path)]=sha(path.read_bytes());outputs[label]=out
                    except FileNotFoundError as error:missing.append(dict(condition=label,id=row['id'],error=str(error)))
                    except Exception as error:violations.append(dict(condition=label,id=row['id'],error=str(error)))
                if 'dense' not in outputs:continue
                try:paired,grades=pairs(row,outputs)
                except Exception as error:
                    violations.append(dict(id=row['id'],error=str(error)));continue
                graded.extend(grades)
                for p in paired:
                    label=p['condition'];out=outputs[label]
                    raw.append(dict(**p,sub_domain=row['sub_domain'],difficulty=row['difficulty'],length_band=row['length_band'],
                        input_tokens=len(row['prompt_tokens']),truncated=row['truncated'],prediction=out['prediction'],completion_tokens=out['completion_tokens']))
                    accumulate_marginals(dimensions,out['records'],'longbench_v2',label)
                    for r in out['records']:
                        item=dict(id=row['id'],condition=label,**r)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(item));writer.writeheader()
                        writer.writerow(item)
    complete=len(raw)==900 and len(conditions)==9 and not missing and not violations
    audit=dict(complete=False,raw_complete=complete,completed=len(raw),expected=900,missing=missing,violations=violations,
        fingerprint=contract['fingerprint'],sources=sources)
    _write(root/'audit.json',audit)
    if not complete:return audit
    temp.replace(root/'per_layer_head_step.csv.gz')
    groups=defaultdict(list);taskgroups=defaultdict(list);subgroups=defaultdict(list)
    for r in raw:
        groups[r['condition']].append(r);taskgroups[r['condition'],r['task']].append(r)
        subgroups[r['condition'],r['sub_domain']].append(r)
    rows=[]
    for label in setup['conditions']:
        c=conditions[label]
        rows.append(dict(benchmark='longbench_v2',condition=label,name=label.rsplit('_s',1)[0] if label!='dense' else label,
            target=c['target'],method=c['config'].get('method','dense'),thresholds=c['thresholds'].get('longbench_v2'),**summarize(groups[label])))
    tasks=[dict(condition=n,task=t,**summarize(g)) for (n,t),g in sorted(taskgroups.items())]
    subtasks=[dict(condition=n,sub_domain=t,**summarize(g)) for (n,t),g in sorted(subgroups.items())]
    compared=comparisons(rows,groups)
    policies=[json.loads((root/'verified_policies/longbench_v2'/f'{r["condition"]}.json').read_text()) for r in rows if r['name']!='dense']
    _write(root/'per_sample.json',raw);_write(root/'summary.json',rows);_write(root/'comparisons.json',compared)
    _write(root/'thresholds.json',policies)
    csv_write(root/'summary.csv',rows);csv_write(root/'per_task.csv',tasks);csv_write(root/'per_subtask.csv',subtasks);csv_write(root/'comparisons.csv',compared)
    csv_write(root/'routing_marginals.csv',[dict(condition=n,axis=a,index=i,attention_type=k,**aggregate([dict(zip(SUM_FIELDS,v))]))
        for (b,n,a,i,k),v in sorted(dimensions.items())])
    sparse=[r for r in rows if r['name']!='dense']
    _write(root/'correlations.json',[dict(metric=k,n=len(sparse),pearson=correlation([r[k] for r in sparse],[r['accuracy'] for r in sparse]),
        interpretation='Descriptive dependent configuration points, not causal evidence') for k in ('overall_mass','token_agreement','overall_relative_error')])
    with (root/'nemo_predictions.jsonl').open('w') as f:
        for g in graded:f.write(json.dumps(g,sort_keys=True,ensure_ascii=False)+'\n')
    figures=plots(root,rows);narrative(root,setup,rows,tasks,compared,policies,figures)
    (root/'findings.md').write_text('# Empirical findings\n\n'+findings(setup,rows,compared))
    names=['per_sample.json','summary.json','summary.csv','per_task.csv','per_subtask.csv','comparisons.json','comparisons.csv',
        'routing_marginals.csv','per_layer_head_step.csv.gz','thresholds.json','nemo_predictions.jsonl','correlations.json','report.md','findings.md',*figures]
    failures=root/'failures.jsonl'
    audit.update(complete=True,failed_attempts=len(failures.read_text().splitlines()) if failures.exists() else 0,
        report_source_sha256=sha(Path(__file__).read_bytes()),artifacts={n:sha((root/n).read_bytes()) for n in names})
    _write(root/'audit.json',audit);return audit
