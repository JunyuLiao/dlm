"""Regenerate final tables/plots only from completed, audited raw outputs."""
import argparse
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path

from experiments.diffusion_gemma_value_aware.report import summary,csv_write,table,plot_series
from experiments.diffusion_gemma_value_aware.report_metrics import (
    aggregate,accumulate_marginals,SUM_FIELDS,matched_blasst,descriptive_correlations,work_opportunities)
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_value_aware.operators import BETAS
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .engine import contract
from .evidence import read_result,pair,merge_sources
from .final import load_contract,PHASES
from .protocol import ROOT,prepare,sha

VIEWS=(('aime26','heldout24','AIME26 previously exposed24'),
    ('aime26','full','AIME26 full30, includes calibration6'),
    ('longbench_v2','full','LongBench v2 short-focused30, 32k cap'))


def expected_figures(rows):
    names=[f'{metric}.png' for metric in ('accuracy','overall_denominator_mass','overall_relative_error','token_agreement')]
    for benchmark,split,_ in VIEWS:
        tag=f'{benchmark}_{split}';names.append(f'{tag}_target_actual.png')
        if any(r['benchmark']==benchmark and r['split']==split and r['method'] in ('compensate','zero_pv') for r in rows):
            names.append(f'{tag}_pv_replacement.png')
    return names


def grouped_summaries(raw,conditions):
    groups=defaultdict(list);tasks=defaultdict(list)
    for row in raw:
        groups[row['benchmark'],'full',row['condition']].append(row)
        if row['benchmark']=='aime26':
            groups['aime26','calibration6' if row['calibration'] else 'heldout24',row['condition']].append(row)
        tasks[row['benchmark'],row['task'],row['condition']].append(row)
    rows=[]
    for (benchmark,split,name),group in sorted(groups.items()):
        condition=conditions[name];config=condition['config']
        rows.append(dict(benchmark=benchmark,split=split,condition=name,
            method=config.get('method','dense'),pooling=config.get('pooling'),mode=config.get('mode'),
            target=condition.get('target'),target_metric=condition['target_metric'],
            thresholds=condition.get('thresholds',{}).get(benchmark),
            beta=BETAS[config['amount']] if config.get('method')=='sol' and config.get('mode')=='gaussian' else None,
            unparsed_answers=sum(r['unparsed_answer'] for r in group),**summary(group)))
    per_task=[dict(benchmark=b,task=t,condition=c,unparsed_answers=sum(r['unparsed_answer'] for r in g),**summary(g))
        for (b,t,c),g in sorted(tasks.items())]
    return rows,per_task


def comparison_intervals(comparisons,raw):
    for comparison in comparisons:
        if not comparison['comparable_within_three_points']:
            continue
        group=[r for r in raw if r['benchmark']==comparison['benchmark']
            and (comparison['split']!='heldout24' or not r['calibration'])]
        a={r['id']:r['accuracy'] for r in group if r['condition']==comparison['candidate']}
        b={r['id']:r['accuracy'] for r in group if r['condition']==comparison['reference']}
        if set(a)!=set(b):
            raise ValueError('paired candidate/reference sample mismatch')
        comparison['paired_ci95']=paired_bootstrap_ci([a[i]-b[i] for i in sorted(a)])
    return comparisons


def plots(dest,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=dest/'figures';folder.mkdir(exist_ok=True)
    views=VIEWS
    for metric,label in [('accuracy','Accuracy'),('overall_denominator_mass','Retained attention mass'),
            ('overall_relative_error','Relative attention-output error'),('token_agreement','Token agreement')]:
        fig,axes=plt.subplots(1,3,figsize=(18,4.5))
        for ax,(benchmark,split,title) in zip(axes,views):
            for name,points in plot_series(rows,benchmark,split).items():
                ax.plot([100*r['overall_physical_sparsity'] for r in points],[r[metric] for r in points],
                    'o-',label=name,markersize=3)
            ax.set(title=title,xlabel='Measured physical deletion (%)',ylabel=label);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(folder/f'{metric}.png',dpi=170);plt.close(fig)
    for benchmark,split,title in views:
        selected=[r for r in rows if r['benchmark']==benchmark and r['split']==split]
        tag=f'{benchmark}_{split}'
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,kind in zip(axes,('overall','global','local')):
            for name,points in plot_series(rows,benchmark,split).items():
                points=sorted([r for r in points if r['target'] is not None],key=lambda r:r['target'])
                if points:
                    ax.plot([100*r['target'] for r in points],[100*r[f'{kind}_physical_sparsity'] for r in points],'o-',label=name)
            ax.plot([0,100],[0,100],':',color='grey');ax.set(title=kind,xlabel='Target deletion (%)',ylabel='Actual deletion (%)')
            ax.grid(alpha=.2)
        fig.suptitle(title);axes[-1].legend(fontsize=6);fig.tight_layout()
        fig.savefig(folder/f'{tag}_target_actual.png',dpi=170);plt.close(fig)
        replacements=[r for r in selected if r['method'] in ('compensate','zero_pv')]
        if replacements:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for method in ('compensate','zero_pv'):
                points=sorted([r for r in replacements if r['method']==method],key=lambda r:r['overall_pv_omission'])
                for ax,metric in zip(axes,('accuracy','overall_relative_error')):
                    ax.plot([100*r['overall_pv_omission'] for r in points],[r[metric] for r in points],'o-',label=method)
                    ax.set(xlabel='Full-PV omission (%) — no physical deletion',ylabel=metric);ax.grid(alpha=.2)
            axes[-1].legend();fig.suptitle(title);fig.tight_layout()
            fig.savefig(folder/f'{tag}_pv_replacement.png',dpi=170);plt.close(fig)


def write_markdown(dest,setup,frozen,rows,per_task,comparisons,correlations):
    selected=[r for r in rows if r['split'] in ('full','heldout24')]
    main=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],pooling=r['pooling'],n=r['count'],
        target=100*r['target'] if r['target'] is not None else None,threshold=threshold_text(r),
        accuracy=100*r['accuracy'],dense_delta_pp=100*r['delta'],physical=100*r['overall_physical_sparsity'],
        global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_denominator_mass'],exact_pv_mass=100*r['overall_mass'],
        output_error=r['overall_relative_error'],agreement=100*r['token_agreement'],
        pv_omission=100*r['overall_pv_omission']) for r in selected]
    csv_write(dest/'main_ablation.csv',main)
    work=[dict(benchmark=r['benchmark'],split=r['split'],condition=r['condition'],
        eligible=r['overall_eligible'],physical_sparsity=r['overall_physical_sparsity'],
        pv_omission=r['overall_pv_omission'],softmax_sparsity=r['overall_softmax_sparsity'],
        compensation_fraction=r['overall_compensation_fraction'],prefix_sparsity=r['overall_prefix_sparsity'],
        canvas_sparsity=r['overall_canvas_sparsity'],boundary_sparsity=r['overall_boundary_sparsity'],
        **work_opportunities(r)) for r in selected]
    csv_write(dest/'execution_work.csv',work)
    text='# Value-aware BLASST: AIME26 and LongBench v2\n\n'
    text+=f"Audited {frozen['expected_shards']} sample-condition results, {len(frozen['conditions'])} conditions. {frozen['scope']}.\n\n"
    text+='## Setup\n\n'
    text+='AIME26 uses all30 problems; full30 includes the original calibration6 (IDs2,8,14,20,23,30). Results on the other24 are shown separately, but those questions were exposed in prior work: new-design conclusions are exploratory, not confirmatory. LongBench v2 uses10 single-document QA,10 multi-document QA and10 code-repository-understanding multiple-choice questions, five easy/five hard per domain. Two additional questions per domain calibrate thresholds and two support development; neither overlaps final30. Selection is short-first then seed42 hash, before scoring.\n\n'
    text+='The32,768-token model-input cap uses official-style head/tail truncation. Fourteen final v2 questions are truncated (3 single-doc,2 multi-doc,9 code). This is a short-focused,32k-context subset, **not full-context LongBench v2 performance**. Official zero-shot prompts,128-token answer budget and exact answer extractor are preserved; unparsed answers score0.\n\n'
    text+=f"BF16 model revision `{setup['revision']}`,128×64 physical tiles, prefix+canvas eligible, native GQA/scaling/structural masks. Canvas256, up to48 denoising steps, thinking off, seed reset per sample. Requested temperature0 is the native0.4–0.8 schedule sentinel, **not greedy decoding**. Dense baselines are cached once; identical complete sparse settings may reuse a source across target labels.\n\n"
    text+='## Definitions and thresholds\n\n'
    text+='Physical sparsity is sum(skipped eligible tiles)/sum(eligible tiles), never an average of per-call sparsities. Overall means decoder denoising attention (25 local and5 global layers); encoder/prefill stays dense and is excluded. The mass diagnostic compares masked and dense attention on the **same Q/K/V state on the sparse trajectory**. It does not compare attention tensors from diverged dense and sparse generations. Relative attention-output error is sqrt(sum squared error / sum squared dense output).\n\n'
    text+='Token agreement is matching generated token IDs / max(dense length,sparse length), summed across prompts. All positions are compared even after the first divergence; missing/extra positions disagree and EOS is included. Sequence match requires identical full token lists. Paired prompt-bootstrap95% intervals use the existing20,000-draw seed42 implementation. V2 accuracy is reported per domain before its equal-domain macro. Small selected samples and dependent configurations limit generalization.\n\n'
    text+='BLASST retains the existing inverse-valid-KV-length rule λ(L)=exp(logκ)/L, calibrated separately for local/global layers. Original BLASST caps λ at1; a verified unattainable type uses constant1 at every final length, independently of the other type. The aggressive λ>1 extension keeps the repository’s preceding seen-maximum convention and is labeled separately. New risks use retained pre-block online state and scalar thresholds calibrated by the existing empirical-risk-rank refinement. Calibration never uses final scores. Full thresholds, raw verification points and source hashes are in the frozen contract and policy files.\n\n'
    text+='The no-value control sets the value rule’s magnitude/reference ratio exactly1 while preserving its retained-state convention and scalar-threshold family. Mass-only similarly controls mass×value. These are stronger tests of value-specific benefit than comparing against legacy BLASST alone. Selected pools are value=vector_mean, mass×value=RMS, output-risk=mean; mean and RMS tied for output-risk and the established lexical tie rule selected mean. Pooling was selected on calibration shared-state error, not downstream final accuracy.\n\n'
    text+='Compensation and zero-PV preserve exact denominators and have zero physical deletion; their targets refer to full-PV omission. Retained denominator mass is100% and cannot by itself diagnose their output quality. Exact-PV mass and output error are shown separately. Exact-mass routing requires block softmax before deciding. Sol uses fixed Gaussian β values without recalibration or approximate correction; its value proxy adds log RMS value norm. Guarded Sol/ranking nonempty repair consults only its declared signal. Top-p sums nonnegative true masses or contribution norms, never signed logits.\n\n'
    text+='## Results\n\nPercentage columns are percentages; output error is a ratio. PV-replacement targets are not physical-deletion targets.\n\n'
    columns=['condition','pooling','n','target','threshold','accuracy','dense_delta_pp','physical','global_s','local_s','mass','exact_pv_mass','output_error','agreement','pv_omission']
    for benchmark,split in (('aime26','heldout24'),('aime26','full'),('longbench_v2','full')):
        text+=f'### {benchmark} — {split}\n\n'
        text+=table([r for r in main if r['benchmark']==benchmark and r['split']==split],columns)+'\n\n'
    text+='## LongBench v2 per-domain results\n\n'
    task_rows=[dict(task=r['task'],condition=r['condition'],n=r['count'],accuracy=100*r['accuracy'],
        actual=100*r['overall_physical_sparsity'],unparsed=r['unparsed_answers']) for r in per_task if r['benchmark']=='longbench_v2']
    text+=table(task_rows,['task','condition','n','accuracy','actual','unparsed'])+'\n\n'
    text+='## Actual-sparsity comparisons\n\n'
    text+=table(comparisons,['benchmark','candidate','reference','candidate_sparsity','reference_sparsity',
        'comparable_within_three_points','score_delta','paired_ci95'])+'\n\n'
    text+='Comparisons use the nearest observed BLASST point within3 percentage points of actual sparsity, with no target-based interpolation. Larger gaps deliberately have no score-delta claim. Positive point estimates alone do not establish a stronger method; repeated paired seeds and uncertainty remain relevant.\n\n'
    for benchmark,split in (('aime26','heldout24'),('longbench_v2','full')):
        dense=next(r for r in rows if r['benchmark']==benchmark and r['split']==split and r['method']=='dense')
        matches=[r for r in comparisons if r['benchmark']==benchmark and r['comparable_within_three_points']]
        gains=[r for r in matches if r['score_delta']>0]
        text+=f"- {benchmark}: dense accuracy {100*dense['accuracy']:.2f}%; {len(gains)}/{len(matches)} nearby actual-sparsity comparisons have positive candidate point estimates. This is descriptive, not proof of equivalence or significance.\n"
    text+='\nPlots separate physical deletion from PV replacement. `routing_marginals.csv` and compressed per-layer/head/step records expose global/local and positional effects; `execution_work.csv` distinguishes operator deletion, softmax work and PV omission. All emulation paths compute dense diagnostics. **No latency, throughput, kernel skipping or speedup is measured or claimed.**\n\n'
    text+='Detailed mechanism diagnosis, calibration misses, streaming compatibility and the next justified iteration belong in the study’s final synthesis; the broad50 report alone does not complete that study.\n'
    (dest/'report.md').write_text(text)


def regenerate(root,phase):
    setup=prepare(root);execution=contract(root);frozen=load_contract(root,phase,execution)
    dest=root/'reports'/phase;dest.mkdir(parents=True,exist_ok=True)
    missing=[];violations=[];raw=[];sources={};dimensions={}
    temporary=dest/'per_layer_head_step.csv.gz.tmp';writer=None
    with temporary.open('wb') as binary:
        with gzip.GzipFile(filename='',fileobj=binary,mode='wb',compresslevel=3,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as handle:
                for row in setup['final']:
                    try:
                        dense,dense_source=read_result(root,row,'dense','dense',{},None,execution)
                    except Exception as error:
                        violations.append(dict(id=row['id'],condition='dense',error=str(error)));continue
                    sources=merge_sources(sources,{dense_source['path']:dense_source['sha256']})
                    for name,condition in sorted(frozen['conditions'].items()):
                        try:
                            if name=='dense':
                                out,source=dense,dense_source
                            else:
                                out,source=read_result(root,row,'final',name,condition['config'],
                                    condition.get('thresholds',{}).get(row['benchmark']),execution)
                            raw.append(dict(condition=name,**pair(row,out,dense)))
                            sources=merge_sources(sources,{source['path']:source['sha256']})
                            records=[r for r in out['records'] if r['probe']=='execution']
                            accumulate_marginals(dimensions,records,row['benchmark'],name)
                            for record in records:
                                dimension=dict(id=row['id'],benchmark=row['benchmark'],condition=name,**record)
                                if writer is None:
                                    writer=csv.DictWriter(handle,fieldnames=list(dimension));writer.writeheader()
                                writer.writerow(dimension)
                        except FileNotFoundError as error:
                            missing.append(dict(id=row['id'],condition=name,error=str(error)))
                        except Exception as error:
                            violations.append(dict(id=row['id'],condition=name,error=str(error)))
    expected=frozen['expected_shards'];counts=defaultdict(int)
    for row in raw:counts[row['condition']]+=1
    audit=dict(raw_complete=not missing and not violations and len(raw)==expected
        and all(counts[n]==60 for n in frozen['conditions']),expected=expected,completed=len(raw),
        complete=False,
        phase=phase,missing=missing,violations=violations,per_condition=dict(counts),
        fingerprint=execution['fingerprint'],sources=sources,
        contract_sha256=sha((root/'final_contracts'/f'{phase}.json').read_bytes()),
        report_source_sha256=sha(Path(__file__).read_bytes()),
        source_scope='completed final outputs and cached dense outputs only; no inference/calibration during regeneration')
    _write(dest/'audit.json',audit)
    if not audit['raw_complete']:
        return audit
    temporary.replace(dest/'per_layer_head_step.csv.gz')
    rows,per_task=grouped_summaries(raw,frozen['conditions'])
    comparisons=comparison_intervals(matched_blasst(rows),raw);correlations=descriptive_correlations(rows)
    _write(dest/'summary.json',rows);csv_write(dest/'summary.csv',rows)
    _write(dest/'per_sample.json',raw);csv_write(dest/'per_task.csv',per_task)
    _write(dest/'matched_sparsity.json',comparisons);_write(dest/'correlations.json',correlations)
    csv_write(dest/'routing_marginals.csv',[dict(benchmark=b,condition=n,axis=a,index=i,attention_type=k,
        **aggregate([dict(zip(SUM_FIELDS,v))])) for (b,n,a,i,k),v in sorted(dimensions.items())])
    plots(dest,rows);write_markdown(dest,setup,frozen,rows,per_task,comparisons,correlations)
    names=('summary.json','summary.csv','per_sample.json','per_task.csv','matched_sparsity.json',
        'correlations.json','routing_marginals.csv','per_layer_head_step.csv.gz','main_ablation.csv',
        'execution_work.csv','report.md')
    audit['artifacts']={name:sha((dest/name).read_bytes()) for name in names}
    audit['artifacts'].update({f'figures/{name}':sha((dest/'figures'/name).read_bytes()) for name in expected_figures(rows)})
    audit['complete']=True;_write(dest/'audit.json',audit)
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--phase',choices=PHASES,default='broad50');args=p.parse_args()
    audit=regenerate(args.output,args.phase)
    print(json.dumps({k:audit[k] for k in ('complete','completed','expected','missing','violations')}))
    if not audit['complete']:raise SystemExit(1)


if __name__=='__main__':
    main()
