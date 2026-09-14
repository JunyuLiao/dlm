"""Raw-only report for the user-requested fixed25 quick50 review gate."""
from collections import Counter,defaultdict
import csv
import gzip
import io
from pathlib import Path

from experiments.diffusion_gemma_value_aware.report import summary,csv_write,table
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate,accumulate_marginals,SUM_FIELDS,work_opportunities
from experiments.diffusion_gemma_value_aware.scientific_report import threshold_text
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import sha
from .evidence import read_result,pair,merge_sources
from .quick_review import load,destination,EXPECTED,REFERENCES,DOMAINS


def summarize(raw,conditions):
    groups=defaultdict(list); tasks=defaultdict(list)
    for r in raw:
        groups[r['benchmark'],r['condition']].append(r)
        tasks[r['benchmark'],r['task'],r['condition']].append(r)
    rows=[]
    for (benchmark,name),group in sorted(groups.items()):
        c=conditions[name]; config=c['config']
        rows.append(dict(benchmark=benchmark,condition=name,split='quick_review',
            method=config.get('method','dense'),pooling=config.get('pooling'),mode=config.get('mode'),
            target=c['target'],target_metric=c['target_metric'],thresholds=c.get('thresholds',{}).get(benchmark),
            correct=sum(r['accuracy'] for r in group),unparsed_answers=sum(r['unparsed_answer'] for r in group),
            **summary(group)))
    per_task=[dict(benchmark=b,task=t,condition=n,correct=sum(r['accuracy'] for r in g),
        unparsed_answers=sum(r['unparsed_answer'] for r in g),**summary(g))
        for (b,t,n),g in sorted(tasks.items())]
    return rows,per_task


def comparisons(rows,raw):
    index={(r['benchmark'],r['condition']):r for r in rows}; results=[]
    scores=defaultdict(dict)
    for r in raw:scores[r['benchmark'],r['condition']][r['id']]=r['accuracy']
    for row in rows:
        if row['condition']=='dense' or row['condition'] in REFERENCES: continue
        b=row['benchmark']; name=row['condition']; refs=[index[b,n] for n in REFERENCES]
        better=max(refs,key=lambda r:r['correct'])
        item=dict(benchmark=b,condition=name,count=row['count'],correct=row['correct'],
            better_reference=better['condition'],extra_wrong_vs_better=better['correct']-row['correct'],
            provisional_quality_keep=better['correct']-row['correct']<=1,
            quality_rule='At most1 extra wrong vs better BLASST; agent-proposed tolerance, not user-specified or equivalence proof',
            physical_sparsity=row['overall_physical_sparsity'],pv_omission=row['overall_pv_omission'])
        for ref in refs:
            tag='original' if ref['condition']==REFERENCES[0] else 'aggressive'
            a=scores[b,name];other=scores[b,ref['condition']]
            if set(a)!=set(other): raise ValueError('unpaired quick-review comparisons')
            item[f'{tag}_correct']=ref['correct']
            item[f'delta_vs_{tag}_pp']=100*(row['accuracy']-ref['accuracy'])
            item[f'paired_ci95_vs_{tag}']=paired_bootstrap_ci([a[i]-other[i] for i in sorted(a)])
            for kind in ('overall','global','local'):
                item[f'{kind}_sparsity_gap_vs_{tag}_pp']=100*(row[f'{kind}_physical_sparsity']-ref[f'{kind}_physical_sparsity'])
            item[f'matched_physical_budget_vs_{tag}']=row['method'] not in ('compensate','zero_pv') and abs(item[f'overall_sparsity_gap_vs_{tag}_pp'])<=3
        results.append(item)
    for row in results:
        group=[r for r in results if r['condition']==row['condition']]
        row['provisional_keep_on_both_benchmarks']=len(group)==2 and all(r['provisional_quality_keep'] for r in group)
    return results


def plot(dest,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=dest/'figures';folder.mkdir(exist_ok=True);names=[]
    for b in EXPECTED:
        group=[r for r in rows if r['benchmark']==b]
        deletion=[r for r in group if r['method'] not in ('compensate','zero_pv')]
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for ax,metric,label in zip(axes,('accuracy','overall_mass','token_agreement'),('Accuracy (%)','Retained mass (%)','Token agreement (%)')):
            for r in deletion:
                x=100*r['overall_physical_sparsity'];y=100*r[metric]
                ax.scatter(x,y,s=25);ax.annotate(r['condition'].removesuffix('_s50'),(x,y),fontsize=6,xytext=(3,3),textcoords='offset points')
            ax.set(xlabel='Measured physical deletion (%)',ylabel=label);ax.grid(alpha=.2)
        fig.suptitle(f'{b}: fixed quick-review subset; 50% target, not a full sparsity curve')
        fig.tight_layout();name=f'{b}_tradeoffs.png';fig.savefig(folder/name,dpi=160);plt.close(fig);names.append(name)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,pv in zip(axes,(False,True)):
            chosen=[r for r in group if r['method']!='dense' and (r['method'] in ('compensate','zero_pv'))==pv]
            labels=[r['condition'].removesuffix('_s50') for r in chosen]
            for j,kind in enumerate(('overall','global','local')):
                field=f'{kind}_'+('pv_omission' if pv else 'physical_sparsity')
                ax.bar([i+(j-1)*.25 for i in range(len(chosen))],[100*r[field] for r in chosen],width=.25,label=kind)
            ax.axhline(50,color='black',linestyle=':',linewidth=1);ax.set_xticks(range(len(chosen)),labels,rotation=75,ha='right')
            ax.set(ylabel='PV omission (%)' if pv else 'Physical deletion (%)',title='No physical deletion' if pv else 'Actual tile-count weighting')
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        fig.suptitle(b);fig.tight_layout();name=f'{b}_target_actual.png';fig.savefig(folder/name,dpi=160);plt.close(fig);names.append(name)
    return names


def markdown(dest,frozen,rows,tasks,comp):
    main=[]
    for r in rows:
        main.append(dict(benchmark=r['benchmark'],condition=r['condition'],pooling=r['pooling'],
            target=100*r['target'],threshold=threshold_text(r),score=f"{r['correct']:g}/{r['count']}",
            accuracy=100*r['accuracy'],dense_delta_pp=100*r['delta'],physical=100*r['overall_physical_sparsity'],
            global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
            mass=100*r['overall_denominator_mass'],exact_pv_mass=100*r['overall_mass'],error=r['overall_relative_error'],
            agreement=100*r['token_agreement'],pv_omission=100*r['overall_pv_omission'],
            unparsed=r['unparsed_answers'],length_terminated=r['length_terminated']))
    csv_write(dest/'main_ablation.csv',main)
    text='# Quick50 review: value-aware BLASST on AIME26 and LongBench v2\n\n'
    text+='**Complete: 13 conditions ×25 identical cached questions =325 results. PAUSED for user review. No later targets/full study are launched.**\n\n'
    text+='## Setup and selection\n\n'
    text+='The user replaced the running60-question sweep with10 AIME26 questions and5 questions per v2 domain. This is a new frozen subset, not a rewrite of the old contracts. All compatible old outputs, prompts, token IDs, seeds and budgets are reused; remaining outputs are saved in the original resumable cache. Selection never reads predictions or scores.\n\n'
    text+=f"{frozen['selection']['rule']}. AIME excludes the six calibration questions but comes from24 previously exposed problems: exploratory, not fresh confirmatory evidence. V2 calibration/development remain disjoint. {frozen['selection']['truncated_v2']}/15 v2 prompts are head/tail truncated at32,768 input tokens. These are multiple-choice single-document QA, multi-document QA and code-repository understanding, not generative summarization/code completion or full-context v2.\n\n"
    text+='The pinned BF16 model,128×64 tiles, prefix+canvas eligibility, native GQA/scaling/local masks, canvas256, up to48 denoising steps and seed42 remain unchanged. AIME budget2048; official v2 budget128 and answer extractor. Requested temperature0 retains native0.4–0.8 sampling, not greedy. Dense is cached, not regenerated.\n\n'
    text+='## Measurements and thresholds\n\n'
    text+='Physical sparsity is total skipped eligible tiles / total eligible tiles, separately overall/global/local. Overall covers decoder denoising attention; dense encoder/prefill is excluded. Attention mass/error compare the approximate and exact operators on the same Q/K/V **on the sparse trajectory**, not hidden states from different generations. Relative error is sqrt(total error²/total dense output²). Token agreement compares every token-ID position even after divergence; missing/extra positions disagree.\n\n'
    text+='Thresholds and pooling are reused unchanged from the completed calibration/development study. Original BLASST uses the established inverse-valid-length rule with cap1; verified unattainable attention types use constantλ1 at every length. Aggressive BLASST allowsλ>1 under its documented preceding-maximum semantics. Value risks use retained pre-block state and independently calibrated local/global scalar thresholds. Selected pools: vector_mean for value, RMS for mass×value, mean for risk. Complete threshold dictionaries and original verification sources are in contract.json and its parent frozen contract. No quick-review accuracy recalibrates them.\n\n'
    text+='Compensation/zero-PV target PV omission and have zero physical deletion; their denominator mass remains100%. Exact-PV retained mass and output error are separate. Exact-mass selection already needs block softmax. No custom kernels or measured speedup claims.\n\n'
    cols=['condition','pooling','target','threshold','score','dense_delta_pp','physical','global_s','local_s','mass','exact_pv_mass','error','agreement','pv_omission']
    for b in EXPECTED:
        text+=f'## {b}: all13 conditions\n\n'+table([r for r in main if r['benchmark']==b],cols)+'\n\n'
    text+='Percent columns are percentages; output error is a ratio. Target50 is not assumed to equal actual sparsity.\n\n'
    text+='## V2 per-domain scores before aggregation\n\n'
    taskview=[dict(task=r['task'],condition=r['condition'],score=f"{r['correct']:g}/{r['count']}",physical=100*r['overall_physical_sparsity'],unparsed=r['unparsed_answers']) for r in tasks if r['benchmark']=='longbench_v2']
    text+=table(taskview,['task','condition','score','physical','unparsed'])+'\n\n'
    text+='Each domain has five questions, so the reported equal-domain macro equals15-question accuracy.\n\n'
    text+='## Provisional ideas to retain for discussion\n\n'
    text+='The user requested “not much worse” than both BLASST references but did not specify a numeric tolerance. The **provisional** flag permits at most one extra wrong answer per benchmark versus the better of original/aggressive BLASST (10pp AIME,6.67pp v2). It is a review aid, not statistical equivalence or an automatic later-stage decision. Both reference deltas and actual budget gaps are shown. A quality flag at a mismatched sparsity budget does not establish an improved frontier; PV replacement is not comparable to physical deletion. Paired bootstrap intervals and local/global gaps are in comparisons.csv/json.\n\n'
    view=[dict(benchmark=r['benchmark'],condition=r['condition'],original_delta_pp=r['delta_vs_original_pp'],aggressive_delta_pp=r['delta_vs_aggressive_pp'],extra_wrong=r['extra_wrong_vs_better'],keep_both=r['provisional_keep_on_both_benchmarks'],original_budget_gap_pp=r['overall_sparsity_gap_vs_original_pp'],aggressive_budget_gap_pp=r['overall_sparsity_gap_vs_aggressive_pp']) for r in comp]
    text+=table(view,['benchmark','condition','original_delta_pp','aggressive_delta_pp','extra_wrong','keep_both','original_budget_gap_pp','aggressive_budget_gap_pp'])+'\n\n'
    text+='Small samples, exposure of AIME and single-seed variation limit conclusions. The earlier calibration-only compensation diagnosis documents repetition despite low same-state error; this report exposes length limits and invalid answers without changing scorers. See the preserved parent diagnostics for nearly constant scalar V norms, vector cancellation, mass-bound looseness and streaming metadata/work costs.\n\n'
    text+='## Exact selected source IDs\n\n'
    for b in EXPECTED:
        text+=f"- {b}: "+', '.join(r['id'] for r in frozen['selection']['rows'] if r['benchmark']==b)+'\n'
    text+='\nAll325 raw sources and derived artifact hashes are in audit.json. Per-layer/head/step data and marginal sums retain physical counts and prefix/canvas breakdowns. **Next action: user review of this complete50% table, not an automatic later sweep.**\n'
    (dest/'report.md').write_text(text)


def regenerate(root):
    frozen,execution=load(root);dest=destination(root);conditions=frozen['conditions']
    _write(dest/'review_ready.json',dict(complete=False,pause_for_user_review=True,
        later_targets_authorized=False,reason='Raw and derived artifact audit in progress'))
    raw=[];sources={};missing=[];violations=[];dimensions={};writer=None
    temporary=dest/'per_layer_head_step.csv.gz.tmp'
    with temporary.open('wb') as binary:
        with gzip.GzipFile(filename='',fileobj=binary,mode='wb',compresslevel=3,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as handle:
                for row in frozen['selection']['rows']:
                    try: dense,dsource=read_result(root,row,'dense','dense',{},None,execution)
                    except Exception as error:
                        violations.append(dict(id=row['id'],condition='dense',error=str(error)));continue
                    sources=merge_sources(sources,{dsource['path']:dsource['sha256']})
                    for name,c in sorted(conditions.items()):
                        try:
                            out,source=(dense,dsource) if name=='dense' else read_result(root,row,'final',name,c['config'],c['thresholds'][row['benchmark']],execution)
                            raw.append(dict(condition=name,source=source,prediction=out['prediction'],
                                completion_tokens=out['completion_tokens'],**pair(row,out,dense)))
                            sources=merge_sources(sources,{source['path']:source['sha256']})
                            records=[r for r in out['records'] if r['probe']=='execution']
                            accumulate_marginals(dimensions,records,row['benchmark'],name)
                            for record in records:
                                dimension=dict(id=row['id'],benchmark=row['benchmark'],condition=name,**record)
                                if writer is None:writer=csv.DictWriter(handle,fieldnames=list(dimension));writer.writeheader()
                                writer.writerow(dimension)
                        except FileNotFoundError as error:missing.append(dict(id=row['id'],condition=name,error=str(error)))
                        except Exception as error:violations.append(dict(id=row['id'],condition=name,error=str(error)))
    counts=Counter(r['condition'] for r in raw)
    per_benchmark=Counter((r['condition'],r['benchmark']) for r in raw)
    complete=not missing and not violations and len(raw)==325 and all(
        counts[n]==25 and all(per_benchmark[n,b]==v for b,v in EXPECTED.items()) for n in conditions)
    audit=dict(complete=False,raw_complete=complete,expected=325,completed=len(raw),per_condition=dict(counts),
        missing=missing,violations=violations,sources=sources,contract_sha256=sha((dest/'contract.json').read_bytes()),
        report_source_sha256=sha(Path(__file__).read_bytes()),later_targets_authorized=False,
        source_scope='completed exact-setting raw outputs only; no inference, selection or calibration during regeneration')
    _write(dest/'audit.json',audit)
    if not complete:return audit
    temporary.replace(dest/'per_layer_head_step.csv.gz')
    rows,tasks=summarize(raw,conditions);comp=comparisons(rows,raw)
    _write(dest/'summary.json',rows);csv_write(dest/'summary.csv',rows)
    _write(dest/'per_sample.json',raw);csv_write(dest/'per_task.csv',tasks)
    _write(dest/'comparisons.json',comp);csv_write(dest/'comparisons.csv',comp)
    csv_write(dest/'routing_marginals.csv',[dict(benchmark=b,condition=n,axis=a,index=i,attention_type=k,
        **aggregate([dict(zip(SUM_FIELDS,v))])) for (b,n,a,i,k),v in sorted(dimensions.items())])
    csv_write(dest/'execution_work.csv',[dict(benchmark=r['benchmark'],condition=r['condition'],
        physical_sparsity=r['overall_physical_sparsity'],pv_omission=r['overall_pv_omission'],
        prefix_sparsity=r['overall_prefix_sparsity'],canvas_sparsity=r['overall_canvas_sparsity'],
        **work_opportunities(r)) for r in rows])
    figures=plot(dest,rows);markdown(dest,frozen,rows,tasks,comp)
    names=('summary.json','summary.csv','per_sample.json','per_task.csv','comparisons.json','comparisons.csv',
        'routing_marginals.csv','per_layer_head_step.csv.gz','main_ablation.csv','execution_work.csv','report.md')
    audit['artifacts']={name:sha((dest/name).read_bytes()) for name in names}
    audit['artifacts'].update({f'figures/{n}':sha((dest/'figures'/n).read_bytes()) for n in figures})
    audit['complete']=True;_write(dest/'audit.json',audit)
    _write(dest/'review_ready.json',dict(complete=True,expected=325,report=str(dest/'report.md'),
        pause_for_user_review=True,later_targets_authorized=False))
    return audit
