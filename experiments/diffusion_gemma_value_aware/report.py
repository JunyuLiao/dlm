"""Scientific tables/audits from immutable shards, never partial final claims."""
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import ROOT,prepare,fingerprint,score,sha
from .report_metrics import aggregate,sample_distributions,descriptive_correlations,matched_blasst,accumulate_marginals,SUM_FIELDS
from .execution import provenance,require_refinement_smoke
from .run import shard_path
from .operators import BETAS


def csv_write(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def table(rows,cols):
    def fmt(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.4f}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+
        ['| '+' | '.join(fmt(r.get(c)) for c in cols)+' |' for r in rows])


def inspect_shard(path,row,fp,condition):
    return inspect_output(json.loads(path.read_text()),row,fp,condition)


def inspect_output(d,row,fp,condition):
    """Shared in-memory audit for final and candidate-selection evidence."""
    violations=[]
    if d['fingerprint']!=fp:violations.append('fingerprint')
    for key in ('id','prompt_hash','seed','generation_budget'):
        if d[key]!=row[key]:violations.append(key)
    if d['config']!=condition['config']:violations.append('method config')
    for key in ('refinement_sha256','ranking_guard_sha256'):
        expected=provenance(condition['config']).get(key)
        if d.get(key)!=expected or condition.get(key)!=expected:violations.append(key)
    expected=condition.get('thresholds',{}).get(row['benchmark'])
    if d['thresholds']!=expected:violations.append('threshold provenance')
    records=[r for r in d['records'] if r['probe']=='execution']
    if {r['layer'] for r in records}!=set(range(30)):violations.append('layers')
    if {r['head'] for r in records}!=set(range(16)):violations.append('heads')
    if {r['attention_type'] for r in records}!={'local','global'}:violations.append('types')
    if any({r['head'] for r in records if r['layer']==layer}!=set(range(16)) for layer in range(30)):
        violations.append('per-layer head coverage')
    keys=[(r['layer'],r['head'],r['step']) for r in records]
    if len(keys)!=len(set(keys)):violations.append('duplicate routing bucket')
    for r in records:
        values=[r[k] for k in ('eligible','skipped','pv_omitted','mass_sum','rows','error_sq','dense_sq')]
        if not np.isfinite(values).all():violations.append('nonfinite stats');break
        if not 0<=r['skipped']<=r['pv_omitted']<=r['eligible']:violations.append('work counts');break
        if not 0<=r['softmax_skipped']<=r['skipped']:violations.append('softmax work counts');break
        if not 0<=r['compensated']<=r['pv_omitted']:violations.append('compensation work counts');break
        if 'exact_mass' in condition['config'].get('mode','') and r['softmax_skipped']!=0:
            violations.append('exact-mass routing cannot skip block softmax');break
        if condition['config'].get('method')=='diagnostic' and condition['config'].get('pooling') in ('mass','contribution') and r['softmax_skipped']!=0:
            violations.append('true mass/contribution selection already requires softmax');break
        if r['prefix_eligible']+r['canvas_eligible']+r['boundary_eligible']!=r['eligible']:violations.append('region total');break
        if r['prefix_skipped']+r['canvas_skipped']+r['boundary_skipped']!=r['skipped']:violations.append('region skips');break
        tolerance=max(.001,1e-5*r['rows'])
        if not -tolerance<=r['mass_sum']<=r['rows']+tolerance:violations.append('mass bounds');break
        if r['attention_type']!=('global' if r['layer']%6==5 else 'local'):violations.append('layer classification');break
    if not d['finite_calls'] or d['termination_reason'] not in ('eos','length'):violations.append('generation coverage')
    if not isinstance(d['completion_tokens'],list) or any(not isinstance(x,int) for x in d['completion_tokens']):
        violations.append('invalid completion token IDs')
    return d,records,violations


def summary(group):
    tasks=defaultdict(list)
    for row in group:tasks[row['task']].append(row['accuracy'])
    result=dict(count=len(group),accuracy=float(np.mean([np.mean(v) for v in tasks.values()])),
        dense_accuracy=float(np.mean([r['dense_accuracy'] for r in group])),
        delta=float(np.mean([r['accuracy']-r['dense_accuracy'] for r in group])),
        paired_ci95=paired_bootstrap_ci([r['accuracy']-r['dense_accuracy'] for r in group]),
        matching_tokens=sum(r['matching'] for r in group),compared_tokens=sum(r['compared'] for r in group),
        token_agreement=sum(r['matching'] for r in group)/sum(r['compared'] for r in group) if sum(r['compared'] for r in group) else 1.,
        sequence_exact_match=float(np.mean([r['exact_match'] for r in group])))
    result['relative_accuracy']=result['accuracy']/result['dense_accuracy'] if result['dense_accuracy'] else None
    result.update(length_terminated=sum(r.get('termination_reason')=='length' for r in group),
        empty_outputs=sum(r.get('output_length',1)==0 for r in group))
    for kind in ('overall','global','local'):
        parts=[]
        for row in group:
            if 'aggregates' in row:parts.append(row['aggregates'][kind])
            else:parts.extend(r for r in row['records'] if kind=='overall' or r['attention_type']==kind)
        result.update({f'{kind}_{k}':v for k,v in aggregate(parts).items()})
    if all('aggregates' in r for r in group):result.update(sample_distributions(group))
    return result


def report(root=ROOT):
    setup=prepare(root);fp=fingerprint(root)
    if not (root/'final_contract.json').exists():
        _write(root/'audit.json',dict(complete=False,stage='screening_or_development',
            reason='No frozen final contract or complete end-to-end sweep yet',fingerprint=fp))
        return
    contract=json.loads((root/'final_contract.json').read_text());assert contract['fingerprint']==fp
    require_refinement_smoke(root,fp,[c['config'] for c in contract['conditions'].values()])
    violations=[];missing=[];raw=[]
    for source,expected_hash in {**setup['source_hashes'],**contract['sources']}.items():
        if not Path(source).exists() or sha(Path(source).read_bytes())!=expected_hash:
            violations.append(dict(source=source,error='frozen calibration/selection provenance changed'))
    smoke=json.loads((root/'smoke.json').read_text())
    if not smoke['passed'] or smoke['fingerprint']!=fp:violations.append(dict(error='CUDA smoke provenance'))
    # Stream the large layer/head/step table; retain only per-example sufficient
    # sums in RAM. A full sweep can contain tens of millions of routing buckets.
    dimensional_tmp=root/'per_layer_head_step.csv.gz.tmp';writer=None
    dimensions={}
    with gzip.open(dimensional_tmp,'wt',newline='',compresslevel=3) as dimensional_file:
        for row in setup['final']:
            dense_path=shard_path(root,'final','dense',row['id'])
            dense=json.loads(dense_path.read_text()) if dense_path.exists() else None
            for name,condition in contract['conditions'].items():
                path=shard_path(root,'final',name,row['id'])
                if not path.exists():missing.append(dict(id=row['id'],condition=name));continue
                d,records,errors=inspect_shard(path,row,fp,condition)
                violations.extend(dict(id=row['id'],condition=name,error=e) for e in errors)
                if dense is None:continue
                for key in ('native_canvas_length','thinking','sampling','denoising_configuration'):
                    if d['generation_metadata'].get(key)!=dense['generation_metadata'].get(key):
                        violations.append(dict(id=row['id'],condition=name,error=f'unrelated decoding setting changed: {key}'))
                matching,compared=token_counts(dense['completion_tokens'],d['completion_tokens'])
                aggregates={kind:aggregate([r for r in records if kind=='overall' or r['attention_type']==kind]) for kind in ('overall','global','local')}
                raw.append(dict(id=row['id'],benchmark=row['benchmark'],task=row['task'],calibration=row['calibration'],
                    condition=name,accuracy=score(row,d['prediction']),dense_accuracy=score(row,dense['prediction']),
                    matching=matching,compared=compared,exact_match=d['completion_tokens']==dense['completion_tokens'],aggregates=aggregates,
                    output_length=len(d['completion_tokens']),termination_reason=d['termination_reason']))
                for r in records:
                    dimension=dict(id=row['id'],benchmark=row['benchmark'],condition=name,**r)
                    if writer is None:
                        writer=csv.DictWriter(dimensional_file,fieldnames=list(dimension));writer.writeheader()
                    writer.writerow(dimension)
                accumulate_marginals(dimensions,records,row['benchmark'],name)
    expected=len(setup['final'])*len(contract['conditions'])
    audit=dict(complete=not missing and not violations and len(raw)==expected,expected=expected,completed=len(raw),
        missing=missing,violations=violations,fingerprint=fp,contract_sha256=sha((root/'final_contract.json').read_bytes()))
    _write(root/'audit.json',audit)
    # Never present incomplete final conditions as completed evaluations.
    if not audit['complete']:return
    dimensional_tmp.replace(root/'per_layer_head_step.csv.gz')
    csv_write(root/'routing_marginals.csv',[dict(benchmark=b,condition=n,axis=a,index=i,attention_type=k,**aggregate([dict(zip(SUM_FIELDS,v))]))
        for (b,n,a,i,k),v in dimensions.items()])
    groups=defaultdict(list);tasks=defaultdict(list)
    for row in raw:
        groups[row['benchmark'],'full',row['condition']].append(row)
        if row['benchmark']=='aime26':groups[row['benchmark'],'calibration6' if row['calibration'] else 'heldout24',row['condition']].append(row)
        tasks[row['benchmark'],row['task'],row['condition']].append(row)
    rows=[]
    for (benchmark,split,name),group in groups.items():
        config=contract['conditions'][name]
        rows.append(dict(benchmark=benchmark,split=split,condition=name,method=config['config'].get('method','dense'),
            pooling=config['config'].get('pooling'),target=config.get('target'),
            beta=BETAS[config['config']['amount']] if config['config'].get('method')=='sol' and config['config'].get('mode')=='gaussian' else None,
            mode=config['config'].get('mode'),target_metric=config.get('target_metric','physical_sparsity'),
            thresholds=config.get('thresholds',{}).get(benchmark),**summary(group)))
    per_task=[dict(benchmark=b,task=t,condition=c,**summary(g)) for (b,t,c),g in tasks.items()]
    _write(root/'summary.json',rows);csv_write(root/'summary.csv',rows);csv_write(root/'per_task.csv',per_task)
    _write(root/'per_sample.json',[{k:v for k,v in r.items() if k!='records'} for r in raw])
    comparisons=matched_blasst(rows);correlations=descriptive_correlations(rows)
    for comparison in comparisons:
        if not comparison['comparable_within_three_points']:continue
        group=[r for r in raw if r['benchmark']==comparison['benchmark']
            and (comparison['split']!='heldout24' or not r['calibration'])]
        candidate={r['id']:r['accuracy'] for r in group if r['condition']==comparison['candidate']}
        reference={r['id']:r['accuracy'] for r in group if r['condition']==comparison['reference']}
        assert set(candidate)==set(reference)
        comparison['paired_ci95']=paired_bootstrap_ci([candidate[k]-reference[k] for k in sorted(candidate)])
    csv_write(root/'matched_actual_sparsity.csv',comparisons);_write(root/'descriptive_correlations.json',correlations)
    plot(root,rows)
    from .scientific_report import write_report
    write_report(root,setup,contract,rows,per_task,comparisons,correlations)


REVISION_TEXT='f7f5b7f5fa82ffc52addd066915886d497f5517b'


def plot_family(condition):
    """Strip only a terminal numeric budget, never `_plain` or a mode name."""
    import re
    return re.sub(r'_(?:s|p)\d+$','',condition)


def plot_series(rows,benchmark,split,primary_only=False):
    groups=defaultdict(list)
    for row in rows:
        if row['benchmark']!=benchmark or row['split']!=split:continue
        if row['method'] in ('compensate','zero_pv'):continue
        if primary_only and row['method'] in ('sol','diagnostic'):continue
        groups[plot_family(row['condition'])].append(row)
    return {name:sorted(points,key=lambda r:r['overall_physical_sparsity'])
        for name,points in sorted(groups.items())}


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'figures';dest.mkdir(exist_ok=True)
    views=[('aime26','heldout24','AIME26 heldout24'),
        ('aime26','full','AIME26 full30 (includes calibration6)'),
        ('longbench','full','LongBench50')]
    for metric,label in [('accuracy','Task score'),('overall_denominator_mass','Retained attention mass'),
                         ('overall_relative_error','Relative attention-output error'),('token_agreement','Token agreement')]:
        # The full-data view remains available, explicitly distinguished from
        # heldout24. A focused view separates streaming candidates/references
        # from global-ranking/Sol diagnostics without dropping their full plots.
        for primary_only,panels,prefix in [(False,views,''),(True,[views[0],views[2]],'primary_')]:
            fig,axes=plt.subplots(1,len(panels),figsize=(6*len(panels),4.5))
            for ax,(benchmark,split,title) in zip(axes,panels):
                for method,points in plot_series(rows,benchmark,split,primary_only).items():
                    ax.plot([100*r['overall_physical_sparsity'] for r in points],[r[metric] for r in points],'o-',label=method,markersize=3)
                ax.set(title=title,xlabel='Measured physical deletion (%)',ylabel=label);ax.grid(alpha=.2)
            axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(dest/f'{prefix}{metric}.png',dpi=170);plt.close(fig)
    # Different work types have separate horizontal axes and curves.
    for benchmark,split,title in views:
        tag=benchmark if split=='full' else f'{benchmark}_{split}'
        points=[r for r in rows if r['benchmark']==benchmark and r['split']==split]
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,kind in zip(axes,('overall','global','local')):
            for method in sorted({plot_family(r['condition']) for r in points}):
                group=[r for r in points if plot_family(r['condition'])==method and r['target'] is not None and r['method'] not in ('compensate','zero_pv')]
                if not group:continue
                group.sort(key=lambda r:r['target'])
                ax.plot([100*r['target'] for r in group],[100*r[f'{kind}_physical_sparsity'] for r in group],'o-',label=method,markersize=3)
            ax.plot([0,100],[0,100],':',color='grey',linewidth=1)
            ax.set(title=f'{title}: {kind}',xlabel='Target physical deletion (%)',ylabel='Measured physical deletion (%)');ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(dest/f'{tag}_target_vs_actual.png',dpi=170);plt.close(fig)
        replacements=[r for r in points if r['method'] in ('compensate','zero_pv')]
        if replacements:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for method in sorted({plot_family(r['condition']) for r in replacements}):
                group=sorted([r for r in replacements if plot_family(r['condition'])==method],key=lambda r:r['overall_pv_omission'])
                for ax,metric,label in zip(axes,('accuracy','overall_relative_error'),('Task score','Relative attention-output error')):
                    ax.plot([100*r['overall_pv_omission'] for r in group],[r[metric] for r in group],'o-',label=method)
                    ax.set(title=title,xlabel='Measured full-PV omission (%) — no deletion',ylabel=label);ax.grid(alpha=.2)
            axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/f'{tag}_pv_replacement.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args();report(args.output)


if __name__=='__main__':main()
