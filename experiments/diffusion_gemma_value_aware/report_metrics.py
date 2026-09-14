"""Expanded reporting sums; do not alter frozen initial screening summaries."""
import math
import numpy as np
from .calibration import aggregate as base_aggregate,FIELDS

EXTRA=('rescued_rows','prefix_eligible','prefix_skipped','prefix_replaced',
    'canvas_eligible','canvas_skipped','canvas_replaced','boundary_eligible','boundary_skipped','boundary_replaced')
SUM_FIELDS=FIELDS+EXTRA


def accumulate_marginals(store,records,benchmark,condition):
    values=np.asarray([[r.get(k,0.) for k in SUM_FIELDS] for r in records],dtype=np.float64)
    for axis in ('layer','head','step'):
        keys=[(r[axis],r['attention_type']) for r in records]
        unique=sorted(set(keys));indices={k:i for i,k in enumerate(unique)}
        totals=np.zeros((len(unique),len(SUM_FIELDS)),dtype=np.float64)
        np.add.at(totals,[indices[k] for k in keys],values)
        for (index,kind),sums in zip(unique,totals):
            key=(benchmark,condition,axis,index,kind)
            if key in store:store[key]+=sums
            else:store[key]=sums


def aggregate(records):
    result=base_aggregate(records)
    result.update({k:sum(r.get(k,0.) for r in records) for k in EXTRA})
    n=max(result['eligible'],1);q=max(result['rows'],1)
    result.update(softmax_sparsity=result['softmax_skipped']/n,
        compensation_fraction=result['compensated']/n,
        mass_selection_gap=(result['same_budget_best_mass_sum']-result['mass_sum'])/q,
        wrongly_replaced_fraction=result['wrongly_replaced_vs_mass']/n,
        rescued_row_rate=result['rescued_rows']/q,
        **{f'{region}_sparsity':result[f'{region}_skipped']/max(result[f'{region}_eligible'],1)
            for region in ('prefix','canvas','boundary')})
    return result


def correlation(x,y):
    a=np.asarray(x,dtype=float);b=np.asarray(y,dtype=float)
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def sample_distributions(group):
    result={}
    for kind in ('overall','global','local'):
        for field in ('mass','relative_error','physical_sparsity'):
            values=[r['aggregates'][kind][field] for r in group]
            result[f'{kind}_{field}_sample_p10_p50_p90']=np.quantile(values,[.1,.5,.9]).tolist()
    return result


def descriptive_correlations(rows):
    result=[]
    for benchmark in sorted({r['benchmark'] for r in rows}):
        split='heldout24' if benchmark=='aime26' else 'full'
        for scope in ('deletion','pv_replacement'):
            selected=[r for r in rows if r['benchmark']==benchmark and r['split']==split
                and r['method']!='dense' and ((r['method'] in ('compensate','zero_pv'))==(scope=='pv_replacement'))]
            for metric in ('overall_denominator_mass','overall_mass','token_agreement','overall_relative_error'):
                result.append(dict(benchmark=benchmark,split=split,scope=scope,metric=metric,n=len(selected),
                    pearson_with_accuracy=correlation([r[metric] for r in selected],[r['accuracy'] for r in selected]),
                    inference='Descriptive across dependent configurations; not causal, independent samples, or a significance test'))
    return result


def matched_blasst(rows,tolerance=.03):
    """Nearest actual-budget comparison only; no target-based interpolation."""
    result=[]
    for row in rows:
        if row['split']!=('heldout24' if row['benchmark']=='aime26' else 'full'):continue
        if row['method'] in ('dense','blasst','diagnostic','compensate','zero_pv'):continue
        refs=[r for r in rows if r['benchmark']==row['benchmark'] and r['split']==row['split'] and r['method']=='blasst']
        if not refs:continue
        ref=min(refs,key=lambda r:abs(r['overall_physical_sparsity']-row['overall_physical_sparsity']))
        gap=abs(ref['overall_physical_sparsity']-row['overall_physical_sparsity'])
        result.append(dict(benchmark=row['benchmark'],split=row['split'],candidate=row['condition'],
            reference=ref['condition'],candidate_sparsity=row['overall_physical_sparsity'],reference_sparsity=ref['overall_physical_sparsity'],
            actual_sparsity_gap=gap,comparable_within_three_points=gap<=tolerance,
            candidate_score=row['accuracy'],reference_score=ref['accuracy'],
            score_delta=row['accuracy']-ref['accuracy'] if gap<=tolerance else None,
            claim='Observed point comparison, not evidence of equivalence or significance' if gap<=tolerance else 'No matched actual-sparsity reference; do not infer a win'))
    return result


def work_opportunities(row):
    """Selection-aware opportunities, not measurements of kernel work/time."""
    contribution=row['method']=='diagnostic' and row.get('pooling')=='contribution'
    normalized=row['method']=='diagnostic' and row.get('pooling') in ('mass','contribution')
    normalized=normalized or row['method'] in ('compensate','zero_pv') or 'exact_mass' in (row.get('mode') or '')
    return dict(operator_deleted_tiles=row['overall_skipped'],
        operator_full_PV_replaced_tiles=row['overall_pv_omitted'],
        selection_requires_all_block_softmax=normalized,
        selection_requires_all_block_PV=contribution,
        selection_aware_softmax_omission_tiles=0. if normalized else row['overall_softmax_skipped'],
        selection_aware_full_PV_omission_tiles=0. if contribution else row['overall_pv_omitted'],
        interpretation='Potential work after including selection dependencies; emulation computes full diagnostic work; not measured kernel skipping or speedup')
