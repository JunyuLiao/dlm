"""Calibration/development-only mechanism diagnosis and information costs.

Shared dense-state budget rankings are NOT sparse-generation accuracy results.
No final outputs or final scores are read, and this analysis changes no policy.
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

from experiments.diffusion_gemma_value_aware.report import csv_write,table
from experiments.diffusion_gemma_value_aware.scientific_report import compatibility_rows
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .engine import contract
from .policies import inputs
from .protocol import ROOT,prepare,sha

FEATURES={
    'max_value_gap_minus_no_value':('value_max','no_value_control',0.),
    'mean_value_gap_minus_no_value':('value_mean','no_value_control',0.),
    'rms_value_gap_minus_no_value':('value_rms','no_value_control',0.),
    'p95_value_gap_minus_no_value':('value_p95','no_value_control',0.),
    'vector_mean_gap_minus_no_value':('value_vector_mean','no_value_control',0.),
    'aligned_gap_minus_no_value':('aligned','no_value_control',0.),
    'mass_rms_value_minus_mass':('mass_value_rms','mass',0.),
    'output_risk_mean_minus_mass_minus_log2':('risk_mean','mass',math.log(2.)),
}


def finite_delta(candidate,reference,offset=0.):
    a=np.asarray(candidate,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=1 or np.isnan(a).any() or np.isnan(b).any():
        raise ValueError('invalid or unaligned same-state risk arrays')
    if not np.array_equal(np.isposinf(a),np.isposinf(b)) or not np.array_equal(np.isneginf(a),np.isneginf(b)):
        raise ValueError('risk criteria disagree on mandatory/ineligible support')
    finite=np.isfinite(a)&np.isfinite(b)
    return a[finite]-b[finite]-offset,int((~finite).sum())


def delta_statistics(parts,excluded):
    values=np.concatenate(parts)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('missing/invalid finite value-signal differences')
    absolute=np.abs(values)
    return dict(finite_physical_candidates=len(values),excluded_nonfinite_candidates=excluded,
        mean=float(values.mean()),std=float(values.std()),
        signed_p01_p50_p99=np.quantile(values,[.01,.5,.99]).tolist(),
        abs_p50_p90_p99_max=np.quantile(absolute,[.5,.9,.99,1.]).tolist(),
        fraction_abs_below_0p001=float((absolute<1e-3).mean()),
        weighting='one observation per eligible physical query/KV tile/head/sampled call; mandatory nonfinite rows excluded explicitly')


def selected_rows(summary,configs):
    probes=dict(gap_only='screen/blasst',no_value_control='screen/no_value_control',
        mass='screen/mass',aligned='screen/aligned',centered='screen/centered',
        compensate='screen/compensate',zero_pv='screen/zero_pv',mass_exact='refine/mass_exact',
        centered_previous='refine/centered_previous',centered_previous_exact='refine/centered_previous_exact',
        contribution_diagnostic='guarded_diagnostic/contribution',mass_diagnostic='guarded_diagnostic/mass',
        qk_max_diagnostic='guarded_diagnostic/qk')
    for name in ('value','mass_value','risk'):
        probes[name]=f'screen/{name}_{configs[name]["pooling"]}'
    lookup={prefix:name for name,prefix in probes.items()};rows=[]
    for row in summary:
        if row['split'] not in ('calibration','development'):
            raise ValueError('shared-state diagnosis cannot use final results')
        prefix,_,suffix=row['probe'].rpartition('/')
        if prefix not in lookup or suffix not in ('s25','s50','s75','s90'):
            continue
        rows.append(dict(row,name=lookup[prefix],target=int(suffix[1:])/100.,
            scope='shared dense-state risk ranking at a budget; NOT calibrated sparse execution',
            target_metric='pv_omission' if lookup[prefix] in ('compensate','zero_pv') else 'physical_sparsity'))
    return rows


def comparisons(rows):
    index={(r['benchmark'],r['split'],r['attention_type'],r['target'],r['name']):r for r in rows}
    result=[]
    pairs=[('value','no_value_control'),('mass_value','mass'),('risk','mass'),('aligned','gap_only'),
        ('centered','gap_only'),('centered_previous','centered'),('centered_previous_exact','centered_previous'),
        ('mass_exact','mass'),('contribution_diagnostic','gap_only')]
    for key,candidate in index.items():
        for name,baseline in pairs:
            if candidate['name']!=name:
                continue
            ref=index.get((*key[:-1],baseline))
            if ref is None:raise ValueError('missing matched shared-state reference')
            gap=abs(candidate['physical_sparsity']-ref['physical_sparsity'])
            matched=gap<=1e-10
            result.append(dict(benchmark=key[0],split=key[1],attention_type=key[2],target=key[3],
                candidate=name,reference=baseline,physical_sparsity=candidate['physical_sparsity'],
                reference_physical_sparsity=ref['physical_sparsity'],budget_gap=gap,exact_budget_match=matched,
                candidate_error=candidate['relative_error'],reference_error=ref['relative_error'],
                relative_error_change=candidate['relative_error']/ref['relative_error']-1. if matched and ref['relative_error'] else None,
                mass_change=candidate['mass']-ref['mass'] if matched else None,
                scope='Matched shared-state diagnostic only; no downstream accuracy conclusion'))
    return result


def risk_changes(root,setup,audit):
    parts=defaultdict(list);excluded=defaultdict(int);sources={}
    for row in setup['calibration']:
        if row['split']!='calibration':
            raise ValueError('value-signal quantification uses calibration only')
        source=audit['sources'][row['id']+'/supplement']
        if source['split']!='calibration':
            raise ValueError('value-signal source is not calibration')
        path=Path(source['arrays_path'])
        if sha(path.read_bytes())!=source['arrays_sha256']:
            raise ValueError('value-signal arrays changed')
        sources[str(path)]=source['arrays_sha256']
        with np.load(path) as arrays:
            for feature,(a,b,offset) in FEATURES.items():
                for kind in ('local','global'):
                    values,n=finite_delta(arrays[f'{a}__{kind}'],arrays[f'{b}__{kind}'],offset)
                    key=row['benchmark'],kind,feature;parts[key].append(values);excluded[key]+=n
    return [dict(benchmark=b,attention_type=k,feature=f,**delta_statistics(v,excluded[b,k,f]))
        for (b,k,f),v in sorted(parts.items())],sources


def build(root):
    setup=prepare(root);execution=contract(root);_,configs=inputs(root,execution)
    audit=json.loads((root/'screen_analysis_audit.json').read_text())
    if not audit['complete'] or audit['heldout_used']:
        raise ValueError('complete non-final shared-state evidence required')
    data=json.loads((root/'screen_summary.json').read_text())
    rows=selected_rows(data,configs);paired=comparisons(rows)
    signals,sources=risk_changes(root,setup,audit)
    # Bind the model-side explanation to actual installed source, not a generic
    # assumption that learned value norms must contain useful magnitude signal.
    from transformers.models.diffusion_gemma import modeling_diffusion_gemma as model_source
    model_path=Path(model_source.__file__);model_text=model_path.read_text()
    if model_text.count('self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)')!=2:
        raise ValueError('review the model normalization explanation against changed source')
    sources[str(model_path)]=sha(model_path.read_bytes())
    for name in ('screen_summary.json','screen_analysis_audit.json','pooling_selection.json','threshold_proposals.json'):
        path=root/name;sources[str(path)]=sha(path.read_bytes())
    dest=root/'diagnostics';dest.mkdir(exist_ok=True)
    _write(dest/'selected_shared_states.json',rows);csv_write(dest/'selected_shared_states.csv',rows)
    _write(dest/'matched_shared_state_comparisons.json',paired);csv_write(dest/'matched_shared_state_comparisons.csv',paired)
    _write(dest/'value_signal_differences.json',signals);csv_write(dest/'value_signal_differences.csv',signals)
    compatibility=compatibility_rows()
    compatibility.insert(1,dict(method='Matched no-value retained-state control',
        decision='After token QK/block maximum; retained pre-block m and Z',metadata='No value-magnitude metadata needed for its mathematical decision',
        extra='Original value risk with magnitude/reference factor exactly1; reference emulation still computes diagnostics',
        information='Streaming; scalar threshold; isolates the value factor from state and inverse-L conventions',
        omission='Softmax and PV, NOT QK; no measured speedup'))
    csv_write(dest/'flashattention_compatibility.csv',compatibility)
    text='# Mechanism diagnosis — calibration/development only\n\n'
    text+='These tables compare budget-ranked risks on shared **dense** states, not calibrated sparse decoding. They cannot establish benchmark accuracy gains or stable behavior over a changed denoising trajectory. All six requested families still proceed to end-to-end evaluation. No final outputs were read and no thresholds were changed.\n\n'
    text+='## What the extra value information is doing\n\n'
    text+='Both model attention classes normalize each value token with scale-free RMSNorm before attention. Before BF16 rounding, the squared norm after this normalization is d·s/(s+ε), where s is the pre-normalization mean square. For s≫ε it is close to d. This explains why max/mean/RMS/p95 token-norm factors can add very little information; vector averaging can still expose direction/cancellation. The checked model source and its SHA are recorded in audit.json.\n\n'
    text+='`value_signal_differences.csv` measures the actual change in physical-tile log risk, weighted by candidate count, not an average of per-example variances. For pooled value-gap routing this equals log(pooled magnitude/reference), up to floating-point rounding, because the tile scalar is common to its query rows. Mandatory nonfinite tiles are counted and excluded explicitly. Output-risk differences subtract log2, its constant limit when magnitude/reference=1. These are risk-signal statistics, not raw token-norm measurements.\n\n'
    text+='## Matched50% shared-state output errors\n\n'
    selected=[dict(benchmark=r['benchmark'],split=r['split'],method=r['name'],
        physical=100*r['physical_sparsity'],pv_omission=100*r['pv_omission'],
        exact_pv_mass=100*r['mass'],error=r['relative_error']) for r in rows
        if r['target']==.5 and r['attention_type']=='overall']
    text+=table(selected,['benchmark','split','method','physical','pv_omission','exact_pv_mass','error'])+'\n\n'
    text+='Compensation and zero-PV rows have no physical deletion; their target counts PV replacements. Their exact denominator mass remains100%, while the mass column above measures the part receiving exact PV. They are not equal-work deletion comparisons.\n\n'
    for benchmark,split in (('aime26','calibration'),('longbench_v2','calibration'),('longbench_v2','development')):
        group=[r for r in paired if r['benchmark']==benchmark and r['split']==split and r['target']==.5 and r['attention_type']=='overall']
        for method,reference in (('value','no_value_control'),('mass_value','mass'),('mass_exact','mass'),('contribution_diagnostic','gap_only')):
            point=next(r for r in group if r['candidate']==method and r['reference']==reference)
            if point['relative_error_change'] is not None:
                text+=f"- {benchmark}/{split}: {method} versus {reference}: output-error change {100*point['relative_error_change']:+.2f}%, retained-mass change {100*point['mass_change']:+.2f} percentage points.\n"
            else:
                text+=f'- {benchmark}/{split}: {method} versus {reference}: nonempty repair changes the actual budgets; no pure selection-effect comparison is claimed.\n'
    text+='\nGap-only and the retained-state no-value control coincide on dense shared states because no prior blocks have been deleted; their distinction must be tested during sparse execution. Small one-call error improvements need not survive repeated denoising or improve discrete task accuracy.\n\n'
    text+='## Failure hypotheses and justified next checks\n\n'
    text+='The mass-only versus exact-online-mass comparison tests the looseness of replacing a block exp-sum with count×exp(max). The centered previous-output comparison tests whether a better output estimate fixes the centered rule; it is not automatically an improvement. True contribution ranking uses actual normalized weighted value vectors and captures cancellation/alignment missed by scalar norms, but it already requires PV and is a diagnostic opportunity, not a deployable oracle or an accuracy guarantee.\n\n'
    text+='Prioritize the ongoing end-to-end comparison and its no-value/mass-only controls. If a meaningful gap to contribution ranking survives downstream evaluation, a bounded follow-up can test a low-dimensional estimate of the score-weighted value vector after block softmax, using retained/previous state only. That would be an explicitly new PV-only refinement, with its projected-PV and metadata costs counted; it is not implemented or claimed successful here. Do not expand to another blind pooling Cartesian product.\n\n'
    text+='## Streaming information and overhead\n\n'
    text+=table(compatibility,['method','decision','metadata','extra','information','omission'])+'\n\n'
    text+='For B KV blocks of width64 and value dimension d, a deployed selected pooled norm needs O(B) scalar metadata; centered/mean-compensation methods need O(Bd) means plus radii; aligned weighting needs O(64B) token norms. Computing a norm/mean/radius pass costs O(64Bd); exact p95 adds within-block selection/sorting. The full-valid-KV reference RMS also needs a metadata reduction before routing. Prefix summaries can be reused when content, shape and validity masks match; this reference cache additionally clones/checks prefix V and recomputes after shape/mask changes. Those audit costs are not deployment overhead measurements.\n\n'
    text+='The reference computes full QK, block softmax/PV diagnostics and all pooling metadata even for methods whose deployed rule would not need them. Claimed omission opportunities must account for what selection itself has already computed. There are no custom kernels and **no measured latency, throughput or speedup claims**.\n'
    (dest/'report.md').write_text(text)
    _write(dest/'audit.json',dict(complete=True,fingerprint=execution['fingerprint'],heldout_used=False,
        sources=sources,code_sha256=sha(Path(__file__).read_bytes()),shared_state_rows=len(rows),
        matched_comparisons=len(paired),signal_distributions=len(signals),
        scope='Calibration/development mechanism diagnostics, not final task evaluation'))
    return rows,signals


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args()
    rows,signals=build(args.output);print(json.dumps(dict(shared_state_rows=len(rows),signal_distributions=len(signals))))


if __name__=='__main__':
    main()
