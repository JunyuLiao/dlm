"""CPU-only mechanistic diagnosis of completed shared-state screens.

Reads calibration/development data only, never held-out scores. This is not an
end-to-end accuracy report. Aligned risk-array differences can recover exact
block-metadata ratios for criteria whose row-independent factor commutes with
the all-row maximum.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .protocol import ROOT,prepare,sha
from .run import shard_path
from .report import csv_write,table
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def ratio_from_risks(numerator,denominator):
    if numerator.shape!=denominator.shape:raise ValueError('unaligned physical risk arrays')
    valid=np.isfinite(numerator)&np.isfinite(denominator)
    return np.exp(numerator[valid].astype(np.float64)-denominator[valid].astype(np.float64))


def mechanisms(root=ROOT):
    setup=prepare(root);summary=json.loads((root/'screen_summary.json').read_text())
    audit=json.loads((root/'screen_audit.json').read_text());assert audit['complete']
    metadata=defaultdict(list);norm_sources={}
    for row in setup['calibration']:
        path=shard_path(root,'screen','dense',row['id']).with_suffix('.npz')
        norm_sources[str(path)]=sha(path.read_bytes())
        with np.load(path) as arrays:
            for kind in ('local','global'):
                mass=arrays[f'mass__{kind}']
                for pooling in ('max','mean','rms','p95','vector_mean'):
                    metadata[row['benchmark'],kind,pooling+'_over_ref'].append(
                        ratio_from_risks(arrays[f'mass_value_{pooling}__{kind}'],mass))
                metadata[row['benchmark'],kind,'radius_over_ref'].append(ratio_from_risks(arrays[f'compensate__{kind}'],mass))
    ratios=[]
    for (benchmark,kind,name),parts in metadata.items():
        v=np.concatenate(parts)
        q=np.quantile(v,[0,.01,.1,.5,.9,.99,1])
        ratios.append(dict(benchmark=benchmark,attention_type=kind,quantity=name,count=len(v),mean=float(v.mean()),
            std=float(v.std()),**{key:float(x) for key,x in zip(('min','p01','p10','median','p90','p99','max'),q)}))
    _write(root/'value_metadata_diagnosis.json',dict(ratios=ratios,sources=norm_sources,
        derivation='exp(max_q(log alpha_q + log metadata_factor)-max_q(log alpha_q)); factor is block/head-specific and row-independent',
        exclusion='mandatory first-support physical tiles have infinity risks and are excluded from the ratio reconstruction',
        model_mechanism='scale-free per-token V RMSNorm; norm-of-mean and radius retain directional/coherence information'))
    csv_write(root/'value_metadata_diagnosis.csv',ratios)
    comparisons=[]
    for r in summary:
        if r['split']!='calibration' or r['probe']=='execution':continue
        suffix=r['probe'].split('/')[-1]
        reference=next((x for x in summary if x['split']==r['split'] and x['benchmark']==r['benchmark']
            and x['attention_type']==r['attention_type'] and x['probe']=='screen/blasst/'+suffix),None)
        if reference is None:continue
        comparisons.append(dict(benchmark=r['benchmark'],attention_type=r['attention_type'],probe=r['probe'],
            physical_sparsity=r['physical_sparsity'],pv_omission=r['pv_omission'],mass=r['mass'],
            relative_error=r['relative_error'],blasst_relative_error=reference['relative_error'],
            relative_error_ratio=r['relative_error']/reference['relative_error'] if reference['relative_error'] else None,
            mass_delta=r['mass']-reference['mass'],same_budget_best_mass=r['same_budget_best_mass_sum']/max(r['rows'],1),
            unnecessarily_retained_fraction=r['unnecessarily_retained_vs_mass']/max(r['eligible'],1),
            wrongly_replaced_fraction=r['wrongly_replaced_vs_mass']/max(r['eligible'],1),
            work_type='PV replacement' if r['probe'].startswith(('screen/compensate/','screen/zero_pv/')) else 'deletion'))
    csv_write(root/'screen_comparisons.csv',comparisons)
    distributions=json.loads((root/'proxy_distributions.json').read_text());groups=defaultdict(list)
    for r in distributions:
        if r['split']=='calibration':groups[r['benchmark'],r['attention_type'],r['value_aware']].append(r)
    gaussian=[]
    for (b,k,value),rs in groups.items():
        n=sum(r['count'] for r in rs)
        entry=dict(benchmark=b,attention_type=k,value_aware=value,count=n,
            mean_skew=sum(r['count']*r['skew'] for r in rs)/n,
            mean_fourth_moment=sum(r['count']*r['fourth_moment'] for r in rs)/n)
        for target in (25,50,75,90):
            p=f'sol/{"value" if value else "plain"}/gaussian/s{target}'
            entry[f'actual_s{target}']=next(r['physical_sparsity'] for r in summary if r['benchmark']==b
                and r['attention_type']==k and r['split']=='calibration' and r['probe']==p)
        gaussian.append(entry)
    csv_write(root/'sol_distribution_diagnosis.csv',gaussian)
    _write(root/'sol_distribution_diagnosis.json',dict(rows=gaussian,
        caveat='Historical screen: nonempty repair uses exact attention mass, not proxy-only rescue. Consult initial_ranking_fallback_audit and the corrected guarded_ranking_screen_summary. Moments are correlated per-row-standardized descriptors (Gaussian skew0/fourth moment3), not an independent-sample normality test; achieved skips include these historical guards.'))
    selected=[r for r in comparisons if r['attention_type']=='overall' and r['probe'].split('/')[-1]=='s50'
        and (r['probe'].startswith('oracle/') or r['probe'] in ('screen/blasst/s50','screen/value_vector_mean/s50','screen/mass/s50','screen/centered/s50','screen/compensate/s50'))]
    text='# Mechanistic diagnosis from the completed calibration screen\n\n'
    text+='**Historical-screen caveat:** Sol nonempty repair here used exact attention mass, not only the proxy. See `initial_ranking_fallback_audit.json` for its frequency and `guarded_ranking_screen_summary.csv` for corrected signal-only routing after the queued replay. Main streaming BLASST/value calibration is unaffected.\n\n'
    text+='No held-out accuracy is used here. These are matched-budget **dense-state ranking diagnostics**, not independently thresholded streaming runs.\n\n'
    text+=table(selected,['benchmark','probe','work_type','physical_sparsity','pv_omission','mass','relative_error','relative_error_ratio','wrongly_replaced_fraction'])+'\n\n'
    text+='## Value metadata\n\n'
    text+=table(ratios,['benchmark','attention_type','quantity','mean','std','p01','median','p99'])+'\n\n'
    text+='The model applies scale-free RMSNorm to each token value vector. Near-unit norm/ref ratios make magnitude-only additions largely redundant with the score criterion. Norm of the mean vector and radius remain directional/coherence signals; neither is the average token norm. Reconstruction excludes mandatory first-support tiles with infinite risk, so it is not a complete unconditional distribution.\n\n'
    text+='## Sol proxy distribution\n\n'
    text+=table(gaussian,['benchmark','attention_type','value_aware','mean_skew','mean_fourth_moment','actual_s25','actual_s50','actual_s75','actual_s90'])+'\n\n'
    text+='Gaussian thresholds are unchanged. Value-aware Sol here adds log RMS V, which should be nearly a tile-independent shift after value normalization and hence disappear under row standardization. The Gaussian approximation is assessed descriptively, without claiming independent samples or a formal normality test.\n'
    (root/'mechanism_diagnosis.md').write_text(text)
    return ratios,comparisons,gaussian


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args();mechanisms(args.output)


if __name__=='__main__':main()
