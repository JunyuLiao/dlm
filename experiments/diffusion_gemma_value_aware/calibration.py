"""Calibration-only screening summaries and independent threshold proposals.

This file never reads final-generation shards or held-out scores. Threshold
proposals are NOT frozen policies until verified on sparse trajectories.
"""
from collections import defaultdict
from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import numpy as np
from experiments.diffusion_gemma_aime26.calibration import fit as fit_blasst
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .protocol import ROOT, TARGETS, prepare, fingerprint, frozen_write, sha
from .routing import screen_configs
from .run import shard_path

FIELDS=('eligible','skipped','softmax_skipped','pv_omitted','compensated','mass_sum','rows','error_sq','dense_sq',
    'denominator_mass_sum','same_budget_best_mass_sum','wrongly_replaced_vs_mass','unnecessarily_retained_vs_mass')


def aggregate(records):
    sums={k:sum(r.get(k,0.) for r in records) for k in FIELDS}
    return dict(sums,physical_sparsity=sums['skipped']/max(sums['eligible'],1),
        pv_omission=sums['pv_omitted']/max(sums['eligible'],1),
        mass=sums['mass_sum']/max(sums['rows'],1),
        denominator_mass=sums['denominator_mass_sum']/max(sums['rows'],1),
        relative_error=math.sqrt(sums['error_sq']/max(sums['dense_sq'],1e-30)))


def quantile_threshold(parts, target):
    """Strict < comparison, true physical-count weighting, finite boundary."""
    values=np.sort(np.concatenate(parts).astype(np.float64))
    if len(values)==0 or np.isnan(values).any(): raise ValueError('empty/NaN calibration risk')
    finite=values[np.isfinite(values)]
    if len(finite)==0: raise ValueError('no skippable calibration blocks')
    ceiling=len(finite)/len(values)
    index=min(len(finite)-1,int(math.floor(target*len(values))))
    point=finite[index]
    # Metadata/risk comparisons are FP32; choose adjacent representable float32.
    above=float(np.nextafter(np.float32(point),np.float32(np.inf)))
    choices=[float(point),above,float(finite[-1]+1)] if target>=ceiling else [float(point),above]
    selected=min(choices,key=lambda x:abs(np.searchsorted(values,x,side='left')/len(values)-target))
    return dict(log_threshold=selected,target=target,
        achieved_dense_sparsity=float(np.searchsorted(values,selected,side='left')/len(values)),
        eligible=len(values),finite_candidates=len(finite),dense_ceiling=ceiling,
        unattainable=target>ceiling,comparison='skip strictly below threshold; retain ties')


def screen_summary(root=ROOT, require_complete=True):
    setup=prepare(root); fp=fingerprint(root); records=[]; parts=defaultdict(list); provenance={}; missing=[]
    distributions=[]
    for row in setup['calibration']+setup['development']:
        path=shard_path(root,'screen','dense',row['id'])
        if not path.exists(): missing.append(row['id']);continue
        result=json.loads(path.read_text())
        if result['fingerprint']!=fp or not result['screen']: raise RuntimeError(f'incompatible screen {path}')
        assert all(result[k]==row[k] for k in ('id','prompt_hash','seed','generation_budget'))
        provenance[row['id']]=dict(path=str(path),sha256=sha(path.read_bytes()),split=row['split'])
        for r in result['records']:
            records.append(dict(benchmark=row['benchmark'],split=row['split'],id=row['id'],**r))
        distributions.extend(dict(benchmark=row['benchmark'],split=row['split'],id=row['id'],**r) for r in result['distributions'])
        if row['split']=='calibration':
            with np.load(path.with_suffix('.npz')) as arrays:
                for key in arrays:
                    name,kind=key.split('__');parts[row['benchmark'],name,kind].append(arrays[key])
    if missing and require_complete: raise RuntimeError(f'screen incomplete: {missing}')
    grouped=defaultdict(list)
    for r in records:
        for kind in ('overall',r['attention_type']): grouped[r['benchmark'],r['split'],r['probe'],kind].append(r)
    summary=[dict(benchmark=b,split=s,probe=p,attention_type=k,**aggregate(rows)) for (b,s,p,k),rows in grouped.items()]
    _write(root/'screen_summary.json',summary)
    _write(root/'proxy_distributions.json',distributions)
    _write(root/'screen_audit.json',dict(complete=not missing,missing=missing,fingerprint=fp,sources=provenance))
    if missing: return summary
    policies={}
    for (benchmark,name,kind),arrays in parts.items():
        policies.setdefault(benchmark,{}).setdefault(name,{})[kind]={str(t):quantile_threshold(arrays,t) for t in TARGETS}
        if name=='blasst':
            # Reuse paper fit + physical-margin monotonic correction verbatim.
            fit=fit_blasst(arrays)
            policies[benchmark][name][kind]={'fit':fit,'targets':fit['targets']}
    frozen_write(root/'threshold_proposals.json',dict(fingerprint=fp,provenance=provenance,policies=policies,
        status='dense-calibration proposals; sparse verification required before final evaluation'))
    selection=select_pooling(summary)
    frozen_write(root/'pooling_selection.json',dict(fingerprint=fp,selected=selection,
        rule='For each family, minimum mean relative output error across benchmarks, local/global, targets50/75 on calibration only; tie by pooling name',
        heldout_used=False,source_sha=sha((root/'screen_summary.json').read_bytes())))
    return summary


def select_pooling(summary):
    selected={}
    for family in ('value','mass_value','risk'):
        candidates={}
        for pool in ('max','mean','rms','p95','vector_mean'):
            prefix=f'screen/{family}_{pool}/'
            rows=[r for r in summary if r['split']=='calibration' and r['attention_type'] in ('local','global')
                and r['probe'].startswith(prefix) and r['probe'].split('/')[-1] in ('s50','s75')]
            if len(rows)!=8: raise ValueError(f'expected both benchmarks/types/targets for {family}/{pool}, got {len(rows)}')
            candidates[pool]=float(np.mean([r['relative_error'] for r in rows]))
        best=min(candidates,key=lambda p:(candidates[p],p))
        selected[family]=dict(pooling=best,mean_error=candidates[best],all_pooling_errors=candidates)
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--partial',action='store_true')
    args=p.parse_args();screen_summary(args.output,not args.partial)


if __name__=='__main__':main()
