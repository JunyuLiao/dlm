"""Distribution-scaled threshold refinement for NEW value/mass criteria only.

Their risks can be tightly clustered near1; a generic unit-sized log-threshold
step is inappropriate. Search in the measured dense-calibration CDF rank,
then map back to a scalar threshold. BLASST's established inverse-L calibration
is not changed. No final/development data participate in this search.
"""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from .protocol import sha
from .run import shard_path
from .calibration import quantile_threshold
from .refinements import CONFIGS as REFINEMENTS


def resume_observations(prior,trace_path):
    """Detach mutable work from the immutable prior; prefer saved extensions.

    A worker may finish verification and checkpoint its trace, then fail while
    publishing the policy. Those completed points must not be inferred again.
    """
    saved=prior['trace'] if prior else []
    if trace_path.exists():
        checkpoint=json.loads(trace_path.read_text())
        if saved and checkpoint[:len(saved)]!=saved:
            raise ValueError('checkpoint is not an extension of the existing policy history')
        saved=checkpoint
    return deepcopy(saved)


def distributions(root,rows,name,config):
    stage='refinement_screen' if name in REFINEMENTS else 'screen'
    key=name if name in REFINEMENTS else (f'{name}_{config["pooling"]}' if name in ('value','mass_value','risk') else name)
    parts={k:[] for k in ('local','global')};sources={}
    for row in rows:
        if row['split']!='calibration':raise ValueError('only calibration rows may select thresholds')
        path=shard_path(root,stage,'dense',row['id']).with_suffix('.npz')
        with np.load(path) as arrays:
            for kind in parts:parts[kind].append(arrays[f'{key}__{kind}'].astype(np.float64))
        sources[str(path)]=sha(path.read_bytes())
    values={k:np.sort(np.concatenate(v)) for k,v in parts.items()}
    if any(np.isnan(v).any() or not np.isfinite(v).any() for v in values.values()):
        raise ValueError('invalid dense calibration risk distribution')
    return values,dict(source_stage=stage,source_ids=[r['id'] for r in rows],sources=sources,
        code_sha256=sha(Path(__file__).read_bytes()),
        rule='calibration-only physical-risk empirical CDF; bracket/secant in rank coordinates, else bounded rank residual update; map to strict-less-than FP32 quantile; keep types already within2pp unchanged')


def cdf(sorted_values,threshold):
    return float(np.searchsorted(sorted_values,threshold,side='left')/len(sorted_values))


def verify_resume(root,observations,rows,config,fp):
    """A completed trace point must still have all its matching raw shards."""
    for point in observations:
        if set(point['source_ids'])!={r['id'] for r in rows}:raise ValueError('resume calibration IDs mismatch')
        for row in rows:
            path=shard_path(root,'calibration',point['source_condition'],row['id'])
            out=json.loads(path.read_text())
            if out['fingerprint']!=fp or out['config']!=config or out['thresholds']!=point['policy']:
                raise ValueError('resume calibration operator/threshold mismatch')
            if any(out[k]!=row[k] for k in ('id','prompt_hash','seed','generation_budget')):
                raise ValueError('resume calibration prompt/seed mismatch')


def next_policy(observations,target,values,tolerance=.02):
    current=observations[-1];updated={}
    for kind in ('local','global'):
        entry=dict(current['policy'][kind]);actual=current['achieved'][kind]
        if abs(actual-target)<=tolerance:updated[kind]=entry;continue
        array=values[kind];ceiling=np.isfinite(array).sum()/len(array)
        points=[(cdf(array,r['policy'][kind]['log_threshold']),r['achieved'][kind]) for r in observations]
        below=sorted((r,s) for r,s in points if s<target)
        above=sorted((r,s) for r,s in points if s>=target)
        if below and above and below[-1][0]<above[0][0]:
            lo,ls=below[-1];hi,hs=above[0]
            fraction=np.clip((target-ls)/(hs-ls),.1,.9)
            rank=lo+float(fraction)*(hi-lo)
        else:
            rank=cdf(array,entry['log_threshold'])+float(np.clip(target-actual,-.25,.25))
        rank=float(np.clip(rank,0.,ceiling))
        entry['log_threshold']=quantile_threshold([array],rank)['log_threshold']
        updated[kind]=entry
    return updated
