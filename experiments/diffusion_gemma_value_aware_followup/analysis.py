"""Shared-state analysis and proposals, using existing calibration helpers.

No final-generation files are read. The only selection inputs are the explicit
calibration manifest; development diagnostics are summarized but never fitted.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from experiments.diffusion_gemma_value_aware.calibration import (
    aggregate,quantile_threshold,select_pooling,fit_blasst,
)
from experiments.diffusion_gemma_value_aware.protocol import frozen_write,sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .engine import check_result,contract
from .protocol import ROOT,TARGETS,prepare


def screen_source(root,row,execution,imports):
    if row['split'] not in ('calibration','development'):
        raise ValueError('screen analysis cannot read a final example')
    path=shard_path(root,'screen','dense',row['id'])
    fp=execution['fingerprint'];entry=None
    if not path.exists():
        entry=imports.get(f'screen/dense/{row["id"]}')
        if entry is None:raise FileNotFoundError(row['id'])
        if row['benchmark']!='aime26' or entry['fingerprint']!=execution['previous_fingerprint']:
            raise ValueError('only matching AIME screens can be imported')
        path=Path(entry['path']);fp=entry['fingerprint']
    raw=path.read_bytes();data=json.loads(raw)
    check_result(data,row,fp,{},None,screen=True)
    arrays_path=path.with_suffix('.npz')
    source=dict(path=str(path),sha256=sha(raw),arrays_path=str(arrays_path),
        arrays_sha256=sha(arrays_path.read_bytes()),split=row['split'])
    if entry and any(entry[k]!=source[k] for k in ('path','sha256','arrays_path','arrays_sha256')):
        raise ValueError('imported calibration/development source changed')
    return data,arrays_path,source


def screen_bundle(root,row,execution,imports):
    # JointScreen intentionally covers refinements/guarded rankings only. The
    # supplemental observer supplies base pools plus the no-value control on
    # exactly the same dense trajectory. Count execution records just once.
    from .supplement import source as supplemental_source
    joint=screen_source(root,row,execution,imports)
    supplemental=supplemental_source(root,row,execution,joint[0])
    return (joint,supplemental)


def summarize(root=ROOT,require_complete=True):
    setup=prepare(root);execution=contract(root)
    imports=json.loads((root/'imported_sources.json').read_text()) if (root/'imported_sources.json').exists() else {}
    grouped=defaultdict(list);parts=defaultdict(list);sources={};missing=[];distributions=[]
    for row in setup['calibration']+setup['development']:
        try:bundle=screen_bundle(root,row,execution,imports)
        except FileNotFoundError:missing.append(row['id']);continue
        seen_keys=set()
        for index,(data,path,source) in enumerate(bundle):
            sources[row['id']+('/supplement' if index else '')]=source
            for record in data['records']:
                if index and record['probe']=='execution':continue
                for kind in ('overall',record['attention_type']):
                    grouped[row['benchmark'],row['split'],record['probe'],kind].append(record)
            distributions.extend(dict(benchmark=row['benchmark'],split=row['split'],id=row['id'],**r)
                for r in data['distributions'])
            if row['split']=='calibration':
                with np.load(path) as arrays:
                    for key in arrays:
                        if key in seen_keys:raise ValueError('duplicate shared-state risk distribution')
                        seen_keys.add(key)
                        name,kind=key.rsplit('__',1)
                        if kind not in ('local','global'):raise ValueError('unknown calibration attention type')
                        values=arrays[key]
                        if values.size==0 or np.isnan(values).any():raise ValueError('invalid calibration margins')
                        parts[row['benchmark'],name,kind].append(values)
    rows=[dict(benchmark=b,split=s,probe=p,attention_type=k,**aggregate(v))
        for (b,s,p,k),v in grouped.items()]
    _write(root/'screen_analysis_audit.json',dict(complete=not missing,missing=missing,sources=sources,
        fingerprint=execution['fingerprint'],heldout_used=False,expected_sources=36))
    if missing and require_complete:raise RuntimeError(f'screen incomplete: {missing}')
    _write(root/('screen_summary_partial.json' if missing else 'screen_summary.json'),rows)
    _write(root/'proxy_distributions.json',distributions)
    if missing:return rows
    policies={}
    for (benchmark,name,kind),arrays in parts.items():
        entry={str(t):quantile_threshold(arrays,t) for t in TARGETS}
        if name=='blasst':
            fitted=fit_blasst(arrays)
            entry=dict(fit=fitted,targets=fitted['targets'])
        policies.setdefault(benchmark,{}).setdefault(name,{})[kind]=entry
    # Reuse the same pooling selection criterion, now on AIME and actual v2
    # calibration only. A changed pooling cannot silently reuse an old policy.
    selection=select_pooling(rows)
    calibration_files=[Path(__file__),
        Path('experiments/diffusion_gemma_value_aware/calibration.py'),
        Path('experiments/diffusion_gemma_aime26/calibration.py')]
    code_sources={str(p):sha(p.read_bytes()) for p in calibration_files}
    frozen_write(root/'threshold_proposals.json',dict(fingerprint=execution['fingerprint'],policies=policies,
        code_sources=code_sources,
        calibration_sources={k:v for k,v in sources.items() if v['split']=='calibration'},
        status='Dense-state proposals only; sparse calibration verification still required'))
    frozen_write(root/'pooling_selection.json',dict(fingerprint=execution['fingerprint'],selected=selection,
        code_sources=code_sources,
        heldout_used=False,source_sha256=sha((root/'screen_summary.json').read_bytes()),
        rule='Existing minimum mean output-error rule across both benchmarks, local/global and50/75 calibration targets; development and finals excluded'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--partial',action='store_true')
    args=p.parse_args();rows=summarize(args.output,not args.partial)
    print(json.dumps(dict(summary_rows=len(rows),partial_allowed=args.partial)))


if __name__=='__main__':main()
