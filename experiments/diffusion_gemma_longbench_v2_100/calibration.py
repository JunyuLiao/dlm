"""New disjoint calibration data, unchanged inverse-L / empirical-rank search."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from experiments.diffusion_gemma_value_aware.protocol import frozen_write,sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware.routing import Attention as DenseObserver
from experiments.diffusion_gemma_value_aware.operators import Config,screen_risks
from experiments.diffusion_gemma_value_aware_followup import calibrate as search,policies
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from .protocol import PARENT,METHODS,TARGETS,CONFIGS
from .runner import cached,generate,backend


def calibration_rows(setup,benchmark='longbench_v2'):
    rows=setup['calibration']
    if benchmark!='longbench_v2' or len(rows)!=12 or len({r['id'] for r in rows})!=12:
        raise ValueError('Expected exactly12 unique LongBench v2 calibration rows')
    if any(r['split']!='calibration' or r['benchmark']!=benchmark for r in rows):
        raise ValueError('Non-calibration input in threshold fitting')
    if {r['id'] for r in rows}&{r['id'] for r in setup['final']+setup['development']}:
        raise ValueError('Calibration overlaps development/final set')
    return rows


def starting_policy(proposals,benchmark,name,config,target):
    """Historical scalars are proposals only; no historical measured point is imported."""
    path=PARENT/'final_configs'/f'{name}_s{int(target*100)}.json'
    prior=json.loads(path.read_text())
    if prior['config']!=config:raise ValueError('Warm-start operator differs')
    result={}
    for kind,entry in prior['thresholds'][benchmark].items():
        if entry.get('cap_one') and entry.get('unattainable'):
            result[kind]=dict(log_threshold=0.,cap_one=True)
        elif 'log_scale' in entry:
            result[kind]=dict(log_scale=entry['log_scale'],cap_one=bool(entry.get('cap_one')))
        else:
            result[kind]=dict(log_threshold=entry['log_threshold'])
            if entry.get('cap_one'):result[kind]['cap_one']=True
        result[kind]['source']='Historical scalar warm start; must reverify on the new12 calibration examples'
    return result


class Observer(DenseObserver):
    """Only the two requested existing risk arrays, without unrelated screen sweeps."""
    def __init__(self,config=None,thresholds=None,validate=False):
        if config or thresholds:raise ValueError('Calibration observer must be dense')
        super().__init__(screen=True)

    def _screen(self,state,meta,q,k,valid,scale,layer,step,kind,prefix,length):
        for name,config in (('mass',Config(method='mass')),('risk_mean',Config(method='risk',pooling='mean'))):
            risks,_=screen_risks(state,meta,config)
            self.risk_arrays.setdefault(f'{name}__{kind}',[]).append(risks[state['eligible']].float().cpu().numpy())


def collect(adapter,root,row,contract):
    if row['split']!='calibration':raise ValueError('Dense risk collection is calibration-only')
    path=shard_path(root,'calibration_dense','dense',row['id'])
    array_path=path.with_suffix('.npz')
    observer_sha=sha(Path(__file__).read_bytes())
    if path.exists():
        out=json.loads(path.read_text());check_result(out,row,contract['fingerprint'],{},None)
        if out['observer_sha256']!=observer_sha or sha(array_path.read_bytes())!=out['arrays_sha256']:
            raise ValueError('Dense calibration observer or arrays changed')
        return out
    if adapter is None:raise FileNotFoundError(path)
    _write(root/'progress.json',dict(pid=os.getpid(),stage='calibration_dense',condition='dense',id=row['id'],started=time.time()))
    observers=[]
    def factory(*args,**kwargs):
        observer=Observer(*args,**kwargs);observers.append(observer);return observer
    with patch.object(backend,'Attention',factory):out=generate(adapter,row)
    arrays=observers[0].arrays()
    if set(arrays)!={'mass__local','mass__global','risk_mean__local','risk_mean__global'}:
        raise ValueError('Missing requested calibration distributions')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=array_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays);temporary.replace(array_path)
    out.update(fingerprint=contract['fingerprint'],observer_sha256=observer_sha,
        arrays_sha256=sha(array_path.read_bytes()),backend='native_dense_with_existing_calibration_risk_observer')
    check_result(out,row,contract['fingerprint'],{},None);_write(path,out)
    _append(root/'completed.jsonl',dict(stage='calibration_dense',condition='dense',id=row['id'],path=str(path),finished=time.time()))
    print('dense calibration collected',row['id'],flush=True)
    return out


def distributions(root,rows,name,config,contract):
    parts={k:[] for k in ('local','global')};sources={}
    for row in rows:
        collect(None,root,row,contract)
        path=shard_path(root,'calibration_dense','dense',row['id']);array_path=path.with_suffix('.npz')
        sources[row['id']]=dict(path=str(path),sha256=sha(path.read_bytes()),arrays_path=str(array_path),
            arrays_sha256=sha(array_path.read_bytes()),split='calibration')
        with np.load(array_path) as arrays:
            for kind in parts:parts[kind].append(arrays[f'{policies.probe_key(name,config)}__{kind}'].astype(np.float64))
    values={k:np.sort(np.concatenate(v)) for k,v in parts.items()}
    if any(np.isnan(v).any() or not np.isfinite(v).any() for v in values.values()):
        raise ValueError('Invalid calibration risk distribution')
    return values,dict(sources=sources,source_ids=[r['id'] for r in rows],heldout_used=False,
        rule='Existing empirical-rank refinement from the new12 dense calibration trajectories only')


def run_group(adapter,root,rows,condition,config,policy,contract):
    if len(rows)!=12 or any(r['split']!='calibration' for r in rows):raise ValueError('Calibration-only group required')
    outputs=[];sources={};failures=[]
    for row in sorted(rows,key=lambda r:len(r['prompt_tokens'])):
        try:
            out=cached(adapter,root,row,'calibration',condition,config,policy,contract)
            path=shard_path(root,'calibration',condition,row['id'])
            sources[row['id']]=dict(path=str(path),sha256=sha(path.read_bytes()),fingerprint=contract['fingerprint'],stage='calibration')
            outputs.append(out)
        except Exception:
            error=dict(stage='calibration',condition=condition,id=row['id'],traceback=traceback.format_exc())
            failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
            import torch
            torch.cuda.empty_cache()
    return outputs,sources,failures


def prepare_policies(adapter,root,setup,contract):
    rows=calibration_rows(setup)
    sources=[Path(__file__),Path(search.__file__),Path(policies.__file__)]
    sources += [Path('experiments/diffusion_gemma_value_aware')/n for n in ('policy_search.py','rank_calibration.py','repair_original.py')]
    sources += [PARENT/'final_configs'/f'{n}_s{int(t*100)}.json' for n in METHODS for t in TARGETS]
    frozen_write(root/'calibration_protocol.json',dict(fingerprint=contract['fingerprint'],calibration_ids=[r['id'] for r in rows],
        sources={str(p):sha(p.read_bytes()) for p in sources},heldout_used=False,max_total_points=3,
        rule='Unchanged inverse-valid-KV-length BLASST search and empirical-CDF mass/risk refinement; joint verified point; independent original BLASST lambda1 boundaries'))
    failures=[]
    for row in sorted(rows,key=lambda r:len(r['prompt_tokens'])):
        try:collect(adapter,root,row,contract)
        except Exception:
            error=dict(stage='dense_calibration',id=row['id'],traceback=traceback.format_exc())
            failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
            import torch
            torch.cuda.empty_cache()
    with patch.object(search,'calibration_rows',calibration_rows),patch.object(search,'run_group',run_group),\
         patch.object(search,'starting_policy',starting_policy),patch.object(search,'distributions',distributions):
        for target in TARGETS:
            for name in METHODS:
                try:
                    trace=root/'calibration_traces/longbench_v2'/f'{name}_s{int(target*100)}.json'
                    count=len(json.loads(trace.read_text())) if trace.exists() else 0
                    search.calibrate_one(adapter,root,setup,contract,None,name,CONFIGS[name],target,'longbench_v2',max(0,3-count))
                except Exception:
                    error=dict(stage='policy',name=name,target=target,traceback=traceback.format_exc())
                    failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
    return failures


def condition(root,setup,contract,name,target):
    if name=='dense':return dict(config={},target=0.,target_metric='physical_sparsity',thresholds={},policy_sources={})
    if name not in METHODS or target not in TARGETS:raise ValueError('Unrequested condition')
    path=root/'verified_policies/longbench_v2'/f'{name}_s{int(target*100)}.json'
    p=json.loads(path.read_text())
    if p['name']!=name or p['target']!=target or p['config']!=CONFIGS[name] or p['heldout_used']:
        raise ValueError('Policy identity/selection mismatch')
    rows=calibration_rows(setup)
    policies.audit_policy(p,rows,CONFIGS[name],contract)
    for point in p['trace']:policies.audit_point(point,rows,CONFIGS[name],contract)
    search_source=p.get('search_provenance')
    if search_source is not None:
        if search_source['heldout_used'] or set(search_source['source_ids'])!={r['id'] for r in rows}:
            raise ValueError('Risk-CDF source contamination')
        if set(search_source['sources'])!={r['id'] for r in rows}:raise ValueError('Incomplete CDF provenance')
        for row in rows:
            source=search_source['sources'][row['id']]
            raw_path=shard_path(root,'calibration_dense','dense',row['id'])
            if source['split']!='calibration' or Path(source['path'])!=raw_path or Path(source['arrays_path'])!=raw_path.with_suffix('.npz'):
                raise ValueError('CDF source is not this calibration row')
            for key,digest in (('path','sha256'),('arrays_path','arrays_sha256')):
                if sha(Path(source[key]).read_bytes())!=source[digest]:raise ValueError('CDF raw source changed')
    return dict(config=CONFIGS[name],target=target,target_metric='physical_sparsity',thresholds={'longbench_v2':p['policy']},
        policy_sources={'longbench_v2':dict(path=str(path),sha256=sha(path.read_bytes()))})
