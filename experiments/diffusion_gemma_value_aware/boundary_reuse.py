"""Reuse a fully verified joint-lambda1 calibration for an unattainable target.

Only target metadata changes. The full attention/generation configuration and
effective local/global thresholds must match; no new inference or fitting occurs.
This never applies to a feasible type or a non-boundary source checkpoint.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from .protocol import ROOT,TARGETS,prepare,fingerprint,frozen_write,sha
from .execution import assert_policy_provenance,provenance
from .policy_audit import point_for,verify_measurement
from .report import inspect_output
from .repair_original import scale_upper
from .run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def proves_exact_one(entry,upper):
    """Prove the actual routing.py expression is zero for every valid length.

    A lambda_at_one annotation alone is not proof: the router does not use it.
    upper conservatively bounds log(valid KV length) for this generation setup.
    """
    key='log_scale' if 'log_scale' in entry else 'log_threshold'
    value=entry.get(key)
    if not entry.get('cap_one') or value is None or not math.isfinite(value):return False
    return bool(entry.get('unattainable') or value >= (upper if key=='log_scale' else 0.))


def retarget(source,target,metrics,upper,source_path,source_hash):
    if source['name']!='blasst_original' or source['config']!=dict(method='blasst') or source['heldout_used']:
        raise ValueError('only the unchanged original BLASST calibration can be reused')
    if target not in TARGETS:raise ValueError('unsupported target')
    point=point_for(source)
    if not all(proves_exact_one(source['policy'][k],upper) for k in ('local','global')):
        raise ValueError('source is not a proven joint-lambda1 boundary')
    measured={k:metrics[k]['physical_sparsity'] for k in ('local','global')}
    if not all(v<target-.02 for v in measured.values()):
        raise ValueError('reuse requires both types to be unattainable at the boundary')
    stage=point.get('verification_stage',source.get('verification_stage','calibration'))
    if stage not in ('calibration','original_boundary_repair'):
        raise ValueError('boundary evidence cannot come from development or final outputs')
    policy={k:dict(log_threshold=0.,cap_one=True,unattainable=True,
        source='reused verified joint-lambda1 boundary; no new inference') for k in ('local','global')}
    error=max(abs(v-target) for v in measured.values())
    trace=[dict(iteration=0,policy=policy,achieved=measured,error=error,metrics=metrics,
        source_condition=point['source_condition'],source_ids=deepcopy(source['calibration_ids']),
        verification_stage=stage,reused_exact_lambda1_source=point['source_condition'])]
    return dict(fingerprint=source['fingerprint'],benchmark=source['benchmark'],name='blasst_original',
        target=target,config=deepcopy(source['config']),policy=policy,measured=measured,error=error,
        cap_one_unattainable={'local':True,'global':True},selection_version='independent_cap_one_v2',
        search_version='verified_joint_boundary_reuse_v1',within_two_points=False,
        unattained_in_tested_range={'local':True,'global':True},trace=trace,selected_round=0,
        calibration_ids=deepcopy(source['calibration_ids']),heldout_used=False,metric='physical deleted tiles',
        verification_stage=stage,boundary_reuse=dict(source_policy=str(source_path),source_policy_sha256=source_hash,
            source_target=source['target'],source_selected_round=source['selected_round'],
            valid_log_length_upper=upper,code_sha256=sha(Path(__file__).read_bytes()),
            explanation='Same complete model/prompt/seed/generation/attention configuration and effective lambda1 for both types. New target is a label, not a new execution or achieved-sparsity claim.'))


def reuse(root=ROOT,target=.9,source_target=.75):
    setup=prepare(root);fp=fingerprint(root);results=[];sources={}
    for benchmark in ('aime26','longbench'):
        path=root/'verified_policies'/benchmark/f'blasst_original_s{int(source_target*100)}.json'
        source=json.loads(path.read_text());assert_policy_provenance(source,dict(method='blasst'),fp)
        if source['benchmark']!=benchmark or source['target']!=source_target:raise ValueError('source policy identity')
        rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
        if not rows or any(r['split']!='calibration' for r in rows):raise ValueError('calibration-only source required')
        point=point_for(source);stage=point.get('verification_stage',source.get('verification_stage','calibration'))
        if stage not in ('calibration','original_boundary_repair'):raise ValueError('invalid source stage')
        upper=scale_upper(rows);outputs=[];sources[str(path)]=sha(path.read_bytes())
        for row in rows:
            file=shard_path(root,stage,point['source_condition'],row['id']);out=json.loads(file.read_text())
            condition=dict(config=source['config'],thresholds={benchmark:out['thresholds']},**provenance(source['config']))
            _,_,errors=inspect_output(out,row,fp,condition)
            if errors:raise ValueError(f'raw boundary source {row["id"]}: {errors}')
            if not all(proves_exact_one(out['thresholds'][k],upper) for k in ('local','global')):
                raise ValueError('raw source does not execute exactlambda1 for both types')
            outputs.append(out);sources[str(file)]=sha(file.read_bytes())
        metrics=verify_measurement(source,outputs,rows)
        policy=retarget(source,target,metrics,upper,path,sources[str(path)])
        verify_measurement(policy,outputs,rows)
        dest=root/'verified_policies'/benchmark/f'blasst_original_s{int(target*100)}.json'
        frozen_write(dest,policy);sources[str(dest)]=sha(dest.read_bytes())
        results.append(dict(benchmark=benchmark,n=len(rows),policy=str(dest),measured=policy['measured'],
            unattainable={'local':True,'global':True},source_condition=point['source_condition']))
    audit=dict(complete=True,heldout_used=False,inference_runs=0,fingerprint=fp,target=target,
        source_target=source_target,measurements=results,sources=sources)
    _write(root/'boundary_reuse_audit.json',audit)
    return audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--target',type=float,default=.9);parser.add_argument('--source-target',type=float,default=.75)
    args=parser.parse_args();audit=reuse(args.output,args.target,args.source_target)
    print(json.dumps({k:audit[k] for k in ('complete','inference_runs','measurements')},indent=2))


if __name__=='__main__':main()
