"""Prove frozen policies from selected raw calibration shards, not flags alone."""
import argparse
import json
import math
from pathlib import Path
from .protocol import ROOT,TARGETS,prepare,fingerprint,sha
from .execution import assert_policy_provenance,provenance
from .run import shard_path
from .calibration import aggregate
from .policy_search import is_one
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def point_for(policy):
    points=[p for p in policy['trace'] if p['iteration']==policy['selected_round']]
    if len(points)!=1 or points[0]['policy']!=policy['policy']:
        raise ValueError('canonical policy is not the selected calibration point')
    return points[0]


def verify_measurement(policy,outputs,rows):
    point=point_for(policy)
    if len(outputs)!=len(rows) or {r['id'] for r in outputs}!={r['id'] for r in rows}:
        raise ValueError('incomplete or duplicated calibration sample set')
    if set(policy['calibration_ids'])!={r['id'] for r in rows} or policy['heldout_used']:
        raise ValueError('calibration manifest contamination')
    manifest={r['id']:r for r in rows}
    upper=math.log(max((len(r['prompt_tokens']) if isinstance(r['prompt_tokens'],list) else r['prompt_tokens'])
        +r['generation_budget']+512 for r in rows))+1e-6
    for out in outputs:
        row=manifest[out['id']]
        for key in ('prompt_hash','seed','generation_budget'):
            if out[key]!=row[key]:raise ValueError(f'calibration shard {key} mismatch')
        if out['fingerprint']!=policy['fingerprint'] or out['config']!=policy['config']:
            raise ValueError('calibration operator mismatch')
        if out.get('refinement_sha256')!=policy.get('refinement_sha256'):
            raise ValueError('calibration refinement source mismatch')
        if out['thresholds']!=policy['policy']:
            # The historical repair may store exact1 under a clearer equivalent
            # representation. This exception never permits a non-boundary value.
            reused=point.get('reused_exact_lambda1_source')
            if not reused or not all(is_one(out['thresholds'][k],upper) and is_one(policy['policy'][k],upper) for k in ('local','global')):
                raise ValueError('selected calibration threshold mismatch')
    metrics={kind:aggregate([r for out in outputs for r in out['records']
        if r['probe']=='execution' and r['attention_type']==kind]) for kind in ('local','global')}
    field='pv_omission' if policy['config']['method'] in ('compensate','zero_pv') else 'physical_sparsity'
    for kind in metrics:
        measured=metrics[kind][field]
        if abs(measured-policy['measured'][kind])>1e-8 or abs(measured-point['achieved'][kind])>1e-8:
            raise ValueError(f'{kind} policy sparsity does not match raw eligible/skipped counts')
    if policy['name']=='blasst_original':
        if not all(e.get('cap_one') for e in policy['policy'].values()):
            raise ValueError('original BLASST must cap lambda at1')
        fixed=policy.get('cap_one_unattainable') or policy.get('independent_cap1_repair',{}).get('fixed_at_one',{})
        for kind,unattainable in fixed.items():
            if unattainable and not is_one(policy['policy'][kind],upper):
                raise ValueError('unattainable type must deploy exact lambda1')
    return metrics


def audit(root=ROOT,names=None,targets=TARGETS):
    if not names:raise ValueError('explicit candidate/reference names required')
    from .evaluate import candidate_configs
    from .report import inspect_shard
    setup=prepare(root);fp=fingerprint(root);configs=candidate_configs(root)
    missing=[];violations=[];sources={};verified=[]
    for name in names:
        for target in targets:
            for benchmark in ('aime26','longbench'):
                path=root/'verified_policies'/benchmark/f'{name}_s{int(target*100)}.json'
                if not path.exists():missing.append(str(path));continue
                try:
                    policy=json.loads(path.read_text());assert_policy_provenance(policy,configs[name],fp)
                    if policy['name']!=name or policy['benchmark']!=benchmark or policy['target']!=target:
                        raise ValueError('policy path and identity mismatch')
                    point=point_for(policy);rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
                    stage=point.get('verification_stage',policy.get('verification_stage','calibration'))
                    if stage not in ('calibration','original_boundary_repair'):
                        raise ValueError('policy cannot be verified from development or final generations')
                    outputs=[];sources[str(path)]=sha(path.read_bytes())
                    for row in rows:
                        shard=shard_path(root,stage,point['source_condition'],row['id'])
                        if not shard.exists():missing.append(str(shard));continue
                        out=json.loads(shard.read_text())
                        condition=dict(config=policy['config'],thresholds={benchmark:out['thresholds']},**provenance(policy['config']))
                        # Structural/count/finite coverage is checked in addition
                        # to the policy/trajectory comparison below.
                        _,_,errors=inspect_shard(shard,row,fp,condition)
                        if errors:raise ValueError(f'raw calibration shard audit: {errors}')
                        outputs.append(out);sources[str(shard)]=sha(shard.read_bytes())
                    if len(outputs)!=len(rows):continue
                    metrics=verify_measurement(policy,outputs,rows)
                    verified.append(dict(name=name,target=target,benchmark=benchmark,n=len(rows),metrics=metrics,
                        calibration_source_stage=stage,selected_source_condition=point['source_condition']))
                except (AssertionError,KeyError,ValueError,RuntimeError) as error:
                    violations.append(dict(path=str(path),error=str(error)))
    expected=len(names)*len(targets)*2
    result=dict(complete=not missing and not violations and len(verified)==expected,fingerprint=fp,
        names=names,targets=list(targets),expected_policies=expected,verified_policies=len(verified),
        missing=missing,violations=violations,sources=sources,measurements=verified,heldout_used=False)
    _write(root/'policy_audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--targets',nargs='+',type=float,default=list(TARGETS));args=p.parse_args()
    result=audit(args.output,args.methods,args.targets)
    print(json.dumps({k:result[k] for k in ('complete','verified_policies','expected_policies','missing','violations')},indent=2))


if __name__=='__main__':main()
