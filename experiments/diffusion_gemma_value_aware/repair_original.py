"""Repair cap1 calibration independently by attention type, preserving traces.

An unattainable LOCAL target must not force GLOBAL lambda to1. Boundary probes
are classified independently; feasible types are then jointly reverified with
the other type fixed at1. This is calibration bookkeeping, not a routing change.
Run only after the initial calibration writer and queued GPU stages finish.
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from dllm.models import create_adapter
from .protocol import ROOT,MODEL,REVISION,prepare,fingerprint,frozen_write,sha
from .evaluate import run_group
from .calibration import aggregate
from .queue import process_state
from .policy_search import is_one
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_fingerprint


def at_one(entry):
    return is_one(entry)


def boundary_plan(trace,target):
    boundary={}
    for kind in ('local','global'):
        points=[r for r in trace if at_one(r['policy'][kind])]
        if not points and min(abs(r['achieved'][kind]-target) for r in trace)>.02:
            raise ValueError(f'missing explicit lambda1 evidence for uncalibrated {kind}')
        boundary[kind]=points[-1] if points else None
    fixed={kind:boundary[kind] is not None and boundary[kind]['achieved'][kind]<target-.02 for kind in boundary}
    policy={}
    for kind in ('local','global'):
        if fixed[kind]:
            policy[kind]=dict(log_threshold=0.,cap_one=True,unattainable=True,
                source='exact lambda1 fallback for this attention type only')
        else:
            point=min(trace,key=lambda r:abs(r['achieved'][kind]-target))
            policy[kind]=dict(point['policy'][kind])
    return policy,fixed,{kind:boundary[kind]['achieved'][kind] if boundary[kind] is not None else None for kind in boundary}


def scale_upper(rows):
    # Prompt IDs already include the official/chat template. Native generation
    # commits fixed256-token canvases; budget plus two canvases is conservative
    # for the in-flight partial last canvas. exp(upper)/L thus saturates at1.
    lengths=[(len(r['prompt_tokens']) if isinstance(r['prompt_tokens'],list) else r['prompt_tokens'])
        +r['generation_budget']+512 for r in rows]
    return math.log(max(lengths))+1e-6


def repair(root=ROOT,max_rounds=6):
    setup=prepare(root);fp=fingerprint(root)
    job=json.loads((root/'job.json').read_text())
    state=process_state(job['pid'])
    if state is not None and state['state'] not in ('Z','X'):raise RuntimeError('another experiment worker is still live; do not overlap GPUs/writers')
    queued=root/'queued_stage.json'
    if queued.exists():
        state=process_state(json.loads(queued.read_text())['pid'])
        if state is not None and state['state'] not in ('Z','X'):
            raise RuntimeError('the queued GPU stage is still live; wait for it to finish')
    adapter=None;repaired=[]
    for path in sorted((root/'verified_policies').glob('*/blasst_original_*.json')):
        old=json.loads(path.read_text());assert old['fingerprint']==fp
        if old.get('independent_cap1_repair') or old.get('selection_version')=='independent_cap_one_v2':repaired.append(str(path));continue
        benchmark=old['benchmark'];target=old['target'];rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
        trace=old['trace'];upper=scale_upper(rows)
        # If a policy already attains both targets, no boundary fallback repair
        # is required. Preserve it verbatim (e.g. a later25% condition).
        if all(abs(old['measured'][k]-target)<=.02 for k in ('local','global')):continue
        policy,fixed,boundary=boundary_plan(trace,target)
        archive=root/'policy_revisions'/'original_cap1_selection_v1'/benchmark/path.name
        frozen_write(archive,old)
        observations=[]
        # Reuse an identical joint boundary even when one type's boundary is
        # already within tolerance (e.g. LongBench global75). Attainability
        # labels differ; the deployed lambda1 operator does not.
        if all(at_one(policy[k]) for k in ('local','global')):
            joint=[r for r in trace if all(at_one(r['policy'][k]) for k in ('local','global'))]
            if not joint:raise ValueError('missing joint lambda1 probe')
            chosen=joint[-1]
            observations=[dict(chosen,policy=policy,reused_exact_lambda1_source=chosen['source_condition'],verification_stage='calibration')]
        else:
            if adapter is None:adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
            for iteration in range(max_rounds):
                condition=f'{path.stem}/{benchmark}/{_fingerprint(policy)[:16]}'
                outputs,failures=run_group(adapter,root,rows,'original_boundary_repair',condition,dict(method='blasst'),policy,fp)
                if failures or len(outputs)!=len(rows):raise RuntimeError(f'incomplete repair verification: {condition}')
                metrics={kind:aggregate([r for d in outputs for r in d['records'] if r['attention_type']==kind and r['probe']=='execution']) for kind in ('local','global')}
                achieved={kind:v['physical_sparsity'] for kind,v in metrics.items()}
                error=max(abs(achieved[k]-target) for k in ('local','global') if not fixed[k])
                observations.append(dict(iteration=iteration,policy=policy,achieved=achieved,error=error,metrics=metrics,
                    source_condition=condition,source_ids=[r['id'] for r in rows],verification_stage='original_boundary_repair'))
                _write(root/'original_boundary_repair_traces'/benchmark/path.name,observations)
                if error<=.02:break
                updated={k:dict(v) for k,v in policy.items()}
                for kind in ('local','global'):
                    if fixed[kind] or abs(achieved[kind]-target)<=.02:continue
                    points=[]
                    for r in trace+observations:
                        entry=r['policy'][kind]
                        x=upper if at_one(entry) else min(upper,entry.get('log_scale',upper))
                        points.append((x,r['achieved'][kind]))
                    below=[x for x,s in points if s<target];above=[x for x,s in points if s>=target]
                    if below and above:value=(max(below)+min(above))/2
                    else:value=min(upper,policy[kind].get('log_scale',upper)+8*(target-achieved[kind]))
                    updated[kind]=dict(log_scale=float(value),cap_one=True,source='independent feasible-type calibration with other unattainable types fixed at1')
                if updated==policy:break
                policy=updated
        best=min(observations,key=lambda r:max((abs(r['achieved'][k]-target) for k in ('local','global') if not fixed[k]),default=0.))
        new=dict(old,policy=best['policy'],measured=best['achieved'],initial_trace=old['trace'],trace=observations,
            selected_round=best.get('iteration',0),verification_stage=best['verification_stage'],
            error=max(abs(v-target) for v in best['achieved'].values()),
            within_two_points=all(abs(v-target)<=.02 for v in best['achieved'].values()),
            within_two_points_on_attainable_types=all(abs(best['achieved'][k]-target)<=.02 for k in ('local','global') if not fixed[k]),
            target_miss={k:abs(v-target)>.02 for k,v in best['achieved'].items()},unattained_in_tested_range=fixed,
            independent_cap1_repair=dict(reason='A cap1 failure for one attention type must not force the other to cap1',
                archive=str(archive),archive_sha256=sha(archive.read_bytes()),boundary_sparsity=boundary,fixed_at_one=fixed,
                selection='minimize error only over attainable types; explicitlambda1 for unattainable type; all selected joint settings verified',
                scale_upper=upper,trace=observations,code_sha256=sha(Path(__file__).read_bytes())))
        _write(path,new);repaired.append(str(path))
    issue=root/'policy_review_issues.json'
    if issue.exists():
        d=json.loads(issue.read_text())
        other=[x for x in d['unresolved'] if x['id']!='original_cap1_cross_type_fallback']
        _write(issue,dict(unresolved=other,resolved=d.get('resolved',[])+[dict(id='original_cap1_cross_type_fallback',time=time.time(),policies=repaired)]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--max-rounds',type=int,default=6)
    args=p.parse_args();repair(args.output,args.max_rounds)


if __name__=='__main__':main()
