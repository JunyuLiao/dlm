"""Sparse calibration and frozen full evaluation, reusing the generation engine.

The initial sweep is limited to targets50/75 and one screened pooling per
family. Other targets/final candidates are explicit later stages, not an
automatic Cartesian product. No final score can affect these calibrations.
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append,_fingerprint
from .protocol import ROOT,MODEL,REVISION,AIME,TARGETS,prepare,fingerprint,frozen_write,sha
from .operators import Config
from .routing import screen_configs
from .calibration import aggregate,quantile_threshold,screen_summary
from .run import cache,shard_path,smoke
from .execution import CONFIGS as REFINEMENTS,execution_cache,provenance,require_refinement_smoke,assert_policy_provenance
from .policy_search import next_policy,select_point
from .ranking_guards import GUARD
from .rank_calibration import distributions as rank_distributions,next_policy as rank_next,verify_resume,resume_observations


def candidate_configs(root):
    selection=json.loads((root/'pooling_selection.json').read_text())['selected']
    c={'blasst_original':dict(method='blasst'),'blasst_aggressive':dict(method='blasst'),
       'mass':dict(method='mass'),'aligned':dict(method='aligned'),'centered':dict(method='centered'),
       'compensate':dict(method='compensate'),'zero_pv':dict(method='zero_pv')}
    c.update({name:dict(method=name,pooling=value['pooling']) for name,value in selection.items()})
    c.update(REFINEMENTS)
    return c


def starting_policy(root,benchmark,name,config,target):
    if name in REFINEMENTS:
        p=json.loads((root/'refinement_threshold_proposals.json').read_text())
        assert p['fingerprint']==fingerprint(root) and p['refinement_sha256']==provenance(config)['refinement_sha256']
        return {kind:dict(log_threshold=p['policies'][benchmark][name][kind][str(target)]['log_threshold'],
            source='independent calibration-only refined dense-state risk quantile',
            proposal_sha256=sha((root/'refinement_threshold_proposals.json').read_bytes())) for kind in ('local','global')}
    proposal=json.loads((root/'threshold_proposals.json').read_text())['policies'][benchmark]
    key='blasst' if name.startswith('blasst') else (
        f'{name}_{config["pooling"]}' if name in ('value','mass_value','risk') else name)
    if name=='blasst_aggressive' and benchmark=='aime26':
        prior=json.loads((AIME/'policy.json').read_text())['targets'][str(target)]
        return {kind:dict(log_scale=value,source='existing AIME26 refined aggressive policy',
            source_sha256=sha((AIME/'policy.json').read_bytes())) for kind,value in prior['log_scales'].items()}
    if name.startswith('blasst'):
        return {kind:dict(log_scale=proposal[key][kind]['targets'][str(target)]['log_scale'],
            cap_one=name=='blasst_original',source='existing exponential fit plus dense-margin correction on new-setting calibration') for kind in ('local','global')}
    return {kind:dict(log_threshold=proposal[key][kind][str(target)]['log_threshold'],
        dense_unattainable=proposal[key][kind][str(target)]['unattainable']) for kind in ('local','global')}


def run_group(adapter,root,rows,stage,name,config,policy,fp):
    outputs=[];failures=[]
    for row in sorted(rows,key=lambda r:(r['generation_budget'],r['id'])):
        try:
            path=shard_path(root,stage,name,row['id'])
            output=execution_cache(config)(adapter,row,path,fp,config,policy,root=root)
            for key in ('id','prompt_hash','seed','generation_budget'):
                if output[key]!=row[key]:raise ValueError(f'cached output {key} mismatch')
            outputs.append(output)
        except Exception:
            failure=dict(id=row['id'],stage=stage,condition=name,traceback=traceback.format_exc())
            failures.append(failure);_append(root/'failures.jsonl',failure);print(failure,flush=True)
        finally: torch.cuda.empty_cache()
    return outputs,failures


def calibrate(root,names,targets,max_rounds):
    setup=prepare(root);fp=fingerprint(root)
    # Complete shared states/proposals required, no partial selection allowed.
    screen_summary(root)
    configs=candidate_configs(root);names=names or [n for n in configs if n not in REFINEMENTS]
    if not set(names)<=set(configs): raise ValueError('unknown candidate')
    require_refinement_smoke(root,fp,[configs[n] for n in names])
    protocol=dict(fingerprint=fp,names=names,targets=targets,max_rounds=max_rounds,
        extra_provenance={n:provenance(configs[n]) for n in names},
        search_version='empirical_rank_v1',max_rounds_semantics='new verification points, preserving/reusing already completed observations',
        rule='empirical dense-calibration rank coordinates for new scalar risks; unchanged inverse-length BLASST fitting; independent cap-one fallback; no downstream accuracy selection')
    tag=_fingerprint(protocol)[:16];frozen_write(root/'calibration_jobs'/f'{tag}.json',protocol)
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    smoke(adapter,setup,root)
    for name in names:
        for target in targets:
            for benchmark in ('aime26','longbench'):
                final=root/'verified_policies'/benchmark/f'{name}_s{int(target*100)}.json'
                prior=None;archive=None
                if final.exists():
                    prior=json.loads(final.read_text());assert_policy_provenance(prior,configs[name],fp)
                    # Preserve complete good calibrations and already tested new
                    # searches. Revisit only bad NEW-risk legacy log-step fits.
                    if name.startswith('blasst') or prior['error']<=.02 or prior.get('search_version')=='empirical_rank_v1':continue
                rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
                from .repair_original import scale_upper
                upper=scale_upper(rows)
                values=search_source=None
                if not name.startswith('blasst'):values,search_source=rank_distributions(root,rows,name,configs[name])
                trace_path=root/'calibration_traces'/benchmark/f'{name}_s{int(target*100)}.json'
                observations=resume_observations(prior,trace_path)
                if observations:
                    verify_resume(root,observations,rows,configs[name],fp)
                    trace_hash=_fingerprint(observations)[:16]
                    frozen_write(root/'calibration_trace_revisions'/benchmark/f'{name}_s{int(target*100)}_{trace_hash}.json',observations)
                if prior:
                    archive=root/'policy_revisions'/'empirical_rank_v1'/benchmark/f'{final.stem}_{sha(final.read_bytes())[:16]}.json'
                    frozen_write(archive,prior)
                policy=(rank_next(observations,target,values) if values is not None else next_policy(observations,target,name=='blasst_original',upper)) if observations else starting_policy(root,benchmark,name,configs[name],target)
                failed=False;start=max((r['iteration'] for r in observations),default=-1)+1
                rounds=0 if observations and min(r['error'] for r in observations)<=.02 else max_rounds
                for iteration in range(start,start+rounds):
                    token=_fingerprint(policy)[:16]
                    condition=f'{name}_s{int(target*100)}/{benchmark}/{token}'
                    outputs,failures=run_group(adapter,root,rows,'calibration',condition,configs[name],policy,fp)
                    if failures or len(outputs)!=len(rows): failed=True;break
                    metrics={kind:aggregate([r for o in outputs for r in o['records'] if r['probe']=='execution' and r['attention_type']==kind]) for kind in ('local','global')}
                    field='pv_omission' if name in ('compensate','zero_pv') else 'physical_sparsity'
                    achieved={kind:metrics[kind][field] for kind in metrics}
                    error=max(abs(value-target) for value in achieved.values())
                    record=dict(iteration=iteration,policy=policy,achieved=achieved,error=error,metrics=metrics,
                        source_condition=condition,source_ids=[r['id'] for r in rows])
                    observations.append(record)
                    _write(root/'calibration_traces'/benchmark/f'{name}_s{int(target*100)}.json',observations)
                    if error<=.02: break
                    updated=rank_next(observations,target,values) if values is not None else next_policy(observations,target,name=='blasst_original',upper)
                    if updated==policy: break
                    policy=updated
                if failed: continue
                best,fixed=select_point(observations,target,name=='blasst_original',upper)
                result=dict(fingerprint=fp,benchmark=benchmark,name=name,target=target,
                    **provenance(configs[name]),
                    config=configs[name],policy=best['policy'],measured=best['achieved'],error=best['error'],
                    cap_one_unattainable=fixed if name=='blasst_original' else None,
                    selection_version='independent_cap_one_v2',
                    search_version='empirical_rank_v1',search_provenance=search_source,
                    previous_policy_archive=str(archive) if archive else None,
                    within_two_points=best['error']<=.02,unattained_in_tested_range={k:v<target-.02 for k,v in best['achieved'].items()},
                    trace=observations,selected_round=best['iteration'],calibration_ids=[r['id'] for r in rows],
                    heldout_used=False,metric='PV omission, not complete block sparsity' if name in ('compensate','zero_pv') else 'physical deleted tiles')
                if prior:
                    assert json.loads(final.read_text())==prior,'another writer changed the policy during refinement'
                    _write(final,result)
                else:frozen_write(final,result)
    _write(root/'calibration_job_terminal.json',dict(finished=time.time(),protocol=protocol))


def development(root,names,targets):
    setup=prepare(root);fp=fingerprint(root);configs=candidate_configs(root);names=names or [n for n in configs if n not in REFINEMENTS]
    require_refinement_smoke(root,fp,[configs[n] for n in names])
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    smoke(adapter,setup,root)
    for name in names:
        for target in targets:
            path=root/'verified_policies'/'longbench'/f'{name}_s{int(target*100)}.json'
            if not path.exists():
                _append(root/'failures.jsonl',dict(stage='development',condition=name,target=target,error='missing verified policy'));continue
            policy=json.loads(path.read_text());assert_policy_provenance(policy,configs[name],fp)
            run_group(adapter,root,setup['development'],'development',f'{name}_s{int(target*100)}',
                configs[name],policy['policy'],fp)


def final_condition_order(conditions,only=None):
    """Dense always precedes sparse work, including after sorted-JSON reload.

    Explicit sparse-only resumes still verify/reuse the dense cache first;
    unknown condition names must not silently turn a requested run into a no-op.
    """
    if 'dense' not in conditions or conditions['dense']['config']:
        raise ValueError('final execution requires an unmodified dense baseline')
    if only is not None and (not only or not set(only)<=set(conditions)):
        raise ValueError('unknown or empty final condition selection')
    selected=set(conditions) if only is None else set(only)
    return ['dense']+[name for name in conditions if name!='dense' and name in selected]


def run_final(root,only=None):
    from .efficient_iteration import guard_legacy_run
    guard_legacy_run(root,'run-final')
    setup=prepare(root);fp=fingerprint(root)
    contract=json.loads((root/'final_contract.json').read_text())
    assert contract['fingerprint']==fp and contract['heldout_used_for_selection'] is False
    order=final_condition_order(contract['conditions'],only)
    require_refinement_smoke(root,fp,[c['config'] for c in contract['conditions'].values()])
    for c in contract['conditions'].values():
        for key in ('refinement_sha256','ranking_guard_sha256'):
            assert c.get(key)==provenance(c['config']).get(key),'frozen operator source mismatch'
    # Each sparse condition must have benchmark-specific verified provenance.
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    smoke(adapter,setup,root)
    for name in order:
        condition=contract['conditions'][name]
        for benchmark in ('aime26','longbench'):
            policy=condition.get('thresholds',{}).get(benchmark)
            rows=[r for r in setup['final'] if r['benchmark']==benchmark]
            if name=='dense':
                pending=[]
                for row in rows:
                    src=shard_path(root,'screen','dense',row['id'])
                    dest=shard_path(root,'final','dense',row['id'])
                    if src.exists() and not dest.exists():
                        reuse_dense(src,dest,row,fp)
                    else:pending.append(row)
                rows=pending
            run_group(adapter,root,rows,'final',name,condition['config'],policy,fp)
    _write(root/'final_run_terminal.json',dict(finished=time.time(),fingerprint=fp))


def reuse_dense(source,dest,row,fp):
    """Reuse matched dense generations, keeping the full screening source."""
    d=json.loads(source.read_text())
    assert d['fingerprint']==fp and not d['config'] and d['thresholds'] is None
    assert all(d[k]==row[k] for k in ('id','prompt_hash','seed','generation_budget'))
    d.update(records=[r for r in d['records'] if r['probe']=='execution'],distributions=[],screen=False,
        reused_dense_source=dict(path=str(source),sha256=sha(source.read_bytes()),source_screen=True,
            reason='identical operator/prompt/settings; observer never alters dense output'))
    _write(dest,d)


def merge_frozen_sources(*maps):
    """Retain raw evidence hashes, refusing inconsistent overlapping snapshots."""
    sources={}
    for entries in maps:
        for path,digest in entries.items():
            if path in sources and sources[path]!=digest:
                raise ValueError(f'inconsistent frozen evidence hash: {path}')
            sources[path]=digest
    return sources


def freeze(root,names):
    """An explicit, evidence-backed decision must precede final held-out use."""
    if not names:raise ValueError('explicit selected methods required')
    setup=prepare(root);fp=fingerprint(root);configs=candidate_configs(root)
    issues=root/'policy_review_issues.json'
    if issues.exists() and json.loads(issues.read_text())['unresolved']:
        raise RuntimeError('unresolved calibration review issues; repair them before freezing final conditions')
    routing_issues=root/'routing_review_issues.json'
    if routing_issues.exists() and json.loads(routing_issues.read_text())['unresolved']:
        raise RuntimeError('unresolved routing review issues; finish corrected ranking CUDA smoke/screen before freezing')
    decision=json.loads((root/'candidate_decision.json').read_text())
    assert decision['fingerprint']==fp and decision['heldout_used'] is False
    assert set(decision['selected_methods'])==set(names) and decision['rationale']
    # Selection must be backed by complete requested calibration/development
    # comparisons, not merely an assertion that no heldout data were used.
    evidence=json.loads((root/'development_audit.json').read_text())
    assert evidence['complete'] and not evidence['heldout_used'] and evidence['fingerprint']==fp
    assert evidence.get('raw_sparse_and_dense_checks') is True,'development evidence needs the shared raw sparse/dense audit'
    assert set(names)<=set(evidence['methods'])
    assert decision['development_audit_sha256']==sha((root/'development_audit.json').read_bytes())
    for source,expected in evidence['sources'].items():
        assert sha(Path(source).read_bytes())==expected,'development evidence changed after selection'
    conditions={'dense':dict(config={},target=0.)}
    from .policy_audit import audit as audit_policies
    policy_audit=audit_policies(root,list(dict.fromkeys(['blasst_original','blasst_aggressive']+names)),TARGETS)
    assert policy_audit['complete'],'selected policies must pass raw-shard audit at all four targets'
    sources=merge_frozen_sources(
        {str(root/p):sha((root/p).read_bytes()) for p in ('candidate_decision.json','development_audit.json','development_summary.json','policy_audit.json')},
        evidence['sources'],policy_audit['sources'])
    from .pruning import evidence as pruning_evidence
    _,pruning_sources=pruning_evidence(root)
    sources=merge_frozen_sources(sources,pruning_sources)
    for name in ['blasst_original','blasst_aggressive']+names:
        for target in TARGETS:
            policies={}
            for benchmark in ('aime26','longbench'):
                path=root/'verified_policies'/benchmark/f'{name}_s{int(target*100)}.json'
                p=json.loads(path.read_text())
                assert_policy_provenance(p,configs[name],fp)
                assert not p['heldout_used']
                assert set(p['calibration_ids'])=={r['id'] for r in setup['calibration'] if r['benchmark']==benchmark}
                policies[benchmark]=p['policy'];sources=merge_frozen_sources(sources,{str(path):sha(path.read_bytes())})
            conditions[f'{name}_s{int(target*100)}']=dict(config=configs[name],target=target,thresholds=policies,
                **provenance(configs[name]),
                target_metric='pv_omission' if name in ('compensate','zero_pv') else 'physical_sparsity')
    for signal in ('qk','mass','contribution'):
        for target in TARGETS:
            conditions[f'{signal}_topk_s{int(target*100)}']=dict(config=dict(method='diagnostic',pooling=signal,mode='topk',amount=target),target=target)
        if signal!='qk':
            for retained in (.95,.99):
                conditions[f'{signal}_topp_p{int(retained*100)}']=dict(config=dict(method='diagnostic',pooling=signal,mode='topp',amount=retained),
                    target=None,retained_nonnegative_signal=retained)
    # Smaller secondary study: Gaussian all four targets, same-proxy ranking
    # control at50%; both plain mean-QK and value-aware variants. The shared
    # dense-state screen already covers ranking at all four targets.
    for value in (False,True):
        for target in TARGETS:
            conditions[f'sol_{"value" if value else "plain"}_gaussian_s{int(target*100)}']=dict(config=dict(method='sol',value_proxy=value,mode='gaussian',amount=target),target=target)
        conditions[f'sol_{"value" if value else "plain"}_topk_s50']=dict(config=dict(method='sol',value_proxy=value,mode='topk',amount=.5),target=.5)
    contract=dict(fingerprint=fp,conditions=conditions,heldout_used_for_selection=False,sources=sources,
        frozen_at=time.time(),expected_per_condition={'aime26':30,'longbench':50},
        secondary_scope='10 end-to-end Sol configurations; Gaussian all targets, same-proxy ranking control50; shared screening ranking all targets',
        compensation='target counts refer to PV omission, never claimed physical deletion',
        report_full30_and_heldout24=True)
    # Proxy/ranking families may not consult true attention mass just to repair
    # an empty row. Streaming BLASST/value methods are unchanged.
    for condition in conditions.values():
        if condition['config'].get('method') in ('sol','diagnostic'):
            condition['config']['aggregation']=GUARD
            condition.update(provenance(condition['config']))
            condition['nonempty_rule']='highest declared signal on each empty row structural support'
    require_refinement_smoke(root,fp,[c['config'] for c in conditions.values()])
    sources[str(root/'ranking_guard_smoke.json')]=sha((root/'ranking_guard_smoke.json').read_bytes())
    if (root/'final_contract.json').exists():
        previous=json.loads((root/'final_contract.json').read_text());contract['frozen_at']=previous['frozen_at']
    frozen_write(root/'final_contract.json',contract)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['calibrate','development','freeze','run-final'])
    parser.add_argument('--output',type=Path,default=ROOT);parser.add_argument('--methods',nargs='+')
    parser.add_argument('--targets',nargs='+',type=float,default=[.5,.75]);parser.add_argument('--max-rounds',type=int,default=3)
    args=parser.parse_args()
    if not set(args.targets)<=set(TARGETS): raise ValueError('unsupported target')
    if args.command=='calibrate':calibrate(args.output,args.methods,args.targets,args.max_rounds)
    elif args.command=='development':development(args.output,args.methods,args.targets)
    elif args.command=='freeze':freeze(args.output,args.methods)
    else:run_final(args.output,args.methods)


if __name__=='__main__':main()
