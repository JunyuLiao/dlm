"""Freeze and execute the requested broad comparison, with no heldout tuning.

The first50% phase is explicitly interim. Full scope retains every proposed
family at25/50/75/90 plus the smaller Sol study and diagnostic rankings.
All phases use the same final60 and cached dense baseline; no scope pruning.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

from experiments.diffusion_gemma_value_aware.protocol import frozen_write
from experiments.diffusion_gemma_value_aware.ranking_guards import GUARD
from experiments.diffusion_gemma_value_aware.evaluate import final_condition_order
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from .engine import contract
from .execution import cached,provenance
from .evidence import read_result,pair,check_sources,merge_sources,require_sparse_smoke
from .development import policy_for
from .policies import METHODS,BENCHMARKS,inputs,candidate_configs
from .protocol import ROOT,MODEL,REVISION,TARGETS,prepare,sha

PHASES=('broad50','full')


def phase_targets(phase):
    if phase not in PHASES:
        raise ValueError('unknown final phase')
    return [.5] if phase=='broad50' else list(TARGETS)


def secondary_conditions():
    result={}
    for signal in ('qk','mass','contribution'):
        for target in TARGETS:
            result[f'{signal}_topk_s{round(target*100)}']=dict(
                config=dict(method='diagnostic',pooling=signal,mode='topk',amount=target,aggregation=GUARD),
                target=target,target_metric='physical_sparsity')
        if signal!='qk':
            for retained in (.95,.99):
                result[f'{signal}_topp_p{round(retained*100)}']=dict(
                    config=dict(method='diagnostic',pooling=signal,mode='topp',amount=retained,aggregation=GUARD),
                    target=None,target_metric='nonnegative_signal_mass',retained_nonnegative_signal=retained)
    for value in (False,True):
        label='value' if value else 'plain'
        for target in TARGETS:
            result[f'sol_{label}_gaussian_s{round(target*100)}']=dict(
                config=dict(method='sol',value_proxy=value,mode='gaussian',amount=target,aggregation=GUARD),
                target=target,target_metric='physical_sparsity')
        result[f'sol_{label}_topk_s50']=dict(
            config=dict(method='sol',value_proxy=value,mode='topk',amount=.5,aggregation=GUARD),
            target=.5,target_metric='physical_sparsity')
    for condition in result.values():
        condition.update(provenance(condition['config']))
    return result


def validate_scope(frozen):
    targets=phase_targets(frozen['phase'])
    expected={'dense'}|{f'{n}_s{round(t*100)}' for n in METHODS for t in targets}
    if frozen['phase']=='full':
        expected|=set(secondary_conditions())
    if set(frozen['conditions'])!=expected:
        raise ValueError('final contract must retain every requested family/reference; no silent scope reduction')
    if frozen['expected_per_condition']!={'aime26':30,'longbench_v2':30}:
        raise ValueError('every final condition must cover all30 AIME and all30 v2')
    if frozen['conditions']['dense']['config']:
        raise ValueError('dense baseline cannot have a sparse operator')
    pools={n:dict(pooling=frozen['conditions'][f'{n}_s50']['config'].get('pooling'))
        for n in ('value','mass_value','risk')}
    expected_configs=candidate_configs(dict(selected=pools))
    if frozen['heldout_used_for_selection'] or not frozen['aime24_previously_exposed']:
        raise ValueError('invalid heldout/selection disclosure')
    for name in METHODS:
        for target in targets:
            c=frozen['conditions'][f'{name}_s{round(target*100)}']
            if c['config']!=expected_configs[name]:
                raise ValueError('final family label and operator config differ')
            if c['target']!=target or set(c['thresholds'])!=set(BENCHMARKS):
                raise ValueError('missing benchmark-specific verified thresholds')
            expected_metric='pv_omission' if c['config']['method'] in ('compensate','zero_pv') else 'physical_sparsity'
            if c['target_metric']!=expected_metric:
                raise ValueError('PV replacement cannot be labeled physical deletion')
    if frozen['phase']=='full':
        for name,expected in secondary_conditions().items():
            if frozen['conditions'][name]!=expected:
                raise ValueError('secondary routing or fixed budget changed')


def freeze(root,phase,rationale):
    """No final output files are consulted by this selection/provenance gate."""
    if not rationale or len(rationale.strip())<20:
        raise ValueError('document the calibration/development rationale before freezing')
    targets=phase_targets(phase);setup=prepare(root);execution=contract(root)
    _,configs=inputs(root,execution)
    development=json.loads((root/'development_audit.json').read_text())
    if (not development['complete'] or development['fingerprint']!=execution['fingerprint']
            or development['heldout_used'] or not development['raw_sparse_and_dense_checks']
            or not set(METHODS)<=set(development['methods']) or .5 not in development['targets']):
        raise ValueError('complete broad50 calibration/development comparison required')
    check_sources(development['sources'])
    sources=merge_sources(development['sources'],require_sparse_smoke(root,setup,execution,configs))
    if phase=='full':
        from .secondary import require_smoke
        sources=merge_sources(sources,require_smoke(root,setup,execution))
    conditions={'dense':dict(config={},target=0.,target_metric='physical_sparsity')}
    for name in METHODS:
        for target in targets:
            thresholds={};policy_sources={}
            for benchmark in BENCHMARKS:
                policy,source=policy_for(root,setup,execution,benchmark,name,configs[name],target)
                thresholds[benchmark]=policy['policy'];policy_sources[benchmark]=source
                sources[source['path']]=source['sha256']
                for point in policy['trace']:
                    sources=merge_sources(sources,{s['path']:s['sha256'] for s in point['sources'].values()})
            conditions[f'{name}_s{round(target*100)}']=dict(config=configs[name],target=target,
                thresholds=thresholds,policy_sources=policy_sources,**provenance(configs[name]),
                target_metric='pv_omission' if name in ('compensate','zero_pv') else 'physical_sparsity')
    if phase=='full':
        conditions.update(secondary_conditions())
    for filename in ('execution_contract.json','setup.json','dataset_audit.json','threshold_proposals.json',
            'pooling_selection.json','screen_analysis_audit.json','supplemental_audit.json',
            'development_audit.json','development_summary.json','development_per_sample.json'):
        path=root/filename;sources[str(path)]=sha(path.read_bytes())
    sources=merge_sources(sources,execution['sources'])
    for name in ('final.py','evidence.py','policies.py','execution.py','calibrate.py','development.py'):
        path=Path(__file__).with_name(name);sources[str(path)]=sha(path.read_bytes())
    decision=dict(fingerprint=execution['fingerprint'],phase=phase,rationale=rationale,
        retained_methods=list(METHODS),heldout_used=False,
        rule='All six requested families remain visible, including negative results; pooling/thresholds use calibration; development supports mechanism diagnosis',
        development_audit_sha256=sha((root/'development_audit.json').read_bytes()),
        broad50_does_not_complete_full_scope=True)
    decision_path=root/'decisions'/f'{phase}.json';frozen_write(decision_path,decision)
    sources[str(decision_path)]=sha(decision_path.read_bytes())
    frozen=dict(fingerprint=execution['fingerprint'],phase=phase,conditions=conditions,sources=sources,
        expected_per_condition=dict(aime26=30,longbench_v2=30),heldout_used_for_selection=False,
        aime24_previously_exposed=True,expected_shards=60*len(conditions),
        scope='Interim broad50 comparison; all four-target curves still required' if phase=='broad50'
            else 'All six families/controls at four targets, diagnostic rankings, smaller plain/value Sol study',
        temperature_note='Seeded native0.4–0.8 temperature schedule, not greedy',
        input_budget=setup['input_budget'])
    validate_scope(frozen);check_sources(sources)
    frozen_write(root/'final_contracts'/f'{phase}.json',frozen)
    return frozen


def load_contract(root,phase,execution):
    data=json.loads((root/'final_contracts'/f'{phase}.json').read_text())
    if data['fingerprint']!=execution['fingerprint'] or data['phase']!=phase:
        raise ValueError('frozen final execution mismatch')
    validate_scope(data);check_sources(data['sources'])
    configs=candidate_configs(json.loads((root/'pooling_selection.json').read_text()))
    for name in METHODS:
        for target in phase_targets(phase):
            c=data['conditions'][f'{name}_s{round(target*100)}']
            if c['config']!=configs[name]:
                raise ValueError('final pooling differs from frozen calibration selection')
            for benchmark in BENCHMARKS:
                source=c['policy_sources'][benchmark]
                if data['sources'].get(source['path'])!=source['sha256']:
                    raise ValueError('final threshold policy is not a frozen source')
                policy=json.loads(Path(source['path']).read_text())
                if (policy['config']!=c['config'] or policy['policy']!=c['thresholds'][benchmark]
                        or policy['benchmark']!=benchmark or policy['name']!=name or policy['target']!=target):
                    raise ValueError('final thresholds differ from the verified selected policy')
    for condition in data['conditions'].values():
        for key,expected in provenance(condition['config']).items():
            if condition.get(key)!=expected:
                raise ValueError('frozen final operator source changed')
    return data


def reuse_equivalent(root,row,name,conditions,execution):
    """Reuse a completed identical operator, e.g. several capped-lambda1 targets.

    A target label is not a model setting. Config, both threshold dictionaries,
    prompt, seed, budget and actual source hash must match; no interpolation or
    approximate threshold equivalence is permitted here.
    """
    requested=conditions[name];policy=requested.get('thresholds',{}).get(row['benchmark'])
    for other,candidate in conditions.items():
        if other in (name,'dense') or candidate['config']!=requested['config']:
            continue
        if candidate.get('thresholds',{}).get(row['benchmark'])!=policy:
            continue
        try:
            result,source=read_result(root,row,'final',other,candidate['config'],policy,execution)
        except FileNotFoundError:
            continue
        path=root/'final_aliases.json';aliases=json.loads(path.read_text()) if path.exists() else {}
        entry=dict(source,source_stage='final',source_condition=other,
            reason='identical executed config/thresholds/prompt/seed/budget; target label alone differs')
        key=f'{name}/{row["id"]}'
        if key in aliases and aliases[key]!=entry:
            raise ValueError('final reuse alias changed')
        aliases[key]=entry;_write(path,aliases)
        return result
    raise FileNotFoundError(f'no completed equivalent final output: {name}/{row["id"]}')


def run(root,phase,only=None):
    setup=prepare(root);execution=contract(root);frozen=load_contract(root,phase,execution)
    order=final_condition_order(frozen['conditions'],only)
    rows=sorted(setup['final'],key=lambda r:(r['generation_budget'],r['id']))
    # Dense is always verified first, even for a sparse-only resume. No dense
    # inference is allowed here: initial-stage cached baselines are required.
    for row in rows:
        read_result(root,row,'dense','dense',{},None,execution)
    adapter=None;failures=[]
    import torch
    from dllm.models import create_adapter
    for name in order:
        if name=='dense':
            continue
        condition=frozen['conditions'][name]
        for row in rows:
            policy=condition.get('thresholds',{}).get(row['benchmark'])
            try:
                try:
                    result,_=read_result(root,row,'final',name,condition['config'],policy,execution)
                except FileNotFoundError:
                    try:
                        result=reuse_equivalent(root,row,name,frozen['conditions'],execution)
                    except FileNotFoundError:
                        try:
                            # Explicit matching historical AIME reuse precedes
                            # model loading; no v1 result may stand in for v2.
                            result=cached(None,root,row,'final',name,condition['config'],policy,execution)
                        except FileNotFoundError:
                            if adapter is None:
                                adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',
                                    precision='bfloat16',revision=REVISION).load()
                            result=cached(adapter,root,row,'final',name,condition['config'],policy,execution)
                dense,_=read_result(root,row,'dense','dense',{},None,execution)
                pair(row,result,dense)
                _append(root/'final_completed.jsonl',dict(phase=phase,condition=name,id=row['id'],finished=time.time()))
            except Exception:
                error=dict(stage='final',phase=phase,condition=name,id=row['id'],traceback=traceback.format_exc())
                failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
            finally:
                torch.cuda.empty_cache()
    _write(root/'final_runs'/f'{phase}_terminal.json',dict(finished=time.time(),phase=phase,
        attempted_conditions=order,failed_attempts=len(failures),
        scope='Execution terminal only; report raw-shard audit determines completion'))
    if failures:
        raise RuntimeError('some final samples failed; independent samples/configurations continued')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'))
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--phase',choices=PHASES,default='broad50')
    p.add_argument('--rationale');p.add_argument('--only',nargs='+');args=p.parse_args()
    if args.command=='freeze':
        data=freeze(args.output,args.phase,args.rationale)
        print(json.dumps(dict(phase=args.phase,conditions=len(data['conditions']),expected=data['expected_shards'])))
    else:
        run(args.output,args.phase,args.only)


if __name__=='__main__':
    main()
