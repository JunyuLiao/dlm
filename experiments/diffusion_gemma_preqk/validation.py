"""One frozen quality-only confirmation; no tuning or deployment claim.

Uses the same online router/generator, scorer, reports, and GPU supervisor. Seed
replicates have distinct storage IDs but retain prompt_id for correct bootstrap
clustering. New dense baselines are needed for the predeclared unseen seeds.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append,_fingerprint
from .config import ROOT,RouterConfig
from .run import digest,sample_key
from .online_run import generate,validate_router,router_options,fingerprint as source_fingerprint
from .online_report import report
from .performance import dense_generate
from .work_volume import analyze


def reserved_rows(protocol,benchmarks):
    dev_ids={r['id'] for r in protocol['development']};dev_hashes={r['prompt_hash'] for r in protocol['development']}
    rows=[]
    for row in protocol['heldout_reserved']:
        if row['benchmark'] not in benchmarks:continue
        if row['id'] in dev_ids or row['prompt_hash'] in dev_hashes:raise ValueError('held-out contamination')
        if row['paired_seeds']!=[314159,271828]:raise ValueError('reserved seeds changed')
        for seed in row['paired_seeds']:
            rows.append(dict(row,id=f"{row['id']}::seed={seed}",prompt_id=row['id'],seed=seed))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate held-out storage ID')
    return rows


def freeze(root,stage,parent,names):
    report(parent)
    evidence=json.loads((parent/'summary.json').read_text())
    if not evidence['audit']['complete']:raise RuntimeError('complete development evidence required')
    source=json.loads((parent/'freeze.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    conditions={n:source['conditions'][n] for n in names};benchmarks=[e['benchmark'] for e in conditions.values()]
    if not names or len(benchmarks)!=len(set(benchmarks)):raise ValueError('at most one frozen candidate per benchmark')
    for name,entry in conditions.items():
        if entry['config']['history_source']!='sparse' or router_options(entry):raise ValueError('only plain own-history candidate may advance')
        metrics=next(r for r in evidence['conditions'] if r['condition']==name)
        if metrics['accuracy_delta'] < -protocol['epsilon'][entry['benchmark']]:raise ValueError('candidate fails development quality tolerance')
    rows=reserved_rows(protocol,set(benchmarks))
    payload=dict(schema='preqk_heldout_quality_v1',conditions=conditions,samples=rows,
        cached_dense={},dense_cache_manifest=str(stage/'dense_cache.json'),epsilon=protocol['epsilon'],
        parent_stage=str(parent),parent_code_fingerprint=source_fingerprint(parent),
        evidence={str(p):digest(p) for p in (root/'protocol.json',parent/'summary.json',parent/'freeze.json',root/'reference_cost_v1'/'summary.json')},
        model=protocol['model'],revision=protocol['revision'],
        purpose='Quality-only confirmation of temporal ranking on reserved IDs/seeds; reference deployment remains rejected for cost. No held-out tuning or subsequent threshold search.',
        gates={'aime24':'rejected: no development candidate meets epsilon',
               'allocation':'rejected: total retained work exceeds dense',
               'ruler_livecodebench':'no expansion: no net-cost deployment finalist'},
        id_contract='id is a unique storage key; prompt_id is the original manifest ID and independent bootstrap unit',
        source_snapshot=str(parent/'source_snapshot'/'manifest.json'),
        decision_rule='compare paired prompt-level mean score difference and its 95% task-stratified bootstrap CI to -epsilon; average seeds within prompt; no retuning')
    path=stage/'freeze.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('immutable held-out freeze changed')
    _write(path,payload);return payload


def fingerprint(stage):
    setup=json.loads((stage/'freeze.json').read_text());parent=Path(setup['parent_stage'])
    if source_fingerprint(parent)!=setup['parent_code_fingerprint']:raise RuntimeError('frozen inference source changed')
    return _fingerprint(dict(freeze=digest(stage/'freeze.json'),inference=setup['parent_code_fingerprint'],
        harness=digest(Path(__file__)),dense_smoke_source=digest(Path(__file__).with_name('performance.py'))))


def smoke(stage,adapter):
    setup=json.loads((stage/'freeze.json').read_text());fp=fingerprint(stage);path=stage/'smoke.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['passed'] and old['fingerprint']==fp:return
    source=json.loads((Path(setup['parent_stage'])/'freeze.json').read_text());tests=[]
    for name,entry in setup['conditions'].items():
        row=dict(next(r for r in source['samples'] if r['benchmark']==entry['benchmark']),generation_budget=32)
        dense=dense_generate(adapter,row)
        out,router,_=generate(adapter,row,RouterConfig(sparsity=0.),diagnostics=True)
        assert dense.completion_tokens==out.completion_tokens and router.costs['retained_tiles']==router.costs['eligible_tiles']
        config=RouterConfig(**entry['config']);a,r,_=generate(adapter,row,config,diagnostics=True)
        b,_,_=generate(adapter,row,config,diagnostics=False)
        assert a.completion_tokens==b.completion_tokens
        validate_router(r,adapter.model.config.text_config.num_hidden_layers)
        assert {v['head'] for v in r.records}==set(range(adapter.model.config.text_config.num_attention_heads))
        tests.append(dict(condition=name,no_skip_dense_parity=True,diagnostic_parity=True,source='development only'))
    _write(path,dict(passed=True,fingerprint=fp,tests=tests))


def save_generation(stage,path,row,name,config,adapter,fp):
    began=time.monotonic();out,router,producer=generate(adapter,row,config,diagnostics=True)
    validate_router(router,adapter.model.config.text_config.num_hidden_layers)
    assert out.completion_tokens and out.metadata['sampling']['native_temperature_schedule']
    assert fingerprint(stage)==fp
    _write(path,dict(row,condition=name,config=config.to_dict(),fingerprint=fp,prediction=out.text,
        completion_tokens=out.completion_tokens,termination_reason=out.termination_reason,metadata=out.metadata,
        model_evaluations=out.model_evaluations,records=router.records,costs=router.costs,
        max_state_bytes=router.max_state_bytes,attention_calls=router.calls,router_options={},allocation_audit=None,
        prefix_value_elements=getattr(producer,'prefix_value_elements',0),canvas_value_elements=getattr(producer,'canvas_value_elements',0),
        diagnostic_e2e_seconds=time.monotonic()-began,executed_qk_saved=0,executed_pv_saved=0))


def pipeline(stage):
    setup=json.loads((stage/'freeze.json').read_text());fp=fingerprint(stage)
    for path,sha in setup['evidence'].items():assert digest(path)==sha,'frozen advancement evidence changed'
    adapter=create_adapter('diffusion_gemma',setup['model'],device='cuda',precision='bfloat16',revision=setup['revision']).load()
    smoke(stage,adapter)
    cache_path=Path(setup['dense_cache_manifest']);cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    failures=[];completed=0;rows=sorted(setup['samples'],key=lambda r:(r['generation_budget'],r['prompt_tokens'],r['id']))
    jobs=[('dense',row,RouterConfig(sparsity=0.)) for row in rows]
    jobs += [(name,row,RouterConfig(**entry['config'])) for name,entry in setup['conditions'].items() for row in rows if row['benchmark']==entry['benchmark']]
    for name,row,config in jobs:
        path=stage/('dense' if name=='dense' else 'conditions/'+name)/'shards'/f'{sample_key(row)}.json'
        _write(stage/'progress.json',dict(condition=name,id=row['id'],started=time.time(),completed=completed,expected=len(jobs)))
        try:
            if name!='dense':
                assert row['id'] in cache and digest(cache[row['id']]['path'])==cache[row['id']]['sha256'],'missing verified paired dense baseline'
            if path.exists():
                old=json.loads(path.read_text());assert old['fingerprint']==fp and old['config']==config.to_dict()
                assert all(old[k]==row[k] for k in ('prompt','seed','generation_budget','prompt_id'))
            else:save_generation(stage,path,row,name,config,adapter,fp)
            if name=='dense':
                cache[row['id']]=dict(path=str(path),sha256=digest(path));_write(cache_path,cache)
            completed+=1;print(time.strftime('%FT%TZ',time.gmtime()),name,row['id'],'completed',flush=True)
        except Exception:
            error=dict(condition=name,id=row['id'],traceback=traceback.format_exc());failures.append(error)
            _append(stage/'failures.jsonl',error);print(error,flush=True)
        finally:torch.cuda.empty_cache()
    _write(stage/'run_status.json',dict(finished=True,expected=len(jobs),completed=completed,failures=failures))
    validation_report(stage)


def validation_report(stage):
    report(stage);analyze(stage)
    setup=json.loads((stage/'freeze.json').read_text());summary=json.loads((stage/'summary.json').read_text())
    cache=json.loads(Path(setup['dense_cache_manifest']).read_text());dense_missing=[]
    for row in setup['samples']:
        if row['id'] not in cache:dense_missing.append(row['id']);continue
        item=cache[row['id']];assert digest(item['path'])==item['sha256'];data=json.loads(Path(item['path']).read_text())
        assert data['fingerprint']==json.loads((stage/'smoke.json').read_text())['fingerprint']
        assert all(data[k]==row[k] for k in ('prompt','seed','generation_budget','prompt_id'))
        assert data['costs']['eligible_tiles']==data['costs']['retained_tiles']
    results=[]
    for r in summary['conditions']:
        epsilon=setup['epsilon'][r['benchmark']]
        results.append(dict(condition=r['condition'],prompts=r['prompts'],generations=r['generations'],epsilon=epsilon,
            delta=r['accuracy_delta'],ci_low=r['accuracy_delta_ci_low'],ci_high=r['accuracy_delta_ci_high'],
            point_within_tolerance=r['accuracy_delta']>=-epsilon,
            bootstrap_interval_within_tolerance=r['accuracy_delta_ci_low']>=-epsilon,
            deployment_recommended=False,heldout_retuning=False))
    _write(stage/'validation_audit.json',dict(complete=summary['audit']['complete'] and not dense_missing,
        dense_generations=len(cache),dense_missing=dense_missing,paired_seeds=[314159,271828],decisions=results,
        purpose=setup['purpose'],source_shards=summary['audit']['source_shards'],heldout_retuning=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','pipeline','report'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--stage',type=Path,default=ROOT/'heldout_quality_v1')
    p.add_argument('--parent',type=Path,default=ROOT/'refinement_v1')
    p.add_argument('--conditions',nargs='+',default=['longbench__last_mass_control_s50__sparse']);a=p.parse_args()
    if a.command=='freeze':freeze(a.root,a.stage,a.parent,a.conditions)
    elif a.command=='report':validation_report(a.stage)
    else:pipeline(a.stage)
