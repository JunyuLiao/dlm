"""Evidence-gated, resumable development rollouts for a small fixed shortlist.

This CLI cannot evaluate held-out IDs. It requires a complete stage-one audit and
an explicit hypothesis/selection artifact before freezing paired history modes.
"""
import argparse
from dataclasses import replace
import json
import time
import traceback
from pathlib import Path
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.config import MODEL,REVISION
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append,_install_dense,_request,_set_context,_fingerprint
from .config import ROOT,RouterConfig,screening_configs
from .run import digest,sample_key
from .online import OnlineAttention,CacheIdentity
from .summaries import ProducerSummaries


def build_conditions(selection):
    if not selection.get('hypothesis') or not selection.get('decision_rationale'):
        raise ValueError('record the hypothesis and evidence-based advancement decision')
    if selection.get('phase')=='refinement':
        from .refinement import build_conditions as build_refinement
        return build_refinement(selection)
    entries=selection['entries'];predictors=screening_configs()
    if not 2<=len(entries)<=6:raise ValueError('small shortlist required: 2–6 benchmark/predictor/budget entries')
    policy=dict(refresh_interval=4,exploration_tiles=1,renormalization='estimated_coverage')
    overrides=selection.get('online_policy',{})
    if set(overrides)-set(policy):raise ValueError('unsupported online refinement; keep changes explicit and bounded')
    policy.update(overrides)
    seen=set();conditions={}
    for entry in entries:
        benchmark=entry['benchmark'];predictor=entry['predictor'];target=float(entry['target_sparsity'])
        if benchmark not in ('aime24','longbench') or predictor not in predictors:raise ValueError('unknown development configuration')
        if target not in ((.4,.5) if benchmark=='aime24' else (.5,.75)):raise ValueError('budget was not screened')
        identity=(benchmark,predictor,target)
        if identity in seen:raise ValueError('duplicate candidate')
        seen.add(identity)
        for source in ('dense_reference','sparse'):
            # Only the source of past observations differs within each pair.
            config=replace(predictors[predictor],history_source=source,sparsity=target,execution='reference',**policy)
            name=f'{benchmark}__{predictor}_s{int(target*100)}__{source}'
            conditions[name]=dict(benchmark=benchmark,predictor=predictor,config=config.to_dict())
    if {e[0] for e in seen}!={'longbench','aime24'}:raise ValueError('include both development benchmarks')
    for benchmark,predictor,target in seen:
        if (benchmark,'last_mass',target) not in seen:raise ValueError('each budget requires a plain last-mass control')
    return conditions


def freeze(root,stage,selection_path):
    from .report import report
    report(root) # reconstruct evidence from shards; never trust a stale complete flag
    audit=json.loads((root/'screen_audit.json').read_text())
    if not audit['complete']:raise RuntimeError('complete development screening required before selecting online candidates')
    selection=json.loads(selection_path.read_text());conditions=build_conditions(selection)
    protocol=json.loads((root/'protocol.json').read_text())
    payload=dict(schema='preqk_online_development_v1',selection=selection,conditions=conditions,
        samples=protocol['development'],cached_dense=protocol['cached_dense'],protocol_sha256=digest(root/'protocol.json'),
        screen_evidence={name:digest(root/name) for name in ('screen_audit.json','screen_summary.csv','screen_paired_ablations.csv')},
        screen_fingerprints=audit['fingerprints'],model=protocol['model'],revision=protocol['revision'],
        history_reference='paid dense previous-step observations on the candidate sparse trajectory, not replay of original dense trajectory',
        history_sparse='observed masked-softmax mass; explicit renormalization/refresh/exploration policy in each config; all observations charged and exploration inside budget',
        diagnostic_source='same current sparse state, all calls; never used in current decisions or sparse-history update',
        executed_backend='dense-mask reference; executed QK/PV savings are zero',heldout_evaluated=False)
    if selection.get('phase')=='refinement':
        from .online_report import report as online_report
        parent=Path(selection['parent_stage']);online_report(parent)
        parent_audit=json.loads((parent/'audit.json').read_text())
        if not parent_audit['complete']:raise RuntimeError('complete parent online evidence required')
        evidence={name:digest(parent/name) for name in ('audit.json','summary.csv','freeze.json')}
        if evidence!=selection['parent_evidence']:raise RuntimeError('refinement parent evidence changed')
        payload['parent_evidence']=evidence
    path=stage/'freeze.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('immutable shortlist differs; use a new stage namespace')
    _write(path,payload);return payload


def load(stage):
    payload=json.loads((stage/'freeze.json').read_text())
    if payload['schema']!='preqk_online_development_v1' or payload['heldout_evaluated']:
        raise ValueError('only frozen development studies are accepted here')
    return payload


def fingerprint(stage):
    paths=[Path(__file__).with_name(n) for n in ('online_run.py','online.py','state.py','config.py','summaries.py','routing.py','allocation.py','refinement.py')]
    paths += [Path('src/dllm/attention/blasst/core.py'),Path('src/dllm/attention/blasst/integration.py'),
        Path('src/dllm/models/adapters/diffusion_gemma.py')]
    return _fingerprint(dict(freeze=digest(stage/'freeze.json'),code={str(p):digest(p) for p in paths},
        torch=torch.__version__,transformers=__import__('transformers').__version__))


def generate(adapter,row,config,*,diagnostics=True,isolate=None,allocation_policy=None):
    binding=_install_dense(adapter)
    producer=CacheIdentity(adapter) if config.value_weight=='none' else ProducerSummaries(adapter)
    router=OnlineAttention(row['id'],config,producer,diagnostics=diagnostics,isolate=isolate,allocation_policy=allocation_policy)
    binding.runtime.attention_override=router
    try:
        _set_context(binding,row);out=adapter.generate(_request(row))
    finally:producer.close();binding.close()
    return out,router,producer


def validate_router(router,layers):
    assert router.calls and router.records
    assert {r['attention_type'] for r in router.records}=={'local','global'}
    assert len({r['layer'] for r in router.records})==layers
    for r in router.records:
        assert 0<=r['skipped']<=r['eligible']
        assert r['prefix_eligible']+r['canvas_eligible']+r['mixed_eligible']==r['eligible']
        assert r['prefix_skipped']+r['canvas_skipped']+r['mixed_skipped']==r['skipped']
        assert all(__import__('math').isfinite(v) for v in r.values() if isinstance(v,(int,float)))
    if router.isolate is not None:
        assert router.costs['isolated_sparse_calls']==1,'isolated perturbation must fire exactly once'
    if router.allocator:
        assert router.allocator.audit
        assert all(a['quota_delta']==0 and a['max_abs_row_delta']==0 for a in router.allocator.audit),'allocation changed matched per-row total work'
        if router.diagnostics:
            assert sum(r['eligible']-r['skipped'] for r in router.records)==sum(r['shadow_uniform_retained'] for r in router.records)


def router_options(entry):
    return {k:entry[k] for k in ('isolate','allocation_policy') if k in entry}


def smoke(stage,adapter=None):
    setup=load(stage)
    path=stage/'smoke.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('passed') and previous.get('fingerprint')==fingerprint(stage):return previous
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for online smoke')
    adapter=adapter or create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    tests=[]
    for benchmark in ('longbench','aime24'):
        row=dict(next(r for r in setup['samples'] if r['benchmark']==benchmark),generation_budget=32)
        binding=_install_dense(adapter)
        try:_set_context(binding,row);dense=adapter.generate(_request(row))
        finally:binding.close()
        for diagnostic in (False,True):
            out,router,_=generate(adapter,row,RouterConfig(sparsity=0.),diagnostics=diagnostic)
            assert out.completion_tokens==dense.completion_tokens,'online no-skip exact parity failed'
            validate_router(router,adapter.model.config.text_config.num_hidden_layers)
            assert sum(r['skipped'] for r in router.records)==0
        entries=[(name,v) for name,v in setup['conditions'].items() if v['benchmark']==benchmark and v['config']['history_source']=='sparse']
        for name,entry in entries:
            config=RouterConfig(**entry['config']);a,router,_=generate(adapter,row,config,diagnostics=True,**router_options(entry))
            b,plain,_=generate(adapter,row,config,diagnostics=False,**router_options(entry))
            assert a.completion_tokens==b.completion_tokens,'diagnostics changed sparse rollout'
            validate_router(router,adapter.model.config.text_config.num_hidden_layers)
            validate_router(plain,adapter.model.config.text_config.num_hidden_layers)
            tests.append(dict(benchmark=benchmark,condition=name,passed=True,attention_calls=router.calls,
            skipped=sum(r['skipped'] for r in router.records),eligible=sum(r['eligible'] for r in router.records),
            layer_ids=sorted({r['layer'] for r in router.records}),head_ids=sorted({r['head'] for r in router.records}),
            denoising_steps=sorted({r['step'] for r in router.records}),no_skip_exact_parity=True,diagnostic_exact_parity=True,
            isolated_sparse_calls=router.costs['isolated_sparse_calls'],allocation_audit=router.allocator.audit if router.allocator else None))
    result=dict(passed=True,fingerprint=fingerprint(stage),tests=tests);_write(path,result);return result


def run(stage,adapter=None):
    setup=load(stage);fp=fingerprint(stage);audit=json.loads((stage/'smoke.json').read_text())
    if not audit['passed'] or audit['fingerprint']!=fp:raise RuntimeError('matching online CUDA smoke required')
    adapter=adapter or create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    failures=[];expected=completed=0
    for name,entry in setup['conditions'].items():
        config=RouterConfig(**entry['config'])
        rows=sorted([r for r in setup['samples'] if r['benchmark']==entry['benchmark']],key=lambda r:(r['generation_budget'],r['prompt_tokens'],r['id']))
        for row in rows:
            expected+=1;path=stage/'conditions'/name/'shards'/f'{sample_key(row)}.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['fingerprint']==fp;completed+=1;continue
            _write(stage/'progress.json',dict(condition=name,id=row['id'],started=time.time()))
            try:
                cache=setup['cached_dense'][row['id']];assert digest(cache['path'])==cache['sha256']
                dense=json.loads(Path(cache['path']).read_text())
                assert all(dense[k]==row[k] for k in ('prompt','seed','generation_budget'))
                start=time.monotonic();out,router,producer=generate(adapter,row,config,**router_options(entry))
                validate_router(router,adapter.model.config.text_config.num_hidden_layers)
                assert out.completion_tokens and fingerprint(stage)==fp
                assert out.metadata['sampling'].get('native_temperature_schedule')
                _write(path,dict(row,condition=name,config=config.to_dict(),fingerprint=fp,prediction=out.text,
                    completion_tokens=out.completion_tokens,termination_reason=out.termination_reason,
                    metadata=out.metadata,model_evaluations=out.model_evaluations,records=router.records,costs=router.costs,
                    max_state_bytes=router.max_state_bytes,attention_calls=router.calls,
                    router_options=router_options(entry),allocation_audit=router.allocator.audit if router.allocator else None,
                    prefix_value_elements=getattr(producer,'prefix_value_elements',0),canvas_value_elements=getattr(producer,'canvas_value_elements',0),
                    dense_cache=cache,diagnostic_e2e_seconds=time.monotonic()-start,executed_qk_saved=0,executed_pv_saved=0))
                completed+=1;print(time.strftime('%FT%TZ',time.gmtime()),name,row['id'],'completed',flush=True)
            except Exception:
                error=dict(condition=name,id=row['id'],traceback=traceback.format_exc());failures.append(error)
                _append(stage/'failures.jsonl',error);print(error,flush=True)
            finally:torch.cuda.empty_cache()
    _write(stage/'run_status.json',dict(finished=True,expected=expected,completed=completed,failures=failures))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','smoke','run','pipeline','report'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--stage',type=Path,default=ROOT/'online_v1')
    p.add_argument('--selection',type=Path);args=p.parse_args()
    if args.command=='freeze':
        if not args.selection:p.error('--selection required')
        freeze(args.root,args.stage,args.selection)
    elif args.command=='report':
        from .online_report import report
        report(args.stage)
    elif args.command=='pipeline':
        adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
        smoke(args.stage,adapter);run(args.stage,adapter)
        from .online_report import report
        report(args.stage)
    else:globals()[args.command](args.stage)


if __name__=='__main__':main()
