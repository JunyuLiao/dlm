"""Gaussian centered rank1 AIME26 and ranks1/2/8 LongBench-v2-100.

Versioned orchestration only: reuse frozen numerical cores, calibration and
NeMo evaluation; preserve all predecessor workers/configurations/results.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
from dllm.models import create_adapter
from experiments import diffusion_gemma_jl_aime_tiny_dimensions as tiny
from experiments import diffusion_gemma_jl_remaining50 as remaining
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint
from experiments.diffusion_gemma_value_aware.run import shard_path

ROOT=Path('results/diffusion_gemma_jl_lowrank_aime30_longbench100_v10')
MODULE='experiments.diffusion_gemma_jl_lowrank_multibench'
AIME=tiny.ROOT
PRIOR=tiny.previous.prior
V6,V7,LB=PRIOR.ROOT,remaining.ROOT,remaining.LB
NEW={f'jl_gaussian_r{r}':dict(family='gaussian',rank=r) for r in (1,2,8)}
ALL={**tiny.ALL,**NEW}
BASELINES=PRIOR.parent.BASELINES
CONFIGS={**BASELINES,**ALL}
NEW_BY_BENCHMARK={'aime26':['jl_gaussian_r1'],'longbench_v2':list(NEW)}
REUSED_BY_BENCHMARK={'aime26':tiny.CONDITIONS,'longbench_v2':remaining.CONDITIONS}
CONDITIONS_BY_BENCHMARK={b:old+[n+'_s50' for n in NEW_BY_BENCHMARK[b]] for b,old in REUSED_BY_BENCHMARK.items()}
CONDITIONS=list(dict.fromkeys(n for names in CONDITIONS_BY_BENCHMARK.values() for n in names))
COUNTS={'aime26':30,'longbench_v2':100}
EXPECTED,TEST_COUNT=1260,13
read,sha,frozen_write=tiny.read,tiny.sha,tiny.frozen_write
fb,backend,continuation=tiny.fb,tiny.backend,tiny.continuation
runner,screen,shared_analysis,validation=tiny.runner,tiny.screen,tiny.shared_analysis,tiny.validation


@contextmanager
def dimension_scope():
    with patch.object(tiny.previous.configuration,'DIMENSIONS',(1,2,4,8,16,24,32)),patch.object(runner,'PROJECTED',ALL):
        yield


def calibration_rows(setup,benchmark):
    expected=[r for r in read(V6/'setup.json')['calibration'] if r['benchmark']==benchmark]
    rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
    if len(rows)!=6 or rows!=expected or [r['id'] for r in rows]!=setup['calibration_ids'][benchmark]:
        raise ValueError('Require unchanged six calibration rows from the existing convention')
    if any(r['generation_budget']!=(2048 if benchmark=='aime26' else 4096) for r in rows):
        raise ValueError('Calibration must use full output budgets')
    finals={r['id']:r for r in setup['final']}
    for r in rows:
        if r['id'] not in finals or any(r[k]!=finals[r['id']][k] for k in ('prompt_hash','prompt_tokens','generation_budget','seed')):
            raise ValueError('Expected explicit, inference-identical calibration/final overlap')
    return rows


def prepare(root=ROOT):
    old,lb,aime=read(V6/'setup.json'),read(LB/'setup.json'),read(AIME/'setup.json')
    cal=deepcopy(old['calibration']);ids={b:[r['id'] for r in cal if r['benchmark']==b] for b in COUNTS}
    rows=deepcopy(aime['final'])+[dict(deepcopy(r),calibration=r['id'] in ids['longbench_v2']) for r in lb['final']]
    for b,count in COUNTS.items():
        group=[r for r in rows if r['benchmark']==b]
        if len(group)!=count or any(len({r[k] for r in group})!=count for k in ('id','prompt_hash')):
            raise ValueError('Unexpected or duplicated evaluation cohort')
        if any(r['generation_budget']!=(2048 if b=='aime26' else 4096) or r['seed']!=42 or sha(r['prompt'])!=r['prompt_hash'] for r in group):
            raise ValueError('Prompt, budget or seed changed')
    for key in ('model','revision','precision','tile_size','regions','input_budget'):
        if not (old[key]==lb[key]==aime[key]):raise ValueError(f'Incompatible cached setting: {key}')
    old50={r['id'] for r in old['final'] if r['benchmark']=='longbench_v2'}
    other50={r['id'] for r in read(V7/'setup.json')['final']}
    if old50&other50 or old50|other50!={r['id'] for r in lb['final']}:raise ValueError('Prior50+remaining50 do not cover cached100')
    setup=dict(schema='jl_lowrank_multibench_v10',**{k:deepcopy(old[k]) for k in
        ('model','revision','precision','tile_size','regions','decoding','input_budget')},
        final=rows,calibration=cal,development=deepcopy(old['development']),calibration_ids=ids,
        calibration_overlap=ids,conditions=CONDITIONS,conditions_by_benchmark=CONDITIONS_BY_BENCHMARK,
        configs=CONFIGS,targets=[.5],expected_counts=COUNTS,new_by_benchmark=NEW_BY_BENCHMARK,
        projection_seed=1729,projection_family='gaussian',
        selection=dict(longbench_domains=dict(sorted(Counter(r['task'] for r in lb['final']).items())),
            longbench_subtasks=dict(sorted(Counter(r['sub_domain'] for r in lb['final']).items())),
            rule='Exact existing100 LongBench and30 AIME, no score-based selection; calibration flags are annotations only'),
        calibration_policy='Same six AIME and six LongBench IDs as v6, full2048/4096 budgets, independent local/global scalar thresholds. Existing empirical-CDF proposal/scalar fallback, max12 joint points, require overall/global/local48-52%. Only sparsity selects; freeze before final inference.',
        proposal_policy='Same540 historical calibration QKV snapshots as v6; regenerate rank1/2/8 CDFs without dense replay. Historical LongBench proposal states differ from the six full-budget verification IDs, as before.',
        exposure='All samples previously examined. Both headline denominators include six calibration members; noncalibration24/94 additionally reported. Not fresh held-out confirmation.',
        final_generation_slots=330,reused_reference_and_baseline_slots=930,monitoring_interval_seconds=900,
        validation_policy='Fresh rank1 two-input actual-model parity; exact rank2/rank8 and native smoke shards reused from audited predecessors; no algorithm/kernel changes.')
    for b in COUNTS:calibration_rows(setup,b)
    frozen_write(root/'setup.json',setup)
    for split in ('final','calibration','development'):frozen_write(root/f'{split}_manifest.json',setup[split])
    frozen_write(root/'dataset_audit.json',dict(passed=True,expected=EXPECTED,counts=COUNTS,
        source_rows_verbatim_except_calibration_annotation=True,calibration_overlap=ids,
        new_final_slots=330,cached_slots=930,longbench_truncated=sum(r['truncated'] for r in lb['final']),
        selection=setup['selection'],fresh_heldout_claim=False))
    return setup


def execution(root=ROOT):
    import triton,transformers
    from experiments import diffusion_gemma_jl_lowrank_multibench_report as reporting
    prepare(root);old=read(AIME/'execution_contract.json')
    runtime=dict(torch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__)
    if runtime!=old['runtime']:raise ValueError('Cached runtime mismatch')
    tests=root/'tests.xml';suites=ET.parse(tests).getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=TEST_COUNT,failures=0,errors=0,skipped=0):raise ValueError(f'Incomplete test gate: {counts}')
    sources={}
    for folder in (AIME,V6,V7,LB):
        sources=backend.evidence.merge_sources(sources,read(folder/'execution_contract.json')['sources'])
    files=(Path(__file__),Path(reporting.__file__),Path('tests/test_jl_lowrank_multibench.py'),tests,root/'setup.json')
    sources=backend.evidence.merge_sources(sources,{str(p):sha(p.read_bytes()) for p in files});fb.check_sources(sources)
    data=dict(schema='jl_lowrank_multibench_execution_v10',sources=sources,runtime=runtime,
        parent_fingerprint=old['fingerprint'],setup_sha256=sha((root/'setup.json').read_bytes()),
        numerical_kernels_unchanged=True,scoped_allowed_dimensions=[1,2,4,8,16,24,32],
        requested_new=NEW_BY_BENCHMARK,projection_seed=1729,projection_dtype='float32',matmul_allow_tf32=False,
        target_sparsity=.5,tolerance=.02,max_verified_points=12,expected=EXPECTED,new_final_slots=330,
        calibration_budgets={'aime26':2048,'longbench_v2':4096},hardware_speedup_claim=False)
    data['fingerprint']=_fingerprint(data);frozen_write(root/'execution_contract.json',data);return data


def imported_policy_path(benchmark,name):
    return (AIME if benchmark=='aime26' else V6)/'policies'/benchmark/f'{name}_s50.json'


def audit_policy(root,p,setup,contract):
    b,n=p['benchmark'],p['name']
    if p['fingerprint']!=contract['fingerprint'] or p['heldout_used'] or p['target']!=.5 or n+'_s50' not in CONDITIONS_BY_BENCHMARK[b] or p['config']!=CONFIGS[n]:
        raise ValueError('Invalid policy scope or identity')
    if p.get('imported_policy'):
        src=imported_policy_path(b,n)
        if p['imported_policy']!=str(src):raise ValueError('Unexpected cached policy source')
        fb.check_sources(p['sources']);old=read(src)
        if any(p[k]!=old[k] for k in ('name','benchmark','target','config','policy','measured')):raise ValueError('Cached policy changed')
    else:
        if n not in NEW_BY_BENCHMARK[b]:raise ValueError('Unexpected new calibration')
        with patch.object(fb,'CONFIG',CONFIGS[n]),patch.object(fb,'calibration_rows',calibration_rows):fb.audit_policy(root,p,setup,contract)


def source_for(benchmark,label,identifier):
    if benchmark=='aime26':return AIME
    if label in remaining.BASELINE_LABELS:return LB
    previous_ids={r['id'] for r in read(V6/'setup.json')['final']}
    return V6 if identifier in previous_ids else V7


def reuse(root,setup,contract):
    proof={}
    for folder,count in ((AIME,330),(V6,640),(V7,300),(LB,900)):
        proof.update(remaining.require_complete(folder,count))
    validation_sources={}
    for module,folder in ((tiny,AIME),(tiny.previous,tiny.PREVIOUS),(PRIOR,V6)):
        validation_sources.update(module.smoke_audit(folder,read(folder/'setup.json'),read(folder/'execution_contract.json')))
    frozen_write(root/'inherited_validation.json',dict(passed=True,sources=validation_sources))
    for b,labels in REUSED_BY_BENCHMARK.items():
        for label in labels:
            name='dense' if label=='dense' else label.rsplit('_s',1)[0]
            threshold=None
            if name!='dense':
                src=imported_policy_path(b,name);old=read(src)
                module=tiny if b=='aime26' else PRIOR;folder=AIME if b=='aime26' else V6
                module.audit_policy(folder,old,read(folder/'setup.json'),read(folder/'execution_contract.json'))
                p={k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
                p.update(fingerprint=contract['fingerprint'],heldout_used=False,imported_policy=str(src),
                    sources={str(src):sha(src.read_bytes()),**proof},calibration_ids=old.get('calibration_ids'),
                    cap_one_unattainable=old.get('cap_one_unattainable'))
                audit_policy(root,p,setup,contract);frozen_write(root/'policies'/b/f'{label}.json',p);threshold=p['policy']
            for row in (r for r in setup['final'] if r['benchmark']==b):
                folder=source_for(b,label,row['id']);old_contract=read(folder/'execution_contract.json')
                cfg=read(folder/'final_configs'/f'{label}.json');fb.check_sources(cfg['sources'])
                expected={} if name=='dense' else CONFIGS[name]
                if cfg['config']!=expected or cfg['thresholds'][b]!=threshold:raise ValueError('Cached output operator/threshold mismatch')
                stage='dense' if name=='dense' else 'final'
                fb.alias(root,row,stage,label,name,expected,threshold,contract,shard_path(folder,stage,label,row['id']),old_contract['fingerprint'])
            print('reused',b,label,COUNTS[b],flush=True)
    frozen_write(root/'predecessor_audit.json',dict(passed=True,reused_outputs=930,sources=proof))
    states=read(V6/'shared_state_index.json');historical=read(PRIOR.parent.OLD/'setup.json')['calibration']
    if len(states)!=540 or any(s['split']!='calibration' or s['id'] not in {r['id'] for r in historical} for s in states):
        raise ValueError('Unexpected historical proposal/diagnostic state population')
    fb.check_sources({s['path']:s['sha256'] for s in states});frozen_write(root/'shared_state_index.json',states)
    for b,folder in (('aime26',AIME),('longbench_v2',V6)):
        for item in read(folder/'shared_diagnostics_index.json'):
            src=Path(item['path']);data=read(src);ident=data['identity']
            if not ident['source']['id'].startswith(b+'/') or ident['name']+'_s50' not in REUSED_BY_BENCHMARK[b]:continue
            fb.check_sources({str(src):item['sha256'],**ident['diagnostic_sources']})
            policy=root/'policies'/b/f"{ident['name']}_s50.json"
            if read(ident['policy_path'])['policy']!=read(policy)['policy']:raise ValueError('Shared diagnostic policy changed')
            ident.update(fingerprint=contract['fingerprint'],policy_path=str(policy),policy_sha256=sha(policy.read_bytes()))
            frozen_write(root/'shared_diagnostics'/src.name,data)
    reuse_smoke(root,setup,contract,validation_sources)


def reuse_smoke(root,setup,contract,validation_sources):
    imports={};examples=[setup['calibration'][0],max(setup['development'],key=lambda r:len(r['prompt_tokens']))]
    dest=root/'validation'/contract['fingerprint'];aime_contract=read(AIME/'execution_contract.json')
    for original in examples:
        row=dict(original,generation_budget=16)
        src=AIME/'validation'/aime_contract['fingerprint']/(sha(row['id'])+'.native.json')
        if str(src) not in validation_sources:raise ValueError('Native smoke source not raw-audited')
        path=dest/src.name;frozen_write(path,read(src));imports[str(path)]=dict(source=str(src),sha256=sha(src.read_bytes()))
        for name,folder in (('jl_gaussian_r2',AIME),('jl_gaussian_r8',tiny.PREVIOUS)):
            fp=read(folder/'execution_contract.json')['fingerprint'];source=folder/'validation'/fp
            for unpruned in (True,False):
                label=f'{name}/{"unpruned" if unpruned else "pruned"}'
                policy={k:dict(log_threshold=-1000.,unpruned=True,tau=0.) if unpruned else dict(log_threshold=-2.) for k in ('local','global')}
                for stage in (('smoke',) if unpruned else ('smoke','reference')):
                    src=shard_path(source,stage,label,row['id'])
                    if str(src) not in validation_sources:raise ValueError('Sparse smoke source not raw-audited')
                    fb.alias(dest,row,stage,label,name,NEW[name],policy,contract,src,fp)
    frozen_write(root/'native_smoke_imports.json',imports)


def smoke_audit(root,setup,contract):
    with patch.object(backend,'PROJECTED',NEW):sources=PRIOR.BASE_SMOKE(root,setup,contract)
    inherited=read(root/'inherited_validation.json');fb.check_sources(inherited['sources']);sources.update(inherited['sources']);return sources


def proposals(root,contract):
    with dimension_scope(),patch.object(screen,'PROJECTED',NEW),patch.object(screen,'BASELINES',{}),patch.object(screen,'SENSITIVITY_SEEDS',()):
        return screen.shared_screen(root/'rank_proposals',read(root/'shared_state_index.json'),contract)


def warm_start(root,name,benchmark,target):
    if name not in NEW_BY_BENCHMARK[benchmark] or target!=.5:raise ValueError('Unexpected calibration request')
    path=imported_policy_path(benchmark,'jl_gaussian_r32');values,sources=screen.distributions(root/'rank_proposals',name,benchmark)
    sources[str(path)]=sha(path.read_bytes());return read(path),{k:np.maximum(v,-1e30) for k,v in values.items()},sources


def calibrate(adapter,root,setup,contract,name,benchmark):
    with dimension_scope(),patch.object(continuation,'NEW',NEW),patch.object(continuation,'audit_policy',audit_policy), \
         patch.object(continuation,'warm_start',lambda n,b,t:warm_start(root,n,b,t)),patch.object(fb,'calibration_rows',calibration_rows):
        return continuation.calibrate(adapter,root,setup,contract,name,benchmark,.5)


def freeze_conditions(root,setup,contract):
    sources={**contract['sources'],**smoke_audit(root,setup,contract),**read(root/'predecessor_audit.json')['sources']};conditions={}
    for label in CONDITIONS:
        name='dense' if label=='dense' else label.rsplit('_s',1)[0];thresholds={};own=dict(sources)
        for b in COUNTS:
            if label not in CONDITIONS_BY_BENCHMARK[b]:continue
            if name=='dense':thresholds[b]=None;continue
            path=root/'policies'/b/f'{label}.json'
            if not path.exists():continue
            p=read(path);audit_policy(root,p,setup,contract);thresholds[b]=p['policy'];own[str(path)]=sha(path.read_bytes())
        if not thresholds:continue
        c=dict(fingerprint=contract['fingerprint'],name=name,config={} if name=='dense' else CONFIGS[name],target=0. if name=='dense' else .5,
            thresholds=thresholds,sources=own,expected_per_benchmark={b:COUNTS[b] for b in thresholds})
        frozen_write(root/'final_configs'/f'{label}.json',c);conditions[label]=c
    return conditions


def accounting(root):
    a=tiny.previous.accounting(root)
    smoke=[read(p) for p in (root/'validation').glob('**/shards/*.json')]
    a.update(final_generation_slots=330,reused_final_baselines_and_references=930,
        reused_sparse_smoke_calls=sum(bool(o.get('imported_source')) for o in smoke),
        note='330 new final slots plus930 cached comparisons. Calibration and6 new short rank1 validation calls additional;12 compatible rank2/8 smoke calls and2 native calls reused. AIME selected calibration6 aliases final; LongBench18 selected calibration outputs likewise alias final.')
    _write(root/'inference_accounting.json',a);return a


def regenerate(root=ROOT):
    from experiments.diffusion_gemma_jl_lowrank_multibench_report import regenerate as report
    return report(root)


def verify(root=ROOT):
    before=read(root/'audit.json')
    if not before['complete'] or before['completed']!=EXPECTED:raise ValueError('Require all1260 results')
    fb.check_sources({str(root/p):h for p,h in before['artifacts'].items()})
    if regenerate(root)!=before:raise ValueError('Independent raw-only regeneration changed')
    result=dict(passed=True,completed=EXPECTED,inference_performed=False,audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json',result);return result


def work(root):
    setup,contract=prepare(root),execution(root);fb.phase(root,'reuse_cached_comparisons');reuse(root,setup,contract);accounting(root)
    torch.backends.cuda.matmul.allow_tf32=False;fb.phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',setup['model'],device='cuda',precision='bfloat16',revision=setup['revision']).load()
    with dimension_scope():
        fb.phase(root,'rank1_actual_model_smoke_with_cached_rank2_8')
        with patch.object(validation,'PROJECTED',NEW):validation.smoke(adapter,root,setup,contract)
        smoke_audit(root,setup,contract);accounting(root);fb.phase(root,'calibration_state_proposals');proposals(root,contract)
        for b,names in NEW_BY_BENCHMARK.items():
            for name in names:
                try:calibrate(adapter,root,setup,contract,name,b)
                except Exception:fb.failure(root,'calibration',method=name,benchmark=b,target=.5)
                accounting(root)
        conditions=freeze_conditions(root,setup,contract);fb.phase(root,'shared_state_diagnostics')
        try:
            with patch.object(shared_analysis,'PROJECTED',ALL),patch.object(shared_analysis,'BASELINES',BASELINES),patch.object(shared_analysis,'TARGETS',(.5,)):
                shared_analysis.analyze(root,read(root/'shared_state_index.json'),contract)
        except Exception:fb.failure(root,'shared_state_diagnostics')
        for b,names in NEW_BY_BENCHMARK.items():
            for name in names:
                label=name+'_s50'
                if label not in conditions or b not in conditions[label]['thresholds']:continue
                p=read(root/'policies'/b/f'{label}.json');selected=next(x for x in p['trace'] if x['iteration']==p['selected_round'])
                fb.phase(root,'final',benchmark=b,condition=label,expected=COUNTS[b])
                for row in sorted((r for r in setup['final'] if r['benchmark']==b),key=lambda r:(len(r['prompt_tokens']),r['id'])):
                    try:
                        if row['id'] in setup['calibration_ids'][b]:
                            fb.alias(root,row,'final',label,name,NEW[name],p['policy'],contract,Path(selected['sources'][row['id']]['path']),contract['fingerprint'])
                        else:runner.cached(adapter,root,row,'final',label,name,NEW[name],p['policy'],contract)
                    except Exception:fb.failure(root,'final',benchmark=b,condition=label,id=row['id'])
                accounting(root)
    del adapter;torch.cuda.empty_cache();fb.phase(root,'report');audit=regenerate(root)
    if audit['complete']:fb.phase(root,'independent_report_verification');verify(root)
    _write(root/'terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=EXPECTED,finished=time.time()))
    fb.phase(root,'finished',complete=audit['complete'])


def supervise(root):
    with patch.object(tiny.previous,'MODULE',MODULE):tiny.previous.supervise(root)


def launch(root):
    execution(root);fb.gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,new_final_slots=330,reused_slots=930,requested=NEW_BY_BENCHMARK,target=.5)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','launch','supervise','work','report','verify'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    result=(regenerate if args.command=='report' else globals()[args.command])(args.output)
    if args.command in ('prepare','report','verify'):print(json.dumps(dict(command=args.command,complete=result.get('complete'),passed=result.get('passed'))))
