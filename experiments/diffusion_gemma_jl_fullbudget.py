"""Full-budget recalibration, reference first; no frozen prior source is edited."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
from dllm.models import create_adapter
from experiments.diffusion_gemma_jl_focused import protocol as parent,report as report_backend
from experiments.diffusion_gemma_jl_focused.reuse import dispatch,audit_policy as parent_audit
from experiments.diffusion_gemma_jl_output_aware import runner,shared_analysis
from experiments.diffusion_gemma_jl_output_aware.screen import distributions
from experiments.diffusion_gemma_jl_output_aware.calibration import audit_policy as old_audit
from experiments.diffusion_gemma_value_aware.rank_calibration import next_policy as rank_next
from experiments.diffusion_gemma_value_aware.policy_search import next_policy as scalar_next,select_point
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware_followup.policies import measurements
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append,_fingerprint
from experiments import diffusion_gemma_jl_reference_first as reference_report

ROOT=Path('results/diffusion_gemma_jl_fullbudget_reference_v3')
PREVIOUS=parent.ROOT
MODULE='experiments.diffusion_gemma_jl_fullbudget'
LABELS=reference_report.LABELS
CONDITIONS=reference_report.REPORT_LABELS
BASELINES=reference_report.BASELINES
CONFIG={'family':'identity'}
MAX_POINTS=12
TOLERANCE=.02
read,sha,frozen_write=parent.read,parent.sha,parent.frozen_write


def calibration_rows(setup,benchmark):
    rows=[r for r in setup['calibration'] if r['benchmark']==benchmark]
    budget=2048 if benchmark=='aime26' else 4096
    if len(rows)!=6 or any(r['split']!='calibration' or r['generation_budget']!=budget for r in rows):
        raise ValueError('Require six fixed full-budget calibration rows')
    if benchmark=='longbench_v2':
        for key in ('id','prompt_hash'):
            if {r[key] for r in rows}&{r[key] for r in setup['final']+setup['development']}:
                raise ValueError('LongBench calibration contamination')
    return rows


def prepare(root=ROOT):
    previous=parent.prepare(PREVIOUS)
    setup=deepcopy(previous)
    setup.update(schema='jl_fullbudget_reference_v3',conditions=list(CONDITIONS),
        configs={**parent.BASELINES,'full_centered':CONFIG},projected_methods_paused=True,
        calibration_policy='Same six calibration IDs per benchmark, now full generation budgets2048 AIME/4096 LongBench. Existing scalar local/global empirical-rank refinement, generic existing scalar bracket fallback if CDF proposals stall; at most12 jointly verified points. Deploy only a point within2pp in overall/local/global physical count-weighted sparsity. No final-score fitting.',
        policy_selection_scores_used=False,baseline_source=str(PREVIOUS),
        old_policy_role='Warm starts only;512-token verification is not accepted. Full-budget raw calibration points may be reused if exactly matching.',
        failure_policy='Preserve failed calibration/final attempts; continue independent targets/benchmarks. Do not deploy a policy missing the calibration tolerance. Projected methods stay paused pending corrected full-dimensional results.')
    for benchmark in ('aime26','longbench_v2'):calibration_rows(setup,benchmark)
    path=root/'setup.json'
    frozen_write(path,setup)
    for split in ('calibration','development','final'):frozen_write(root/f'{split}_manifest.json',setup[split])
    return setup


def execution(root=ROOT):
    setup=prepare(root);previous=parent.execution(PREVIOUS)
    files=[Path(__file__),Path(reference_report.__file__),Path('tests/test_jl_fullbudget.py'),root/'tests.xml',
        PREVIOUS/'execution_contract.json',PREVIOUS/'setup.json',PREVIOUS/'test_gate.json']
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=4,failures=0,errors=0,skipped=0):raise ValueError(f'Full-budget tests incomplete: {counts}')
    sources={**previous['sources'],**{str(p):sha(p.read_bytes()) for p in files}}
    check_sources(sources)
    data=dict(schema='jl_fullbudget_execution_v3',parent_fingerprint=previous['fingerprint'],
        setup_sha256=sha((root/'setup.json').read_bytes()),sources=sources,runtime=previous['runtime'],
        unchanged_inference_algorithms=True,hardware_speedup_claim=False,calibration_budgets={'aime26':2048,'longbench_v2':4096},
        calibration_tolerance=TOLERANCE,max_verified_points=MAX_POINTS)
    data['fingerprint']=_fingerprint(data);frozen_write(root/'execution_contract.json',data)
    return data


def alias(root,row,stage,label,name,cfg,threshold,contract,source,source_fp):
    """Reuse only identical prompts, full budgets, operator and thresholds."""
    path=shard_path(root,stage,label,row['id'])
    if path.exists():return runner.cached(None,root,row,stage,label,name,cfg,threshold,contract)
    out=runner.load_output(source);check_result(out,row,source_fp,cfg,threshold)
    out.update(fingerprint=contract['fingerprint'],candidate=name,
        imported_source=dict(path=str(source),sha256=sha(source.read_bytes()),original_fingerprint=source_fp))
    check_result(out,row,contract['fingerprint'],cfg,threshold)
    if out.get('records_source'):
        # Lossless existing gzip buckets are immutable and need not be copied.
        saved=dict(out);saved.pop('records');_write(path,saved)
    else:runner.write_output(path,out)
    return out


def measured(outputs):
    metrics,achieved=measurements(outputs,{'method':'centered'})
    rs=[r for o in outputs for r in o['records'] if r['probe']=='execution']
    achieved['overall']=sum(r['skipped'] for r in rs)/sum(r['eligible'] for r in rs)
    return metrics,achieved


def within(achieved,target):
    return all(abs(achieved[k]-target)<=TOLERANCE+1e-12 for k in ('overall','local','global'))


def audit_policy(root,p,setup,contract):
    if p['fingerprint']!=contract['fingerprint'] or p['heldout_used']:raise ValueError('Policy identity or final-data fitting violation')
    if p.get('imported_baseline'):
        src=Path(p['imported_baseline']);check_sources(p['sources']);old=read(src)
        parent_audit(PREVIOUS,old,parent.prepare(PREVIOUS),parent.execution(PREVIOUS))
        if (p['config'],p['policy'])!=(old['config'],old['policy']):raise ValueError('Baseline policy changed')
        return
    rows=calibration_rows(setup,p['benchmark'])
    if p['calibration_ids']!=[r['id'] for r in rows]:raise ValueError('Calibration sample identity changed')
    check_sources(p['distribution_sources'])
    for point in p['trace']:
        if set(point['sources'])!={r['id'] for r in rows}:raise ValueError('Incomplete full-budget calibration point')
        outputs=[]
        for row in rows:
            src=point['sources'][row['id']];path=Path(src['path'])
            if sha(path.read_bytes())!=src['sha256']:raise ValueError('Calibration raw shard changed')
            out=runner.load_output(path);check_result(out,row,contract['fingerprint'],CONFIG,point['policy'])
            if out.get('imported_source'):check_sources({out['imported_source']['path']:out['imported_source']['sha256']})
            outputs.append(out)
        _,actual=measured(outputs)
        if actual!=point['achieved']:raise ValueError('Reported calibration sparsity differs from summed physical counts')
    selected=next(x for x in p['trace'] if x['iteration']==p['selected_round'])
    if p['policy']!=selected['policy'] or p['measured']!=selected['achieved'] or not within(p['measured'],p['target']):
        raise ValueError('Unverified or off-target policy cannot be deployed')


def reuse(root,setup,contract):
    previous=parent.execution(PREVIOUS);sources={}
    for label in BASELINES:
        c=read(PREVIOUS/'final_configs'/f'{label}.json');check_sources(c['sources']);sources.update(c['sources'])
        for benchmark in ('aime26','longbench_v2'):
            if label=='dense':continue
            src=PREVIOUS/'policies'/benchmark/f'{label}.json';old=read(src)
            p=dict(fingerprint=contract['fingerprint'],benchmark=benchmark,name=c['name'],config=c['config'],target=c['target'],
                policy=c['thresholds'][benchmark],heldout_used=False,imported_baseline=str(src),
                sources={str(src):sha(src.read_bytes())},measured=old.get('measured'),
                calibration_budget='Unchanged historical full-budget baseline calibration; no refit',
                cap_one_unattainable=old.get('cap_one_unattainable'))
            audit_policy(root,p,setup,contract);frozen_write(root/'policies'/benchmark/f'{label}.json',p)
        for row in setup['final']:
            stage='dense' if label=='dense' else 'final'
            alias(root,row,stage,label,c['name'],c['config'],c['thresholds'][row['benchmark']],contract,
                shard_path(PREVIOUS,stage,label,row['id']),previous['fingerprint'])
        print('reused baseline',label,80,flush=True)
    # Reuse previously verified actual-model parity: algorithms, prompt policy,
    # runtime and source hashes are unchanged; only calibration/scheduling differ.
    validation=report_backend.smoke_audit(PREVIOUS,parent.prepare(PREVIOUS),previous)
    frozen_write(root/'inherited_validation.json',dict(passed=True,sources=validation,
        prior_fingerprint=previous['fingerprint'],note='Raw-audited two-example native/unpruned and pruned trusted-reference parity; unchanged numerical code'))
    states=read(PREVIOUS/'shared_state_index.json')
    check_sources({r['path']:r['sha256'] for r in states});frozen_write(root/'shared_state_index.json',states)


def proposal(trace,target,values):
    candidate=rank_next(trace,target,values)
    if any(candidate==x['policy'] for x in trace):candidate=scalar_next(trace,target)
    return candidate


def calibrate(adapter,root,setup,contract,benchmark,target):
    label=f'full_centered_s{int(target*100)}';dest=root/'policies'/benchmark/f'{label}.json'
    if dest.exists():p=read(dest);audit_policy(root,p,setup,contract);return p
    rows=calibration_rows(setup,benchmark)
    old_path=parent.OLD/'policies'/benchmark/f'{label}.json';old=read(old_path)
    old_audit(parent.OLD,old,read(parent.OLD/'setup.json'),read(parent.OLD/'execution_contract.json'))
    values,dist_sources=distributions(parent.OLD,'full_centered',benchmark)
    values={k:np.maximum(v,-1e30) for k,v in values.items()}
    dist_sources[str(old_path)]=sha(old_path.read_bytes())
    trace_path=root/'calibration_traces'/benchmark/f'{label}.json'
    trace=read(trace_path) if trace_path.exists() else []
    previous=parent.execution(PREVIOUS)
    for iteration in range(len(trace),MAX_POINTS):
        if trace:
            best,_=select_point(trace,target)
            if within(best['achieved'],target):break
        policy=deepcopy(old['policy']) if not trace else proposal(trace,target,values)
        if any(policy==p['policy'] for p in trace):break
        tag=f'{label}/{benchmark}/{_fingerprint(policy)[:16]}'
        phase(root,'full_budget_calibration',benchmark=benchmark,target=target,point=iteration+1,
            generation_budget=rows[0]['generation_budget'],previous_achieved=trace[-1]['achieved'] if trace else None)
        outputs=[];sources={}
        for row in rows:
            path=shard_path(root,'calibration',tag,row['id'])
            # Only the six calibration IDs may borrow their existing full-budget
            # generations. No score is read for threshold selection.
            source=shard_path(PREVIOUS,'final',label,row['id'])
            if not path.exists() and source.exists():
                saved=runner.load_output(source)
                try:check_result(saved,row,previous['fingerprint'],CONFIG,policy)
                except ValueError:pass
                else:alias(root,row,'calibration',tag,'full_centered',CONFIG,policy,contract,source,previous['fingerprint'])
            out=runner.cached(adapter,root,row,'calibration',tag,'full_centered',CONFIG,policy,contract)
            outputs.append(out);sources[row['id']]=dict(path=str(path),sha256=sha(path.read_bytes()))
        metrics,achieved=measured(outputs)
        trace.append(dict(iteration=iteration,policy=policy,achieved=achieved,metrics=metrics,sources=sources,
            source_ids=[r['id'] for r in rows],source_condition=tag))
        _write(trace_path,trace);print('full-budget verified',benchmark,target,iteration+1,achieved,flush=True)
    best,_=select_point(trace,target)
    if not within(best['achieved'],target):
        _write(root/'calibration_failures'/benchmark/f'{label}.json',dict(target=target,trace=trace,
            best_achieved=best['achieved'],deploy=False,reason='No jointly verified full-budget point within2pp; do not label nearest point as target-achieving'))
        raise RuntimeError(f'Full-budget calibration failed tolerance: {benchmark}/{label} {best["achieved"]}')
    p=dict(fingerprint=contract['fingerprint'],benchmark=benchmark,name='full_centered',config=CONFIG,target=target,
        policy=best['policy'],measured=best['achieved'],trace=trace,selected_round=best['iteration'],
        calibration_ids=[r['id'] for r in rows],calibration_budget=rows[0]['generation_budget'],heldout_used=False,
        distribution_sources=dist_sources,tolerance=TOLERANCE,
        rule='Existing empirical-rank proposals, existing scalar bracket fallback on repeated proposal; full-budget sparse-trajectory verification, max12 joint points. Only achieved physical sparsity selects thresholds; never accuracy.')
    audit_policy(root,p,setup,contract);frozen_write(dest,p);return p


def freeze_conditions(root,setup,contract):
    result={}
    for label in CONDITIONS:
        name='dense' if label=='dense' else label.rsplit('_s',1)[0]
        target=0. if label=='dense' else int(label.rsplit('_s',1)[1])/100
        cfg={} if name=='dense' else CONFIG if name=='full_centered' else parent.BASELINES[name]
        thresholds={};sources=dict(contract['sources']);missing=[]
        for benchmark in ('aime26','longbench_v2'):
            if name=='dense':thresholds[benchmark]=None;continue
            path=root/'policies'/benchmark/f'{label}.json'
            if not path.exists():missing.append(benchmark);continue
            p=read(path);audit_policy(root,p,setup,contract);thresholds[benchmark]=p['policy'];sources[str(path)]=sha(path.read_bytes())
        c=dict(fingerprint=contract['fingerprint'],name=name,target=target,config=cfg,thresholds=thresholds,sources=sources,
            unavailable_benchmarks=missing,expected_per_benchmark={'aime26':30,'longbench_v2':50})
        if not missing:frozen_write(root/'final_configs'/f'{label}.json',c)
        else:_write(root/'pending_configs'/f'{label}.json',c)
        result[label]=c
    return result


def inherited_smoke(root,setup,contract):
    p=read(root/'inherited_validation.json')
    if not p['passed']:raise ValueError('Inherited validation missing')
    check_sources(p['sources'])
    return {**p['sources'],str(root/'inherited_validation.json'):sha((root/'inherited_validation.json').read_bytes())}


def write_report(root,setup,rows,tasks,compared,diag,audit,policies):
    reference_report.write_reference_report(root,setup,rows,tasks,compared,diag,audit,policies)
    path=root/'report.md';text=path.read_text()
    text=text.replace('# Full-dimensional centered reference: feasibility gate','# Full-dimensional centered reference: full-budget calibration v3')
    text=text.replace('Full-centered calibrated local/global scalar thresholds are reused unchanged from the completed JL calibration:6 questions/benchmark,512-token verification; LongBench calibration spans3 domains.',
        'Full-centered local/global scalar thresholds are newly verified on the same6 calibration questions at2048 AIME/4096 LongBench output budgets. A policy is deployed only if overall/local/global summed-tile sparsity is within2 percentage points of target. The existing empirical-rank search and scalar bracket fallback are allowed at most12 joint verification points. LongBench calibration spans3 domains. Old512-token policies are warm starts only, never accepted on short-budget evidence.')
    text=text.replace('The39 earlier projected outputs and all prior logs are preserved outside this scoped report.',
        'The39 earlier projected outputs,34 earlier full-centered outputs and all prior logs are preserved in the v2 bundle; they are not mixed with corrected-policy final results. Exact full-budget calibration or baseline aliases retain original source hashes.')
    text=text.replace('experiments.diffusion_gemma_jl_reference_first','experiments.diffusion_gemma_jl_fullbudget')
    text+='\n## Why recalibration was necessary\n\nOn the identical six AIME calibration examples, the previous50% reference policy measured51.10% overall at512 tokens, but62.51% at2048 tokens (global71.50%, local60.27%). All six first512 generated-token prefixes match. This is later-trajectory sparsity drift, not changed decoding. New full-budget verification addresses that mismatch; disjoint final LongBench achieved sparsity can still differ and is never retuned using final scores. Shared-QKV diagnostics reuse early512-token dense calibration states and therefore do not exhaustively cover later generation; full-run mass/error/count metrics cover actual denoising calls.\n'
    path.write_text(text)


def regenerate(root=ROOT):
    with patch.object(report_backend,'prepare',prepare),patch.object(report_backend,'execution',execution),\
         patch.object(report_backend,'PROJECTED',{'full_centered':CONFIG}),patch.object(report_backend,'audit_policy',audit_policy),\
         patch.object(report_backend,'smoke_audit',inherited_smoke),patch.object(report_backend.common,'shared_seed_summary',lambda unused:([],{})),\
         patch.object(report_backend,'write_report',write_report):
        return report_backend.regenerate(root)


def verify(root=ROOT):
    before=read(root/'audit.json')
    if not before['complete'] or before['completed']!=720:raise ValueError('Require720 completed corrected-policy results')
    check_sources({str(root/p):v for p,v in before['artifacts'].items()})
    after=regenerate(root)
    if before!=after:raise ValueError('Full-budget raw-only report regeneration changed')
    proof=dict(passed=True,completed=720,inference_performed=False,audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json',proof);return proof


def phase(root,stage,**details):
    _write(root/'phase.json',dict(stage=stage,started=time.time(),**details));print('phase',stage,details,flush=True)


def failure(root,stage,**details):
    d=dict(stage=stage,**details,traceback=traceback.format_exc());_append(root/'failures.jsonl',d);print(d,flush=True)
    torch.cuda.empty_cache()


def work(root):
    setup=prepare(root);contract=execution(root)
    phase(root,'reuse_exact_baselines_and_validation');reuse(root,setup,contract)
    torch.backends.cuda.matmul.allow_tf32=False;phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',parent.MODEL,device='cuda',precision='bfloat16',revision=parent.REVISION).load()
    with dispatch():
        for target in (.5,.75):
            for benchmark in ('aime26','longbench_v2'):
                try:calibrate(adapter,root,setup,contract,benchmark,target)
                except Exception:failure(root,'calibration',benchmark=benchmark,target=target)
        conditions=freeze_conditions(root,setup,contract)
        phase(root,'shared_operator_diagnostics')
        try:
            with patch.object(shared_analysis,'PROJECTED',{'full_centered':CONFIG}),patch.object(shared_analysis,'BASELINES',parent.BASELINES):
                shared_analysis.analyze(root,read(root/'shared_state_index.json'),contract)
        except Exception:failure(root,'shared_operator_diagnostics')
        for label in LABELS:
            c=conditions[label];phase(root,'final',condition=label,expected=80)
            for row in sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id'])):
                if row['benchmark'] not in c['thresholds']:continue
                try:
                    threshold=c['thresholds'][row['benchmark']]
                    if row['benchmark']=='aime26' and row['calibration']:
                        p=read(root/'policies/aime26'/f'{label}.json');point=next(x for x in p['trace'] if x['iteration']==p['selected_round'])
                        alias(root,row,'final',label,'full_centered',CONFIG,threshold,contract,
                            Path(point['sources'][row['id']]['path']),contract['fingerprint'])
                    else:runner.cached(adapter,root,row,'final',label,'full_centered',CONFIG,threshold,contract)
                except Exception:failure(root,'final',condition=label,id=row['id'])
    del adapter;torch.cuda.empty_cache();phase(root,'report')
    audit=regenerate(root)
    if audit['complete']:phase(root,'independent_report_verification');verify(root)
    _write(root/'terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=720,
        projected_methods_paused=True,finished=time.time()))
    phase(root,'await_reference_decision',complete=audit['complete'])


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
        with (root/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:code=child.wait(timeout=1800);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    state={k:read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase','progress')}
                    _append(root/'monitor.jsonl',dict(time=time.time(),pid=child.pid,gpu=gpu.stdout.strip(),disk_free_bytes=shutil.disk_usage(root).free,**state))
            _write(root/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    execution(root);gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,output=str(root),full_budget_calibration=True,projected_methods_paused=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','launch','supervise','work','report','verify'))
    p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='report':print(json.dumps(regenerate(a.output)))
    elif a.command=='verify':print(json.dumps(verify(a.output)))
    else:globals()[a.command](a.output)
