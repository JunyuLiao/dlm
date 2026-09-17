"""Explicit user acceptance of one measured calibration point; no global relaxation."""
from copy import deepcopy
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

from experiments import diffusion_gemma_jl_fullbudget as fb
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write,_append
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle

ROOT=fb.ROOT
MODULE='experiments.diffusion_gemma_jl_accepted_round2'
READ=fb.read
BASE_AUDIT=fb.audit_policy
BASE_WITHIN=fb.within
BASE_FREEZE=fb.freeze_conditions
BASE_REPORT=fb.write_report
EXPECTED={'global':0.7255676461669522,'local':0.7663615396249244,'overall':0.7327736740713108}


def folder(root):return root/'accepted_round2'


def point(root):
    path=root/'calibration_traces/longbench_v2/full_centered_s75.json'
    trace=READ(path)
    selected=next(p for p in trace if p['iteration']==1)
    if selected['achieved']!=EXPECTED:raise ValueError('Round2 is not the point accepted by the user')
    return trace,selected,path


def is_exception(p):
    return (p.get('benchmark')=='longbench_v2' and p.get('name')=='full_centered'
        and p.get('config')==fb.CONFIG and p.get('target')==.75 and p.get('selected_round')==1
        and p.get('measured')==EXPECTED and p.get('user_accepted_tolerance_exception') is True)


def audit(root,p,setup,contract):
    if not p.get('user_accepted_tolerance_exception'):return BASE_AUDIT(root,p,setup,contract)
    if not is_exception(p):raise ValueError('Tolerance exception is restricted to the explicitly accepted point')
    _,selected,trace_path=point(root)
    approval=folder(root)/'acceptance.json'
    record=READ(approval)
    if record['selected_trace_sha256']!=fb.sha(trace_path.read_bytes()) or p['policy']!=selected['policy']:
        raise ValueError('Accepted point provenance changed')
    if p['acceptance_source']!={'path':str(approval),'sha256':fb.sha(approval.read_bytes())}:
        raise ValueError('Acceptance authorization source changed')
    # The original raw audit still checks every calibration ID, budget, source,
    # physical count and the selected jointly verified threshold. Only this
    # named point is allowed past the numerical tolerance predicate.
    with patch.object(fb,'within',lambda actual,target:BASE_WITHIN(actual,target) or (target==.75 and actual==EXPECTED)):
        BASE_AUDIT(root,p,setup,contract)


def prepare(root=ROOT):
    setup=fb.prepare(root);contract=fb.execution(root)
    trace,selected,path=point(root)
    test_path=folder(root)/'tests.xml'
    suites=ET.parse(test_path).getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=2,failures=0,errors=0,skipped=0):raise ValueError('Acceptance-extension tests incomplete')
    approval=folder(root)/'acceptance.json'
    fb.frozen_write(approval,dict(authorization='yes, Accept round 2 and start final evaluation',
        benchmark='longbench_v2',method='full_centered',target=.75,selected_iteration=1,
        measured=EXPECTED,strict_two_point_pass=False,global_target_miss_pp=100*(.75-EXPECTED['global']),
        selected_trace=str(path),selected_trace_sha256=fb.sha(path.read_bytes()),
        thresholds=selected['policy'],other_policies_unchanged=True,final_scores_used=False))
    dest=root/'policies/longbench_v2/full_centered_s75.json'
    if not dest.exists():
        old_path=fb.parent.OLD/'policies/longbench_v2/full_centered_s75.json'
        _,sources=fb.distributions(fb.parent.OLD,'full_centered','longbench_v2')
        sources[str(old_path)]=fb.sha(old_path.read_bytes())
        p=dict(fingerprint=contract['fingerprint'],benchmark='longbench_v2',name='full_centered',config=fb.CONFIG,target=.75,
            policy=selected['policy'],measured=selected['achieved'],trace=trace,selected_round=1,
            calibration_ids=[r['id'] for r in fb.calibration_rows(setup,'longbench_v2')],calibration_budget=4096,
            heldout_used=False,distribution_sources=sources,tolerance=.02,within_two_points=False,
            user_accepted_tolerance_exception=True,acceptance_source=dict(path=str(approval),sha256=fb.sha(approval.read_bytes())),
            rule='User explicitly accepted jointly verified full-budget round2 despite a2.443pp global miss. No additional fitting; never claim strict2pp success.')
        audit(root,p,setup,contract);fb.frozen_write(dest,p)
    else:audit(root,READ(dest),setup,contract)
    sources={str(p):fb.sha(p.read_bytes()) for p in (Path(__file__),Path('tests/test_jl_accepted_round2.py'),test_path,approval,dest)}
    extra=dict(parent_fingerprint=contract['fingerprint'],sources=sources,labels=list(fb.LABELS),
        calibration_complete_by_user_acceptance=True,algorithm_change=False,projected_methods_paused=True)
    fb.frozen_write(folder(root)/'extension_contract.json',extra)
    return setup,contract,extra


def freeze(root,setup,contract):
    extra=READ(folder(root)/'extension_contract.json');fb.check_sources(extra['sources'])
    extended=dict(contract,sources={**contract['sources'],**extra['sources'],
        str(folder(root)/'extension_contract.json'):fb.sha((folder(root)/'extension_contract.json').read_bytes())})
    return BASE_FREEZE(root,setup,extended)


def reused(root,setup,contract):
    fb.inherited_smoke(root,setup,contract)
    count=0
    for label in fb.BASELINES:
        stage='dense' if label=='dense' else 'final'
        for row in setup['final']:
            p=fb.shard_path(root,stage,label,row['id'])
            if not p.exists():raise ValueError('Previously completed baseline import is missing')
            count+=1
    if count!=560:raise ValueError('Baseline reuse scope mismatch')


def already_calibrated(adapter,root,setup,contract,benchmark,target):
    p=READ(root/'policies'/benchmark/f'full_centered_s{int(target*100)}.json')
    audit(root,p,setup,contract);return p


def report(root,setup,rows,tasks,compared,diag,checked,policies):
    BASE_REPORT(root,setup,rows,tasks,compared,diag,checked,policies)
    path=root/'report.md';text=path.read_text()
    text=text.replace('A policy is deployed only if overall/local/global summed-tile sparsity is within2 percentage points of target.',
        'Three policies pass the2-percentage-point calibration requirement. The user explicitly accepted LongBench75 round2 at73.28% overall,72.56% global,76.64% local; its2.44pp global miss is an exception, not a strict-tolerance pass.')
    text=text.replace('experiments.diffusion_gemma_jl_fullbudget','experiments.diffusion_gemma_jl_accepted_round2')
    text+='\n## Explicit calibration acceptance\n\nThe user accepted the already measured second full-budget LongBench75 calibration point and requested final evaluation. `accepted_round2/acceptance.json` pins the exact thresholds, raw trace and authorization. Refinement was stopped; the other three policies, all samples, budgets and numerical algorithms are unchanged. This exception does not change actual reported sparsity or permit tuning from final scores.\n'
    path.write_text(text)


def execute(root,command):
    prepare(root)
    with patch.object(fb,'audit_policy',audit),patch.object(fb,'freeze_conditions',freeze),\
         patch.object(fb,'reuse',reused),patch.object(fb,'calibrate',already_calibrated),patch.object(fb,'write_report',report):
        return {'work':fb.work,'report':fb.regenerate,'verify':fb.verify}[command](root)


def supervise(root):
    with (folder(root)/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_idle()
        with (folder(root)/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(folder(root)/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:code=child.wait(timeout=1800);break
                except subprocess.TimeoutExpired:
                    state={k:READ(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase','progress')}
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    _append(folder(root)/'monitor.jsonl',dict(time=time.time(),pid=child.pid,gpu=gpu.stdout.strip(),**state))
            _write(folder(root)/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    prepare(root);gpu_idle()
    with (folder(root)/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,accepted_round=2,final_reference_runs=160,projected_methods_paused=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','supervise','work','report','verify'))
    p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args()
    if a.command in ('work','report','verify'):print(json.dumps(execute(a.output,a.command)))
    else:globals()[a.command](a.output)
