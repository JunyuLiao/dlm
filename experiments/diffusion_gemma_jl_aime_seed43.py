"""Queued seed-sensitivity replication; frozen operators and seed42 policies."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
from dllm.models import create_adapter
from experiments import diffusion_gemma_jl_lowrank_multibench_v2 as parent
from experiments import diffusion_gemma_jl_lowrank_multibench_v2_report as reporting
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint

ROOT=Path('results/diffusion_gemma_jl_aime_seed43_v12')
MODULE='experiments.diffusion_gemma_jl_aime_seed43'
PREVIOUS=parent.AIME
SEED=43
ALL={n:parent.ALL[n] for n in ('full_centered','jl_gaussian_r8')}
CONFIGS={n:parent.CONFIGS[n] for n in ('blasst_aggressive','mass','full_centered','jl_gaussian_r8')}
CONDITIONS=['dense']+[n+'_s50' for n in CONFIGS]
CONDITIONS_BY_BENCHMARK={'aime26':CONDITIONS}
NEW_BY_BENCHMARK={'aime26':list(CONFIGS)}
COUNTS={'aime26':30}
EXPECTED,TEST_COUNT=150,9
read,sha,frozen_write=parent.read,parent.sha,parent.frozen_write
fb,backend,runner=parent.fb,parent.backend,parent.runner
shared_analysis,validation,shard_path=parent.shared_analysis,parent.validation,parent.shard_path
dimension_scope=parent.dimension_scope


def prepare(root=ROOT):
    old=read(PREVIOUS/'setup.json')
    rows=[dict(deepcopy(r),seed=SEED) for r in old['final']]
    if len(rows)!=30 or any(r['benchmark']!='aime26' or r['generation_budget']!=2048 or sha(r['prompt'])!=r['prompt_hash'] for r in rows):
        raise ValueError('Require unchanged30 AIME prompts and2048 budget')
    if len({r['id'] for r in rows})!=30 or len({r['prompt_hash'] for r in rows})!=30:raise ValueError('Duplicate final examples')
    ids=deepcopy(old['calibration_ids']);cal=[r for r in rows if r['id'] in ids['aime26']]
    if len(cal)!=6:raise ValueError('Historical calibration membership changed')
    development=[max((r for r in rows if r['id']!=cal[0]['id']),key=lambda r:len(r['prompt_tokens']))]
    data=dict(schema='jl_aime_seed43_v12',**{k:deepcopy(old[k]) for k in ('model','revision','precision','tile_size','regions','decoding','input_budget')},
        final=rows,calibration=cal,development=development,calibration_ids=ids,calibration_overlap=ids,
        conditions=CONDITIONS,conditions_by_benchmark=CONDITIONS_BY_BENCHMARK,configs=CONFIGS,targets=[.5],expected_counts=COUNTS,
        generation_seed=SEED,previous_generation_seed=42,projection_seed=1729,projection_family='gaussian',projection_dimension=8,
        policy_source=str(PREVIOUS),queue_after=str(parent.ROOT),queue_parent_fingerprint=read(parent.ROOT/'execution_contract.json')['fingerprint'],
        calibration_policy='No new calibration. Exact previous seed42 local/global policies for each method, including any original length adjustment. Measure achieved seed43 sparsity; do not retune.',
        validation_policy='Reuse audited unchanged kernel tests and seed42 raw smoke; fresh seed43 rank8 native/unpruned and trusted/pruned checks on two AIME inputs before final inference.',
        exposure='Same previously examined30 questions, including six historical calibration IDs. No fresh held-out claim. Only generation seed changes; projection directions remain fixed.',
        final_generation_slots=150,reused_final_slots=0,monitoring_interval_seconds=900)
    frozen_write(root/'setup.json',data)
    for split in ('final','calibration','development'):frozen_write(root/f'{split}_manifest.json',data[split])
    frozen_write(root/'dataset_audit.json',dict(passed=True,expected=EXPECTED,only_row_change='seed42->43',
        exact_previous_ids=True,budget=2048,historical_calibration_ids=ids['aime26'],fresh_heldout=False))
    return data


def execution(root=ROOT):
    import triton,transformers
    prepare(root);old=read(PREVIOUS/'execution_contract.json')
    runtime=dict(torch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__)
    if runtime!=old['runtime']:raise ValueError('Runtime changed')
    tests=root/'tests.xml';suites=ET.parse(tests).getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=TEST_COUNT,failures=0,errors=0,skipped=0):raise ValueError(f'Incomplete CPU orchestration tests: {counts}')
    files=[Path(__file__),Path(reporting.__file__),Path(parent.__file__),Path('tests/test_jl_aime_seed43.py'),tests,
        root/'setup.json',PREVIOUS/'execution_contract.json',PREVIOUS/'per_sample.json',parent.ROOT/'execution_contract.json']
    sources=backend.evidence.merge_sources(old['sources'],{str(p):sha(p.read_bytes()) for p in files});fb.check_sources(sources)
    data=dict(schema='jl_aime_seed43_execution_v12',sources=sources,runtime=runtime,previous_fingerprint=old['fingerprint'],
        setup_sha256=sha((root/'setup.json').read_bytes()),generation_seed=SEED,projection_seed=1729,
        algorithms_and_kernels_unchanged=True,no_threshold_refitting=True,expected=EXPECTED,
        projection_dtype='float32',matmul_allow_tf32=False,hardware_speedup_claim=False)
    data['fingerprint']=_fingerprint(data);frozen_write(root/'execution_contract.json',data);return data


def source_policy(name):return PREVIOUS/'policies/aime26'/f'{name}_s50.json'


def audit_policy(root,p,setup,contract):
    if p['benchmark']!='aime26' or p['name'] not in CONFIGS or p['target']!=.5 or p['heldout_used'] or p['fingerprint']!=contract['fingerprint']:
        raise ValueError('Invalid imported policy identity')
    path=source_policy(p['name']);old=read(path)
    if p['imported_policy']!=str(path) or p['config']!=CONFIGS[p['name']]:raise ValueError('Wrong policy source/operator')
    fb.check_sources(p['sources'])
    if any(p[k]!=old[k] for k in ('name','benchmark','target','config','policy','measured')):raise ValueError('Previously calibrated threshold changed')


def reuse(root,setup,contract):
    proof=parent.remaining.require_complete(PREVIOUS,330)
    old_setup,old_contract=read(PREVIOUS/'setup.json'),read(PREVIOUS/'execution_contract.json')
    sources=parent.tiny.smoke_audit(PREVIOUS,old_setup,old_contract)
    frozen_write(root/'inherited_validation.json',dict(passed=True,sources=sources,seed=42))
    for name in CONFIGS:
        path=source_policy(name);old=read(path);parent.tiny.audit_policy(PREVIOUS,old,old_setup,old_contract)
        p={k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
        p.update(fingerprint=contract['fingerprint'],heldout_used=False,imported_policy=str(path),
            sources={str(path):sha(path.read_bytes()),**proof},calibration_generation_seed=42,evaluation_generation_seed=SEED,
            calibration_ids=old.get('calibration_ids'),cap_one_unattainable=old.get('cap_one_unattainable'))
        audit_policy(root,p,setup,contract);frozen_write(root/'policies/aime26'/f'{name}_s50.json',p)
    frozen_write(root/'predecessor_audit.json',dict(passed=True,reused_outputs=0,sources=proof))
    states=read(PREVIOUS/'shared_state_index.json');fb.check_sources({s['path']:s['sha256'] for s in states})
    frozen_write(root/'shared_state_index.json',states)
    imported=[]
    for item in read(PREVIOUS/'shared_diagnostics_index.json'):
        src=Path(item['path']);data=read(src);ident=data['identity']
        if ident['name'] not in CONFIGS:continue
        fb.check_sources({str(src):item['sha256'],**ident['diagnostic_sources']})
        path=root/'policies/aime26'/f"{ident['name']}_s50.json"
        if read(ident['policy_path'])['policy']!=read(path)['policy']:raise ValueError('Diagnostic threshold mismatch')
        ident.update(fingerprint=contract['fingerprint'],policy_path=str(path),policy_sha256=sha(path.read_bytes()))
        dest=root/'shared_diagnostics'/src.name;frozen_write(dest,data);imported.append(dict(path=str(dest),sha256=sha(dest.read_bytes())))
    frozen_write(root/'shared_diagnostics_index.json',imported)
    reporting_study=reporting.study
    with patch.object(reporting,'study',sys.modules[__name__]):reporting.audit_diagnostics(root,setup)
    assert reporting.study is reporting_study


def smoke_audit(root,setup,contract):
    folder=root/'validation'/contract['fingerprint'];summary=folder/'smoke.json';proof=read(summary)
    if not proof['passed'] or proof['fingerprint']!=contract['fingerprint'] or len(proof['cases'])!=4:raise ValueError('Incomplete seed43 smoke')
    sources={str(summary):sha(summary.read_bytes())}
    for original in (setup['calibration'][0],setup['development'][0]):
        row=dict(original,generation_budget=16);p=folder/(sha(row['id'])+'.native.json');native=read(p);sources[str(p)]=sha(p.read_bytes())
        for unpruned in (True,False):
            name='jl_gaussian_r8';label=f'{name}/{"unpruned" if unpruned else "pruned"}'
            threshold={k:dict(log_threshold=-1000.,unpruned=True,tau=0.) if unpruned else dict(log_threshold=-2.) for k in ('local','global')}
            outputs={}
            for stage in (('smoke',) if unpruned else ('smoke','reference')):
                out=runner.cached(None,folder,row,stage,label,name,ALL[name],threshold,contract);outputs[stage]=out
                path=shard_path(folder,stage,label,row['id']);sources[str(path)]=sha(path.read_bytes());sources[out['records_source']['path']]=out['records_source']['sha256']
            out=outputs['smoke']
            if out.get('imported_source') or out['seed']!=SEED or not out['kernel_validation'] or not out['finite_calls']:raise ValueError('Missing fresh finite seed43 validation')
            if unpruned:
                if out['completion_tokens']!=native['completion_tokens'] or any(r['skipped'] for r in out['records']):raise ValueError('Unpruned native parity failed')
            else:
                ref=outputs['reference']
                if out['completion_tokens']!=ref['completion_tokens']:raise ValueError('Trusted parity failed')
                for kind in ('local','global'):
                    for k in ('eligible','skipped'):
                        if sum(r[k] for r in out['records'] if r['attention_type']==kind)!=sum(r[k] for r in ref['records'] if r['attention_type']==kind):raise ValueError('Physical counts mismatch')
            for key in backend.evidence.DECODING_FIELDS:
                if out['generation_metadata'].get(key)!=native['generation_metadata'].get(key):raise ValueError('Unrelated decoding changed')
    inherited=read(root/'inherited_validation.json');fb.check_sources(inherited['sources']);sources.update(inherited['sources']);return sources


def freeze_conditions(root,setup,contract):
    with patch.object(parent,'CONDITIONS',CONDITIONS),patch.object(parent,'CONFIGS',CONFIGS),patch.object(parent,'COUNTS',COUNTS), \
         patch.object(parent,'CONDITIONS_BY_BENCHMARK',CONDITIONS_BY_BENCHMARK),patch.object(parent,'smoke_audit',smoke_audit),patch.object(parent,'audit_policy',audit_policy):
        return parent.freeze_conditions(root,setup,contract)


def accounting(root):
    data=parent.tiny.previous.accounting(root)
    native=len(list((root/'validation').glob('*/[0-9a-f]*.native.json')))
    data.update(final_generation_slots=150,reused_final_baselines_and_references=0,new_native_smoke_calls=native,
        completed_new_inference=data['completed_new_inference']+native,
        note='150 fresh seed43 final calls, no calibration calls or cached seed42 final aliases; six sparse and two native short smoke calls additional. Shared diagnostics reuse old seed42 QKV.')
    _write(root/'inference_accounting.json',data);return data


def seed_comparisons(current,previous):
    from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import paired_bootstrap_ci
    result=[]
    for label in CONDITIONS:
        now={r['id']:r for r in current if r['condition']==label};old={r['id']:r for r in previous if r['condition']==label}
        if len(now)!=30 or set(now)!=set(old):raise ValueError('Require paired30 questions for seed comparison')
        for i,r in now.items():
            if r['seed']!=SEED or old[i]['seed']!=42 or any(r[k]!=old[i][k] for k in ('prompt_hash','generation_budget')):raise ValueError('Seed comparison identity mismatch')
        differences=[now[i]['accuracy']-old[i]['accuracy'] for i in sorted(now)]
        result.append(dict(condition=label,count=30,seed42_correct=sum(r['accuracy'] for r in old.values()),seed43_correct=sum(r['accuracy'] for r in now.values()),
            delta_pp=100*sum(differences)/30,paired_ci95_pp=[100*v for v in paired_bootstrap_ci(differences)]))
    return result


def write_report(root,setup,rows,tasks,direct,diag,audit):
    core=reporting.core
    display=[dict(split=r['split'],condition=r['condition'],threshold=core.threshold_text(r),correct=r['correct'],count=r['count'],
        accuracy=100*r['accuracy'],delta_pp=100*r['delta'],ci95_pp=[100*v for v in r['paired_ci95']],
        whole=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_mass'],agreement=100*r['token_agreement'],local_operator_error=r['overall_relative_error']) for r in rows]
    core.csv_write(root/'main_results.csv',display)
    text=f"# AIME26 generation-seed43 replication\n\nStatus: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/{EXPECTED} audited outputs.\n\n"
    text+='Same30 AIME26 questions and2048-token output budgets, pinned BF16 DiffusionGemma revision '+setup['revision']+'. Generation seed43 instead of42 for every condition; projection seed1729 and all Gaussian8 matrices unchanged. Native0.4–0.8 sampling schedule, canvas256, max48 denoising steps, thinkingFalse; temperature0 sentinel is NOT greedy.128-query×64-KV tiles, prefix+canvas eligible, native masks and GQA.\n\n'
    text+='All150 final outputs are fresh and paired against the new seed43 dense outputs. No seed42 outputs are substituted. Exact previous seed42-calibrated local/global policies are frozen without refitting; achieved sparsity can drift. Aggressive BLASST retains its existing lambda/length rule allowing lambda>1. Mass is the existing max-based mass bound. Full and Gaussian8 use unchanged attention-weighted centered updates, first-support/tie retention and ordinary retained attention renormalization.\n\n'
    text+='## Seed43 results\n\n'
    for split in ('full','noncalibration24','calibration6'):
        text+=f'### {split}\n\n'+core.table([r for r in display if r['split']==split],tuple(display[0]) if display else ())+'\n\n'
    if audit['complete']:
        comparison=seed_comparisons(read(root/'per_sample.json'),read(PREVIOUS/'per_sample.json'))
        _write(root/'seed_comparison.json',comparison);core.csv_write(root/'seed_comparison.csv',comparison)
        text+='## Seed43 minus seed42, paired by question\n\n'+core.table(comparison,tuple(comparison[0]))+'\n\n'
    text+='## Metrics and limitations\n\nPhysical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), with whole/global/local counts, excluding dense prefill. Retained mass and full-dimensional local output error use dense attention on each run\'s corresponding QKV. Shared-QKV diagnostics are reused from32 selected early seed42 states per method; they are not new seed43 trajectory diagnostics. Execution-local measurements are fresh. Token agreement uses the seed43 dense sequence, all positions including after divergence and EOS, with missing/extra positions disagreeing.\n\n'
    text+='Six historically calibrated questions remain in headline30; noncalibration24 is separately reported. All examples were previously examined. Two generation seeds with one fixed projection seed are a sensitivity check, not broad seed robustness or fresh held-out confirmation. Paired prompt-bootstrap95% intervals are exploratory and not multiplicity-corrected. Compare actual sparsity, not target labels. No policy or seed selection using these scores.\n\n'
    text+='Numerical kernels and routing algorithms are unchanged. Existing CPU/CUDA evidence is reused, plus fresh seed43 native/unpruned and trusted/pruned rank8 checks on two inputs before evaluation. QK, block softmax and projected PV are still computed; retained native PV is a masked dense-shaped matmul. Projection/cache overhead is recorded; no FlashAttention or hardware-speedup claim.\n\n'
    text+=f"![Tradeoffs](figures/aime26_tradeoffs.png)\n\nMissing={len(audit['missing'])}; violations={len(audit['violations'])}. Raw-only reproduction: `python -m {MODULE} report`, then `verify`.\n"
    (root/'report.md').write_text(text)


def regenerate(root=ROOT):
    with patch.object(reporting,'study',sys.modules[__name__]),patch.object(reporting,'write_report',write_report):
        audit=reporting.regenerate(root)
    if audit['complete']:
        for name in ('seed_comparison.json','seed_comparison.csv'):audit['artifacts'][name]=sha((root/name).read_bytes())
    _write(root/'audit.json',audit);return audit


def verify(root=ROOT):
    before=read(root/'audit.json')
    if not before['complete'] or before['completed']!=EXPECTED:raise ValueError('Require all150 fresh results')
    fb.check_sources({str(root/p):h for p,h in before['artifacts'].items()})
    if regenerate(root)!=before:raise ValueError('Raw-only report regeneration changed')
    result=dict(passed=True,completed=EXPECTED,inference_performed=False,audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json',result);return result


def parent_ready():
    folder=parent.ROOT
    if not (folder/'supervisor_terminal.json').exists():return None
    status=read(folder/'supervisor_terminal.json')
    if status['exit_code']!=0:raise ValueError('Queued predecessor failed; preserve it and investigate before proceeding')
    terminal=read(folder/'terminal.json')
    if not terminal['complete'] or terminal['completed']!=parent.EXPECTED:raise ValueError('Queued predecessor incomplete')
    return parent.remaining.require_complete(folder,parent.EXPECTED)


def work(root):
    setup,contract=prepare(root),execution(root)
    proof=parent_ready()
    if proof is None:raise ValueError('Cannot start before predecessor final audit/report completes')
    frozen_write(root/'queue_parent_audit.json',dict(passed=True,sources=proof))
    fb.phase(root,'import_frozen_policies_and_diagnostics');reuse(root,setup,contract)
    torch.backends.cuda.matmul.allow_tf32=False;fb.phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',setup['model'],device='cuda',precision='bfloat16',revision=setup['revision']).load()
    with dimension_scope():
        fb.phase(root,'seed43_actual_model_smoke')
        with patch.object(validation,'PROJECTED',{'jl_gaussian_r8':ALL['jl_gaussian_r8']}):validation.smoke(adapter,root,setup,contract)
        smoke_audit(root,setup,contract);conditions=freeze_conditions(root,setup,contract)
        for label in CONDITIONS:
            c=conditions[label];stage='dense' if label=='dense' else 'final';fb.phase(root,stage,condition=label,expected=30)
            for row in sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id'])):
                try:runner.cached(adapter,root,row,stage,label,c['name'],c['config'],c['thresholds']['aime26'],contract)
                except Exception:fb.failure(root,stage,condition=label,id=row['id'])
            accounting(root)
    del adapter;torch.cuda.empty_cache();fb.phase(root,'report');audit=regenerate(root)
    if audit['complete']:fb.phase(root,'independent_report_verification');verify(root)
    _write(root/'terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=EXPECTED,finished=time.time()))
    fb.phase(root,'finished',complete=audit['complete'])


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _write(root/'queue.json',dict(status='waiting',supervisor_pid=os.getpid(),after=str(parent.ROOT),started=time.time()))
        while parent_ready() is None:time.sleep(60)
        execution(root);fb.gpu_idle()
        _write(root/'queue.json',dict(status='released',supervisor_pid=os.getpid(),after=str(parent.ROOT),released=time.time()))
        with (root/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:code=child.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    state={k:read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase','progress')}
                    _append(root/'monitor.jsonl',dict(time=time.time(),pid=child.pid,**state))
            _write(root/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    execution(root)
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,queued_after=str(parent.ROOT),generation_seed=SEED,projection_seed=1729,new_final_slots=EXPECTED)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','launch','supervise','work','report','verify'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    result=(regenerate if args.command=='report' else globals()[args.command])(args.output)
    if args.command in ('prepare','report','verify'):print(json.dumps(dict(command=args.command,complete=result.get('complete'),passed=result.get('passed'))))
