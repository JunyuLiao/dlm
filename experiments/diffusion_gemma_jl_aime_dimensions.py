"""AIME26 Gaussian centered-routing rank8/16/24 follow-on at50% sparsity.

The original numerical sources stay frozen. Rank24 is enabled by a scoped
configuration allow-list extension; existing generic padded kernels are reused.
"""
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

import numpy as np
import torch
from dllm.models import create_adapter
from experiments import diffusion_gemma_jl_directional50 as prior
from experiments import diffusion_gemma_jl_remaining50 as transfer
from experiments.diffusion_gemma_jl_output_aware import config as configuration
from experiments.diffusion_gemma_jl_output_aware import runner, screen, shared_analysis, validation
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint

ROOT = Path('results/diffusion_gemma_jl_aime_gaussian_dimensions_v8')
PREVIOUS = prior.ROOT
MODULE = 'experiments.diffusion_gemma_jl_aime_dimensions'
NEW = {f'jl_gaussian_r{r}': dict(family='gaussian', rank=r) for r in (8,16,24)}
REFERENCE = dict(jl_gaussian_r32=dict(family='gaussian', rank=32), full_centered=dict(family='identity'))
ALL = {**REFERENCE, **NEW}
CONFIGS = {**prior.parent.BASELINES, **ALL}
REUSED = ['dense'] + [n+'_s50' for n in {**prior.parent.BASELINES, **REFERENCE}]
CONDITIONS = REUSED + [n+'_s50' for n in NEW]
EXPECTED, TEST_COUNT = 270, 11
read, sha, frozen_write = prior.read, prior.sha, prior.frozen_write
fb, backend, continuation = prior.fb, prior.backend, prior.continuation
BASE_PLOTS = backend.plots


@contextmanager
def dimension_scope():
    # Config resolves DIMENSIONS in its defining module at construction time.
    # No frozen source file, projection rule or numerical kernel is modified.
    with patch.object(configuration, 'DIMENSIONS', (8,16,24,32)), patch.object(runner, 'PROJECTED', ALL):
        yield


def prepare(root=ROOT):
    old = read(PREVIOUS/'setup.json')
    setup = deepcopy(old)
    setup.update(schema='jl_aime_gaussian_dimensions_v8',
        final=[r for r in old['final'] if r['benchmark']=='aime26'],
        calibration=[r for r in old['calibration'] if r['benchmark']=='aime26'],
        # Keep the existing long-context development input for16-token smoke
        # validation only; no LongBench final evaluation or calibration occurs.
        conditions=CONDITIONS, configs=CONFIGS, targets=[.5],
        calibration_ids={'aime26':old['calibration_ids']['aime26']},
        calibration_overlap={'aime26':old['calibration_overlap']['aime26']},
        projection_policy='Gaussian centered rank8/16/24, seed1729, unchanged rank-dependent matrix seed/hash construction. Scoped rank24 allow-list extension only. Cached Gaussian32/full-dimensional controls unchanged.',
        calibration_policy='Same six AIME IDs, full2048-token budgets. Existing independent local/global empirical-rank proposal and scalar fallback, max12 joint points; require overall/global/local48-52%. Common initial scalar trial is the prior full-budget Gaussian32 policy, never treated as calibrated for a different rank. No accuracy-based selection.',
        initial_policy=str(PREVIOUS/'policies/aime26/jl_gaussian_r32_s50.json'),
        proposal_policy='Reuse historical rank8/16 calibration CDFs; compute rank24 CDF on the same240 saved AIME calibration QKV states. No new dense replay. Only full-budget sparse-trajectory verification can accept a policy.',
        failure_policy='Preserve failures, continue independent ranks; no final run for a rank lacking an accepted policy. Never retune from final scores.',
        final_generation_slots=90, reused_reference_and_baseline_slots=180,
        calibration_validation_accounting='90 final slots; calibration and short validation calls separately counted. Six selected calibration outputs/rank are exact final aliases, not repeated inference.',
        headline_accuracy='All30 AIME questions including six calibration members; noncalibration24 additionally reported. Previously examined data, not fresh held-out confirmation.',
        monitoring_interval_seconds=900)
    if len(setup['final']) != 30 or len({r['id'] for r in setup['final']}) != 30:
        raise ValueError('Require the existing30 AIME questions')
    prior.calibration_rows(setup, 'aime26')
    if any(r['generation_budget'] != 2048 or r['seed'] != 42 or sha(r['prompt']) != r['prompt_hash'] for r in setup['final']):
        raise ValueError('AIME budget, seed or prompt changed')
    frozen_write(root/'setup.json', setup)
    for split in ('final','calibration','development'):
        frozen_write(root/f'{split}_manifest.json', setup[split])
    frozen_write(root/'dataset_audit.json', dict(passed=True, exact_previous30=True, generation_budget=2048,
        calibration_ids=setup['calibration_ids']['aime26'], calibration_included_in_headline=True,
        expected=EXPECTED, new_final_slots=90, cached_slots=180))
    return setup


def execution(root=ROOT):
    import triton, transformers
    setup = prepare(root); old = read(PREVIOUS/'execution_contract.json')
    runtime = dict(torch=torch.__version__, triton=triton.__version__, transformers=transformers.__version__)
    if runtime != old['runtime']:
        raise ValueError('Runtime differs from compatible cached results')
    tests = root/'tests.xml'; suites = ET.parse(tests).getroot().findall('testsuite')
    counts = {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=TEST_COUNT, failures=0, errors=0, skipped=0):
        raise ValueError(f'CPU/CUDA dimension tests incomplete: {counts}')
    files = (Path(__file__), Path(transfer.__file__), Path('tests/test_jl_aime_dimensions.py'), tests,
             root/'setup.json', PREVIOUS/'execution_contract.json')
    sources = {**old['sources'], **{str(p):sha(p.read_bytes()) for p in files}}
    fb.check_sources(sources)
    data = dict(schema='jl_aime_dimensions_execution_v8', sources=sources, runtime=runtime,
        parent_fingerprint=old['fingerprint'], setup_sha256=sha((root/'setup.json').read_bytes()),
        numerical_kernels_unchanged=True, scoped_allowed_dimensions=[8,16,24,32],
        new_dimensions=[8,16,24], projection_seed=1729, projection_dtype='float32',
        matmul_allow_tf32=False, target_sparsity=.5, tolerance=.02, max_verified_points=12,
        calibration_budget=2048, final_generation_slots=90, hardware_speedup_claim=False)
    data['fingerprint'] = _fingerprint(data); frozen_write(root/'execution_contract.json', data)
    return data


def audit_policy(root, p, setup, contract):
    if p['fingerprint'] != contract['fingerprint'] or p['heldout_used'] or p['target'] != .5 or p['benchmark'] != 'aime26':
        raise ValueError('Invalid policy identity or calibration scope')
    if p.get('imported_policy'):
        fb.check_sources(p['sources']); old = read(p['imported_policy'])
        if p['imported_policy'] != str(PREVIOUS/'policies/aime26'/f"{p['name']}_s50.json"):
            raise ValueError('Unexpected prior threshold source')
        if any(p[k] != old[k] for k in ('name','benchmark','target','config','policy','measured')):
            raise ValueError('Cached policy/operator changed')
        return
    if p['name'] not in NEW or p['config'] != NEW[p['name']]:
        raise ValueError('Unrequested new rank')
    with patch.object(fb, 'CONFIG', NEW[p['name']]), patch.object(fb, 'calibration_rows', prior.calibration_rows):
        fb.audit_policy(root, p, setup, contract)


def reuse(root, setup, contract):
    proof = transfer.require_complete(PREVIOUS, 640)
    old_setup, old_contract = read(PREVIOUS/'setup.json'), read(PREVIOUS/'execution_contract.json')
    validation_sources = prior.smoke_audit(PREVIOUS, old_setup, old_contract)
    frozen_write(root/'inherited_validation.json', dict(passed=True, sources=validation_sources,
        note='Raw-audited cached baseline/rank32/full-reference validation; all three newly requested ranks receive fresh CUDA smoke.'))
    for label in REUSED:
        cpath = PREVIOUS/'final_configs'/f'{label}.json'; c = read(cpath); fb.check_sources(c['sources'])
        if label != 'dense':
            src = PREVIOUS/'policies/aime26'/f'{label}.json'; old = read(src)
            prior.audit_policy(PREVIOUS, old, old_setup, old_contract)
            p = {k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
            p.update(fingerprint=contract['fingerprint'], heldout_used=False, imported_policy=str(src),
                sources={str(src):sha(src.read_bytes()), **proof}, calibration_ids=old.get('calibration_ids'),
                cap_one_unattainable=old.get('cap_one_unattainable'))
            audit_policy(root,p,setup,contract); frozen_write(root/'policies/aime26'/f'{label}.json',p)
        for row in setup['final']:
            stage = 'dense' if label == 'dense' else 'final'
            fb.alias(root,row,stage,label,c['name'],c['config'],c['thresholds']['aime26'],contract,
                shard_path(PREVIOUS,stage,label,row['id']),old_contract['fingerprint'])
        print('reused AIME condition',label,30,flush=True)
    frozen_write(root/'predecessor_audit.json',dict(passed=True,reused_outputs=180,sources=proof))
    states = [s for s in read(PREVIOUS/'shared_state_index.json') if s['id'].startswith('aime26/')]
    if len(states)!=240 or any(s['split']!='calibration' or s['id'] not in setup['calibration_ids']['aime26'] for s in states):
        raise ValueError('Invalid AIME calibration QKV coverage')
    fb.check_sources({s['path']:s['sha256'] for s in states}); frozen_write(root/'shared_state_index.json',states)
    for item in read(PREVIOUS/'shared_diagnostics_index.json'):
        src=Path(item['path']); data=read(src); ident=data['identity']
        if not ident['source']['id'].startswith('aime26/') or ident['name'] not in {**prior.parent.BASELINES,**REFERENCE} or ident['target']!=.5:
            continue
        fb.check_sources({str(src):item['sha256']})
        policy=root/'policies/aime26'/f"{ident['name']}_s50.json"
        if read(ident['policy_path'])['policy'] != read(policy)['policy']:
            raise ValueError('Shared diagnostic policy mismatch')
        ident.update(fingerprint=contract['fingerprint'],policy_path=str(policy),policy_sha256=sha(policy.read_bytes()))
        frozen_write(root/'shared_diagnostics'/src.name,data)
    # Native dense16 validation is independent of sketch rank. Reuse its exact
    # unchanged inputs/decoding, while fresh rank-specific sparse checks follow.
    examples=[setup['calibration'][0],max(setup['development'],key=lambda r:len(r['prompt_tokens']))]
    imports={}
    for row in examples:
        src=PREVIOUS/'validation'/old_contract['fingerprint']/(sha(row['id'])+'.native.json')
        if str(src) not in validation_sources:
            raise ValueError('Native smoke cache was not raw-audited')
        dest=root/'validation'/contract['fingerprint']/src.name
        frozen_write(dest,read(src)); imports[str(dest)]=dict(source=str(src),sha256=sha(src.read_bytes()))
    frozen_write(root/'native_smoke_imports.json',imports)


def smoke_audit(root,setup,contract):
    with patch.object(backend,'PROJECTED',NEW):
        sources=prior.BASE_SMOKE(root,setup,contract)
    inherited=read(root/'inherited_validation.json'); fb.check_sources(inherited['sources'])
    sources.update(inherited['sources'])
    return sources


def proposals(root,contract):
    folder=root/'rank24_proposals'
    with dimension_scope(),patch.object(screen,'PROJECTED',{'jl_gaussian_r24':NEW['jl_gaussian_r24']}), \
         patch.object(screen,'BASELINES',{}),patch.object(screen,'SENSITIVITY_SEEDS',()):
        return screen.shared_screen(folder,read(root/'shared_state_index.json'),contract)


def warm_start(root,name,benchmark,target):
    if name not in NEW or benchmark!='aime26' or target!=.5:
        raise ValueError('Unexpected calibration request')
    path=PREVIOUS/'policies/aime26/jl_gaussian_r32_s50.json'; old=read(path)
    proposal_root=root/'rank24_proposals' if name=='jl_gaussian_r24' else prior.parent.OLD
    values,sources=screen.distributions(proposal_root,name,benchmark)
    sources[str(path)]=sha(path.read_bytes())
    return old,{k:np.maximum(v,-1e30) for k,v in values.items()},sources


def calibrate(adapter,root,setup,contract,name):
    with dimension_scope(),patch.object(continuation,'NEW',NEW),patch.object(continuation,'audit_policy',audit_policy), \
         patch.object(continuation,'warm_start',lambda n,b,t:warm_start(root,n,b,t)), \
         patch.object(fb,'calibration_rows',prior.calibration_rows):
        return continuation.calibrate(adapter,root,setup,contract,name,'aime26',.5)


def freeze_conditions(root,setup,contract):
    sources={**contract['sources'],**smoke_audit(root,setup,contract),
             **read(root/'predecessor_audit.json')['sources']}
    result={}
    for label in CONDITIONS:
        name='dense' if label=='dense' else label.rsplit('_s',1)[0]
        own=dict(sources); threshold=None
        if name!='dense':
            path=root/'policies/aime26'/f'{label}.json'
            if not path.exists():
                continue
            p=read(path); audit_policy(root,p,setup,contract); threshold=p['policy'];own[str(path)]=sha(path.read_bytes())
        c=dict(fingerprint=contract['fingerprint'],name=name,config={} if name=='dense' else CONFIGS[name],
            target=0. if name=='dense' else .5,thresholds={'aime26':threshold},sources=own,expected_per_benchmark={'aime26':30})
        frozen_write(root/'final_configs'/f'{label}.json',c);result[label]=c
    return result


def accounting(root):
    counts={}
    for stage in ('calibration','final','dense'):
        outputs=[read(p) for p in (root/stage).glob('**/shards/*.json')]
        counts[stage]=dict(stored=len(outputs),new=sum(not o.get('imported_source') for o in outputs),
                           reused=sum(bool(o.get('imported_source')) for o in outputs))
    smoke=sum(1 for p in (root/'validation').glob('**/shards/*.json') if not read(p).get('imported_source'))
    data=dict(final_generation_slots=90,reused_final_baselines_and_references=180,completed_by_stage=counts,
        new_short_validation_calls=smoke,reused_native_smoke_calls=len(read(root/'native_smoke_imports.json')) if (root/'native_smoke_imports.json').exists() else 0,
        completed_new_inference=sum(v['new'] for v in counts.values())+smoke,
        note='Calibration/validation calls additional to90 final slots. Selected calibration outputs alias final6 per rank; shared-state proposal/diagnostic GPU work is not a generation call.')
    _write(root/'inference_accounting.json',data);return data


def write_report(root,setup,rows,tasks,compared,diag,audit,policies):
    audit.update(expected=EXPECTED,complete=audit['completed']==EXPECTED and not audit['missing'] and not audit['violations'])
    display=[dict(split=r['split'],condition=r['condition'],threshold=backend.threshold_text(r),
        score=f"{r['correct']:g}/{r['count']}",accuracy=100*r['accuracy'],delta_pp=100*r['delta'],
        ci95_pp=[100*v for v in r['paired_ci95']],whole=100*r['overall_physical_sparsity'],
        global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_mass'],agreement=100*r['token_agreement'],local_operator_error=r['overall_relative_error'],
        length_limited=r['length_terminated']) for r in rows]
    backend.csv_write(root/'main_results.csv',display)
    text='# AIME26 Gaussian projection dimensions8/16/24 versus32 and full-dimensional\n\n'
    text+=f"Status: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/{EXPECTED} audited outputs.90 new final slots plus180 cached results; calibration and validation calls separately counted.\n\n"
    text+='## Setup\n\nAll existing30 AIME26 questions,2048-token output budget, identical cached prompts and scorer, seed42, BF16 DiffusionGemma revision '+setup['revision']+'. Canvas256,max48 denoising steps,thinkingFalse; unchanged native0.4–0.8 schedule (temperature0 sentinel is not greedy).128-query×64-KV physical tiles; prefix+canvas eligible with native masks and GQA. All30, including the six calibration members, count in headline accuracy; noncalibration24 is additional. Previously examined samples, not fresh held-out confirmation. No LongBench final evaluation is run.\n\n'
    text+='## Algorithms and calibration\n\nAll four Gaussian ranks use the same attention-weighted projected centered online-update criterion, fixed seed1729 and unchanged rank-dependent matrix generation/hash rules. The full-dimensional identity control is unchanged. No projection directions are selected using scores. Rank24 is enabled only through a process-local configuration allow-list; the existing padded FP32 GPU kernels, structural masks, first-support/tie retention, skipped-state convention, valid-KV RMS reference and original-V renormalized output are unchanged. Fresh CUDA tensor and actual-model trusted-reference checks validate ranks8/16/24. Two compatible native-dense16 smoke outputs are reused; sparse smoke is fresh.\n\n'
    text+='Each new rank starts from the same prior full-budget Gaussian32 scalar thresholds as an unverified initial trial. The unchanged empirical-rank proposal/scalar fallback procedure verifies the same six AIME IDs at the full2048-token budget, for at most12 joint points. Acceptance requires physical overall/global/local sparsity within48–52%. Rank8/16 reuse historical calibration-state CDFs; rank24 computes its CDF from the same240 saved AIME states. Only sparsity, never accuracy, selects thresholds. Final policies freeze before final inference. Cached rank32/full/dense/originalBLASST/aggressiveBLASST/mass policies and outputs remain unchanged. No final retuning. Calibration IDs: '+', '.join(setup['calibration_ids']['aime26'])+'.\n\n'
    text+='BLASST retains its exact-QK, row-max threshold convention and existing local/global inverse-length or scalar policies; original caps lambda at1 and aggressive permits larger values. Its physical tile skips only when all valid rows skip. Mass-only is the existing max-based candidate mass bound, not mass_exact. Compare actual physical sparsities and local/global allocations, not target labels.\n\n'
    text+='## Results\n\n'+backend.findings(rows,compared)+'\n\n'
    cols=('condition','threshold','score','accuracy','delta_pp','ci95_pp','whole','global_s','local_s','mass','agreement','local_operator_error','length_limited')
    for split in ('full','noncalibration24','calibration6'):
        text+=f'### {split}\n\n'+backend.table([r for r in display if r['split']==split],cols)+'\n\n'
    text+='## Measurements and limitations\n\nPhysical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), reported overall/global/local and by layer/head/step/region. Dense prefill is excluded. Retained mass and full-dimensional local operator error use dense attention on each execution\'s corresponding QKV, separately from generated-token divergence. Shared-QKV diagnostics use identical historical early calibration states (32 selected states from240); they do not exhaust all final generation states. No end-to-end projection-seed sensitivity claim is made.\n\n'
    text+='Token agreement counts positional matches over all generated positions after divergence; missing/extra positions disagree and EOS is included. Paired prompt bootstrap95% intervals condition on fixed policies/seeds, are exploratory and not multiplicity-corrected. Small differences on30 reused questions may be inconclusive. Changing rank also changes the deterministic random matrix; this is one fixed-seed dimension study, not an average over directions.\n\n'
    text+='QK, block softmax and projected PV remain computed; full-dimensional PV is diagnostic-only for projected routing. Retained native PV is still a dense-shaped masked matmul. Work-accounting exports sketch projection/storage/refresh/reuse; no hardware-speedup or FlashAttention claim.\n\n'
    text+='![AIME tradeoffs](figures/aime26_tradeoffs.png)\n\n![Target versus achieved](figures/aime26_target_actual.png)\n\n'
    text+=f"Missing={len(audit['missing'])}; violations={len(audit['violations'])}. Failures are preserved and independent ranks continue. Regenerate without inference: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m {MODULE} report`; independently repeat with `verify`.\n"
    (root/'report.md').write_text(text)


def regenerate(root=ROOT):
    def plots(folder,rows):
        from matplotlib.figure import Figure
        original=Figure.suptitle
        def title(fig,text,*args,**kwargs):
            return original(fig,text.replace('50/75% targets','50% target; Gaussian rank sweep'),*args,**kwargs)
        with patch.object(Figure,'suptitle',title):
            return [p for p in BASE_PLOTS(folder,rows) if 'aime26' in p]
    with dimension_scope(),patch.object(backend,'prepare',prepare),patch.object(backend,'execution',execution), \
         patch.object(backend,'PROJECTED',ALL),patch.object(backend,'TARGETS',(.5,)),patch.object(backend.common,'TARGETS',(.5,)), \
         patch.object(backend,'audit_policy',audit_policy),patch.object(backend,'smoke_audit',smoke_audit), \
         patch.object(backend.common,'shared_seed_summary',lambda unused:([],{})), \
         patch.object(backend,'write_report',write_report),patch.object(backend,'plots',plots):
        audit=backend.regenerate(root)
    accounting(root);audit['artifacts']['inference_accounting.json']=sha((root/'inference_accounting.json').read_bytes())
    _write(root/'audit.json',audit);return audit


def verify(root=ROOT):
    before=read(root/'audit.json')
    if not before['complete'] or before['completed']!=EXPECTED:
        raise ValueError('Require all270 audited results')
    fb.check_sources({str(root/p):h for p,h in before['artifacts'].items()})
    if regenerate(root)!=before:
        raise ValueError('Raw-only independent regeneration changed')
    result=dict(passed=True,completed=EXPECTED,inference_performed=False,audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json',result);return result


def work(root):
    setup,contract=prepare(root),execution(root)
    fb.phase(root,'reuse_cached_comparisons');reuse(root,setup,contract);accounting(root)
    torch.backends.cuda.matmul.allow_tf32=False;fb.phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',setup['model'],device='cuda',precision='bfloat16',revision=setup['revision']).load()
    with dimension_scope():
        fb.phase(root,'three_rank_actual_model_smoke')
        with patch.object(validation,'PROJECTED',NEW):
            validation.smoke(adapter,root,setup,contract)
        smoke_audit(root,setup,contract);accounting(root)
        fb.phase(root,'rank24_calibration_state_proposals');proposals(root,contract)
        for name in NEW:
            try:
                calibrate(adapter,root,setup,contract,name)
            except Exception:
                fb.failure(root,'calibration',method=name,benchmark='aime26',target=.5)
            accounting(root)
        conditions=freeze_conditions(root,setup,contract)
        fb.phase(root,'shared_state_diagnostics')
        try:
            with patch.object(shared_analysis,'PROJECTED',ALL),patch.object(shared_analysis,'BASELINES',prior.parent.BASELINES),patch.object(shared_analysis,'TARGETS',(.5,)):
                shared_analysis.analyze(root,read(root/'shared_state_index.json'),contract)
        except Exception:
            fb.failure(root,'shared_state_diagnostics')
        for name,cfg in NEW.items():
            label=name+'_s50'
            if label not in conditions:
                continue
            c=conditions[label];p=read(root/'policies/aime26'/f'{label}.json')
            selected=next(x for x in p['trace'] if x['iteration']==p['selected_round'])
            fb.phase(root,'final',condition=label,expected=30)
            for row in sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id'])):
                try:
                    if row['id'] in setup['calibration_ids']['aime26']:
                        fb.alias(root,row,'final',label,name,cfg,p['policy'],contract,
                            Path(selected['sources'][row['id']]['path']),contract['fingerprint'])
                    else:
                        runner.cached(adapter,root,row,'final',label,name,cfg,p['policy'],contract)
                except Exception:
                    fb.failure(root,'final',condition=label,id=row['id'])
            accounting(root)
    del adapter;torch.cuda.empty_cache();fb.phase(root,'report');audit=regenerate(root)
    if audit['complete']:
        fb.phase(root,'independent_report_verification');verify(root)
    _write(root/'terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=EXPECTED,finished=time.time()))
    fb.phase(root,'finished',complete=audit['complete'])


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fb.gpu_idle()
        with (root/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'work','--output',str(root)],stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:
                    code=child.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    state={k:read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase','progress')}
                    _append(root/'monitor.jsonl',dict(time=time.time(),pid=child.pid,**state))
            _write(root/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:raise SystemExit(code)


def launch(root):
    transfer.require_complete(PREVIOUS,640);execution(root);fb.gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,final_slots=90,reused_slots=180,dimensions=[8,16,24],target=.5,calibration_validation_additional=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','launch','supervise','work','report','verify'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    result=(regenerate if args.command=='report' else globals()[args.command])(args.output)
    if args.command in ('prepare','report','verify'):
        print(json.dumps(dict(command=args.command,complete=result.get('complete'),passed=result.get('passed'))))
