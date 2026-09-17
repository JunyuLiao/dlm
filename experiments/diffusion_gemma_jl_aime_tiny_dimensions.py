"""Add Gaussian centered ranks2/4 to the frozen AIME rank sweep.

Reuse the existing kernels, calibration, validation and reporting machinery;
never edit a predecessor's numerical sources, policies or result artifacts.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
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
from experiments import diffusion_gemma_jl_aime_dimensions as previous
from experiments import diffusion_gemma_jl_aime_dimensions_analysis as paired
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _fingerprint
from experiments.diffusion_gemma_value_aware.run import shard_path

ROOT = Path('results/diffusion_gemma_jl_aime_gaussian_tiny_dimensions_v9')
PREVIOUS = previous.ROOT
MODULE = 'experiments.diffusion_gemma_jl_aime_tiny_dimensions'
NEW = {f'jl_gaussian_r{r}': dict(family='gaussian', rank=r) for r in (2,4)}
ALL = {**previous.ALL, **NEW}
CONFIGS = {**previous.CONFIGS, **NEW}
REUSED = previous.CONDITIONS
CONDITIONS = REUSED + [n+'_s50' for n in NEW]
EXPECTED, TEST_COUNT = 330, 11
read, sha, frozen_write = previous.read, previous.sha, previous.frozen_write
fb, backend, continuation = previous.fb, previous.backend, previous.continuation
PREVIOUS_AUDIT = previous.audit_policy
runner, screen, shared_analysis, validation = previous.runner, previous.screen, previous.shared_analysis, previous.validation
PAIRS = tuple((n+'_s50', ref+'_s50') for n in NEW
              for ref in ('jl_gaussian_r8','jl_gaussian_r16','jl_gaussian_r24','jl_gaussian_r32','full_centered'))


@contextmanager
def dimension_scope():
    with patch.object(previous.configuration,'DIMENSIONS',(2,4,8,16,24,32)), patch.object(runner,'PROJECTED',ALL):
        yield


def prepare(root=ROOT):
    setup=deepcopy(read(PREVIOUS/'setup.json'))
    setup.update(schema='jl_aime_gaussian_tiny_dimensions_v9', conditions=CONDITIONS, configs=CONFIGS,
        projection_policy='Gaussian centered ranks2/4, seed1729, unchanged rank-dependent matrix construction and FP32 generic kernels. Scoped dimension allow-list only; cached ranks8/16/24/32/full and baselines unchanged.',
        initial_policy=str(PREVIOUS/'policies/aime26/jl_gaussian_r32_s50.json'),
        proposal_policy='Compute rank2/4 empirical CDFs on the same240 saved AIME calibration QKV states; no dense replay. Only full2048-token verification accepts a policy.',
        final_generation_slots=60,reused_reference_and_baseline_slots=270,
        calibration_validation_accounting='60 new final slots plus270 cached slots. Calibration and12 fresh short validation calls additional; selected calibration6/rank alias final slots.')
    if setup['final']!=read(PREVIOUS/'final_manifest.json') or len(setup['final'])!=30:
        raise ValueError('Require unchanged30 AIME rows')
    previous.prior.calibration_rows(setup,'aime26')
    if any(r['generation_budget']!=2048 or r['seed']!=42 or sha(r['prompt'])!=r['prompt_hash'] for r in setup['final']):
        raise ValueError('Generation settings changed')
    frozen_write(root/'setup.json',setup)
    for split in ('final','calibration','development'):
        frozen_write(root/f'{split}_manifest.json',setup[split])
    frozen_write(root/'dataset_audit.json',dict(passed=True,exact_previous30=True,generation_budget=2048,
        calibration_ids=setup['calibration_ids']['aime26'],calibration_included_in_headline=True,
        expected=EXPECTED,new_final_slots=60,cached_slots=270))
    return setup


def execution(root=ROOT):
    import triton,transformers
    prepare(root);old=read(PREVIOUS/'execution_contract.json')
    runtime=dict(torch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__)
    if runtime!=old['runtime']:raise ValueError('Cached runtime mismatch')
    tests=root/'tests.xml';suites=ET.parse(tests).getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=TEST_COUNT,failures=0,errors=0,skipped=0):raise ValueError(f'Incomplete tests: {counts}')
    files=(Path(__file__),Path(paired.__file__),Path('tests/test_jl_aime_tiny_dimensions.py'),tests,
           root/'setup.json',PREVIOUS/'execution_contract.json')
    sources={**old['sources'],**{str(p):sha(p.read_bytes()) for p in files}};fb.check_sources(sources)
    data=dict(schema='jl_aime_tiny_dimensions_execution_v9',sources=sources,runtime=runtime,
        parent_fingerprint=old['fingerprint'],setup_sha256=sha((root/'setup.json').read_bytes()),
        numerical_kernels_unchanged=True,scoped_allowed_dimensions=[2,4,8,16,24,32],new_dimensions=[2,4],
        projection_seed=1729,projection_dtype='float32',matmul_allow_tf32=False,target_sparsity=.5,
        tolerance=.02,max_verified_points=12,calibration_budget=2048,final_generation_slots=60,hardware_speedup_claim=False)
    data['fingerprint']=_fingerprint(data);frozen_write(root/'execution_contract.json',data);return data


def audit_policy(root,p,setup,contract):
    # The previous generic auditor already checks exact imported policy identity
    # or all raw full-budget calibration shards for a newly calibrated rank.
    with patch.object(previous,'PREVIOUS',PREVIOUS),patch.object(previous,'NEW',NEW):
        return PREVIOUS_AUDIT(root,p,setup,contract)


def reuse(root,setup,contract):
    proof=previous.transfer.require_complete(PREVIOUS,270)
    old_setup,old_contract=read(PREVIOUS/'setup.json'),read(PREVIOUS/'execution_contract.json')
    sources=previous.smoke_audit(PREVIOUS,old_setup,old_contract)
    frozen_write(root/'inherited_validation.json',dict(passed=True,sources=sources))
    for label in REUSED:
        c=read(PREVIOUS/'final_configs'/f'{label}.json');fb.check_sources(c['sources'])
        if label!='dense':
            src=PREVIOUS/'policies/aime26'/f'{label}.json';old=read(src)
            previous.audit_policy(PREVIOUS,old,old_setup,old_contract)
            p={k:deepcopy(old[k]) for k in ('name','benchmark','target','config','policy','measured')}
            p.update(fingerprint=contract['fingerprint'],heldout_used=False,imported_policy=str(src),
                sources={str(src):sha(src.read_bytes()),**proof},calibration_ids=old.get('calibration_ids'),
                cap_one_unattainable=old.get('cap_one_unattainable'))
            audit_policy(root,p,setup,contract);frozen_write(root/'policies/aime26'/f'{label}.json',p)
        for row in setup['final']:
            stage='dense' if label=='dense' else 'final'
            fb.alias(root,row,stage,label,c['name'],c['config'],c['thresholds']['aime26'],contract,
                shard_path(PREVIOUS,stage,label,row['id']),old_contract['fingerprint'])
        print('reused AIME condition',label,30,flush=True)
    frozen_write(root/'predecessor_audit.json',dict(passed=True,reused_outputs=270,sources=proof))
    states=read(PREVIOUS/'shared_state_index.json')
    if len(states)!=240 or any(s['split']!='calibration' or s['id'] not in setup['calibration_ids']['aime26'] for s in states):
        raise ValueError('Invalid calibration states')
    fb.check_sources({s['path']:s['sha256'] for s in states});frozen_write(root/'shared_state_index.json',states)
    for item in read(PREVIOUS/'shared_diagnostics_index.json'):
        src=Path(item['path']);fb.check_sources({str(src):item['sha256']});data=read(src);ident=data['identity']
        if ident['target']!=.5 or ident['name'] not in previous.CONFIGS:raise ValueError('Unexpected diagnostic source')
        policy=root/'policies/aime26'/f"{ident['name']}_s50.json"
        if read(ident['policy_path'])['policy']!=read(policy)['policy']:raise ValueError('Diagnostic policy mismatch')
        ident.update(fingerprint=contract['fingerprint'],policy_path=str(policy),policy_sha256=sha(policy.read_bytes()))
        frozen_write(root/'shared_diagnostics'/src.name,data)
    imports={}
    for row in (setup['calibration'][0],max(setup['development'],key=lambda r:len(r['prompt_tokens']))):
        src=PREVIOUS/'validation'/old_contract['fingerprint']/(sha(row['id'])+'.native.json')
        if str(src) not in sources:raise ValueError('Native smoke not raw-audited')
        dest=root/'validation'/contract['fingerprint']/src.name;frozen_write(dest,read(src))
        imports[str(dest)]=dict(source=str(src),sha256=sha(src.read_bytes()))
    frozen_write(root/'native_smoke_imports.json',imports)


def smoke_audit(root,setup,contract):
    # Same raw checks as the established auditor, generalized from12 to8 cases.
    folder=root/'validation'/contract['fingerprint'];path=folder/'smoke.json';proof=read(path)
    if not proof['passed'] or proof['fingerprint']!=contract['fingerprint'] or len(proof['cases'])!=8:
        raise ValueError('Incomplete two-rank actual-model smoke')
    sources={str(path):sha(path.read_bytes())}
    for original in (setup['calibration'][0],max(setup['development'],key=lambda r:len(r['prompt_tokens']))):
        row=dict(original,generation_budget=16);p=folder/(sha(row['id'])+'.native.json');native=read(p);sources[str(p)]=sha(p.read_bytes())
        for name,cfg in NEW.items():
            for unpruned in (True,False):
                thresholds={k:dict(log_threshold=-1000.,unpruned=True,tau=0.) if unpruned else dict(log_threshold=-2.) for k in ('local','global')}
                label=f'{name}/{"unpruned" if unpruned else "pruned"}'
                def load(stage):
                    out=runner.cached(None,folder,row,stage,label,name,cfg,thresholds,contract)
                    p=shard_path(folder,stage,label,row['id']);sources[str(p)]=sha(p.read_bytes())
                    sources[out['records_source']['path']]=out['records_source']['sha256'];return out
                out=load('smoke')
                if not out['kernel_validation'] or not out['finite_calls']:raise ValueError('Missing finite CUDA checks')
                if unpruned:
                    if out['completion_tokens']!=native['completion_tokens'] or any(r['skipped'] for r in out['records']):
                        raise ValueError('Native unpruned parity failed')
                else:
                    ref=load('reference')
                    if out['completion_tokens']!=ref['completion_tokens']:raise ValueError('Trusted generation parity failed')
                    for kind in ('local','global'):
                        for key in ('eligible','skipped'):
                            if sum(r[key] for r in out['records'] if r['attention_type']==kind)!=sum(r[key] for r in ref['records'] if r['attention_type']==kind):
                                raise ValueError('Trusted physical counts mismatch')
                for key in backend.evidence.DECODING_FIELDS:
                    if out['generation_metadata'].get(key)!=native['generation_metadata'].get(key):raise ValueError('Decoding changed')
    inherited=read(root/'inherited_validation.json');fb.check_sources(inherited['sources']);sources.update(inherited['sources'])
    return sources


def proposals(root,contract):
    with dimension_scope(),patch.object(screen,'PROJECTED',NEW),patch.object(screen,'BASELINES',{}),patch.object(screen,'SENSITIVITY_SEEDS',()):
        return screen.shared_screen(root/'rank_proposals',read(root/'shared_state_index.json'),contract)


def warm_start(root,name,benchmark,target):
    if name not in NEW or benchmark!='aime26' or target!=.5:raise ValueError('Unexpected calibration request')
    path=PREVIOUS/'policies/aime26/jl_gaussian_r32_s50.json'
    values,sources=screen.distributions(root/'rank_proposals',name,benchmark);sources[str(path)]=sha(path.read_bytes())
    return read(path),{k:np.maximum(v,-1e30) for k,v in values.items()},sources


def calibrate(adapter,root,setup,contract,name):
    with dimension_scope(),patch.object(continuation,'NEW',NEW),patch.object(continuation,'audit_policy',audit_policy), \
         patch.object(continuation,'warm_start',lambda n,b,t:warm_start(root,n,b,t)), \
         patch.object(fb,'calibration_rows',previous.prior.calibration_rows):
        return continuation.calibrate(adapter,root,setup,contract,name,'aime26',.5)


def freeze_conditions(root,setup,contract):
    with patch.object(previous,'CONDITIONS',CONDITIONS),patch.object(previous,'CONFIGS',CONFIGS), \
         patch.object(previous,'smoke_audit',smoke_audit),patch.object(previous,'audit_policy',audit_policy):
        return previous.freeze_conditions(root,setup,contract)


def accounting(root):
    a=previous.accounting(root)
    a.update(final_generation_slots=60,reused_final_baselines_and_references=270,
        note='60 new final slots plus270 cached comparisons; calibration and12 short validation calls additional. Selected calibration6/rank are exact final aliases.')
    _write(root/'inference_accounting.json',a);return a


def write_report(root,setup,rows,tasks,compared,diag,audit,policies):
    audit.update(expected=EXPECTED,complete=audit['completed']==EXPECTED and not audit['missing'] and not audit['violations'])
    display=[dict(split=r['split'],condition=r['condition'],threshold=backend.threshold_text(r),correct=r['correct'],count=r['count'],
        accuracy=100*r['accuracy'],delta_pp=100*r['delta'],ci95_pp=[100*v for v in r['paired_ci95']],
        whole=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_mass'],agreement=100*r['token_agreement'],local_operator_error=r['overall_relative_error']) for r in rows]
    backend.csv_write(root/'main_results.csv',display)
    text=f"# AIME26 Gaussian dimensions2/4 versus8/16/24/32 and full-dimensional\n\nStatus: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/{EXPECTED} audited outputs.60 new final slots and270 reused comparisons.\n\n"
    text+='## Setup and calibration\n\nUnchanged30 AIME26 questions,2048-token budget, prompts/scorer, seed42, BF16 model revision '+setup['revision']+'. Canvas256,up to48 denoising steps,thinkingFalse and native0.4–0.8 schedule (temperature0 sentinel is not greedy).128-query×64-KV physical tiles; prefix+canvas eligible with native masks and GQA. Gaussian centered routing uses FP32 sketches/state, seed1729 and unchanged rank-dependent matrix/hash construction. Only a scoped configuration allow-list is extended; existing generic kernels, first-support/tie handling and original-V renormalized output are unchanged.\n\n'
    text+='Each new rank starts from the same unverified Gaussian32 scalar trial, with its own empirical CDF from240 saved calibration QKV states. The existing local/global rank-proposal/scalar fallback checks the same six questions at2048 tokens, max12 joint points, requiring overall/global/local48–52%. Only sparsity selects thresholds, never accuracy. All new policies freeze before final inference. The six calibration questions are included in headline30; noncalibration24 is separately reported and was previously examined, not fresh held-out confirmation. IDs: '+', '.join(setup['calibration_ids']['aime26'])+'.\n\n'
    text+='All270 predecessor outputs/policies and256 shared-state diagnostics are reused unchanged in numerical content. New dimensions receive CPU/CUDA tests and fresh unpruned-native/pruned-trusted generation checks on two inputs. Two compatible native smoke outputs are reused. Failures are preserved; independent ranks continue.\n\n'
    text+='## Results\n\n'+backend.findings(rows,compared)+'\n\n'
    for split in ('full','noncalibration24','calibration6'):
        text+=f'### {split}\n\n'+backend.table([r for r in display if r['split']==split],tuple(display[0]))+'\n\n'
    text+='## Metrics and limitations\n\nPhysical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), not an average of percentages; dense prefill excluded. Retained mass and local operator error use dense attention on each execution\'s QKV, not cached dense-generation QKV after divergence. Shared-QKV diagnostics separately use32 selected early calibration states per method, one selected head/query tile per state; error is sqrt(SUM(error squared)/SUM(dense output squared)). Token agreement includes positions after divergence and treats missing/extra tokens as disagreements, including EOS. Raw per-layer/head/step/region counts, sketch work and projection hashes are exported.\n\n'
    text+='Original/aggressive BLASST, mass-only and all previous directional policies remain unchanged. Original BLASST caps lambda at1 and cannot attain50% here; its lower actual sparsity is not matched. Mass-only is the existing max-based mass bound. Compare actual whole/global/local sparsity rather than target labels.\n\n'
    text+='One fixed projection seed; rank changes also change random directions. No seed robustness or monotonic dimension claim follows. Paired prompt-bootstrap95% intervals condition on policies/seeds, are exploratory and not multiplicity-corrected. QK, block softmax and projected PV are still computed; full PV is diagnostic-only for sketch routing. Retained native PV remains a masked dense-shaped matmul. No hardware-speedup claim.\n\n'
    text+='![AIME tradeoffs](figures/aime26_tradeoffs.png)\n\n![Target versus actual](figures/aime26_target_actual.png)\n\n'
    if audit['complete']:
        with patch.object(paired,'PAIRS',PAIRS):direct=paired.analyze(read(root/'per_sample.json'))
        _write(root/'dimension_pairwise.json',direct);backend.csv_write(root/'dimension_pairwise.csv',direct)
        text+='## New ranks versus existing dimensions\n\n'+backend.table([r for r in direct if r['split']=='full'],
            ('candidate','reference','candidate_correct','reference_correct','delta_pp','paired_ci95_pp','overall_sparsity_gap_pp','global_sparsity_gap_pp','local_sparsity_gap_pp','within2pp_all_types'))+'\n\n'
    text+=f"Missing={len(audit['missing'])}; violations={len(audit['violations'])}. Raw-only regeneration: `python -m {MODULE} report`; independently repeat with `verify`.\n"
    (root/'report.md').write_text(text)


def regenerate(root=ROOT):
    def plots(folder,rows):
        from matplotlib.figure import Figure
        original=Figure.suptitle
        with patch.object(Figure,'suptitle',lambda fig,title,*a,**kw:original(fig,title.replace('50/75% targets','50% target; Gaussian dimensions2–32'),*a,**kw)):
            return [p for p in previous.BASE_PLOTS(folder,rows) if 'aime26' in p]
    with dimension_scope(),patch.object(backend,'prepare',prepare),patch.object(backend,'execution',execution), \
         patch.object(backend,'PROJECTED',ALL),patch.object(backend,'TARGETS',(.5,)),patch.object(backend.common,'TARGETS',(.5,)), \
         patch.object(backend,'audit_policy',audit_policy),patch.object(backend,'smoke_audit',smoke_audit), \
         patch.object(backend.common,'shared_seed_summary',lambda unused:([],{})), \
         patch.object(backend,'write_report',write_report),patch.object(backend,'plots',plots):
        audit=backend.regenerate(root)
    accounting(root)
    for name in ('inference_accounting.json','dimension_pairwise.json','dimension_pairwise.csv'):
        if (root/name).exists():audit['artifacts'][name]=sha((root/name).read_bytes())
    _write(root/'audit.json',audit);return audit


def verify(root=ROOT):
    before=read(root/'audit.json')
    if not before['complete'] or before['completed']!=EXPECTED:raise ValueError('Require all330 results')
    fb.check_sources({str(root/p):h for p,h in before['artifacts'].items()})
    if regenerate(root)!=before:raise ValueError('Independent raw-only regeneration changed')
    result=dict(passed=True,completed=EXPECTED,inference_performed=False,audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json',result);return result


def work(root):
    setup,contract=prepare(root),execution(root)
    fb.phase(root,'reuse_cached_comparisons');reuse(root,setup,contract);accounting(root)
    torch.backends.cuda.matmul.allow_tf32=False;fb.phase(root,'load_model')
    adapter=create_adapter('diffusion_gemma',setup['model'],device='cuda',precision='bfloat16',revision=setup['revision']).load()
    with dimension_scope():
        fb.phase(root,'two_rank_actual_model_smoke')
        with patch.object(validation,'PROJECTED',NEW):validation.smoke(adapter,root,setup,contract)
        smoke_audit(root,setup,contract);accounting(root)
        fb.phase(root,'rank2_4_calibration_state_proposals');proposals(root,contract)
        for name in NEW:
            try:calibrate(adapter,root,setup,contract,name)
            except Exception:fb.failure(root,'calibration',method=name,benchmark='aime26',target=.5)
            accounting(root)
        conditions=freeze_conditions(root,setup,contract);fb.phase(root,'shared_state_diagnostics')
        try:
            with patch.object(shared_analysis,'PROJECTED',ALL),patch.object(shared_analysis,'BASELINES',previous.prior.parent.BASELINES),patch.object(shared_analysis,'TARGETS',(.5,)):
                shared_analysis.analyze(root,read(root/'shared_state_index.json'),contract)
        except Exception:fb.failure(root,'shared_state_diagnostics')
        for name,cfg in NEW.items():
            label=name+'_s50'
            if label not in conditions:continue
            p=read(root/'policies/aime26'/f'{label}.json');selected=next(x for x in p['trace'] if x['iteration']==p['selected_round'])
            fb.phase(root,'final',condition=label,expected=30)
            for row in sorted(setup['final'],key=lambda r:(len(r['prompt_tokens']),r['id'])):
                try:
                    if row['id'] in setup['calibration_ids']['aime26']:
                        fb.alias(root,row,'final',label,name,cfg,p['policy'],contract,Path(selected['sources'][row['id']]['path']),contract['fingerprint'])
                    else:runner.cached(adapter,root,row,'final',label,name,cfg,p['policy'],contract)
                except Exception:fb.failure(root,'final',condition=label,id=row['id'])
            accounting(root)
    del adapter;torch.cuda.empty_cache();fb.phase(root,'report');audit=regenerate(root)
    if audit['complete']:fb.phase(root,'independent_report_verification');verify(root)
    _write(root/'terminal.json',dict(complete=audit['complete'],completed=audit['completed'],expected=EXPECTED,finished=time.time()))
    fb.phase(root,'finished',complete=audit['complete'])


def supervise(root):
    with patch.object(previous,'MODULE',MODULE):previous.supervise(root)


def launch(root):
    previous.transfer.require_complete(PREVIOUS,270);execution(root);fb.gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'supervise','--output',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,final_slots=60,reused_slots=270,dimensions=[2,4],target=.5,calibration_validation_additional=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','launch','supervise','work','report','verify'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    result=(regenerate if args.command=='report' else globals()[args.command])(args.output)
    if args.command in ('prepare','report','verify'):print(json.dumps(dict(command=args.command,complete=result.get('complete'),passed=result.get('passed'))))
