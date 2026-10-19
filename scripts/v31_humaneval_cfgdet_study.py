"""Corrected, bounded HumanEval evaluation with fixed K128/K256 budgets."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from scripts import v31_humaneval_value_study as inherited
from scripts import v31_v2_local_threshold_sweep as base
from scripts.v31_humaneval_cfgdet_worker import CONTROL,V2,V3A,REFINEMENTS

ROOT=next(p for p in (Path.cwd(),*Path(__file__).absolute().parents) if (p/'.git').exists())
INVESTIGATION=ROOT/'results/humaneval_reproducibility_20261019/attempt001'
PARENT=ROOT/'results/humaneval_value_k256_20261018/attempt001'
SPEC=ROOT/'docs/HUMANEVAL_CFGDET_K256_K128_PLAN_20261019.md'
PREFLIGHT=ROOT/'results/humaneval_cfgdet_preflight_20261019/attempt001'
MODES=inherited.MODES
CANDIDATES=inherited.CANDIDATES
REFERENCES=inherited.REFERENCES
LABELS=inherited.LABELS
GRID=inherited.GRID
read,write,sha=base.read,base.write,base.sha
entry=inherited.entry
tau_label=inherited.tau_label
first_canvas=inherited.first_canvas
work=inherited.work
match=inherited.match
choose_threshold=inherited.choose_threshold
refinement_points=inherited.refinement_points
extract=inherited.extract


def phase_caps():
    add=read(INVESTIGATION/'allocation_addendum.json')
    if (sha(ROOT/add['spec_file'])!=add['spec_sha256'] or
        sha(INVESTIGATION/'protocol.json')!=add['prior_protocol_sha256']):
        raise RuntimeError('GPU allocation addendum drift')
    if sum(add['caps_s'].values())!=43200:
        raise RuntimeError('unauthorized total GPU-budget change')
    return add['caps_s']


def prepare(study,budget):
    if budget not in (128,256) or study.exists():
        raise RuntimeError('new fixed-budget attempt required')
    qualification=INVESTIGATION/'qualification/attempt001/baseline_runtime_qualified.json'
    q=read(qualification)
    if q['status']!='baseline_runtime_qualified':raise RuntimeError('baseline runtime unqualified')
    gate=read(PREFLIGHT/'preflight_complete.json')
    if gate['status']!='complete' or sha(PREFLIGHT/'tests.xml')!=gate['tests_sha256']:
        raise RuntimeError('qualified H100 tests required')
    for rel,digest in read(PARENT/'verified_complete.json')['files_sha256'].items():
        if sha(PARENT/rel)!=digest:raise RuntimeError('original campaign drift')
    study.mkdir(parents=True)
    execution=study/'private/execution'
    shutil.copytree(PARENT/'private/execution',execution)
    # The inherited selector/native execution remains byte-identical. Overlay
    # only the generalized harness and the qualified numerical-policy worker.
    overlays=['v31_humaneval_cfgdet_worker.py','v31_humaneval_cfgdet_study.py',
              'v31_humaneval_cfgdet_report.py','v31_humaneval_cfgdet_verify.py']
    for name in overlays:
        shutil.copyfile(ROOT/'scripts'/name,execution/'scripts'/name)
    shutil.copyfile(INVESTIGATION/'private/source_addenda/attempt027/worker.py',
                    execution/'scripts/v31_humaneval_cfgdet_policy_worker.py')
    shutil.copytree(PARENT/'private/pools',study/'private/pools')
    shutil.copytree(PARENT/'private/schedules',study/'private/schedules')
    shutil.copytree(INVESTIGATION/'private/cache',study/'private/cache')
    cohort=read(INVESTIGATION/'protocol.json')['cohort']
    for mode in MODES:
        folder=study/f'private/schedules/{mode}'
        cells=read(folder/'cells.json')
        selected=[next(c for c in cells if c['index']==i) for i in cohort]
        # Qualification's seed42 prefix matches the first primary shard's
        # request/arm order exactly. The extra seed follows that prefix only.
        write(folder/'qualification.json',selected+[dict(c,seed=43) for c in selected],True)
        rest=[c for c in cells if c['index'] not in cohort]
        write(folder/'primary_shard001.json',selected+rest[:76],True)
        write(folder/'primary_shard002.json',rest[76:],True)
        write(folder/'all_work.json',selected+rest,True)
    shutil.copyfile(PARENT/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
    shutil.copyfile(SPEC,study/'spec.md')
    shutil.copyfile(PREFLIGHT/'tests.xml',study/'preflight_tests.xml')
    original=read(PARENT/'protocol.json')
    settings=dict(inherited.SETTINGS,mage_k=budget)
    p=dict(schema='humaneval164_cfgdet_fixed_budget_v1',prefix_budget_tokens=budget,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selector_parent_protocol_sha256=sha(PARENT/'protocol.json'),
        baseline_runtime_qualification_sha256=sha(qualification),
        numerical_policy_sha256=sha(execution/'scripts/v31_humaneval_cfgdet_policy_worker.py'),
        execution_fingerprints={str(f.relative_to(execution)):sha(f) for f in execution.rglob('*') if f.is_file()},
        input_fingerprints={str(f.relative_to(study)):sha(f) for root in (study/'private/pools',study/'private/schedules')
                            for f in root.rglob('*') if f.is_file()},
        model_inventory_sha256=sha(study/'model_source_fingerprints.json'),spec_sha256=sha(study/'spec.md'),
        preflight_tests_sha256=sha(study/'preflight_tests.xml'),runtime=base.runtime_identity(),
        suites={mode:settings for mode in MODES},native=original['native'],manifests=original['manifests'],
        seed=42,extra_seed=43,cohort=cohort,references=REFERENCES,candidates=CANDIDATES,
        per_query_tile='query-head/Q128, prefix KV64 only; dense canvas/boundary and LOCAL',
        calibration='attention work only; previous threshold first for K256; fresh K128; full164 work verification',
        matching_tolerance=dict(sparsity_pp=.25,kept_fraction_relative=.01),
        phase='k'+str(budget),phase_cap_s=phase_caps()['k'+str(budget)],
        maximum_job_reservation_s=1800,passes=['audit','clean'],primary_control='control_before',
        status='frozen_before_generation')
    write(study/'protocol.json',p,True)
    write(study/'receipts/source_fingerprints.json',dict(source_commit=p['source_commit'],
        execution_fingerprints=p['execution_fingerprints'],numerical_policy_sha256=p['numerical_policy_sha256']),True)
    write(study/'state.json',dict(stage='prepared',resumable=True,full_accuracy_released=False))
    return p


def verify(study):
    p=read(study/'protocol.json');base.verify(study,p)
    for rel,key in [('spec.md','spec_sha256'),('model_source_fingerprints.json','model_inventory_sha256'),
                    ('preflight_tests.xml','preflight_tests_sha256')]:
        if sha(study/rel)!=p[key]:raise RuntimeError('protocol/source identity drift')
    if p['prefix_budget_tokens'] not in (128,256):raise RuntimeError('unsupported prefix budget')
    if sha(INVESTIGATION/'qualification/attempt001/baseline_runtime_qualified.json')!=p['baseline_runtime_qualification_sha256']:
        raise RuntimeError('baseline qualification identity drift')
    base.SUITES.update(p['suites'])
    return p


def schedule(cells, warm, family, pass_name, threshold, calibration=None):
    rows, entries, groups = [], [entry(CONTROL)], []
    if family == 'shared':
        warm_entries = [entry(m, threshold if m == V2 else None) for _ in warm for m in CANDIDATES]
        for c in warm:
            rows.extend([c]*len(CANDIDATES))
        entries.extend(warm_entries)
        warm_count = len(rows)
        for c in cells:
            shift = c['index'] % len(CANDIDATES)
            arms = list(CANDIDATES[shift:]+CANDIDATES[:shift])
            if pass_name == 'audit':
                arms.reverse()
            group = [entry(CONTROL, label='control_before')]+[entry(m, threshold if m == V2 else None) for m in arms]+[entry(CONTROL, label='control_after')]
            groups.append(dict(start=len(rows), end=len(rows)+len(group), labels=[g['label'] for g in group]))
            rows.extend([c]*len(group))
            entries.extend(group)
    elif family == 'calibration':
        for c in warm:
            for t in (0., 1.):
                rows.append(c); entries.append(entry(V2, t, tau_label(t)))
        warm_count = len(rows)
        for c in cells:
            group = [entry(CONTROL, label='control_before')]+[entry(V2, t, tau_label(t)) for t in calibration]
            groups.append(dict(start=len(rows), end=len(rows)+len(group), labels=[g['label'] for g in group]))
            rows.extend([c]*len(group)); entries.extend(group)
    elif family == V2:
        for c in warm:
            rows.append(c); entries.append(entry(V2, threshold))
        warm_count = len(rows)
        for c in cells:
            groups.append(dict(start=len(rows), end=len(rows)+1, labels=[V2]))
            rows.append(c); entries.append(entry(V2, threshold))
    else:
        rows.extend([c for c in warm for _ in CANDIDATES])
        warm_count = len(rows)
        for c in cells:
            groups.append(dict(start=len(rows), end=len(rows)+1, labels=[family]))
            rows.append(c)
    return rows, entries, groups, warm_count



def validate(row, cell, family, live, pass_name, oracle=False, prefix_budget=256):
    errors = []
    a = ((row.get('receipts') or {}).get('adapter') or {})
    checks = [(row.get('dataset') == cell['dataset'] and row.get('index') == cell['index'] and row.get('panel_seed') == cell['seed'] and row.get('repeat') == 0, 'input/seed'),
        (row.get('cuda_graph_captures') == 0, 'measured CUDA capture'),
        (row.get('block_size') == 64 and row.get('chunk') == 4096 and row.get('max_model_len') == 9216, 'geometry'),
        (row.get('fix_51994') is True and row.get('fa4_local_fix') is True, 'dense/LOCAL fixes'),
        (row.get('value_clean_timing') == (pass_name == 'clean') and row.get('value_audit') == (pass_name == 'audit'), 'pass flags'),
        (row.get('denoise_forwards', 0) > 0 and row.get('canvases', 0) > 0 and row.get('output_tokens', 0) > 0, 'empty native output'),
        (row.get('budget') == 8192 and row.get('local_kv_budget') is None, 'output/LOCAL budget'),
        (row.get('cuda_matmul_allow_tf32') is False, 'FP32 precision'),
        (row.get('engine_steps') == row.get('prefill_steps', 0)+row.get('sampler_calls', 0), 'native call accounting')]
    if pass_name == 'clean':
        checks += [(row.get('timer_policy') == 'request_and_prefill_boundaries', 'clean boundaries'),
                   (abs(row['wall_s']-row['decode_s']-row['prefill_s']) < 2e-5, 'time sum')]
    if family in ('shared', 'calibration', V2):
        arm = live['arm']
        checks += [(a.get('humaneval_live_arm') == arm, 'live arm'),
                   (a.get('humaneval_threshold') == live.get('threshold'), 'live threshold'),
                   (a.get('humaneval_prefix_budget_tokens') == prefix_budget, 'prefix budget')]
        if arm != V2 and pass_name == 'audit':
            checks.append((a.get('humaneval_fixed_budget_errors') == 0, 'fixed query-tile budget'))
        if arm in REFINEMENTS:
            checks += [(a.get('value_refinement') == arm and a.get('value_refinement_layers') == [29], 'refinement scope'),
                       (a.get('value_refinement_backend') == 'device_exact_v1', 'device backend')]
            if pass_name == 'audit':
                checks.append((a.get('value_refinement_work', {}).get('budget_errors') == 0, 'refined budget'))
                if oracle:
                    checks.append((a.get('value_refinement_work', {}).get('humaneval_native_oracle_checks', 0) > 0, 'native oracle'))
    if pass_name == 'audit':
        checks += [(a.get('global_calls') == 5*row['denoise_forwards'], 'GLOBAL calls'),
                   (a.get('native_local_calls_audited') == 25*row['denoise_forwards'], 'LOCAL calls'),
                   (a.get('native_local_skipped_tiles') == 0, 'LOCAL skips'),
                   (a.get('global_eligible_tiles', 0) > 0 and a.get('native_local_eligible_rectangles', 0) > 0, 'physical denominator')]
        if family in ('shared', 'calibration', V2):
            first_canvas(row)
    for okay, name in checks:
        if not okay:
            errors.append(name)
    return errors


def verify_leaf(study,leaf):
    done=read(leaf/'attempt_complete.json')
    if done['protocol_sha256']!=sha(study/'protocol.json'):
        raise RuntimeError('leaf protocol drift')
    for name,digest in done['files_sha256'].items():
        if sha(leaf/name)!=digest:raise RuntimeError('completed native artifact drift')
    return done


def reserve(study,p,leaf):
    budget=read(INVESTIGATION/'gpu_budget.json');caps=phase_caps();phase=p['phase']
    if any('elapsed_s' not in r for r in budget['reservations']):
        raise RuntimeError('another GPU reservation is open')
    spent=sum(r['elapsed_s'] for r in budget['reservations'] if r['phase']==phase)
    total=sum(r['elapsed_s'] for r in budget['reservations'])
    remaining=min(caps[phase]-spent,sum(caps.values())-total)
    reservation=min(p['maximum_job_reservation_s'],remaining-30)
    if reservation<180:raise RuntimeError('phase GPU budget exhausted; preserve resumable state')
    record=dict(name=str(leaf.relative_to(ROOT)),phase=phase,reserved_s=reservation,
                protocol_sha256=sha(study/'protocol.json'))
    budget['reservations'].append(record);write(INVESTIGATION/'gpu_budget.json',budget)
    return budget,record


def policy_errors(path,cg):
    if not path.exists():return ['missing numerical-policy receipt']
    x=read(path);p=x.get('policy',{});helpers=p.get('native_helpers',{})
    checks=[(bool(x.get('kernels')) and all(k['deterministic'] for k in x.get('kernels',[])),
             'actual deterministic compiler kernels'),
        (p.get('effective_cudagraph_mode')==('FULL_AND_PIECEWISE' if cg=='default' else 'PIECEWISE'),
             'native graph mode'),
        (p.get('effective_inductor_compile_config',{}).get('deterministic') is True,
             'explicit model compiler options'),
        (set(helpers)=={'_softcap_logits','_compute_num_rejected','_compiled_sample_step'} and all(
             h.get('original_python_callable_preserved') is True and
             h.get('options',{}).get('deterministic') is True for h in helpers.values()),
             'unchanged native helper Python functions')]
    return [name for okay,name in checks if not okay]


def read_complete_jsonl(path):
    if not path.exists():return []
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line]


def jit_events(log,warm_count):
    printed=0;events=[]
    for line in log.read_text(errors='replace').splitlines():
        try:
            x=json.loads(line)
            if isinstance(x,dict) and 'denoise_forwards' in x:printed+=1
        except (ValueError,TypeError):pass
        if printed>=warm_count and 'WARNING' in line and 'jit_monitor' in line:
            events.append(dict(after_saved_requests=printed,message=line.split(']')[-1].strip()))
    return events


def run_job(study,stage,mode,pass_name,family,cells_file,threshold=None,calibration=None,oracle=False):
    p=verify(study)
    if (study/'PAUSED_USER_REQUEST.json').exists():raise RuntimeError('campaign paused')
    parent=study/stage/mode/pass_name/family/cells_file.stem
    done=list(parent.glob('attempt*/attempt_complete.json'))
    if len(done)>1:raise RuntimeError('ambiguous completed attempts')
    if done:verify_leaf(study,done[0].parent);return done[0].parent
    number=max([int(x.name[7:]) for x in parent.glob('attempt*')],default=0)+1
    leaf=parent/f'attempt{number:03d}';(leaf/'private').mkdir(parents=True)
    cells=read(cells_file);warm=read(study/f'private/schedules/{mode}/warmup.json')
    rows,entries,groups,warm_count=schedule(cells,warm,family,pass_name,threshold,calibration)
    write(leaf/'private/cells.json',rows,True);write(leaf/'private/arms.json',entries,True)
    write(leaf/'config.json',dict(protocol_sha256=sha(study/'protocol.json'),stage=stage,mode=mode,
        pass_name=pass_name,family=family,threshold=threshold,calibration=calibration,
        groups=groups,warmup_requests=warm_count,oracle=oracle),True)
    base.wait_idle()
    if base.runtime_identity()!=p['runtime']:raise RuntimeError('runtime identity drift')
    execution=study/'private/execution';env=base.env_for(study,mode,CONTROL)
    shared=family in ('shared','calibration',V2)
    native_worker=execution/'scripts'/('v31_humaneval_cfgdet_worker.py' if shared else 'v31_vllm_paired_bench.py')
    policy_worker=execution/'scripts/v31_humaneval_cfgdet_policy_worker.py'
    env.update(STUDY_ARM=family,PYTHONHASHSEED='0',REPEATS='1',TRACE=str(int(pass_name=='audit')),
        PYTHONPATH=str(execution/'src')+':'+str(execution),
        FROZEN_SOURCE_SHA256=sha(study/'receipts/source_fingerprints.json'),
        VALUE_AUDIT=str(int(pass_name=='audit')),VALUE_CLEAN_TIMING=str(int(pass_name=='clean')),
        VALUE_BOUNDARY_TIMING=str(int(pass_name=='clean')),
        HUMANEVAL_REQUEST_ARMS=str(leaf/'private/arms.json'),HUMANEVAL_PREFIX_BUDGET=str(p['prefix_budget_tokens']),
        HUMANEVAL_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'),
        HUMANEVAL_REPRO_BASE=str(native_worker),HUMANEVAL_ORACLE=str(int(oracle)),
        HUMANEVAL_ORACLE_ANCHORS_PER_REQUEST='2',
        TORCHINDUCTOR_DETERMINISTIC='1',HUMANEVAL_KERNEL_INVENTORY=str(leaf/'private/kernel_inventory.json'))
    arm='mage' if shared else 'allkept' if family=='allkept_fa4' else 'dense'
    cg='default' if family=='dense_full_fix51994' else 'PIECEWISE'
    cmd=[base.PYTHON,str(policy_worker),str(base.MODEL),str(study/f'private/pools/{mode}'),
         str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),arm,cg]
    write(leaf/'private/launch.json',dict(command=cmd,environment={k:v for k,v in env.items()
        if k not in os.environ or os.environ[k]!=v}),True)
    budget,reservation=reserve(study,p,leaf)
    write(study/'state.json',dict(stage=stage,mode=mode,pass_name=pass_name,family=family,
        active_attempt=str(leaf.relative_to(study)),resumable=True,full_accuracy_released=False))
    print(dict(event='launch',stage=stage,mode=mode,pass_name=pass_name,family=family,
               requests=len(rows),reserved_s=reservation['reserved_s']),flush=True)
    start=time.monotonic();errors=[];checked=0;last=start
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(leaf/'private/process.json',dict(pid=proc.pid,command=cmd),True)
        while proc.poll() is None:
            public=read_complete_jsonl(leaf/'records.jsonl')
            if len(public)>len(rows):
                errors.append('unexpected extra request records')
                public=public[:len(rows)]
            for i in range(checked,len(public)):
                errors.extend(validate(public[i],rows[i],family,entries[i+1] if shared else None,
                                       pass_name,oracle,p['prefix_budget_tokens']))
            checked=len(public)
            if errors or time.monotonic()-start>=reservation['reserved_s']:
                if not errors:errors.append('GPU reservation expired')
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            if time.monotonic()-last>=1800:
                print(dict(event='progress',stage=stage,mode=mode,pass_name=pass_name,
                           family=family,records=checked,planned=len(rows)),flush=True);last=time.monotonic()
            time.sleep(min(5,max(.05,reservation['reserved_s']-(time.monotonic()-start))))
    elapsed=time.monotonic()-start
    reservation.update(elapsed_s=elapsed,returncode=proc.returncode)
    write(INVESTIGATION/'gpu_budget.json',budget)
    public=base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    private=base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
    if proc.returncode or len(public)!=len(rows) or len(private)!=len(rows):errors.append('worker/coverage failure')
    for i in range(checked,len(public)):
        errors.extend(validate(public[i],rows[i],family,entries[i+1] if shared else None,
                               pass_name,oracle,p['prefix_budget_tokens']))
    errors.extend(policy_errors(leaf/'private/kernel_inventory.json',cg))
    jit=jit_events(leaf/'private/worker.log',warm_count)
    if pass_name=='clean' and jit:errors.append('measured kernel JIT; clean latency unqualified')
    receipt=dict(status='failed' if errors else 'complete',errors=sorted(set(errors)),returncode=proc.returncode,
        records=len(public),measured_records=len(rows)-warm_count,gpu_reservation_s=elapsed,
        protocol_sha256=sha(study/'protocol.json'),post_warmup_jit=jit)
    write(leaf/'receipt.json',receipt,True)
    if errors:
        write(leaf/'failed_attempt.json',dict(**receipt,files_sha256={str(f.relative_to(leaf)):sha(f)
            for f in leaf.rglob('*') if f.is_file()}),True)
        raise RuntimeError('preserved failed attempt '+str(leaf)+': '+str(receipt['errors']))
    exports=defaultdict(list)
    for group in groups:
        for i,label in zip(range(group['start'],group['end']),group['labels']):exports[label].append(i)
    for label,indices in exports.items():
        for name,data in [(f'exports/{label}/records.jsonl',[dict(public[i],study_arm=label) for i in indices]),
            (f'exports/{label}/private/run_{label}.private.jsonl',[private[i] for i in indices])]:
            dest=leaf/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('x') as f:
                for row in data:f.write(json.dumps(row)+'\n')
    receipt['files_sha256']={str(f.relative_to(leaf)):sha(f) for f in leaf.rglob('*') if f.is_file()}
    write(leaf/'attempt_complete.json',receipt,True)
    print(dict(event='complete',stage=stage,mode=mode,pass_name=pass_name,family=family,
               records=len(public),gpu_reservation_s=elapsed),flush=True)
    return leaf


def exported(leaf,label):
    return base.jsonl(leaf/f'exports/{label}/records.jsonl')


def calibrate(study,mode):
    p=verify(study);dest=study/f'calibration/{mode}/threshold.json'
    if dest.exists():return read(dest)['threshold']
    folder=study/f'private/schedules/{mode}';weights=read(folder/'calibration_weights.json')
    candidates={};reference=None;previous=None
    if p['prefix_budget_tokens']==256:
        previous=read(PARENT/f'calibration/{mode}/full164_threshold.json')['threshold']
        leaf=run_job(study,'calibration_previous',mode,'audit','calibration',folder/'calibration.json',calibration=[previous])
        reference=work(exported(leaf,'control_before'),weights)
        candidates[previous]=work(exported(leaf,tau_label(previous)),weights)
    if previous is None or not match(reference,candidates[previous])['matched']:
        grid=list(GRID) if previous is None else [previous/2,previous*2]
        leaf=run_job(study,'calibration_grid',mode,'audit','calibration',folder/'calibration.json',calibration=grid)
        fresh=work(exported(leaf,'control_before'),weights)
        if reference is not None and fresh!=reference:
            raise RuntimeError('V31 work-calibration process repeat changed')
        reference=fresh
        candidates.update({t:work(exported(leaf,tau_label(t)),weights) for t in grid})
        mids=refinement_points(reference,candidates)
        if mids:
            leaf=run_job(study,'calibration_midpoints',mode,'audit','calibration',folder/'calibration.json',calibration=mids)
            if work(exported(leaf,'control_before'),weights)!=reference:
                raise RuntimeError('V31 calibration repeat changed')
            candidates.update({t:work(exported(leaf,tau_label(t)),weights) for t in mids})
    best=choose_threshold(reference,candidates)
    write(dest,dict(status='work_calibrated',threshold=best,reference=reference,
        candidates={str(t):v for t,v in candidates.items()},matching=match(reference,candidates[best]),
        previous_threshold=previous,previous_retained=best==previous,accuracy_used=False,
        question_count=24,weighted_population=164,protocol_sha256=sha(study/'protocol.json')),True)
    return best


def freeze_full_work(study,mode,threshold):
    dest=study/f'calibration/{mode}/full164_threshold.json'
    if dest.exists():return read(dest)['threshold']
    folder=study/f'private/schedules/{mode}';points={};leaves={};reference=None;control=None
    current=threshold
    grid=read(study/f'calibration/{mode}/threshold.json')
    for iteration in range(3):
        leaf=run_job(study,f'full_work{iteration+1:03d}',mode,'audit','calibration',folder/'all_work.json',calibration=[current])
        rows=exported(leaf,'control_before')
        identity=[(r['index'],r['rng_seed'],r['output_hash'],r['denoise_forwards'],r['canvases'],r['output_tokens']) for r in rows]
        if control is not None and identity!=control:
            raise RuntimeError('full-cohort V31 calibration repeat changed')
        control=identity;reference=work(rows)
        points[current]=work(exported(leaf,tau_label(current)))
        leaves[str(current)]=str(leaf.relative_to(study))
        best=choose_threshold(reference,points)
        if match(reference,points[best])['matched']:break
        all_points={float(t):v for t,v in grid['candidates'].items()};all_points.update(points)
        mids=refinement_points(reference,all_points)
        choices=[t for t in mids if t not in points]
        if not choices:break
        current=min(choices,key=lambda t:abs(t-best))
    best=choose_threshold(reference,points);matched=match(reference,points[best])
    write(dest,dict(status='frozen_before_primary_accuracy',threshold=best,reference=reference,
        candidates={str(t):v for t,v in points.items()},matching=matched,work_leaves=leaves,
        accuracy_used=False,protocol_sha256=sha(study/'protocol.json')) ,True)
    if not matched['matched']:
        print(dict(event='closest_work_threshold_unmatched',mode=mode,**matched),flush=True)
    return best


def selected_paths(study,mode,pass_name,stage='runs'):
    result=defaultdict(list)
    for done in sorted((study/stage/mode/pass_name).glob('*/*/attempt*/attempt_complete.json')):
        leaf=done.parent;verify_leaf(study,leaf)
        for label in LABELS:
            path=leaf/f'exports/{label}'
            if path.exists():result[label].append(path)
    return result


def prefix_reproducibility(study):
    comparisons=[]
    for mode in MODES:
        for pass_name in ('clean','audit'):
            qualification=selected_paths(study,mode,pass_name,'qualification')
            primary=selected_paths(study,mode,pass_name)
            for label in LABELS:
                # Qualification contains seed42 followed by the predeclared43.
                old={r['index']:r for path in qualification[label] for r in base.jsonl(path/'records.jsonl') if r['panel_seed']==42}
                new={r['index']:r for path in primary[label] for r in base.jsonl(path/'records.jsonl')}
                if set(old)!=set(read(study/'protocol.json')['cohort']):raise RuntimeError('replication cohort missing')
                differences={k:sum(old[i][k]!=new[i][k] for i in old) for k in (
                    'output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','rng_seed','prompt_sha256')}
                comparisons.append(dict(mode=mode,pass_name=pass_name,label=label,questions=6,
                                        differences=differences,passed=not any(differences.values())))
    return comparisons


def run(study):
    verify(study)
    # Canonical official test programs validate the scoring harness, never a
    # candidate threshold. Reuse the inherited data-bound canonical gate.
    inherited.scorer_gate(study)
    for mode in MODES:
        threshold=freeze_full_work(study,mode,calibrate(study,mode))
        folder=study/f'private/schedules/{mode}'
        for pass_name in ('audit','clean'):
            run_job(study,'qualification',mode,pass_name,'shared',folder/'qualification.json',threshold,
                    oracle=pass_name=='audit')
            for family in REFERENCES:
                run_job(study,'qualification',mode,pass_name,family,folder/'qualification.json')
    # Qualification audit/clean outputs are checked before primary generation.
    from scripts.v31_humaneval_cfgdet_report import qualification_equivalence,make_report
    proof=qualification_equivalence(study)
    if not all(x['passed'] for x in proof):raise RuntimeError('method audit/clean qualification failed')
    path=study/'qualification/methods_qualified.json'
    if not path.exists():write(path,dict(comparisons=proof,protocol_sha256=sha(study/'protocol.json')),True)
    for shard in (1,2):
        for mi,mode in enumerate(MODES):
            folder=study/f'private/schedules/{mode}'
            threshold=read(study/f'calibration/{mode}/full164_threshold.json')['threshold']
            families=['shared',*REFERENCES];shift=(shard+mi)%len(families)
            families=families[shift:]+families[:shift]
            for family in families:
                for pass_name in (('clean','audit') if (shard+mi)%2 else ('audit','clean')):
                    run_job(study,'runs',mode,pass_name,family,folder/f'primary_shard{shard:03d}.json',threshold)
    replica=prefix_reproducibility(study)
    write(study/'receipts/independent_prefix_replication.json',dict(comparisons=replica),True)
    if not all(x['passed'] for x in replica):raise RuntimeError('independent primary-engine repeat failed')
    make_report(study,study/'analysis/attempt001')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['prepare','run']);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--budget',type=int,choices=[128,256])
    a=p.parse_args();study=a.study.absolute()
    if a.action=='prepare':prepare(study,a.budget)
    else:run(study)


if __name__=='__main__':main()


