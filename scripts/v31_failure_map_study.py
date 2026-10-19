"""Bounded V31 budget screen, independent case confirmation, gated repair."""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_refresh_diagnostic_study as prior
from scripts import v31_official as official

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'results/v31_failure_map_20261019/attempt001'
SPEC = 'docs/V31_FAILURE_MAP_PLAN_20261019.md'
BUDGET = dict(qualification=900., screen=5400., confirmation=3600., repair=3600., recovery=900.)
CAP = 14400.
KS = (8192, 4096, 2048)
REFS = ('dense_full_fix51994', 'dense_piecewise', 'allkept_fa4')
SALT = 'v31-failure-map-20261019-metadata-v1'
SUITE = 'longbench_v2'
FILES = (SPEC, 'scripts/v31_failure_map_study.py', 'scripts/v31_failure_map_repair.py',
         'tests/test_v31_failure_map.py')


def key(c):
    return hashlib.sha256(f"{SALT}|{c['dataset']}|{c['index']}".encode()).hexdigest()


def choose(cells):
    groups = []
    for lo, hi in ((16384,32768),(32768,65536),(65536,98304),(98304,123136)):
        pool = sorted((c for c in cells if lo <= prior.token_length(c) < hi), key=key)
        if len(pool) < 5:
            raise ValueError('insufficient cohort/warmup coverage')
        groups.append(pool)
    return [c for g in groups for c in g[:4]], [groups[0][4], groups[-1][4]]


def available(phase, reservations):
    charges = defaultdict(float)
    for r in reservations:
        t = r['reserved_s'] if r.get('elapsed_s') is None else r['elapsed_s']
        if not math.isfinite(t) or t < 0:
            raise ValueError('invalid budget charge')
        charges[r['phase']] += t
    return max(0., min(BUDGET[phase]-charges[phase], CAP-sum(charges.values())))


def schedule(cells, warm, family, seeds, repeat, fixed=False, repair=False):
    rows, arms = [], [prior.entry('control', 'implicit_warmup', warmup=True)]
    # Same three explicit native inputs and budgets for each family.
    for i, c in enumerate((warm[0], warm[1], warm[0])):
        policy = 'fixed16' if fixed and family == 'shared' and i == 2 else 'control'
        rows.append(dict(c, seed=42)); arms.append(prior.entry(policy, 'warmup', warmup=True))
    ordered = sorted(cells, key=key)
    if repeat % 2:
        ordered.reverse()
    for c in ordered:
        for seed in (seeds[::-1] if repeat % 2 else seeds):
            group = f"{c['index']}:{seed}:{repeat}"
            if repair:
                policies = list(prior.FACTORIAL)
                shift = (c['index']+seed+repeat) % 4
                policies = policies[shift:]+policies[:shift]
                if repeat % 2:
                    policies.reverse()
                entries = [prior.entry('control','control_before')]+[prior.entry(x) for x in policies]+[prior.entry('control','control_after')]
            elif family == 'shared':
                entries = [prior.entry('control','control_before')]
                if fixed:
                    entries.append(prior.entry('fixed16','fixed16'))
                entries.append(prior.entry('control','control_after'))
            else:
                entries = [prior.entry('control', family)]
            for a in entries:
                rows.append(dict(c, seed=seed)); arms.append(dict(a, group_id=group))
    return rows, arms


def prepare(study):
    if study.exists():
        raise RuntimeError('new immutable attempt required')
    tracked = subprocess.check_output(['git','ls-files'], cwd=ROOT, text=True).splitlines()
    changed = set(subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).splitlines())
    files = [p for p in tracked if (p.startswith(('scripts/','src/','experiments/','tests/')) and
              Path(p).suffix in ('.py','.json','.yaml','.yml','.toml')) or p == SPEC]
    if not set(FILES) <= set(files) or set(files) & changed:
        raise RuntimeError('commit all execution sources and plan first')
    prior.pause_unchanged(); prior.require_idle()
    parent = prior.LB_PARENT
    cells, warm = choose(base.read(parent/f'private/schedules/{SUITE}/cells.json'))
    study.mkdir(parents=True)
    for rel in files:
        dest = study/'private/execution'/rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel,dest)
    shutil.copytree(parent/f'private/pools/{SUITE}',study/f'private/pools/{SUITE}')
    base.write(study/'private/cohort.json',cells,True)
    base.write(study/'private/warmup.json',warm,True)
    base.check_inputs(study/f'private/pools/{SUITE}',study/'private/cohort.json')
    shutil.copyfile(parent/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
    for item in base.read(study/'model_source_fingerprints.json')['files']:
        p=base.MODEL/item['file']
        if not p.is_file() or p.stat().st_size != item['bytes'] or base.sha(p) != item['sha256']:
            raise RuntimeError('model inventory mismatch')
    settings = dict(base.read(parent/'protocol.json')['suites'][SUITE], memory=.80)
    protocol = dict(schema='v31_failure_map_v1', source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        budgets_seconds=BUDGET,total_gpu_seconds=CAP,runtime=base.runtime_identity(),frozen_native=prior.native_contract(),
        suites={SUITE:settings},spec_sha256=base.sha(ROOT/SPEC),
        execution_fingerprints={rel:base.sha(study/'private/execution'/rel) for rel in files},
        input_fingerprints={str(p.relative_to(study)):base.sha(p) for folder in (study/f'private/pools/{SUITE}',study/'private')
            for p in (folder.rglob('*') if folder.name == SUITE else folder.glob('*.json')) if p.is_file()},
        model_inventory_sha256=base.sha(study/'model_source_fingerprints.json'),
        cohort=[dict(index=c['index'],prompt_tokens=prior.token_length(c),metadata_hash=key(c)) for c in cells],
        warmup_indices=[c['index'] for c in warm],screen_seeds=[1,42],confirmation_seeds=[43,44],
        budgets_tokens=list(KS),reference_arms=list(REFS),screen_requests=320,
        candidate_rule='same type both seeds, FULL and all-kept controls; largest qualifying K; metadata order max8',
        confirmation_rule='same type all four new-seed/process cells; FULL+PIECEWISE+all-kept; max4 for repair',
        development_only=True,older_campaigns_remain_paused=True)
    base.write(study/'protocol.json',protocol,True)
    shutil.copyfile(ROOT/SPEC,study/'spec.md')
    base.write(study/'receipts/source_fingerprints.json',dict(source_commit=protocol['source_commit'],execution_fingerprints=protocol['execution_fingerprints']),True)
    base.write(study.parent/'gpu_budget.json',dict(cap_s=CAP,allocations=BUDGET,reservations=[]),True)
    base.write(study/'state.json',dict(status='prepared',protocol_sha256=base.sha(study/'protocol.json')))
    print(json.dumps(dict(event='prepared',source=protocol['source_commit'],questions=len(cells),cap_s=CAP)),flush=True)


def verify(study):
    p=base.read(study/'protocol.json');prior.pause_unchanged()
    base.verify(study,p)
    if p['budgets_seconds'] != BUDGET or p['total_gpu_seconds'] != CAP:
        raise RuntimeError('budget drift')
    for rel in FILES:
        if base.sha(ROOT/rel) != p['execution_fingerprints'][rel]:
            raise RuntimeError('host execution source changed')
    if base.sha(study/'spec.md') != p['spec_sha256']:
        raise RuntimeError('spec changed')
    base.SUITES.update(p['suites'])
    return p


def validate_record(row, cell, arm, ordinal, family, pass_name, k, p, repair=False):
    plan=dict(phase='repair' if repair else 'screen',family=family,pass_name=pass_name)
    settings=dict(p['suites'][SUITE],mage_k=k)
    errors=prior.validate_row(row,cell,arm,ordinal,plan,settings)
    a=(row.get('receipts') or {}).get('adapter') or {}
    if pass_name == 'audit':
        for field in ('global_eligible_tiles','global_kept_tiles','native_local_skipped_tiles'):
            if field not in a:errors.append('missing audit '+field)
        if family in REFS and a.get('global_kept_tiles') != a.get('global_eligible_tiles'):
            errors.append('dense control skipped work')
    if family == 'shared':
        if row.get('mage_sticky') != 1.386 or row.get('mage_granularity') != 'qblock_max':
            errors.append('selector configuration')
        if a.get('order_errors'):
            errors.append('adapter order')
        if pass_name == 'audit' and (a.get('failure_map_budget_errors') != 0 or a.get('failure_map_budget_units',0)<=0):
            errors.append('actual fixed-K and protected tail validation')
    return errors


def score(study, leaf, rows, raw, cells, arms, family, pass_name, k, phase, repeat, p):
    pools=official.Pools(study/f'private/pools/{SUITE}',fields=('thinking',))
    final,_=official.load_final_response()
    gold=base.read(study/f'private/pools/{SUITE}'/(p['suites'][SUITE]['dataset']+'_gold.json'))
    errors=[];out=[]
    for i,(r,g,c,a) in enumerate(zip(rows,raw,cells,arms[1:])):
        errors+=validate_record(r,c,a,i,family,pass_name,k,p,phase=='repair')
        meta,_,bound=official.bind(g,pools)
        if not bound or any(r.get(f)!=g.get(f) for f in official.COMPARE_FIELDS):
            errors.append('public/private binding')
        if a['warmup']:continue
        row={f:r.get(f) for f in ('dataset','index','panel_seed','rng_seed','output_hash','prompt_tokens','budget',
            'denoise_forwards','canvases','output_tokens','finish_reason','wall_s','decode_s','prefill_s','cuda_graph_captures')}
        row.update(label=a['label'],policy=a['policy'],family=family,k=k if family=='shared' else None,
            phase=phase,pass_name=pass_name,engine_repeat=repeat,engine=str(leaf.relative_to(study)),
            score=float(official.lb_extract_answer(final(g['completion'],bool(meta['thinking'])))==gold[g['id']]),
            native_receipt=(r.get('receipts') or {}).get('adapter'),source_commit=p['source_commit'],
            protocol_sha256=base.sha(study/'protocol.json'))
        out.append(row)
    return sorted(set(errors)),out


def run_job(study,phase,family,k,cells,seeds,repeat=0,fixed=False,pass_name='clean'):
    p=verify(study);ledger_path=study.parent/'gpu_budget.json';ledger=base.read(ledger_path)
    seconds=available(phase,ledger['reservations'])
    if seconds<180:return None
    tag=f'{family}_k{k}_rep{repeat:02d}_{pass_name}'
    leaf=study/phase/tag/'attempt001'
    if leaf.exists():raise RuntimeError('preserve existing leaf; no automatic retry')
    prior.require_idle()
    if base.runtime_identity()!=p['runtime']:raise RuntimeError('runtime changed')
    rows,arms=schedule(cells,base.read(study/'private/warmup.json'),family,seeds,repeat,fixed,phase=='repair')
    base.write(leaf/'private/cells.json',rows,True);base.write(leaf/'private/arms.json',arms,True)
    config=dict(phase=phase,family=family,k=k,seeds=seeds,repeat=repeat,fixed=fixed,pass_name=pass_name,
        source_commit=p['source_commit'],protocol_sha256=base.sha(study/'protocol.json'),
        cells_sha256=base.sha(leaf/'private/cells.json'),arms_sha256=base.sha(leaf/'private/arms.json'),measured_count=len(rows)-3)
    base.write(leaf/'config.json',config,True)
    execution=study/'private/execution';clean=pass_name=='clean'
    env=base.env_for(study,SUITE,'current_v31_control')
    for name in list(env):
        if name.startswith(('REFRESH_','HEADROOM_','FAILURE_MAP_')):env.pop(name)
    env.update(MAGE_K=str(k),STUDY_ARM=tag,PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',REPEATS='1',
        FROZEN_SOURCE_SHA256=base.sha(study/'receipts/source_fingerprints.json'),
        REFRESH_SOURCE_COMMIT=p['source_commit'],REFRESH_PROTOCOL_SHA256=base.sha(study/'protocol.json'),
        REFRESH_SCHEDULE=str(leaf/'private/arms.json'),REFRESH_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'),
        REFRESH_AUDIT='1' if phase=='repair' else '0',HEADROOM_PROFILE='0',
        VALUE_AUDIT=str(int(not clean)),VALUE_CLEAN_TIMING=str(int(clean)),VALUE_BOUNDARY_TIMING=str(int(clean)),
        FAILURE_MAP_REPAIR=str(int(phase=='repair')))
    worker='v31_failure_map_repair.py'
    cmd=[base.PYTHON,str(execution/'scripts'/worker),str(base.MODEL),str(study/f'private/pools/{SUITE}'),
        str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),
        'mage' if family=='shared' else 'allkept' if family=='allkept_fa4' else 'dense',
        'default' if family=='dense_full_fix51994' else 'PIECEWISE']
    reservation=dict(phase=phase,leaf=str(leaf.relative_to(study)),reserved_s=seconds,elapsed_s=None)
    ledger['reservations'].append(reservation);base.write(ledger_path,ledger)
    base.write(study/'state.json',dict(status='running',phase=phase,active_leaf=reservation['leaf']))
    print(json.dumps(dict(event='launch',**config,deadline_s=seconds)),flush=True)
    start=time.monotonic();timed_out=False
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.write(leaf/'private/process.json',dict(pid=proc.pid,command=cmd),True)
        try:
            while proc.poll() is None:
                if time.monotonic()-start>=seconds-5:
                    timed_out=True;prior.stop_owned_process(proc);break
                time.sleep(1)
        finally:
            if proc.poll() is None:prior.stop_owned_process(proc)
    reservation['elapsed_s']=time.monotonic()-start
    reservation['returncode']=proc.returncode;reservation['timed_out']=timed_out
    base.write(ledger_path,ledger)
    public=prior.complete_jsonl(leaf/'records.jsonl',allow_truncated=timed_out)
    raw=prior.complete_jsonl(leaf/'private/generation.private.jsonl',allow_truncated=timed_out)
    errors,scored=score(study,leaf,public,raw,rows,arms,family,pass_name,k,phase,repeat,p)
    full=len(public)==len(raw)==len(rows)
    if proc.returncode or not full:errors.append('worker or coverage incomplete')
    base.write(leaf/'scored.json',scored,True)
    receipt=dict(status='complete' if not errors else 'budget_limited_partial' if timed_out else 'failed',
        errors=errors,returncode=proc.returncode,timed_out=timed_out,records=len(public),private_records=len(raw),
        measured_records=len(scored),planned_measured=len(rows)-3,gpu_reservation_s=reservation['elapsed_s'],
        source_commit=p['source_commit'],protocol_sha256=base.sha(study/'protocol.json'))
    base.write(leaf/'receipt.json',receipt,True)
    if not errors:
        files=('records.jsonl','private/generation.private.jsonl','private/cells.json','private/arms.json','config.json','receipt.json','scored.json')
        base.write(leaf/'attempt_complete.json',dict(receipt,files_sha256={f:base.sha(leaf/f) for f in files}),True)
    print(json.dumps(dict(event='finished',leaf=reservation['leaf'],**receipt)),flush=True)
    if errors:
        raise RuntimeError(f"{receipt['status']}: {reservation['leaf']}: {errors}")
    return scored


def case_type(v, refs):
    """Conservative per-case gate; not a population estimator."""
    if len(v)!=2 or not refs or v[0]['score']!=v[1]['score']:
        return None
    if v[0]['score']==0 and all(r['score']==1 for r in refs):return 'accuracy'
    if not all(r['score']==1 and r['finish_reason'] not in official.CAPPED for r in [*v,*refs]):return None
    nc=lambda r:r['denoise_forwards']/r['canvases']
    lo=min(map(nc,v));ref=max(map(nc,refs));drift=abs(nc(v[0])-nc(v[1]))
    if lo>=1.1*ref and lo-ref>=1 and lo-ref>2*drift:return 'convergence'
    return None


def cases(rows,k,seeds,repeats,reference_labels):
    groups=defaultdict(list)
    for r in rows:groups[(r['index'],r['panel_seed'],r['engine_repeat'])].append(r)
    indices=sorted({r['index'] for r in rows})
    result=[]
    for index in indices:
        kinds=[]
        for rep in repeats:
            for seed in seeds:
                g=groups[(index,seed,rep)]
                v=[r for r in g if r['family']=='shared' and r['k']==k and r['label'] in ('control_before','control_after')]
                refs=[r for r in g if r['label'] in reference_labels]
                kind=case_type(v,refs) if len(refs)==len(reference_labels) and {r['label'] for r in refs}==set(reference_labels) else None
                kinds.append(kind)
        if kinds and kinds[0] is not None and len(set(kinds))==1:
            result.append(dict(index=index,k=k,failure_type=kinds[0]))
    return result


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['family'],r['k'],r['label'])].append(r)
    out=[]
    for (family,k,label),g in groups.items():
        n=np.array([r['denoise_forwards'] for r in g]);c=sum(r['canvases'] for r in g)
        out.append(dict(family=family,k=k,label=label,records=len(g),questions=len({r['index'] for r in g}),
            accuracy=float(np.mean([r['score'] for r in g])),N=float(n.mean()),N_per_C=float(n.sum()/c),
            C=c/len(g),T=float(np.mean([r['output_tokens'] for r in g])),
            W=float(np.mean([r['wall_s'] for r in g])),S=float(np.mean([r['decode_s'] for r in g])),
            S_per_N=sum(r['decode_s'] for r in g)/int(n.sum()),median_N=float(np.median(n)),
            p90_N=float(np.quantile(n,.9)),p95_N=float(np.quantile(n,.95)),max_N=int(n.max()),
            caps=sum(r['finish_reason'] in official.CAPPED for r in g)))
    return out


def run(study,stop_after=None):
    p=verify(study)
    lock=(ROOT/'results/.v31_refresh_h100.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cohort=base.read(study/'private/cohort.json');warm=base.read(study/'private/warmup.json')
    collected={};terminal='complete_bounded_diagnostic'
    try:
        for phase in ('qualification','screen','confirmation','repair'):
            if (study/phase/'phase_complete.json').exists():
                collected[phase]=load_phase(study,phase)
                continue
            if phase=='qualification':
                pool=warm;seeds=[42];repeats=[0]
                jobs=[('shared',k,k==8192) for k in KS]+[(f,8192,False) for f in REFS]
            elif phase=='screen':
                pool=cohort;seeds=[1,42];repeats=[0]
                jobs=[('dense_full_fix51994',8192,False),('shared',8192,True),('dense_piecewise',8192,False),
                      ('shared',4096,False),('allkept_fa4',8192,False),('shared',2048,False)]
            elif phase=='confirmation':
                candidates=[]
                for k in KS:
                    found=cases(collected['screen'],k,[1,42],[0],('dense_full_fix51994','allkept_fa4'))
                    if found:
                        by_id={c['index']:c for c in cohort}
                        candidates=sorted(found,key=lambda x:key(by_id[x['index']]))[:8];break
                base.write(study/'screen/candidate_selection.json',dict(candidates=candidates,rule=p['candidate_rule']),True)
                if not candidates:
                    terminal='complete_no_screen_candidates';break
                pool=[c for c in cohort if c['index'] in {x['index'] for x in candidates}]
                k=candidates[0]['k'];seeds=[43,44];repeats=[0,1]
                jobs=[('shared',k,True)]+[(f,k,False) for f in REFS]
            else:
                selected=base.read(study/'screen/candidate_selection.json')['candidates'];k=selected[0]['k']
                original={x['index']:x['failure_type'] for x in selected}
                confirmed=[x for x in cases(collected['confirmation'],k,[43,44],[0,1],REFS)
                           if x['failure_type']==original[x['index']]]
                by_id={c['index']:c for c in cohort}
                confirmed=sorted(confirmed,key=lambda x:key(by_id[x['index']]))
                base.write(study/'confirmation/confirmed_failures.json',dict(confirmed=confirmed,repair_cases=confirmed[:4],rule=p['confirmation_rule']),True)
                if not confirmed:
                    terminal='complete_no_confirmed_failures';break
                pool=[by_id[x['index']] for x in confirmed[:4]];seeds=[43,44];repeats=[0,1]
                jobs=[('shared',k,False)]
            values=[]
            for rep in repeats:
                for family,k,fixed in (jobs[::-1] if rep%2 else jobs):
                    data=run_job(study,phase,family,k,pool,seeds,rep,fixed,'audit' if phase in ('qualification','repair') else 'clean')
                    if data is None:
                        terminal='budget_limited_partial';break
                    values.extend(data)
                if terminal=='budget_limited_partial':break
            collected[phase]=values
            base.write(study/phase/'aggregate.json',summarize(values),True)
            if terminal=='budget_limited_partial':break
            markers={str(x.relative_to(study)):base.sha(x) for x in (study/phase).glob('*/attempt*/attempt_complete.json')}
            base.write(study/phase/'phase_complete.json',dict(status='complete',records=len(values),engines=markers,
                protocol_sha256=base.sha(study/'protocol.json'),aggregate_sha256=base.sha(study/phase/'aggregate.json')),True)
            if stop_after==phase:
                terminal='phase_stop';break
    except Exception as exc:
        terminal='failed';base.write(study/'failure.json',dict(error=str(exc),phase=phase),True)
        raise
    finally:
        ledger=base.read(study.parent/'gpu_budget.json')
        used=sum(r['reserved_s'] if r.get('elapsed_s') is None else r['elapsed_s'] for r in ledger['reservations'])
        base.write(study/'state.json',dict(status=terminal,phase=phase,gpu_reservation_s=used,cap_s=CAP,
            completed_phases=[x for x in collected if (study/x/'phase_complete.json').exists()],
            protocol_sha256=base.sha(study/'protocol.json'),older_campaigns_remain_paused=True))
        print(json.dumps(dict(event='terminal',status=terminal,gpu_reservation_s=used)),flush=True)


def load_phase(study,phase):
    done=base.read(study/phase/'phase_complete.json');out=[]
    if done['protocol_sha256']!=base.sha(study/'protocol.json'):raise RuntimeError('phase protocol drift')
    if done['aggregate_sha256']!=base.sha(study/phase/'aggregate.json'):raise RuntimeError('aggregate drift')
    for name,h in done['engines'].items():
        marker=study/name
        if base.sha(marker)!=h:raise RuntimeError('phase marker drift')
        m=base.read(marker)
        for rel,digest in m['files_sha256'].items():
            if base.sha(marker.parent/rel)!=digest:raise RuntimeError('completed artifact changed')
        out.extend(base.read(marker.parent/'scored.json'))
    if len(out)!=done['records']:raise RuntimeError('phase coverage changed')
    return out


def repair_qualification(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['index'],r['panel_seed'],r['engine_repeat'])].append(r)
    outcomes=[]
    for (index,seed,repeat),g in sorted(groups.items()):
        by={r['label']:r for r in g};errors=[]
        expected={'control_before','control_after',*prior.FACTORIAL}
        if len(g)!=6 or set(by)!=expected:
            errors.append('incomplete six-arm group')
        else:
            diagnostics=[r['native_receipt']['refresh_diagnostic'] for r in g]
            if not all(d['exposed'] and d['complete_scope'] for d in diagnostics):errors.append('unexposed or incomplete scope')
            signatures=[d.get('prefix_signature') for d in diagnostics]
            if None in signatures or len(set(signatures))!=1:errors.append('unmatched recorded native prefix')
            outcome=lambda r:tuple(r[f] for f in ('output_hash','denoise_forwards','canvases','output_tokens','score','finish_reason'))
            if len({outcome(by[x]) for x in ('control_before','control_after','sparse_hold')})!=1:
                errors.append('V31 bookend or sham outcome drift')
            if any(d.get('added_model_forwards')!=0 or d.get('added_sampler_calls')!=0 for d in diagnostics):
                errors.append('added native calls')
            if any(r['native_receipt'].get('failure_map_budget_errors')!=0 for r in g):errors.append('budget error')
        effects={}
        if not errors:
            control=by['sparse_hold'];nc=lambda r:r['denoise_forwards']/r['canvases']
            for label in prior.FACTORIAL:
                r=by[label]
                effects[label]=dict(score_delta=r['score']-control['score'],N_delta=r['denoise_forwards']-control['denoise_forwards'],
                    NC_delta=nc(r)-nc(control),T_delta=r['output_tokens']-control['output_tokens'],
                    improved_without_score_loss=r['score']>=control['score'] and
                    (r['score']>control['score'] or (nc(r)<nc(control) and r['finish_reason'] not in official.CAPPED)))
        outcomes.append(dict(index=index,seed=seed,repeat=repeat,qualified=not errors,errors=errors,effects=effects))
    leads=[]
    for index in sorted({x['index'] for x in outcomes}):
        case=[x for x in outcomes if x['index']==index]
        if len(case)==4 and all(x['qualified'] for x in case):
            for label in prior.FACTORIAL:
                if all(x['effects'][label]['improved_without_score_loss'] for x in case):
                    leads.append(dict(index=index,policy=label,interpretation='replicated selected-case mechanistic lead, not deployable algorithm'))
    return dict(groups=outcomes,qualified_groups=sum(x['qualified'] for x in outcomes),mechanistic_leads=leads)


def paired_intervals(rows):
    """Paired question-cluster resampling; bookends averaged within cell."""
    groups=defaultdict(list)
    for r in rows:groups[(r['family'],r['k'],r['label'])].append(r)
    ref={(r['index'],r['panel_seed'],r['engine_repeat']):r for r in rows if r['label']=='dense_full_fix51994'}
    output=[]
    for k in KS:
        v=[r for r in rows if r['family']=='shared' and r['k']==k and r['label'] in ('control_before','control_after')]
        by=defaultdict(list)
        for r in v:by[r['index']].append(r)
        values=[]
        for index,g in sorted(by.items()):
            cells=defaultdict(list)
            for r in g:cells[(r['index'],r['panel_seed'],r['engine_repeat'])].append(r)
            sums=np.zeros(10)
            for identity,bookends in cells.items():
                if identity not in ref or len(bookends)!=2:continue
                dense=ref[identity]
                fields=('score','denoise_forwards','canvases','wall_s','output_tokens')
                sums[:5]+=np.mean([[r[f] for f in fields] for r in bookends],axis=0)
                sums[5:]+=[dense[f] for f in fields]
            if cells:values.append(sums/len(cells))
        if not values:continue
        a=np.array(values);rng=np.random.default_rng(1729)
        boot=a[rng.integers(len(a),size=(20000,len(a)))].mean(1)
        point=a.mean(0)
        metrics={'accuracy_delta_pp':lambda x:100*(x[...,0]-x[...,5]),
                 'N_ratio':lambda x:x[...,1]/x[...,6],
                 'NC_ratio':lambda x:(x[...,1]/x[...,2])/(x[...,6]/x[...,7]),
                 'W_ratio':lambda x:x[...,3]/x[...,8], 'T_ratio':lambda x:x[...,4]/x[...,9]}
        estimates={name:dict(estimate=float(fn(point)),ci95=np.quantile(fn(boot),[.025,.975]).tolist()) for name,fn in metrics.items()}
        output.append(dict(k=k,questions=len(values),estimates=estimates,
            population_flag_accuracy=estimates['accuracy_delta_pp']['estimate']<=-3,
            population_flag_convergence=estimates['NC_ratio']['estimate']>=1.10))
    return output


def report(study):
    p=verify(study);state=base.read(study/'state.json');destination=study/'analysis/attempt001'
    if destination.exists():raise RuntimeError('preserve prior analysis')
    phases={phase:load_phase(study,phase) for phase in state['completed_phases']}
    summaries={phase:summarize(rows) for phase,rows in phases.items()}
    comparisons={phase:paired_intervals(rows) for phase,rows in phases.items() if phase in ('screen','confirmation')}
    repair=repair_qualification(phases.get('repair',[]))
    drift=[]
    for phase,rows in phases.items():
        grouped=defaultdict(dict)
        for r in rows:
            if r['label'] in ('control_before','control_after'):
                grouped[(r['index'],r['panel_seed'],r['engine_repeat'],r['k'])][r['label']]=r
        fields=('output_hash','score','denoise_forwards','canvases','output_tokens')
        counts={f:0 for f in fields};total=0
        for g in grouped.values():
            if len(g)==2:
                total+=1
                for f in fields:counts[f]+=g['control_before'][f]!=g['control_after'][f]
        drift.append(dict(phase=phase,pairs=total,differences=counts))
    candidates=base.read(study/'screen/candidate_selection.json') if (study/'screen/candidate_selection.json').exists() else None
    confirmed=base.read(study/'confirmation/confirmed_failures.json') if (study/'confirmation/confirmed_failures.json').exists() else None
    value=dict(state=state,summaries=summaries,paired_question_cluster_intervals=comparisons,bookend_drift=drift,
        candidate_selection=candidates,confirmed_failures=confirmed,repair=repair,
        source_commit=p['source_commit'],protocol_sha256=base.sha(study/'protocol.json'),
        uncertainty='20000 question-cluster bootstrap draws, seed1729; selected confirmation is not population inference')
    base.write(destination/'analysis.json',value,True)
    lines=['# Bounded V31 failure-map diagnostic','',f"Status: {state['status']}. GPU reservation {state['gpu_reservation_s']:.1f}/14400 seconds.",'',
        f"Execution source `{p['source_commit']}`; protocol `{base.sha(study/'protocol.json')}`.",'',
        'Development LongBench questions; native sampler and manifest output budgets preserved. Audit and clean outcomes are separate. N excludes per-canvas commit calls and C is derived by the inherited recorder from output length; N/C does not align causal states. No result here establishes held-out generalization or noninferiority.','']
    for phase,summary in summaries.items():
        lines += [f'## {phase}','','| Arm | K | Requests | Accuracy | N | N/C | T | W (s) |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            lines.append(f"| {row['label']} | {row['k'] or 'dense'} | {row['records']} | {100*row['accuracy']:.2f}% | {row['N']:.3f} | {row['N_per_C']:.3f} | {row['T']:.1f} | {row['W']:.4f} |")
        lines += ['']
    lines += ['## Gates','',f"Screen candidates: {len(candidates['candidates']) if candidates else 'not assessed'}. Confirmed failures: {len(confirmed['confirmed']) if confirmed else 'not assessed'}.",
        f"Qualified repair groups: {repair['qualified_groups']}/{len(repair['groups'])}. Replicated mechanistic leads: {len(repair['mechanistic_leads'])}.",'',
        'See analysis.json for all paired intervals, bookend drift, selected-case reversals, and repair qualification errors. Repair and qualification W are instrumented and are not speed measurements. Older campaigns remain paused.','']
    (destination/'summary.md').write_text('\n'.join(lines))
    complete=state['status'].startswith('complete_')
    marker=dict(status='complete' if complete else 'partial',source_commit=p['source_commit'],protocol_sha256=base.sha(study/'protocol.json'),
        files_sha256={str(f.relative_to(study)):base.sha(f) for f in (destination/'analysis.json',destination/'summary.md',study/'state.json')})
    base.write(destination/'verified_complete.json' if complete else destination/'partial_analysis.json',marker,True)
    if complete:base.write(study/'final_complete.json',marker,True)
    print(json.dumps(dict(event='reported',status=state['status'],qualified_repair_groups=repair['qualified_groups'])),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','run','report'))
    ap.add_argument('--study',type=Path,default=DEFAULT);ap.add_argument('--stop-after',choices=('qualification','screen','confirmation','repair'))
    a=ap.parse_args()
    if a.action=='prepare':prepare(a.study)
    elif a.action=='run':run(a.study,a.stop_after)
    else:report(a.study)


if __name__=='__main__':main()
