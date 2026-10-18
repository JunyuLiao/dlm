"""Frozen 45-minute numerical and attribution campaign; no selector changes."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from scripts import v31_refresh_external_profile as ext
from scripts import v31_refresh_diagnostic_study as study
from scripts import v31_v2_local_threshold_sweep as base

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'results/v31_refresh_repeatability_20261018'
DEFAULT=CAMPAIGN/'attempt001'
SPEC='docs/REFRESH_REPEATABILITY_PLAN_20261018.md'
FILES=(SPEC,'scripts/v31_refresh_followup.py','scripts/v31_refresh_probe_worker.py',
       'scripts/v31_refresh_operator_replay.py','tests/test_v31_refresh_followup.py')
BUDGET=dict(capture=900.,replay=600.,attribution=900.,recovery=300.)
JOBS=(('capture','capture',900.),('replay001','replay',300.),('replay002','replay',300.),
      ('clean','attribution',450.),('nvtx','attribution',450.))


def available(phase,reservations):
    counts=defaultdict(float)
    for r in reservations:counts[r['phase']]+=r['reserved_s'] if r.get('elapsed_s') is None else r['elapsed_s']
    return max(0.,min(BUDGET[phase]-counts[phase],2700.-sum(counts.values())))


def prepare(output):
    if output.exists():raise RuntimeError('preserve previous attempt')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    changed=set(subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).splitlines())
    if not set(FILES)<=tracked or set(FILES)&changed:raise RuntimeError('commit frozen protocol, implementation and tests first')
    original,plan,leaf=ext.verify_original()
    output.mkdir(parents=True);execution=output/'private/execution'
    shutil.copytree(ext.ORIGINAL/'private/execution',execution)
    for rel in FILES:
        dest=execution/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    originals=ext.inventory(ext.ORIGINAL/'private/cache')
    for job,phase,limit in JOBS:
        dest=output/job;dest.mkdir();shutil.copytree(ext.ORIGINAL/'private/cache',dest/'private/cache')
        if ext.inventory(dest/'private/cache')!=originals:raise RuntimeError('cache copy mismatch')
    cells=base.read(leaf/'private/cells.json');arms=base.read(leaf/'private/arms.json')
    for name in ('clean','nvtx'):
        base.write(output/name/'private/cells.json',cells,True);base.write(output/name/'private/arms.json',arms,True)
    source_cells=base.read(study.LB_PARENT/'private/schedules/longbench_v2/cells.json')
    chosen=[]
    for index,seed in ((329,42),(125,1),(125,1),(329,42),(329,42),(254,42),(254,42)):
        cell=dict(next(c for c in source_cells if c['index']==index));cell['seed']=seed;chosen.append(cell)
    capture_arms=[dict(policy='control',label='implicit',warmup=True)]+[
        dict(policy='control',label='capture_%02d'%i,warmup=i==0) for i in range(len(chosen))]
    base.write(output/'capture/private/cells.json',chosen,True);base.write(output/'capture/private/arms.json',capture_arms,True)
    protocol=dict(schema='refresh_repeatability_v1',status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        native_source_commit=original['source_commit'],original_protocol_sha256=base.sha(ext.ORIGINAL/'protocol.json'),
        native_fingerprints=original['execution_fingerprints'],runner_fingerprints={r:base.sha(ROOT/r) for r in FILES},
        execution_fingerprints={str(p.relative_to(execution)):base.sha(p) for p in execution.rglob('*') if p.is_file()},
        original_cache_manifest=originals,runtime=original['runtime'],budgets_seconds=BUDGET,total_gpu_seconds=2700,
        jobs=[dict(name=n,phase=p,maximum_s=s) for n,p,s in JOBS],
        schedule_sha256={str(p.relative_to(output)):base.sha(p) for n in ('capture','clean','nvtx') for p in (output/n/'private').glob('*.json')},
        nsys_sha256=base.sha(ext.NSYS.resolve()),nsys_version=subprocess.check_output([str(ext.NSYS),'--version'],text=True).strip())
    import vllm
    runtime_root=Path(vllm.__file__).parent
    protocol['additional_runtime_fingerprints']={rel:base.sha(runtime_root/rel) for rel in
        ('vllm_flash_attn/cute/interface.py','vllm_flash_attn/cute/cute_dsl_utils.py',
         'vllm_flash_attn/cute/cache_utils.py','vllm_flash_attn/cute/mask.py','vllm_flash_attn/cute/utils.py')}
    base.write(output/'protocol.json',protocol,True)
    if not (CAMPAIGN/'gpu_budget.json').exists():base.write(CAMPAIGN/'gpu_budget.json',dict(reservations=[]),True)
    print(json.dumps(dict(status='prepared',source=protocol['source_commit'],protocol_sha256=base.sha(output/'protocol.json'))),flush=True)


def verify(output):
    p=base.read(output/'protocol.json');original,plan,leaf=ext.verify_original()
    assert p['original_protocol_sha256']==base.sha(ext.ORIGINAL/'protocol.json')
    assert p['runtime']==original['runtime']
    for rel,h in p['runner_fingerprints'].items():assert base.sha(ROOT/rel)==h,rel
    for rel,h in p['execution_fingerprints'].items():assert base.sha(output/'private/execution'/rel)==h,rel
    for rel,h in p['schedule_sha256'].items():assert base.sha(output/rel)==h,rel
    assert ext.inventory(ext.ORIGINAL/'private/cache')==p['original_cache_manifest']
    assert base.sha(ext.NSYS.resolve())==p['nsys_sha256']
    import vllm
    for rel,h in p['additional_runtime_fingerprints'].items():assert base.sha(Path(vllm.__file__).parent/rel)==h,rel
    return p,original,leaf


def command(output,name,original,seconds):
    leaf=output/name;execution=output/'private/execution';base.SUITES.update(original['suites'])
    env=base.env_for(leaf,'longbench_v2','current_v31_control')
    for key in list(env):
        if key.startswith(('REFRESH_','HEADROOM_','PROBE_')):env.pop(key)
    env.update(PYTHONPATH=f'{execution}/src:{execution}',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
        STUDY_ARM='refresh_shared',FROZEN_SOURCE_SHA256=base.sha(ext.ORIGINAL/'receipts/source_fingerprints.json'),
        REFRESH_SOURCE_COMMIT=original['source_commit'],REFRESH_PROTOCOL_SHA256=base.sha(ext.ORIGINAL/'protocol.json'),
        REFRESH_SCHEDULE=str(leaf/'private/arms.json'),REFRESH_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'),
        REFRESH_AUDIT='0',HEADROOM_PROFILE='0',VALUE_AUDIT='0',VALUE_CLEAN_TIMING='1',VALUE_BOUNDARY_TIMING='1',
        PROBE_MODE=name,PROBE_OUTPUT=str(leaf),PROBE_ORIGINAL_WORKER=str(execution/'scripts/v31_refresh_diagnostic_worker.py'))
    if name.startswith('replay'):
        return [base.PYTHON,str(execution/'scripts/v31_refresh_operator_replay.py'),'--capture',str(output/'capture/private'),'--output',str(leaf/'analysis')],env
    script='v31_refresh_diagnostic_worker.py' if name=='clean' else 'v31_refresh_probe_worker.py'
    cmd=[base.PYTHON,str(execution/'scripts'/script),str(base.MODEL),str(ext.ORIGINAL/'private/pools/longbench_v2'),
         str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
    if name=='nvtx':
        flags=[x for x in ext.PROFILE_FLAGS if not x.startswith('--trace=')]
        cmd=[str(ext.NSYS),'profile','--trace=cuda,nvtx',*flags,'--wait=all','--kill=sigterm',
             '--duration='+str(max(1,int(seconds-50))),'--output='+str(leaf/'private/trace'),*cmd]
    return cmd,env


def validate_native(output,name,original,old_leaf):
    leaf=output/name;rows=study.complete_jsonl(leaf/'records.jsonl',allow_truncated=True)
    raw=study.complete_jsonl(leaf/'private/generation.private.jsonl',allow_truncated=True)
    cells=base.read(leaf/'private/cells.json');arms=base.read(leaf/'private/arms.json')
    errors=[]
    if len(rows)!=len(cells) or len(raw)!=len(cells):errors.append('incomplete public/private coverage')
    plan=dict(phase='headroom',family='shared',suite='longbench_v2',pass_name='clean')
    errors+=sorted({e for i,(r,c,a) in enumerate(zip(rows,cells,arms[1:])) for e in study.validate_row(r,c,a,i,plan,original['suites']['longbench_v2'])})
    ext.check_private_bind(rows,raw,ext.official.Pools(ext.ORIGINAL/'private/pools/longbench_v2',fields=('thinking',)))
    parity=ext.parity_check(rows,base.jsonl(old_leaf/'records.jsonl'),arms) if name!='capture' else None
    if name=='nvtx':
        parity_new=ext.parity_check(rows,base.jsonl(output/'clean/records.jsonl'),arms)
        base.write(leaf/'parity_new_clean.json',parity_new,True)
    if parity is not None:base.write(leaf/'parity_original.json',parity,True)
    if name=='capture':
        probe=base.read(leaf/'probe.json')
        if len(probe['captures'])!=6:errors.append('six snapshots not exposed')
    report=dict(status='complete' if not errors else 'failed_or_partial',records=len(rows),errors=errors,
                parity=None if parity is None else parity['status'])
    base.write(leaf/'validation.json',report,True);return report


def run(output,only=None):
    with (ROOT/'results/.v31_refresh_h100.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p,original,old_leaf=verify(output)
        for name,phase,maximum in JOBS:
            if only and name!=only:continue
            leaf=output/name
            if (leaf/'receipt.json').exists():continue
            if (leaf/'private/worker.log').exists():raise RuntimeError('preserve incomplete job')
            if name.startswith('replay') and len(list((output/'capture/private').glob('snapshot*.pt')))!=6:raise RuntimeError('capture gate missing')
            if name=='nvtx' and base.read(output/'clean/validation.json')['status']!='complete':raise RuntimeError('clean control incomplete')
            ledger=base.read(CAMPAIGN/'gpu_budget.json')
            if any(r.get('elapsed_s') is None for r in ledger['reservations']):raise RuntimeError('unresolved reservation')
            seconds=min(maximum,available(phase,ledger['reservations']))
            if seconds<120:raise RuntimeError('phase budget exhausted')
            study.require_idle();cmd,env=command(output,name,original,seconds)
            reservation=dict(attempt=output.name,job=name,phase=phase,reserved_s=seconds,elapsed_s=None,status='started')
            ledger['reservations'].append(reservation);base.write(CAMPAIGN/'gpu_budget.json',ledger)
            start=time.monotonic();error=None;process=None;validation=None;export=None
            print(json.dumps(dict(event='launch',job=name,limit_s=seconds)),flush=True)
            try:
                process=ext.bounded_process(cmd,cwd=output/'private/execution',env=env,log_path=leaf/'private/worker.log',deadline=start+seconds-20)
                if process['returncode']==0:
                    if not name.startswith('replay'):validation=validate_native(output,name,original,old_leaf)
                    else:validation=dict(status='complete' if (leaf/'analysis/summary.json').exists() else 'incomplete')
                if name=='nvtx' and process['returncode']==0:
                    export=ext.bounded_process([str(ext.NSYS),'export','--type=sqlite','--force-overwrite=false',
                        '--output='+str(leaf/'private/trace.sqlite'),str(leaf/'private/trace.nsys-rep')],cwd=leaf,env=env,
                        log_path=leaf/'private/export.log',deadline=start+seconds)
            except BaseException as exc:error=type(exc).__name__+': '+str(exc)
            finally:
                reservation.update(elapsed_s=time.monotonic()-start,status='finished' if error is None else 'failed')
                base.write(CAMPAIGN/'gpu_budget.json',ledger)
            receipt=dict(status='complete_diagnostic' if not error and process and process['returncode']==0 and validation and validation['status']=='complete' else 'failed_or_partial',
                error=error,process=process,validation=validation,export=export,reservation_seconds=reservation['elapsed_s'],
                protocol_sha256=base.sha(output/'protocol.json'),source_commit=p['source_commit'],native_source_commit=p['native_source_commit'])
            base.write(leaf/'receipt.json',receipt,True);print(json.dumps(dict(event='finished',job=name,**receipt)),flush=True)
            if receipt['status']!='complete_diagnostic':break


def union(intervals):
    intervals=sorted(intervals);total=0;last=None
    for a,b in intervals:
        if b<a:raise ValueError('negative interval')
        if last is None or a>last[1]:
            if last:total+=last[1]-last[0]
            last=[a,b]
        else:last[1]=max(last[1],b)
    return total+(last[1]-last[0] if last else 0)


def attribute(ranges,apis,activities,metadata):
    """Launch-correlation attribution. Each activity belongs to one deepest range."""
    import bisect
    by_thread=defaultdict(list)
    for r in ranges:by_thread[r['thread']].append(r)
    for values in by_thread.values():values.sort(key=lambda r:r['start'])
    starts={t:[r['start'] for r in v] for t,v in by_thread.items()}
    linked={};ambiguous=0
    for api in apis:
        values=by_thread.get(api['thread'],[]);end=bisect.bisect_right(starts.get(api['thread'],[]),api['start'])
        # Scopes are shallow and nonoverlapping except for ancestry. Search back
        # through candidates without assuming all CUDA APIs occur inside a range.
        candidates=[r for r in values[max(0,end-16):end] if r['start']<=api['start'] and api['end']<=r['end']]
        if candidates:
            candidates.sort(key=lambda r:(r['end']-r['start'],r['id']))
            if len(candidates)>1 and candidates[0]['end']-candidates[0]['start']==candidates[1]['end']-candidates[1]['start']:ambiguous+=1
            linked[api['correlation']]=candidates[0]['id']
    stats=defaultdict(lambda:dict(cpu_ns=0,gpu_sum_ns=0,gpu_intervals=[],activities=0,calls=0))
    for r in ranges:
        m=metadata[r['id']];key=(m['request'],m['label'])
        stats[key]['cpu_ns']+=r['end']-r['start'];stats[key]['calls']+=1
    unlinked=0;per_range=Counter()
    for a in activities:
        ident=linked.get(a['correlation'])
        if ident is None:unlinked+=1;continue
        visited=set()
        while ident is not None:
            if ident in visited:raise ValueError('cyclic scope ancestry')
            visited.add(ident);m=metadata[ident];s=stats[(m['request'],m['label'])]
            per_range[ident]+=1
            s['gpu_sum_ns']+=a['end']-a['start'];s['gpu_intervals'].append((a['start'],a['end']));s['activities']+=1
            ident=m['parent']
    rows=[]
    for (request,label),s in stats.items():
        intervals=s.pop('gpu_intervals');s['gpu_union_ns']=union(intervals)
        rows.append(dict(request=request,label=label,**s))
    return dict(rows=rows,unlinked_activities=unlinked,ambiguous_launches=ambiguous,
                per_range_activity_counts=dict(per_range),
                interpretation='inclusive ancestry; never add child to parent; unlinked model work retained')


def analyze(output):
    leaf=output/'nvtx';rows=base.jsonl(leaf/'records.jsonl');arms=base.read(leaf/'private/arms.json')
    basic=ext.sqlite_summary(leaf/'private/trace.sqlite',rows,arms)
    db=sqlite3.connect((leaf/'private/trace.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    names=dict(db.execute('SELECT id,value FROM StringIds'))
    cols=[r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')]
    nvtx=[dict(zip(cols,r)) for r in db.execute('SELECT * FROM NVTX_EVENTS')]
    ranges=[]
    for r in nvtx:
        label=r.get('text') or names.get(r.get('textId'),'')
        if str(label).startswith('refresh_probe:') and r.get('end') is not None:
            ranges.append(dict(id=int(label.split(':')[1]),start=r['start'],end=r['end'],thread=r['globalTid']))
    metadata={r['id']:r for r in base.read(leaf/'probe.json')['scopes']}
    assert len(ranges)==len(metadata) and {r['id'] for r in ranges}==set(metadata),'NVTX scope coverage mismatch'
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    apis=[]
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:apis += [dict(start=a,end=b,thread=t,correlation=c) for a,b,t,c in db.execute(f'SELECT start,end,globalTid,correlationId FROM {table}')]
    acts=[]
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:acts += [dict(start=a,end=b,correlation=c) for a,b,c in db.execute(f'SELECT start,end,correlationId FROM {table}')]
    report=attribute(ranges,apis,acts,metadata);db.close()
    counts=[]
    for index,row in enumerate(rows,1):
        c=Counter(m['label'] for m in metadata.values() if m['request']==index)
        adapter=row['receipts']['adapter'];errors=[]
        for label,key in (('initial','mage_selections'),('refresh','mage_reselections'),('held','mage_reused_calls'),('carried','mage_carried_calls'),('bootstrap_dense','mage_warm_dense_calls')):
            if c['denoise/global_'+label]!=adapter.get(key,0):errors.append(label)
        attention_count=sum(v for label,v in c.items() if '/global_' in label or label.endswith('/local_attention'))
        if attention_count!=30*row['engine_steps']:errors.append('total native attention scopes')
        for phase,steps in (('prefill',row['prefill_steps']),('commit',row['canvases'])):
            if c[phase+'/global_dense']!=5*steps or c[phase+'/local_attention']!=25*steps:errors.append(phase+' scope counts')
        if c['denoise/local_attention']!=25*row['denoise_forwards']:errors.append('LOCAL denoise count')
        counts.append(dict(request=index,counts=dict(c),native_counter_mismatches=errors))
    result=dict(status='complete_attribution_analysis',request_segmentation=basic,attribution=report,scope_counts=counts,
                source_protocol_sha256=base.sha(output/'protocol.json'))
    base.write(leaf/'attribution.json',result,True)
    print(json.dumps(dict(status=result['status'],scope_count=len(metadata),ambiguous=report['ambiguous_launches'],count_mismatches=sum(bool(x['native_counter_mismatches']) for x in counts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','analyze']);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--only',choices=[j[0] for j in JOBS])
    a=parser.parse_args()
    if a.action=='prepare':prepare(a.output)
    elif a.action=='run':run(a.output,a.only)
    else:analyze(a.output)
