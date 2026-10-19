"""Immutable, budgeted diagnostics preceding the HumanEval K128 campaign."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_humaneval_value_study as old

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'results/humaneval_value_k256_20261018/attempt001'
DEFAULT = ROOT/'results/humaneval_reproducibility_20261019/attempt001'
SPEC = ROOT/'docs/HUMANEVAL_REPRODUCIBILITY_PLAN_20261019.md'
write, read, sha = base.write, base.read, base.sha
CAPS = dict(diagnosis=7200, k256=18000, k128=18000)


def prepare(study):
    if study.exists():
        raise RuntimeError('new immutable study required')
    proof = read(PARENT/'verified_complete.json')
    for rel, h in proof['files_sha256'].items():
        if sha(PARENT/rel) != h:
            raise RuntimeError('original artifact drift')
    # Selection uses only metadata and trajectories, never answer correctness.
    paths = old.selected_paths(PARENT, 'thinking_on', 'clean')
    clean = {r['index']:r for p in paths['control_before'] for r in base.jsonl(p/'records.jsonl')}
    paths = old.selected_paths(PARENT, 'thinking_on', 'audit')
    audit = {r['index']:r for p in paths['control_before'] for r in base.jsonl(p/'records.jsonl')}
    cells = read(PARENT/'private/schedules/thinking_on/cells.json')
    lengths = sorted(cells, key=lambda c:(len(c['prompt_tokens']), c['index']))
    ix = list(dict.fromkeys([0, lengths[0]['index'], lengths[-1]['index'], lengths[82]['index'],
        max(clean, key=lambda i:abs(clean[i]['denoise_forwards']-audit[i]['denoise_forwards'])),
        max(clean, key=lambda i:clean[i]['denoise_forwards'])]))
    for c in lengths:
        if len(ix) >= 6:
            break
        if c['index'] not in ix:
            ix.append(c['index'])
    study.mkdir(parents=True)
    shutil.copytree(PARENT/'private/execution', study/'private/execution')
    for rel in ('scripts/v31_humaneval_repro_worker.py', 'scripts/v31_humaneval_repro_study.py'):
        shutil.copyfile(ROOT/rel, study/'private/execution'/rel)
    shutil.copytree(PARENT/'private/pools', study/'private/pools')
    shutil.copytree(PARENT/'private/cache', study/'private/cache')
    for mode in old.MODES:
        cells = read(PARENT/f'private/schedules/{mode}/cells.json')
        write(study/f'private/schedules/{mode}/cohort.json', [cells[i] for i in ix], True)
        shutil.copyfile(PARENT/f'private/schedules/{mode}/warmup.json',
                        study/f'private/schedules/{mode}/warmup.json')
    shutil.copyfile(SPEC, study/'spec.md')
    p = dict(schema='humaneval_repro_v1', starting_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        parent_protocol_sha256=sha(PARENT/'protocol.json'), parent_seal_sha256=sha(PARENT/'verified_complete.json'),
        runtime=base.runtime_identity(), native=read(PARENT/'protocol.json')['native'],
        model_inventory_sha256=sha(PARENT/'model_source_fingerprints.json'),
        cohort=ix, cohort_selection='length extremes/median and original N divergence/tail; no scores',
        caps_s=CAPS, seed=42, prefix_budget_tokens=256,
        execution={str(p.relative_to(study/'private/execution')):sha(p)
                   for p in (study/'private/execution').rglob('*.py')}, spec_sha256=sha(study/'spec.md'))
    write(study/'protocol.json',p,True)
    write(study/'gpu_budget.json',dict(caps_s=CAPS,reservations=[]),True)
    write(study/'state.json',dict(status='prepared',full_runs_qualified=False))
    return p


def job(study, name, mode, family, pass_name, trace=False, overrides=None):
    p = read(study/'protocol.json')
    if base.runtime_identity() != p['runtime']:
        raise RuntimeError('runtime drift')
    if (study/'PAUSED_USER_REQUEST.json').exists():
        raise RuntimeError('paused')
    leaf = study/'diagnosis'/name
    if leaf.exists():
        raise RuntimeError('new immutable job name required')
    (leaf/'private').mkdir(parents=True)
    cells = read(study/f'private/schedules/{mode}/cohort.json')
    warm = read(study/f'private/schedules/{mode}/warmup.json')
    rows = [c for c in warm for _ in range(4)] + cells + cells
    write(leaf/'private/cells.json',rows,True)
    execution = study/'private/execution'
    base.SUITES[mode]=old.SETTINGS
    env = base.env_for(study,mode,old.CONTROL)
    env.update(PYTHONHASHSEED='0', REPEATS='1', TRACE=str(int(pass_name=='audit')),
        VALUE_AUDIT=str(int(pass_name=='audit')),VALUE_CLEAN_TIMING=str(int(pass_name=='clean')),
        VALUE_BOUNDARY_TIMING=str(int(pass_name=='clean')), STUDY_ARM=family)
    env.update(overrides or {})
    worker = execution/'scripts/v31_vllm_paired_bench.py'
    if trace:
        env.update(HUMANEVAL_REPRO_TRACE=str(leaf/'private/trace'),HUMANEVAL_REPRO_BASE=str(worker))
        worker = execution/'scripts/v31_humaneval_repro_worker.py'
    cmd = [base.PYTHON,str(worker),str(base.MODEL),str(study/f'private/pools/{mode}'),
           str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),
           'mage' if family=='v31' else 'allkept' if family=='allkept' else 'dense',
           'default' if family=='dense_full' else 'PIECEWISE']
    selected_env={k:v for k,v in env.items() if k not in os.environ or os.environ[k]!=v}
    write(leaf/'private/launch.json',dict(command=cmd,environment=selected_env),True)
    write(leaf/'config.json',dict(mode=mode,family=family,pass_name=pass_name,trace=trace,
        warmups=8,repetitions=2,cohort=p['cohort'],overrides=overrides or {},protocol_sha256=sha(study/'protocol.json')),True)
    base.wait_idle()
    budget=read(study/'gpu_budget.json')
    charged=sum(r.get('elapsed_s',r['reserved_s']) for r in budget['reservations'] if r['phase']=='diagnosis')
    remaining=CAPS['diagnosis']-charged
    if remaining < 180:
        raise RuntimeError('diagnosis budget exhausted')
    reserve=min(900,remaining)
    entry=dict(name=name,phase='diagnosis',reserved_s=reserve)
    budget['reservations'].append(entry);write(study/'gpu_budget.json',budget)
    write(study/'state.json',dict(status='running',job=name,full_runs_qualified=False))
    start=time.monotonic()
    print(json.dumps(dict(event='launch',job=name,reserved_s=reserve)),flush=True)
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
        last=start
        while proc.poll() is None:
            if time.monotonic()-start>reserve:
                proc.terminate();proc.wait(timeout=30)
                break
            if time.monotonic()-last>=1800:
                print(json.dumps(dict(event='progress',job=name,elapsed_s=time.monotonic()-start)),flush=True)
                last=time.monotonic()
            time.sleep(5)
    elapsed=time.monotonic()-start
    entry.update(elapsed_s=elapsed,returncode=proc.returncode);write(study/'gpu_budget.json',budget)
    records=base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    good=proc.returncode==0 and len(records)==len(rows)
    marker=dict(status='complete' if good else 'failed',elapsed_s=elapsed,returncode=proc.returncode,
        records=len(records),expected=len(rows),protocol_sha256=sha(study/'protocol.json'),
        files_sha256={str(f.relative_to(leaf)):sha(f) for f in leaf.rglob('*') if f.is_file()})
    write(leaf/('attempt_complete.json' if good else 'failed_attempt.json'),marker,True)
    write(study/'state.json',dict(status='diagnosis_pending',last_job=name,full_runs_qualified=False))
    print(json.dumps(dict(event='end',job=name,**{k:marker[k] for k in ('status','elapsed_s','records')})),flush=True)
    if not good:
        raise RuntimeError('preserved failed diagnostic; inspect worker log')


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','job'])
    a.add_argument('--study',type=Path,default=DEFAULT);a.add_argument('--name')
    a.add_argument('--mode',default='thinking_on');a.add_argument('--family',default='dense_full')
    a.add_argument('--pass-name',default='clean');a.add_argument('--trace',action='store_true')
    a.add_argument('--overrides',type=json.loads,default={});args=a.parse_args()
    if args.action=='prepare':
        print(json.dumps(prepare(args.study)['cohort']))
    else:
        job(args.study,args.name,args.mode,args.family,args.pass_name,args.trace,args.overrides)


if __name__=='__main__':
    main()
