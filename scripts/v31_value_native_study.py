"""Source-frozen native development for diagnostically promoted value variants.

Never starts clean timing, changes the sampler, or resumes an older campaign.
Promotion is supplied by a hash-bound decision receipt after diagnostic review.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scripts import v31_value_decision_replay as io
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_official as official_rules
from experiments.numerical_qk_reuse.v31_value_refinement_runtime import MODES

CONTROLS=['current_v31_control','dense_full_fix51994','dense_piecewise','allkept_fa4']
NATIVE_CODE=['experiments/numerical_qk_reuse/'+p+'.py' for p in
    ['v31_value_decision','v31_value_online_kernel','v31_value_reuse_guard','v31_value_reuse_merge_kernel','v31_value_refinement_runtime','v31_same_state_suffix']]
NATIVE_CODE+=['scripts/v31_value_refinement_worker.py']
TEST_CODE=['tests/'+p+'.py' for p in ['test_v31_value_decision','test_v31_value_reuse_guard',
    'test_v31_same_state_suffix','test_v31_value_refinement_runtime','test_v31_value_native_study']]


def verify_h100_gate(gate):
    receipt=io.ROOT/gate['h100_tests_receipt']
    if io.sha(receipt)!=gate['h100_tests_receipt_sha256']:raise RuntimeError('H100 test evidence changed')
    proof=io.read(receipt)
    if (proof.get('status')!='passed' or 'H100' not in proof.get('device','') or
            proof.get('tests_passed',0)<1 or proof.get('tests_skipped')!=0):
        raise RuntimeError('complete H100 correctness tests required')
    for rel in NATIVE_CODE+TEST_CODE:
        if proof.get('source_sha256',{}).get(rel)!=io.sha(io.ROOT/rel):
            raise RuntimeError('H100 tests do not cover the current native source: '+rel)
    return receipt


def prepare(study,gate):
    if (study/'protocol.json').exists():
        verify(study);return
    if study.exists():raise RuntimeError('preserve partial preparation; use a new attempt')
    g=io.read(gate)
    if g.get('status')!='passed' or not g.get('promoted_modes') or any(m not in MODES for m in g['promoted_modes']) or len(set(g['promoted_modes']))!=len(g['promoted_modes']):
        raise RuntimeError('completed decision promotion is required')
    if set(g.get('native_source_sha256',{}))!=set(NATIVE_CODE):raise RuntimeError('native source is not bound to promotion')
    for p,digest in g['native_source_sha256'].items():
        if io.sha(io.ROOT/p)!=digest:raise RuntimeError('native implementation changed after promotion')
    h100=verify_h100_gate(g)
    batch=io.read(study.parent/'decision_batch_complete.json')
    if io.sha(study.parent/'decision_batch_complete.json')!=g['decision_batch_complete_sha256']:
        raise RuntimeError('decision evidence changed')
    for suite,digest in batch['reports_sha256'].items():
        if io.sha(study.parent/f'decisions_{suite}_attempt001/report/summary.json')!=digest:
            raise RuntimeError('decision report changed')
    execution=study/'private/execution';execution.mkdir(parents=True)
    files=subprocess.check_output(['git','ls-files'],cwd=io.ROOT,text=True).splitlines()
    for rel in files:
        p=io.ROOT/rel
        if not p.is_file() or p.suffix not in ('.py','.json','.yaml','.yml','.toml') or rel.split('/')[0] not in ('scripts','src','experiments','tests'):continue
        dst=execution/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    fingerprints={str(p.relative_to(execution)):io.sha(p) for p in execution.rglob('*') if p.is_file()}
    if 'scripts/v31_value_native_study.py' not in fingerprints:raise RuntimeError('commit native coordinator before freezing')
    shutil.copyfile(gate,study/'decision_promotion.json')
    shutil.copyfile(h100,study/'h100_tests.json')
    oldroot=io.ROOT/'results/v31_v2_local_threshold_sweep_20261014_attempt002'
    old=io.read(oldroot/'protocol.json')
    shutil.copyfile(oldroot/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
    for f in io.read(study/'model_source_fingerprints.json')['files']:
        model_file=io.MODEL/f['file']
        if model_file.stat().st_size!=f['bytes'] or io.sha(model_file)!=f['sha256']:
            raise RuntimeError('model bytes differ from frozen inventory')
    settings={s:dict(v) for s,v in base.SUITES.items()}
    settings['longbench_v2'].update(count=64,memory=.80)
    settings['ruler128k']=dict(dataset='ruler128k_v33ofc_subset',count=104,shards=4,
        max_len=136192,block=32,chunk=16384,memory=.85,mage_k=8192)
    inputs={}
    for suite,cfg in settings.items():
        source=oldroot if suite!='ruler128k' else io.ROOT/'results/v31_v2_ruler128k_seed42_tau021_20261015'
        src=source/f'private/pools/{suite}';pool=study/f'private/pools/{suite}'
        shutil.copytree(src,pool)
        cells=io.read(source/f'private/schedules/{suite}/cells.json')[:cfg['count']]
        if len(cells)!=cfg['count']:raise RuntimeError('panel size changed')
        schedule=study/f'private/schedules/{suite}';schedule.mkdir(parents=True)
        io.write(schedule/'cells.json',cells,True);base.check_inputs(pool,schedule/'cells.json')
        io.write(schedule/'smoke.json',cells[:2],True)
        for shard in range(cfg['shards']):io.write(schedule/f'shard{shard+1:03d}.json',cells[shard::cfg['shards']],True)
        for folder in (pool,schedule):
            for p in folder.rglob('*'):
                if p.is_file():inputs[str(p.relative_to(study))]=io.sha(p)
    ruler=io.read(io.ROOT/'results/v31_v2_ruler128k_seed42_tau021_20261015/private/scorer_location.json')
    io.write(study/'private/scorer_location.json',ruler,True)
    inputs['private/scorer_location.json']=io.sha(study/'private/scorer_location.json')
    inputs['decision_promotion.json']=io.sha(study/'decision_promotion.json')
    inputs['h100_tests.json']=io.sha(study/'h100_tests.json')
    protocol=dict(schema='value_native_development_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        execution_fingerprints=fingerprints,input_fingerprints=inputs,runtime=old['runtime'],
        model_inventory_sha256=io.sha(study/'model_source_fingerprints.json'),suites=settings,
        frozen_native=old['frozen_native'],
        arms=CONTROLS+g['promoted_modes'],scope='value changes only at GLOBAL layer29; other GLOBAL layers retain V31 and LOCAL remains dense',
        budget='same fixed prefix K in every head/query tile, plus common mandatory/carry phases',
        native_sampler='unchanged proposal, acceptance, re-noising, self-conditioning, stopping, temperatures and budgets',
        timing='paused; instrumented audit only',full503='requires separate completed development promotion and new frozen protocol',
        promotion='no observed accuracy loss or larger N in any workload, at least one accuracy gain; p95/cap increase <=5 percent; exploratory not noninferiority')
    io.write(study/'protocol.json',protocol,True)
    io.write(study/'receipts/source_fingerprints.json',dict(source_commit=protocol['source_commit'],execution_fingerprints=fingerprints),True)
    verify(study)


def verify(study):
    p=io.read(study/'protocol.json');base.verify(study,p)
    if io.sha(study/'model_source_fingerprints.json')!=p['model_inventory_sha256']:raise RuntimeError('model inventory changed')
    base.SUITES.update(p['suites'])
    return p


def validate(rows,suite,arm,planned):
    errors=base.validate_records(rows,suite,arm,None,planned)
    for row in rows:
        a=row['receipts']['adapter']
        if row.get('study_arm')!=arm:errors.append('wrong study arm')
        expected_graph='FULL' if arm=='dense_full_fix51994' else 'PIECEWISE'
        if set(a.get('decode_cudagraph_modes',{}))!={expected_graph}:errors.append('wrong compiled execution substrate')
        if arm in MODES:
            if a.get('value_refinement')!=arm or a.get('value_refinement_layers')!=[29]:errors.append('wrong value refinement/scope')
            if a.get('value_refinement_prefix_budget')!=base.SUITES[suite]['mage_k']//64:errors.append('wrong configured tile budget')
            if a.get('value_refinement_work',{}).get('budget_errors',0):errors.append('fixed budget violation')
            expected='unchanged native FA4' if arm.endswith('swap8') else 'native FA4 plus cached correction'
            if a.get('value_refinement_output_consumer')!=expected:errors.append('wrong correction consumer')
        elif a.get('value_refinement') is not None:errors.append('control changed by refinement')
    return sorted(set(errors))


def run_shard(study,suite,arm,shard,qual=False):
    p=verify(study);family=study/('qualification' if qual else 'runs')/suite/arm/shard.stem
    for marker in sorted(family.glob('attempt*/attempt_complete.json')):
        m=io.read(marker);leaf=marker.parent
        if m['protocol_sha256']!=io.sha(study/'protocol.json') or m['records_sha256']!=io.sha(leaf/'records.jsonl'):raise RuntimeError('completed shard changed')
        for rel,digest in m['private_sha256'].items():
            if io.sha(leaf/'private'/rel)!=digest:raise RuntimeError('private generation changed after completion')
        if validate(base.jsonl(leaf/'records.jsonl'),suite,arm,io.read(shard)):raise RuntimeError('completed receipt invalid')
        return leaf
    prior=[int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()]
    leaf=family/f'attempt{max(prior,default=0)+1:03d}';(leaf/'private').mkdir(parents=True)
    base.wait_idle()
    if base.runtime_identity()!=p['runtime']:raise RuntimeError('runtime/host/GPU changed')
    env=base.env_for(study,suite,arm);env['STUDY_ARM']=arm
    env['FROZEN_SOURCE_SHA256']=io.sha(study/'receipts/source_fingerprints.json')
    worker='v31_vllm_paired_bench.py'
    if arm in MODES:
        worker='v31_value_refinement_worker.py'
        env.update(VALUE_REFINEMENT=arm,VALUE_REFINEMENT_GATE=str(study/'decision_promotion.json'))
    actual='mage' if arm in MODES or arm=='current_v31_control' else ('allkept' if arm=='allkept_fa4' else 'dense')
    graph='default' if arm=='dense_full_fix51994' else 'PIECEWISE'
    execution=study/'private/execution'
    command=[base.PYTHON,str(execution/'scripts'/worker),str(io.MODEL),str(study/f'private/pools/{suite}'),
        str(shard),str(leaf/'records.jsonl'),str(leaf/'private'/f'run_{arm}.private.jsonl'),actual,graph]
    io.write(leaf/'config.json',dict(arm=arm,suite=suite,protocol_sha256=io.sha(study/'protocol.json'),
        cells_sha256=io.sha(shard),source_commit=p['source_commit'],expected_cells=len(io.read(shard))),True)
    io.write(study/'state.json',dict(stage='running',suite=suite,arm=arm,attempt=str(leaf.relative_to(study)),timing='paused'))
    start=time.monotonic()
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT);last=start
        while proc.poll() is None:
            if time.monotonic()-last>=1800:
                print(json.dumps(dict(event='progress',suite=suite,arm=arm,attempt=str(leaf.relative_to(study)))),flush=True);last=time.monotonic()
            time.sleep(5)
    rows=base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    errors=validate(rows,suite,arm,io.read(shard));verify(study)
    receipt=dict(status='complete' if proc.returncode==0 and not errors else 'failed',returncode=proc.returncode,
        validation_errors=errors,gpu_reservation_s=time.monotonic()-start,records=len(rows),
        protocol_sha256=io.sha(study/'protocol.json'),records_sha256=io.sha(leaf/'records.jsonl') if rows else None,
        private_sha256={f.name:io.sha(f) for f in (leaf/'private').glob('*.private.jsonl')})
    io.write(leaf/'receipt.json',receipt,True)
    if receipt['status']!='complete':
        io.write(study/'state.json',dict(stage='failed',attempt=str(leaf.relative_to(study)),resumable=True,timing='paused'))
        raise RuntimeError('preserved failed native attempt')
    io.write(leaf/'attempt_complete.json',receipt,True)
    return leaf


def score(study,suite,arm,attempts):
    if suite!='ruler128k':return base.score_arm(study,suite,arm,attempts)
    dest=study/f'scoring/{suite}/{arm}';dest.mkdir(parents=True,exist_ok=True)
    prefix=dest/'official';summary=Path(str(prefix)+'.summary.json')
    if not summary.exists():
        execution=study/'private/execution';pool=study/f'private/pools/{suite}'
        ruler=io.read(study/'private/scorer_location.json')['ruler']
        files=[str(f) for a in attempts for f in (a/'private').glob('*.private.jsonl')]
        cmd=[base.PYTHON,str(execution/'scripts/v31_score_ruler.py'),str(prefix),ruler,str(pool),str(pool),
            '--cells',str(study/f'private/schedules/{suite}/cells.json'),*files]
        env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}',PYTHONNOUSERSITE='1')
        with (dest/'scorer.log').open('w') as log:subprocess.run(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return io.read(summary),io.read(str(prefix)+'.official.json')[arm]


def paired(suite,control,candidate,task_of):
    if control.keys()!=candidate.keys():raise RuntimeError('official score cohorts differ')
    groups=defaultdict(list);scale=1 if suite=='ruler128k' else 100
    for key in control:
        cluster=task_of[int(key.split('|')[1])] if suite=='ruler128k' else tuple(key.split('|')[:2])
        groups[cluster].append(scale*(float(candidate[key])-float(control[key])))
    delta=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(1729)
    boot=delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1)
    return dict(delta_score_points=float(delta.mean()),ci95=list(map(float,np.quantile(boot,[.025,.975]))),
        clusters=len(groups),clustering='task' if suite=='ruler128k' else 'problem',draws=20000,
        inference='exploratory paired difference, not a noninferiority conclusion')


def paired_trajectory(suite,control,candidate,task_of):
    c={base.key(r):r for r in control};v={base.key(r):r for r in candidate}
    if c.keys()!=v.keys():raise RuntimeError('native trajectory cohorts differ')
    result={}
    for field in ('denoise_forwards','canvases','output_tokens'):
        groups=defaultdict(list)
        for key in c:
            cluster=task_of[int(key.split('|')[1])] if suite=='ruler128k' else tuple(key.split('|')[:2])
            groups[cluster].append(v[key][field]-c[key][field])
        delta=np.array([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(1729)
        boot=delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1)
        result[field]=dict(delta=float(delta.mean()),ci95=list(map(float,np.quantile(boot,[.025,.975]))),
            fewer_cells=sum(v[k][field]<c[k][field] for k in c),
            more_cells=sum(v[k][field]>c[k][field] for k in c),clusters=len(groups))
    ccan=np.mean([r['canvases'] for r in control]);vcan=np.mean([r['canvases'] for r in candidate])
    cr=sum(r['denoise_forwards'] for r in control)/sum(r['canvases'] for r in control)
    vr=sum(r['denoise_forwards'] for r in candidate)/sum(r['canvases'] for r in candidate)
    result['N_change_decomposition']=dict(canvas_count_component=float((vcan-ccan)*(vr+cr)/2),
        calls_per_canvas_component=float((vr-cr)*(vcan+ccan)/2),
        interpretation='algebraic aggregate decomposition, not a causal estimate')
    result['output_hash_mismatches']=sum(c[k]['output_hash']!=v[k]['output_hash'] for k in c)
    return result


def finish(study,paths):
    p=verify(study);summaries={};scored={};rows_by={};bindings={}
    task_of={r['index']:r['task'] for r in io.read(study/'private/pools/ruler128k/ruler128k_v33ofc_subset_rowinfo.json')['rows']}
    for suite,cfg in p['suites'].items():
        for arm in p['arms']:
            attempts=paths[suite,arm];rows=[r for leaf in attempts for r in base.jsonl(leaf/'records.jsonl')]
            official,values=score(study,suite,arm,attempts)
            c=official['coverage'][arm]
            if c['scored']!=cfg['count'] or c['missing'] or c['dropped']:raise RuntimeError('official scorer coverage failed')
            work=base.aggregate(rows);extra=defaultdict(int)
            local=sum(r['receipts']['adapter']['native_local_eligible_rectangles'] for r in rows)
            work.update(local_eligible_tiles=local,local_kept_tiles=local,
                overall_kept_rectangles=work['overall_eligible_rectangles']-work['overall_skipped_rectangles'],
                overall_denominator='actual decode GLOBAL full-support and LOCAL window-intersecting H x Q128 x KV64 rectangles')
            for r in rows:
                for k,v in r['receipts']['adapter'].get('value_refinement_work',{}).items():extra[k]+=v
            result=dict(arm=arm,suite=suite,score_points=float(np.mean(list(values.values())))*(1 if suite=='ruler128k' else 100),
                items=len(rows),work=work,extra_refinement_work=dict(extra),
                p95_N=float(np.quantile([r['denoise_forwards'] for r in rows],.95)),
                cap_rate=sum(official_rules.finish_class(r['finish_reason'])=='capped' for r in rows)/len(rows),
                attempts=[str(x.relative_to(study)) for x in attempts],timing='audit only; clean timing paused')
            dest=study/f'reports/{suite}/{arm}';io.write(dest/'summary.json',result,True)
            io.write(dest/'final_complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
                summary_sha256=io.sha(dest/'summary.json'),
                official_sha256=io.sha(study/f'scoring/{suite}/{arm}/official.official.json'),
                binding_sha256=io.sha(study/f'scoring/{suite}/{arm}/official.binding.json')),True)
            summaries[suite,arm]=result;scored[suite,arm]=values;rows_by[suite,arm]=rows
            bindings.setdefault(suite,{}).update(io.read(study/f'scoring/{suite}/{arm}/official.binding.json'))
        official_rules.check_comparable(bindings[suite],p['arms'],set(scored[suite,'current_v31_control']))
    comparisons=[];survivors=[]
    for arm in p['arms']:
        if arm in CONTROLS:continue
        passed=True;gain=False
        for suite in p['suites']:
            c=summaries[suite,'current_v31_control'];v=summaries[suite,arm]
            pair=paired(suite,scored[suite,'current_v31_control'],scored[suite,arm],task_of)
            ok=(v['score_points']>=c['score_points'] and v['work']['N']<=c['work']['N'] and
                v['p95_N']<=1.05*c['p95_N'] and v['cap_rate']<=1.05*c['cap_rate'])
            passed&=ok;gain|=v['score_points']>c['score_points']
            trajectory=paired_trajectory(suite,rows_by[suite,'current_v31_control'],rows_by[suite,arm],task_of)
            comparisons.append(dict(suite=suite,arm=arm,passes_screen=bool(ok),trajectory=trajectory,**pair))
        if passed and gain:survivors.append(arm)
    io.write(study/'development_promotion.json',dict(status='complete',full503_candidates=survivors,
        comparisons=comparisons,timing='paused',requires_new_full503_protocol=True),True)
    lines=['# Native value-refinement development audit','',
        'Exploratory source-matched official scores. Changes are restricted to GLOBAL layer29. Clean timing remains paused.', '',
        '| Workload | Method | Score | N/problem-trial | N/canvas | Output tokens/problem-trial |',
        '|---|---|---:|---:|---:|---:|']
    for (suite,arm),r in summaries.items():
        w=r['work'];lines.append(f"| {suite} | {arm} | {r['score_points']:.3f} | {w['mean_N']:.3f} | {w['N_per_C']:.3f} | {w['mean_T']:.1f} |")
    (study/'summary.md').write_text('\n'.join(lines)+'\n')
    io.write(study/'final_complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
        development_promotion_sha256=io.sha(study/'development_promotion.json'),timing='paused'),True)
    io.write(study/'state.json',dict(stage='complete',full503_candidates=survivors,timing='paused'))


def run(study):
    p=verify(study);paths={}
    # Mandatory execution/accounting smokes precede the panel. These are
    # exploratory development items and are not independent confirmation.
    for suite in p['suites']:
        for arm in p['arms']:run_shard(study,suite,arm,study/f'private/schedules/{suite}/smoke.json',True)
    for suite,cfg in p['suites'].items():
        arms=p['arms']
        for i in range(cfg['shards']):
            order=arms[i%len(arms):]+arms[:i%len(arms)]
            for arm in order:
                leaf=run_shard(study,suite,arm,study/f'private/schedules/{suite}/shard{i+1:03d}.json')
                paths.setdefault((suite,arm),[]).append(leaf)
    finish(study,paths)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--gate',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.gate is None:p.error('--gate is required for prepare')
        prepare(a.study.resolve(),a.gate.resolve())
    else:run(a.study.resolve())
