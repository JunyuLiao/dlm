"""Immutable full503 seed42 clean/audit campaign with resumable engine shards."""
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_value_native_study as native
from scripts import v31_official as official

ROOT=Path(__file__).resolve().parents[1]
CONTROL='current_v31_control'
MODES=['full_swap8','residual_direction_guard']
REFERENCES=['dense_full_fix51994','dense_piecewise','allkept_fa4']
SUITE='longbench_v2'
read,sha,write=base.read,base.sha,base.write


def schedule(cells,warm,family,pass_name):
    rows=[];arms=[CONTROL];groups=[]
    if family=='shared':
        for c in warm:
            rows.extend([c]*len(MODES));arms.extend(MODES)
        warm_count=len(rows)
        for c in cells:
            shift=(c['index']+c['seed'])%2
            modes=MODES[shift:]+MODES[:shift]
            if pass_name=='audit':modes=modes[::-1]
            labels=['control_before']+modes+['control_after']
            group=[CONTROL]+modes+[CONTROL]
            groups.append(dict(start=len(rows),end=len(rows)+4,arms=group,labels=labels))
            rows.extend([c]*4);arms.extend(group)
    else:
        rows.extend(warm);warm_count=len(rows)
        for c in cells:
            groups.append(dict(start=len(rows),end=len(rows)+1,arms=[family],labels=[family]))
            rows.append(c)
    return rows,arms,groups,warm_count


def prepare(study):
    if study.exists():raise RuntimeError('new frozen campaign directory required')
    root=study.parent;parent=ROOT/'results/v31_v2_local_threshold_sweep_20261014_attempt002'
    replay=root/'replay_fast/attempt002';r=read(replay/'summary.json')
    done=read(replay/'final_complete.json')
    if done['summary_sha256']!=sha(replay/'summary.json'):raise RuntimeError('replay binding changed')
    exact=r['totals']['exact']
    if (exact['units']!=256 or any(exact[k] for k in ('mask_mismatches','swap_count_mismatches','budget_errors'))
        or r['moment_mismatches'] or r['dense_reduction_mismatches']):raise RuntimeError('exact backend failed replay')
    for rel,digest in read(replay/'config.json')['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('replayed source changed: '+rel)
    xml=root/'qualification_fast/attempt002/tests.xml';suites=ET.parse(xml).getroot()
    tests=sum(int(s.attrib.get('tests',0)) for s in suites.findall('testsuite'))
    if tests!=86 or any(int(s.attrib.get(k,0)) for s in suites.findall('testsuite') for k in ('failures','errors','skipped')):
        raise RuntimeError('complete H100 test gate required')
    execution=study/'private/execution';execution.mkdir(parents=True)
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    for rel in files:
        p=ROOT/rel
        if rel.split('/')[0] not in ('scripts','src','experiments','tests') or p.suffix not in ('.py','.json','.yaml','.yml','.toml') or not p.is_file():continue
        dst=execution/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    if not (execution/'scripts/v31_value_e2e_study.py').exists():raise RuntimeError('commit coordinator before freezing')
    pool=study/f'private/pools/{SUITE}'
    shutil.copytree(parent/f'private/pools/{SUITE}',pool)
    cells=read(parent/f'private/schedules/{SUITE}/cells.json')
    if len(cells)!=503 or {c['index'] for c in cells}!=set(range(503)):raise RuntimeError('full503 coverage required')
    cells=[dict(c,seed=42) for c in cells]
    folder=study/f'private/schedules/{SUITE}'
    write(folder/'cells.json',cells,True);base.check_inputs(pool,folder/'cells.json')
    warm=cells[:2]
    write(folder/'warmup.json',warm,True);write(folder/'smoke.json',warm,True)
    for i in range(16):write(folder/f'shard{i+1:03d}.json',cells[i::16],True)
    shutil.copyfile(parent/'model_source_fingerprints.json',study/'model_source_fingerprints.json')
    for f in read(study/'model_source_fingerprints.json')['files']:
        p=base.MODEL/f['file']
        if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise RuntimeError('model source changed')
    evidence=study/'qualification_evidence';evidence.mkdir()
    for src,name in [(xml,'tests.xml'),(replay/'config.json','replay_config.json'),
        (replay/'summary.json','replay_summary.json'),(replay/'final_complete.json','replay_complete.json')]:shutil.copyfile(src,evidence/name)
    spec=ROOT/'docs/V31_VALUE_E2E_LONGBENCH503_PLAN_20261017.md';shutil.copyfile(spec,study/'spec.md')
    config=dict(schema='value_full503_clean_audit_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        execution_fingerprints={str(p.relative_to(execution)):sha(p) for p in execution.rglob('*') if p.is_file()},
        input_fingerprints={str(p.relative_to(study)):sha(p) for path in (pool,folder,evidence) for p in path.rglob('*') if p.is_file()},
        model_inventory_sha256=sha(study/'model_source_fingerprints.json'),runtime=base.runtime_identity(),
        suites={SUITE:dict(base.SUITES[SUITE],count=503,shards=16,memory=.80)},
        frozen_native=native.native_contract(base.MODEL),panel_seed=42,
        arms=[CONTROL]+MODES+REFERENCES,passes=['clean','audit'],
        value_scope='GLOBAL layer29 only; fixed Q128/KV64 prefix K128, common mandatory tail/carry; native LOCAL dense',
        backend='device_exact_v1',spec_sha256=sha(study/'spec.md'),
        ordering='16 index-strided shards; clean/audit order alternates; four engine families rotate by shard; candidates rotate by index and reverse across passes',
        warmup='two fixed original inputs, full native budget; both candidate methods before measured groups; implicit initial V31 warmup',
        timing='clean request/prefill boundaries, no step synchronization or diagnostic kernels; audit independently scored',
        failure_policy='stop on structural failure; preserve attempt; resume new attempt only; never retry valid unfavorable outcomes',
        bootstrap=dict(draws=20000,seed=1729,cluster='question; additional engine-shard sensitivity'),
        status='frozen_before_native_qualification')
    write(study/'protocol.json',config,True)
    write(study/'receipts/source_fingerprints.json',dict(source_commit=config['source_commit'],execution_fingerprints=config['execution_fingerprints']),True)
    verify(study)
    print(dict(event='frozen',questions=503,seed=42,source=config['source_commit']),flush=True)


def verify(study):
    p=read(study/'protocol.json');base.verify(study,p)
    if sha(study/'model_source_fingerprints.json')!=p['model_inventory_sha256']:raise RuntimeError('model inventory changed')
    if sha(study/'spec.md')!=p['spec_sha256']:raise RuntimeError('spec changed')
    base.SUITES.update(p['suites']);return p


def validate(row,cell,arm,pass_name,oracle=False):
    errors=[];a=(row.get('receipts') or {}).get('adapter') or {}
    if pass_name=='audit':errors+=native.validate([dict(row,study_arm=arm)],SUITE,arm,[cell])
    for ok,name in [
        (row.get('dataset')==cell['dataset'] and row.get('index')==cell['index'] and row.get('panel_seed')==42 and row.get('repeat')==0,'input/seed'),
        (row.get('cuda_graph_captures')==0,'measured graph capture'),
        (row.get('block_size')==32 and row.get('chunk')==16384 and row.get('max_model_len')==141312,'geometry'),
        (row.get('fix_51994') is True and row.get('fa4_local_fix') is True,'dense/local fixes'),
        (row.get('value_clean_timing')==(pass_name=='clean'),'pass identity'),
        (row.get('timer_policy')==('request_and_prefill_boundaries' if pass_name=='clean' else 'synchronized_each_engine_step'),'timer boundaries'),
        (row.get('value_audit')==(pass_name=='audit'),'instrumentation'),
        (row.get('local_kv_budget') is None,'LOCAL budget'),
        (row.get('denoise_forwards',0)>0 and row.get('canvases',0)>0 and row.get('output_tokens',0)>0,'empty native trajectory'),
        (row.get('engine_steps',0)==row.get('prefill_steps',0)+row.get('sampler_calls',0),'engine/sample accounting')]:
        if not ok:errors.append(name)
    if pass_name=='clean' and abs(row['prefill_s']+row['decode_s']-row['wall_s'])>2e-5:errors.append('boundary time sum')
    if arm in MODES:
        if a.get('value_refinement_backend')!='device_exact_v1' or a.get('value_refinement_layers')!=[29]:errors.append('backend/scope')
        if a.get('value_refinement_prefix_budget')!=128:errors.append('prefix budget')
        if a.get('value_refinement_budget_audited')!=(pass_name=='audit'):errors.append('budget audit flag')
        work=a.get('value_refinement_work',{})
        if pass_name=='audit' and work.get('budget_errors')!=0:errors.append('budget errors/missing audit')
        if oracle and not work.get('native_same_state_oracle_checks'):errors.append('missing native oracle')
    elif a.get('value_refinement') is not None:errors.append('modified control')
    return errors


def verify_leaf(study,leaf):
    r=read(leaf/'attempt_complete.json')
    if r['protocol_sha256']!=sha(study/'protocol.json'):raise RuntimeError('attempt protocol changed')
    for name,digest in r['files_sha256'].items():
        if sha(leaf/name)!=digest:raise RuntimeError('attempt record changed')
    return r


def run_job(study,pass_name,family,shard,qual=False):
    p=verify(study);root=study/('qualification' if qual else 'runs')/pass_name/family/shard.stem
    markers=list(root.glob('attempt*/attempt_complete.json'))
    if markers:
        if len(markers)!=1:raise RuntimeError('ambiguous completion')
        leaf=markers[0].parent;verify_leaf(study,leaf);return leaf
    number=max([int(x.name[7:]) for x in root.glob('attempt*')],default=0)+1
    leaf=root/f'attempt{number:03d}';(leaf/'private').mkdir(parents=True)
    cells,arms,groups,warm_count=schedule(read(shard),read(study/f'private/schedules/{SUITE}/warmup.json'),family,pass_name)
    write(leaf/'private/cells.json',cells,True);write(leaf/'private/arms.json',arms,True)
    write(leaf/'config.json',dict(protocol_sha256=sha(study/'protocol.json'),family=family,pass_name=pass_name,
        groups=groups,warmup_requests=warm_count,cells_sha256=sha(leaf/'private/cells.json'),arms_sha256=sha(leaf/'private/arms.json')),True)
    base.wait_idle()
    if base.runtime_identity()!=p['runtime']:raise RuntimeError('runtime identity changed')
    execution=study/'private/execution';env=base.env_for(study,SUITE,CONTROL if family=='shared' else family)
    env.update(STUDY_ARM=family,PYTHONHASHSEED='0',
        FROZEN_SOURCE_SHA256=sha(study/'receipts/source_fingerprints.json'),
        VALUE_AUDIT=str(int(pass_name=='audit')),VALUE_CLEAN_TIMING=str(int(pass_name=='clean')),
        VALUE_BOUNDARY_TIMING=str(int(pass_name=='clean')),
        NATIVE_INTERLEAVED_ARMS=str(leaf/'private/arms.json'),
        NATIVE_INTERLEAVED_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
    worker='v31_vllm_paired_bench.py'
    if family=='shared':worker='v31_value_e2e_qualification_worker.py' if qual and pass_name=='audit' else 'v31_value_e2e_worker.py'
    arm='mage' if family=='shared' else 'allkept' if family=='allkept_fa4' else 'dense'
    graph='default' if family=='dense_full_fix51994' else 'PIECEWISE'
    cmd=[base.PYTHON,str(execution/'scripts'/worker),str(base.MODEL),str(study/f'private/pools/{SUITE}'),
        str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),arm,graph]
    write(study/'state.json',dict(stage='qualification' if qual else 'full503',pass_name=pass_name,family=family,
        attempt=str(leaf.relative_to(study)),resumable=True,timing='enabled only for this study'))
    start=time.monotonic()
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT);last=start
        while proc.poll() is None:
            if time.monotonic()-last>=1800:
                count=sum(1 for _ in (leaf/'records.jsonl').open()) if (leaf/'records.jsonl').exists() else 0
                print(dict(event='progress',pass_name=pass_name,family=family,shard=shard.stem,records=count,planned=len(cells)),flush=True);last=time.monotonic()
            time.sleep(10)
    rows=base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    private=base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
    errors=[];warm_work={m:defaultdict(int) for m in MODES};drift=[]
    if proc.returncode or len(rows)!=len(cells) or len(private)!=len(cells):errors.append('worker/coverage failure')
    if not errors:
        for i,(row,cell) in enumerate(zip(rows,cells)):
            method=arms[i+1] if family=='shared' else family
            errors+=validate(row,cell,method,pass_name,qual and pass_name=='audit' and family=='shared')
            a=(row.get('receipts') or {}).get('adapter') or {}
            if family=='shared' and (a.get('comparison_arm')!=method or a.get('comparison_request_index')!=i+1):errors.append('live schedule mismatch')
            if i<warm_count and method in MODES:
                for k,v in a.get('value_refinement_work',{}).items():warm_work[method][k]+=v
        if family=='shared':
            for method,work in warm_work.items():
                if not work.get('anchor_events') or (method=='residual_direction_guard' and not work.get('corrected_calls')):errors.append('incomplete candidate warmup')
            for group in groups:
                a,b=rows[group['start']],rows[group['end']-1]
                drift.append(dict(index=a['index'],hash_equal=a['output_hash']==b['output_hash'],N_before=a['denoise_forwards'],N_after=b['denoise_forwards']))
    receipt=dict(status='failed' if errors else 'complete',returncode=proc.returncode,errors=sorted(set(errors)),
        protocol_sha256=sha(study/'protocol.json'),gpu_reservation_s=time.monotonic()-start,records=len(rows),
        warmup_work={k:dict(v) for k,v in warm_work.items()},control_drift=drift)
    write(leaf/'receipt.json',receipt,True)
    if errors:raise RuntimeError('preserved failed attempt: '+str(leaf))
    exports=defaultdict(list)
    for group in groups:
        for i,label in zip(range(group['start'],group['end']),group['labels']):exports[label].append(i)
    files=['records.jsonl','private/generation.private.jsonl','receipt.json','config.json']
    for label,indices in exports.items():
        for rel,data in [(f'exports/{label}/records.jsonl',[dict(rows[i],study_arm=label) for i in indices]),
            (f'exports/{label}/private/run_{label}.private.jsonl',[private[i] for i in indices])]:
            path=leaf/rel;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as f:
                for row in data:f.write(json.dumps(row)+'\n')
            files.append(rel)
    receipt['files_sha256']={name:sha(leaf/name) for name in files}
    write(leaf/'attempt_complete.json',receipt,True)
    print(dict(event='engine_complete',pass_name=pass_name,family=family,shard=shard.stem,records=len(rows),gpu_s=receipt['gpu_reservation_s']),flush=True)
    return leaf


def score_pass(study,pass_name,paths,qual=False):
    execution=study/'private/execution';pool=study/f'private/pools/{SUITE}'
    planned=study/f'private/schedules/{SUITE}'/('smoke.json' if qual else 'cells.json')
    result={}
    for label,folders in paths.items():
        dest=study/('qualification_scoring' if qual else 'scoring')/pass_name/label
        dest.mkdir(parents=True,exist_ok=True);prefix=dest/'official'
        files=[str(f) for folder in folders for f in (folder/'private').glob('*.private.jsonl')]
        if not (dest/'score_complete.json').exists():
            cmd=[base.PYTHON,str(execution/'scripts/v31_score_longbench_official.py'),str(prefix),str(pool),str(pool),'--cells',str(planned),*files]
            with (dest/'scorer.log').open('w') as log:
                subprocess.run(cmd,cwd=execution,env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}',PYTHONNOUSERSITE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            write(dest/'score_complete.json',dict(status='complete',protocol_sha256=sha(study/'protocol.json'),
                input_sha256={str(Path(f).relative_to(study)):sha(f) for f in files},
                outputs_sha256={p.name:sha(p) for p in dest.glob('official.*.json')}),True)
        proof=read(dest/'score_complete.json')
        for rel,digest in proof['input_sha256'].items():
            if sha(study/rel)!=digest:raise RuntimeError('scored input changed')
        for rel,digest in proof['outputs_sha256'].items():
            if sha(dest/rel)!=digest:raise RuntimeError('official score changed')
        result[label]=dict(official=read(str(prefix)+'.official.json')[label],
            records=[r for folder in folders for r in base.jsonl(folder/'records.jsonl')])
    return result


def run(study,qualification_only=False):
    p=verify(study)
    if sha(__file__)!=p['execution_fingerprints']['scripts/v31_value_e2e_study.py']:raise RuntimeError('use frozen coordinator')
    for qual in (True,False):
        if not qual and qualification_only:return
        paths={name:defaultdict(list) for name in ('clean','audit')}
        shards=[study/f'private/schedules/{SUITE}/smoke.json'] if qual else [study/f'private/schedules/{SUITE}/shard{i:03d}.json' for i in range(1,17)]
        for ordinal,shard in enumerate(shards):
            passes=['audit','clean'] if qual or ordinal%2==0 else ['clean','audit']
            families=['shared']+REFERENCES
            shift=ordinal%len(families);families=families[shift:]+families[:shift]
            for pass_name in passes:
                for family in families:
                    leaf=run_job(study,pass_name,family,shard,qual)
                    for folder in (leaf/'exports').iterdir():paths[pass_name][folder.name].append(folder)
        for pass_name in paths:
            result=score_pass(study,pass_name,paths[pass_name],qual)
            if qual:
                write(study/f'qualification/{pass_name}/qualification_complete.json',dict(status='passed',
                    protocol_sha256=sha(study/'protocol.json'),labels=list(result),
                    policy='same-state oracle plus structure/official scorer; every valid outcome retained'),True)
            else:
                from scripts.v31_value_e2e_report import report
                report(study,pass_name,result)
        if qual:print(dict(event='native_clean_and_audit_qualified'),flush=True)
    from scripts.v31_value_e2e_report import finish
    finish(study)
    write(study/'state.json',dict(stage='complete',resumable=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','qualification','run'],required=True)
    args=parser.parse_args();study=args.study.resolve()
    if args.stage=='prepare':prepare(study)
    else:
        try:run(study,args.stage=='qualification')
        except Exception:
            state=read(study/'state.json') if (study/'state.json').exists() else {}
            write(study/'state.json',dict(state,stage='failed',resumable=True))
            raise
