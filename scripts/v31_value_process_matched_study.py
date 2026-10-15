"""Frozen native development with same-engine controls for value comparisons.

The old independent-process panel remains held. This named replacement retains
its inputs and methods, while blocking the incremental comparison by engine.
"""
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scripts import v31_value_decision_replay as io
from scripts import v31_value_native_study as native
from scripts.v31_value_interleaved_worker import CONTROL

OVERLAY=['scripts/v31_value_interleaved_worker.py','scripts/v31_value_process_matched_study.py',
         'tests/test_v31_value_interleaved_worker.py','tests/test_v31_value_process_matched_study.py']
FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','rng_seed')


def expand(cells,modes):
    rows=[];arms=[];groups=[]
    for cell in cells:
        shift=(int(cell['index'])+int(cell['seed']))%len(modes)
        order=modes[shift:]+modes[:shift]
        group=[CONTROL]+order+[CONTROL]
        groups.append(dict(start=len(rows),end=len(rows)+len(group),arms=group))
        rows.extend([cell]*len(group));arms.extend(group)
    return rows,[CONTROL]+arms,groups


def prepare(study,parent,qualification):
    if study.exists():raise RuntimeError('use a new named campaign')
    old=native.verify(parent)
    done=io.read(qualification/'final_complete.json');summary=io.read(qualification/'summary.json')
    if done['summary_sha256']!=io.sha(qualification/'summary.json') or summary.get('gate')!='passed':
        raise RuntimeError('complete process-matched qualification required')
    qp=io.read(qualification/'protocol.json')
    if summary['protocol_sha256']!=io.sha(qualification/'protocol.json') or qp['parent_protocol_sha256']!=io.sha(parent/'protocol.json'):
        raise RuntimeError('qualification source or parent differs')
    if qp['worker_sha256']!=io.sha(io.ROOT/OVERLAY[0]):raise RuntimeError('qualified request switcher changed')
    execution=study/'private/execution';execution.parent.mkdir(parents=True)
    shutil.copytree(parent/'private/execution',execution)
    for rel in OVERLAY:
        dest=execution/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/rel,dest)
    shutil.copytree(parent/'private/pools',study/'private/pools')
    shutil.copytree(parent/'private/schedules',study/'private/schedules',ignore=shutil.ignore_patterns('shard*.json'))
    shutil.copyfile(parent/'private/scorer_location.json',study/'private/scorer_location.json')
    for name in ('decision_promotion.json','h100_tests.json','model_source_fingerprints.json'):
        shutil.copyfile(parent/name,study/name)
    proofdir=study/'receipts/process_qualification';proofdir.mkdir(parents=True)
    for name in ('config.json','summary.json','final_complete.json'):
        shutil.copyfile(qualification/name,proofdir/name)
    p=json.loads(json.dumps(old));p['schema']='value_native_process_matched_v1'
    p['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    p['deployment_recipe']=dict(base_source_commit=old['source_commit'],base_protocol_sha256=io.sha(parent/'protocol.json'),
        overlay_commit=p['source_commit'],overlay_files=OVERLAY,description='frozen parent source plus explicitly listed overlay; file hashes define deployment')
    p['comparison_design']=dict(methods='V31 and candidates share each engine; per-cell control bookends',
        order='candidate Latin rotation by (item index + seed) modulo candidate count',
        references='dense FULL, dense PIECEWISE and all-kept remain separate named engine paths',
        process_variation='unresolved; process matching qualifies incremental V31 comparisons, not cross-process numerical reproducibility',
        qualification_sha256=io.sha(proofdir/'final_complete.json'),
        uncertainty='problem/task clusters plus a separate process-cluster sensitivity interval',
        interpretation='exploratory previously examined development inputs; no held-out or noninferiority claim')
    for suite,n in dict(aime26=10,longbench_v2=8,math500=10,ruler128k=13).items():
        p['suites'][suite]['shards']=n
        folder=study/f'private/schedules/{suite}';cells=io.read(folder/'cells.json')
        for i in range(n):io.write(folder/f'shard{i+1:03d}.json',cells[i::n],True)
    p['execution_fingerprints']={str(f.relative_to(execution)):io.sha(f) for f in execution.rglob('*') if f.is_file()}
    inputs={}
    for folder in (study/'private/pools',study/'private/schedules',proofdir):
        inputs.update({str(f.relative_to(study)):io.sha(f) for f in folder.rglob('*') if f.is_file()})
    for name in ('decision_promotion.json','h100_tests.json','private/scorer_location.json'):
        inputs[name]=io.sha(study/name)
    p['input_fingerprints']=inputs
    io.write(study/'protocol.json',p,True)
    io.write(study/'receipts/source_fingerprints.json',dict(source_commit=p['source_commit'],deployment_recipe=p['deployment_recipe'],
        execution_fingerprints=p['execution_fingerprints']),True)
    native.verify(study)


def identical(a,b):return all(a[k]==b[k] for k in FIELDS)


def validate_group(rows,arms):
    errors=[];reference=rows[0]
    if not identical(reference,rows[-1]):errors.append('control bookend drift')
    for row,arm in zip(rows[1:-1],arms[1:-1]):
        work=row['receipts']['adapter'].get('value_refinement_work',{})
        active=bool(work.get('swapped_blocks',0) or work.get('corrected_calls',0))
        if not active and not identical(reference,row):errors.append('inactive candidate drift: '+arm)
    return errors


def shared(study,suite,shard,qual=False):
    p=native.verify(study);modes=[x for x in p['arms'] if x not in native.CONTROLS]
    planned=io.read(shard);cells,arms,groups=expand(planned,modes)
    family=study/('shared_qualification' if qual else 'shared_runs')/suite/shard.stem
    markers=list(family.glob('attempt*/attempt_complete.json'))
    if markers:
        if len(markers)!=1:raise RuntimeError('ambiguous completed shared shard')
        proof=io.read(markers[0]);leaf=markers[0].parent
        if proof['protocol_sha256']!=io.sha(study/'protocol.json') or proof['records_sha256']!=io.sha(leaf/'records.jsonl'):
            raise RuntimeError('completed shared shard changed')
        if proof['private_sha256']!=io.sha(leaf/'private/generation.private.jsonl'):
            raise RuntimeError('completed shared private generation changed')
        for rel,digest in proof['exports'].items():
            if io.sha(study/rel)!=digest:raise RuntimeError('derived arm completion changed')
            marker=study/rel;binding=io.read(marker);dest=marker.parent
            if io.sha(dest/'records.jsonl')!=binding['records_sha256']:raise RuntimeError('derived records changed')
            for name,expected in binding['private_sha256'].items():
                if io.sha(dest/'private'/name)!=expected:raise RuntimeError('derived private generation changed')
        return {arm:study/rel for arm,rel in proof['paths'].items()}
    ordinal=max([int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()],default=0)+1
    leaf=family/f'attempt{ordinal:03d}';(leaf/'private').mkdir(parents=True)
    io.write(leaf/'private/cells.json',cells,True);io.write(leaf/'private/arms.json',arms,True)
    io.write(leaf/'config.json',dict(protocol_sha256=io.sha(study/'protocol.json'),source_commit=p['source_commit'],
        planned_cells_sha256=io.sha(shard),expanded_cells_sha256=io.sha(leaf/'private/cells.json'),
        expanded_arms_sha256=io.sha(leaf/'private/arms.json'),groups=groups),True)
    native.base.wait_idle()
    if native.base.runtime_identity()!=p['runtime']:raise RuntimeError('runtime changed')
    env=native.base.env_for(study,suite,CONTROL);execution=study/'private/execution'
    env.update(STUDY_ARM='shared_engine_comparison',PYTHONHASHSEED='0',
        FROZEN_SOURCE_SHA256=io.sha(study/'receipts/source_fingerprints.json'),
        VALUE_REFINEMENT_GATE=str(study/'decision_promotion.json'),
        NATIVE_INTERLEAVED_ARMS=str(leaf/'private/arms.json'),
        NATIVE_INTERLEAVED_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
    cmd=[native.base.PYTHON,str(execution/OVERLAY[0]),str(io.MODEL),str(study/f'private/pools/{suite}'),
        str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
    io.write(study/'state.json',dict(stage='shared_qualification' if qual else 'shared_panel',suite=suite,
        attempt=str(leaf.relative_to(study)),timing='paused'))
    start=time.monotonic()
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT);last=start
        while proc.poll() is None:
            if time.monotonic()-last>=1800:
                print(dict(event='progress',suite=suite,attempt=str(leaf.relative_to(study))),flush=True);last=time.monotonic()
            time.sleep(5)
    rows=native.base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    private=native.base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
    errors=[];selected=defaultdict(list);private_by=defaultdict(list)
    if proc.returncode or len(rows)!=len(cells) or len(private)!=len(cells):errors.append('worker or coverage failure')
    if len(rows)==len(cells) and len(private)==len(cells):
        for i,(row,arm,cell) in enumerate(zip(rows,arms[1:],cells)):
            a=row['receipts']['adapter']
            if a.get('comparison_request_index')!=i+1 or a.get('comparison_arm')!=arm:errors.append('live request arm mismatch')
            if row.get('study_arm')!='shared_engine_comparison':errors.append('raw study label changed')
            errors+=native.validate([dict(row,study_arm=arm)],suite,arm,[cell])
            if row['cuda_graph_captures']:errors.append('unexpected request graph capture')
        for group in groups:
            lo,hi=group['start'],group['end'];errors+=validate_group(rows[lo:hi],group['arms'])
            if private[lo]['completion']!=private[hi-1]['completion']:errors.append('control decoded text drift')
            for i in range(lo,hi-1):
                arm=arms[i+1]
                selected[arm].append(dict(rows[i],study_arm=arm,comparison_engine=str(leaf.relative_to(study)),comparison_raw_ordinal=i))
                private_by[arm].append(private[i])
    proof=dict(status='failed' if errors else 'complete',validation_errors=sorted(set(errors)),returncode=proc.returncode,
        protocol_sha256=io.sha(study/'protocol.json'),gpu_reservation_s=time.monotonic()-start,
        records_sha256=io.sha(leaf/'records.jsonl') if rows else None,
        private_sha256=io.sha(leaf/'private/generation.private.jsonl') if private else None)
    io.write(leaf/'receipt.json',proof,True)
    if errors:raise RuntimeError('preserved failed process-matched shard')
    native.verify(study);exports={};paths={}
    for arm in [CONTROL]+modes:
        dest=study/('qualification' if qual else 'runs')/suite/arm/shard.stem/f'attempt{ordinal:03d}'
        (dest/'private').mkdir(parents=True)
        for path,data in ((dest/'records.jsonl',selected[arm]),(dest/'private'/f'run_{arm}.private.jsonl',private_by[arm])):
            with path.open('x') as f:
                for row in data:f.write(json.dumps(row)+'\n')
        binding=dict(status='complete',validation_errors=[],protocol_sha256=io.sha(study/'protocol.json'),
            records_sha256=io.sha(dest/'records.jsonl'),private_sha256={f'run_{arm}.private.jsonl':io.sha(dest/'private'/f'run_{arm}.private.jsonl')},
            raw_records_sha256=proof['records_sha256'],raw_attempt=str(leaf.relative_to(study)),
            requires_shared_completion=str((leaf/'attempt_complete.json').relative_to(study)),
            derivation='live receipt and frozen ordinal select rows; study_arm is normalized; control bookend retained only in raw audit',
            gpu_reservation_s=0,accounting='GPU reservation belongs to the shared raw attempt; do not count it once per exported arm')
        io.write(dest/'receipt.json',binding,True);io.write(dest/'attempt_complete.json',binding,True)
        exports[str((dest/'attempt_complete.json').relative_to(study))]=io.sha(dest/'attempt_complete.json')
        paths[arm]=dest
    io.write(leaf/'attempt_complete.json',dict(proof,exports=exports,paths={a:str(x.relative_to(study)) for a,x in paths.items()}),True)
    print(dict(event='shared_shard_complete',suite=suite,shard=shard.stem,qualified_controls=len(groups)),flush=True)
    return paths


def process_sensitivity(study,paths):
    """Engine-cluster resampling alongside the original problem/task intervals.

    This is a sensitivity analysis for the unresolved process effect; it is
    not a claim that engine and problem effects are independent or that their
    separate intervals form a joint confidence region.
    """
    p=native.verify(study);results=[]
    for suite in p['suites']:
        _,control=native.score(study,suite,CONTROL,paths[suite,CONTROL])
        cr={native.base.key(r):r for leaf in paths[suite,CONTROL] for r in native.base.jsonl(leaf/'records.jsonl')}
        for arm in [x for x in p['arms'] if x not in native.CONTROLS]:
            _,candidate=native.score(study,suite,arm,paths[suite,arm])
            vr={native.base.key(r):r for leaf in paths[suite,arm] for r in native.base.jsonl(leaf/'records.jsonl')}
            if cr.keys()!=vr.keys() or control.keys()!=candidate.keys() or control.keys()!=cr.keys():
                raise RuntimeError('process sensitivity cohorts differ')
            groups=defaultdict(list);scale=1 if suite=='ruler128k' else 100
            for key in cr:
                engine=cr[key]['comparison_engine']
                if vr[key]['comparison_engine']!=engine:raise RuntimeError('candidate and control used different engines')
                groups[engine].append([scale*(float(candidate[key])-float(control[key])),vr[key]['denoise_forwards']-cr[key]['denoise_forwards']])
            # All planned shards have equal cell counts. Refuse unequal groups
            # instead of accidentally changing the item-weighted estimand.
            if len({len(v) for v in groups.values()})!=1:raise RuntimeError('unequal process blocks require weighted bootstrap')
            means=np.array([np.mean(v,axis=0) for v in groups.values()]);rng=np.random.default_rng(1729)
            boot=means[rng.integers(len(means),size=(20000,len(means)))].mean(1)
            results.append(dict(suite=suite,arm=arm,process_clusters=len(means),
                delta_score_points=float(means[:,0].mean()),score_ci95=list(map(float,np.quantile(boot[:,0],[.025,.975]))),
                delta_N=float(means[:,1].mean()),N_ci95=list(map(float,np.quantile(boot[:,1],[.025,.975])))))
    return dict(status='complete',results=results,draws=20000,
        inference='process-cluster sensitivity alongside problem/task clustered intervals; not a joint-coverage guarantee',timing='paused')


def write_or_verify(path,data,exclusive=False):
    """Resume a deterministic report by verifying existing immutable JSON."""
    path=Path(path)
    if exclusive and path.exists():
        if io.read(path)!=data:raise RuntimeError('existing immutable report differs: '+str(path))
        return
    return ORIGINAL_WRITE(path,data,exclusive)


ORIGINAL_WRITE=io.write


def run(study):
    p=native.verify(study)
    if io.sha(__file__)!=p['execution_fingerprints'][OVERLAY[1]]:raise RuntimeError('execute the frozen coordinator')
    references=[x for x in native.CONTROLS if x!=CONTROL];paths={}
    for suite in p['suites']:
        marker=study/f'qualification/{suite}/qualification_complete.json'
        leaves=shared(study,suite,study/f'private/schedules/{suite}/smoke.json',True)
        for arm in references:leaves[arm]=native.run_shard(study,suite,arm,study/f'private/schedules/{suite}/smoke.json',True)
        bindings={};keys=None
        for arm,leaf in leaves.items():
            official,values=native.score(study,suite,arm,[leaf],qual=True);coverage=official['coverage'][arm]
            if coverage['scored']!=2 or coverage['missing'] or coverage['dropped']:raise RuntimeError('smoke scorer coverage')
            if keys is not None and keys!=set(values):raise RuntimeError('smoke cohorts differ')
            keys=set(values);bindings.update(io.read(study/f'qualification_scoring/{suite}/{arm}/official.binding.json'))
            if arm in native.MODES:
                work=defaultdict(int)
                for row in native.base.jsonl(leaf/'records.jsonl'):
                    for k,v in row['receipts']['adapter']['value_refinement_work'].items():work[k]+=v
                if not work.get('anchor_events') or (not arm.endswith('swap8') and not work.get('corrected_calls')):
                    raise RuntimeError('sparse smoke exposure missing')
        native.official_rules.check_comparable(bindings,p['arms'],keys)
        write_or_verify(marker,dict(status='passed',protocol_sha256=io.sha(study/'protocol.json'),
            scope='same-engine control bookends, inactive parity wherever observed, active sparse exposure, budgets, scope, source and official scorer'),True)
    for suite,cfg in p['suites'].items():
        for i in range(cfg['shards']):
            shard=study/f'private/schedules/{suite}/shard{i+1:03d}.json'
            leaves=shared(study,suite,shard)
            order=references[i%len(references):]+references[:i%len(references)]
            for arm in order:leaves[arm]=native.run_shard(study,suite,arm,shard)
            for arm,leaf in leaves.items():paths.setdefault((suite,arm),[]).append(leaf)
    write_or_verify(study/'process_cluster_sensitivity.json',process_sensitivity(study,paths),True)
    final=study/'final_complete.json'
    if final.exists():
        done=io.read(final)
        if done['protocol_sha256']!=io.sha(study/'protocol.json') or done['development_promotion_sha256']!=io.sha(study/'development_promotion.json'):
            raise RuntimeError('completed study marker changed')
        for (suite,arm),_ in paths.items():
            report=study/f'reports/{suite}/{arm}';binding=io.read(report/'final_complete.json')
            for key,file in [('summary_sha256',report/'summary.json'),
                             ('official_sha256',study/f'scoring/{suite}/{arm}/official.official.json'),
                             ('binding_sha256',study/f'scoring/{suite}/{arm}/official.binding.json')]:
                if binding[key]!=io.sha(file):raise RuntimeError('completed score evidence changed')
        return
    # Original generation records are never changed. Deterministic report
    # writes may resume only when existing immutable JSON is exactly equal.
    io.write=write_or_verify
    try:native.finish(study,paths)
    finally:io.write=ORIGINAL_WRITE


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','run'],required=True)
    a.add_argument('--study',type=Path,required=True);a.add_argument('--parent',type=Path);a.add_argument('--qualification',type=Path)
    args=a.parse_args()
    if args.stage=='prepare':prepare(args.study.resolve(),args.parent.resolve(),args.qualification.resolve())
    else:run(args.study.resolve())
