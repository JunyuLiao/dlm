"""Replicated common-engine native evaluation; execution drift is retained.

Separate frozen study: never reclassifies a failed exact-parity attempt.
Method code, native sampling, fixed K, and official scorers are unchanged.
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
from scripts.v31_value_process_matched_study import validate_group,write_or_verify

OVERLAY=['scripts/v31_value_replicated_study.py','tests/test_v31_value_replicated_study.py']
BEFORE='control_before';AFTER='control_after'


def schedule(cells,modes,warm_cells,repeat):
    if repeat not in (1,2):raise ValueError('exactly two engine replicates')
    rows=[];arms=[CONTROL];groups=[]
    for cell in warm_cells:
        rows.extend([cell]*len(modes));arms.extend(modes)
    warm_count=len(rows)
    for cell in cells:
        shift=(int(cell['index'])+int(cell['seed']))%len(modes)
        order=modes[shift:]+modes[:shift]
        if repeat==2:order=order[::-1]
        group=[CONTROL]+order+[CONTROL]
        groups.append(dict(start=len(rows),end=len(rows)+len(group),arms=group,
                           labels=[BEFORE]+order+[AFTER]))
        rows.extend([cell]*len(group));arms.extend(group)
    return rows,arms,groups,warm_count


def drift(rows,private,arms):
    errors=validate_group(rows,arms)
    if private[0]['completion']!=private[-1]['completion']:errors.append('control decoded text drift')
    for row,text,arm in zip(rows[1:-1],private[1:-1],arms[1:-1]):
        work=row['receipts']['adapter'].get('value_refinement_work',{})
        if not (work.get('swapped_blocks',0) or work.get('corrected_calls',0)) and text['completion']!=private[0]['completion']:
            errors.append('inactive candidate decoded text drift: '+arm)
    return sorted(set(errors))


def prepare(study,parent,factor):
    if study.exists():raise RuntimeError('preserve prior study attempts')
    old=native.verify(parent)
    done=io.read(factor/'final_complete.json');fs=io.read(factor/'summary.json')
    if done['summary_sha256']!=io.sha(factor/'summary.json') or fs['status']!='complete' or len(fs['results'])!=4:
        raise RuntimeError('complete four-job warmup factor required')
    if io.read(factor/'protocol.json')['parent_protocol_sha256']!=io.sha(parent/'protocol.json'):
        raise RuntimeError('warmup factor source differs')
    execution=study/'private/execution';execution.parent.mkdir(parents=True)
    shutil.copytree(parent/'private/execution',execution,ignore=shutil.ignore_patterns('__pycache__'))
    for rel in OVERLAY:shutil.copyfile(io.ROOT/rel,execution/rel)
    for name in ('pools','schedules'):shutil.copytree(parent/'private'/name,study/'private'/name)
    shutil.copyfile(parent/'private/scorer_location.json',study/'private/scorer_location.json')
    for name in ('decision_promotion.json','h100_tests.json','model_source_fingerprints.json'):
        shutil.copyfile(parent/name,study/name)
    evidence=study/'receipts/warmup_factor';evidence.mkdir(parents=True)
    for name in ('config.json','summary.json','final_complete.json'):shutil.copyfile(factor/name,evidence/name)
    p=json.loads(json.dumps(old));p['schema']='value_native_replicated_v1'
    p['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=io.ROOT,text=True).strip()
    p['deployment_recipe']=dict(base_source_commit=old['source_commit'],base_protocol_sha256=io.sha(parent/'protocol.json'),
        overlay_commit=p['source_commit'],overlay_files=OVERLAY,description='qualified parent execution plus explicit coordinator/test overlay; method code unchanged')
    p['comparison_design']=dict(repeats=2,controls_per_engine=2,
        estimand='per-cell candidate minus mean of two control bookends, averaged over two fresh engines; no answer ensemble',
        ordering='Latin rotation then reversed candidate order in the second engine',
        warmup='all four methods, long AIME factor input or both qualified workload smoke inputs; chosen before factor result',
        drift='retain and score every structurally valid outcome; exact parity is an audit outcome, not an inclusion rule',
        uncertainty='separate problem/task and paired engine-shard block bootstrap intervals; 20000 draws, seed1729',
        references='fresh dense FULL, PIECEWISE and all-kept once per cell; unresolved process variation limits fine comparisons',
        factor_summary_sha256=io.sha(evidence/'summary.json'),timing='paused')
    for suite in p['suites']:
        folder=study/f'private/schedules/{suite}';cells=io.read(folder/'cells.json')
        warm=[next(c for c in cells if c['index']==10 and c['seed']==42)] if suite=='aime26' else io.read(folder/'smoke.json')
        io.write(folder/'warmup.json',warm,True)
    p['execution_fingerprints']={str(f.relative_to(execution)):io.sha(f) for f in execution.rglob('*') if f.is_file()}
    inputs={}
    for folder in (study/'private/pools',study/'private/schedules',evidence):
        inputs.update({str(f.relative_to(study)):io.sha(f) for f in folder.rglob('*') if f.is_file()})
    for name in ('decision_promotion.json','h100_tests.json','private/scorer_location.json'):inputs[name]=io.sha(study/name)
    p['input_fingerprints']=inputs
    io.write(study/'protocol.json',p,True)
    io.write(study/'receipts/source_fingerprints.json',dict(source_commit=p['source_commit'],deployment_recipe=p['deployment_recipe'],execution_fingerprints=p['execution_fingerprints']),True)
    native.verify(study)


def verify_leaf(study,leaf):
    proof=io.read(leaf/'attempt_complete.json')
    if proof['protocol_sha256']!=io.sha(study/'protocol.json'):raise RuntimeError('completion protocol changed')
    for rel,digest in proof['files_sha256'].items():
        if io.sha(leaf/rel)!=digest:raise RuntimeError('completed raw or exported record changed')
    return proof


def shared(study,suite,shard,repeat,qual=False):
    p=native.verify(study);modes=[x for x in p['arms'] if x not in native.CONTROLS]
    family=study/('shared_qualification' if qual else 'shared_runs')/suite/shard.stem/f'repeat{repeat:03d}'
    markers=list(family.glob('attempt*/attempt_complete.json'))
    if markers:
        if len(markers)!=1:raise RuntimeError('ambiguous engine completion')
        leaf=markers[0].parent;proof=verify_leaf(study,leaf)
        return leaf,proof
    ordinal=max([int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()],default=0)+1
    leaf=family/f'attempt{ordinal:03d}';(leaf/'private').mkdir(parents=True)
    planned=io.read(shard);warm_cells=io.read(study/f'private/schedules/{suite}/warmup.json')
    cells,arms,groups,warm_count=schedule(planned,modes,warm_cells,repeat)
    io.write(leaf/'private/cells.json',cells,True);io.write(leaf/'private/arms.json',arms,True)
    io.write(leaf/'config.json',dict(protocol_sha256=io.sha(study/'protocol.json'),repeat=repeat,
        groups=groups,warmup_requests=warm_count,cells_sha256=io.sha(leaf/'private/cells.json'),arms_sha256=io.sha(leaf/'private/arms.json')),True)
    native.base.wait_idle()
    if native.base.runtime_identity()!=p['runtime']:raise RuntimeError('runtime changed')
    execution=study/'private/execution';env=native.base.env_for(study,suite,CONTROL)
    env.update(STUDY_ARM='replicated_common_engine',PYTHONHASHSEED='0',
        FROZEN_SOURCE_SHA256=io.sha(study/'receipts/source_fingerprints.json'),
        VALUE_REFINEMENT_GATE=str(study/'decision_promotion.json'),
        NATIVE_INTERLEAVED_ARMS=str(leaf/'private/arms.json'),
        NATIVE_INTERLEAVED_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
    command=[native.base.PYTHON,str(execution/'scripts/v31_value_interleaved_worker.py'),str(io.MODEL),
        str(study/f'private/pools/{suite}'),str(leaf/'private/cells.json'),str(leaf/'records.jsonl'),
        str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
    io.write(study/'state.json',dict(stage='qualification' if qual else 'replicated_panel',suite=suite,
        attempt=str(leaf.relative_to(study)),timing='paused'))
    start=time.monotonic()
    with (leaf/'private/worker.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT);last=start
        while proc.poll() is None:
            if time.monotonic()-last>=1800:
                print(dict(event='progress',suite=suite,attempt=str(leaf.relative_to(study))),flush=True);last=time.monotonic()
            time.sleep(5)
    rows=native.base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    private=native.base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
    errors=[];audits=[];warm_work={arm:defaultdict(int) for arm in modes}
    if proc.returncode or len(rows)!=len(cells) or len(private)!=len(cells):errors.append('worker or coverage failure')
    if len(rows)==len(cells) and len(private)==len(cells):
        for i,(row,arm,cell) in enumerate(zip(rows,arms[1:],cells)):
            receipt=row['receipts']['adapter']
            if receipt.get('comparison_arm')!=arm or receipt.get('comparison_request_index')!=i+1:errors.append('live arm mismatch')
            if row.get('study_arm')!='replicated_common_engine':errors.append('raw study label mismatch')
            errors+=native.validate([dict(row,study_arm=arm)],suite,arm,[cell])
            if row['cuda_graph_captures']:errors.append('measured graph capture')
            if i<warm_count:
                for name,value in receipt.get('value_refinement_work',{}).items():warm_work[arm][name]+=value
        for arm,work in warm_work.items():
            if not work.get('anchor_events',0) or (not arm.endswith('swap8') and not work.get('corrected_calls',0)):
                errors.append('warmup did not exercise '+arm)
        for group in groups:
            lo,hi=group['start'],group['end'];rr=rows[lo:hi]
            audits.append(dict(index=rr[0]['index'],seed=rr[0]['panel_seed'],
                drift=drift(rr,private[lo:hi],group['arms']),
                control_N=[rr[0]['denoise_forwards'],rr[-1]['denoise_forwards']]))
    proof=dict(status='failed' if errors else 'complete',structural_errors=sorted(set(errors)),returncode=proc.returncode,
        protocol_sha256=io.sha(study/'protocol.json'),gpu_reservation_s=time.monotonic()-start,
        repeat=repeat,groups=audits,warmup_work={k:dict(v) for k,v in warm_work.items()},
        parity_policy='all structural-valid outcomes retained, including drift; no parity-based retries')
    io.write(leaf/'receipt.json',proof,True)
    if errors:raise RuntimeError('preserved structurally invalid replicated engine')
    native.verify(study)
    exported=defaultdict(list);public=defaultdict(list)
    for group in groups:
        for i,label in zip(range(group['start'],group['end']),group['labels']):
            name=f'r{repeat}_{label}'
            exported[name].append(private[i])
            public[name].append(dict(rows[i],study_arm=arms[i+1],comparison_engine=str(leaf.relative_to(study)),
                                    comparison_shard=shard.stem,comparison_repeat=repeat,comparison_raw_ordinal=i))
    files=['records.jsonl','private/generation.private.jsonl','receipt.json','config.json']
    for name in exported:
        for rel,values in [(f'exports/{name}/records.jsonl',public[name]),(f'exports/{name}/private/run_{name}.private.jsonl',exported[name])]:
            path=leaf/rel;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as f:
                for row in values:f.write(json.dumps(row)+'\n')
            files.append(rel)
    proof['files_sha256']={f:io.sha(leaf/f) for f in files}
    io.write(leaf/'attempt_complete.json',proof,True)
    print(dict(event='replicated_engine_complete',suite=suite,shard=shard.stem,repeat=repeat,
        groups=len(audits),drifting_groups=sum(bool(g['drift']) for g in audits)),flush=True)
    return leaf,proof


def official(study,suite,paths,qual=False):
    """All repeat/position labels score independently, with unchanged native keys."""
    execution=study/'private/execution';pool=study/f'private/pools/{suite}'
    family=study/('qualification_scoring' if qual else 'scoring')/suite
    markers=list(family.glob('attempt*/scoring_complete.json'))
    if len(markers)>1:raise RuntimeError('ambiguous scoring completion')
    if markers:dest=markers[0].parent
    else:
        ordinal=max([int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()],default=0)+1
        dest=family/f'attempt{ordinal:03d}';dest.mkdir(parents=True)
    prefix=dest/'official';marker=dest/'scoring_complete.json'
    files=[f for leaves in paths.values() for leaf in leaves for f in sorted((leaf/'private').glob('*.private.jsonl'))]
    inputs={str(f.relative_to(study)):io.sha(f) for f in files}
    if marker.exists():
        proof=io.read(marker)
        if proof['inputs']!=inputs or proof['protocol_sha256']!=io.sha(study/'protocol.json'):raise RuntimeError('scored inputs changed')
        for name,digest in proof['outputs'].items():
            if io.sha(dest/name)!=digest:raise RuntimeError('official scoring changed')
    else:
        if suite=='ruler128k':
            ruler=io.read(study/'private/scorer_location.json')['ruler']
            command=[native.base.PYTHON,str(execution/'scripts/v31_score_ruler.py'),str(prefix),ruler,str(pool),str(pool)]
        elif suite=='longbench_v2':command=[native.base.PYTHON,str(execution/'scripts/v31_score_longbench_official.py'),str(prefix),str(pool),str(pool)]
        elif suite=='aime26':command=[native.base.PYTHON,str(execution/'scripts/v31_score_aime.py'),str(prefix),str(pool),str(pool/'aime2026_answers.jsonl')]
        else:command=[native.base.PYTHON,str(execution/'scripts/v31_score_math500_bound.py'),str(prefix),str(pool)]
        command+=['--cells',str(study/f'private/schedules/{suite}'/('smoke.json' if qual else 'cells.json')),*map(str,files)]
        env=dict(os.environ,PYTHONPATH=f'{execution}/src:{execution}',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
        with (dest/'scorer.log').open('w') as log:subprocess.run(command,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        proof=dict(protocol_sha256=io.sha(study/'protocol.json'),inputs=inputs,attempt=str(dest.relative_to(study)),
            outputs={p.name:io.sha(p) for p in dest.glob('official.*.json')})
    summary=io.read(str(prefix)+'.summary.json');scores=io.read(str(prefix)+'.official.json');bindings=io.read(str(prefix)+'.binding.json')
    expected=2 if qual else native.verify(study)['suites'][suite]['count']
    if set(scores)!=set(paths):raise RuntimeError('official arm coverage differs')
    for label in paths:
        cov=summary['coverage'][label]
        if cov['scored']!=expected or cov['missing'] or cov['dropped']:raise RuntimeError('official coverage failed')
    native.official_rules.check_comparable(bindings,list(paths),set(next(iter(scores.values()))))
    write_or_verify(marker,proof,True)
    return scores,proof


def clustered(values,groups):
    """Equal-size problem/task or engine blocks; refuse estimand-changing sizes."""
    collected=defaultdict(list)
    for key in sorted(values):collected[groups[key]].append(values[key])
    if len({len(v) for v in collected.values()})!=1:raise RuntimeError('unequal clusters require explicit weighted estimand')
    means=np.array([np.mean(collected[key],axis=0) for key in sorted(collected,key=str)]);rng=np.random.default_rng(1729)
    boot=means[rng.integers(len(means),size=(20000,len(means)))].mean(1)
    return dict(delta=list(map(float,means.mean(0))),ci95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),clusters=len(means))


def paired_values(scores,rows,mode):
    keys=set(scores[f'r1_{mode}']);values={};replicas={}
    for rep in (1,2):
        v=scores[f'r{rep}_{mode}'];a=scores[f'r{rep}_{BEFORE}'];b=scores[f'r{rep}_{AFTER}']
        if set(v)!=keys or set(a)!=keys or set(b)!=keys:raise RuntimeError('comparison key mismatch')
        out={}
        for key in sorted(keys):
            vr=rows[f'r{rep}_{mode}'][key];ar=rows[f'r{rep}_{BEFORE}'][key];br=rows[f'r{rep}_{AFTER}'][key]
            out[key]=[float(v[key])-(float(a[key])+float(b[key]))/2]+[
                vr[f]-(ar[f]+br[f])/2 for f in ('denoise_forwards','canvases','output_tokens')]
        replicas[rep]=out
    for key in sorted(keys):values[key]=np.mean([replicas[1][key],replicas[2][key]],axis=0).tolist()
    return values,replicas


def report_suite(study,suite,paths,engine_proofs):
    p=native.verify(study);scores,score_proof=official(study,suite,paths)
    rows={label:{native.base.key(r):r for leaf in leaves for r in native.base.jsonl(leaf/'records.jsonl')} for label,leaves in paths.items()}
    modes=[a for a in p['arms'] if a not in native.CONTROLS];keys=set(next(iter(scores.values())))
    if any(set(r)!=keys for r in rows.values()):raise RuntimeError('report cohort mismatch')
    scale=1 if suite=='ruler128k' else 100
    tasks={r['index']:r['task'] for r in io.read(study/'private/pools/ruler128k/ruler128k_v33ofc_subset_rowinfo.json')['rows']} if suite=='ruler128k' else {}
    clusters={k:tasks[int(k.split('|')[1])] if suite=='ruler128k' else tuple(k.split('|')[:2]) for k in keys}
    engines={k:rows[f'r1_{BEFORE}'][k]['comparison_shard'] for k in keys}
    summaries={};per_repeat={}
    for arm in p['arms']:
        labels=([f'r{r}_{pos}' for r in (1,2) for pos in (BEFORE,AFTER)] if arm==CONTROL else
                [f'r{r}_{arm}' for r in (1,2)] if arm in modes else [arm])
        rr=[row for label in labels for row in rows[label].values()];work=native.base.aggregate(rr);extra=defaultdict(int)
        local=sum(r['receipts']['adapter']['native_local_eligible_rectangles'] for r in rr)
        work.update(local_eligible_tiles=local,local_kept_tiles=local,overall_kept_rectangles=work['overall_eligible_rectangles']-work['overall_skipped_rectangles'],
                    overall_denominator='decode GLOBAL full-support plus LOCAL window-intersecting H x Q128 x KV64 rectangles',
                    mean_global_kept_tiles=work['global_kept_tiles']/len(rr),mean_global_eligible_tiles=work['global_eligible_tiles']/len(rr),
                    audit_S_per_N=work['audit_decode_s']/work['N'],latency_inference='instrumented audit only; no clean speedup')
        for row in rr:
            for name,value in row['receipts']['adapter'].get('value_refinement_work',{}).items():extra[name]+=value
        summaries[arm]=dict(score_points=scale*float(np.mean([float(v) for label in labels for v in scores[label].values()])),
            problem_trials=len(rr),distinct_cells=len(keys),work=work,extra_refinement_work=dict(extra),
            p95_N=float(np.quantile([r['denoise_forwards'] for r in rr],.95)),
            cap_rate=float(np.mean([native.official_rules.finish_class(r['finish_reason'])=='capped' for r in rr])))
        per_repeat[arm]=[{ 'label':label,'score_points':scale*float(np.mean(list(scores[label].values()))),
                          'mean_N':float(np.mean([r['denoise_forwards'] for r in rows[label].values()]))} for label in labels]
    comparisons=[]
    for mode in modes:
        values,replicas=paired_values(scores,rows,mode)
        for value in values.values():value[0]*=scale
        item=clustered(values,clusters);engine=clustered(values,engines)
        rep_delta={str(rep):np.mean(list(v.values()),axis=0).tolist() for rep,v in replicas.items()}
        for v in rep_delta.values():v[0]*=scale
        c=summaries[CONTROL];v=summaries[mode]
        passed=(v['score_points']>=c['score_points'] and v['work']['mean_N']<=c['work']['mean_N'] and
                v['p95_N']<=1.05*c['p95_N'] and v['cap_rate']<=1.05*c['cap_rate'])
        comparisons.append(dict(arm=mode,passes_screen=bool(passed),metrics=['score_points','N','C','T'],
            problem_or_task_cluster=item,paired_engine_block=engine,replicate_delta=rep_delta))
    noise={};cross_engine={};bookend_hash=0;cross_hash=0
    for key in sorted(keys):
        dv=[]
        for rep in (1,2):
            a=f'r{rep}_{BEFORE}';b=f'r{rep}_{AFTER}'
            dv.append([scale*(float(scores[b][key])-float(scores[a][key]))]+[rows[b][key][f]-rows[a][key][f] for f in ('denoise_forwards','canvases','output_tokens')])
            bookend_hash+=rows[a][key]['output_hash']!=rows[b][key]['output_hash']
        noise[key]=np.mean(dv,axis=0).tolist()
        cross_engine[key]=[scale*(sum(float(scores[f'r2_{x}'][key]) for x in (BEFORE,AFTER))-sum(float(scores[f'r1_{x}'][key]) for x in (BEFORE,AFTER)))/2]+[
            (sum(rows[f'r2_{x}'][key][f] for x in (BEFORE,AFTER))-sum(rows[f'r1_{x}'][key][f] for x in (BEFORE,AFTER)))/2 for f in ('denoise_forwards','canvases','output_tokens')]
        cross_hash+=rows[f'r1_{BEFORE}'][key]['output_hash']!=rows[f'r2_{BEFORE}'][key]['output_hash']
    result=dict(status='complete_workload',suite=suite,protocol_sha256=io.sha(study/'protocol.json'),summaries=summaries,
        separate_repeat_positions=per_repeat,comparisons=comparisons,
        control_noise=dict(bookend_hash_mismatches=bookend_hash,bookend_pairs=2*len(keys),
            cross_engine_first_control_hash_mismatches=cross_hash,cross_engine_pairs=len(keys),
            bookend_delta=clustered(noise,clusters),between_engine_delta=clustered(cross_engine,clusters)),
        engine_audits=engine_proofs,official_scoring=score_proof,
        inference='exploratory repeated evaluation, not pass@2; separate item/task and engine-block intervals are not a joint confidence region',timing='paused')
    folder=study/'reports'/suite
    write_or_verify(folder/'summary.json',result,True)
    write_or_verify(folder/'final_complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),summary_sha256=io.sha(folder/'summary.json')),True)
    lines=[f'# Replicated {suite} development evaluation','',result['inference']+'. Clean timing remains paused.','',
        '| Method | Score | N/trial | N/canvas | C/trial | T/trial | GLOBAL kept/trial |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in summaries.items():
        w=r['work'];lines.append(f"| {arm} | {r['score_points']:.3f} | {w['mean_N']:.3f} | {w['N_per_C']:.3f} | {w['mean_C']:.3f} | {w['mean_T']:.1f} | {w['mean_global_kept_tiles']:.1f} |")
    content='\n'.join(lines)+'\n';path=folder/'summary.md'
    if path.exists() and path.read_text()!=content:raise RuntimeError('immutable workload summary changed')
    if not path.exists():path.write_text(content)
    print(dict(event='replicated_workload_complete',suite=suite,control_bookend_mismatches=bookend_hash),flush=True)
    return result


def run(study):
    p=native.verify(study)
    if io.sha(__file__)!=p['execution_fingerprints'][OVERLAY[0]]:raise RuntimeError('execute frozen coordinator')
    refs=[a for a in native.CONTROLS if a!=CONTROL];results={}
    for qual in (True,False):
        for suite,cfg in p['suites'].items():
            paths=defaultdict(list);proofs=[]
            shards=[study/f'private/schedules/{suite}/smoke.json'] if qual else [study/f'private/schedules/{suite}/shard{i:03d}.json' for i in range(1,cfg['shards']+1)]
            for shard in shards:
                # Reverse engine replicate launch order on alternate shards.
                reps=(1,2) if qual or int(shard.stem[5:])%2 else (2,1)
                for rep in reps:
                    leaf,proof=shared(study,suite,shard,rep,qual)
                    proofs.append(dict(attempt=str(leaf.relative_to(study)),groups=proof['groups'],repeat=rep,
                        gpu_reservation_s=proof['gpu_reservation_s']))
                    for folder in sorted((leaf/'exports').iterdir()):paths[folder.name].append(folder)
                for arm in refs:
                    paths[arm].append(native.run_shard(study,suite,arm,shard,qual))
            if qual:
                official(study,suite,paths,True)
                write_or_verify(study/f'qualification/{suite}/qualification_complete.json',dict(status='passed',
                    protocol_sha256=io.sha(study/'protocol.json'),engine_audits=proofs,
                    scope='structural receipt, fixed K, source, scope, sparse warmup and official scorer; drift retained'),True)
                print(dict(event='replicated_suite_qualified',suite=suite),flush=True)
            else:results[suite]=report_suite(study,suite,paths,proofs)
    candidates=[]
    for mode in [a for a in p['arms'] if a not in native.CONTROLS]:
        comps=[next(c for c in r['comparisons'] if c['arm']==mode) for r in results.values()]
        if all(c['passes_screen'] for c in comps) and any(c['problem_or_task_cluster']['delta'][0]>0 for c in comps):candidates.append(mode)
    write_or_verify(study/'development_promotion.json',dict(status='complete',full503_candidates=candidates,
        inference='exploratory screen only; uncertainty and repeat consistency must be reviewed; separate full503 protocol required',timing='paused'),True)
    write_or_verify(study/'final_complete.json',dict(status='complete',protocol_sha256=io.sha(study/'protocol.json'),
        report_sha256={suite:io.sha(study/f'reports/{suite}/summary.json') for suite in results},
        development_promotion_sha256=io.sha(study/'development_promotion.json'),timing='paused'),True)
    io.write(study/'state.json',dict(stage='complete',full503_candidates=candidates,timing='paused'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','run'],required=True)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--parent',type=Path);parser.add_argument('--factor',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.study.resolve(),args.parent.resolve(),args.factor.resolve())
    else:
        try:run(args.study.resolve())
        except Exception:
            io.write(args.study.resolve()/'state.json',dict(stage='failed',resumable=True,
                policy='preserve failed attempts; no output-parity exclusion',timing='paused'))
            raise
