"""Frozen repeatability diagnosis; never changes native sampling or its seed.

Each job deliberately repeats one identical cell twice after the usual warmup.
This isolates repeatability before attributing inactive-variant differences.
It uses the exact execution source and environment of native development.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_value_decision_replay as io
from scripts import v31_value_native_study as native


def repeated_outputs_identical(rows,fields):
    groups={}
    for r in rows:groups.setdefault(native.base.key(r),[]).append(r)
    return all(all(all(r[k]==group[0][k] for k in fields) for r in group) for group in groups.values())


def prepare(study,parent,python_hash_seed=None,trace=False,layer_trace=False,cublas_workspace_config=None,component_trace=False,control_history=False):
    if study.exists():raise RuntimeError('use a new immutable diagnostic attempt')
    p=native.verify(parent)
    cells=io.read(parent/'private/schedules/aime26/cells.json')
    cell=next(c for c in cells if c['index']==15 and c['seed']==44)
    planned=[cell,cell]
    if control_history:
        planned=[next(c for c in cells if c['index']==index and c['seed']==42)
                 for index in (0,10,20) for _ in range(6)]
    study.mkdir(parents=True);(study/'private').mkdir()
    trace_source=io.ROOT/'scripts/v31_native_rng_trace_worker.py'
    if trace:shutil.copyfile(trace_source,study/'private/trace_worker.py')
    io.write(study/'private/cells.json',planned,True)
    shutil.copyfile(__file__,study/'coordinator_source.py')
    jobs=[dict(name='original_repeat001',arm='current_v31_control'),
          dict(name='original_repeat002',arm='current_v31_control'),
          dict(name='refinement_repeat001',arm='full_swap8')]
    if python_hash_seed is not None:
        jobs.append(dict(name='refinement_repeat002',arm='full_swap8'))
    if trace or control_history:jobs=jobs[:2]  # Original V31 only.
    protocol=dict(schema='native_noop_repeatability_v1',parent=str(parent),
        parent_protocol_sha256=io.sha(parent/'protocol.json'),source_commit=p['source_commit'],
        coordinator_sha256=io.sha(study/'coordinator_source.py'),
        cells_sha256=io.sha(study/'private/cells.json'),suite='aime26',
        jobs=jobs,python_hash_seed=python_hash_seed,trace=trace,layer_trace=layer_trace,
        cublas_workspace_config=cublas_workspace_config,
        component_trace=component_trace,
        control_history=control_history,
        trace_source_sha256=io.sha(trace_source) if trace else None,
        scope=('control-only history: AIME indices0,10,20 seed42 repeated six times each in order; no task-accuracy inference'
               if control_history else 'one previously examined inactive cell, repeated with identical request seed; no task-accuracy inference'),
        timing='paused; diagnostic only')
    io.write(study/'protocol.json',protocol,True)


def run(study):
    p=io.read(study/'protocol.json');parent=Path(p['parent'])
    if p['coordinator_sha256']!=io.sha(__file__):raise RuntimeError('execute the frozen coordinator')
    if p['parent_protocol_sha256']!=io.sha(parent/'protocol.json'):raise RuntimeError('parent protocol drift')
    if p['cells_sha256']!=io.sha(study/'private/cells.json'):raise RuntimeError('diagnostic cells drift')
    if p.get('trace') and p['trace_source_sha256']!=io.sha(study/'private/trace_worker.py'):
        raise RuntimeError('diagnostic observer source changed')
    source=native.verify(parent);results=[];planned=io.read(study/'private/cells.json')
    execution=parent/'private/execution'
    for job in p['jobs']:
        family=study/job['name'];markers=list(family.glob('attempt*/attempt_complete.json'))
        if markers:
            if len(markers)!=1:raise RuntimeError('ambiguous completed diagnostic')
            marker=markers[0];leaf=marker.parent;proof=io.read(marker)
            if proof['records_sha256']!=io.sha(leaf/'records.jsonl'):raise RuntimeError('diagnostic records drift')
        else:
            ids=[int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()]
            leaf=family/f'attempt{max(ids,default=0)+1:03d}';(leaf/'private').mkdir(parents=True)
            native.base.wait_idle()
            if native.base.runtime_identity()!=source['runtime']:raise RuntimeError('runtime changed')
            arm=job['arm'];env=native.base.env_for(parent,'aime26',arm)
            env.update(STUDY_ARM=arm,FROZEN_SOURCE_SHA256=io.sha(parent/'receipts/source_fingerprints.json'))
            if p.get('python_hash_seed') is not None:env['PYTHONHASHSEED']=str(p['python_hash_seed'])
            if p.get('cublas_workspace_config') is not None:
                env['CUBLAS_WORKSPACE_CONFIG']=p['cublas_workspace_config']
            worker='v31_vllm_paired_bench.py'
            if arm in native.MODES:
                worker='v31_value_refinement_worker.py'
                env.update(VALUE_REFINEMENT=arm,VALUE_REFINEMENT_GATE=str(parent/'decision_promotion.json'))
            worker_path=execution/'scripts'/worker
            if p.get('trace'):
                env.update(NATIVE_RNG_TRACE_DIR=str(leaf/'private/trace'),NATIVE_RNG_BASE_WORKER=str(worker_path))
                if p.get('layer_trace'):env['NATIVE_LAYER_TRACE']='1'
                if p.get('component_trace'):env['NATIVE_COMPONENT_TRACE']='1'
                worker_path=study/'private/trace_worker.py'
            cmd=[native.base.PYTHON,str(worker_path),str(io.MODEL),
                 str(parent/'private/pools/aime26'),str(study/'private/cells.json'),
                 str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
            io.write(study/'state.json',dict(stage='running',job=job['name'],attempt=str(leaf.relative_to(study)),timing='paused'))
            start=time.monotonic()
            with (leaf/'private/worker.log').open('w') as log:
                proc=subprocess.run(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
            rows=native.base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
            errors=[]
            if proc.returncode or len(rows)!=len(planned):errors.append('worker or repeated-cell coverage failure')
            if len(rows)==len(planned):
                seeds={}
                for r,cell in zip(rows,planned):
                    key=native.base.key(r)
                    if key in seeds and seeds[key]!=r['rng_seed']:errors.append('repeated request seed changed')
                    seeds[key]=r['rng_seed']
                    errors+=native.validate([r],'aime26',arm,[cell])
                    if r['cuda_graph_captures']:errors.append('unexpected request capture')
            proof=dict(status='failed' if errors else 'complete',errors=errors,
                returncode=proc.returncode,gpu_reservation_s=time.monotonic()-start,
                records_sha256=io.sha(leaf/'records.jsonl') if rows else None,
                private_sha256=io.sha(leaf/'private/generation.private.jsonl') if rows else None)
            io.write(leaf/'receipt.json',proof,True)
            if errors:raise RuntimeError('preserved diagnostic worker failure')
            io.write(leaf/'attempt_complete.json',proof,True)
        rows=native.base.jsonl(leaf/'records.jsonl')
        fields=('output_hash','denoise_forwards','canvases','output_tokens','rng_seed')
        identical=repeated_outputs_identical(rows,fields)
        results.append(dict(job=job['name'],arm=job['arm'],records=[{k:r[k] for k in fields} for r in rows],
            identical_repeated_output=identical,
            cell_order=[dict(index=r['index'],seed=r['panel_seed']) for r in rows],
            refinement_work=[r['receipts']['adapter'].get('value_refinement_work',{}) for r in rows]))
        print(json.dumps({'event':'diagnostic_job_complete','job':job['name'],
                          'identical_repeated_output':results[-1]['identical_repeated_output']}),flush=True)
    reference=results[0]['records']
    summary=dict(status='complete',scope=p['scope'],results=results,
        all_process_outputs_identical=all(job['records']==reference for job in results),
        python_hash_seed=p.get('python_hash_seed'),
        protocol_sha256=io.sha(study/'protocol.json'),timing='paused')
    io.write(study/'summary.json',summary,True)
    io.write(study/'final_complete.json',dict(status='complete',summary_sha256=io.sha(study/'summary.json')),True)
    io.write(study/'state.json',dict(stage='complete',timing='paused'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','run'],required=True)
    a.add_argument('--study',type=Path,required=True);a.add_argument('--parent',type=Path)
    a.add_argument('--python-hash-seed',type=int)
    a.add_argument('--trace',action='store_true')
    a.add_argument('--layer-trace',action='store_true')
    a.add_argument('--cublas-workspace-config',choices=[':4096:8',':16:8'])
    a.add_argument('--component-trace',action='store_true')
    a.add_argument('--control-history',action='store_true')
    args=a.parse_args();study=args.study.resolve()
    if args.layer_trace and not args.trace:a.error('--layer-trace requires --trace')
    if args.component_trace and not args.layer_trace:a.error('--component-trace requires --layer-trace')
    if args.control_history and args.trace:a.error('--control-history uses unobserved original V31 only')
    if args.stage=='prepare':prepare(study,args.parent.resolve(),args.python_hash_seed,args.trace,args.layer_trace,args.cublas_workspace_config,args.component_trace,args.control_history)
    else:run(study)
