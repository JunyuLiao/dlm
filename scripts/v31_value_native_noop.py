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


def prepare(study,parent,python_hash_seed=None,trace=False):
    if study.exists():raise RuntimeError('use a new immutable diagnostic attempt')
    p=native.verify(parent)
    cells=io.read(parent/'private/schedules/aime26/cells.json')
    cell=next(c for c in cells if c['index']==15 and c['seed']==44)
    study.mkdir(parents=True);(study/'private').mkdir()
    trace_source=io.ROOT/'scripts/v31_native_rng_trace_worker.py'
    if trace:shutil.copyfile(trace_source,study/'private/trace_worker.py')
    io.write(study/'private/cells.json',[cell,cell],True)
    shutil.copyfile(__file__,study/'coordinator_source.py')
    jobs=[dict(name='original_repeat001',arm='current_v31_control'),
          dict(name='original_repeat002',arm='current_v31_control'),
          dict(name='refinement_repeat001',arm='full_swap8')]
    if python_hash_seed is not None:
        jobs.append(dict(name='refinement_repeat002',arm='full_swap8'))
    protocol=dict(schema='native_noop_repeatability_v1',parent=str(parent),
        parent_protocol_sha256=io.sha(parent/'protocol.json'),source_commit=p['source_commit'],
        coordinator_sha256=io.sha(study/'coordinator_source.py'),
        cells_sha256=io.sha(study/'private/cells.json'),suite='aime26',
        jobs=jobs,python_hash_seed=python_hash_seed,trace=trace,
        trace_source_sha256=io.sha(trace_source) if trace else None,
        scope='one previously examined inactive cell, repeated with identical request seed; no task-accuracy inference',
        timing='paused; diagnostic only')
    io.write(study/'protocol.json',protocol,True)


def run(study):
    p=io.read(study/'protocol.json');parent=Path(p['parent'])
    if p['coordinator_sha256']!=io.sha(__file__):raise RuntimeError('execute the frozen coordinator')
    if p['parent_protocol_sha256']!=io.sha(parent/'protocol.json'):raise RuntimeError('parent protocol drift')
    if p['cells_sha256']!=io.sha(study/'private/cells.json'):raise RuntimeError('diagnostic cells drift')
    if p.get('trace') and p['trace_source_sha256']!=io.sha(study/'private/trace_worker.py'):
        raise RuntimeError('diagnostic observer source changed')
    source=native.verify(parent);results=[]
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
            worker='v31_vllm_paired_bench.py'
            if arm in native.MODES:
                worker='v31_value_refinement_worker.py'
                env.update(VALUE_REFINEMENT=arm,VALUE_REFINEMENT_GATE=str(parent/'decision_promotion.json'))
            worker_path=execution/'scripts'/worker
            if p.get('trace'):
                env.update(NATIVE_RNG_TRACE_DIR=str(leaf/'private/trace'),NATIVE_RNG_BASE_WORKER=str(worker_path))
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
            if proc.returncode or len(rows)!=2:errors.append('worker or repeated-cell coverage failure')
            if len(rows)==2:
                if native.base.key(rows[0])!=native.base.key(rows[1]):errors.append('repeated cell changed')
                if rows[0]['rng_seed']!=rows[1]['rng_seed']:errors.append('repeated request seed changed')
                for r in rows:
                    errors+=native.validate([r],'aime26',arm,[io.read(study/'private/cells.json')[0]])
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
        results.append(dict(job=job['name'],arm=job['arm'],records=[{k:r[k] for k in fields} for r in rows],
            identical_repeated_output=all(rows[0][k]==rows[1][k] for k in fields),
            refinement_work=[r['receipts']['adapter'].get('value_refinement_work',{}) for r in rows]))
        print(json.dumps({'event':'diagnostic_job_complete','job':job['name'],
                          'identical_repeated_output':results[-1]['identical_repeated_output']}),flush=True)
    reference=results[0]['records'][0]
    summary=dict(status='complete',scope=p['scope'],results=results,
        all_process_outputs_identical=all(r==reference for job in results for r in job['records']),
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
    args=a.parse_args();study=args.study.resolve()
    if args.stage=='prepare':prepare(study,args.parent.resolve(),args.python_hash_seed,args.trace)
    else:run(study)
