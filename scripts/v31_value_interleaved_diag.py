"""Qualify process-matched method comparisons before any new accuracy panel."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_value_decision_replay as io
from scripts import v31_value_native_study as native

CONTROL='current_v31_control'
ARMS=[CONTROL,'full_swap8',CONTROL,'residual_direction_guard',CONTROL,
      'residual',CONTROL,'full_swap8_direction_guard',CONTROL]
FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','rng_seed')


def prepare(study,parent):
    if study.exists():raise RuntimeError('use a new diagnostic attempt')
    p=native.verify(parent)
    cell=next(c for c in io.read(parent/'private/schedules/aime26/cells.json')
              if c['index']==15 and c['seed']==44)
    (study/'private').mkdir(parents=True)
    io.write(study/'private/cells.json',[cell for _ in ARMS],True)
    io.write(study/'private/arms.json',[CONTROL]+ARMS,True)
    worker=io.ROOT/'scripts/v31_value_interleaved_worker.py'
    shutil.copyfile(worker,study/'private/worker.py')
    shutil.copyfile(__file__,study/'coordinator_source.py')
    protocol=dict(schema='value_interleaved_repeatability_v1',parent=str(parent),
        parent_protocol_sha256=io.sha(parent/'protocol.json'),parent_source_commit=p['source_commit'],
        overlay_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        worker_sha256=io.sha(study/'private/worker.py'),coordinator_sha256=io.sha(study/'coordinator_source.py'),
        cells_sha256=io.sha(study/'private/cells.json'),arms_sha256=io.sha(study/'private/arms.json'),
        arms=ARMS,jobs=['repeat001','repeat002'],python_hash_seed=0,
        gate='all five control repeats identical; every inactive variant identical to those controls; receipt validation passes',
        scope='one previously examined AIME cell; process-matched execution qualification, not task accuracy',timing='paused')
    io.write(study/'protocol.json',protocol,True)
    public=dict(protocol);public.pop('parent');public['parent_study']=str(parent.relative_to(io.ROOT))
    public['private_runtime_protocol_sha256']=io.sha(study/'protocol.json')
    io.write(study/'config.json',public,True)


def run(study):
    p=io.read(study/'protocol.json');parent=Path(p['parent'])
    for key,file in [('coordinator_sha256',Path(__file__)),('worker_sha256',study/'private/worker.py'),
                     ('cells_sha256',study/'private/cells.json'),('arms_sha256',study/'private/arms.json'),
                     ('parent_protocol_sha256',parent/'protocol.json')]:
        if io.sha(file)!=p[key]:raise RuntimeError('frozen diagnostic source or input changed')
    source=native.verify(parent);execution=parent/'private/execution';results=[]
    cells=io.read(study/'private/cells.json')
    for job in p['jobs']:
        family=study/job;markers=list(family.glob('attempt*/attempt_complete.json'))
        if markers:
            if len(markers)!=1:raise RuntimeError('ambiguous completed attempt')
            leaf=markers[0].parent;proof=io.read(markers[0])
            if proof['records_sha256']!=io.sha(leaf/'records.jsonl'):raise RuntimeError('diagnostic rows changed')
        else:
            ids=[int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()]
            leaf=family/f'attempt{max(ids,default=0)+1:03d}';(leaf/'private').mkdir(parents=True)
            native.base.wait_idle()
            if native.base.runtime_identity()!=source['runtime']:raise RuntimeError('native runtime changed')
            env=native.base.env_for(parent,'aime26',CONTROL)
            env.update(STUDY_ARM='shared_engine_repeatability',PYTHONHASHSEED='0',
                FROZEN_SOURCE_SHA256=io.sha(parent/'receipts/source_fingerprints.json'),
                VALUE_REFINEMENT_GATE=str(parent/'decision_promotion.json'),
                NATIVE_INTERLEAVED_ARMS=str(study/'private/arms.json'),
                NATIVE_INTERLEAVED_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
            cmd=[native.base.PYTHON,str(study/'private/worker.py'),str(io.MODEL),
                 str(parent/'private/pools/aime26'),str(study/'private/cells.json'),
                 str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
            io.write(study/'state.json',dict(stage='running',job=job,timing='paused'))
            start=time.monotonic()
            with (leaf/'private/worker.log').open('w') as log:
                proc=subprocess.run(cmd,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
            rows=native.base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
            errors=[];report=[]
            if proc.returncode or len(rows)!=len(p['arms']):errors.append('worker or request coverage failure')
            if len(rows)==len(p['arms']):
                reference={k:rows[0][k] for k in FIELDS}
                for i,(arm,row,cell) in enumerate(zip(p['arms'],rows,cells)):
                    a=row['receipts']['adapter']
                    if a.get('comparison_request_index')!=i+1 or a.get('comparison_arm')!=arm:
                        errors.append('request arm receipt disagrees with frozen schedule')
                    if row.get('study_arm')!='shared_engine_repeatability':errors.append('raw study label changed')
                    # The raw record is retained unchanged. Validate the arm
                    # named by its live receipt, not the shared engine label.
                    errors+=native.validate([dict(row,study_arm=arm)],'aime26',arm,[cell])
                    if row['cuda_graph_captures']:errors.append('request captured new CUDA graphs')
                    values={k:row[k] for k in FIELDS};equal=values==reference
                    work=a.get('value_refinement_work',{})
                    active=bool(work.get('swapped_blocks',0) or work.get('corrected_calls',0))
                    if arm==CONTROL and not equal:errors.append('control changes after intervening request')
                    if arm!=CONTROL and not active and not equal:errors.append('inactive candidate changes output')
                    report.append(dict(arm=arm,record=values,equal_to_control=equal,active=active,work=work))
            proof=dict(status='failed' if errors else 'complete',errors=sorted(set(errors)),
                returncode=proc.returncode,gpu_reservation_s=time.monotonic()-start,report=report,
                records_sha256=io.sha(leaf/'records.jsonl') if rows else None,
                private_sha256=io.sha(leaf/'private/generation.private.jsonl') if rows else None)
            io.write(leaf/'receipt.json',proof,True)
            if errors:raise RuntimeError('preserved failed process-matched qualification')
            io.write(leaf/'attempt_complete.json',proof,True)
        results.append(dict(job=job,report=proof['report']))
        print(dict(event='interleaved_diagnostic_complete',job=job,status='passed'),flush=True)
    summary=dict(status='complete',scope=p['scope'],results=results,protocol_sha256=io.sha(study/'protocol.json'),
        gate='passed',timing='paused')
    io.write(study/'summary.json',summary,True)
    io.write(study/'final_complete.json',dict(status='complete',summary_sha256=io.sha(study/'summary.json')),True)
    io.write(study/'state.json',dict(stage='complete',timing='paused'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','run'],required=True)
    a.add_argument('--study',type=Path,required=True);a.add_argument('--parent',type=Path)
    args=a.parse_args();study=args.study.resolve()
    if args.stage=='prepare':prepare(study,args.parent.resolve())
    else:run(study)
