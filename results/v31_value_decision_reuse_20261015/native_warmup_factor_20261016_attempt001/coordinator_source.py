"""Control-only versus all-method warmup on the frozen failed input history.

Warmup inputs/counts are identical between policies. Only their request arms
differ. Every measured control bookend is retained, including disagreements.
No official accuracy score is used to select a policy.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import time
from scripts import v31_value_decision_replay as io
from scripts import v31_value_native_study as native
from scripts.v31_value_process_matched_study import expand,validate_group
from scripts.v31_value_interleaved_worker import CONTROL


def measured_group(rows,private,arms):
    errors=validate_group(rows,arms)
    if private[0]['completion']!=private[-1]['completion']:
        errors.append('control decoded text drift')
    for row,text,arm in zip(rows[1:-1],private[1:-1],arms[1:-1]):
        work=row['receipts']['adapter'].get('value_refinement_work',{})
        if not (work.get('swapped_blocks',0) or work.get('corrected_calls',0)):
            if text['completion']!=private[0]['completion']:
                errors.append('inactive candidate decoded text drift: '+arm)
    fields=('output_hash','denoise_forwards','canvases','output_tokens','rng_seed')
    return dict(index=rows[0]['index'],seed=rows[0]['panel_seed'],errors=errors,
                controls=[{k:r[k] for k in fields} for r in (rows[0],rows[-1])])


def prepare(study,parent):
    if study.exists():raise RuntimeError('preserve prior attempts')
    p=native.verify(parent);modes=[x for x in p['arms'] if x not in native.CONTROLS]
    source=parent/'private/execution/scripts/v31_value_interleaved_worker.py'
    if io.sha(source)!=io.sha(io.ROOT/'scripts/v31_value_interleaved_worker.py'):
        raise RuntimeError('qualified request switcher differs')
    for rel in ('scripts/v31_value_process_matched_study.py','scripts/v31_value_native_study.py'):
        if io.sha(parent/'private/execution'/rel)!=io.sha(io.ROOT/rel):
            raise RuntimeError('qualified coordinator helper differs: '+rel)
    cells=io.read(parent/'private/schedules/aime26/shard001.json')
    if len(cells)!=9:raise RuntimeError('freeze exactly the failed nine-cell input history')
    warm=next(c for c in cells if c['index']==10 and c['seed']==42)
    measured,arms,groups=expand(cells,modes)
    (study/'private').mkdir(parents=True)
    shutil.copyfile(source,study/'private/worker.py');shutil.copyfile(__file__,study/'coordinator_source.py')
    io.write(study/'private/cells.json',[warm]*4+measured,True)
    for policy in ('control_only','all_methods'):
        prefix=[CONTROL]*4 if policy=='control_only' else modes
        io.write(study/f'private/arms_{policy}.json',[CONTROL]+prefix+arms[1:],True)
    jobs=[dict(name='control_warm_repeat001',policy='control_only'),dict(name='method_warm_repeat001',policy='all_methods'),
          dict(name='method_warm_repeat002',policy='all_methods'),dict(name='control_warm_repeat002',policy='control_only')]
    protocol=dict(schema='value_warmup_factor_v1',parent=str(parent),parent_protocol_sha256=io.sha(parent/'protocol.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=io.ROOT,text=True).strip(),
        coordinator_sha256=io.sha(study/'coordinator_source.py'),worker_sha256=io.sha(study/'private/worker.py'),
        cells_sha256=io.sha(study/'private/cells.json'),arms_sha256={policy:io.sha(study/f'private/arms_{policy}.json') for policy in ('control_only','all_methods')},
        measured_groups=[dict(start=g['start']+4,end=g['end']+4,arms=g['arms']) for g in groups],
        jobs=jobs,explicit_warmup_requests=4,implicit_control_warmup=True,python_hash_seed=0,
        scope='execution diagnostic on the failed nine-cell AIME shard; warmup factor and all mismatches retained; no accuracy claims',
        hypothesis='first active computation may change persistent allocation or initialization state; not established',timing='paused')
    io.write(study/'protocol.json',protocol,True)
    public=dict(protocol);public.pop('parent');public['parent_study']=str(parent.relative_to(io.ROOT));public['private_runtime_protocol_sha256']=io.sha(study/'protocol.json')
    io.write(study/'config.json',public,True)


def run(study):
    p=io.read(study/'protocol.json');parent=Path(p['parent']);source=native.verify(parent)
    for key,file in [('coordinator_sha256',Path(__file__)),('worker_sha256',study/'private/worker.py'),
                     ('cells_sha256',study/'private/cells.json'),('parent_protocol_sha256',parent/'protocol.json')]:
        if p[key]!=io.sha(file):raise RuntimeError('frozen factor source or input changed')
    results=[];planned=io.read(study/'private/cells.json');execution=parent/'private/execution'
    for job in p['jobs']:
        policy=job['policy'];schedule=study/f'private/arms_{policy}.json'
        if io.sha(schedule)!=p['arms_sha256'][policy]:raise RuntimeError('arm schedule changed')
        arms=io.read(schedule)[1:];family=study/job['name'];markers=list(family.glob('attempt*/attempt_complete.json'))
        if markers:
            if len(markers)!=1:raise RuntimeError('ambiguous factor completion')
            leaf=markers[0].parent;proof=io.read(markers[0])
            if proof['records_sha256']!=io.sha(leaf/'records.jsonl') or proof['private_sha256']!=io.sha(leaf/'private/generation.private.jsonl'):
                raise RuntimeError('factor records changed')
        else:
            ordinal=max([int(x.name[7:]) for x in family.glob('attempt*') if x.is_dir()],default=0)+1
            leaf=family/f'attempt{ordinal:03d}';(leaf/'private').mkdir(parents=True)
            native.base.wait_idle()
            if native.base.runtime_identity()!=source['runtime']:raise RuntimeError('runtime changed')
            env=native.base.env_for(parent,'aime26',CONTROL)
            env.update(STUDY_ARM='warmup_factor',PYTHONHASHSEED='0',
                FROZEN_SOURCE_SHA256=io.sha(parent/'receipts/source_fingerprints.json'),
                VALUE_REFINEMENT_GATE=str(parent/'decision_promotion.json'),NATIVE_INTERLEAVED_ARMS=str(schedule),
                NATIVE_INTERLEAVED_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'))
            command=[native.base.PYTHON,str(study/'private/worker.py'),str(io.MODEL),str(parent/'private/pools/aime26'),
                str(study/'private/cells.json'),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),'mage','PIECEWISE']
            io.write(study/'state.json',dict(stage='running',job=job['name'],timing='paused'))
            start=time.monotonic()
            with (leaf/'private/worker.log').open('w') as log:
                proc=subprocess.run(command,cwd=execution,env=env,stdout=log,stderr=subprocess.STDOUT)
            rows=native.base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
            private=native.base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
            errors=[];groups=[];warmup=[]
            if proc.returncode or len(rows)!=len(planned) or len(private)!=len(planned):errors.append('worker or coverage failure')
            if len(rows)==len(planned) and len(private)==len(planned):
                for i,(row,arm,cell) in enumerate(zip(rows,arms,planned)):
                    receipt=row['receipts']['adapter']
                    if receipt.get('comparison_request_index')!=i+1 or receipt.get('comparison_arm')!=arm:errors.append('live arm index mismatch')
                    if row.get('study_arm')!='warmup_factor':errors.append('raw study label mismatch')
                    errors+=native.validate([dict(row,study_arm=arm)],'aime26',arm,[cell])
                    if row['cuda_graph_captures']:errors.append('unexpected measured CUDA capture')
                for i in range(4):
                    work=rows[i]['receipts']['adapter'].get('value_refinement_work',{})
                    warmup.append(dict(arm=arms[i],N=rows[i]['denoise_forwards'],work=work))
                    if policy=='all_methods' and (not work.get('anchor_events',0) or
                         (not arms[i].endswith('swap8') and not work.get('corrected_calls',0))):
                        errors.append('declared active warmup did not exercise method')
                for group in p['measured_groups']:
                    lo,hi=group['start'],group['end']
                    groups.append(measured_group(rows[lo:hi],private[lo:hi],group['arms']))
            proof=dict(status='failed' if errors else 'complete',structural_errors=sorted(set(errors)),returncode=proc.returncode,
                groups=groups,warmup=warmup,parity_pass=bool(groups) and not any(g['errors'] for g in groups),
                protocol_sha256=io.sha(study/'protocol.json'),gpu_reservation_s=time.monotonic()-start,
                records_sha256=io.sha(leaf/'records.jsonl') if rows else None,
                private_sha256=io.sha(leaf/'private/generation.private.jsonl') if rows else None)
            io.write(leaf/'receipt.json',proof,True)
            if errors:raise RuntimeError('preserved structurally invalid warmup-factor attempt')
            io.write(leaf/'attempt_complete.json',proof,True)
        results.append(dict(job=job['name'],policy=policy,groups=proof['groups'],warmup=proof['warmup'],parity_pass=proof['parity_pass']))
        print(dict(event='warmup_factor_job_complete',job=job['name'],parity_pass=proof['parity_pass']),flush=True)
    summary=dict(status='complete',results=results,protocol_sha256=io.sha(study/'protocol.json'),scope=p['scope'],timing='paused')
    io.write(study/'summary.json',summary,True);io.write(study/'final_complete.json',dict(status='complete',summary_sha256=io.sha(study/'summary.json')),True)
    io.write(study/'state.json',dict(stage='complete',timing='paused'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','run'],required=True)
    a.add_argument('--study',type=Path,required=True);a.add_argument('--parent',type=Path)
    args=a.parse_args()
    if args.stage=='prepare':prepare(args.study.resolve(),args.parent.resolve())
    else:run(args.study.resolve())
