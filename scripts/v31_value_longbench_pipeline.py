"""Serialized, resumable stages for the frozen independent LongBench study.

Completed shards are never relaunched or overwritten. A failed/killed shard
requires a new attempt directory. Threshold calibration is gated on exact
historical development schedule identity and completed audited control data.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess


ARMS=('dense_full_fix51994','dense_piecewise','allkept_fa4','current_v31_control',
      'value_v1_online_discard_mass','value_v2_online_preserve_mass',
      'value_v3a_singleton_delete','value_v3b_greedy_exact','value_v3b_approx_batch8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ('python','model','manifests'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--stage',choices=('reference','calibration','audit','target'),required=True)
    p.add_argument('--attempt-id',required=True,help='new zero-padded stage attempt, e.g. attempt001')
    p.add_argument('--thresholds',help='completed calibration file; required for audit and target')
    p.add_argument('--execution-protocol',help='explicit frozen execution protocol, including any user-authorized amendment')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=root/'results/v31_value_selectors_20261008'
    protocol=json.loads((study/'protocol_original_s1_20261008.json').read_text())
    development=study/'private/cells_development_original_s1.json'
    target=study/'private/cells_target_full503.json'
    if digest(development)!=protocol['development']['schedule_sha256'] or digest(target)!=protocol['target']['schedule_sha256']:
        raise ValueError('frozen original schedule identity mismatch')
    smoke=json.loads((study/'smoke/qualification_complete_attempt003.json').read_text())
    if smoke['status']!='complete' or smoke['arms']!=len(ARMS):
        raise ValueError('all-arm model qualification required')
    base=study/'longbench'/args.stage/args.attempt_id
    base.mkdir(parents=True,exist_ok=True)
    lock=(base/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reference_file=study/'longbench/reference/current_control_reference.json'
    thresholds_file=Path(args.thresholds) if args.thresholds else None

    def run(arm,cells,shard,purpose,threshold=None):
        # Each failed/killed worker retains its own immutable attempt. Completed
        # shards survive a coordinator restart without repeating generation.
        shard.mkdir(parents=True,exist_ok=True)
        for attempt in sorted(shard.glob('attempt[0-9][0-9][0-9]')):
            marker=attempt/'attempt_complete.json'
            if not marker.exists():
                continue
            if json.loads(marker.read_text())['status']!='complete':
                raise ValueError('invalid completion marker')
            config=json.loads((attempt/'config.json').read_text())
            if (config['cells_sha256']!=digest(cells) or config['arm']!=arm
                    or config.get('threshold')!=threshold or config['purpose']!=purpose):
                raise ValueError('completed attempt cannot be rebound')
            for source,expected in config['source_sha256'].items():
                if digest(root/source)!=expected:
                    raise ValueError('completed shard source changed; use a new stage attempt')
            return attempt
        attempts=list(shard.glob('attempt[0-9][0-9][0-9]'))
        next_id=max((int(x.name[-3:]) for x in attempts),default=0)+1
        attempt=shard/f'attempt{next_id:03d}'
        cmd=[args.python,str(root/'scripts/v31_value_campaign.py'),'--python',args.python,
            '--model',args.model,'--manifests',args.manifests,'--cells',str(cells),
            '--attempt',str(attempt),'--arm',arm,'--purpose',purpose]
        if threshold is not None:
            cmd+=['--threshold',str(threshold)]
        subprocess.run(cmd,cwd=root,check=True)
        return attempt

    if args.stage=='reference':
        attempt=run('current_v31_control',development,base/'current_v31_control','development')
        rows=[json.loads(x) for x in (attempt/'records.jsonl').read_text().splitlines()]
        eligible=sum(r['receipts']['adapter']['global_eligible_tiles'] for r in rows)
        kept=sum(r['receipts']['adapter']['global_kept_tiles'] for r in rows)
        config=json.loads((attempt/'config.json').read_text())
        result=dict(status='complete',items=len(rows),questions=32,seeds=[1,2],
            source_attempt=str(attempt.relative_to(study)),source_commit=config['source_commit'],
            cells_sha256=digest(development),global_eligible_tiles=eligible,global_kept_tiles=kept,
            global_sparsity=1-kept/eligible,local_sparsity=0,
            denominator='all decode GLOBAL H x Q128 x KV64 consumer rectangles; includes exact observations and current canvas')
        if reference_file.exists():
            if json.loads(reference_file.read_text())!=result:
                raise ValueError('reference is already frozen; retain it and use a new named calibration family')
        else:
            reference_file.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    if args.stage=='calibration':
        # Calibration owns its immutable directory; a stage lock stays outside
        # that directory so first launch and completed-stage reuse are safe.
        output=base/'search001'
        if (output/'calibration_complete.json').exists():
            print(json.dumps(dict(completed_calibration=str(output.relative_to(study)))),flush=True)
            return
        subprocess.run([args.python,str(root/'scripts/v31_value_calibrate.py'),'--python',args.python,
            '--model',args.model,'--manifests',args.manifests,'--cells',str(development),
            '--reference',str(reference_file),'--output',str(output),
            '--tolerance',str(protocol['calibration_tolerance_absolute']),
            '--max-trials',str(protocol['calibration_max_trials_per_arm'])],cwd=root,check=True)
        return
    if thresholds_file is None:
        raise ValueError('explicit completed calibration required')
    calibration=json.loads(thresholds_file.read_text())
    if calibration['status']!='complete':
        raise ValueError('incomplete calibration')
    thresholds=calibration['thresholds']
    execution_file=Path(args.execution_protocol) if args.execution_protocol else study/'protocol_execution_attempt002_20261008.json'
    execution=json.loads(execution_file.read_text())
    if execution['foundation_protocol_sha256']!=digest(study/'protocol_original_s1_20261008.json'):
        raise ValueError('execution foundation protocol mismatch')
    for source,expected in execution['source_sha256'].items():
        if digest(root/source)!=expected:
            raise ValueError('execution source pin mismatch')
    stage_config=base/'config.json'
    if not stage_config.exists():
        stage_config.write_text(json.dumps(dict(stage=args.stage,arms=ARMS,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            arm_order_by_shard=[list(ARMS[(2*i)%len(ARMS):]+ARMS[:(2*i)%len(ARMS)]) for i in range(4)],
            thresholds_sha256=digest(thresholds_file),protocol_sha256=digest(study/'protocol_original_s1_20261008.json'),
            execution_protocol_sha256=digest(execution_file),
            schedules_sha256=dict(development=digest(development),target=digest(target))),indent=2)+'\n')
    else:
        frozen=json.loads(stage_config.read_text())
        if (frozen['thresholds_sha256']!=digest(thresholds_file)
                or frozen['protocol_sha256']!=digest(study/'protocol_original_s1_20261008.json')
                or frozen['execution_protocol_sha256']!=digest(execution_file)):
            raise ValueError('frozen stage cannot be rebound')
    # Four native-order shards limit lost work. Each arm uses the same shard
    # boundaries and first-cell warm-up. Their union is exactly the 503-item plan.
    cells=json.loads(target.read_text())
    schedule_root=study/'private/target_shards'
    schedule_root.mkdir(parents=True,exist_ok=True)
    shards=[]
    for start in range(0,len(cells),128):
        name=f'shard{len(shards)+1:02d}'
        path=schedule_root/(name+'.json')
        content=json.dumps(cells[start:start+128],indent=1)+'\n'
        if path.exists() and path.read_text()!=content:
            raise ValueError('immutable shard schedule changed')
        if not path.exists():
            path.write_text(content)
        shards.append((name,path))
    orders=json.loads(stage_config.read_text())['arm_order_by_shard']
    for shard_index,(name,schedule) in enumerate(shards):
        for arm in orders[shard_index]:
            run(arm,schedule,base/arm/name,
                'audit' if args.stage=='audit' else 'clean',thresholds.get(arm))
        print(json.dumps(dict(completed_shard=name,stage=args.stage,arms=len(ARMS))),flush=True)
    for arm in ARMS:
        private_files=[]
        completed=[]
        for name,schedule in shards:
            attempt=run(arm,schedule,base/arm/name,
                        'audit' if args.stage=='audit' else 'clean',thresholds.get(arm))
            completed.append(str(attempt.relative_to(study)))
            private_files.append(str(attempt/'private'/('run_'+arm+'.private.jsonl')))
        index_file=base/arm/'completed_shards.json'
        index=dict(status='complete',attempts=completed,cells=503,
                   schedule_sha256=digest(target))
        if index_file.exists() and json.loads(index_file.read_text())!=index:
            raise ValueError('completed shard index changed')
        if not index_file.exists():
            index_file.write_text(json.dumps(index,indent=2)+'\n')
        scoring=base/arm/'scoring'
        scoring.mkdir(exist_ok=True)
        if not (scoring/'official.summary.json').exists():
            subprocess.run([args.python,str(root/'scripts/v31_score_longbench_official.py'),str(scoring/'official'),
                args.manifests,args.manifests,'--cells',str(target),*private_files],cwd=root,check=True)
        print(json.dumps(dict(completed_arm=arm,stage=args.stage,cells=len(cells))),flush=True)
    marker=base/'stage_complete.json'
    if not marker.exists():
        marker.write_text(json.dumps(dict(status='complete',arms=len(ARMS),cells_per_arm=503,
            purpose='separate instrumented target audit' if args.stage=='audit' else 'clean target timing'),indent=2)+'\n')


if __name__=='__main__':
    main()
