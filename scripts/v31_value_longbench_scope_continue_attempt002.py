"""Eight-arm continuation with canonical JSON restart and predecessor reuse.

Generation bytes and schedules stay fixed. Completed matching audit shards are
referenced in place; failed attempts and the original nine-arm family survive.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

from v31_value_longbench_pipeline import ARMS as ORIGINAL_ARMS
from v31_value_summarize import load

EXCLUDED = 'value_v3b_greedy_exact'
ARMS = tuple(arm for arm in ORIGINAL_ARMS if arm != EXCLUDED)
IDENTITY_FIELDS = ('source_sha256', 'model_revision', 'manifests_sha256',
                   'model_source_inventory_sha256', 'host_fingerprint', 'gpu_identity')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f'immutable artifact differs: {path.name}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+'\n')


def orders():
    # Retain each old shard's relative order, removing only exact greedy.
    return [[a for a in ORIGINAL_ARMS[(2*i)%len(ORIGINAL_ARMS):]
             + ORIGINAL_ARMS[:(2*i)%len(ORIGINAL_ARMS)] if a != EXCLUDED]
            for i in range(4)]


def validate_attempt(attempt, cells, arm, purpose, threshold, identity, root):
    config, rows = load(attempt)
    receipt = json.loads((attempt/'receipt.json').read_text())
    if (receipt['status'] != 'complete' or receipt['validation_errors']
            or receipt['returncode'] != 0 or receipt['timed_cuda_captures'] != 0):
        raise ValueError('invalid completed execution receipt')
    schedule = json.loads(cells.read_text())
    if (config['cells_sha256'] != digest(cells) or config['arm'] != arm
            or config['purpose'] != purpose or config['threshold'] != threshold
            or config['cell_count'] != len(schedule)):
        raise ValueError('completed shard cannot be rebound')
    for field in IDENTITY_FIELDS:
        if config[field] != identity[field]:
            raise ValueError('generation identity differs: '+field)
    expected_runtime = dict(identity['runtime_settings'])
    expected_runtime.update(VALUE_AUDIT='1' if purpose == 'audit' else '0',
                            VALUE_CLEAN_TIMING='0' if purpose == 'audit' else '1')
    if config['runtime_settings'] != expected_runtime:
        raise ValueError('frozen runtime settings differ')
    for source, expected in identity['source_sha256'].items():
        if digest(root/source) != expected:
            raise ValueError('generation source changed: '+source)
    key = lambda r: (r['dataset'], r['index'], r['panel_seed'], r['repeat'])
    scheduled_keys = [(r['dataset'], r['index'], r['seed'], r.get('repeat', 0)) for r in schedule]
    if [key(r) for r in rows] != scheduled_keys:
        raise ValueError('shard cell order or coverage differs')
    if not (attempt/'private'/('run_'+arm+'.private.jsonl')).is_file():
        raise ValueError('official scorer input unavailable')
    return config, rows


def main():
    p = argparse.ArgumentParser()
    for name in ('python', 'model', 'manifests', 'thresholds', 'execution-protocol'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--stage', choices=('audit', 'target'), required=True)
    p.add_argument('--attempt-id', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = root/'results/v31_value_selectors_20261008'
    execution_path = Path(args.execution_protocol)
    execution = json.loads(execution_path.read_text())
    foundation = study/'protocol_original_s1_20261008.json'
    thresholds_path = Path(args.thresholds)
    calibration = json.loads(thresholds_path.read_text())
    if (execution['arms'] != list(ARMS)
            or execution['foundation_protocol_sha256'] != digest(foundation)
            or execution['thresholds_sha256'] != digest(thresholds_path)
            or calibration['status'] != 'complete'):
        raise ValueError('scope, foundation or threshold binding mismatch')
    for source, expected in execution['source_sha256'].items():
        if digest(root/source) != expected:
            raise ValueError('continuation source pin mismatch: '+source)
    identity = execution['generation_identity']
    qualification = json.loads((study/'smoke/qualification_complete_attempt003.json').read_text())
    if qualification['status'] != 'complete' or qualification['arms'] != len(ORIGINAL_ARMS):
        raise ValueError('original all-arm qualification required')
    target = study/'private/cells_target_full503.json'
    protocol = json.loads(foundation.read_text())
    if digest(target) != protocol['target']['schedule_sha256']:
        raise ValueError('target schedule changed')
    base = study/'longbench'/(args.stage+'_no_exact')/args.attempt_id
    base.mkdir(parents=True, exist_ok=True)
    lock = (base/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    purpose = 'audit' if args.stage == 'audit' else 'clean'
    config_path = base/'config.json'
    source_commit = (json.loads(config_path.read_text())['source_commit'] if config_path.exists()
                     else subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip())
    write_once(config_path, dict(stage=args.stage, arms=ARMS,
        source_commit=source_commit,
        generation_identity=identity, arm_order_by_shard=orders(),
        thresholds_sha256=digest(thresholds_path), protocol_sha256=digest(foundation),
        execution_protocol_sha256=digest(execution_path),
        schedules_sha256=dict(target=digest(target)),
        predecessor=execution['completed_predecessor_families'][purpose],
        excluded_arm=EXCLUDED, reuse_policy='completed shards with identical generation bytes and execution identity'))
    selected = {}
    reuse = []
    thresholds = calibration['thresholds']
    for i, arm_order in enumerate(orders(), 1):
        shard_name = f'shard{i:02d}'
        cells = study/'private/target_shards'/(shard_name+'.json')
        for arm in arm_order:
            threshold = thresholds.get(arm)
            shard = base/arm/shard_name
            candidates = list(shard.glob('attempt[0-9][0-9][0-9]'))
            for family in execution['completed_predecessor_families'][purpose]:
                candidates += list((study/family/arm/shard_name).glob('attempt[0-9][0-9][0-9]'))
            completed = [a for a in sorted(candidates) if (a/'attempt_complete.json').exists()
                         and json.loads((a/'attempt_complete.json').read_text())['status'] == 'complete']
            if completed:
                attempt = completed[-1]
                config, rows = validate_attempt(attempt, cells, arm, purpose, threshold, identity, root)
                if not attempt.is_relative_to(base):
                    reuse.append(dict(arm=arm, shard=shard_name, cells=len(rows),
                        attempt=str(attempt.relative_to(study)), source_commit=config['source_commit'],
                        config_sha256=digest(attempt/'config.json'),
                        receipt_sha256=digest(attempt/'receipt.json')))
            else:
                shard.mkdir(parents=True, exist_ok=True)
                ordinal = max((int(a.name[-3:]) for a in candidates if a.is_relative_to(base)), default=0)+1
                attempt = shard/f'attempt{ordinal:03d}'
                command = [args.python, str(root/'scripts/v31_value_campaign.py'),
                    '--python', args.python, '--model', args.model, '--manifests', args.manifests,
                    '--cells', str(cells), '--attempt', str(attempt), '--arm', arm, '--purpose', purpose]
                if threshold is not None:
                    command += ['--threshold', str(threshold)]
                subprocess.run(command, cwd=root, check=True)
                validate_attempt(attempt, cells, arm, purpose, threshold, identity, root)
            selected[arm, shard_name] = attempt
        print(json.dumps(dict(completed_shard=shard_name, stage=args.stage, arms=len(ARMS))), flush=True)
    write_once(base/'reused_shards.json', dict(status='identity validated', shards=reuse,
        cells=sum(r['cells'] for r in reuse), excluded_arm=EXCLUDED))
    for arm in ARMS:
        attempts = [selected[arm, f'shard{i:02d}'] for i in range(1,5)]
        write_once(base/arm/'completed_shards.json', dict(status='complete', cells=503,
            schedule_sha256=digest(target), attempts=[str(a.relative_to(study)) for a in attempts]))
        scoring = base/arm/'scoring'
        scoring.mkdir(exist_ok=True)
        if not (scoring/'official.summary.json').exists():
            private_files = [str(a/'private'/('run_'+arm+'.private.jsonl')) for a in attempts]
            subprocess.run([args.python, str(root/'scripts/v31_score_longbench_official.py'),
                str(scoring/'official'), args.manifests, args.manifests,
                '--cells', str(target), *private_files], cwd=root, check=True)
        print(json.dumps(dict(completed_arm=arm, stage=args.stage, cells=503)), flush=True)
    write_once(base/'stage_complete.json', dict(status='complete', arms=len(ARMS), cells_per_arm=503,
        purpose='separate instrumented target audit' if purpose == 'audit' else 'clean target timing'))


if __name__ == '__main__':
    main()
