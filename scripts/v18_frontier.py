"""Bounded v18 frontier calibration using the existing request and resume engine.

`plan-calibration` is CPU only. `run-calibration` needs the coordinator's qualified
single-GPU deploy; it never reads gold, evaluates answers, or runs implicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.v13_seed_runs import arm_config_hash, is_device_error, plan_schedule, run_schedule, validate_resume
from scripts.v18_protocol import sha

PLUGIN = 'experiments.value_direction_hopper.frontier_scope:install'
METHOD_POINTS = ('U50', 'U60', 'T50', 'T60')


def midpoint_policy(frozen, arm):
    method = 'unweighted' if arm.startswith('U') else 'T'
    target = int(arm[1:])
    p50 = frozen['policies'][f'{method}_s50']
    p70 = frozen['policies'][f'{method}_s70']
    if target == 50:
        return p50
    if target == 70:
        return p70
    return {kind: {'log_threshold': (p50[kind]['log_threshold'] + p70[kind]['log_threshold']) / 2}
            for kind in ('local', 'global')}


def config_for(arm, manifest, model, policy_file, library, torch_library, phase, policy):
    from experiments.numerical_qk_reuse.runner import _config, _fingerprint
    args = SimpleNamespace(condition='native_legal_all_layers', phase=phase,
                           ids=[r['id'] for r in json.loads(manifest.read_text())], seeds=[42],
                           manifest=manifest, policy=policy_file, policy_name=('unweighted_s50' if arm.startswith('U') else 'T_s50'),
                           model=model, revision='f7f5b7f5fa82ffc52addd066915886d497f5517b',
                           decision_interval=1, score_refresh_period=1, support='native_legal',
                           output_mode='historical_route_preqk_current_output', selector='legacy_recompute',
                           selector_layers='all', kernel_variant='static', library=library, torch_library=torch_library,
                           plugin=PLUGIN, diagnostic=False, timing_events=False,
                           extra_source=[Path(__file__), Path(__file__).with_name('v18_protocol.py')])
    config = _config(args)
    config.update(frontier_arm=arm, method='unweighted' if arm.startswith('U') else 'T',
                  target=int(arm[1:]), support_geometry='native_legal', fast_t=arm.startswith('T'),
                  policy=policy, collect=True, max_new_tokens=128, thinking=False)
    config['fingerprint'] = _fingerprint({k: v for k, v in config.items() if k != 'fingerprint'})
    return config


def plan_calibration(manifest, policy_file, model, library, torch_library, out, *, authorization,
                     point=0, policies=None, arms=METHOD_POINTS):
    from experiments.numerical_qk_reuse.runner import _rows
    rows = _rows(manifest, allow_task_budgets=True, allow_thinking_off=True)
    if len(rows) != 26 or len({r['id'] for r in rows}) != 26:
        raise ValueError('calibration requires 26 distinct questions')
    if any(k in r for r in rows for k in ('outputs', 'answer', 'expected', 'expected_answer', 'gold')):
        raise ValueError('generation manifest contains gold')
    frozen = json.loads(policy_file.read_text())
    if not 0 <= point < 5 or not arms or set(arms) - set(METHOD_POINTS):
        raise ValueError('calibration point/arms outside bounded plan')
    policies = policies or {arm: midpoint_policy(frozen, arm) for arm in arms}
    phase = f'v18_calibration_p{point}_' + sha(json.dumps([str(manifest.resolve()), sha(manifest.read_bytes()),
                                                        sha(policy_file.read_bytes()), policies, list(arms)], sort_keys=True))[:12]
    configs = {arm: config_for(arm, manifest, model, policy_file, library, torch_library, phase,
                               policies[arm]) for arm in arms}
    hashes = {arm: arm_config_hash(config) for arm, config in configs.items()}
    schedule = plan_schedule(phase, configs[next(iter(arms))]['revision'], hashes,
                             [r['id'] for r in rows], [42], 2026092603, warm_repeats=0)
    protocol = dict(schema='v18_calibration_protocol_v1', protocol_id=phase, authorization=authorization,
                    point_index=point, max_points_per_method=5, selection_metric='minimum maximum absolute error among overall, global, local achieved sparsity; no quality/steps/time selection',
                    source_scope='native_legal_all_layers', ids=[r['id'] for r in rows], seeds=[42],
                    manifest=str(manifest.resolve()), manifest_sha256=sha(manifest.read_bytes()),
                    policy_file=str(policy_file.resolve()), model_path=str(model.resolve()),
                    model_revision=configs[next(iter(arms))]['revision'], arm_hashes=hashes,
                    arms={arm: {'method': configs[arm]['method'], 'target': configs[arm]['target'],
                                'policy': configs[arm]['policy']} for arm in arms},
                    schedule=schedule, planned_executions=len(schedule),
                    ceilings={'executions': len(schedule), 'single_gpu': True})
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f'calibration_p{point}_protocol.json'
    if destination.exists() and json.loads(destination.read_text()) != protocol:
        raise ValueError('calibration protocol already exists with a different identity')
    destination.write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    for arm, config in configs.items():
        path = out / 'configs' / f'p{point}' / f'{arm}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('calibration config already exists with a different identity')
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    return protocol


def run_calibration(protocol_path, private, ledger, lock_path, max_executions, deadline_epoch, timeout):
    import fcntl
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import _atomic, _one, _rows
    from scripts.v9_clean_request_timing import append, disk_cache_entries, redacted
    from scripts.v10_request_runs import import_identity
    from triton.runtime.jit import JITFunction

    protocol = json.loads(protocol_path.read_text())
    arms = tuple(protocol['arms'])
    configs = {arm: json.loads((protocol_path.parent / 'configs' / f"p{protocol['point_index']}" / f'{arm}.json').read_text()) for arm in arms}
    if {arm: arm_config_hash(config) for arm, config in configs.items()} != protocol['arm_hashes']:
        raise ValueError('calibration config hash mismatch')
    for arm, config in configs.items():
        for filename, expected in config['source_hashes'].items():
            source = Path(filename)
            if not source.is_file() or sha(source.read_bytes()) != expected:
                raise ValueError(f'{arm}: source or binary drift: {filename}')
    manifest = Path(protocol['manifest'])
    if sha(manifest.read_bytes()) != protocol['manifest_sha256']:
        raise ValueError('generation manifest byte hash mismatch')
    rows = _rows(manifest, allow_task_budgets=True, allow_thinking_off=True)
    if any(k in r for r in rows for k in ('outputs', 'answer', 'expected', 'expected_answer', 'gold')):
        raise ValueError('generation manifest contains gold')
    if {r['id'] for r in rows} != set(protocol['ids']) or len(rows) != len(protocol['ids']):
        raise ValueError('manifest ID coverage mismatch')
    identity = dict(protocol_id=protocol['protocol_id'], model_revision=protocol['model_revision'],
                    arm_hashes=protocol['arm_hashes'], source_hashes=configs[arms[0]]['source_hashes'],
                    private_root=str(private.resolve()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process_started = time.perf_counter()
        events = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        done = validate_resume(events, identity, protocol['schedule'])
        if len(done) == len(protocol['schedule']):
            return 'complete'
        gpu_uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()[0].strip()
        append(ledger, dict(event='start', pid=os.getpid(), pgid=os.getpgid(0), when=time.time(),
                            import_identity=import_identity(), disk_cache_entries=disk_cache_entries(), **identity))
        adapter = create_adapter('diffusion_gemma', protocol['model_path'], device='cuda',
                                 precision='bfloat16', revision=protocol['model_revision']).load()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        compiles = []
        def before(**kw):
            compiles.append(time.perf_counter())
            return False
        JITFunction.cache_hook = before
        class Timeout(Exception):
            pass
        def alarm(*_):
            raise Timeout()
        signal.signal(signal.SIGALRM, alarm)
        def execute_one(row, seed, config, entry):
            start, count = time.perf_counter(), len(compiles)
            receipt, error = None, None
            signal.alarm(timeout)
            try:
                receipt = _one(adapter, row, seed, config)
            except Timeout:
                error = f'timeout>{timeout}s'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:500]
            finally:
                signal.alarm(0)
            record = dict(ok=receipt is not None, error=error, generation_seed=seed,
                          arm_config_hash=protocol['arm_hashes'][entry['arm']],
                          outer_wall_s=time.perf_counter() - start, triton_misses=len(compiles) - count)
            if receipt is not None:
                if receipt['seed'] != seed:
                    raise AssertionError('generation seed mismatch')
                record.update(redacted(receipt))
            return dict(record=record, receipt=receipt, fatal=bool(error and is_device_error(error)))
        try:
            status = run_schedule(protocol['schedule'], done=done, execute_one=execute_one,
                                  rows={r['id']: r for r in rows}, configs=configs,
                                  ledger_append=lambda r: append(ledger, r), private=private,
                                  save_receipt=_atomic, max_executions=max_executions,
                                  stop_flag=lambda: deadline_epoch is not None and time.time() >= deadline_epoch)
        finally:
            JITFunction.cache_hook = None
        append(ledger, dict(event='worker_end', status=status, when=time.time(),
                            host=socket.gethostname(), gpu_uuid=gpu_uuid,
                            gpu_process_seconds=time.perf_counter() - process_started))
    return status


def measured_density(protocol, ledger_path):
    """Read private routing counters only; answer content is never used."""
    events = [json.loads(s) for s in ledger_path.read_text().splitlines() if s.strip()]
    runs = [e for e in events if e.get('event') == 'run']
    expected = {(e['arm'], e['id'], e['seed']) for e in protocol['schedule']}
    if len(runs) != len(expected) or {(e['arm'], e['id'], e['seed']) for e in runs} != expected:
        raise ValueError('calibration incomplete or duplicate/foreign run')
    counts = {arm: {kind: {'eligible': 0, 'skipped': 0} for kind in ('whole', 'local', 'global')}
              for arm in protocol['arms']}
    for event in runs:
        if not event.get('ok') or not event.get('private_receipt'):
            raise ValueError('calibration execution failed; no density selection')
        receipt = json.loads(Path(event['private_receipt']).read_text())
        if receipt.get('seed') != event['seed'] or receipt.get('id') != event['id']:
            raise ValueError('calibration receipt identity mismatch')
        routing = receipt.get('routing')
        if not isinstance(routing, list) or not routing:
            raise ValueError('calibration routing counters absent')
        for row in routing:
            kind = row['attention_type']
            if kind not in ('local', 'global'):
                raise ValueError('unknown attention kind')
            eligible, skipped = row['eligible'], row['skipped']
            if not isinstance(eligible, int) or not isinstance(skipped, int) or not 0 <= skipped <= eligible:
                raise ValueError('invalid routing counts')
            for bucket in ('whole', kind):
                counts[event['arm']][bucket]['eligible'] += eligible
                counts[event['arm']][bucket]['skipped'] += skipped
    result = {}
    for arm, buckets in counts.items():
        if any(b['eligible'] == 0 for b in buckets.values()):
            raise ValueError('missing eligible tiles in a density stratum')
        result[arm] = {kind: b['skipped'] / b['eligible'] for kind, b in buckets.items()}
    return result, counts


def next_policy(history, target):
    """Coordinate bracket update driven only by measured density."""
    current = history[-1]
    new = {}
    for kind in ('local', 'global'):
        here = current['policy'][kind]['log_threshold']
        samples = [(p['actual'][kind], p['policy'][kind]['log_threshold']) for p in history]
        below = [(density, value) for density, value in samples if density < target]
        above = [(density, value) for density, value in samples if density >= target]
        if below and above:
            guess = (max(below)[1] + min(above)[1]) / 2
        else:
            guess = here + (.35 if current['actual'][kind] < target else -.35)
        new[kind] = {'log_threshold': min(here + .6, max(here - .6, float(guess)))}
    return new


def best_point(history, target):
    return min(history, key=lambda p: (max(abs(p['actual'][k] - target) for k in ('whole', 'local', 'global')),
                                       sum(abs(p['actual'][k] - target) for k in ('local', 'global'))))


def advance_calibration(protocol_path, ledger_path, out):
    """Append one measured point and either freeze or plan the next point."""
    protocol = json.loads(protocol_path.read_text())
    actual, counts = measured_density(protocol, ledger_path)
    history_path = out / 'calibration_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {arm: [] for arm in METHOD_POINTS}
    point = protocol['point_index']
    next_arms, next_policies = [], {}
    for arm in protocol['arms']:
        entries = history.setdefault(arm, [])
        if len(entries) < point or len(entries) > 5:
            raise ValueError('calibration history point mismatch')
        measured = dict(point=point, policy=protocol['arms'][arm]['policy'], actual=actual[arm], counts=counts[arm])
        if len(entries) == point:
            entries.append(measured)
        elif entries[point] != measured:
            raise ValueError('calibration history conflicts with measured receipt point')
        prefix = entries[:point + 1]
        target = int(arm[1:]) / 100
        best = best_point(prefix, target)
        attained = all(abs(best['actual'][k] - target) <= .02 for k in ('whole', 'local', 'global'))
        if not attained and point < 4:
            next_arms.append(arm)
            next_policies[arm] = next_policy(prefix, target)
    temp = history_path.with_suffix('.tmp')
    temp.write_text(json.dumps(history, indent=2, sort_keys=True) + '\n')
    temp.replace(history_path)
    if not next_arms:
        frozen = {arm: dict(policy=best_point(entries, int(arm[1:]) / 100)['policy'],
                            actual=best_point(entries, int(arm[1:]) / 100)['actual'],
                            attained=all(abs(best_point(entries, int(arm[1:]) / 100)['actual'][k] - int(arm[1:]) / 100) <= .02
                                         for k in ('whole', 'local', 'global')),
                            points=len(entries), calibration_ids=protocol['ids'])
                  for arm, entries in history.items() if entries}
        frozen_path = out / 'calibration_frozen.json'
        frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
        return None
    first = json.loads((out / 'calibration_p0_protocol.json').read_text())
    first_config = json.loads((out / 'configs/p0/U50.json').read_text())
    return plan_calibration(Path(first['manifest']), Path(first['policy_file']), Path(first['model_path']),
                            Path(first_config['library']), Path(first_config['torch_library']), out,
                            authorization=first['authorization'], point=point + 1,
                            policies=next_policies, arms=tuple(next_arms))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('plan-calibration')
    for name in ('manifest', 'policy', 'model', 'library', 'torch-library', 'out', 'authorization'):
        a.add_argument('--' + name, required=True)
    b = sub.add_parser('run-calibration')
    for name in ('protocol', 'private', 'ledger', 'lock'):
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--max-executions', type=int, default=104)
    b.add_argument('--deadline-epoch', type=float)
    b.add_argument('--timeout', type=int, default=900)
    c = sub.add_parser('calibrate-all')
    for name in ('out', 'private', 'lock'):
        c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--deadline-epoch', type=float, required=True)
    c.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    if args.action == 'plan-calibration':
        x = plan_calibration(Path(args.manifest), Path(args.policy), Path(args.model),
                             Path(args.library), Path(args.torch_library), Path(args.out),
                             authorization=args.authorization)
        print(json.dumps({'protocol_id': x['protocol_id'], 'planned_executions': x['planned_executions']}))
    elif args.action == 'run-calibration':
        if not 1 <= args.max_executions <= 104:
            p.error('--max-executions must be 1..104')
        print(run_calibration(args.protocol, args.private, args.ledger, args.lock,
                              args.max_executions, args.deadline_epoch, args.timeout))
    else:
        protocol_path = args.out / 'calibration_p0_protocol.json'
        if not protocol_path.is_file():
            p.error('plan-calibration first')
        while protocol_path is not None:
            protocol = json.loads(protocol_path.read_text())
            point = protocol['point_index']
            ledger = args.out / f'calibration_p{point}_ledger.jsonl'
            status = run_calibration(protocol_path, args.private / f'p{point}', ledger, args.lock,
                                     len(protocol['schedule']), args.deadline_epoch, args.timeout)
            print(json.dumps({'point': point, 'status': status}), flush=True)
            if status != 'complete':
                break
            next_protocol = advance_calibration(protocol_path, ledger, args.out)
            protocol_path = None if next_protocol is None else args.out / f"calibration_p{point + 1}_protocol.json"


if __name__ == '__main__':
    main()
