"""Adopt and advance the two-host frozen v18 stages using status metadata only.

Run on the coordinator workstation. Reads stage status and SHA-256 hashes over
SSH every 60 seconds; never reads prompts, gold, completions, or private receipts.
Each stage launcher is immutable and handles its own GPU/request/deadline guards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

HOSTS = {
    'mpk': ('exouser@149.165.151.254', '/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926'),
    'dllm': ('exouser@149.165.159.64', '/home/exouser/dyh/junyu_frontier_v18_20260926'),
}
DEPLOY = 'driver_5b14a21'
STAGES = ('initial', 'aime', 'remainder')
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15')


def marker_state(started, done):
    if done is not None:
        if started is None or done.get('rc') != 0 or done.get('start') != started.get('start'):
            return 'failed'
        return 'complete'
    return 'running' if started is not None else 'absent'


def remote(host, command):
    for attempt in range(3):
        try:
            return subprocess.check_output(['ssh', *SSH_OPTIONS, HOSTS[host][0], command],
                                           text=True, timeout=60).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(60)


def marker(host, stage, suffix):
    root = HOSTS[host][1]
    path = f'{root}/evaluation/status/{host}_{stage}.{suffix}.json'
    value = remote(host, f'if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; fi')
    return json.loads(value) if value else None


def launch(host, stage):
    target, root = HOSTS[host]
    script = f'{root}/deploy/{DEPLOY}/scripts/v18_run_stage.sh'
    log = f'{root}/logs/eval_{stage}.log'
    command = (f'setsid -f bash {shlex.quote(script)} {shlex.quote(DEPLOY)} {shlex.quote(stage)} '
               f'> {shlex.quote(log)} 2>&1 < /dev/null &')
    try:
        subprocess.run(['ssh', *SSH_OPTIONS, target, command], check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Unknown dispatch result: read a marker, never issue a second launch.
        if marker(host, stage, 'started') is None:
            raise RuntimeError(f'{host}/{stage} launch uncertain and no marker; manual inspection required')
    until = time.monotonic() + 15
    while marker(host, stage, 'started') is None:
        if time.monotonic() >= until:
            raise RuntimeError(f'{host}/{stage} dispatched but no started marker; no automatic retry')
        time.sleep(1)


def alive(host, pid):
    if type(pid) is not int or pid <= 0:
        return False
    result = remote(host, f'if kill -0 {pid} 2>/dev/null; then echo yes; else echo no; fi')
    return result == 'yes'


def wait_stage(stage, *, adopt, poll_seconds, deadline_epoch):
    for host in HOSTS:
        state = marker_state(marker(host, stage, 'started'), marker(host, stage, 'done'))
        if state == 'failed':
            raise RuntimeError(f'{host}/{stage} failed; inspect immutable status and ledger, no automatic retry')
        if state == 'absent':
            if adopt:
                raise RuntimeError(f'{host}/{stage} has no started marker to adopt')
            launch(host, stage)
    while True:
        if time.time() >= deadline_epoch:
            raise RuntimeError('campaign hard deadline reached; inspect live workers and preserve ledgers')
        states = {host: marker_state(marker(host, stage, 'started'), marker(host, stage, 'done'))
                  for host in HOSTS}
        print(json.dumps({'stage': stage, 'states': states}), flush=True)
        if 'failed' in states.values() or 'absent' in states.values():
            raise RuntimeError(f'{stage} failed or lost a started marker; no automatic retry')
        if all(state == 'complete' for state in states.values()):
            return
        for host, state in states.items():
            if state == 'running':
                started = marker(host, stage, 'started')
                if not alive(host, started.get('pid')):
                    raise RuntimeError(f'{host}/{stage} started marker has no live supervisor; no retry')
        time.sleep(poll_seconds)


def sync_ledger(source, dataset, staging):
    peer = 'dllm' if source == 'mpk' else 'mpk'
    src_host, src_root = HOSTS[source]
    dst_host, dst_root = HOSTS[peer]
    basename = f'{source}_{dataset}.jsonl'
    src = f'{src_root}/evaluation/ledgers/{basename}'
    dst = f'{dst_root}/evaluation/ledgers/{basename}'
    local = staging / basename
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    f'{src_host}:{src}', str(local)], check=True, timeout=120)
    local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
    source_hash = remote(source, f'sha256sum {shlex.quote(src)}').split()[0]
    if local_hash != source_hash:
        raise RuntimeError('source ledger changed during transfer; preserve and inspect stage')
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    str(local), f'{dst_host}:{dst}'], check=True, timeout=120)
    dest_hash = remote(peer, f'sha256sum {shlex.quote(dst)}').split()[0]
    if dest_hash != local_hash:
        raise RuntimeError('peer ledger copy hash mismatch')
    return dict(source=source, dataset=dataset, sha256=local_hash)


def own_process_alive(host, pid, command_token):
    """Check both PID liveness and command identity; a reused PID is not ours."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('known worker/supervisor PID missing')
    line = remote(host, f'ps -ww -p {pid} -o stat=,args= 2>/dev/null || true')
    if not line:
        return False
    status, _, command = line.partition(' ')
    return not status.startswith('Z') and command_token in command


def open_worker_pids(host):
    """Only these two redacted campaign ledgers can identify own GPU workers."""
    root = HOSTS[host][1]
    pending = []
    for dataset in ('ruler', 'aime'):
        path = f'{root}/evaluation/ledgers/{host}_{dataset}.jsonl'
        payload = remote(host, f'if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; fi')
        open_starts = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') == 'start':
                open_starts.append(event)
            elif event.get('event') == 'worker_end':
                if not open_starts:
                    raise RuntimeError(f'{host}/{dataset} has worker_end without start')
                open_starts.pop(0)
        for start in open_starts:
            if type(start.get('pid')) is not int or start['pid'] <= 0:
                raise RuntimeError(f'{host}/{dataset} open worker start lacks a verifiable PID')
            pending.append(dict(dataset=dataset, pid=start['pid']))
    return pending


def wait_known_writers_quiet(stages, *, poll_seconds, deadline_epoch):
    """Never finalize while a known supervisor or GPU worker still owns a PID."""
    while True:
        snapshot, active = {}, []
        for host in HOSTS:
            host_stages = {}
            for stage in stages:
                started, done = marker(host, stage, 'started'), marker(host, stage, 'done')
                state = marker_state(started, done)
                if started is not None:
                    supervisor_live = own_process_alive(host, started.get('pid'), 'v18_run_stage.sh')
                    if supervisor_live:
                        active.append(dict(host=host, stage=stage, role='supervisor'))
                else:
                    supervisor_live = False
                host_stages[stage] = dict(state=state, supervisor_live=supervisor_live,
                                          done_marker=done is not None)
            workers = open_worker_pids(host)
            for worker in workers:
                worker['live'] = own_process_alive(host, worker['pid'], 'scripts.v18_evaluate')
                if worker['live']:
                    active.append(dict(host=host, dataset=worker['dataset'], role='worker'))
            snapshot[host] = dict(stages=host_stages, open_workers=workers)
        if not active:
            return snapshot
        if time.time() >= deadline_epoch:
            raise RuntimeError('known GPU writers remain live at CPU finalization cutoff: ' +
                               json.dumps(active, sort_keys=True))
        time.sleep(poll_seconds)


def sync_all_ledgers(staging):
    """Hash-check latest redacted ledgers for both hosts and both datasets."""
    return [sync_ledger(host, dataset, staging)
            for dataset in ('ruler', 'aime') for host in HOSTS]


def aime_complete(staging):
    """A clean rc=0 can still be a partial budget stop; never start remainder then."""
    from scripts.v18_stage_driver import completed_keys

    source, root = HOSTS['mpk']
    path = f'{root}/evaluation/aime26_primary_protocol.json'
    local = staging / 'aime26_primary_protocol.json'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    f'{source}:{path}', str(local)], check=True, timeout=120)
    local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
    if local_hash != remote('mpk', f'sha256sum {shlex.quote(path)}').split()[0]:
        raise RuntimeError('AIME protocol changed during completion check')
    protocol = json.loads(local.read_text())
    ledgers = [staging / f'{host}_aime.jsonl' for host in HOSTS]
    done = completed_keys(protocol, ledgers)
    return dict(complete=len(done) == len(protocol['schedule']),
                completed=len(done), planned=len(protocol['schedule']),
                protocol_id=protocol['protocol_id'])


def persist_outcome(path, outcome):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(outcome, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--start-at', choices=STAGES, default='initial',
                        help='Adopt an already started stage when resuming a coordinator')
    parser.add_argument('--budget', type=Path, default=Path('results/junyu_frontier_v18_20260926/campaign_budget.json'))
    parser.add_argument('--after-stages-command-file', type=Path, required=True,
                        help='JSON argv for the vetted CPU-only finalization hook')
    parser.add_argument('--outcome', type=Path,
                        default=Path('results/junyu_frontier_v18_20260926/coordinator_outcome.json'),
                        help='Persistent CPU coordinator result, including partial recovery errors')
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error('status polling must be at least 60 seconds')
    budget = json.loads(args.budget.read_text())
    deadline_epoch = budget['deadline_epoch']
    cpu_deadline = deadline_epoch + 60 * budget['scoring_minutes_reserved']
    outcome = dict(schema='v18_coordinator_outcome_v1', deploy=DEPLOY,
                   started_epoch=time.time(), gpu_cutoff_epoch=deadline_epoch,
                   cpu_finalization_deadline_epoch=cpu_deadline,
                   start_at=args.start_at, stage_results=[], finalization='pending')
    persist_outcome(args.outcome, outcome)
    with tempfile.TemporaryDirectory(prefix='v18_ledgers_') as folder:
        staging = Path(folder)
        known_stages = []
        try:
            for stage in STAGES[STAGES.index(args.start_at):]:
                known_stages.append(stage)
                wait_stage(stage, adopt=stage == args.start_at, poll_seconds=args.poll_seconds,
                           deadline_epoch=deadline_epoch)
                dataset = 'aime' if stage == 'aime' else 'ruler'
                synced = [sync_ledger(host, dataset, staging) for host in HOSTS]
                outcome['stage_results'].append(dict(stage=stage, status='closed',
                                                     ledger_copies_verified=synced))
                persist_outcome(args.outcome, outcome)
                print(json.dumps({'stage': stage, 'ledger_copies_verified': synced}), flush=True)
                if stage == 'aime':
                    completion = aime_complete(staging)
                    outcome['aime_completion'] = completion
                    persist_outcome(args.outcome, outcome)
                    if not completion['complete']:
                        outcome['gpu_stop_reason'] = 'clean_partial_aime'
                        break
        except (Exception, KeyboardInterrupt) as exc:
            outcome['gpu_stop_reason'] = 'stage_failure'
            outcome['stage_error'] = dict(type=type(exc).__name__, message=str(exc)[:1000],
                                          stage=known_stages[-1] if known_stages else None)
            persist_outcome(args.outcome, outcome)
        if 'gpu_stop_reason' not in outcome:
            outcome['gpu_stop_reason'] = 'all_stages_closed'
        try:
            outcome['known_writers'] = wait_known_writers_quiet(
                STAGES, poll_seconds=args.poll_seconds, deadline_epoch=cpu_deadline)
            outcome['writers_quiet'] = True
            persist_outcome(args.outcome, outcome)
            outcome['ledger_copies_verified'] = sync_all_ledgers(staging)
            persist_outcome(args.outcome, outcome)
            argv = json.loads(args.after_stages_command_file.read_text())
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
                raise ValueError('--after-stages-command-file must be a JSON argv array')
            remaining = cpu_deadline - time.time()
            if remaining <= 0:
                raise TimeoutError('CPU finalization reservation exhausted before final hook')
            subprocess.run(argv, check=True, timeout=remaining)
            outcome['finalization'] = 'complete'
        except (Exception, KeyboardInterrupt) as exc:
            outcome['finalization'] = 'failed'
            outcome['finalization_error'] = dict(type=type(exc).__name__, message=str(exc)[:1000])
        outcome['finished_epoch'] = time.time()
        persist_outcome(args.outcome, outcome)
        print(json.dumps({'gpu_stop_reason': outcome['gpu_stop_reason'],
                          'finalization': outcome['finalization'],
                          'outcome': str(args.outcome)}), flush=True)
    if outcome['gpu_stop_reason'] != 'all_stages_closed' or outcome['finalization'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
