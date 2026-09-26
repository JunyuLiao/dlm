"""Adopt and advance the two-host frozen v18 stages using status metadata only.

Run on the coordinator workstation. Reads stage status and SHA-256 hashes over
SSH every 60 seconds; never reads prompts, gold, completions, or private receipts.
Each stage launcher is immutable and handles its own GPU/request/deadline guards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--start-at', choices=STAGES, default='initial',
                        help='Adopt an already started stage when resuming a coordinator')
    parser.add_argument('--budget', type=Path, default=Path('results/junyu_frontier_v18_20260926/campaign_budget.json'))
    parser.add_argument('--after-stages-command-file', type=Path,
                        help='Optional JSON argv for a vetted CPU-only finalization command')
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error('status polling must be at least 60 seconds')
    budget = json.loads(args.budget.read_text())
    deadline_epoch = budget['deadline_epoch']
    with tempfile.TemporaryDirectory(prefix='v18_ledgers_') as folder:
        staging = Path(folder)
        for stage in STAGES[STAGES.index(args.start_at):]:
            wait_stage(stage, adopt=stage == args.start_at, poll_seconds=args.poll_seconds,
                       deadline_epoch=deadline_epoch)
            dataset = 'aime' if stage == 'aime' else 'ruler'
            synced = [sync_ledger(host, dataset, staging) for host in HOSTS]
            print(json.dumps({'stage': stage, 'ledger_copies_verified': synced}), flush=True)
    print(json.dumps({'gpu_stages': 'complete', 'next': 'offline_scoring'}), flush=True)
    if args.after_stages_command_file:
        argv = json.loads(args.after_stages_command_file.read_text())
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
            parser.error('--after-stages-command-file must be a JSON argv array')
        cpu_deadline = deadline_epoch + 60 * budget['scoring_minutes_reserved']
        subprocess.run(argv, check=True, timeout=max(1, int(cpu_deadline - time.time())))


if __name__ == '__main__':
    main()
