"""Read-only, bounded watcher for already-launched v20 stages.

Polls each host's ``<root>/<remote-dir>/<stage>.stage.json`` every 60 seconds.
Only terminal stages trigger SHA-verified downloads of final/partial/failure
profile JSON into a private local host directory. Never dispatches, retries or
changes remote jobs, and never reads per-request logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time


TERMINAL = frozenset(('complete', 'failed', 'deadline_stop', 'supervisor_error'))
ARTIFACT_SUFFIXES = ('.json', '.json.partial.json', '.json.failure.json')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_stage(stage, remote_dir):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', stage):
        raise ValueError('stage must be a simple frozen name')
    parts = PurePosixPath(remote_dir).parts
    if not parts or any(part in ('..', '.') for part in parts) or remote_dir.startswith('/') or not re.fullmatch(r'[A-Za-z0-9_./-]+', remote_dir):
        raise ValueError('remote-dir must be a safe relative directory')


def remote_paths(host, stage, remote_dir):
    root = host['root']
    if not root.startswith('/') or not re.fullmatch(r'[A-Za-z0-9_./-]+', root) or '..' in PurePosixPath(root).parts:
        raise ValueError('host root must be a safe absolute POSIX path')
    base = str(PurePosixPath(root) / remote_dir / stage)
    receipt = base + '.stage.json'
    return receipt, [receipt, *(base + suffix for suffix in ARTIFACT_SUFFIXES)]


def atomic_replace(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with tmp.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class SSHTransport:
    """Two bounded read-only attempts; shell text is fixed and paths validated."""
    def __init__(self, user='exouser', timeout=20, retries=2):
        self.user, self.timeout, self.retries = user, timeout, retries

    def _run(self, command):
        last = None
        for attempt in range(self.retries):
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=self.timeout, check=False)
                if result.returncode == 0:
                    return result.stdout
                last = RuntimeError(f'read-only transport exited {result.returncode}')
            except (OSError, subprocess.TimeoutExpired) as exc:
                last = exc
            if attempt + 1 < self.retries:
                time.sleep(2)
        raise RuntimeError(f'bounded read-only transport failed: {type(last).__name__}')

    def _remote_python(self, host, script, paths):
        command = ' '.join([shlex.quote(host['python']), '-c', shlex.quote(script),
                            *(shlex.quote(path) for path in paths)])
        return self._run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                          '-o', f'ConnectTimeout={self.timeout}',
                          f"{self.user}@{host['host']}", command])

    def read_stage(self, host, path):
        script = ('import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); '
                  'print(p.read_text() if p.is_file() else "null")')
        return json.loads(self._remote_python(host, script, [path]))

    def artifact_manifest(self, host, paths):
        script = ('import hashlib,json,sys,pathlib; '
                  'print(json.dumps({p:{"sha256":hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(),'
                  '"bytes":pathlib.Path(p).stat().st_size} for p in sys.argv[1:] if pathlib.Path(p).is_file()}))')
        return json.loads(self._remote_python(host, script, paths))

    def copy_file(self, host, remote_path, local_path):
        self._run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', f'ConnectTimeout={self.timeout}',
                   f"{self.user}@{host['host']}:{remote_path}", str(local_path)])


def fetch_terminal_artifacts(transport, alias, host, paths, destination):
    manifest = transport.artifact_manifest(host, paths)
    if any(path not in paths for path in manifest):
        raise ValueError('remote artifact manifest contains an unrequested path')
    result = {}
    host_dir = Path(destination) / alias
    host_dir.mkdir(parents=True, exist_ok=True)
    for remote, identity in manifest.items():
        local = host_dir / PurePosixPath(remote).name
        if local.exists() or local.with_name(local.name + '.writing').exists():
            raise FileExistsError(f'local artifact already exists: {local}')
        tmp = local.with_name(local.name + '.writing')
        transport.copy_file(host, remote, tmp)
        data = tmp.read_bytes()
        if len(data) != identity['bytes'] or sha(data) != identity['sha256']:
            tmp.unlink(missing_ok=True)
            raise ValueError(f'remote/local artifact SHA mismatch: {alias}/{local.name}')
        os.replace(tmp, local)
        result[local.name] = dict(path=str(local), sha256=identity['sha256'], bytes=len(data))
    return result


def watch(hosts, stage, remote_dir, output_dir, deadline_epoch, *, transport,
          clock=time.time, sleep=time.sleep, interval=60):
    safe_stage(stage, remote_dir)
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError('at least one named host required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', alias) or
           not all(key in host for key in ('host', 'root', 'python', 'env'))
           for alias, host in hosts.items()):
        raise ValueError('invalid host alias or host/root/python/env schema')
    if deadline_epoch <= clock():
        raise ValueError('watch deadline already expired')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcome = output_dir / f'{stage}.outcome.json'
    status = output_dir / f'{stage}.status.json'
    if outcome.exists() or status.exists() or outcome.with_name(outcome.name+'.writing').exists() or status.with_name(status.name+'.writing').exists():
        raise FileExistsError('old watcher outcome/status exists; preserve it')
    state = {alias: dict(status='pending', artifacts={}) for alias in hosts}
    paths = {alias: remote_paths(host, stage, remote_dir) for alias, host in hosts.items()}
    atomic_replace(status, dict(stage=stage, hosts=state, terminal=False))
    while True:
        changed = False
        for alias, host in hosts.items():
            if state[alias]['status'] in TERMINAL:
                continue
            receipt_path, artifact_paths = paths[alias]
            try:
                receipt = transport.read_stage(host, receipt_path)
            except Exception as exc:
                # No mutation or retry of the remote job; next 60 s poll is allowed.
                state[alias]['last_read_error_type'] = type(exc).__name__
                continue
            if receipt is None:
                continue
            observed = receipt.get('status')
            if observed not in TERMINAL | {'started'}:
                state[alias].update(status='failed', reason='invalid_stage_status')
                changed = True
                continue
            if observed == 'started':
                if state[alias]['status'] != 'started':
                    state[alias]['status'] = 'started'
                    changed = True
                continue
            state[alias].update(status=observed, returncode=receipt.get('returncode'),
                                charged_process_seconds=receipt.get('charged_process_seconds'),
                                end_epoch=receipt.get('end_epoch'))
            try:
                state[alias]['artifacts'] = fetch_terminal_artifacts(
                    transport, alias, host, artifact_paths, output_dir)
                if f'{stage}.stage.json' not in state[alias]['artifacts']:
                    state[alias].update(status='failed', reason='terminal_stage_receipt_missing')
                elif observed == 'complete' and f'{stage}.json' not in state[alias]['artifacts']:
                    state[alias].update(status='failed', reason='complete_stage_missing_final_profile')
            except Exception as exc:
                state[alias].update(status='failed', reason='artifact_fetch_or_hash_failure',
                                    artifact_error_type=type(exc).__name__)
            changed = True
        if changed:
            atomic_replace(status, dict(stage=stage, hosts=state, terminal=False))
        if all(row['status'] in TERMINAL for row in state.values()):
            break
        if clock() >= deadline_epoch:
            for row in state.values():
                if row['status'] not in TERMINAL:
                    row.update(status='deadline_stop', reason='local_watch_deadline')
            break
        sleep(min(interval, max(0, deadline_epoch-clock())))
    success = all(row['status'] == 'complete' and row.get('returncode') == 0 for row in state.values())
    final = dict(schema='v20_stage_watch_v1', stage=stage, remote_dir=remote_dir,
                 hosts=state, all_complete=success, deadline_epoch=deadline_epoch)
    atomic_replace(status, dict(final, terminal=True))
    atomic_replace(outcome, final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosts', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--remote-dir', default='cost')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--user', default='exouser')
    args = parser.parse_args(argv)
    hosts = json.loads(args.hosts.read_text(encoding='utf-8'))
    result = watch(hosts, args.stage, args.remote_dir, args.out_dir,
                   args.deadline_epoch, transport=SSHTransport(args.user))
    print(json.dumps(dict(stage=args.stage, all_complete=result['all_complete'],
                          hosts={k: v['status'] for k, v in result['hosts'].items()})), flush=True)
    return 0 if result['all_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
