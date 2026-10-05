"""One-shot first84 archive collection and CPU-only offline scoring.

The local controller runs only after the existing watcher is terminal. The
same file's ``remote-score`` entry runs on mpk from the pinned CP5 checkout.
Neither entry dispatches generation or retries an uncertain mutation.
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
import sys
import tarfile
import time


SCHEMA = 'v20_collect_score_v1'
SAFE_STAGE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
FROZEN_BINDING_SHA = 'ee06c936e500fecfd219c6a9e277f99e77baed7b39d405ad96625f94e50734b5'
FROZEN_PROTOCOL_SHA = '99e1d287867b0906b8c5a2715d4cf962ec31e6d1f473c1fe0ce2df5891f1aa17'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def safe_archive_members(archive):
    members = archive.getmembers()
    if not members:
        raise ValueError('empty private archive')
    for member in members:
        name = PurePosixPath(member.name)
        if (not member.isfile() or name.is_absolute() or len(name.parts) < 2 or
                name.parts[0] not in ('private', 'ledger') or
                any(part in ('', '.', '..') for part in name.parts)):
            raise ValueError('unsafe or nonregular archive member')
    if not any(PurePosixPath(m.name).parts[0] == 'ledger' for m in members):
        raise ValueError('private archive lacks ledger')
    return members


def extract_archive(path, destination):
    """Extract only regular files under a newly created immutable root."""
    destination = Path(destination)
    with tarfile.open(path, 'r:gz') as archive:
        members = safe_archive_members(archive)
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(1 << 20), b''):
                    output.write(block)
    ledgers = list((destination / 'ledger').glob('*'))
    if len(ledgers) != 1 or not ledgers[0].is_file():
        raise ValueError('archive must contain exactly one ledger')
    return ledgers[0], destination / 'private'


def validate_config(config, hosts):
    if config.get('schema') != SCHEMA or config.get('stage') != 'initial_001':
        raise ValueError('wrong first84 collection schema/stage')
    if set(hosts) != {'mpk', 'dllm'} or any(not all(k in hosts[a] for k in
            ('host', 'root', 'python', 'env')) for a in hosts):
        raise ValueError('expected frozen mpk/dllm hosts')
    for key in ('binding_sha256', 'protocol_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', config.get(key, '')):
            raise ValueError(f'invalid {key}')
    for key in ('remote_protocol', 'remote_binding', 'remote_deploy'):
        value = config.get(key, '')
        if not isinstance(value, str) or not value.startswith('/') or '..' in PurePosixPath(value).parts:
            raise ValueError(f'unsafe {key}')
    for key in ('ruler_gold', 'aime_gold', 'longbench_gold', 'ruler_root'):
        value = config.get(key, '')
        if not isinstance(value, str) or not value.startswith('/') or '..' in PurePosixPath(value).parts:
            raise ValueError(f'unsafe {key}')
    if config.get('watch_dir') is None:
        raise ValueError('watch_dir absent')
    if config.get('binding_sha256') != FROZEN_BINDING_SHA:
        raise ValueError('unfrozen binding')
    if config.get('protocol_sha256') != FROZEN_PROTOCOL_SHA:
        raise ValueError('unfrozen protocol')
    for key, expected in (('binding_local', config['binding_sha256']),
                          ('protocol_local', config['protocol_sha256'])):
        path = Path(config.get(key, ''))
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f'{key} frozen byte drift')
    if not config['remote_deploy'].endswith('/deploy/cp5_05f9947'):
        raise ValueError('offline scorer must use pinned CP5 deploy')


def watch_receipts(config, hosts):
    watch_dir = Path(config['watch_dir'])
    outcome = json.loads((watch_dir / 'initial_001.outcome.json').read_text(encoding='utf-8'))
    if outcome.get('schema') != 'v20_stage_watch_v1' or outcome.get('stage') != 'initial_001' or not outcome.get('all_complete'):
        raise ValueError('both initial stages are not terminal-complete')
    result = {}
    binding = json.loads(Path(config['binding_local']).read_text(encoding='utf-8'))
    for alias, host in hosts.items():
        row = outcome['hosts'].get(alias, {})
        if row.get('status') != 'complete' or row.get('returncode') != 0:
            raise ValueError(f'{alias} stage not complete')
        artifact = row.get('artifacts', {}).get('initial_001.json')
        if not artifact:
            raise ValueError(f'{alias} final stage receipt missing')
        path = Path(artifact['path'])
        if sha_file(path) != artifact['sha256'] or path.stat().st_size != artifact['bytes']:
            raise ValueError(f'{alias} watch artifact hash drift')
        stage = json.loads(path.read_text(encoding='utf-8'))
        public = stage.get('public_identity', {})
        archive = stage.get('private_archive') or {}
        expected_configs = {f'{dataset}/{arm}': item['sha256']
                            for dataset, arms in binding.get('host_configs', {}).get(host['host'], {}).items()
                            for arm, item in arms.items()}
        if (stage.get('status') != 'complete' or stage.get('mode') != 'request' or
                stage.get('stage_name') != 'initial_001' or
                public.get('binding_sha256') != config['binding_sha256'] or
                public.get('protocol_sha256') != config['protocol_sha256'] or
                public.get('host') != host['host'] or
                public.get('execution_config_sha256') != expected_configs or
                archive.get('path') != str(PurePosixPath(host['root']) / 'primary' / 'initial_001.private.tar.gz') or
                not re.fullmatch('[0-9a-f]{64}', archive.get('sha256', '')) or
                not isinstance(archive.get('bytes'), int) or archive['bytes'] <= 0):
            raise ValueError(f'{alias} stage/public/archive identity mismatch')
        result[alias] = dict(stage_sha256=artifact['sha256'], archive=archive,
                             execution=stage.get('execution'))
    return result


class Transport:
    def __init__(self, user='exouser', timeout=60):
        self.user, self.timeout = user, timeout

    def _run(self, argv, *, timeout=None):
        result = subprocess.run(argv, text=True, capture_output=True,
                                timeout=timeout or self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f'transport command failed: {result.returncode}')
        return result.stdout.strip()

    def remote(self, host, argv, *, cwd=None, env=None, timeout=None):
        prefix = ''.join(f'{key}={shlex.quote(value)} ' for key, value in (env or {}).items())
        command = (f'cd {shlex.quote(cwd)} && ' if cwd else '') + prefix + ' '.join(shlex.quote(str(x)) for x in argv)
        return self._run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                          f"{self.user}@{host['host']}", command], timeout=timeout)

    def copy_from(self, host, remote, local):
        self._run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   f"{self.user}@{host['host']}:{remote}", str(local)], timeout=3600)

    def copy_to(self, host, local, remote):
        self._run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   str(local), f"{self.user}@{host['host']}:{remote}"], timeout=3600)

    def remote_sha(self, host, path):
        script = ('import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); '
                  'print(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size)')
        value = self.remote(host, [host['python'], '-c', script, path]).split()
        return value[0], int(value[1])

    def gpu_pids(self, host):
        return self.remote(host, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])


def remote_score(payload):
    """Run on mpk: verify frozen bytes, extract private files, score without GPU."""
    root = Path(payload['score_root'])
    if root.exists():
        raise FileExistsError('offline scoring root already exists; preserve uncertain attempt')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU process active; defer CPU scorer')
    protocol = Path(payload['remote_protocol'])
    binding = Path(payload['remote_binding'])
    if sha_file(protocol) != payload['protocol_sha256'] or sha_file(binding) != payload['binding_sha256']:
        raise ValueError('remote protocol/binding byte drift')
    root.mkdir(parents=True, exist_ok=False)
    ledgers, private_roots = [], {}
    for alias in ('mpk', 'dllm'):
        item = payload['archives'][alias]
        archive = Path(item['path'])
        if sha_file(archive) != item['sha256'] or archive.stat().st_size != item['bytes']:
            raise ValueError('remote archive byte drift')
        ledger, private = extract_archive(archive, root / alias)
        ledgers.append(ledger)
        private_roots[item['host_ip']] = private
    roots_path = root / 'private_roots.json'
    write_new(roots_path, {k: str(v) for k, v in private_roots.items()})
    summary_path = root / 'first84.redacted_summary.json'
    command = [sys.executable, '-m', 'scripts.v20_score', '--protocol', str(protocol),
               '--binding', str(binding), '--ruler-gold', payload['ruler_gold'],
               '--aime-gold', payload['aime_gold'], '--longbench-gold', payload['longbench_gold'],
               '--ruler-root', payload['ruler_root'], '--private-roots', str(roots_path),
               '--out', str(summary_path)]
    for ledger in ledgers:
        command.extend(('--ledger', str(ledger)))
    scored = subprocess.run(command, capture_output=True, text=True, check=False)
    if scored.returncode:
        (root / 'scorer.stdout.private.txt').write_text(scored.stdout, encoding='utf-8')
        (root / 'scorer.stderr.private.txt').write_text(scored.stderr, encoding='utf-8')
        raise RuntimeError(f'offline scorer failed with exit {scored.returncode}')
    result = json.loads(summary_path.read_text(encoding='utf-8'))
    print(json.dumps(dict(status='complete', summary_path=str(summary_path),
                          summary_sha256=sha_file(summary_path),
                          summary_bytes=summary_path.stat().st_size,
                          first84_recorded=result['first84']['recorded_executions'])))


def wait_for_watch(config, deadline, *, clock=time.time, sleep=time.sleep):
    """Read only the already-running local watcher outcome; never poll hosts."""
    path = Path(config['watch_dir']) / 'initial_001.outcome.json'
    while True:
        if path.is_file():
            outcome = json.loads(path.read_text(encoding='utf-8'))
            if outcome.get('schema') != 'v20_stage_watch_v1' or outcome.get('stage') != 'initial_001':
                raise ValueError('local watcher outcome identity mismatch')
            if not outcome.get('all_complete'):
                return dict(status='blocked', reason='existing watcher terminal outcome is not all-complete',
                            host_status={a: outcome.get('hosts', {}).get(a, {}).get('status')
                                         for a in ('mpk', 'dllm')})
            return None
        if deadline is None or clock() >= deadline:
            return dict(status='waiting_stage', reason='local watcher outcome absent at deadline')
        sleep(min(30, max(0, deadline - clock())))


def collect(config, hosts, out_dir, *, transport, wait_until_epoch=None,
            clock=time.time, sleep=time.sleep):
    validate_config(config, hosts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / 'initial_001.collect.status.json'
    if status_path.exists():
        raise FileExistsError('old collection outcome exists')
    lock = out_dir / 'initial_001.collect.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    status = dict(schema=SCHEMA, stage='initial_001', status='started')
    try:
        pending = wait_for_watch(config, wait_until_epoch, clock=clock, sleep=sleep)
        if pending:
            status.update(pending)
            write_new(status_path, status)
            return status
        receipts = watch_receipts(config, hosts)
        status['stage_sha256'] = {a: r['stage_sha256'] for a, r in receipts.items()}
        archives = {}
        for alias, host in hosts.items():
            item = receipts[alias]['archive']
            local = out_dir / f'{alias}.initial_001.private.tar.gz'
            remote = item['path']
            if transport.remote_sha(host, remote) != (item['sha256'], item['bytes']):
                raise ValueError(f'{alias} remote archive drift')
            transport.copy_from(host, remote, local)
            if sha_file(local) != item['sha256'] or local.stat().st_size != item['bytes']:
                raise ValueError(f'{alias} local archive drift')
            with tarfile.open(local, 'r:gz') as archive:
                safe_archive_members(archive)
            archives[alias] = dict(path=str(local), sha256=item['sha256'], bytes=item['bytes'],
                                   host_ip=host['host'])
        mpk = hosts['mpk']
        wait_seconds = int(config.get('gpu_wait_seconds', 0))
        waited = 0
        while transport.gpu_pids(mpk):
            if waited >= wait_seconds:
                status.update(status='waiting_gpu', reason='mpk GPU process active; no scorer dispatched')
                write_new(status_path, status)
                return status
            sleep(min(60, wait_seconds - waited))
            waited += min(60, wait_seconds - waited)
        score_root = str(PurePosixPath(mpk['root']) / 'offline_scoring' / 'initial_001')
        remote_dllm = str(PurePosixPath(mpk['root']) / 'offline_scoring' / 'dllm.initial_001.private.tar.gz')
        remote_payload = str(PurePosixPath(mpk['root']) / 'offline_scoring' / 'initial_001.remote_payload.json')
        remote_helper = str(PurePosixPath(mpk['root']) / 'offline_scoring' / 'v20_collect_score.py')
        prepare = ('import pathlib,sys; root=pathlib.Path(sys.argv[1]); '
                   'targets=[pathlib.Path(x) for x in sys.argv[2:]]; '
                   'assert not any(p.exists() for p in targets), "offline scoring output exists"; '
                   'root.mkdir(parents=True,exist_ok=True)')
        transport.remote(mpk, [mpk['python'], '-c', prepare,
                               str(PurePosixPath(mpk['root']) / 'offline_scoring'),
                               score_root, remote_dllm, remote_payload, remote_helper])
        transport.copy_to(mpk, archives['dllm']['path'], remote_dllm)
        if transport.remote_sha(mpk, remote_dllm) != (archives['dllm']['sha256'], archives['dllm']['bytes']):
            raise ValueError('cross-host archive transfer drift')
        payload = {key: config[key] for key in ('binding_sha256', 'protocol_sha256', 'remote_protocol',
                   'remote_binding', 'ruler_gold', 'aime_gold', 'longbench_gold', 'ruler_root')}
        payload.update(score_root=score_root, archives={
            'mpk': dict(archives['mpk'], path=receipts['mpk']['archive']['path']),
            'dllm': dict(archives['dllm'], path=remote_dllm)})
        payload_path = out_dir / 'initial_001.remote_payload.json'
        write_new(payload_path, payload)
        transport.copy_to(mpk, payload_path, remote_payload)
        transport.copy_to(mpk, Path(__file__), remote_helper)
        result = json.loads(transport.remote(mpk, [mpk['python'], remote_helper, 'remote-score',
                                              '--payload', remote_payload],
                                     cwd=config['remote_deploy'], env=mpk['env'], timeout=3600))
        if result.get('status') != 'complete':
            raise ValueError('offline scorer did not finish')
        summary_local = out_dir / 'initial_001.redacted_summary.json'
        transport.copy_from(mpk, result['summary_path'], summary_local)
        if sha_file(summary_local) != result['summary_sha256'] or summary_local.stat().st_size != result['summary_bytes']:
            raise ValueError('scored summary transfer drift')
        status.update(status='complete', archives={a: {'sha256': x['sha256'], 'bytes': x['bytes']}
                                                  for a, x in archives.items()},
                      summary_sha256=result['summary_sha256'],
                      first84_recorded=result['first84_recorded'])
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__)
        write_new(status_path, status)
        raise
    write_new(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    local = sub.add_parser('collect')
    local.add_argument('--config', type=Path, required=True)
    local.add_argument('--out-dir', type=Path, required=True)
    local.add_argument('--wait-until-epoch', type=float,
                       help='bounded read-only wait for the existing local watcher outcome')
    remote = sub.add_parser('remote-score')
    remote.add_argument('--payload', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'remote-score':
        remote_score(json.loads(args.payload.read_text()))
        return 0
    config = json.loads(args.config.read_text(encoding='utf-8'))
    hosts = json.loads(Path(config['hosts_path']).read_text(encoding='utf-8'))
    result = collect(config, hosts, args.out_dir, transport=Transport(),
                     wait_until_epoch=args.wait_until_epoch)
    print(json.dumps(dict(status=result['status'], first84_recorded=result.get('first84_recorded'))))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
