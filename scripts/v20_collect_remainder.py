"""One-shot CPU collection/scoring of first84 plus a terminal remainder stage.

Consumes only the existing local watcher. A finalized failed wrapper with a
verified archive is scoreable as partial evidence; a missing archive is blocked.
No generation job is launched or retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

try:
    from scripts import v20_collect_score as C
except ImportError:
    # CP5 intentionally predates the collector. The first84 collector was
    # copied beside this helper in the private offline_scoring directory.
    import importlib.util
    sibling = Path(__file__).with_name('v20_collect_score.py')
    spec = importlib.util.spec_from_file_location('v20_collect_score_private', sibling)
    if spec is None or spec.loader is None:
        raise
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)


SCHEMA = 'v20_collect_remainder_v1'
STAGE = 'remainder_001'


def validate(config, hosts):
    if config.get('schema') != SCHEMA or config.get('stage') != STAGE:
        raise ValueError('wrong remainder collector schema/stage')
    base = dict(config, schema=C.SCHEMA, stage='initial_001')
    C.validate_config(base, hosts)
    for key in ('initial_collect_dir', 'initial_watch_dir'):
        if not Path(config.get(key, '')).is_dir():
            raise ValueError(f'{key} absent')
    if (set(config.get('initial_remote_archives', {})) != {'mpk', 'dllm'} or
            set(config.get('initial_archive_sha256', {})) != {'mpk', 'dllm'}):
        raise ValueError('initial remote archive paths and hashes required')
    for alias, path in config['initial_remote_archives'].items():
        if not isinstance(path, str) or not path.startswith(hosts['mpk']['root'] + '/') or '..' in PurePosixPath(path).parts:
            raise ValueError(f'unsafe initial remote archive: {alias}')
        if (not isinstance(config['initial_archive_sha256'][alias], str) or
                len(config['initial_archive_sha256'][alias]) != 64):
            raise ValueError(f'invalid initial archive hash: {alias}')


def terminal_remainder(config, hosts):
    path = Path(config['watch_dir']) / f'{STAGE}.outcome.json'
    outcome = json.loads(path.read_text(encoding='utf-8'))
    if outcome.get('schema') != 'v20_stage_watch_v1' or outcome.get('stage') != STAGE:
        raise ValueError('remainder watcher identity mismatch')
    binding = json.loads(Path(config['binding_local']).read_text(encoding='utf-8'))
    result = {}
    for alias, host in hosts.items():
        row = outcome.get('hosts', {}).get(alias, {})
        if row.get('status') not in ('complete', 'failed'):
            return None, f'{alias} has no finalized wrapper'
        artifact = row.get('artifacts', {}).get(f'{STAGE}.json')
        if not artifact:
            return None, f'{alias} finalized wrapper absent; hard kill or incomplete finalization'
        path = Path(artifact['path'])
        if C.sha_file(path) != artifact.get('sha256') or path.stat().st_size != artifact.get('bytes'):
            raise ValueError(f'{alias} watcher final artifact hash drift')
        receipt = json.loads(path.read_text(encoding='utf-8'))
        public = receipt.get('public_identity', {})
        expected_configs = {f'{dataset}/{arm}': item['sha256']
                            for dataset, arms in binding.get('host_configs', {}).get(host['host'], {}).items()
                            for arm, item in arms.items()}
        archive = receipt.get('private_archive') or {}
        expected_archive = str(PurePosixPath(host['root']) / 'primary' / f'{STAGE}.private.tar.gz')
        if (receipt.get('schema') != 'v20_request_stage_v1' or
                receipt.get('stage_name') != STAGE or receipt.get('mode') != 'request' or
                receipt.get('status') not in ('complete', 'failed') or
                public.get('host') != host['host'] or
                public.get('binding_sha256') != config['binding_sha256'] or
                public.get('protocol_sha256') != config['protocol_sha256'] or
                public.get('execution_config_sha256') != expected_configs or
                archive.get('path') != expected_archive or
                not isinstance(archive.get('bytes'), int) or archive['bytes'] <= 0 or
                not isinstance(archive.get('sha256'), str) or len(archive['sha256']) != 64):
            return None, f'{alias} wrapper/archive identity incomplete'
        if receipt.get('finalization_error_type'):
            return None, f'{alias} archive finalization failed'
        result[alias] = dict(stage_status=row['status'], wrapper_status=receipt['status'],
                             execution_status=receipt.get('execution', {}).get('status'),
                             stage_sha256=artifact['sha256'], archive=archive)
    return result, None


def initial_archives(config, hosts):
    collection = Path(config['initial_collect_dir'])
    status = json.loads((collection / 'initial_001.collect.status.json').read_text(encoding='utf-8'))
    if status.get('schema') != C.SCHEMA or status.get('status') != 'complete' or status.get('first84_recorded') != 84:
        raise ValueError('first84 collection not complete')
    first_watch = dict(config, watch_dir=config['initial_watch_dir'])
    initial_receipts = C.watch_receipts(first_watch, hosts)
    found = {}
    for alias in hosts:
        archive = collection / f'{alias}.initial_001.private.tar.gz'
        expected = status.get('archives', {}).get(alias, {})
        receipt = initial_receipts[alias]['archive']
        if (not archive.is_file() or C.sha_file(archive) != expected.get('sha256') or
                archive.stat().st_size != expected.get('bytes') or
                expected.get('sha256') != config['initial_archive_sha256'][alias] or
                (expected.get('sha256'), expected.get('bytes')) != (receipt['sha256'], receipt['bytes'])):
            raise ValueError(f'{alias} initial archive identity drift')
        with tarfile.open(archive, 'r:gz') as stream:
            C.safe_archive_members(stream)
        found[alias] = dict(path=str(archive), sha256=expected['sha256'], bytes=expected['bytes'],
                            host_ip=hosts[alias]['host'])
    return found


def merge_private_trees(roots, destination):
    """Exclusive regular-file copy; duplicate cell IDs are never overwritten."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for root in roots:
        root = Path(root)
        for source in sorted(root.rglob('*')):
            if source.is_symlink():
                raise ValueError('private tree contains symlink')
            if source.is_dir():
                continue
            if not source.is_file():
                raise ValueError('private tree contains nonregular entry')
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
    return destination


def score_completion(full, wrapper_status):
    """Ledger presence is weaker than a valid paired request block."""
    recorded = full.get('recorded_executions')
    all_rows = full.get('executions_complete') is True
    planned_blocks = full.get('planned_blocks')
    valid_blocks = full.get('complete_valid_pair_blocks')
    complete = (recorded == 700 and all_rows and planned_blocks == 50 and
                valid_blocks == 50 and all(value == 'complete' for value in wrapper_status.values()))
    return dict(status='scored_complete' if complete else 'scored_partial',
                full_recorded=recorded, recorded_all_rows=all_rows,
                complete_valid_pair_blocks=valid_blocks, planned_blocks=planned_blocks)


def remote_score(payload):
    root = Path(payload['score_root'])
    if root.exists():
        raise FileExistsError('remainder offline scoring root exists; preserve uncertain attempt')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU process active; defer CPU scorer')
    protocol, binding = Path(payload['remote_protocol']), Path(payload['remote_binding'])
    if C.sha_file(protocol) != payload['protocol_sha256'] or C.sha_file(binding) != payload['binding_sha256']:
        raise ValueError('remote binding/protocol drift')
    root.mkdir(parents=True, exist_ok=False)
    ledgers, private_roots = [], {}
    for alias in ('mpk', 'dllm'):
        components = []
        for stage in ('initial', 'remainder'):
            item = payload['archives'][alias][stage]
            archive = Path(item['path'])
            if C.sha_file(archive) != item['sha256'] or archive.stat().st_size != item['bytes']:
                raise ValueError(f'{alias}/{stage} archive drift')
            ledger, private = C.extract_archive(archive, root / f'{alias}_{stage}')
            ledgers.append(ledger)
            components.append(private)
        private_roots[payload['archives'][alias]['initial']['host_ip']] = merge_private_trees(
            components, root / 'merged_private' / alias)
    roots_path = root / 'private_roots.json'
    C.write_new(roots_path, {host: str(path) for host, path in private_roots.items()})
    summary_path = root / 'remainder.redacted_summary.json'
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
    completion = score_completion(result['full'], payload['wrapper_status'])
    print(json.dumps(dict(transport_status='scored', summary_path=str(summary_path),
                          summary_sha256=C.sha_file(summary_path), summary_bytes=summary_path.stat().st_size,
                          **completion)))


def collect(config, hosts, out_dir, *, transport, wait_until_epoch=None,
            clock=time.time, sleep=time.sleep):
    validate(config, hosts)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f'{STAGE}.collect.status.json'
    if status_path.exists():
        raise FileExistsError('prior remainder outcome exists')
    with (out_dir / f'{STAGE}.collect.lock').open('x', encoding='utf-8') as stream:
        stream.write(f'pid={os.getpid()}\n')
    status = dict(schema=SCHEMA, stage=STAGE, status='started')
    try:
        watch = dict(config, watch_dir=config['watch_dir'])
        # Reuse the bounded local-only wait, with this stage's frozen outcome.
        while True:
            outcome_path = Path(watch['watch_dir']) / f'{STAGE}.outcome.json'
            if outcome_path.is_file():
                break
            if wait_until_epoch is None or clock() >= wait_until_epoch:
                status.update(status='waiting_stage', reason='local remainder watcher outcome absent at deadline')
                C.write_new(status_path, status)
                return status
            sleep(min(30, max(0, wait_until_epoch-clock())))
        receipts, reason = terminal_remainder(config, hosts)
        if receipts is None:
            status.update(status='blocked', reason=reason)
            C.write_new(status_path, status)
            return status
        first = initial_archives(config, hosts)
        remainder = {}
        for alias, host in hosts.items():
            archive = receipts[alias]['archive']
            if transport.remote_sha(host, archive['path']) != (archive['sha256'], archive['bytes']):
                raise ValueError(f'{alias} remote remainder archive drift')
            local = out_dir / f'{alias}.{STAGE}.private.tar.gz'
            transport.copy_from(host, archive['path'], local)
            if C.sha_file(local) != archive['sha256'] or local.stat().st_size != archive['bytes']:
                raise ValueError(f'{alias} local remainder archive drift')
            with tarfile.open(local, 'r:gz') as stream:
                C.safe_archive_members(stream)
            remainder[alias] = dict(path=str(local), sha256=archive['sha256'],
                                    bytes=archive['bytes'], host_ip=host['host'])
        mpk = hosts['mpk']
        wait_seconds, waited = int(config.get('gpu_wait_seconds', 0)), 0
        while transport.gpu_pids(mpk):
            if waited >= wait_seconds:
                status.update(status='waiting_gpu', reason='mpk GPU occupied; scorer not dispatched')
                C.write_new(status_path, status)
                return status
            span = min(30, wait_seconds - waited)
            sleep(span)
            waited += span
        for alias in ('mpk', 'dllm'):
            remote = config['initial_remote_archives'][alias]
            if transport.remote_sha(mpk, remote) != (first[alias]['sha256'], first[alias]['bytes']):
                raise ValueError(f'{alias} initial archive on scorer host drift')
        offline = str(PurePosixPath(mpk['root']) / 'offline_scoring')
        score_root = str(PurePosixPath(offline) / STAGE)
        remote_dllm_remainder = str(PurePosixPath(offline) / f'dllm.{STAGE}.private.tar.gz')
        remote_payload = str(PurePosixPath(offline) / f'{STAGE}.remote_payload.json')
        remote_helper = str(PurePosixPath(offline) / 'v20_collect_remainder.py')
        prior_helper = str(PurePosixPath(offline) / 'v20_collect_score.py')
        local_helper = Path(C.__file__)
        if transport.remote_sha(mpk, prior_helper) != (C.sha_file(local_helper), local_helper.stat().st_size):
            raise ValueError('first84 collector helper source drift on scorer host')
        prepare = ('import pathlib,sys; base=pathlib.Path(sys.argv[1]); '
                   'targets=[pathlib.Path(x) for x in sys.argv[2:]]; '
                   'assert not any(p.exists() for p in targets), "remainder scoring output exists"; '
                   'base.mkdir(parents=True,exist_ok=True)')
        transport.remote(mpk, [mpk['python'], '-c', prepare, offline,
                               score_root, remote_dllm_remainder, remote_payload, remote_helper])
        transport.copy_to(mpk, remainder['dllm']['path'], remote_dllm_remainder)
        if transport.remote_sha(mpk, remote_dllm_remainder) != (
                remainder['dllm']['sha256'], remainder['dllm']['bytes']):
            raise ValueError('dllm remainder cross-host archive drift')
        payload = {key: config[key] for key in ('binding_sha256', 'protocol_sha256', 'remote_protocol',
                   'remote_binding', 'ruler_gold', 'aime_gold', 'longbench_gold', 'ruler_root')}
        payload.update(score_root=score_root,
                       wrapper_status={a: x['wrapper_status'] for a, x in receipts.items()}, archives={
            'mpk': {'initial': dict(first['mpk'], path=config['initial_remote_archives']['mpk']),
                    'remainder': dict(remainder['mpk'], path=receipts['mpk']['archive']['path'])},
            'dllm': {'initial': dict(first['dllm'], path=config['initial_remote_archives']['dllm']),
                     'remainder': dict(remainder['dllm'], path=remote_dllm_remainder)}})
        payload_path = out_dir / f'{STAGE}.remote_payload.json'
        C.write_new(payload_path, payload)
        transport.copy_to(mpk, payload_path, remote_payload)
        transport.copy_to(mpk, Path(__file__), remote_helper)
        result = json.loads(transport.remote(mpk, [mpk['python'], remote_helper, 'remote-score',
                                              '--payload', remote_payload],
                                     cwd=config['remote_deploy'], env=mpk['env'], timeout=3600))
        if result.get('transport_status') != 'scored' or result.get('status') not in ('scored_complete', 'scored_partial'):
            raise ValueError('remainder offline scorer did not finish')
        summary_local = out_dir / f'{STAGE}.redacted_summary.json'
        transport.copy_from(mpk, result['summary_path'], summary_local)
        if C.sha_file(summary_local) != result['summary_sha256'] or summary_local.stat().st_size != result['summary_bytes']:
            raise ValueError('remainder scored summary transfer drift')
        status.update(status=result['status'],
                      wrapper_status={a: x['wrapper_status'] for a, x in receipts.items()},
                      execution_status={a: x['execution_status'] for a, x in receipts.items()},
                      stage_sha256={a: x['stage_sha256'] for a, x in receipts.items()},
                      archive_sha256={a: x['sha256'] for a, x in remainder.items()},
                      summary_sha256=result['summary_sha256'], full_recorded=result['full_recorded'],
                      recorded_all_rows=result['recorded_all_rows'],
                      complete_valid_pair_blocks=result['complete_valid_pair_blocks'],
                      planned_blocks=result['planned_blocks'])
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__)
        C.write_new(status_path, status)
        raise
    C.write_new(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    local = sub.add_parser('collect')
    local.add_argument('--config', type=Path, required=True)
    local.add_argument('--out-dir', type=Path, required=True)
    local.add_argument('--wait-until-epoch', type=float)
    remote = sub.add_parser('remote-score')
    remote.add_argument('--payload', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'remote-score':
        remote_score(json.loads(args.payload.read_text(encoding='utf-8')))
        return 0
    config = json.loads(args.config.read_text(encoding='utf-8'))
    hosts = json.loads(Path(config['hosts_path']).read_text(encoding='utf-8'))
    result = collect(config, hosts, args.out_dir, transport=C.Transport(),
                     wait_until_epoch=args.wait_until_epoch)
    print(json.dumps(dict(status=result['status'], full_recorded=result.get('full_recorded'))))
    return 0 if result['status'] in ('scored_complete', 'scored_partial') else 2


if __name__ == '__main__':
    raise SystemExit(main())
