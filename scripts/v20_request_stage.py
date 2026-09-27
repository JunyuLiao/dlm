"""One-shot v20 bridge/request wrapper with redacted terminal and private archive.

The child argv is frozen in a JSON config and run without a shell. This wrapper
does not retry/resume, enforce a GPU budget, or inspect raw receipt contents.
Place its output directory outside Git; the archive contains only the named
private receipt directory and the named JSONL ledger/bridge summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time

SCHEMA = 'v20_request_stage_v1'
STAGE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
FORBIDDEN_PARTS = ('gold', 'manifest', '.env', 'credential', 'secret')


def inside_git(path: Path) -> bool:
    return any((parent / '.git').exists() for parent in (path, *path.parents))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _option(argv: list[str], name: str) -> str:
    positions = [i for i, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f'child argv requires one {name}')
    return argv[positions[0] + 1]


def validate_config(config: dict) -> dict:
    if config.get('schema') != SCHEMA or config.get('mode') not in ('bridge', 'request'):
        raise ValueError('wrong one-shot stage schema/mode')
    stage = config.get('stage_name')
    if not isinstance(stage, str) or not STAGE_NAME.fullmatch(stage) or '..' in stage:
        raise ValueError('unsafe stage name')
    argv = config.get('argv')
    if not isinstance(argv, list) or not argv or any(type(x) is not str or not x for x in argv):
        raise ValueError('child argv must be a nonempty string array')
    expected_module = 'scripts.v20_bridge' if config['mode'] == 'bridge' else 'scripts.v20_run'
    prefix = argv[1:4] if config['mode'] == 'bridge' else argv[1:3]
    if prefix != (['-m', expected_module, 'run'] if config['mode'] == 'bridge'
                  else ['-m', expected_module]):
        raise ValueError('child argv must invoke the pinned v20 worker module')
    if Path(argv[0]).name not in ('python', 'python3', 'python3.10', 'python3.11'):
        raise ValueError('child argv must begin with a Python executable')
    private, ledger, output = (Path(config[key]).resolve() for key in
                               ('private_dir', 'ledger', 'output_dir'))
    ledger_option = '--summary' if config['mode'] == 'bridge' else '--ledger'
    if Path(_option(argv, '--private')).resolve() != private or Path(_option(argv, ledger_option)).resolve() != ledger:
        raise ValueError('archive paths differ from child output arguments')
    for name in ('--protocol', '--binding', '--host'):
        _option(argv, name)
    if config['mode'] == 'request':
        _option(argv, '--stage')
    cwd = Path(config.get('cwd', os.getcwd())).resolve()
    if not cwd.is_dir() or output == private or private in output.parents or output in private.parents:
        raise ValueError('invalid stage working/output/private directories')
    if inside_git(output):
        raise ValueError('private archive output must be outside Git')
    return dict(stage=stage, mode=config['mode'], argv=argv, private=private,
                ledger=ledger, output=output, cwd=cwd)


def public_identity(argv: list[str], cwd: Path) -> dict:
    protocol, binding = Path(_option(argv, '--protocol')), Path(_option(argv, '--binding'))
    bound = json.loads(binding.read_text(encoding='utf-8'))
    host = _option(argv, '--host')
    host_configs = bound.get('host_configs', {}).get(host, {})
    config_hashes, sources, metadata = {}, {}, {}
    for dataset, arms in host_configs.items():
        for arm, item in arms.items():
            key = f'{dataset}/{arm}'
            config_hashes[key] = item['sha256']
            actual = digest(Path(item['path']))
            if actual != item['sha256']:
                raise ValueError(f'bound execution config byte drift: {key}')
            values = json.loads(Path(item['path']).read_text(encoding='utf-8'))
            for name, sha in values.get('source_hashes', {}).items():
                if name in sources and sources[name] != sha:
                    raise ValueError('conflicting source hash values')
                sources[name] = sha
            for name, sha in values.get('model_metadata_hashes', {}).items():
                if name in metadata and metadata[name] != sha:
                    raise ValueError('conflicting model metadata hash values')
                metadata[name] = sha
    worker = cwd / 'scripts' / ('v20_bridge.py' if argv[2] == 'scripts.v20_bridge'
                                else 'v20_run.py')
    worker_hash = digest(worker)
    if argv[2] == 'scripts.v20_run':
        frozen_workers = [sha for name, sha in sources.items()
                          if Path(name).name == 'v20_run.py']
        if not frozen_workers or any(sha != worker_hash for sha in frozen_workers):
            raise ValueError('actual child worker source differs from frozen binding')
    return dict(protocol_sha256=digest(protocol), binding_sha256=digest(binding),
                wrapper_sha256=digest(Path(__file__)), worker_sha256=worker_hash,
                execution_config_sha256=config_hashes, source_sha256=sources,
                model_metadata_sha256=metadata, host=host)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def summarize_ledger(mode: str, path: Path) -> dict:
    if not path.is_file():
        return dict(status='absent', events=0)
    rows = _jsonl(path)
    if mode == 'bridge':
        starts = [row for row in rows if row.get('event') == 'start']
        arms = [row for row in rows if row.get('event') == 'arm']
        expected = starts[0].get('arms', []) if len(starts) == 1 else []
        names = [row.get('arm') for row in arms]
        counts = dict(Counter(str(row.get('status', 'missing')) for row in arms))
        complete = bool(expected) and len(expected) == len(arms) and set(expected) == set(names) and len(names) == len(set(names))
        status = 'complete' if complete and counts == {'ok': len(expected)} else 'failed'
        return dict(status=status, events=len(rows), expected_arms=len(expected), arm_rows=len(arms),
                    arm_status_counts=counts, coverage_complete=complete)
    starts = [row for row in rows if row.get('event') == 'start']
    runs = [row for row in rows if row.get('event') == 'run']
    ends = [row for row in rows if row.get('event') == 'worker_end']
    roles = dict(Counter(str(row.get('role', 'missing')) for row in runs))
    outcomes = dict(Counter('ok' if row.get('ok') is True else 'failed' for row in runs))
    return dict(status=ends[-1].get('status', 'missing') if ends else 'missing_worker_end',
                events=len(rows), start_events=len(starts), run_events=len(runs),
                role_counts=roles, run_status_counts=outcomes,
                all_runs_ok=bool(runs) and outcomes.get('failed', 0) == 0,
                worker_end_events=len(ends))


def _safe_relative(path: Path) -> PurePosixPath:
    parts = path.parts
    if not parts or any(part in ('', '.', '..') or any(word in part.casefold() for word in FORBIDDEN_PARTS)
                        for part in parts):
        raise ValueError('archive contains an unsafe/forbidden private path')
    return PurePosixPath(*parts)


def archive_private(private: Path, ledger: Path, destination: Path) -> dict | None:
    if not private.is_dir() or not ledger.is_file():
        return None
    if private.is_symlink():
        raise ValueError('private archive refuses symlink root')
    if destination.exists():
        raise FileExistsError(destination)
    files = sorted(path for path in private.rglob('*') if path.is_file() or path.is_symlink())
    members = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError('private archive refuses symlink/nonregular output')
        relative = _safe_relative(path.relative_to(private))
        members.append((path, str(PurePosixPath('private') / relative)))
    if ledger.is_symlink():
        raise ValueError('ledger archive refuses symlink')
    _safe_relative(Path(ledger.name))
    members.append((ledger, str(PurePosixPath('ledger') / ledger.name)))
    temp = destination.with_name(destination.name + '.writing')
    with temp.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for source, name in members:
                info = archive.gettarinfo(str(source), arcname=name)
                if not info.isfile():
                    raise ValueError('archive member is not a regular file')
                with source.open('rb') as data:
                    archive.addfile(info, data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, destination)
    return dict(path=str(destination), sha256=digest(destination), bytes=destination.stat().st_size,
                private_files=len(files), members=len(members))


def atomic_final(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + '.writing')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run(config_path: Path) -> int:
    config_bytes = config_path.read_bytes()
    paths = validate_config(json.loads(config_bytes))
    output, stage = paths['output'], paths['stage']
    output.mkdir(parents=True, exist_ok=True)
    started = output / f'{stage}.started.json'
    final = output / f'{stage}.json'
    archive = output / f'{stage}.private.tar.gz'
    if final.exists() or archive.exists():
        raise FileExistsError('stage final/archive already exists; never retry')
    start_epoch = time.time()
    with started.open('x', encoding='utf-8') as stream:
        json.dump(dict(schema=SCHEMA, stage_name=stage, status='started',
                       start_epoch=start_epoch, config_sha256=hashlib.sha256(config_bytes).hexdigest()), stream)
        stream.write('\n')
    receipt = dict(schema=SCHEMA, stage_name=stage, mode=paths['mode'],
                   start_epoch=start_epoch, config_sha256=hashlib.sha256(config_bytes).hexdigest())
    child_code = None
    try:
        receipt['public_identity'] = public_identity(paths['argv'], paths['cwd'])
        log = output / f'{stage}.child.log'
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.run(paths['argv'], cwd=paths['cwd'], stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
        child_code = child.returncode
        receipt['child_returncode'] = child_code
    except BaseException as exc:
        receipt['wrapper_error_type'] = type(exc).__name__
    finally:
        try:
            receipt['execution'] = summarize_ledger(paths['mode'], paths['ledger'])
            receipt['private_archive'] = archive_private(paths['private'], paths['ledger'], archive)
        except BaseException as exc:
            receipt['finalization_error_type'] = type(exc).__name__
        receipt['end_epoch'] = time.time()
        receipt['status'] = ('complete' if child_code == 0 and
                             receipt.get('execution', {}).get('status') == 'complete' and
                             receipt.get('private_archive') is not None and
                             not receipt.get('finalization_error_type') else 'failed')
        atomic_final(final, receipt)
    return 0 if receipt['status'] == 'complete' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    raise SystemExit(run(args.config))


if __name__ == '__main__':
    main()
