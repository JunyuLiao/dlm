"""After GPU stages, archive private receipts and score offline on pinned mpk CPU.

No completion or gold values are sent to stdout. The Windows archive stays
outside Git; the redacted summaries remain immutable on the scorer host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

OLD = ('exouser@149.165.151.254', '/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926')
NEW = ('exouser@149.165.159.64', '/home/exouser/dyh/junyu_frontier_v18_20260926')
PY = '/home/exouser/miniconda3/envs/ljy_dlm/bin/python'
RULER_ROOT = '/home/exouser/dyh/paper_first_baseline_anchor_20260923/ruler_official'
RULER_GOLD = '/home/exouser/dyh/numerical_qk_reuse_recovery_20260924/code_cp1/results/query_adaptive_v3/configs/final_manifest.json'
AIME_GOLD = '/media/volume/dllm-1/dyh/numerical_qk_global_multiseed_20260925/private/manifest_all30.json'
GOLD_SHA = {'ruler': 'c0d87ef6f81e07d04e13b34ab723a8680ac4965e650ed298c6bb0f839ef7cbfb',
            'aime': 'a253c357e1a2a2634d895e91620d7e91b1b10640755088a8d4ca7d2b0250b57c'}
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15')


def run_ssh(host, command, *, timeout=300):
    return subprocess.check_output(['ssh', *SSH_OPTIONS, host, command], text=True, timeout=timeout).strip()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_archive(path):
    with tarfile.open(path, 'r:gz') as archive:
        entries = archive.getmembers()
        if not entries or any(not (item.isfile() or item.isdir()) or
                              PurePosixPath(item.name).parts[0] not in ('ruler', 'aime') or
                              PurePosixPath(item.name).is_absolute() or '..' in PurePosixPath(item.name).parts
                              for item in entries):
            raise ValueError('private archive contains an unexpected entry')
        return len(entries)


def archive_private(local_path):
    if local_path.exists():
        return dict(path=str(local_path), sha256=file_hash(local_path), members=inspect_archive(local_path))
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = f'tar -C {shlex.quote(NEW[1] + "/private_eval")} -czf - ruler aime'
    with local_path.open('xb') as output:
        subprocess.run(['ssh', *SSH_OPTIONS, NEW[0], command], stdout=output,
                       check=True, timeout=3600)
    return dict(path=str(local_path), sha256=file_hash(local_path), members=inspect_archive(local_path))


def transfer_archive(local_path, digest):
    remote_archive = OLD[1] + '/scoring/dllm_private.tar.gz'
    remote_root = OLD[1] + '/scoring/dllm_private'
    run_ssh(OLD[0], f'mkdir -p {shlex.quote(OLD[1] + "/scoring")}')
    existing = run_ssh(OLD[0], f'if test -f {shlex.quote(remote_archive)}; then sha256sum {shlex.quote(remote_archive)}; fi')
    if existing:
        if existing.split()[0] != digest:
            raise ValueError('existing remote private archive differs; preserve both')
    else:
        subprocess.run(['scp', *SSH_OPTIONS, str(local_path), f'{OLD[0]}:{remote_archive}'],
                       check=True, timeout=3600)
    actual = run_ssh(OLD[0], f'sha256sum {shlex.quote(remote_archive)}').split()[0]
    if actual != digest:
        raise ValueError('remote private archive hash mismatch')
    marker = remote_root + '/.archive_sha256'
    prior = run_ssh(OLD[0], f'if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi')
    if prior:
        if prior != digest:
            raise ValueError('remote extraction marker differs')
    else:
        if run_ssh(OLD[0], f'if test -e {shlex.quote(remote_root)}; then echo exists; fi'):
            raise ValueError('unmarked private extraction exists; inspect before retry')
        run_ssh(OLD[0], f'mkdir {shlex.quote(remote_root)} && tar -xzf {shlex.quote(remote_archive)} '
                         f'-C {shlex.quote(remote_root)} && printf %s {shlex.quote(digest)} > {shlex.quote(marker)}',
                timeout=3600)
    return remote_root


def score_on_old(deploy, private_root):
    code = OLD[1] + '/deploy/' + deploy
    scoring = OLD[1] + '/scoring'
    outputs = {}
    for dataset, protocol_name, gold in (
            ('ruler', 'ruler4k_primary_protocol.json', RULER_GOLD),
            ('aime', 'aime26_primary_protocol.json', AIME_GOLD)):
        if run_ssh(OLD[0], f'sha256sum {shlex.quote(gold)}').split()[0] != GOLD_SHA[dataset]:
            raise ValueError('original scorer-only gold source byte drift')
        protocol = OLD[1] + '/evaluation/' + protocol_name
        lock = scoring + '/' + dataset + '_scorer_lock.json'
        output = scoring + '/' + dataset + '_redacted_summary.json'
        command = [PY, '-m', 'scripts.v18_summarize', '--protocol', protocol,
                   '--scorer-lock', lock]
        if dataset == 'ruler':
            command.extend(['--ruler-root', RULER_ROOT])
        prefix = f'cd {shlex.quote(code)} && PYTHONPATH=src:. HF_HUB_OFFLINE=1 '
        freeze = prefix + ' '.join(shlex.quote(part) for part in command + ['--freeze-lock'])
        run_ssh(OLD[0], freeze, timeout=300)
        execute = command + ['--ledger', scoring + f'/ledgers/mpk_{dataset}.jsonl',
                             scoring + f'/ledgers/dllm_{dataset}.jsonl', '--gold', gold,
                             '--private-root', f'dllm={private_root}/{dataset}', '--out', output]
        run_ssh(OLD[0], prefix + ' '.join(shlex.quote(part) for part in execute), timeout=3600)
        outputs[dataset] = output
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer-deploy', required=True)
    parser.add_argument('--archive-dir', type=Path, default=Path('E:/dlm/v18_private'))
    args = parser.parse_args()
    local = args.archive_dir / 'dllm_private_eval.tar.gz'
    archive = archive_private(local)
    remote_root = transfer_archive(local, archive['sha256'])
    # Freeze copies after all GPU writers have exited. Never use a live ledger.
    run_ssh(OLD[0], f'mkdir -p {shlex.quote(OLD[1] + "/scoring/ledgers")}')
    for dataset in ('ruler', 'aime'):
        for host in ('mpk', 'dllm'):
            src = OLD[1] + f'/evaluation/ledgers/{host}_{dataset}.jsonl'
            dst = OLD[1] + f'/scoring/ledgers/{host}_{dataset}.jsonl'
            run_ssh(OLD[0], f'cp -n {shlex.quote(src)} {shlex.quote(dst)}')
            if run_ssh(OLD[0], f'sha256sum {shlex.quote(src)}').split()[0] != run_ssh(
                    OLD[0], f'sha256sum {shlex.quote(dst)}').split()[0]:
                raise ValueError('scoring ledger snapshot differs from completed evaluation')
    outputs = score_on_old(args.scorer_deploy, remote_root)
    print(json.dumps({'archive_sha256': archive['sha256'], 'archive_members': archive['members'],
                      'redacted_summaries': outputs}, sort_keys=True))


if __name__ == '__main__':
    main()
