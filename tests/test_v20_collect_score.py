"""CPU tests for terminal-only first84 collection and safe private archives."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import v20_collect_score as C


class CollectScoreTests(unittest.TestCase):
    def fixture(self, root):
        hosts = {a: dict(host=f'10.0.0.{i}', root=f'/remote/{a}', python='/python', env={})
                 for i, a in enumerate(('mpk', 'dllm'), 1)}
        watch = root / 'watch'
        watch.mkdir()
        binding_local = root / 'binding.json'
        binding_local.write_text(json.dumps({'host_configs': {h['host']: {} for h in hosts.values()}}))
        protocol_local = root / 'protocol.json'
        protocol_local.write_text('{}')
        config = dict(schema=C.SCHEMA, stage='initial_001', watch_dir=str(watch),
            hosts_path=str(root / 'hosts.json'),
            binding_local=str(binding_local), protocol_local=str(protocol_local),
            binding_sha256=C.sha_file(binding_local), protocol_sha256=C.sha_file(protocol_local),
            remote_protocol='/remote/mpk/results/frozen_protocol.json',
            remote_binding='/remote/mpk/generation/binding_001.json',
            remote_deploy='/remote/mpk/deploy/cp5_05f9947',
            ruler_gold='/gold/ruler.json', aime_gold='/gold/aime.json',
            longbench_gold='/gold/lb.json', ruler_root='/ruler', gpu_wait_seconds=0)
        (root / 'hosts.json').write_text(json.dumps(hosts))
        archive_bytes = self.archive_bytes()
        archives = {}
        artifacts = {}
        for alias, host in hosts.items():
            local = root / f'{alias}.tar.gz'
            local.write_bytes(archive_bytes)
            remote = f"{host['root']}/primary/initial_001.private.tar.gz"
            archives[remote] = archive_bytes
            receipt = dict(schema='v20_request_stage_v1', status='complete', mode='request',
                           stage_name='initial_001', public_identity=dict(
                               binding_sha256=config['binding_sha256'],
                               protocol_sha256=config['protocol_sha256'], host=host['host'],
                               execution_config_sha256={}),
                           private_archive=dict(path=remote, sha256=C.sha_file(local),
                                                bytes=len(archive_bytes)))
            final = watch / alias / 'initial_001.json'
            final.parent.mkdir()
            final.write_text(json.dumps(receipt))
            artifacts[alias] = dict(status='complete', returncode=0, artifacts={
                'initial_001.json': dict(path=str(final), sha256=C.sha_file(final), bytes=final.stat().st_size)})
        (watch / 'initial_001.outcome.json').write_text(json.dumps(dict(
            schema='v20_stage_watch_v1', stage='initial_001', all_complete=True,
            hosts=artifacts)))
        return config, hosts, archives

    @staticmethod
    def archive_bytes(name='ledger/initial_001.jsonl'):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            for path in ('private/a.json', name):
                data = b'{}\n'
                info = tarfile.TarInfo(path)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return output.getvalue()

    def test_terminal_stage_identity_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, archives = self.fixture(root)
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']):
                C.validate_config(config, hosts)
            receipts = C.watch_receipts(config, hosts)
            self.assertEqual(set(receipts), {'mpk', 'dllm'})
            archive_path = root / 'mpk.tar.gz'
            ledger, private = C.extract_archive(archive_path, root / 'extracted')
            self.assertEqual(ledger.read_bytes(), b'{}\n')
            self.assertTrue((private / 'a.json').is_file())

    def test_refuses_nonterminal_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            config, hosts, _ = self.fixture(Path(directory))
            outcome_path = Path(config['watch_dir']) / 'initial_001.outcome.json'
            outcome = json.loads(outcome_path.read_text())
            outcome['all_complete'] = False
            outcome_path.write_text(json.dumps(outcome))
            with self.assertRaisesRegex(ValueError, 'not terminal-complete'):
                C.watch_receipts(config, hosts)
            outcome['all_complete'] = True
            outcome['hosts']['mpk']['artifacts']['initial_001.json']['sha256'] = '0' * 64
            outcome_path.write_text(json.dumps(outcome))
            with self.assertRaisesRegex(ValueError, 'hash drift'):
                C.watch_receipts(config, hosts)

    def test_rejects_traversal_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.tar.gz'
            path.write_bytes(self.archive_bytes('ledger/../../escaped'))
            with tarfile.open(path, 'r:gz') as archive:
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    C.safe_archive_members(archive)

    def test_gpu_busy_never_dispatches_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, remote_archives = self.fixture(root)
            class BusyTransport:
                def __init__(self):
                    self.uploads = []
                def remote_sha(self, host, path):
                    data = remote_archives[path]
                    return C.hashlib.sha256(data).hexdigest(), len(data)
                def copy_from(self, host, remote, local):
                    Path(local).write_bytes(remote_archives[remote])
                def gpu_pids(self, host):
                    return '1234'
                def copy_to(self, *args):
                    self.uploads.append(args)
            transport = BusyTransport()
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']):
                result = C.collect(config, hosts, root / 'out', transport=transport)
            self.assertEqual(result['status'], 'waiting_gpu')
            self.assertFalse(transport.uploads)

    def test_wait_is_local_bounded_and_times_out_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, _ = self.fixture(root)
            (Path(config['watch_dir']) / 'initial_001.outcome.json').unlink()
            class NoRemote:
                def __getattr__(self, name):
                    raise AssertionError(f'remote transport called: {name}')
            ticks = [0]
            spans = []
            def sleep(seconds):
                spans.append(seconds)
                ticks[0] += seconds
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']):
                result = C.collect(config, hosts, root / 'out', transport=NoRemote(),
                                   wait_until_epoch=65, clock=lambda: ticks[0], sleep=sleep)
            self.assertEqual(result['status'], 'waiting_stage')
            self.assertEqual(spans, [30, 30, 5])
            self.assertTrue((root / 'out' / 'initial_001.collect.lock').exists())
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']):
                with self.assertRaises(FileExistsError):
                    C.collect(config, hosts, root / 'out', transport=NoRemote())

    def test_failed_terminal_outcome_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, _ = self.fixture(root)
            path = Path(config['watch_dir']) / 'initial_001.outcome.json'
            outcome = json.loads(path.read_text())
            outcome['all_complete'] = False
            outcome['hosts']['dllm']['status'] = 'failed'
            path.write_text(json.dumps(outcome))
            class NoRemote:
                def __getattr__(self, name):
                    raise AssertionError(f'remote transport called: {name}')
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']):
                result = C.collect(config, hosts, root / 'out', transport=NoRemote())
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['host_status']['dllm'], 'failed')

    def test_scorer_failure_output_stays_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / 'protocol.json'
            binding = root / 'binding.json'
            protocol.write_text('{}')
            binding.write_text('{}')
            archive = root / 'archive.tar.gz'
            archive.write_bytes(self.archive_bytes())
            payload = dict(score_root=str(root / 'score'), remote_protocol=str(protocol),
                           remote_binding=str(binding), protocol_sha256=C.sha_file(protocol),
                           binding_sha256=C.sha_file(binding),
                           archives={a: dict(path=str(archive), sha256=C.sha_file(archive),
                                             bytes=archive.stat().st_size, host_ip=str(i))
                                     for i, a in enumerate(('mpk', 'dllm'), 1)},
                           ruler_gold='/g/r', aime_gold='/g/a', longbench_gold='/g/l', ruler_root='/r')
            with patch.object(C.subprocess, 'check_output', return_value=''), \
                 patch.object(C.subprocess, 'run', return_value=SimpleNamespace(
                     returncode=1, stdout='private answers', stderr='private diagnostic')):
                with self.assertRaisesRegex(RuntimeError, 'exit 1'):
                    C.remote_score(payload)
            self.assertEqual((root / 'score' / 'scorer.stdout.private.txt').read_text(), 'private answers')
            self.assertEqual((root / 'score' / 'scorer.stderr.private.txt').read_text(), 'private diagnostic')


if __name__ == '__main__':
    unittest.main()
