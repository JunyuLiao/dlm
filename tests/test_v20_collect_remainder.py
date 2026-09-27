"""CPU tests for partial-terminal remainder collection and private merge."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import v20_collect_remainder as R
from scripts import v20_collect_score as C
from tests.test_v20_collect_score import CollectScoreTests


class RemainderTests(unittest.TestCase):
    def fixture(self, root):
        first_config, hosts, _ = CollectScoreTests().fixture(root)
        initial_collect = root / 'initial_collect'
        initial_collect.mkdir()
        archives = {}
        for alias in hosts:
            source = root / f'{alias}.tar.gz'
            target = initial_collect / f'{alias}.initial_001.private.tar.gz'
            target.write_bytes(source.read_bytes())
            archives[alias] = dict(sha256=C.sha_file(target), bytes=target.stat().st_size)
        (initial_collect / 'initial_001.collect.status.json').write_text(json.dumps(dict(
            schema=C.SCHEMA, status='complete', first84_recorded=84, archives=archives)))
        remainder_watch = root / 'remainder_watch'
        remainder_watch.mkdir()
        config = dict(first_config, schema=R.SCHEMA, stage=R.STAGE,
                      initial_collect_dir=str(initial_collect),
                      initial_watch_dir=first_config['watch_dir'], watch_dir=str(remainder_watch),
                      initial_remote_archives={
                          'mpk': hosts['mpk']['root'] + '/primary/initial_001.private.tar.gz',
                          'dllm': hosts['mpk']['root'] + '/offline_scoring/dllm.initial_001.private.tar.gz'},
                      initial_archive_sha256={a: item['sha256'] for a, item in archives.items()})
        outcome = dict(schema='v20_stage_watch_v1', stage=R.STAGE, all_complete=False, hosts={})
        for alias, host in hosts.items():
            archive = root / f'{alias}.tar.gz'
            receipt = dict(schema='v20_request_stage_v1', stage_name=R.STAGE,
                           mode='request', status='failed', execution=dict(status='stopped_before_block'),
                           public_identity=dict(host=host['host'],
                               binding_sha256=config['binding_sha256'],
                               protocol_sha256=config['protocol_sha256'],
                               execution_config_sha256={}),
                           private_archive=dict(path=host['root'] + '/primary/remainder_001.private.tar.gz',
                                                sha256=C.sha_file(archive), bytes=archive.stat().st_size))
            final = remainder_watch / alias / 'remainder_001.json'
            final.parent.mkdir()
            final.write_text(json.dumps(receipt))
            outcome['hosts'][alias] = dict(status='failed', artifacts={R.STAGE + '.json': dict(
                path=str(final), sha256=C.sha_file(final), bytes=final.stat().st_size)})
        (remainder_watch / 'remainder_001.outcome.json').write_text(json.dumps(outcome))
        return config, hosts, outcome

    def test_finalized_failed_wrapper_with_valid_archive_is_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            config, hosts, _ = self.fixture(Path(directory))
            receipts, reason = R.terminal_remainder(config, hosts)
            self.assertIsNone(reason)
            self.assertEqual({x['wrapper_status'] for x in receipts.values()}, {'failed'})
            self.assertEqual({x['execution_status'] for x in receipts.values()}, {'stopped_before_block'})
            first = R.initial_archives(config, hosts)
            self.assertEqual(set(first), {'mpk', 'dllm'})

    def test_hardkill_without_final_archive_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            config, hosts, outcome = self.fixture(Path(directory))
            del outcome['hosts']['dllm']['artifacts'][R.STAGE + '.json']
            (Path(config['watch_dir']) / 'remainder_001.outcome.json').write_text(json.dumps(outcome))
            receipts, reason = R.terminal_remainder(config, hosts)
            self.assertIsNone(receipts)
            self.assertIn('finalized wrapper absent', reason)

    def test_merge_disjoint_and_reject_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a', root / 'b'
            for path, cell in ((a, 'cell1'), (b, 'cell2')):
                target = path / 'cells' / cell / 'attempt00.json'
                target.parent.mkdir(parents=True)
                target.write_text('{}')
            merged = R.merge_private_trees([a, b], root / 'merged')
            self.assertEqual(len(list(merged.rglob('attempt00.json'))), 2)
            collision = b / 'cells' / 'cell1' / 'attempt00.json'
            collision.parent.mkdir(parents=True)
            collision.write_text('{}')
            with self.assertRaises(FileExistsError):
                R.merge_private_trees([a, b], root / 'collision')

    def test_all700_rows_need_50_valid_blocks_and_complete_wrappers(self):
        full = dict(recorded_executions=700, executions_complete=True,
                    planned_blocks=50, complete_valid_pair_blocks=50)
        complete = R.score_completion(full, {'mpk': 'complete', 'dllm': 'complete'})
        self.assertEqual(complete['status'], 'scored_complete')
        full['complete_valid_pair_blocks'] = 49
        bad_warm = R.score_completion(full, {'mpk': 'complete', 'dllm': 'complete'})
        self.assertEqual(bad_warm['status'], 'scored_partial')
        self.assertTrue(bad_warm['recorded_all_rows'])
        full['complete_valid_pair_blocks'] = 50
        failed_wrapper = R.score_completion(full, {'mpk': 'failed', 'dllm': 'complete'})
        self.assertEqual(failed_wrapper['status'], 'scored_partial')

    def score_payload(self, root):
        protocol, binding = root / 'protocol.json', root / 'binding.json'
        protocol.write_text('{}')
        binding.write_text('{}')
        archives = {}
        for alias in ('mpk', 'dllm'):
            archives[alias] = {}
            for stage in ('initial', 'remainder'):
                path = root / f'{alias}_{stage}.tar.gz'
                with tarfile.open(path, 'w:gz') as tar:
                    for member in (f'private/cells/{alias}_{stage}/attempt00.json',
                                   f'ledger/{alias}_{stage}.jsonl'):
                        data = b'{}\n'
                        info = tarfile.TarInfo(member)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
                archives[alias][stage] = dict(path=str(path), sha256=C.sha_file(path),
                    bytes=path.stat().st_size, host_ip=alias)
        return dict(score_root=str(root / 'score'), remote_protocol=str(protocol),
                    remote_binding=str(binding), protocol_sha256=C.sha_file(protocol),
                    binding_sha256=C.sha_file(binding), archives=archives,
                    wrapper_status={'mpk': 'complete', 'dllm': 'failed'},
                    ruler_gold='/gold/r', aime_gold='/gold/a', longbench_gold='/gold/l',
                    ruler_root='/ruler')

    def test_remote_scorer_uses_four_ledgers_and_partial_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.score_payload(root)
            def score(cmd, **kwargs):
                self.assertEqual(cmd.count('--ledger'), 4)
                self.assertIn('--private-roots', cmd)
                summary = dict(full=dict(recorded_executions=700, executions_complete=True,
                                         planned_blocks=50, complete_valid_pair_blocks=50))
                Path(cmd[cmd.index('--out') + 1]).write_text(json.dumps(summary))
                return SimpleNamespace(returncode=0, stdout='private', stderr='')
            with patch.object(R.subprocess, 'check_output', return_value=''), \
                 patch.object(R.subprocess, 'run', side_effect=score), \
                 patch('builtins.print') as printed:
                R.remote_score(payload)
            result = json.loads(printed.call_args.args[0])
            self.assertEqual(result['status'], 'scored_partial')
            self.assertTrue(result['recorded_all_rows'])
            self.assertEqual(result['complete_valid_pair_blocks'], 50)

    def test_remote_scorer_failure_output_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.score_payload(root)
            with patch.object(R.subprocess, 'check_output', return_value=''), \
                 patch.object(R.subprocess, 'run', return_value=SimpleNamespace(
                     returncode=3, stdout='private answers', stderr='private error')):
                with self.assertRaisesRegex(RuntimeError, 'exit 3'):
                    R.remote_score(payload)
            self.assertEqual((root / 'score' / 'scorer.stdout.private.txt').read_text(), 'private answers')
            self.assertEqual((root / 'score' / 'scorer.stderr.private.txt').read_text(), 'private error')

    def test_wait_and_lock_are_local_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, _ = self.fixture(root)
            (Path(config['watch_dir']) / 'remainder_001.outcome.json').unlink()
            class NoRemote:
                def __getattr__(self, name):
                    raise AssertionError(f'remote called: {name}')
            ticks = [0]
            def sleep(span):
                self.assertLessEqual(span, 30)
                ticks[0] += span
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']):
                result = R.collect(config, hosts, root / 'out', transport=NoRemote(),
                                   wait_until_epoch=65, clock=lambda: ticks[0], sleep=sleep)
                self.assertEqual(result['status'], 'waiting_stage')
                with self.assertRaises(FileExistsError):
                    R.collect(config, hosts, root / 'out', transport=NoRemote())


if __name__ == '__main__':
    unittest.main()
