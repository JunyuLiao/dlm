"""CPU-only one-shot stage terminal/archive tests; no GPU or worker process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import v20_request_stage as S


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


class RequestStageTests(unittest.TestCase):
    def fixture(self, root, mode='request'):
        protocol = root / 'protocol.json'
        write_json(protocol, {'schema': 'v20_fan_panel_v1'})
        method = root / 'method.py'
        method.write_bytes(b'public method source')
        scripts = root / 'scripts'
        scripts.mkdir()
        worker = scripts / ('v20_bridge.py' if mode == 'bridge' else 'v20_run.py')
        worker.write_bytes(b'qualified child source in a separate deploy')
        config = root / 'arm.json'
        write_json(config, {'source_hashes': {str(method): S.digest(method),
                                              str(worker): S.digest(worker)},
                            'model_metadata_hashes': {'config.json': 'm'}})
        binding = root / 'binding.json'
        write_json(binding, {'host_configs': {'mpk': {'ruler4k': {
            'D_native': {'path': str(config), 'sha256': S.digest(config)}}}}})
        private = root / 'private'
        private.mkdir()
        (private / 'receipt.json').write_text('{"raw":"do not print"}\n', encoding='utf-8')
        ledger = root / 'bridge.jsonl' if mode == 'bridge' else root / 'ledger.jsonl'
        output = root / 'output'
        argv = ['python', '-m', 'scripts.v20_bridge', 'run'] if mode == 'bridge' else [
            'python', '-m', 'scripts.v20_run']
        argv += ['--protocol', str(protocol), '--binding', str(binding),
                 '--host', 'mpk', '--private', str(private),
                 '--summary' if mode == 'bridge' else '--ledger', str(ledger)]
        if mode == 'request':
            argv += ['--stage', 'initial']
        stage_config = root / 'stage_config.json'
        write_json(stage_config, dict(schema=S.SCHEMA, stage_name='mpk_initial', mode=mode,
                                      argv=argv, private_dir=str(private), ledger=str(ledger),
                                      output_dir=str(output), cwd=str(root)))
        return stage_config, private, ledger, output

    def test_request_terminal_archive_is_exact_and_run_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, private, ledger, output = self.fixture(root)
            ledger.write_text('\n'.join(json.dumps(x) for x in [
                {'event': 'start', 'host': 'mpk'},
                {'event': 'run', 'role': 'attempt0', 'ok': False},
                {'event': 'run', 'role': 'warm', 'ok': True},
                {'event': 'worker_end', 'status': 'complete'}]) + '\n', encoding='utf-8')
            with patch.object(S.subprocess, 'run', return_value=type('Result', (), {'returncode': 0})()):
                self.assertEqual(S.run(config), 0)
                with self.assertRaises(FileExistsError):
                    S.run(config)
            final = json.loads((output / 'mpk_initial.json').read_text())
            self.assertEqual(final['status'], 'complete')
            self.assertEqual(final['execution']['run_status_counts'], {'failed': 1, 'ok': 1})
            self.assertFalse(final['execution']['all_runs_ok'])
            self.assertEqual(final['public_identity']['execution_config_sha256']['ruler4k/D_native'],
                             S.digest(root / 'arm.json'))
            self.assertEqual(final['public_identity']['wrapper_sha256'], S.digest(Path(S.__file__)))
            self.assertEqual(final['public_identity']['worker_sha256'],
                             S.digest(root / 'scripts' / 'v20_run.py'))
            self.assertNotEqual(final['public_identity']['worker_sha256'],
                                S.digest(Path(S.__file__).with_name('v20_run.py')))
            archive = output / 'mpk_initial.private.tar.gz'
            self.assertEqual(final['private_archive']['sha256'], S.digest(archive))
            with tarfile.open(archive, 'r:gz') as package:
                self.assertEqual(sorted(package.getnames()), ['ledger/ledger.jsonl',
                                                              'private/receipt.json'])
            self.assertEqual((private / 'receipt.json').read_text(), '{"raw":"do not print"}\n')

    def test_bridge_cli_zero_but_failed_row_is_terminal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, ledger, output = self.fixture(root, 'bridge')
            ledger.write_text('\n'.join(json.dumps(x) for x in [
                {'event': 'start', 'arms': ['D_native', 'M1']},
                {'event': 'arm', 'arm': 'D_native', 'status': 'ok'},
                {'event': 'arm', 'arm': 'M1', 'status': 'failed'}]) + '\n', encoding='utf-8')
            with patch.object(S.subprocess, 'run', return_value=type('Result', (), {'returncode': 0})()):
                self.assertEqual(S.run(config), 1)
            final = json.loads((output / 'mpk_initial.json').read_text())
            self.assertEqual(final['execution']['status'], 'failed')
            self.assertEqual(final['child_returncode'], 0)
            self.assertEqual(final['status'], 'failed')

    def test_child_failure_preserves_ledger_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, ledger, output = self.fixture(root)
            ledger.write_text('{"event":"run","ok":false}\n', encoding='utf-8')
            with patch.object(S.subprocess, 'run', return_value=type('Result', (), {'returncode': 7})()):
                self.assertEqual(S.run(config), 1)
            final = json.loads((output / 'mpk_initial.json').read_text())
            self.assertEqual(final['child_returncode'], 7)
            self.assertEqual(final['execution']['status'], 'missing_worker_end')
            self.assertEqual(ledger.read_text(), '{"event":"run","ok":false}\n')

    def test_actual_child_source_drift_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _, output = self.fixture(root)
            (root / 'scripts' / 'v20_run.py').write_bytes(b'changed child bytes')
            with patch.object(S.subprocess, 'run') as child:
                self.assertEqual(S.run(config), 1)
                child.assert_not_called()
            final = json.loads((output / 'mpk_initial.json').read_text())
            self.assertEqual(final['wrapper_error_type'], 'ValueError')
            self.assertEqual(final['status'], 'failed')

    def test_rejects_wrong_child_archive_paths_and_forbidden_or_symlink_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, private, ledger, output = self.fixture(root)
            data = json.loads(config.read_text())
            data['argv'][data['argv'].index('--private') + 1] = str(root / 'other')
            with self.assertRaisesRegex(ValueError, 'archive paths'):
                S.validate_config(data)
            data['argv'][data['argv'].index('--private') + 1] = str(private)
            data['argv'] = ['sh', '-c', 'anything']
            with self.assertRaisesRegex(ValueError, 'pinned v20'):
                S.validate_config(data)
            (root / '.git').write_text('gitdir: elsewhere\n', encoding='utf-8')
            data['argv'] = json.loads(config.read_text())['argv']
            with self.assertRaisesRegex(ValueError, 'outside Git'):
                S.validate_config(data)
            (root / '.git').unlink()
            ledger.write_text('{}\n', encoding='utf-8')
            output.mkdir()
            (private / 'gold.json').write_text('{}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unsafe/forbidden'):
                S.archive_private(private, ledger, output / 'bad.tar.gz')


if __name__ == '__main__':
    unittest.main()
