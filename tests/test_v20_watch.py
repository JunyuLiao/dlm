"""Fake-transport tests; no SSH, SCP, GPU or remote job is touched."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import v20_watch as W


class FakeTransport:
    def __init__(self, statuses, payloads):
        self.statuses = {key: list(value) for key, value in statuses.items()}
        self.payloads = payloads
        self.reads = []
        self.copies = []

    def read_stage(self, host, path):
        alias = host['host']
        self.reads.append((alias, path))
        sequence = self.statuses[alias]
        return sequence.pop(0) if len(sequence) > 1 else sequence[0]

    def artifact_manifest(self, host, paths):
        return {path: {'sha256': W.sha(data), 'bytes': len(data)}
                for path, data in self.payloads.get(host['host'], {}).items() if path in paths}

    def copy_file(self, host, remote_path, local_path):
        self.copies.append((host['host'], remote_path))
        Path(local_path).write_bytes(self.payloads[host['host']][remote_path])


class V20WatchTests(unittest.TestCase):
    def hosts(self):
        return {key: {'host': key, 'root': f'/private/{key}',
                      'python': '/env/bin/python', 'env': {}}
                for key in ('mpk', 'dllm')}

    def test_terminal_fetch_sha_and_no_repoll(self):
        hosts = self.hosts()
        payloads = {key: {f'/private/{key}/cost/screen_002.stage.json': b'{"status":"complete"}\n',
                          f'/private/{key}/cost/screen_002.json': b'{"ok":true}\n'}
                    for key in hosts}
        transport = FakeTransport(dict(mpk=[{'status': 'started'},
                                           {'status': 'complete', 'returncode': 0}],
                                       dllm=[{'status': 'complete', 'returncode': 0}]), payloads)
        now = [0.]
        def sleep(seconds):
            now[0] += seconds
        with tempfile.TemporaryDirectory() as directory:
            result = W.watch(hosts, 'screen_002', 'cost', directory, 180,
                             transport=transport, clock=lambda: now[0], sleep=sleep)
            self.assertTrue(result['all_complete'])
            self.assertEqual(len(transport.copies), 4)
            self.assertEqual([key for key, _ in transport.reads].count('dllm'), 1)
            self.assertTrue((Path(directory)/'screen_002.outcome.json').exists())
            self.assertTrue((Path(directory)/'mpk'/'screen_002.json').exists())
            with self.assertRaises(FileExistsError):
                W.watch(hosts, 'screen_002', 'cost', directory, 180,
                        transport=transport, clock=lambda: now[0], sleep=sleep)

    def test_failure_and_missing_final_return_unsuccessful(self):
        hosts = self.hosts()
        payloads = {key: {f'/private/{key}/cost/screen_003.stage.json': b'{}\n'}
                    for key in hosts}
        transport = FakeTransport(dict(mpk=[{'status': 'complete', 'returncode': 0}],
                                       dllm=[{'status': 'failed', 'returncode': 2}]), payloads)
        with tempfile.TemporaryDirectory() as directory:
            result = W.watch(hosts, 'screen_003', 'cost', directory, 120,
                             transport=transport, clock=lambda: 0., sleep=lambda _: None)
            self.assertFalse(result['all_complete'])
            self.assertEqual(result['hosts']['mpk']['reason'], 'complete_stage_missing_final_profile')
            self.assertEqual(result['hosts']['dllm']['status'], 'failed')

    def test_deadline_stops_without_dispatch(self):
        hosts = self.hosts()
        transport = FakeTransport({key: [None] for key in hosts}, {})
        now = [0.]
        with tempfile.TemporaryDirectory() as directory:
            result = W.watch(hosts, 'screen_004', 'cost', directory, 61,
                             transport=transport, clock=lambda: now[0],
                             sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
            self.assertFalse(result['all_complete'])
            self.assertTrue(all(x['status'] == 'deadline_stop' for x in result['hosts'].values()))
            self.assertEqual(transport.copies, [])

    def test_unsafe_remote_path_rejected(self):
        with self.assertRaises(ValueError):
            W.safe_stage('../screen', 'cost')
        with self.assertRaises(ValueError):
            W.safe_stage('screen_002', '../cost')


if __name__ == '__main__':
    unittest.main()
