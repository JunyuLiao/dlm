"""CPU tests for frozen historical-only archive collection and scoring."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import v20_collect_historical as H
from scripts import v20_collect_score as C
from tests.test_v20_collect_score import CollectScoreTests


class HistoricalTests(unittest.TestCase):
    def fixture(self, root):
        base, hosts, _ = CollectScoreTests().fixture(root)
        watch = root / 'historical_watch'
        watch.mkdir()
        config = dict(base, schema=H.SCHEMA, stage=H.STAGE, watch_dir=str(watch),
                      main_score_root=hosts['mpk']['root'] + '/offline_scoring/remainder_001',
                      main_summary_sha256=H.FROZEN_MAIN_SHA)
        outcome = dict(schema='v20_stage_watch_v1', stage=H.STAGE, all_complete=False, hosts={})
        for alias, host in hosts.items():
            archive = root / f'{alias}.tar.gz'
            receipt = dict(schema='v20_request_stage_v1', stage_name=H.STAGE,
                mode='request', status='failed', execution=dict(status='stopped_before_block'),
                public_identity=dict(host=host['host'], binding_sha256=config['binding_sha256'],
                    protocol_sha256=config['protocol_sha256'], execution_config_sha256={}),
                private_archive=dict(path=host['root'] + '/historical/historical_001.private.tar.gz',
                                     sha256=C.sha_file(archive), bytes=archive.stat().st_size))
            final = watch / alias / 'historical_001.json'
            final.parent.mkdir()
            final.write_text(json.dumps(receipt))
            outcome['hosts'][alias] = dict(status='failed', artifacts={H.STAGE + '.json': dict(
                path=str(final), sha256=C.sha_file(final), bytes=final.stat().st_size)})
        (watch / 'historical_001.outcome.json').write_text(json.dumps(outcome))
        return config, hosts, outcome

    def test_valid_finalized_partial_and_missing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, outcome = self.fixture(root)
            receipts, reason = H.terminal_historical(config, hosts)
            self.assertIsNone(reason)
            self.assertEqual({x['wrapper_status'] for x in receipts.values()}, {'failed'})
            del outcome['hosts']['dllm']['artifacts'][H.STAGE + '.json']
            (Path(config['watch_dir']) / 'historical_001.outcome.json').write_text(json.dumps(outcome))
            receipts, reason = H.terminal_historical(config, hosts)
            self.assertIsNone(receipts)
            self.assertIn('absent', reason)

    def test_complete_requires_100_rows_50_warm_and_complete_wrappers(self):
        extension = dict(planned_executions=100, recorded_executions=100,
            execution_rows_complete=True, datasets={'aime26': dict(warm_accepted=12, first_success=12),
              'ruler4k': dict(warm_accepted=26, first_success=26),
              'longbench_v2': dict(warm_accepted=12, first_success=12)})
        self.assertEqual(H.historical_completion(extension, {'mpk':'complete','dllm':'complete'})['status'],
                         'scored_complete')
        extension['datasets']['aime26']['warm_accepted'] = 11
        self.assertEqual(H.historical_completion(extension, {'mpk':'complete','dllm':'complete'})['status'],
                         'scored_partial')
        extension['datasets']['aime26']['warm_accepted'] = 12
        self.assertEqual(H.historical_completion(extension, {'mpk':'failed','dllm':'complete'})['status'],
                         'scored_partial')

    def test_core_paths_hash_and_four_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'remainder_001'
            root.mkdir()
            summary = root / 'remainder.redacted_summary.json'
            summary.write_text('{}')
            private = {}
            for alias in ('mpk', 'dllm'):
                p = root / 'merged_private' / alias
                p.mkdir(parents=True)
                private[alias] = str(p)
                for stage in ('initial', 'remainder'):
                    ledger = root / f'{alias}_{stage}' / 'ledger' / 'run.jsonl'
                    ledger.parent.mkdir(parents=True)
                    ledger.write_text('{}\n')
            (root / 'private_roots.json').write_text(json.dumps(private))
            payload = dict(main_score_root=str(root), main_summary_sha256=C.sha_file(summary),
                           host_ips={'mpk':'mpk','dllm':'dllm'})
            _, _, ledgers = H.core_paths(payload)
            self.assertEqual(len(ledgers), 4)
            summary.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'summary byte drift'):
                H.core_paths(payload)

    def test_work_dataset_mapping_and_failed_or_rejected_warm_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hist_specs, core_specs, hist_rows, core_rows = [], [], [], []
            for block, dataset in enumerate(('aime26', 'ruler4k', 'longbench_v2')):
                for role, repeat in (('attempt0', 0), ('warm', 1)):
                    base = dict(dataset=dataset, id=f'{dataset}/private', seed=101, block=block,
                                role=role, repeat=repeat, host='host1', gpu_uuid='GPU-1')
                    h = dict(base, arm='G75L30_nativeQ128', cell_id=f'h{block}')
                    n = dict(base, arm='D_native', cell_id=f'n{block}')
                    hist_specs.append(h)
                    core_specs.append(n)
                    row_h = dict(h, event='run', execution_key=H._execution_key(h), ok=dataset != 'ruler4k',
                                 decoder_calls=6 if dataset == 'longbench_v2' else 4,
                                 canvases=3 if dataset == 'longbench_v2' else 2,
                                 per_canvas_calls=[2, 2, 2] if dataset == 'longbench_v2' else [2, 2],
                                 output_tokens=10, termination='length', api_wall_s=3,
                                 router_phase_evidence=dict(A=1, D=2, H=3),
                                 phase_evidence=dict(per_canvas=[dict(iteration_cap=True)],
                                                     prefill_end_to_finish_gpu_s=2),
                                 acceptance=dict(accepted=dataset == 'longbench_v2'))
                    row_n = dict(n, event='run', execution_key=H._execution_key(n), ok=True,
                                 decoder_calls=3, api_wall_s=6, acceptance=dict(accepted=True))
                    hist_rows.append(row_h)
                    core_rows.append(row_n)
            hp, cp = root / 'hist.jsonl', root / 'core.jsonl'
            hp.write_text('\n'.join(json.dumps(x) for x in hist_rows))
            cp.write_text('\n'.join(json.dumps(x) for x in core_rows))
            protocol = dict(ids={d: [] for d in ('aime26', 'ruler4k', 'longbench_v2')},
                            historical_extension=dict(schedule=hist_specs), schedule=core_specs)
            work = H.historical_work(protocol, [cp], [hp])['datasets']
            self.assertEqual(work['aime26']['first_success'], 1)
            self.assertEqual(work['aime26']['decoder_calls_total'], 4)
            self.assertEqual(work['aime26']['accepted_warm_by_host'], {})
            self.assertEqual(work['ruler4k']['first_success'], 0)
            lb = work['longbench_v2']
            self.assertEqual(lb['calls_per_canvas_distribution'], {2: 3})
            self.assertEqual(lb['iteration_cap_canvases'], 1)
            host = lb['accepted_warm_by_host']['host1']
            self.assertEqual(host['paired_total_request_time_ratio'], .5)
            self.assertEqual(host['paired_decoder_call_ratio'], 2)
            self.assertEqual(host['paired_amortized_wall_per_call_ratio'], .25)

    def test_local_wait_lock_never_contacts_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, hosts, _ = self.fixture(root)
            (Path(config['watch_dir']) / 'historical_001.outcome.json').unlink()
            class NoRemote:
                def __getattr__(self, name):
                    raise AssertionError(f'remote called: {name}')
            ticks = [0]
            def sleep(span):
                self.assertLessEqual(span, 30)
                ticks[0] += span
            with patch.object(C, 'FROZEN_BINDING_SHA', config['binding_sha256']), \
                 patch.object(C, 'FROZEN_PROTOCOL_SHA', config['protocol_sha256']), \
                 patch.object(H, 'FROZEN_MAIN_SHA', config['main_summary_sha256']):
                result = H.collect(config, hosts, root / 'out', transport=NoRemote(),
                                   wait_until_epoch=35, clock=lambda: ticks[0], sleep=sleep)
                self.assertEqual(result['status'], 'waiting_stage')
                with self.assertRaises(FileExistsError):
                    H.collect(config, hosts, root / 'out', transport=NoRemote())

    def test_remote_scorer_six_ledgers_and_private_failure_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / 'remainder_001'
            main.mkdir()
            summary = main / 'remainder.redacted_summary.json'
            summary.write_text('{}')
            roots = {}
            for alias in ('mpk', 'dllm'):
                p = main / 'merged_private' / alias
                p.mkdir(parents=True)
                roots[alias] = str(p)
                for stage in ('initial', 'remainder'):
                    ledger = main / f'{alias}_{stage}' / 'ledger' / 'run.jsonl'
                    ledger.parent.mkdir(parents=True)
                    ledger.write_text('{}\n')
            (main / 'private_roots.json').write_text(json.dumps(roots))
            protocol, binding = root / 'protocol.json', root / 'binding.json'
            protocol.write_text(json.dumps(dict(schedule=[], historical_extension=dict(schedule=[]), ids={})))
            binding.write_text('{}')
            archives = {}
            for alias in ('mpk', 'dllm'):
                path = root / f'{alias}.tar.gz'
                with tarfile.open(path, 'w:gz') as tar:
                    for member in (f'private/cells/{alias}/attempt00.json', f'ledger/{alias}.jsonl'):
                        data = b'{}\n'
                        info = tarfile.TarInfo(member)
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
                archives[alias] = dict(path=str(path), sha256=C.sha_file(path),
                                       bytes=path.stat().st_size, host_ip=alias)
            payload = dict(score_root=str(root / 'score'), main_score_root=str(main),
                main_summary_sha256=C.sha_file(summary), host_ips={'mpk':'mpk','dllm':'dllm'},
                remote_protocol=str(protocol), remote_binding=str(binding),
                protocol_sha256=C.sha_file(protocol), binding_sha256=C.sha_file(binding),
                archives=archives, wrapper_status={'mpk':'complete','dllm':'complete'},
                ruler_gold='/g/r', aime_gold='/g/a', longbench_gold='/g/l', ruler_root='/r')
            def score(cmd, **kwargs):
                self.assertEqual(cmd.count('--ledger'), 4)
                self.assertEqual(cmd.count('--historical-ledger'), 2)
                self.assertIn('--historical-private-roots', cmd)
                Path(cmd[cmd.index('--out')+1]).write_text(json.dumps(dict(
                    historical_extension=dict(planned_executions=100, recorded_executions=100,
                        execution_rows_complete=True, datasets={'ruler4k':dict(warm_accepted=50,first_success=50)}))))
                return SimpleNamespace(returncode=0, stdout='private', stderr='')
            with patch.object(H.subprocess, 'check_output', return_value=''), \
                 patch.object(H.subprocess, 'run', side_effect=score), \
                 patch('builtins.print') as printed:
                H.remote_score(payload)
            self.assertEqual(json.loads(printed.call_args.args[0])['status'], 'scored_complete')
            self.assertEqual(C.sha_file(summary), payload['main_summary_sha256'])


if __name__ == '__main__':
    unittest.main()
