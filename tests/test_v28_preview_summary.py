"""Public CPU toy bundle tests, runnable with stdlib unittest alone."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import v28_preview_summary as summary
from scripts.v27_vllm_bind import fingerprint


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def toy_bundle(root):
    deploy, model = root / 'deploy', root / 'toy_model'
    deploy.mkdir()
    model.mkdir()
    (deploy / 'DEPLOY_SHA').write_text('public-toy-deploy', encoding='utf-8')
    write_json(model / 'config.json', {'public_toy_model': True})
    write_json(model / 'generation_config.json', {'eos_token_id': 1})
    for relative in summary.SOURCES:
        path = deploy / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# public toy source, never executed\n', encoding='utf-8')
    datasets = {f'toy_length_{n}': dict(indices=[0, 1], repeats=list(range(8))) for n in (32, 64, 96)}
    blocks = [dict(engine_seed=7001 + block, repeats=[2 * block, 2 * block + 1]) for block in range(4)]
    private_cells = [dict(dataset=ds, index=i, id=f'PRIVATE_TOY_ID_{ds}_{i}') for ds in datasets for i in (0, 1)]
    groups = {}
    for _, query, canvas, group in summary.VARIANTS.values():
        if group in groups:
            continue
        frozen = root / group
        frozen.mkdir()
        config = dict(condition='public_toy', q_block=query)
        config['fingerprint'] = fingerprint(config)
        required = dict(score_period=64, decision_interval=6, risk_state='dense_prefix',
                        carry_first=True, fused_observe=True, async_route=True, consumer='fa4', q_block=query)
        adapter = dict(lifecycle='request_clear', alias_splits=2)
        if canvas != 'legacy':
            adapter['canvas_buffers'] = canvas
        spec = dict(schema='v27_vllm_spec_v1', name='public toy preview', protocol_id='v28_toy_' + group,
                    arms=['dense', 'method'], controls=['native', 'allkept'], datasets=datasets,
                    control_indices={ds: [0, 1] for ds in datasets}, blocks=blocks, order_seed=77,
                    settings=dict(max_model_len=106496, chunk=16384, gpu_memory_utilization=.85,
                                  block_size=32, cpu_threads=1), primary_receipt_method=required,
                    arm_settings={arm: dict(compilation_config='default' if arm == 'dense' else 'PIECEWISE',
                                            cudagraph_mode='default' if arm == 'dense' else 'PIECEWISE')
                                  for arm in summary.old.ARMS}, adapter_settings=adapter, jit_event_receipts=True)
        write_json(frozen / 'spec.json', spec)
        write_json(frozen / 'config.private.json', config)
        write_json(frozen / 'cells.private.json', private_cells)
        manifests = {}
        for ds in datasets:
            path = frozen / (ds + '.private.json')
            write_json(path, [dict(id=cell['id'], prompt='PRIVATE_TOY_PROMPT', prompt_tokens=[1, 2],
                                   generation_budget=512) for cell in private_cells if cell['dataset'] == ds])
            manifests[ds] = str(path)
        paths = list(frozen.glob('*.json')) + [deploy / name for name in summary.SOURCES] + list(model.glob('*.json'))
        binding = dict(protocol_id=spec['protocol_id'], deploy_commit='public-toy-deploy', deploy=str(deploy),
                       model=str(model), host='toy_alias', gpu_uuid='GPU-TOY', spec=str(frozen / 'spec.json'),
                       config=str(frozen / 'config.private.json'), cells=str(frozen / 'cells.private.json'),
                       manifests=manifests, files={str(path): summary.qualify.panel.digest(path) for path in paths})
        binding_path = frozen / 'binding.private.json'
        write_json(binding_path, binding)
        groups[group] = binding_path, binding, spec, config
    variants = {}
    walls = dict(dense=12, native=11, allkept_release=10, main_legacy=10,
                 main_release=9, q64_release=8)
    for variant, (arm, query, canvas, group) in summary.VARIANTS.items():
        binding_path, binding, spec, config = groups[group]
        settings = summary.qualify.execution_settings(query, 'request_clear', canvas)
        settings['jit_monitor_events'] = True
        run_dirs = []
        for block in range(4):
            directory = root / f'{variant}_block_{block}'
            directory.mkdir()
            run_dirs.append(str(directory))
            records, completions = [], []
            n = 10 + block * 2
            for cell in private_cells:
                for repeat in blocks[block]['repeats']:
                    wall = walls[variant] * (1 + cell['index'] / 10)
                    receipts = None
                    if arm != 'dense':
                        memory = dict(num_alloc_retries=1, num_ooms=0, allocated_bytes=100,
                                      reserved_bytes=200, peak_allocated_bytes=150, peak_reserved_bytes=250)
                        adapter = dict(begins=n, observes=n, global_calls=5 * n, order_errors=0,
                                       split_fa4_calls=0 if arm == 'native' else n * 5,
                                       v28_config=settings, kv_probe_timed_checks=0,
                                       allocator=summary.qualify.allocator_receipt(memory, memory),
                                       kv_layout_checks=[dict(layer=layer, max_abs_error=0, page_size=64)
                                                         for layer in (5, 11, 17, 23, 29)],
                                       canvas_release_calls=1, canvas_release_layers=5, canvas_release_bytes=1024)
                        method = None
                        if arm == 'method':
                            method = dict(effective_method=dict(spec['primary_receipt_method'],
                                                                 q_regroup=False, q_carry64=False),
                                          active_layers=[5, 11, 17, 23, 29], gated_native_calls=0,
                                          layer_native_calls=0, unsupported_mask_refreshes=0,
                                          fused_observations=5, dp_routes=5, q64_refined_routes=5, q64_list_builds=5)
                        receipts = dict(adapter=adapter, method=method, timing=None)
                    row = dict(schema=summary.old.SCHEMA, protocol_id=spec['protocol_id'],
                               deploy_commit=binding['deploy_commit'], host=binding['host'], gpu_uuid=binding['gpu_uuid'],
                               dataset=cell['dataset'], index=cell['index'], repeat=repeat, arm=arm,
                               engine_seed=blocks[block]['engine_seed'], seed_applied=False, qualification_only=False,
                               measurement_mode='request_boundary_sync', run_id=directory.name,
                               receipts=receipts, wall_s=wall, decode_span_s=wall / 2, prefill_s=2.,
                               denoise_forward_count=n, scheduler_denoise_forward_count=n - 1,
                               speculative_unused_denoising=1, commit_forward_count=2,
                               execution_count_source='vllm_existing_async_cpu_snapshot', graph_captures_timed=0,
                               output_tokens=16, finish_reason='stop', torch='toy_torch', vllm='toy_vllm',
                               compilation_deltas=dict(num_backend_compilations=0, num_inductor_compiles=0,
                                                       **{'jit_monitor_' + key: 0 for key in
                                                          ('triton_events', 'cute_events', 'unknown_events', 'all_events')}),
                               adapter_sha256=None if arm == 'dense' else binding['files'][str(deploy / summary.SOURCES[-1])],
                               method_fingerprint=config['fingerprint'] if arm == 'method' else None,
                               **spec['settings'], **spec['arm_settings'][arm])
                    records.append(row)
                    completions.append(dict(dataset=row['dataset'], index=row['index'], repeat=repeat,
                                            arm=arm, run_id=row['run_id'], finish_reason='stop', id=cell['id'],
                                            completion='<channel|>Answer: B PRIVATE_TOY_COMPLETION<turn|>'))
            write_rows(directory / 'records.jsonl', records)
            write_rows(directory / 'completions.private.jsonl', completions)
            write_json(directory / 'terminal.json', dict(complete=True, mode='benchmark', arm=arm, block=block,
                                                        protocol_id=spec['protocol_id'], run_id=directory.name,
                                                        expected_timed=12, completed_timed=12, v28_config=settings))
        variants[variant] = dict(binding=str(binding_path), run_dirs=run_dirs)
    family_path = root / 'family.private.json'
    write_json(family_path, dict(variants=variants))
    golds = {}
    for ds in datasets:
        path = root / (ds + '.gold.private.json')
        write_json(path, {cell['id']: 'B' for cell in private_cells if cell['dataset'] == ds})
        golds[ds] = str(path)
    return family_path, variants, golds


class PreviewSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.family, self.variants, self.golds = toy_bundle(self.root)

    def ledger(self, variant='dense', block=0, private=False):
        name = 'completions.private.jsonl' if private else 'records.jsonl'
        return Path(self.variants[variant]['run_dirs'][block]) / name

    def test_complete_preview_scores_unchanged_and_publishes_no_private_material(self):
        out = self.root / 'new_public_out'
        # External final-channel and NeMo dependencies are replaced only in the
        # public toy test; score_panel and v15 task.score themselves run intact.
        with patch.object(summary.old.task, 'final_text', side_effect=lambda raw:
                          raw.partition('<channel|>')[2].partition('<turn|>')[0]), \
                patch.object(summary.old.task, 'predict', side_effect=lambda texts: ['B'] * len(texts)) as predict:
            result = summary.main(['--family', str(self.family), '--out-dir', str(out),
                                   '--bootstrap-reps', '80'] +
                                  [value for ds, path in self.golds.items() for value in ('--gold', ds + '=' + path)])
        self.assertEqual(len(predict.call_args.args[0]), 288)
        self.assertTrue(all(text.startswith('Answer: B') for text in predict.call_args.args[0]))
        self.assertTrue(result['preview_only'])
        self.assertEqual((result['question_clusters'], result['requests_per_variant']), (6, 48))
        extras = [row for row in result['comparisons'] if row['comparison_role'] == 'descriptive_extra']
        self.assertEqual([(row['candidate'], row['reference']) for row in extras], [('main_legacy', 'dense')])
        primary = result['comparisons'][0]
        self.assertEqual((primary['candidate'], primary['reference']), ('q64_release', 'main_release'))
        for metric in ('W', 'S', 'SN'):
            self.assertAlmostEqual(primary[metric]['ratio'], 8 / 9)
        self.assertEqual(primary['N']['ratio'], 1)
        self.assertEqual(primary['P']['ratio'], 1)
        self.assertEqual(primary['strict_correct']['candidate_count'], 48)
        dense = result['variants'][0]
        self.assertEqual(dense['N']['sum'], 624)
        self.assertEqual(dense['N']['mean'], 13)
        self.assertEqual(dense['N']['distribution'], [{'N': n, 'requests': 12} for n in (10, 12, 14, 16)])
        self.assertEqual([row['requests'] for row in dense['by_engine_seed']], [12] * 4)
        for path in out.iterdir():
            public = path.read_text(encoding='utf-8')
            for forbidden in ('PRIVATE_TOY_', str(self.root), 'run_id', 'fingerprint', 'protocol_id', '<channel|>', 'receipts'):
                self.assertNotIn(forbidden, public)

    def test_original_scorer_strict_correctness_still_requires_eos(self):
        cells, completions = summary.load_family(self.family)
        key = next(iter(cells))
        cells[key]['q64_release']['finish_reason'] = 'length'
        completions[key + ('q64_release',)]['finish_reason'] = 'length'
        with patch.object(summary.old.task, 'final_text', return_value='Answer: B'), \
                patch.object(summary.old.task, 'predict', side_effect=lambda texts: ['B'] * len(texts)):
            summary.old.score_panel(cells, completions, self.golds)
        rows = [values['q64_release'] for values in cells.values()]
        self.assertEqual(sum(row['task_correct'] for row in rows), 48)
        self.assertEqual(sum(row['strict_correct'] for row in rows), 47)

    def test_public_record_guard_failures(self):
        path = self.ledger()
        original = path.read_text(encoding='utf-8')
        mutations = [dict(qualification_only=True), dict(qualification_only=None), dict(engine_seed=9000),
                     dict(seed_applied=True), dict(cpu_threads=4), dict(cpu_threads=True), dict(chunk=8192),
                     dict(deploy_commit='different'), dict(host='other_alias'), dict(gpu_uuid='GPU-OTHER'),
                     dict(graph_captures_timed=1), dict(graph_captures_timed=None), dict(wall_s=0),
                     dict(decode_span_s=-1), dict(prefill_s=float('nan')), dict(denoise_forward_count=11),
                     dict(execution_count_source='scheduler_only'), dict(adapter_sha256='unwanted'),
                     dict(protocol_id='v28_other'), dict(compilation_config='PIECEWISE')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = read_rows(path)
                rows[0].update(mutation)
                write_rows(path, rows)
                with self.assertRaises(ValueError):
                    summary.load_family(self.family)
                path.write_text(original, encoding='utf-8')

    def test_timed_monitor_backend_compile_and_missing_counter_rejected(self):
        path = self.ledger()
        original = path.read_text(encoding='utf-8')
        for field in ('num_backend_compilations', 'num_inductor_compiles', 'jit_monitor_triton_events',
                      'jit_monitor_cute_events', 'jit_monitor_unknown_events', 'jit_monitor_all_events'):
            for value in (1, None):
                with self.subTest(field=field, value=value):
                    rows = read_rows(path)
                    rows[0]['compilation_deltas'][field] = value
                    write_rows(path, rows)
                    with self.assertRaises(ValueError):
                        summary.load_family(self.family)
                    path.write_text(original, encoding='utf-8')

    def test_missing_duplicate_and_unexpected_cell_rejected(self):
        path = self.ledger()
        original = path.read_text(encoding='utf-8')
        rows = read_rows(path)
        alternatives = [rows[:-1], rows + rows[:1], rows[:-1] + rows[:1],
                        [dict(row, repeat=2) if i == 0 else row for i, row in enumerate(rows)],
                        [dict(row, index=8) if i == 0 else row for i, row in enumerate(rows)]]
        for alternative in alternatives:
            with self.subTest(rows=len(alternative)):
                write_rows(path, alternative)
                with self.assertRaises(ValueError):
                    summary.load_family(self.family)
                path.write_text(original, encoding='utf-8')

    def test_terminal_open_failed_bad_counts_identity_mode_or_block_rejected(self):
        path = self.ledger().parent / 'terminal.json'
        original = path.read_text(encoding='utf-8')
        for mutation in (dict(complete=False), dict(complete=1), dict(mode='qualification'),
                         dict(expected_timed=13), dict(completed_timed=11), dict(run_id='wrong'),
                         dict(arm='method'), dict(block=1), dict(protocol_id='wrong'), dict(v28_config={})):
            with self.subTest(mutation=mutation):
                status = json.loads(original)
                status.update(mutation)
                write_json(path, status)
                with self.assertRaises(ValueError):
                    summary.load_family(self.family)
                path.write_text(original, encoding='utf-8')
        path.unlink()
        with self.assertRaises(ValueError):
            summary.load_family(self.family)

    def test_private_missing_duplicate_wrong_id_or_wrong_execution_rejected(self):
        path = self.ledger(private=True)
        original = path.read_text(encoding='utf-8')
        rows = read_rows(path)
        alternatives = [rows[:-1], rows + rows[:1]]
        for mutation in (dict(id='PRIVATE_TOY_OTHER'), dict(run_id='different'),
                         dict(finish_reason='length'), dict(arm='method'), dict(completion=None)):
            alternatives.append([dict(row, **mutation) if i == 0 else row for i, row in enumerate(rows)])
        for alternative in alternatives:
            with self.subTest(rows=len(alternative)):
                write_rows(path, alternative)
                with self.assertRaises(ValueError):
                    summary.load_family(self.family)
                path.write_text(original, encoding='utf-8')

    def test_sources_and_deployment_receipt_are_rechecked_from_bytes(self):
        source = self.root / 'deploy' / summary.SOURCES[-1]
        original = source.read_bytes()
        source.write_bytes(original + b'# drift\n')
        with self.assertRaisesRegex(ValueError, 'byte drift'):
            summary.load_family(self.family)
        source.write_bytes(original)
        (self.root / 'deploy' / 'DEPLOY_SHA').write_text('other-deploy', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'deployment commit drift'):
            summary.load_family(self.family)

    def mutate_spec(self, variant, mutation):
        path = Path(self.variants[variant]['binding'])
        binding = json.loads(path.read_text(encoding='utf-8'))
        spec = json.loads(Path(binding['spec']).read_text(encoding='utf-8'))
        mutation(spec)
        write_json(Path(binding['spec']), spec)
        binding['files'][binding['spec']] = summary.qualify.panel.digest(binding['spec'])
        binding['protocol_id'] = spec['protocol_id']
        write_json(path, binding)

    def test_repeat_labels_cannot_replace_four_distinct_seed_blocks(self):
        self.mutate_spec('q64_release', lambda spec: spec['blocks'][1].update(engine_seed=7001))
        with self.assertRaisesRegex(ValueError, 'independent engine seeds'):
            summary.load_family(self.family)

    def test_seed_schedule_drift_rejected_even_when_each_variant_is_individually_frozen(self):
        self.mutate_spec('q64_release', lambda spec: spec['blocks'][0].update(engine_seed=9999))
        with self.assertRaisesRegex(ValueError, 'family question/seed'):
            summary.load_family(self.family)

    def test_distinct_protocols_cannot_be_relabelled_to_one_id(self):
        self.mutate_spec('q64_release', lambda spec: spec.update(protocol_id='v28_toy_q128_release'))
        with self.assertRaisesRegex(ValueError, 'distinct identities'):
            summary.load_family(self.family)

    def test_software_drift_across_blocks_rejected(self):
        path = self.ledger(block=1)
        rows = read_rows(path)
        for row in rows:
            row['vllm'] = 'other_vllm'
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, 'software drift'):
            summary.load_family(self.family)

    def test_missing_duplicate_or_unknown_family_variants_rejected(self):
        original = json.loads(self.family.read_text(encoding='utf-8'))
        alternatives = []
        missing = copy.deepcopy(original)
        del missing['variants']['native']
        alternatives.append(missing)
        unknown = copy.deepcopy(original)
        unknown['variants']['extra'] = unknown['variants']['native']
        alternatives.append(unknown)
        duplicate = copy.deepcopy(original)
        duplicate['variants']['dense']['run_dirs'][1] = duplicate['variants']['dense']['run_dirs'][0]
        alternatives.append(duplicate)
        for alternative in alternatives:
            write_json(self.family, alternative)
            with self.assertRaises(ValueError):
                summary.load_family(self.family)

    def test_invalid_family_never_scores_or_creates_output(self):
        path = self.ledger()
        rows = read_rows(path)
        rows[0]['engine_seed'] = 42
        write_rows(path, rows)
        out = self.root / 'must_not_exist'
        with patch.object(summary.old, 'score_panel') as scorer:
            with self.assertRaises(ValueError):
                summary.main(['--family', str(self.family), '--gold', 'toy=' + str(self.root / 'missing'),
                              '--out-dir', str(out)])
        scorer.assert_not_called()
        self.assertFalse(out.exists())

    def test_receipt_method_query_and_lifecycle_guards_are_not_bypassed(self):
        path = self.ledger('q64_release')
        original = path.read_text(encoding='utf-8')
        for mutation in ('qblock', 'release', 'allocator', 'probe'):
            rows = read_rows(path)
            receipt = rows[0]['receipts']
            if mutation == 'qblock':
                receipt['method']['effective_method']['q_block'] = 128
            elif mutation == 'release':
                receipt['adapter']['canvas_release_calls'] = 0
            elif mutation == 'allocator':
                receipt['adapter']['allocator'] = {}
            else:
                receipt['adapter']['kv_probe_timed_checks'] = 1
            write_rows(path, rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summary.load_family(self.family)
            path.write_text(original, encoding='utf-8')


class QuestionClusterTests(unittest.TestCase):
    def test_all_repeats_of_question_remain_a_cluster(self):
        groups = {(ds, index): [100 * index] * 8 for ds in ('toy32', 'toy64', 'toy96') for index in (0, 1)}
        ci = summary.question_ci(groups, geometric=False, reps=2000)
        self.assertGreater(ci[1] - ci[0], 50)
        self.assertEqual(ci, summary.question_ci({key: values * 2 for key, values in groups.items()},
                                                geometric=False, reps=2000))

    def test_length_stratum_weights_are_fixed(self):
        groups = {(ds, index): [value] * 8 for ds, value in (('toy32', 1), ('toy64', 100), ('toy96', 10000))
                  for index in (0, 1)}
        for bound in summary.question_ci(groups, reps=80):
            self.assertAlmostEqual(bound, 100)


if __name__ == '__main__':
    unittest.main()
