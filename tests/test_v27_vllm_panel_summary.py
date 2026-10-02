"""Synthetic CPU guard tests; no GPU, model, private dataset or NeMo dependency."""
import json

import pytest

from scripts import v15_longbench_task as task
from scripts import v27_vllm_panel_summary as summary


def fixtures():
    protocol = dict(schema='v27_vllm_panel_spec_v1', name='synthetic panel', protocol_id='p1',
                    arms=['dense', 'method'], controls=['native', 'allkept'],
                    datasets={'lb32': {'indices': [0, 1, 2], 'repeats': [0, 1]}},
                    control_indices={'lb32': [0, 1]},
                    settings=dict(max_model_len=40000, chunk=8192, gpu_memory_utilization=.9, block_size=64),
                    primary_receipt_method={'decision_interval': 6, 'score_period': 64,
                                            'risk_state': 'dense_prefix', 'carry_first': True},
                    arm_settings={arm: dict(compilation_config='default' if arm == 'dense' else 'PIECEWISE',
                                            cudagraph_mode='default' if arm == 'dense' else 'PIECEWISE')
                                  for arm in summary.ARMS})
    records, completions = [], []
    for dataset, index, repeat, arm in sorted(summary.expected_inventory(protocol)):
        wall = 8 if arm == 'method' else 10
        row = dict(schema=summary.SCHEMA, protocol_id='p1', deploy_commit='deploy', host='aliasA',
                   gpu_uuid='GPU-A', dataset=dataset, index=index, repeat=repeat, arm=arm,
                   engine_seed=123, seed_applied=False, measurement_mode='request_boundary_sync',
                   graph_captures_timed=0, wall_s=wall, decode_span_s=wall / 2, prefill_s=2,
                   denoise_forward_count=10, output_tokens=16, finish_reason='stop',
                   receipts=None,
                   adapter_sha256=None if arm == 'dense' else 'adapter',
                   method_fingerprint='method-source' if arm == 'method' else None,
                   max_model_len=40000, chunk=8192, gpu_memory_utilization=.9, block_size=64,
                   torch='2.9', vllm='0.30.0', run_id=f'{arm}-run')
        row.update(protocol['arm_settings'][arm])
        if arm != 'dense':
            row['receipts'] = dict(adapter=dict(begins=10, observes=10, order_errors=0, global_calls=50,
                                                split_fa4_calls=50 if arm != 'native' else 0),
                                   method=None, timing=None)
        if arm == 'method':
            row['receipts']['method'] = dict(effective_method=dict(protocol['primary_receipt_method']),
                                            active_layers=[5, 11, 17, 23, 29], gated_native_calls=0,
                                            layer_native_calls=0, unsupported_mask_refreshes=0,
                                            fused_observations=5, dp_routes=5)
        records.append(row)
        completions.append(dict(dataset=dataset, index=index, repeat=repeat, arm=arm,
                                run_id=row['run_id'], finish_reason=row['finish_reason'],
                                id=f'private-item-{index}', completion='<channel|>Answer: B<turn|>'))
    return protocol, records, completions


def inputs(tmp_path, protocol=None, records=None, completions=None):
    default_protocol, default_records, default_completions = fixtures()
    protocol = default_protocol if protocol is None else protocol
    records = default_records if records is None else records
    completions = default_completions if completions is None else completions
    record_path = tmp_path / 'records.jsonl'
    completion_path = tmp_path / 'private.jsonl'
    protocol_path = tmp_path / 'protocol.json'
    record_path.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    completion_path.write_text('\n'.join(json.dumps(r) for r in completions), encoding='utf-8')
    protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
    return record_path, completion_path, protocol_path


def load(tmp_path, protocol=None, records=None, completions=None):
    public, private, spec = inputs(tmp_path, protocol, records, completions)
    return summary.load_panel([public], [private], json.loads(spec.read_text(encoding='utf-8')))


@pytest.fixture
def predict(monkeypatch):
    seen = []
    def fake_predict(texts):
        seen.extend(texts)
        return ['B' if 'Answer: B' in text else None for text in texts]
    monkeypatch.setattr(task, 'predict', fake_predict)
    return seen


def test_complete_panel_ratios_controls_and_private_redaction(tmp_path, predict):
    public, private, spec = inputs(tmp_path)
    protocol, records, _ = fixtures()
    public_paths = []
    for arm in summary.ARMS:
        directory = tmp_path / arm
        directory.mkdir()
        rows = [row for row in records if row['arm'] == arm]
        record_path = directory / 'records.jsonl'
        record_path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        terminal = dict(complete=True, protocol_id='p1', arm=arm, run_id=f'{arm}-run',
                        completed_timed=len(rows), expected_timed=len(rows))
        (directory / 'terminal.json').write_text(json.dumps(terminal), encoding='utf-8')
        public_paths.append(str(record_path))
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps({f'private-item-{i}': 'B' for i in range(3)}), encoding='utf-8')
    out = tmp_path / 'out'
    result = summary.main(['--records', *public_paths, '--completions', str(private), '--protocol', str(spec),
                           '--gold', f'lb32={gold}', '--out-dir', str(out), '--bootstrap-reps', '80'])
    primary = result['comparisons'][0]
    assert primary['cells'] == 6 and primary['items'] == 3
    assert primary['W'] == pytest.approx(.8)
    assert primary['S'] == pytest.approx(.8) and primary['SN'] == pytest.approx(.8)
    assert primary['N'] == 1 and primary['W_ci95'] == pytest.approx([.8, .8])
    assert primary['strict_correct'] == primary['base_strict_correct'] == 6
    assert {row['base'] for row in result['comparisons'] if row['arm'] == 'method'} == {'dense', 'native', 'allkept'}
    assert [row['cells'] for row in result['arms']] == [6, 6, 4, 4]
    assert predict and all(text == 'Answer: B' for text in predict)
    for path in out.iterdir():
        text = path.read_text(encoding='utf-8')
        assert 'private-item' not in text and '<channel|>' not in text
        assert 'adapter' not in text and 'predicted' not in text and str(gold) not in text
    assert 'not a sampling seed' in (out / 'summary.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('private', [False, True])
def test_duplicate_records_never_replace_first(tmp_path, private):
    protocol, records, completions = fixtures()
    if private:
        completions.append(dict(completions[0], completion='replacement'))
    else:
        records.append(dict(records[0], wall_s=.01))
    with pytest.raises(ValueError, match='duplicate'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('private', [False, True])
def test_missing_records_cannot_shrink_panel(tmp_path, private):
    protocol, records, completions = fixtures()
    (completions if private else records).pop()
    with pytest.raises(ValueError, match='inventory is incomplete'):
        load(tmp_path, protocol, records, completions)


def test_empty_public_records_fail(tmp_path):
    with pytest.raises(ValueError, match='inventory is incomplete'):
        load(tmp_path, records=[])


def test_unknown_execution_fails(tmp_path):
    protocol, records, completions = fixtures()
    records[0]['index'] = 99
    with pytest.raises(ValueError, match='outside frozen inventory'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('field,value', [('host', 'aliasB'), ('gpu_uuid', 'GPU-B'),
                                        ('deploy_commit', 'different'), ('engine_seed', 456),
                                        ('max_model_len', 50000), ('chunk', 4096),
                                        ('gpu_memory_utilization', .8), ('block_size', 128),
                                        ('torch', 'different'), ('vllm', 'different')])
def test_same_cell_environment_and_settings_drift_fail(tmp_path, field, value):
    protocol, records, completions = fixtures()
    records[0][field] = value
    with pytest.raises(ValueError, match='drift'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('value', [None, 1, -1, False, '0'])
def test_unknown_or_timed_graph_capture_fails(tmp_path, value):
    protocol, records, completions = fixtures()
    records[0]['graph_captures_timed'] = value
    with pytest.raises(ValueError, match='graph captures'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('field,value', [('wall_s', 0), ('decode_span_s', -1), ('prefill_s', 0),
                                        ('wall_s', float('nan')), ('decode_span_s', float('inf')),
                                        ('denoise_forward_count', 0), ('denoise_forward_count', 1.5),
                                        ('output_tokens', -1), ('wall_s', True)])
def test_invalid_numeric_measurements_fail(tmp_path, field, value):
    protocol, records, completions = fixtures()
    records[0][field] = value
    with pytest.raises(ValueError, match='positive|integer'):
        load(tmp_path, protocol, records, completions)


def test_predeclared_graph_difference_is_required(tmp_path):
    protocol, records, completions = fixtures()
    records[0]['cudagraph_mode'] = 'FULL'
    with pytest.raises(ValueError, match='predeclared'):
        load(tmp_path, protocol, records, completions)
    protocol.pop('arm_settings')
    with pytest.raises(ValueError, match='predeclare'):
        load(tmp_path, protocol)


@pytest.mark.parametrize('field,value', [('protocol_id', 'other'), ('schema', 'old'),
                                        ('seed_applied', True), ('measurement_mode', 'per_step_sync')])
def test_execution_contract_is_fixed(tmp_path, field, value):
    protocol, records, completions = fixtures()
    records[0][field] = value
    with pytest.raises(ValueError, match='identity|contract'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('field', ['run_id', 'finish_reason', 'id'])
def test_private_execution_join_is_exact(tmp_path, field):
    protocol, records, completions = fixtures()
    completions[0][field] = 'length' if field == 'finish_reason' else 'other'
    with pytest.raises(ValueError, match='identity differs'):
        load(tmp_path, protocol, records, completions)


def test_adapter_source_drift_is_rejected_but_dense_null_is_allowed(tmp_path):
    protocol, records, completions = fixtures()
    assert load(tmp_path, protocol, records, completions)
    next(r for r in records if r['arm'] == 'native')['adapter_sha256'] = 'different'
    with pytest.raises(ValueError, match='adapter source'):
        load(tmp_path, protocol, records, completions)


def test_method_receipt_checks_frozen_config(tmp_path):
    protocol, records, completions = fixtures()
    method = next(row for row in records if row['arm'] == 'method')
    method['receipts']['method']['effective_method']['decision_interval'] = 12
    with pytest.raises(ValueError, match='execution receipt'):
        load(tmp_path, protocol, records, completions)


def test_method_fingerprint_drift_is_rejected(tmp_path):
    protocol, records, completions = fixtures()
    next(r for r in records if r['arm'] == 'method')['method_fingerprint'] = 'changed'
    with pytest.raises(ValueError, match='fingerprint drift'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('field,value', [('order_errors', 1), ('begins', 9), ('observes', 9), ('global_calls', 49)])
def test_adapter_clock_and_global_coverage_are_required(tmp_path, field, value):
    protocol, records, completions = fixtures()
    method = next(row for row in records if row['arm'] == 'method')
    method['receipts']['adapter'][field] = value
    with pytest.raises(ValueError, match='execution receipt'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('field,value', [('active_layers', [5, 11]), ('gated_native_calls', 1),
                                        ('layer_native_calls', 1), ('unsupported_mask_refreshes', 1),
                                        ('fused_observations', 0), ('dp_routes', 0)])
def test_sparse_path_receipts_cannot_hide_native_fallback(tmp_path, field, value):
    protocol, records, completions = fixtures()
    method = next(row for row in records if row['arm'] == 'method')
    method['receipts']['method'][field] = value
    with pytest.raises(ValueError, match='execution receipt'):
        load(tmp_path, protocol, records, completions)


def test_all_arms_same_seed_drift_still_violates_frozen_block(tmp_path):
    protocol, records, completions = fixtures()
    protocol['blocks'] = [{'engine_seed': 123, 'repeats': [0]}, {'engine_seed': 456, 'repeats': [1]}]
    for row in records:
        if row['repeat'] == 1:
            row['engine_seed'] = 456
    assert load(tmp_path, protocol, records, completions)
    for row in records:
        if row['repeat'] == 1:
            row['engine_seed'] = 789
    with pytest.raises(ValueError, match='frozen block'):
        load(tmp_path, protocol, records, completions)


def test_stop_and_length_follow_v15_strict_quality(tmp_path, predict):
    protocol, records, completions = fixtures()
    for rows in (records, completions):
        rows[0]['finish_reason'] = 'length'
    cells, joined = load(tmp_path, protocol, records, completions)
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps({f'private-item-{i}': 'B' for i in range(3)}), encoding='utf-8')
    summary.score_panel(cells, joined, {'lb32': gold})
    first = cells[('lb32', 0, 0)][records[0]['arm']]
    assert first['task_correct'] is True and first['strict_correct'] is False
    assert 'predicted' not in first


def test_thinking_channel_is_never_mined(tmp_path, predict):
    protocol, records, completions = fixtures()
    completions[0]['completion'] = '<|channel>thought Answer: B'
    cells, joined = load(tmp_path, protocol, records, completions)
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps({f'private-item-{i}': 'B' for i in range(3)}), encoding='utf-8')
    summary.score_panel(cells, joined, {'lb32': gold})
    first = cells[('lb32', 0, 0)][records[0]['arm']]
    assert not first['parsed'] and not first['strict_correct']
    assert predict[0] == ''


def test_missing_gold_never_becomes_an_incorrect_answer(tmp_path, predict):
    cells, joined = load(tmp_path)
    gold = tmp_path / 'gold.json'
    gold.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='missing from gold'):
        summary.score_panel(cells, joined, {'lb32': gold})
    assert predict == []


def test_cluster_bootstrap_keeps_repeated_requests_together():
    # Twenty perfectly correlated repeats per item must retain item uncertainty.
    clustered = summary.cluster_ci({0: [1.] * 20, 1: [4.] * 20}, reps=400)
    assert clustered == pytest.approx([1., 4.])
    assert summary.cluster_ci({0: [1., 4.]}, reps=400) == [None, None]
    assert summary.cluster_ci({0: [-1.] * 20, 1: [1.] * 20}, geometric=False, reps=400) == [-1., 1.]


def test_item_accuracy_difference_ci_and_exploratory_label(tmp_path, predict):
    cells, joined = load(tmp_path)
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps({f'private-item-{i}': 'B' for i in range(3)}), encoding='utf-8')
    summary.score_panel(cells, joined, {'lb32': gold})
    for key, rows in cells.items():
        if key[1] == 0:
            rows['method']['strict_correct'] = False
    result = summary.summarize(cells, bootstrap_reps=400)
    primary = result['comparisons'][0]
    assert primary['accuracy_difference'] == pytest.approx(-1 / 3)
    assert primary['accuracy_difference_ci95'] == [-1., 0.]
    assert primary['exploratory_mcnemar_exact_p'] == pytest.approx(.5)
    assert 'exploratory only' in result['inference']


def test_disjoint_hosts_for_different_items_are_allowed(tmp_path):
    protocol, records, completions = fixtures()
    for row in records:
        if row['index'] == 2:
            row['host'], row['gpu_uuid'] = 'aliasB', 'GPU-B'
    cells, _ = load(tmp_path, protocol, records, completions)
    assert len(cells) == 6


@pytest.mark.parametrize('field', ['indices', 'repeats'])
def test_duplicate_protocol_schedule_is_rejected(tmp_path, field):
    protocol, records, completions = fixtures()
    protocol['datasets']['lb32'][field].append(protocol['datasets']['lb32'][field][0])
    with pytest.raises(ValueError, match='duplicate'):
        load(tmp_path, protocol, records, completions)


def test_summary_never_overwrites_existing_artifact(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'summary.md').write_text('immutable', encoding='utf-8')
    with pytest.raises(ValueError, match='already exist'):
        summary.write_summary({}, out)
    assert (out / 'summary.md').read_text(encoding='utf-8') == 'immutable'


def test_records_split_across_files_keep_exact_inventory(tmp_path):
    protocol, records, completions = fixtures()
    public, private, _ = inputs(tmp_path, protocol, records[:10], completions)
    extra = tmp_path / 'extra.jsonl'
    extra.write_text('\n'.join(json.dumps(r) for r in records[10:]), encoding='utf-8')
    assert len(summary.load_panel([public, extra], [private], protocol)[0]) == 6


@pytest.mark.parametrize('field,value', [('max_model_len', 50000), ('chunk', 4096),
                                        ('gpu_memory_utilization', .8), ('block_size', 128)])
def test_all_arms_settings_drift_violates_frozen_protocol(tmp_path, field, value):
    protocol, records, completions = fixtures()
    for row in records:
        row[field] = value
    with pytest.raises(ValueError, match='frozen protocol'):
        load(tmp_path, protocol, records, completions)


@pytest.mark.parametrize('field', ['num_backend_compilations', 'num_inductor_compiles'])
@pytest.mark.parametrize('value', [1, -1, None, False])
def test_timed_compilation_is_rejected(tmp_path, field, value):
    protocol, records, completions = fixtures()
    records[0]['compilation_deltas'] = {field: value}
    with pytest.raises(ValueError, match='compilation delta'):
        load(tmp_path, protocol, records, completions)


def test_zero_timed_compilation_deltas_are_allowed(tmp_path):
    protocol, records, completions = fixtures()
    for row in records:
        row['compilation_deltas'] = dict(num_backend_compilations=0, num_inductor_compiles=0)
    assert load(tmp_path, protocol, records, completions)


def closed_run(tmp_path):
    protocol, records, _ = fixtures()
    rows = [row for row in records if row['arm'] == 'dense']
    path = tmp_path / 'records.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    terminal = dict(protocol_id='p1', arm='dense', run_id='dense-run', complete=True,
                    completed_timed=len(rows), expected_timed=len(rows))
    terminal_path = tmp_path / 'terminal.json'
    terminal_path.write_text(json.dumps(terminal), encoding='utf-8')
    return protocol, path, terminal, terminal_path


def test_complete_worker_terminal_is_required(tmp_path):
    protocol, path, terminal, terminal_path = closed_run(tmp_path)
    summary.prevalidate_closed_runs([path], protocol)
    terminal_path.unlink()
    with pytest.raises(ValueError, match='terminal is missing'):
        summary.prevalidate_closed_runs([path], protocol)


@pytest.mark.parametrize('field,value,match', [('complete', False, 'incomplete or failed'),
                                             ('complete', 1, 'incomplete or failed'),
                                             ('arm', 'method', 'identity differs'),
                                             ('run_id', 'other', 'identity differs'),
                                             ('protocol_id', 'other', 'protocol identity'),
                                             ('completed_timed', 5, 'timed count'),
                                             ('expected_timed', 7, 'timed count'),
                                             ('completed_timed', None, 'timed count')])
def test_failed_mismatched_or_partial_worker_is_rejected(tmp_path, field, value, match):
    protocol, path, terminal, terminal_path = closed_run(tmp_path)
    terminal[field] = value
    terminal_path.write_text(json.dumps(terminal), encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        summary.prevalidate_closed_runs([path], protocol)


def test_terminal_counts_cannot_hide_truncated_record_file(tmp_path):
    protocol, path, terminal, terminal_path = closed_run(tmp_path)
    path.write_text('\n'.join(path.read_text(encoding='utf-8').splitlines()[:-1]), encoding='utf-8')
    with pytest.raises(ValueError, match='timed count'):
        summary.prevalidate_closed_runs([path], protocol)


def test_cli_rejects_open_worker_before_scoring(tmp_path, predict):
    public, private, spec = inputs(tmp_path)
    with pytest.raises(ValueError, match='terminal is missing'):
        summary.main(['--records', str(public), '--completions', str(private), '--protocol', str(spec),
                      '--gold', 'lb32=unused.json', '--out-dir', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists() and predict == []


def test_same_private_item_cannot_be_counted_as_two_questions(tmp_path):
    protocol, records, completions = fixtures()
    for row in completions:
        if row['index'] == 1:
            row['id'] = 'private-item-0'
    with pytest.raises(ValueError, match='duplicate private item'):
        load(tmp_path, protocol, records, completions)
