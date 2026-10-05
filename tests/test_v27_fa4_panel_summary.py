"""Timing joins must retain the scorer's same-host, immutable-first contract."""
import csv
import json

import pytest

from scripts.v27_fa4_panel_summary import main, read_cells


def _inputs(tmp_path, runs=None, scores=None, identity=None):
    identity = identity or dict(event='start', host='hostA', gpu_uuid='GPU-A',
                               protocol_id='protocol', model_revision='model',
                               source_hashes={'code': 'source'}, substrate='piecewise_v5')
    runs = runs or [dict(event='run', role='attempt0', ok=True, dataset='task', id='q', seed=101,
                        arm=arm, cell_id=arm, host='hostA', gpu_uuid='GPU-A', api_wall_s=wall,
                        phase_evidence={'prefill_end_to_finish_gpu_s': wall / 2},
                        decoder_calls=10, canvases=1, output_tokens=256,
                        substrate={'substrate': 'piecewise_v5'}, substrate_new_graphs=0)
                    for arm, wall in [('D_fa4_allkept', 10), ('method', 8)]]
    scores = scores or [dict(dataset=r['dataset'], id=r['id'], seed=str(r['seed']), arm=r['arm'],
                             cell_id=r['cell_id'], host=r['host'], gpu_uuid=r['gpu_uuid'],
                             first_status='success', scored_first='True', strict_correct='True') for r in runs]
    scored, ledger = tmp_path / 'scored.csv', tmp_path / 'ledger.jsonl'
    with scored.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    ledger.write_text('\n'.join(json.dumps(e) for e in [identity, *runs]) + '\n', encoding='utf-8')
    return scored, ledger, runs, scores, identity


def test_summary_preserves_paired_ratios(tmp_path):
    scored, ledger, *_ = _inputs(tmp_path)
    out = tmp_path / 'summary.csv'
    main([str(tmp_path / 'summary.md'), str(out), str(scored), str(ledger)])
    row = next(r for r in csv.DictReader(out.open(encoding='utf-8')) if r['arm'] == 'method')
    assert float(row['W']) == .8 and float(row['S']) == .8 and float(row['SN']) == .8
    assert row['correct'] == row['base_correct'] == '1'


@pytest.mark.parametrize('failed_first', [False, True])
def test_duplicate_first_cannot_replace_earlier_output(tmp_path, failed_first):
    scored, ledger, runs, *_ = _inputs(tmp_path)
    if failed_first:
        runs[0]['ok'] = False
    ledger.write_text(ledger.read_text(encoding='utf-8').splitlines()[0] + '\n' +
                      '\n'.join(json.dumps(e) for e in [*runs, dict(runs[0], ok=True, api_wall_s=1)]),
                      encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate ledger first'):
        read_cells(scored, [ledger])


@pytest.mark.parametrize('field,value', [('host', 'hostB'), ('gpu_uuid', 'GPU-B'), ('cell_id', 'other')])
def test_score_and_timing_execution_must_match(tmp_path, field, value):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    scores[0][field] = value
    _inputs(tmp_path, runs, scores, identity)
    with pytest.raises(ValueError, match='scored execution identity'):
        read_cells(scored, [ledger])


@pytest.mark.parametrize('field,value', [('host', 'hostB'), ('gpu_uuid', 'GPU-B'),
                                        ('substrate', 'piecewise_v4'), ('source_hashes', {'code': 'other'}),
                                        ('protocol_id', 'other'), ('model_revision', 'other')])
def test_arms_cannot_be_paired_across_execution_conditions(tmp_path, field, value):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    changed = dict(identity, **{field: value})
    if field in ('host', 'gpu_uuid'):
        runs[1][field] = scores[1][field] = value
    if field == 'substrate':
        runs[1]['substrate'] = {'substrate': value}
    _inputs(tmp_path, runs, scores, identity)
    ledger.write_text('\n'.join(json.dumps(e) for e in [identity, runs[0], changed, runs[1]]), encoding='utf-8')
    with pytest.raises(ValueError, match='cell arms differ'):
        read_cells(scored, [ledger])


def test_unscored_success_cannot_count_as_incorrect(tmp_path):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    scores[0]['scored_first'] = 'False'
    _inputs(tmp_path, runs, scores, identity)
    with pytest.raises(ValueError, match='qualified score'):
        read_cells(scored, [ledger])


def test_missing_host_ledger_cannot_silently_shrink_panel(tmp_path):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    _inputs(tmp_path, runs[:1], scores, identity)
    with pytest.raises(ValueError, match='timing ledger inventory'):
        read_cells(scored, [ledger])


def test_duplicate_score_is_rejected(tmp_path):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    _inputs(tmp_path, runs, scores + [scores[0]], identity)
    with pytest.raises(ValueError, match='duplicate scored first'):
        read_cells(scored, [ledger])


def test_unknown_graph_counter_is_excluded_from_clean_timing(tmp_path):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    runs[1]['substrate_new_graphs'] = None
    _inputs(tmp_path, runs, scores, identity)
    out = tmp_path / 'summary.csv'
    main([str(tmp_path / 'summary.md'), str(out), str(scored), str(ledger)])
    row = next(r for r in csv.DictReader(out.open(encoding='utf-8')) if r['arm'] == 'method')
    assert float(row['W']) == .8
    assert row['Wc'] == '' and row['timed_graph_counter_unknown'] == '1'


def test_disjoint_hosts_can_be_pooled_by_question(tmp_path):
    scored, ledger, runs, scores, identity = _inputs(tmp_path)
    second = dict(identity, host='hostB', gpu_uuid='GPU-B')
    extra = [dict(r, id='q2', cell_id=r['arm'] + '2', host='hostB', gpu_uuid='GPU-B') for r in runs]
    extra_scores = [dict(s, id='q2', cell_id=s['arm'] + '2', host='hostB', gpu_uuid='GPU-B') for s in scores]
    _inputs(tmp_path, runs, scores + extra_scores, identity)
    ledger.write_text('\n'.join(json.dumps(e) for e in [identity, *runs, second, *extra]), encoding='utf-8')
    assert len(read_cells(scored, [ledger])) == 2
