"""Scope changes must preserve completed shard identity and schedule coverage."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from v31_value_longbench_scope_continue_attempt002 import ARMS, EXCLUDED, orders, validate_attempt, write_once


@pytest.fixture
def completed(tmp_path):
    root = tmp_path/'root'
    root.mkdir()
    (root/'source.py').write_text('frozen source\n')
    cells = tmp_path/'cells.json'
    cells.write_text(json.dumps([dict(dataset='lb', index=3, seed=1)]))
    identity = dict(source_sha256={'source.py': hashlib.sha256((root/'source.py').read_bytes()).hexdigest()},
        model_revision='rev', manifests_sha256={'lb':'manifest'}, model_source_inventory_sha256='inventory',
        host_fingerprint='host', gpu_identity='gpu', runtime_settings=dict(VALUE_AUDIT='1',VALUE_CLEAN_TIMING='0'))
    config = dict(identity, arm=ARMS[0], purpose='audit', threshold=None, cell_count=1,
        cells_sha256=hashlib.sha256(cells.read_bytes()).hexdigest())
    attempt = tmp_path/'attempt001'
    (attempt/'private').mkdir(parents=True)
    (attempt/'config.json').write_text(json.dumps(config))
    receipt = dict(status='complete', validation_errors=[], returncode=0, timed_cuda_captures=0)
    (attempt/'receipt.json').write_text(json.dumps(receipt))
    (attempt/'attempt_complete.json').write_text(json.dumps(receipt))
    rows = [dict(dataset='lb', index=3, panel_seed=1, repeat=0, cuda_graph_captures=0)]
    (attempt/'records.jsonl').write_text(json.dumps(rows[0])+'\n')
    (attempt/'private'/('run_'+ARMS[0]+'.private.jsonl')).write_text('{}\n')
    return attempt, cells, identity, root


def test_every_rotated_shard_has_exactly_the_eight_remaining_arms():
    assert len(ARMS) == 8 and EXCLUDED not in ARMS
    assert all(len(o) == 8 and set(o) == set(ARMS) for o in orders())
    assert orders()[1][0] == 'allkept_fa4'
    assert orders()[1][-2:] == ['dense_full_fix51994', 'dense_piecewise']


def test_reuse_accepts_completed_matching_generation(completed):
    attempt, cells, identity, root = completed
    config, rows = validate_attempt(attempt, cells, ARMS[0], 'audit', None, identity, root)
    assert len(rows) == 1 and config['purpose'] == 'audit'


@pytest.mark.parametrize('change', ['source', 'gpu', 'seed', 'threshold', 'receipt', 'capture'])
def test_reuse_rejects_changed_generation_or_invalid_receipt(completed, change):
    attempt, cells, identity, root = completed
    if change == 'source':
        (root/'source.py').write_text('changed\n')
    elif change == 'gpu':
        identity = copy.deepcopy(identity)
        identity['gpu_identity'] = 'other'
    elif change in ('seed', 'capture'):
        row = json.loads((attempt/'records.jsonl').read_text())
        row['panel_seed' if change == 'seed' else 'cuda_graph_captures'] = 2
        (attempt/'records.jsonl').write_text(json.dumps(row)+'\n')
    elif change == 'threshold':
        config = json.loads((attempt/'config.json').read_text())
        config['threshold'] = .1
        (attempt/'config.json').write_text(json.dumps(config))
    else:
        receipt = json.loads((attempt/'receipt.json').read_text())
        receipt['validation_errors'] = ['bad']
        (attempt/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        validate_attempt(attempt, cells, ARMS[0], 'audit', None, identity, root)


def test_offline_fresh_selector_scope_excludes_exact():
    import v31_value_offline_diagnostic_no_exact as diagnostic
    assert EXCLUDED not in diagnostic.SELECTORS
    assert 'value_v3b_approx_batch8' in diagnostic.SELECTORS
    assert len(diagnostic.SELECTORS) == 4


def test_restart_preserves_json_roundtrip_and_original_file(tmp_path):
    path = tmp_path/'config.json'
    value = dict(arms=ARMS, nested={'order': [ARMS]}, source_commit='first')
    write_once(path, value)
    original = path.read_bytes()
    write_once(path, value)
    write_once(path, json.loads(json.dumps(value)))
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        write_once(path, dict(value, source_commit='different'))
    assert path.read_bytes() == original
