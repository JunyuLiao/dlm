"""v27 generic panel: freezer output passes the runner validator; guards fire."""
import hashlib
import json

import pytest

from scripts.v21_freeze_panel import freeze_v27
from scripts.v21_run import validate_protocol


def _row(i):
    prompt = f'prompt {i}'
    return dict(id=i, prompt=prompt, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                generation_budget=64, thinking=True)


def _setup(tmp_path, arms, warm=True, seeds=(101, 202)):
    original = dict(block_assignments={'0': dict(host='hA', gpu_uuid='GPU-A'),
                                       '1': dict(host='hB', gpu_uuid='GPU-B')})
    op = tmp_path / 'v20_protocol.json'
    op.write_text(json.dumps(original))
    binding = dict(panel_protocol_sha256=hashlib.sha256(op.read_bytes()).hexdigest(), status='frozen',
                   policy_sha256='p' * 64)
    bp = tmp_path / 'binding.json'
    bp.write_text(json.dumps(binding))
    pool = tmp_path / 'pool'
    pool.mkdir()
    (pool / 'longbench_v2_pool_manifest.json').write_text(json.dumps([_row('lb/1'), _row('lb/2'), _row('lb/3')]))
    (pool / 'aime26_pool_manifest.json').write_text(json.dumps([_row('aime26/1')]))
    spec = dict(name='t', ids={'longbench_v2': ['lb/1', 'lb/3'], 'aime26': ['aime26/1']},
                seeds=list(seeds), warm=warm, arms=arms)
    sp = tmp_path / 'spec.json'
    sp.write_text(json.dumps(spec))
    return sp, op, bp, pool


ARMS = {'D_native': dict(kind='native'), 'T_scope': dict(kind='fresh_T'),
        'M2c': dict(kind='method', parent='M1_R1_A8_current_output', extra=dict(mu_mode='pooled_compact')),
        'M3_R6_A16_p1': dict(kind='method', parent='M3_R3_A8_current_output',
                             extra=dict(decision_interval=6, score_period=16, threshold_shift='plus_ln2')),
        'B_A16': dict(kind='method', parent='B_A8_matched', extra=dict(hold_only=True, score_period=16))}


def test_freeze_balances_hosts_and_validates(tmp_path):
    sp, op, bp, pool = _setup(tmp_path, ARMS)
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert protocol['planned_executions'] == 3 * 2 * len(ARMS) * 2
    hosts = [(a['dataset'], a['id'], a['seed'], a['host']) for a in protocol['block_assignments'].values()]
    assert [h for *_, h in hosts[:4]] == ['hA', 'hB', 'hB', 'hA']
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))


def test_first_only_panel_and_single_seed(tmp_path):
    sp, op, bp, pool = _setup(tmp_path, ARMS, warm=False, seeds=(303,))
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert protocol['planned_executions'] == 3 * len(ARMS)
    assert {e['role'] for e in protocol['schedule']} == {'attempt0'}


@pytest.mark.parametrize('bad', [
    dict(kind='method', parent='M3_R3_A8_current_output', extra=dict(decision_interval=5)),
    dict(kind='method', parent='M3_R3_A8_current_output', extra=dict(score_period=32)),
    dict(kind='method', parent='M3_R3_A8_current_output', extra=dict(threshold_shift='plus_9ln2')),
    dict(kind='method', parent='M3_R3_A8_current_output', extra=dict(unknown=1)),
    dict(kind='weird')])
def test_bad_arm_rejected(tmp_path, bad):
    sp, op, bp, pool = _setup(tmp_path, dict(ARMS, bad=bad))
    with pytest.raises(ValueError):
        freeze_v27(sp, op, bp, pool, tmp_path / 'out')


def test_gold_or_prompt_drift_rejected(tmp_path):
    sp, op, bp, pool = _setup(tmp_path, ARMS)
    rows = json.loads((pool / 'longbench_v2_pool_manifest.json').read_text())
    rows[0]['prompt'] = 'changed'
    (pool / 'longbench_v2_pool_manifest.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        freeze_v27(sp, op, bp, pool, tmp_path / 'out')


def test_long_ruler_and_dense_c64_panel(tmp_path):
    sp, op, bp, pool = _setup(tmp_path, dict(ARMS, D_c64=dict(kind='dense_c64')))
    (pool / 'ruler32k_pool_manifest.json').write_text(json.dumps([_row('ruler32k/ruler_32768_cwe_p0000')]))
    spec = json.loads(sp.read_text())
    spec['ids'] = {'ruler32k': ['ruler32k/ruler_32768_cwe_p0000']}
    spec['extra_gold_sha256'] = {'ruler32k': 'a' * 64}
    sp.write_text(json.dumps(spec))
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert protocol['arm_contracts']['D_c64'] == dict(kind='v27_dense', control='D_c64',
                                                      scope='GLOBAL_ONLY_NATIVE_LOCAL')
    assert protocol['extra_gold_sha256'] == {'ruler32k': 'a' * 64}
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))
