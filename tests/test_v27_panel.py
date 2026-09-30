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


def test_execution_variant_fields_freeze(tmp_path):
    arms = dict(ARMS, M3f=dict(kind='method', parent='M3_R3_A8_current_output',
                               extra=dict(score_period=64, decision_interval=6, share_layers='one',
                                          consumer64=2, memory_caps='long', fused_observe=True)))
    sp, op, bp, pool = _setup(tmp_path, arms)
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert protocol['arm_contracts']['M3f']['fused_observe'] is True
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))


@pytest.mark.parametrize('dataset,row,ok', [
    ('ruler32k', dict(thinking=False, generation_budget=128), True),
    ('ruler64k', dict(thinking=False, generation_budget=30), True),
    ('ruler64k', dict(thinking=True, generation_budget=30), False),
    ('ruler32k', dict(thinking=False, generation_budget=8192), False),
    ('aime26', dict(thinking=True, generation_budget=8192), True),
    ('longbench_v2', dict(thinking=False, generation_budget=8192), False)])
def test_long_ruler_task_contract(dataset, row, ok):
    from scripts.v21_run import check_task_row
    if ok:
        check_task_row(dataset, row)
    else:
        with pytest.raises(ValueError):
            check_task_row(dataset, row)


def test_score_gold_paths_accept_long_only_panels():
    from pathlib import Path
    from scripts.v21_score import gold_paths_from
    assert gold_paths_from(None, None, None, []) is None
    assert gold_paths_from(None, None, None, ['ruler32k=/g/32.json', 'ruler64k=/g/64.json']) == {
        'ruler32k': Path('/g/32.json'), 'ruler64k': Path('/g/64.json')}
    assert set(gold_paths_from('r', 'a', 'l', [])) == {'ruler4k', 'aime26', 'longbench_v2'}


def test_long_longbench_bins_use_the_lb_contract(tmp_path):
    from scripts.v21_run import check_task_row
    from scripts.v27_datasets import EXTRA_GOLD, base_task
    assert base_task('longbench_v2_64k') == 'longbench_v2' and 'longbench_v2_32k' in EXTRA_GOLD
    check_task_row('longbench_v2_32k', dict(thinking=True, generation_budget=8192))
    with pytest.raises(ValueError):
        check_task_row('longbench_v2_64k', dict(thinking=False, generation_budget=128))
    sp, op, bp, pool = _setup(tmp_path, ARMS)
    (pool / 'longbench_v2_64k_pool_manifest.json').write_text(json.dumps([_row('longbench_v2/x64')]))
    spec = json.loads(sp.read_text())
    spec['ids'] = {'longbench_v2_64k': ['longbench_v2/x64']}
    spec['extra_gold_sha256'] = {'longbench_v2_64k': 'b' * 64}
    sp.write_text(json.dumps(spec))
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert set(protocol['ids']) == {'longbench_v2_64k'}
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))


def test_long_lb_gold_is_sha_pinned(tmp_path):
    import hashlib
    from scripts.v21_score import _long_lb_gold
    gold = tmp_path / 'g.json'
    gold.write_text(json.dumps({'longbench_v2/x64': {'answer': 'A'}}))
    protocol = dict(ids={'longbench_v2_64k': ['longbench_v2/x64']},
                    extra_gold_sha256={'longbench_v2_64k': hashlib.sha256(gold.read_bytes()).hexdigest()})
    assert _long_lb_gold(protocol, {'longbench_v2_64k': gold}, ['longbench_v2_64k'])['longbench_v2_64k']
    protocol['extra_gold_sha256']['longbench_v2_64k'] = 'c' * 64
    with pytest.raises(ValueError):
        _long_lb_gold(protocol, {'longbench_v2_64k': gold}, ['longbench_v2_64k'])


def test_g75_port_arms_freeze_and_validate(tmp_path):
    arms = dict(ARMS, G75L0_c64=dict(kind='g75_c64', local_fraction=0.0),
                G75L15_c64=dict(kind='g75_c64', local_fraction=0.15),
                G75L30_c64=dict(kind='g75_c64', local_fraction=0.3))
    sp, op, bp, pool = _setup(tmp_path, arms)
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    c = protocol['arm_contracts']
    assert c['G75L0_c64'] == dict(kind='v27_g75', parent_v20_arm='D_matched', local_fraction=0.0,
                                  scope='GLOBAL_ONLY_NATIVE_LOCAL')
    assert c['G75L30_c64']['scope'] == 'ALL_NATIVE_LEGAL'
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))
    (tmp_path / 'bad').mkdir()
    sp2, op2, bp2, pool2 = _setup(tmp_path / 'bad', dict(ARMS, G=dict(kind='g75_c64', local_fraction=0.5)))
    with pytest.raises(ValueError):
        freeze_v27(sp2, op2, bp2, pool2, tmp_path / 'bad' / 'out')


def test_fa4_piecewise_panel_freezes_and_validates(tmp_path):
    fa4 = dict(consumer64=2, memory_caps='long', fa4_consumer=True, route_pipeline=True)
    arms = dict(ARMS, D_fa4=dict(kind='dense_fa4'), D_fa4_allkept=dict(kind='dense_fa4_allkept'),
                G75L0_fa4=dict(kind='g75_c64', local_fraction=0.0, consumer='fa4'),
                G75L30_fa4=dict(kind='g75_c64', local_fraction=0.3, consumer='fa4'),
                M1_fa4=dict(kind='method', parent='M1_R1_A8_current_output', extra=fa4))
    sp, op, bp, pool = _setup(tmp_path, arms)
    spec = json.loads(sp.read_text())
    spec['substrate'] = 'piecewise_v1'
    sp.write_text(json.dumps(spec))
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    c = protocol['arm_contracts']
    assert c['D_fa4']['control'] == 'D_fa4' and c['D_fa4_allkept']['control'] == 'D_fa4_allkept'
    assert c['G75L0_fa4']['consumer'] == 'fa4' and c['G75L30_fa4']['scope'] == 'ALL_NATIVE_LEGAL'
    assert c['M1_fa4']['fa4_consumer'] is True
    assert protocol['substrate'] == 'piecewise_v1' and protocol['timing_eligible'] is True
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))
    # the substrate is part of the protocol identity
    (tmp_path / 'eager').mkdir()
    sp2, op2, bp2, pool2 = _setup(tmp_path / 'eager', arms)
    assert freeze_v27(sp2, op2, bp2, pool2, tmp_path / 'eager' / 'out')['protocol_id'] != protocol['protocol_id']


def test_unknown_substrate_and_g75_consumer_rejected(tmp_path):
    sp, op, bp, pool = _setup(tmp_path, dict(ARMS, G=dict(kind='g75_c64', local_fraction=0.0, consumer='hopper')))
    with pytest.raises(ValueError):
        freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    (tmp_path / 's').mkdir()
    sp2, op2, bp2, pool2 = _setup(tmp_path / 's', ARMS)
    spec = json.loads(sp2.read_text())
    spec['substrate'] = 'full_graph'
    sp2.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        freeze_v27(sp2, op2, bp2, pool2, tmp_path / 's' / 'out')


@pytest.mark.parametrize('config,mode', [
    (dict(v20_scope='GLOBAL_ONLY_NATIVE_LOCAL'), 'graph'),
    (dict(v20_scope='ALL_NATIVE_LEGAL'), 'eager'),
    (dict(parent_config=dict(v20_scope='GLOBAL_ONLY_NATIVE_LOCAL')), 'graph'),
    (dict(parent_config=dict(v20_scope='ALL_NATIVE_LEGAL')), 'eager'),
    (dict(condition='native_dense'), 'graph'),
])
def test_substrate_local_mode_follows_attention_scope(config, mode):
    from experiments.numerical_qk_reuse.v27_substrate import local_mode_for
    assert local_mode_for(config) == mode


def test_single_host_panel(tmp_path):
    sp, op, bp, pool = _setup(tmp_path, ARMS, warm=False)
    spec = json.loads(sp.read_text())
    spec['hosts'] = ['hB']
    sp.write_text(json.dumps(spec))
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert {a['host'] for a in protocol['block_assignments'].values()} == {'hB'}
    assert {e['gpu_uuid'] for e in protocol['schedule']} == {'GPU-B'}
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))
    (tmp_path / 'x').mkdir()
    sp2, op2, bp2, pool2 = _setup(tmp_path / 'x', ARMS)
    spec2 = json.loads(sp2.read_text())
    spec2['hosts'] = ['hC']
    sp2.write_text(json.dumps(spec2))
    with pytest.raises(ValueError):
        freeze_v27(sp2, op2, bp2, pool2, tmp_path / 'x' / 'out')


def test_compact_sliding_cache_releases_the_prompt_buffer_and_keeps_values():
    import torch
    from types import SimpleNamespace
    from experiments.numerical_qk_reuse.v27_substrate import compact_sliding_cache
    full = torch.randn(1, 8, 5000, 16)
    sliding = SimpleNamespace(is_sliding=True, keys=full[:, :, -1023:, :], values=full[:, :, -1023:, :].clone()[..., ::1])
    dense = SimpleNamespace(is_sliding=False, keys=full[:, :, :100, :], values=full[:, :, :100, :])
    cache = SimpleNamespace(layers=[sliding, dense])
    before = sliding.keys.clone()
    assert compact_sliding_cache(cache) == 1          # the slice view is copied; the clone already was compact
    assert sliding.keys.is_contiguous() and torch.equal(sliding.keys, before)
    assert sliding.keys.untyped_storage().nbytes() == before.numel() * 4    # no longer holds the 5000-token buffer
    assert not dense.keys.is_contiguous()            # full-attention layers are left alone


def test_compiled_sampler_state_is_mirrored_to_the_current_sampler():
    from types import SimpleNamespace
    from experiments.numerical_qk_reuse.v27_substrate import mirrored_accept
    first, current = SimpleNamespace(accepted_token_mask=None), SimpleNamespace(accepted_token_mask=None)

    def compiled_accept(step):               # bound to the first sampler, as the official compiled path is
        first.accepted_token_mask = ('mask', step)
        return step
    accept = mirrored_accept(first, current, compiled_accept)
    assert accept(3) == 3 and current.accepted_token_mask == ('mask', 3)
    assert accept(4) == 4 and current.accepted_token_mask == ('mask', 4)
