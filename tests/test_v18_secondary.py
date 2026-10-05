"""CPU-only protocol and density-selection gates for secondary v18 arms."""
from collections import Counter
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.v18_protocol import sha
from scripts.v18_evaluate import logical_protocol_digest
from scripts.v18_secondary import (ALLOCATION_SEED, bind_control_config, calibration_schedule,
                                   inherit_block_assignments, initial_policies, mechanism_subset,
                                   prerequisite_gate, secondary_plan, stage_completion,
                                   update_density_history)
from scripts.v18_secondary_coordinate import budget_plan, usage_by_host


def seal(protocol):
    protocol['status'] = 'frozen'
    for field in ('ids', 'seeds', 'arm_hashes', 'manifest_sha256', 'calibration_sha256',
                  'binary_hashes', 'model_tensor_identity_sha256', 'assigned_hosts',
                  'bridge_receipt_sha256', 'initial_prefix_blocks'):
        protocol.setdefault(field, None)
    protocol['logical_protocol_sha256'] = logical_protocol_digest(protocol)
    return protocol


def panel(per_task, prefix):
    return [dict(id=f'{prefix}/{task}/{index}', task=f'task{task:02d}',
                 prompt=f'{prefix} prompt {task} {index}',
                 prompt_hash=sha(f'{prefix} prompt {task} {index}'),
                 generation_budget=128)
            for task in range(13) for index in range(per_task)]


def test_secondary_order_and_exact_execution_counts():
    ruler, cal = panel(10, 'eval'), panel(2, 'cal')
    ids = [r['id'] for r in ruler]
    warm = ids[::5]
    draft = dict(dataset='ruler4k', ids=ids, seeds=[101, 202, 303], warm_ids=warm)
    plan = secondary_plan(ruler, cal, draft)
    assert plan['stage_order'][-2:] == ['ruler_secondary70', 'ruler_secondary_allocation']
    assert plan['core70']['planned_executions'] == 936
    assert Counter(e['role'] for e in plan['core70']['schedule']) == {'attempt0': 780, 'warm': 156}
    assert len(plan['allocation']['ids']) == 26
    assert plan['allocation']['planned_executions'] == 156
    assert {e['role'] for e in plan['allocation']['schedule']} == {'attempt0'}
    assert plan['allocation']['runnable'] is False
    assert plan['allocation']['runtime_binding_available'] is True
    assert plan['allocation']['arms']['T60_uniform']['allocation_rng_seed'] == ALLOCATION_SEED
    assert plan == secondary_plan(ruler, cal, draft)
    assert Counter(r['task'] for r in ruler if r['id'] in mechanism_subset(ruler)) == {
        f'task{task:02d}': 2 for task in range(13)}


def test_independent_calibration_and_panel_are_required():
    ruler, cal = panel(10, 'eval'), panel(2, 'cal')
    draft = dict(dataset='ruler4k', ids=[r['id'] for r in ruler],
                 seeds=[101, 202, 303], warm_ids=[r['id'] for r in ruler][::5])
    with pytest.raises(ValueError, match='independent'):
        secondary_plan(ruler, [dict(cal[0], id=ruler[0]['id'])] + cal[1:], draft)
    with pytest.raises(ValueError, match='exact frozen'):
        secondary_plan(ruler, cal, dict(draft, ids=draft['ids'][::-1]))


def test_initial_70_is_only_bracket_and_controls_start_from_primary_t60():
    old = {'policies': {'unweighted_s50': {'local': {'log_threshold': -3.}, 'global': {'log_threshold': -4.}},
           'unweighted_s70': {'local': {'log_threshold': -2.}, 'global': {'log_threshold': -3.}},
           'T_s50': {'local': {'log_threshold': -1.}, 'global': {'log_threshold': -2.}},
           'T_s70': {'local': {'log_threshold': -.5}, 'global': {'log_threshold': -1.5}}}}
    primary = {'T60': dict(attained=True, policy={'local': {'log_threshold': -.7},
                                                  'global': {'log_threshold': -2.8}})}
    proposal = initial_policies(old, primary)
    assert proposal['U70']['local']['log_threshold'] == -2.
    assert proposal['T70']['global']['log_threshold'] == -1.5
    assert proposal['T60_shuffled'] == proposal['T60_uniform'] == primary['T60']['policy']
    calibration = calibration_schedule([r['id'] for r in panel(2, 'cal')], proposal, 0)
    assert calibration['planned_executions'] == 104
    assert Counter(e['arm'] for e in calibration['schedule']) == {
        arm: 26 for arm in proposal}
    assert {e['seed'] for e in calibration['schedule']} == {42}
    assert {e['role'] for e in calibration['schedule']} == {'attempt0'}
    with pytest.raises(ValueError, match='ceiling'):
        calibration_schedule(calibration['ids'], proposal, 5)
    with pytest.raises(ValueError, match='attained'):
        initial_policies(old, {'T60': dict(primary['T60'], attained=False)})


def test_density_only_selection_and_five_pair_ceiling():
    policy = {'local': {'log_threshold': -1.}, 'global': {'log_threshold': -2.}}
    history = []
    for i in range(4):
        state = update_density_history('T60_uniform', history, policy,
                                       {'whole': .4, 'local': .4, 'global': .4})
        assert not state['done']
        history, policy = state['history'], state['next_policy']
    final = update_density_history('T60_uniform', history, policy,
                                   {'whole': .59, 'local': .60, 'global': .61})
    assert final['done'] and final['attained'] and len(final['history']) == 5
    assert final['next_policy'] is None and final['best_point'] == 4
    with pytest.raises(ValueError, match='ceiling'):
        update_density_history('T60_uniform', final['history'], policy,
                               {'whole': .6, 'local': .6, 'global': .6})
    with pytest.raises(ValueError, match='all density strata'):
        update_density_history('U70', [], policy, {'whole': .7, 'score': 1.})


def test_named_control_binding_and_normal_arm_defaults(monkeypatch):
    from experiments.value_direction_hopper import frontier_scope as scope
    captured = []

    class Model:
        def named_modules(self):
            return []

    class Binding:
        def __init__(self):
            self.runtime = SimpleNamespace(attention_override=None)
        def close(self):
            pass

    class Router:
        def __init__(self, *_args, **_kwargs):
            self.support_geometry = 'native_legal'
            self.calls = 0
            self.cache = SimpleNamespace(projected_tokens=0, reused_tokens=0, close=lambda: None)

    def fake_state(method, router, **kwargs):
        captured.append((method, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(scope, '_install_dense', lambda adapter: Binding())
    monkeypatch.setattr(scope, 'Attention', Router)
    monkeypatch.setattr(scope, 'State', fake_state)
    adapter = SimpleNamespace(model=Model(), is_blasst_attention_module=lambda *_: False)
    base = dict(frontier_arm='T60', method='T', target=60, diagnostic=False,
                policy={'local': {'log_threshold': -1.}, 'global': {'log_threshold': -2.}},
                library='kernel', torch_library='bridge', m_ref=1., beta=3., gamma=.5)
    with scope.install(adapter, base, 'native_legal_all_layers'):
        pass
    assert 'allocation' not in captured[-1][1] and 'seed' not in captured[-1][1]
    for arm, allocation in (('T60_shuffled', 'shuffle'), ('T60_uniform', 'uniform')):
        config = dict(base, frontier_arm=arm, allocation=allocation, allocation_seed=ALLOCATION_SEED)
        with scope.install(adapter, config, 'native_legal_all_layers'):
            pass
        assert captured[-1][0] == 'T'
        assert captured[-1][1]['allocation'] == allocation
        assert captured[-1][1]['seed'] == ALLOCATION_SEED
        with pytest.raises(ValueError, match='independent RNG seed'):
            with scope.install(adapter, dict(config, allocation_seed=42), 'native_legal_all_layers'):
                pass


def test_existing_causal_t_allocation_semantics():
    from experiments.value_direction_hopper.query_adaptive import State
    canvas = torch.zeros((1, 256), dtype=torch.long)
    original = torch.zeros((1, 256, 2))
    flipped = original.clone()
    flipped[:, :64, 1] = 2.
    flipped[:, :64, 0] = -2.
    for allocation in ('shuffle', 'uniform'):
        state = State('T', None, m_ref=1., diagnostics=False, fast_t=True,
                      allocation=allocation, seed=ALLOCATION_SEED)
        state.begin(48, canvas)
        assert state.used_weights is None  # no completed prior step
        state.observe_logits(original, None, 48)
        state.begin(47, canvas)
        state.observe_logits(flipped, None, 47)
        state.begin(46, canvas)
        values = state.used_weights
        assert values.shape == (1, 256)
        if allocation == 'shuffle':
            assert sorted(values[0, :128].tolist()) == sorted(([2.5] * 64) + ([1.] * 64))
            assert torch.all(values[0, 128:] == 1.)
        else:
            assert torch.all(values == 1.375)  # entire canvas mean, not tile mean


def test_control_config_binds_fresh_scope_and_fingerprint():
    from experiments.value_direction_hopper import frontier_scope
    from pathlib import Path
    scope_path = Path(frontier_scope.__file__).resolve()
    base = dict(condition='native_legal_all_layers', frontier_arm='T60', method='T', target=60,
                fast_t=True, source_hashes={str(scope_path): sha(scope_path.read_bytes())},
                policy={'local': {'log_threshold': -1.}, 'global': {'log_threshold': -2.}})
    bound = bind_control_config(base, 'T60_uniform', base['policy'])
    assert bound['allocation_seed'] == ALLOCATION_SEED and bound['allocation'] == 'uniform'
    assert bound['fingerprint'] != base.get('fingerprint')
    assert base['frontier_arm'] == 'T60'
    with pytest.raises(ValueError, match='fresh control-capable'):
        bind_control_config(dict(base, source_hashes={}), 'T60_uniform', base['policy'])


def test_secondary_blocks_inherit_primary_host_by_question_seed():
    primary = dict(schedule=[dict(id='a', seed=101, block=0), dict(id='b', seed=101, block=1),
                             dict(id='a', seed=101, block=0)],
                   block_assignments={'0': dict(host='old', gpu_uuid='GPU-A'),
                                      '1': dict(host='new', gpu_uuid='GPU-B')})
    # Secondary block order is reversed, but host/GPU pairing must not move.
    secondary = [dict(id='b', seed=101, block=0), dict(id='a', seed=101, block=1),
                 dict(id='a', seed=101, block=1)]
    assert inherit_block_assignments(primary, secondary) == {
        '0': dict(host='new', gpu_uuid='GPU-B'),
        '1': dict(host='old', gpu_uuid='GPU-A')}
    with pytest.raises(ValueError, match='absent from primary'):
        inherit_block_assignments(primary, [dict(id='c', seed=101, block=0)])


def test_secondary_gate_lists_missing_prerequisite_executions(tmp_path):
    def protocol(stage):
        row = dict(arm='T60', id='q', seed=101, role='attempt0', repeat=0,
                   cell_id=stage, block=0)
        return seal(dict(stage=stage, protocol_id=stage, schedule=[row],
                         block_assignments={'0': dict(host='old', gpu_uuid='GPU-A')}))
    ruler, aime, core = (protocol(stage) for stage in
                         ('ruler4k_primary', 'aime26_primary', 'ruler_secondary70'))
    def ledger(name, item):
        path = tmp_path / (name + '.jsonl')
        path.write_text(json.dumps(dict(item['schedule'][0],
                                        event='run', execution_key=f"{name}:attempt0:0",
                                        host='old', gpu_uuid='GPU-A')) + '\n')
        return [path]
    ruler_ledger, aime_ledger, core_ledger = (ledger(name, item) for name, item in
                                              (('ruler4k_primary', ruler), ('aime26_primary', aime),
                                               ('ruler_secondary70', core)))
    assert stage_completion(ruler, ruler_ledger)['complete']
    allocation = dict(stage='ruler_secondary_allocation',
                      prerequisite_protocol_ids={'ruler4k_primary': ruler['protocol_id'],
                                                 'aime26_primary': aime['protocol_id'],
                                                 'ruler_secondary70': core['protocol_id']})
    gate = prerequisite_gate(allocation, (ruler, ruler_ledger), (aime, aime_ledger), (core, []))
    assert not gate['ready'] and len(gate['missing']) == 1
    assert gate['missing'][0]['stage'] == 'ruler_secondary70'
    assert prerequisite_gate(allocation, (ruler, ruler_ledger), (aime, aime_ledger),
                             (core, core_ledger))['ready']
    with pytest.raises(ValueError, match='foreign or duplicate'):
        stage_completion(ruler, ruler_ledger + ruler_ledger)


def test_secondary_budget_charges_primary_and_calibration_before_launch(tmp_path):
    def stage(name):
        row = dict(index=0, block=0, arm='T60', id='q', seed=101, role='attempt0',
                   repeat=0, cell_id=name)
        return seal(dict(stage=name, protocol_id=name, schedule=[row],
                         block_assignments={'0': dict(host='old', gpu_uuid='GPU-A')}))
    ruler, aime, secondary = (stage(x) for x in ('ruler4k_primary', 'aime26_primary',
                                                 'ruler_secondary70'))
    secondary['prerequisite_protocol_ids'] = {'ruler4k_primary': ruler['protocol_id'],
                                              'aime26_primary': aime['protocol_id']}
    secondary['assigned_hosts'] = [dict(host='old', gpu_uuid='GPU-A')]
    def write_run(name, item):
        path = tmp_path / (name + '.jsonl')
        event = dict(item['schedule'][0], event='run', execution_key=f'{name}:attempt0:0',
                     host='old', gpu_uuid='GPU-A')
        path.write_text('\n'.join((json.dumps(dict(event='start', when=10.)),
                                    json.dumps(event),
                                    json.dumps(dict(event='worker_end', when=15., host='old',
                                                    gpu_process_seconds=6.)))))
        return path
    ruler_path = write_run('ruler4k_primary', ruler)
    aime_path = write_run('aime26_primary', aime)
    secondary_path = tmp_path / 'secondary.jsonl'
    budget = dict(gpu_cap_s_by_host={'old': 100.}, request_cap_by_host={'old': 10},
                  baseline_gpu_s_by_host={'old': 20.}, baseline_requests_by_host={'old': 1},
                  baseline_requests=1, request_cap=10, deadline_epoch=1000., scoring_minutes_reserved=1,
                  block_guard_s=5.)
    prior = {'ruler4k_primary': (ruler, [ruler_path]), 'aime26_primary': (aime, [aime_path])}
    summaries = {}
    for item in (ruler, aime):
        path = tmp_path / (item['stage'] + '_summary.json')
        path.write_text(json.dumps(dict(schema='v18_redacted_summary_v1',
                                        protocol_id=item['protocol_id'], stage=item['stage'],
                                        complete=True, planned_executions=1, recorded_executions=1)))
        summaries[item['stage']] = path
    result = budget_plan(secondary, prior, [secondary_path],
                         {'old': [ruler_path, aime_path, secondary_path]}, budget, 'old', now=100.,
                         primary_summaries=summaries, gpu_idle=True)
    assert result['gate']['ready'] and result['allowed']
    assert result['gpu_used_s_by_host']['old'] == 32.
    assert result['requests_used_by_host']['old'] == 3
    assert result['next_block_requests'] == 1
    capped = dict(budget, request_cap_by_host={'old': 3}, request_cap=3)
    assert not budget_plan(secondary, prior, [secondary_path],
                           {'old': [ruler_path, aime_path, secondary_path]}, capped,
                           'old', now=100., primary_summaries=summaries,
                           gpu_idle=True)['allowed']
