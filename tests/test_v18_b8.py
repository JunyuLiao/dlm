"""CPU qualification for the opt-in B8 risk-export removal."""
from types import SimpleNamespace
import json

import pytest
import torch


def router(mode):
    from experiments.value_direction_hopper.cvm import CVMRouter

    r = CVMRouter.__new__(CVMRouter)
    r.mode, r.period, r.canvas, r.step, r.epoch = mode, 8, 0, 0, 0
    r.layers = {}
    r.counts = dict(calls=0, anchors=0, ordinary=0, fallback_native=0, restored_tiles=0,
                    planner_calls=0, kept_prefix_tiles=0, prefix_tiles=0)
    r.kernel = SimpleNamespace(export=False, protect_tile=-1, last_masks=None, last_export=None)
    seen = []

    def fresh(*_args, **_kwargs):
        seen.append(r.kernel.export)
        r.kernel.last_masks = (torch.zeros(1, 1, 1, 5, dtype=torch.bool),
                               torch.ones(1, 1, 1, 5, dtype=torch.bool))
        r.kernel.last_export = torch.zeros(1, 1, 5, 128) if r.kernel.export else None
        return 'fresh-output'

    r.fresh = fresh
    r._native = lambda *_args, **_kwargs: 'native-output'
    r._support = SimpleNamespace(attention=lambda *_args, **_kwargs: (torch.zeros(1, 1, 128, 2), None,
                                                                      torch.zeros(1, 1, 128, dtype=torch.bool), None))
    r.guard_mode = 'async'
    r.v5_identity = dict(key='test')
    r.support_identity = dict(key='test')
    return r, seen


@pytest.mark.parametrize('mode,export', [('B8P', True), ('B8NR', False), ('CVM', True)])
def test_anchor_storage_and_reset(mode, export):
    r, seen = router(mode)
    module = SimpleNamespace(layer_idx=1, is_sliding=False)
    q = torch.zeros(1, 1, 128, 2)
    k = torch.zeros(1, 1, 385, 2)
    assert r(module, q, k, k, None) == 'fresh-output'
    assert seen == [export] and r.kernel.export is False
    st = r.layers[1]
    assert ('log_rho' in st) is export
    assert r.counters()['metadata_bytes'] == (5 * 128 * 4 if export else 0) + 5 * 2
    r.begin_step(1, 0)
    assert not r.layers
    r.close = lambda: None


def test_export_resets_after_anchor_exception():
    r, _ = router('B8NR')
    r.fresh = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('anchor failure'))
    q = torch.zeros(1, 1, 128, 2)
    k = torch.zeros(1, 1, 385, 2)
    with pytest.raises(RuntimeError, match='anchor failure'):
        r(SimpleNamespace(layer_idx=1, is_sliding=False), q, k, k, None)
    assert r.kernel.export is False and r.layers == {}


def test_opt_in_plugin_rejects_other_condition():
    from experiments.value_direction_hopper.cvm import install_no_risk_export

    with pytest.raises(ValueError, match='requires global_B8P'):
        with install_no_risk_export(None, {}, 'global_CVM'):
            pass


def test_ledger_verifier_requires_every_field_and_catches_variant(tmp_path):
    from scripts.v18_b8_runs import verify_ledger
    from scripts.v13_seed_runs import execution_key

    ids = [f'longbench_v2/{i}' for i in range(4)]
    schedule = []
    for i, qid in enumerate(ids):
        for role in ('attempt0', 'warm'):
            for arm in ('B8_P', 'B8_no_risk_export'):
                schedule.append(dict(index=len(schedule), id=qid, seed=17, arm=arm, role=role,
                                     repeat=int(role == 'warm'), cell_id=f'{i}-{arm}'))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(schema='v18_b8_no_risk_export_v1', schedule=schedule, seeds=[17],
                                        arms={'B8_P': {}, 'B8_no_risk_export': {}})))
    ledger = tmp_path / 'ledger.jsonl'
    runs = [dict(event='run', execution_key=execution_key(e), ok=True, completion_token_hash='same',
                 per_canvas_calls=[2, 3], termination='eos', phases={'anchors': 1, 'ordinary': 4,
                 'fallback_native': 0, 'calls': 5}, acceptance={'accepted': True}, **e) for e in schedule]
    ledger.write_text('\n'.join(json.dumps(e) for e in runs))
    assert verify_ledger(protocol, ledger)['equivalent']
    del runs[0]['completion_token_hash']
    ledger.write_text('\n'.join(json.dumps(e) for e in runs))
    assert not verify_ledger(protocol, ledger)['equivalent']


def test_b8_phases_rejects_missing_or_inconsistent_counts():
    from scripts.v18_b8_runs import b8_phases

    good = {'anchors': 2, 'ordinary': 5, 'fallback_native': 1, 'calls': 8}
    assert b8_phases({'counters': good}) == good
    assert b8_phases({'counters': dict(good, calls=7)}) is None
    assert b8_phases({'counters': dict(good, anchors=None)}) is None
