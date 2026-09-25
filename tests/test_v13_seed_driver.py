"""v13 CP0: the multi-seed driver/summarizer is seed-safe (CPU mocks)."""
import json
import math
from pathlib import Path

import pytest

from scripts.v13_seed_runs import (arm_config_hash, cell_id, execution_key, plan_schedule, receipt_path, run_schedule,
                                   validate_resume, warm_acceptance)
from scripts.v13_summarize import aggregate, build_cells

ARM_HASHES = dict(D_native='d' * 64, G3='g' * 64, T_G='t' * 64)


def schedule(ids=('aime26/2', 'aime26/8'), seeds=(17, 29)):
    return plan_schedule('proto', 'rev', ARM_HASHES, list(ids), list(seeds), 7)


def test_1_two_seeds_are_two_cells_with_distinct_attempt0_paths():
    s = schedule()
    first = [e for e in s if e['role'] == 'attempt0' and e['arm'] == 'G3' and e['id'] == 'aime26/2']
    assert sorted(e['seed'] for e in first) == [17, 29]
    assert first[0]['cell_id'] != first[1]['cell_id']
    assert receipt_path(Path('/p'), first[0]) != receipt_path(Path('/p'), first[1])
    assert len({e['cell_id'] for e in s}) == 3 * 2 * 2 and len(s) == 3 * 2 * 2 * 2


def test_2_one_receives_the_scheduled_seed_and_it_reaches_the_records(tmp_path):
    s = schedule(ids=('aime26/2',))
    seen, ledger = [], []

    def execute_one(row, seed, config, entry):
        seen.append(seed)
        return dict(record=dict(ok=True, generation_seed=seed), receipt=dict(seed=seed))
    run_schedule(s, done=set(), execute_one=execute_one, rows={'aime26/2': {}}, configs=dict.fromkeys(ARM_HASHES),
                 ledger_append=ledger.append, private=tmp_path, save_receipt=lambda p, r: None)
    assert set(seen) == {17, 29} and 42 not in seen
    assert all(r['seed'] == r['generation_seed'] for r in ledger)


def test_3_same_seed_repeat_is_timing_repeat_other_seed_is_not():
    s = schedule(ids=('aime26/2',))
    by_cell = {}
    for e in s:
        by_cell.setdefault(e['cell_id'], []).append((e['role'], e['repeat'], e['seed']))
    for roles in by_cell.values():
        assert sorted(r[:2] for r in roles) == [('attempt0', 0), ('warm', 1)]
        assert len({r[2] for r in roles}) == 1
    warm = next(e for e in s if e['role'] == 'warm')
    first_other_seed = next(e for e in s if e['role'] == 'attempt0' and e['arm'] == warm['arm'] and e['seed'] != warm['seed'])
    assert first_other_seed['cell_id'] != warm['cell_id']


def identity():
    return dict(protocol_id='proto', model_revision='rev', arm_hashes=ARM_HASHES, source_hashes={'a': '1'}, private_root='/p')


def test_4_resume_refuses_changed_identity_or_foreign_records():
    s = schedule()
    ok_event = dict(event='start', **identity())
    assert validate_resume([ok_event], identity(), s) == set()
    for field, value in (('model_revision', 'other'), ('arm_hashes', dict(ARM_HASHES, G3='x' * 64)),
                         ('source_hashes', {'a': '2'}), ('private_root', '/q'), ('protocol_id', 'v12')):
        with pytest.raises(RuntimeError):
            validate_resume([dict(ok_event, **{field: value})], identity(), s)
    old_seed42 = dict(event='run', execution_key=f"{cell_id('proto', 'rev', 'g' * 64, 'aime26/2', 42)}:attempt0:0")
    with pytest.raises(RuntimeError):
        validate_resume([ok_event, old_seed42], identity(), s)
    real = dict(event='run', execution_key=execution_key(s[0]))
    assert validate_resume([ok_event, real], identity(), s) == {execution_key(s[0])}
    with pytest.raises(RuntimeError):
        validate_resume([ok_event, real, real], identity(), s)


def golden():
    s = schedule(ids=('aime26/1', 'aime26/2'))
    records, quality = {}, {}
    times = {('G3', 'aime26/1'): 10., ('T_G', 'aime26/1'): 20., ('D_native', 'aime26/1'): 15.,
             ('G3', 'aime26/2'): 30., ('T_G', 'aime26/2'): 30., ('D_native', 'aime26/2'): 30.}
    right = {('G3', 'aime26/1', 17), ('T_G', 'aime26/1', 17), ('T_G', 'aime26/1', 29), ('D_native', 'aime26/2', 29)}
    for e in s:
        base = dict(event='run', execution_key=execution_key(e), ok=True, **{k: e[k] for k in ('arm', 'id', 'seed', 'role', 'cell_id')},
                    completion_token_hash=f"{e['cell_id']}h", per_canvas_calls=[5, 5], termination='eos', triton_misses=0,
                    decoder_calls=10 if e['arm'] == 'G3' else 20, api_wall_s=times[(e['arm'], e['id'])] * (1 if e['seed'] == 17 else 2))
        records[execution_key(e)] = base
        if e['role'] == 'attempt0':
            quality[e['cell_id']] = dict(correct=(e['arm'], e['id'], e['seed']) in right)
    return s, records, quality


def test_5_golden_multi_seed_aggregates():
    s, records, quality = golden()
    cells = build_cells(s, records, quality)
    out = aggregate(cells, ['aime26/1', 'aime26/2'], [17, 29], ['aime26/1'], pairs=[('G3', 'T_G')], boot=200)
    assert out['quality']['G3']['per_seed'] == {'17': 1, '29': 0} and out['quality']['T_G']['combined'] == 2
    p = out['pairs']['G3/T_G']
    # per-question mean log ratio: q1 log(0.5), q2 log(1) -> geo = exp(mean) = sqrt(0.5)
    assert p['geometric_time_ratio'] == pytest.approx(math.sqrt(.5))
    assert p['summed_time_ratio'] == pytest.approx((10 + 20 + 30 + 60) / (20 + 40 + 30 + 60))
    assert p['call_factor'] == pytest.approx(40 / 80)
    assert p['amortized_time_per_call_factor'] == pytest.approx(p['summed_time_ratio'] / p['call_factor'])
    assert p['paired_disagreements'] == dict(both=1, candidate_only=0, reference_only=1, neither=2)
    assert p['quality_diff_mean_per_question'] == pytest.approx(((.5 - 1) + 0) / 2)
    assert set(p['per_seed']) == {'17', '29'} and 'reserved' in p['strata']
    assert set(p['leave_one_question_out']) == {'aime26/1', 'aime26/2'}


def test_6_warm_acceptance_rejects_but_preserves_bad_rows():
    s, records, quality = golden()
    warm = next(k for k, r in records.items() if r['role'] == 'warm' and r['arm'] == 'G3' and r['id'] == 'aime26/1' and r['seed'] == 17)
    records[warm] = dict(records[warm], triton_misses=3)
    other = next(k for k, r in records.items() if r['role'] == 'warm' and r['arm'] == 'T_G' and r['id'] == 'aime26/2')
    records[other] = dict(records[other], completion_token_hash='different')
    cells = build_cells(s, records, quality)
    bad = cells[('G3', 'aime26/1', 17)]
    assert bad['time'] is None and 'new_compilation' in bad['timing_status'] and bad['warm'][0]['triton_misses'] == 3
    assert 'tokens_mismatch' in cells[(records[other]['arm'], 'aime26/2', records[other]['seed'])]['timing_status']
    out = aggregate(cells, ['aime26/1', 'aime26/2'], [17, 29], ['aime26/1'], pairs=[('G3', 'T_G')], boot=50)
    p = out['pairs']['G3/T_G']
    assert p['timing_complete'] is False and 'geometric_time_ratio' not in p and len(p['missing_timings']) == 2
    assert warm_acceptance(None, dict(ok=True, triton_misses=0))['reasons'] == ['attempt0_missing_or_failed']


def test_7_reporting_metadata_comes_from_the_protocol_not_v10_notes():
    import inspect
    import scripts.v13_summarize as summ
    source = inspect.getsource(summ.main)
    assert "protocol['ids']" in source and "protocol['seeds']" in source and 'effective_arm_fields' in source
    for stale in ('4 independent questions', 'seed 42', 'legacy Junyu local mask', 'collect=True'):
        assert stale not in inspect.getsource(summ)
    assert arm_config_hash(dict(a=1, seeds=[17], ids=['x'], fingerprint='f')) == arm_config_hash(dict(a=1, seeds=[42]))


def test_8_config_hash_is_independent_of_checkout_root():
    from scripts.v13_seed_runs import REPO
    a = dict(x=1, source_hashes={REPO + '/experiments/a.py': 'h1', '/site/transformers/g.py': 'h2'})
    b = dict(x=1, source_hashes={'<repo>/experiments/a.py': 'h1', '/site/transformers/g.py': 'h2'})
    assert arm_config_hash(a) == arm_config_hash(b)
    assert arm_config_hash(a) != arm_config_hash(dict(a, source_hashes={REPO + '/experiments/a.py': 'CHANGED', '/site/transformers/g.py': 'h2'}))
