from copy import deepcopy
import json

import pytest
from experiments import diffusion_gemma_jl_remaining50 as study


def test_score_blind_exact_complement_and_duplicate_rejection():
    pool = study.read(study.LB/'setup.json')['final']
    old = study.read(study.PREVIOUS/'setup.json')['final']
    selected = study.complement(pool, old)
    assert len(selected) == 50
    assert {r['id'] for r in selected}.isdisjoint(r['id'] for r in old)
    altered = list(reversed(deepcopy(pool)))
    for i, row in enumerate(altered):
        row.update(score=i, prediction=str(i))
    assert [r['id'] for r in selected] == [r['id'] for r in study.complement(altered, old)]
    bad = deepcopy(pool); bad[0]['id'] = bad[1]['id']
    with pytest.raises(ValueError, match='Duplicate'):
        study.complement(bad, old)


def test_prepare_preserves_prompts_settings_and_disjointness(tmp_path):
    setup = study.prepare(tmp_path)
    original = {r['id']: r for r in study.read(study.LB/'setup.json')['final']}
    assert setup['targets'] == [.5] and len(setup['conditions']) == 6
    assert setup['final_generation_slots'] == 100 and setup['reused_baseline_slots'] == 200
    assert all(r == original[r['id']] for r in setup['final'])
    assert not setup['calibration'] and not setup['development']
    assert sum(setup['selection']['domains'].values()) == 50
    assert setup == study.prepare(tmp_path)


def test_thresholds_are_exact_transfers_not_new_fits(tmp_path):
    setup = study.prepare(tmp_path); contract = dict(fingerprint='test')
    for name in study.NEW:
        src = study.PREVIOUS/'policies/longbench_v2'/f'{name}_s50.json'
        p = study.read(src)
        p.update(fingerprint='test', imported_policy=str(src), sources={str(src): study.sha(src.read_bytes())})
        study.audit_policy(tmp_path, p, setup, contract)
        p['policy']['global']['log_threshold'] += .001
        with pytest.raises(ValueError, match='threshold or operator changed'):
            study.audit_policy(tmp_path, p, setup, contract)


def test_complete_predecessor_gate_rejects_stale_proof(tmp_path):
    a = dict(complete=True, completed=640, expected=640, missing=[], violations=[], artifacts={})
    (tmp_path/'audit.json').write_text(json.dumps(a))
    v = dict(passed=True, completed=640, inference_performed=False, audit_sha256=study.sha((tmp_path/'audit.json').read_bytes()))
    (tmp_path/'regeneration_verification.json').write_text(json.dumps(v))
    assert len(study.require_complete(tmp_path, 640)) == 2
    a['missing'] = ['one']; (tmp_path/'audit.json').write_text(json.dumps(a))
    with pytest.raises(ValueError):
        study.require_complete(tmp_path, 640)


def test_accounting_separates100_new_from200_reused(tmp_path):
    for label in study.CONDITIONS:
        stage = 'dense' if label == 'dense' else 'final'
        folder = tmp_path/stage/label/'shards'; folder.mkdir(parents=True)
        for i in range(50):
            (folder/f'{i}.json').write_text(json.dumps({'imported_source': {'path':'old'}} if label in study.BASELINE_LABELS else {}))
    a = study.accounting(tmp_path)
    assert a['completed_new_generations'] == 100 and a['completed_reused'] == 200
    assert a['new_calibration_or_smoke_calls'] == 0


def test_complete300_report_raw_roundtrip_and_missing_detection(tmp_path, monkeypatch):
    report = study.backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup = study.prepare(tmp_path)
    monkeypatch.setattr(study, 'prepare', lambda root: setup)
    monkeypatch.setattr(study, 'execution', lambda root: dict(fingerprint='test', sources={}))
    monkeypatch.setattr(study, 'audit_policy', lambda *args: None)
    monkeypatch.setattr(study, 'smoke_audit', lambda *args: {})
    monkeypatch.setattr(report.common, 'diagnostic_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report.common, 'audit_matrices', lambda *args: None)
    monkeypatch.setattr(report.common, 'shared_seed_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report, 'selected_sources', lambda *args: {})
    monkeypatch.setattr(study, 'BASE_PLOTS', lambda *args: [])
    for name in ('shared_state_index', 'shared_diagnostics_index'):
        (tmp_path/f'{name}.json').write_text('[]')
    raw = tmp_path/'raw.json'; raw.write_text('{}')
    monkeypatch.setattr(report, 'shard_path', lambda *args: raw)
    def output(adapter, root, row, stage, label, name, cfg, thresholds, contract):
        assert adapter is None
        records = []
        for i in range(2):
            r = dict.fromkeys(FIELDS, 0)
            r.update(layer=i*5, step=0, head=0, attention_type=('local','global')[i], probe='execution',
                eligible=10*(i+1), skipped=0 if label == 'dense' else 5*(i+1), rows=2,
                mass_sum=2 if label == 'dense' else 1.5, dense_sq=2, error_sq=0 if label == 'dense' else .1)
            records.append(r)
        return dict(prediction='A', completion_tokens=[1,2], records=records, backend='synthetic',
            records_source=dict(path=str(raw), sha256=study.sha(raw.read_bytes())))
    monkeypatch.setattr(report, 'cached', output)
    def pair(row, out, dense):
        return dict(id=row['id'], benchmark=row['benchmark'], task=row['task'], calibration=False,
            accuracy=1., dense_accuracy=1., matching=2, compared=3, exact_match=False,
            termination_reason='length', unparsed_answer=False, output_length=2,
            aggregates={k: aggregate([r for r in out['records'] if k == 'overall' or r['attention_type'] == k])
                for k in ('overall','local','global')})
    monkeypatch.setattr(report, 'nemo_pairs', lambda row, outs:
        ([dict(condition=n, **pair(row,o,outs['dense'])) for n,o in outs.items()], []))
    (tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name = label.rsplit('_s',1)[0] if label != 'dense' else label
        c = dict(fingerprint='test', name=name, target=0. if name == 'dense' else .5,
            config={} if name == 'dense' else study.CONFIGS[name], sources={},
            thresholds={'longbench_v2': None if name == 'dense' else {k:dict(log_threshold=-2.) for k in ('local','global')}})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    audit = study.regenerate(tmp_path)
    assert audit['complete'] and audit['completed'] == audit['expected'] == 300
    assert len(study.read(tmp_path/'summary.json')) == 6
    assert len(study.read(tmp_path/'cohort_comparison.json')) == 18
    text = (tmp_path/'report.md').read_text()
    assert '300 audited' in text and 'No recalibration' in text and 'remaining50' in text
    assert audit == study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    def missing(*args, **kwargs):
        if args[2]['id'] == setup['final'][0]['id'] and args[4] == 'jl_gaussian_r32_s50':
            raise FileNotFoundError('synthetic missing')
        return output(*args, **kwargs)
    monkeypatch.setattr(report, 'cached', missing)
    assert not study.regenerate(tmp_path)['complete']
