"""Continuation scope, predecessor gate, policy identity and raw report proof."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import diffusion_gemma_jl_projection_fullbudget as study


def test_exact_scope_full_budgets_and_unchanged_sample_ids(tmp_path):
    setup = study.prepare(tmp_path)
    prior = study.fb.prepare(study.REFERENCE)
    assert setup['final'] == prior['final'] and setup['calibration'] == prior['calibration']
    assert setup['conditions'] == study.parent.CONDITIONS and len(setup['conditions']) == 13
    assert len(study.NEW) == 2
    assert all(c['family'] == 'gaussian' and c['rank'] == 32 for c in study.NEW.values())
    assert setup['projection_seed'] == 1729 and setup['monitoring_interval_seconds'] == 900
    for benchmark, budget in (('aime26', 2048), ('longbench_v2', 4096)):
        assert all(r['generation_budget'] == budget for r in study.fb.calibration_rows(setup, benchmark))
    assert study.prepare(tmp_path) == setup


def test_reference_generation_is_not_enough_without_independent_audit(tmp_path):
    with pytest.raises(FileNotFoundError):
        study.require_reference(tmp_path)
    audit = dict(complete=True, completed=720, expected=720, missing=[], violations=[], artifacts={})
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    proof = dict(passed=True, completed=720, inference_performed=False,
        audit_sha256=study.sha((tmp_path/'audit.json').read_bytes()))
    (tmp_path/'regeneration_verification.json').write_text(json.dumps(proof))
    (tmp_path/'terminal.json').write_text(json.dumps(dict(complete=True, completed=720)))
    (tmp_path/'accepted_round2').mkdir()
    exit_path = tmp_path/'accepted_round2/supervisor_terminal.json'
    exit_path.write_text(json.dumps(dict(exit_code=0)))
    assert len(study.require_reference(tmp_path)) == 4
    exit_path.write_text(json.dumps(dict(exit_code=1)))
    with pytest.raises(ValueError, match='must finish first'):
        study.require_reference(tmp_path)
    exit_path.write_text(json.dumps(dict(exit_code=0)))
    (tmp_path/'audit.json').write_text(json.dumps(dict(audit, completed=719)))
    with pytest.raises(ValueError):
        study.require_reference(tmp_path)


def test_projected_auditor_preserves_exact_config_and_strict_tolerance(monkeypatch):
    seen = []
    old_config = deepcopy(study.fb.CONFIG)
    def audit(root, policy, setup, contract):
        seen.append(deepcopy(study.fb.CONFIG))
        assert not study.fb.within(study.accepted.EXPECTED, .75)
    monkeypatch.setattr(study.fb, 'audit_policy', audit)
    for name, cfg in study.NEW.items():
        p = dict(fingerprint='test', heldout_used=False, name=name, config=cfg)
        study.audit_policy(None, p, {}, dict(fingerprint='test'))
        with pytest.raises(ValueError):
            study.audit_policy(None, dict(p, config=dict(cfg, rank=16)), {}, dict(fingerprint='test'))
        with pytest.raises(ValueError):
            study.audit_policy(None, dict(p, heldout_used=True), {}, dict(fingerprint='test'))
    assert seen == list(study.NEW.values())
    assert study.fb.CONFIG == old_config


def test_short_budget_projected_alias_cannot_enter_full_budget_calibration(tmp_path):
    setup = study.prepare(tmp_path)
    row = study.fb.calibration_rows(setup, 'aime26')[0]
    name = 'jl_gaussian_r32'
    old = study.read(study.parent.OLD/'policies/aime26/jl_gaussian_r32_s50.json')
    point = next(x for x in old['trace'] if x['iteration'] == old['selected_round'])
    source = Path(point['sources'][row['id']]['path'])
    fp = study.read(study.parent.OLD/'execution_contract.json')['fingerprint']
    with pytest.raises(ValueError, match='generation_budget'):
        study.fb.alias(tmp_path, row, 'calibration', 'short', name, study.NEW[name], old['policy'],
            dict(fingerprint='test'), source, fp)
    assert not (tmp_path/'calibration').exists()


def test_rank32_warm_proposals_are_existing_calibration_only_sources():
    for name, cfg in study.NEW.items():
        old, arrays, sources = study.warm_start(name, 'aime26', .5)
        assert old['config'] == cfg and old['heldout_used'] is False
        assert set(arrays) == {'local', 'global'}
        assert all(len(v) and (v[1:] >= v[:-1]).all() for v in arrays.values())
        assert sources and all('/final/' not in p for p in sources)


def test_shared_diagnostic_reuse_preserves_metrics_and_checks_policy(tmp_path, monkeypatch):
    previous = tmp_path/'reference'; previous.mkdir()
    root = tmp_path/'next'; root.mkdir()
    policy = dict(benchmark='aime26', config={'family': 'identity'},
        policy={k: dict(log_threshold=-2.) for k in ('local', 'global')})
    old_path = previous/'policies/aime26/full_centered_s50.json'
    new_path = root/'policies/aime26/full_centered_s50.json'
    for path in (old_path, new_path):
        path.parent.mkdir(parents=True); path.write_text(json.dumps(policy))
    src = previous/'state_full_centered_s50.json'
    original = dict(identity=dict(fingerprint='old', policy_path=str(old_path),
        policy_sha256=study.sha(old_path.read_bytes()), diagnostic_sources={},
        source={'path': 'unchanged_qkv'}, name='full_centered', target=.5),
        metrics={'error_sq': 1.234, 'retained_mass_sum': 4.567})
    src.write_text(json.dumps(original))
    (previous/'shared_diagnostics_index.json').write_text(json.dumps([
        dict(path=str(src), sha256=study.sha(src.read_bytes()))]))
    monkeypatch.setattr(study, 'REFERENCE', previous)
    study.reuse_shared_diagnostics(root, {'fingerprint': 'new'})
    copied = study.read(root/'shared_diagnostics'/src.name)
    assert copied['metrics'] == original['metrics']
    expected = dict(original['identity'], fingerprint='new', policy_path=str(new_path),
        policy_sha256=study.sha(new_path.read_bytes()))
    assert copied['identity'] == expected
    assert study.read(src) == original
    new_path.write_text(json.dumps(dict(policy, config={'family': 'gaussian', 'rank': 32})))
    with pytest.raises(ValueError, match='changed operator'):
        study.reuse_shared_diagnostics(root, {'fingerprint': 'new'})


def test_full1040_raw_report_roundtrip_and_missing_detection(tmp_path, monkeypatch):
    report = study.backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup = study.prepare(tmp_path)
    monkeypatch.setattr(study, 'prepare', lambda root: setup)
    monkeypatch.setattr(study, 'execution', lambda root: dict(fingerprint='test', sources={}))
    monkeypatch.setattr(study, 'audit_policy', lambda *args: None)
    monkeypatch.setattr(study.fb, 'inherited_smoke', lambda *args: {})
    monkeypatch.setattr(report.common, 'diagnostic_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report.common, 'audit_matrices', lambda *args: None)
    monkeypatch.setattr(report.common, 'shared_seed_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report, 'selected_sources', lambda *args: {})
    monkeypatch.setattr(report, 'plots', lambda *args: [])
    for name in ('shared_state_index', 'shared_diagnostics_index'):
        (tmp_path/f'{name}.json').write_text('[]')
    raw = tmp_path/'raw.json'; raw.write_text('{}')
    monkeypatch.setattr(report, 'shard_path', lambda *args: raw)
    def output(adapter, root, row, stage, label, name, cfg, thresholds, contract):
        assert adapter is None
        records = []
        for index in range(2):
            r = dict.fromkeys(FIELDS, 0)
            r.update(layer=index*5, step=0, head=0, attention_type=('local', 'global')[index],
                probe='execution', eligible=10*(index+1), skipped=0 if label == 'dense' else 5*(index+1),
                rows=2, mass_sum=2 if label == 'dense' else 1.5, dense_sq=2,
                error_sq=0 if label == 'dense' else .1)
            records.append(r)
        return dict(prediction='A', completion_tokens=[1, 2], records=records, backend='synthetic',
            records_source=dict(path=str(raw), sha256=study.sha(raw.read_bytes())))
    monkeypatch.setattr(report, 'cached', output)
    def pair(row, out, dense):
        return dict(id=row['id'], benchmark=row['benchmark'], task=row['task'], calibration=row['calibration'],
            accuracy=1., dense_accuracy=1., matching=2, compared=3, exact_match=False,
            termination_reason='length', unparsed_answer=False, output_length=2,
            aggregates={k: aggregate([r for r in out['records'] if k == 'overall' or r['attention_type'] == k])
                for k in ('overall', 'local', 'global')})
    monkeypatch.setattr(report.evidence, 'pair', pair)
    monkeypatch.setattr(report, 'nemo_pairs', lambda row, outs:
        ([dict(condition=n, **pair(row, o, outs['dense'])) for n, o in outs.items()], []))
    (tmp_path/'final_configs').mkdir()
    for label in setup['conditions']:
        name = label.rsplit('_s', 1)[0] if label != 'dense' else label
        cfg = {} if name == 'dense' else study.parent.CONFIGS[name]
        c = dict(fingerprint='test', name=name, target=0. if name == 'dense' else int(label.rsplit('_s', 1)[1])/100,
            config=cfg, sources={}, thresholds={b: None if name == 'dense' else
                {k: dict(log_threshold=-2.) for k in ('local', 'global')} for b in ('aime26', 'longbench_v2')})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    audit = study.regenerate(tmp_path)
    assert audit['complete'] and audit['completed'] == audit['expected'] == 1040
    assert len(study.read(tmp_path/'summary.json')) == 52
    text = (tmp_path/'report.md').read_text()
    assert 'Full-budget Gaussian rank32' in text and 'No tolerance exception is inherited' in text
    assert 'Only uncentered rank32 is newly calibrated' not in text
    assert audit == study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    def missing(*args, **kwargs):
        if args[2]['id'] == setup['final'][0]['id'] and args[4] == 'jl_gaussian_r32_s50':
            raise FileNotFoundError('synthetic missing')
        return output(*args, **kwargs)
    monkeypatch.setattr(report, 'cached', missing)
    assert not study.regenerate(tmp_path)['complete']
