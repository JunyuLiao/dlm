"""50%-only scope, score-blind calibration, guard semantics and raw audit."""
from copy import deepcopy
import json
import math

import pytest
import torch

from experiments import diffusion_gemma_jl_directional50 as study


def test_score_blind_one_per_domain_selection():
    rows = study.fb.prepare(study.fb.ROOT)['final']
    selected = study.select_longbench_calibration(rows)
    assert len(selected) == 6 and len({r['task'] for r in selected}) == 6
    changed = list(reversed(deepcopy(rows)))
    for i, row in enumerate(changed):
        row.update(prediction=str(i), accuracy=i/80)
    again = study.select_longbench_calibration(changed)
    assert [r['id'] for r in selected] == [r['id'] for r in again]
    assert {r['id'] for r in selected} <= {r['id'] for r in rows}


def test_explicit_profiles_preserve_all80_inference_settings(tmp_path):
    with pytest.raises(ValueError, match='Explicit'):
        study.prepare(tmp_path/'unspecified')
    before = study.fb.prepare(study.fb.ROOT)['final']
    for profile, overlap in (('final50', 6), ('historical_disjoint', 0)):
        root = tmp_path/profile; setup = study.prepare(root, profile)
        assert len(setup['conditions']) == 8 and setup['targets'] == [.5]
        assert setup['final_generation_slots'] == 240
        assert len(setup['calibration_overlap']['longbench_v2']) == overlap
        assert len(setup['final']) == 80 and len(setup['calibration_overlap']['aime26']) == 6
        for a, b in zip(before, setup['final']):
            assert {k: v for k, v in a.items() if k != 'calibration'} == {k: v for k, v in b.items() if k != 'calibration'}
        assert study.prepare(root) == setup


def test_full_budget_membership_and_strict_two_point_gate(tmp_path):
    setup = study.prepare(tmp_path, 'final50')
    for benchmark, budget in (('aime26', 2048), ('longbench_v2', 4096)):
        assert all(r['generation_budget'] == budget for r in study.calibration_rows(setup, benchmark))
    bad = deepcopy(setup); bad['calibration'][-1]['generation_budget'] = 512
    with pytest.raises(ValueError):
        study.calibration_rows(bad, 'longbench_v2')
    bad = deepcopy(setup); bad['calibration'][-1]['prompt_hash'] = 'changed'
    with pytest.raises(ValueError):
        study.calibration_rows(bad, 'longbench_v2')
    assert study.fb.within({'overall': .5, 'local': .48, 'global': .52}, .5)
    assert not study.fb.within({'overall': .5, 'local': .479, 'global': .50}, .5)
    assert not study.fb.within(study.accepted.EXPECTED, .75)


def test_existing_guard32_uses_independent_sketch_only_in_declared_region():
    from experiments.diffusion_gemma_jl_output_aware.config import Config
    from experiments.diffusion_gemma_jl_output_aware.reference import route
    cfg = Config(**study.NEW['cancel_guard_gaussian_r32'], log_threshold=math.log(.5))
    assert cfg.rank == 32 and cfg.projection_seed == 1729 and cfg.guard_seed == 2718
    mu = torch.zeros(1, 1, 2, 2, 64); mu[..., 1, 32] = 2.
    state = dict(logz=torch.zeros(1, 1, 2, 2), mu=mu,
        mean_norm=torch.ones(1, 1, 2, 2), active=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        eligible=torch.ones(1, 1, 2, dtype=torch.bool))
    guarded = route(state, torch.ones(1, 1), cfg)
    primary_state = dict(state, mu=mu[..., :32])
    plain = route(primary_state, torch.ones(1, 1), Config(family='gaussian', rank=32, log_threshold=math.log(.5)))
    assert not guarded.any()  # First support and independent risk>=threshold.
    assert plain.tolist() == [[[False, True]]]
    weak = dict(state, logz=state['logz'].clone()); weak['logz'][..., 1] = -10.
    assert route(weak, torch.ones(1, 1), cfg).tolist() == [[[False, True]]]


def test_call_accounting_distinguishes_reuse_calibration_and_validation(tmp_path):
    for stage, values in {'calibration': [{}, {'imported_source': {'path': 'old'}}],
                          'final': [{}, {'imported_source': {'path': 'calibration'}}]}.items():
        for i, data in enumerate(values):
            p = tmp_path/stage/'condition/shards'/f'{i}.json'; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
    for stage in ('smoke', 'reference'):
        p = tmp_path/'validation/fingerprint'/stage/'condition/shards/one.json'
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    (tmp_path/'validation/fingerprint/example.native.json').write_text('{}')
    out = study.accounting(tmp_path)
    assert out['completed_new_inference'] == 5 and out['completed_validation_inference'] == 3
    assert out['completed_by_stage']['final']['imported_or_exact_reused'] == 1
    assert out['final_generation_slots'] == 240


def test_predecessor_gate_requires400_audit_and_worker_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(study.fb, 'ROOT', tmp_path)
    folder = study.reference50.control(tmp_path); view = folder/'report'; view.mkdir(parents=True)
    audit = dict(complete=True, completed=400, expected=400, missing=[], violations=[], artifacts={})
    (view/'audit.json').write_text(json.dumps(audit))
    proof = dict(passed=True, completed=400, inference_performed=False, audit_sha256=study.sha((view/'audit.json').read_bytes()))
    (view/'regeneration_verification.json').write_text(json.dumps(proof))
    (folder/'terminal.json').write_text(json.dumps(dict(complete=True, completed=400, expected=400)))
    p = folder/'supervisor_terminal.json'; p.write_text(json.dumps(dict(exit_code=0)))
    assert len(study.require_reference()) == 4
    p.write_text(json.dumps(dict(exit_code=1)))
    with pytest.raises(ValueError):
        study.require_reference()


def test_complete640_raw_roundtrip_and_missing_detection(tmp_path, monkeypatch):
    report = study.backend
    from experiments.diffusion_gemma_value_aware_gpu.routing import FIELDS
    from experiments.diffusion_gemma_value_aware.report_metrics import aggregate
    setup = study.prepare(tmp_path, 'final50')
    monkeypatch.setattr(study, 'prepare', lambda root: setup)
    monkeypatch.setattr(study, 'execution', lambda root: dict(fingerprint='test', sources={}))
    monkeypatch.setattr(study, 'audit_policy', lambda *args: None)
    monkeypatch.setattr(study, 'smoke_audit', lambda *args: {})
    monkeypatch.setattr(report.common, 'diagnostic_summary', lambda *args: ([], {}))
    monkeypatch.setattr(report.common, 'audit_matrices', lambda *args: None)
    def seed_summary(*args):
        assert report.common.TARGETS == (.5,)
        return [], {}
    monkeypatch.setattr(report.common, 'shared_seed_summary', seed_summary)
    monkeypatch.setattr(report, 'selected_sources', lambda *args: {})
    monkeypatch.setattr(report, 'plots', lambda *args: [])
    for name in ('shared_state_index', 'shared_diagnostics_index'):
        (tmp_path/f'{name}.json').write_text('[]')
    raw = tmp_path/'raw.json'; raw.write_text('{}')
    monkeypatch.setattr(report, 'shard_path', lambda *args: raw)
    def output(adapter, root, row, stage, label, name, cfg, thresholds, contract):
        assert adapter is None
        records = []
        for i in range(2):
            r = dict.fromkeys(FIELDS, 0)
            r.update(layer=i*5, step=0, head=0, attention_type=('local', 'global')[i], probe='execution',
                eligible=10*(i+1), skipped=0 if label == 'dense' else 5*(i+1), rows=2,
                mass_sum=2 if label == 'dense' else 1.5, dense_sq=2, error_sq=0 if label == 'dense' else .1)
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
        cfg = {} if name == 'dense' else study.CONFIGS[name]
        c = dict(fingerprint='test', name=name, target=0. if name == 'dense' else .5, config=cfg, sources={},
            thresholds={b: None if name == 'dense' else {k: dict(log_threshold=-2.) for k in ('local', 'global')}
                for b in ('aime26', 'longbench_v2')})
        (tmp_path/'final_configs'/f'{label}.json').write_text(json.dumps(c))
    audit = study.regenerate(tmp_path)
    assert audit['complete'] and audit['completed'] == audit['expected'] == 640
    assert len(study.read(tmp_path/'summary.json')) == 32
    text = (tmp_path/'report.md').read_text()
    assert '640 audited' in text and 'random-sign32' in text and 'overlap is 6/50' in text
    assert 'Uncentered control:' not in text and 'Both50% and75%' not in text
    assert audit == study.regenerate(tmp_path) and study.verify(tmp_path)['passed']
    def missing(*args, **kwargs):
        if args[2]['id'] == setup['final'][0]['id'] and args[4] == 'jl_sign_r32_s50':
            raise FileNotFoundError('synthetic missing')
        return output(*args, **kwargs)
    monkeypatch.setattr(report, 'cached', missing)
    assert not study.regenerate(tmp_path)['complete']
