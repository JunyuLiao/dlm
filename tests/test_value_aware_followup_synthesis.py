import json
import pytest

from experiments.diffusion_gemma_value_aware_followup import synthesis as s


def point(name, target, actual, score=.5, benchmark='longbench_v2'):
    return dict(benchmark=benchmark, split='heldout24' if benchmark=='aime26' else 'full',
        condition=f'{name}_s{int(target*100)}', target=target, method=name,
        accuracy=score, overall_mass=.9, overall_relative_error=.1,
        **{f'{kind}_{metric}': actual for kind in ('overall', 'global', 'local')
            for metric in ('physical_sparsity', 'pv_omission')})


def sample(condition, identity, score, benchmark='longbench_v2', calibration=False):
    return dict(condition=condition, id=identity, accuracy=score, benchmark=benchmark, calibration=calibration)


def test_control_matching_uses_actual_not_target_and_reports_type_gaps():
    a = point('value', .75, .49, .6)
    a['local_physical_sparsity'] = .8
    b = point('no_value_control', .5, .50, .5)
    c = point('no_value_control', .75, .73, .5)
    raw = [sample(p['condition'], str(i), float(i < n)) for p,n in ((a,6),(b,5),(c,5)) for i in range(10)]
    out = s.matched_controls([a,b,c], raw)[0]
    assert out['reference'] == b['condition']
    assert out['matched_overall'] and not out['matched_both_attention_types']
    assert out['score_delta'] == pytest.approx(.1)
    assert len(out['paired_ci95']) == 2


def test_unmatched_budget_has_no_delta_or_interval_claim():
    a = point('mass_value', .5, .5, .9)
    b = point('mass', .5, .8, .2)
    out = s.matched_controls([a,b], [])[0]
    assert not out['matched_overall']
    assert out['score_delta'] is None and out['paired_ci95'] is None
    assert out['output_error_delta'] is None


def test_compensation_compares_pv_replacement_not_zero_physical_deletion():
    a = point('compensate', .5, 0.)
    b = point('zero_pv', .5, 0.)
    a['overall_pv_omission'] = .5
    b['overall_pv_omission'] = .7
    out = s.matched_controls([a,b], [])[0]
    assert out['budget_metric'] == 'pv_omission'
    assert not out['matched_overall']


def test_aime_control_pairs_exclude_original_calibration_and_reject_duplicates():
    raw = [sample('value_s50', 'test', 1., 'aime26'),
        sample('value_s50', 'cal', 0., 'aime26', True)]
    assert s.selected_samples(raw, 'aime26', 'heldout24', 'value_s50') == {'test': 1.}
    with pytest.raises(ValueError, match='duplicate'):
        s.selected_samples(raw+raw[:1], 'aime26', 'heldout24', 'value_s50')


def test_transfer_uses_frozen_calibration_type_measurements_and_cap_label(tmp_path):
    row = point('blasst_original', .5, .3)
    row['global_physical_sparsity'] = .55
    thresholds = {'local': dict(log_threshold=0., cap_one=True, unattainable=True),
        'global': dict(log_scale=2., cap_one=True)}
    policy = dict(policy=thresholds, heldout_used=False, measured={'local': .25, 'global': .49})
    path = tmp_path/'policy.json'
    path.write_text(json.dumps(policy))
    source = dict(path=str(path), sha256=s.sha(path.read_bytes()))
    frozen = dict(conditions={row['condition']: dict(policy_sources=dict(longbench_v2=source),
        thresholds=dict(longbench_v2=thresholds), target_metric='physical_sparsity')})
    out = s.calibration_transfer([row], frozen)
    assert out[0]['cap_one_unattainable'] and not out[1]['cap_one_unattainable']
    assert out[0]['calibration_to_final_shift'] == pytest.approx(.05)
    assert out[1]['final_error'] == pytest.approx(.05)
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        s.calibration_transfer([row], frozen)


def test_point_score_description_does_not_assert_equivalence():
    rows = []
    for benchmark in ('aime26', 'longbench_v2'):
        rows.extend([dict(point('dense',0.,0.,.5,benchmark), method='dense'),
            point('value', .5, .5, .5, benchmark), point('value', .9, .9, .4, benchmark)])
    out = s.point_score_preserving(rows)
    value = [r for r in out if r['method']=='value']
    assert len(value)==2 and all(r['actual']==.5 for r in value)
    assert all('not statistical equivalence' in r['statement'] for r in value)


def test_synthesis_rejects_partial_report_before_reading_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'contract', lambda _: dict(fingerprint='fp'))
    monkeypatch.setattr(s, 'load_contract', lambda *_: dict(expected_shards=780))
    folder = tmp_path/'reports'/'broad50'
    folder.mkdir(parents=True)
    (folder/'audit.json').write_text(json.dumps(dict(complete=False)))
    with pytest.raises(ValueError, match='complete audited final report'):
        s.audited_inputs(tmp_path, 'broad50')


def test_synthesis_writes_all_derived_tables_and_never_claims_goal_completion(tmp_path, monkeypatch):
    rows = []
    raw = []
    for benchmark in ('aime26', 'longbench_v2'):
        for name in ('dense', *s.METHODS):
            row = point(name, .5 if name!='dense' else 0., .5 if name!='dense' else 0., benchmark=benchmark)
            row.update(count=2, delta=0., token_agreement=.9, thresholds=None, pooling=None,
                overall_denominator_mass=.9, global_denominator_mass=.9, local_denominator_mass=.9,
                global_eligible=100., local_eligible=100.)
            rows.append(row)
            raw.extend(sample(row['condition'], str(i), float(i%2), benchmark) for i in range(2))
    data = dict(summary=rows, per_sample=raw, matched_sparsity=[], correlations=[])
    monkeypatch.setattr(s, 'audited_inputs', lambda *_: (dict(fingerprint='fp'), {}, data))
    monkeypatch.setattr(s, 'calibration_transfer', lambda *_:
        [dict(benchmark='longbench_v2', split='full', target=.5)])
    out = s.build(tmp_path, 'broad50')
    assert set(out) == {'matched_controls', 'calibration_transfer', 'layer_behavior', 'point_score_preserving'}
    folder = tmp_path/'synthesis'/'broad50'
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['complete'] and not audit['goal_completion_claim']
    assert len(audit['artifacts']) == 10
    for name, digest in audit['artifacts'].items():
        assert s.sha((folder/name).read_bytes()) == digest
    text = (folder/'report.md').read_text()
    assert 'Interim50%' in text and 'remaining target curves' in text
