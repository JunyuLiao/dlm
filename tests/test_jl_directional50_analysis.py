from copy import deepcopy
import pytest
from experiments.diffusion_gemma_jl_directional50_analysis import analyze, PAIRS, LABELS, plot_rows


def samples():
    rows = []
    names = sorted({n for pair in PAIRS for n in pair})
    for benchmark, count in (('aime26', 30), ('longbench_v2', 50)):
        for name in names:
            for index in range(count):
                eligible = 1 if index == 0 else 99
                skipped = 1 if index == 0 else 49
                rows.append(dict(benchmark=benchmark, condition=name, id=f'{benchmark}/{index}',
                    calibration=index < 6, accuracy=float(index % 2 == 0),
                    aggregates={kind: dict(eligible=eligible, skipped=skipped) for kind in ('overall', 'global', 'local')}))
    return rows


def test_all_questions_pairing_and_count_weighted_sparsity():
    rows = analyze(samples())
    assert len(rows) == 6 and all(r['accuracy_delta_pp'] == 0 for r in rows)
    assert all(r['paired_ci95_pp'] == [0., 0.] for r in rows)
    first = rows[0]
    assert first['count'] == 30 and first['candidate_correct'] == 15
    assert first['candidate_overall_sparsity_pct'] == 100*(1+29*49)/(1+29*99)
    assert first['candidate_within48_52_all_types']
    assert rows == analyze(samples())


def test_missing_or_unpaired_samples_are_rejected():
    raw = samples()
    with pytest.raises(ValueError):
        analyze(raw[1:])
    changed = deepcopy(raw); changed[0]['id'] = 'wrong_id'
    with pytest.raises(ValueError, match='not paired'):
        analyze(changed)


def test_local_global_mismatch_is_not_hidden_by_overall():
    raw = samples()
    for row in raw:
        if row['condition'] == 'jl_sign_r32_s50':
            row['aggregates']['local']['skipped'] = 0
    compared = analyze(raw)[0]
    assert compared['overall_sparsity_gap_pp'] == 0
    assert not compared['within3pp_all_types']
    assert not compared['candidate_within48_52_all_types']


def test_figures_include_only_complete_fifty_percent_conditions():
    rows = [dict(benchmark='aime26', split='full', condition=name,
                 target=0 if name == 'dense' else .5) for name in LABELS]
    assert [r['condition'] for r in plot_rows(rows, 'aime26')] == list(LABELS)
    with pytest.raises(ValueError, match='eight complete'):
        plot_rows(rows[:-1], 'aime26')
    rows[-1]['target'] = .75
    with pytest.raises(ValueError, match='another target'):
        plot_rows(rows, 'aime26')
