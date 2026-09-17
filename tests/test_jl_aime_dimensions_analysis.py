from copy import deepcopy
import pytest
from experiments.diffusion_gemma_jl_aime_dimensions_analysis import analyze, PAIRS


def samples():
    rows = []
    for name in sorted({n for pair in PAIRS for n in pair}):
        for i in range(30):
            rows.append(dict(benchmark='aime26', condition=name, id=f'aime26/{i}',
                calibration=i < 6, accuracy=float(i % 2 == 0),
                aggregates={k:dict(eligible=1 if i == 0 else 99, skipped=1 if i == 0 else 49)
                            for k in ('overall','global','local')}))
    return rows


def test_all_ranks_paired_and_count_weighted_with_calibration_separate():
    rows = analyze(samples())
    assert len(rows) == 21 and all(r['delta_pp'] == 0 and r['paired_ci95_pp'] == [0.,0.] for r in rows)
    assert rows[0]['candidate_correct'] == 15 and rows[0]['count'] == 30
    assert rows[0]['candidate_overall_sparsity_pct'] == 100*(1+29*49)/(1+29*99)
    assert rows[7]['count'] == 24 and rows[14]['count'] == 6
    assert rows == analyze(samples())


def test_missing_duplicate_or_unpaired_rejected():
    raw = samples()
    with pytest.raises(ValueError): analyze(raw[1:])
    with pytest.raises(ValueError): analyze(raw+[deepcopy(raw[0])])
    raw[0]['id'] = 'incorrect'
    with pytest.raises(ValueError, match='not paired'): analyze(raw)


def test_local_gap_and_invalid_counts_not_hidden_by_overall():
    raw = samples()
    for row in raw:
        if row['condition'] == 'jl_gaussian_r8_s50': row['aggregates']['local']['skipped'] = 0
    first = analyze(raw)[0]
    assert first['overall_sparsity_gap_pp'] == 0 and not first['within2pp_all_types']
    raw[0]['aggregates']['local']['skipped'] = -1
    # Make aggregate counts invalid, even after summing all rows.
    raw[0]['aggregates']['local']['skipped'] = -100000
    with pytest.raises(ValueError, match='Invalid physical'): analyze(raw)
