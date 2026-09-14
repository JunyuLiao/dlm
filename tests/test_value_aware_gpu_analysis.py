import pytest

from experiments.diffusion_gemma_value_aware_gpu.analysis import analyze


def test_mass_control_is_paired_and_reports_type_budget_mismatch():
    rows, raw = [], []
    for label, score, overall, local, glob in (
        ('mass_s50', .5, .5, .5, .5),
        ('value_s50', 1., .51, .56, .46),
        ('blasst_original_s50', .5, .3, .25, .4),
    ):
        name = label.rsplit('_s', 1)[0]
        rows.append(dict(benchmark='aime26', split='full', condition=label, name=name,
            target=.5, accuracy=score, overall_physical_sparsity=overall,
            local_physical_sparsity=local, global_physical_sparsity=glob,
            overall_mass=.9, token_agreement=.5, overall_relative_error=.1,
            thresholds={'local':dict(cap_one=True, unattainable=True)} if name=='blasst_original' else {}))
        for i in range(2):
            raw.append(dict(benchmark='aime26', condition=label, id=f'aime26/{i}',
                            calibration=False, accuracy=1. if name=='value' or i==0 else 0.))
    result = analyze(rows, raw)
    control, = result['mass_controls']
    assert control['accuracy_delta_pp'] == 50.
    assert control['paired_ci95_pp'] == [0., 100.]
    assert control['within3pp_overall'] and not control['within3pp_all_types']
    assert control['local_sparsity_gap_pp'] == pytest.approx(6.)
    assert result['unattainable'] == [dict(benchmark='aime26',condition='blasst_original_s50',
        attention_type='local',target=50.,threshold=1.,final_actual=25.)]
    assert all(r['pearson_with_accuracy'] is None for r in result['correlations'])
    raw[-3]['id'] = 'wrong-example'
    with pytest.raises(ValueError, match='identical sample IDs'):
        analyze(rows, raw)
