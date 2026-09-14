"""Plot groupings must preserve routing mechanisms and confirmatory splits."""
import pytest
from experiments.diffusion_gemma_value_aware.report import plot_family,plot_series,plot


@pytest.mark.parametrize('name,expected',[
    ('sol_plain_gaussian_s25','sol_plain_gaussian'),
    ('sol_plain_topk_s50','sol_plain_topk'),
    ('sol_value_gaussian_s90','sol_value_gaussian'),
    ('mass_exact_s50','mass_exact'),('mass_s50','mass'),
    ('contribution_topp_p95','contribution_topp'),
    ('dense','dense'),('name_plain','name_plain')])
def test_strip_only_numeric_terminal_budget(name,expected):
    assert plot_family(name)==expected


def fixtures():
    rows=[]
    for benchmark,split in [('aime26','full'),('aime26','heldout24'),('longbench','full')]:
        for name,method,target in [('dense','dense',0.),('blasst_original_s25','blasst',.25),
            ('mass_s50','mass',.5),('mass_exact_s50','mass',.5),
            ('sol_plain_gaussian_s25','sol',.25),('sol_plain_topk_s50','sol',.5),
            ('contribution_topp_p95','diagnostic',None),('zero_pv_s50','zero_pv',.5)]:
            rows.append(dict(benchmark=benchmark,split=split,condition=name,method=method,target=target,
                accuracy=.5,overall_denominator_mass=.9,overall_relative_error=.2,token_agreement=.1,
                overall_physical_sparsity=target or .1,global_physical_sparsity=target or .1,
                local_physical_sparsity=target or .1,overall_pv_omission=.5))
    return rows


def test_modes_and_splits_stay_separate():
    series=plot_series(fixtures(),'aime26','heldout24')
    assert 'sol_plain_gaussian' in series and 'sol_plain_topk' in series
    assert 'mass' in series and 'mass_exact' in series
    assert 'zero_pv' not in series
    assert all(r['split']=='heldout24' for points in series.values() for r in points)
    assert set(plot_series(fixtures(),'aime26','heldout24',True))=={
        'dense','blasst_original','mass','mass_exact'}
    assert not plot_series(fixtures(),'longbench','heldout24')


def test_render_all_and_focused_figures(tmp_path):
    plot(tmp_path,fixtures())
    files={p.name for p in (tmp_path/'figures').iterdir()}
    assert {'accuracy.png','primary_accuracy.png','primary_overall_denominator_mass.png',
        'aime26_target_vs_actual.png','aime26_heldout24_target_vs_actual.png',
        'longbench_target_vs_actual.png','aime26_heldout24_pv_replacement.png'}<=files
    assert all(p.stat().st_size>1000 for p in (tmp_path/'figures').iterdir())
