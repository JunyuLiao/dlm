"""Reduced scope must preserve held-out separation and cache identity."""
import pytest
from experiments.diffusion_gemma_value_aware.efficient_iteration import (
    CANDIDATES, SCREEN_METHODS, final_condition_names, guard_legacy_run, link_input, matching_sources,
    screen_conditions, screen_rows, select_candidate,
)


def test_nine_screen_conditions_preserve_exact_config_objects():
    names = ['dense']+[f'{m}_s{t}' for m in SCREEN_METHODS for t in (25,50,75,90)]
    contract = dict(conditions={n:dict(config=dict(name=n)) for n in names})
    selected = screen_conditions(contract)
    assert len(selected)==9 and len(selected)*16==144
    assert list(selected)[0]=='dense'
    assert all(selected[n] is contract['conditions'][n] for n in selected)
    assert not any(n.endswith(('_s25','_s90')) for n in selected)


def test_explicit_scope_change_prevents_old_sweep_resurrection(tmp_path):
    guard_legacy_run(tmp_path,'run-final')
    dest=tmp_path/'efficient_iteration';dest.mkdir();(dest/'iteration_plan.json').write_text('{}')
    for stage in ('finalize','run-final'):
        with pytest.raises(RuntimeError,match='superseded'):guard_legacy_run(tmp_path,stage)
    guard_legacy_run(tmp_path,'report')
    guard_legacy_run(dest,'run-final')


def test_selected_final_is_thirteen_conditions_not_cartesian_sweep():
    names = final_condition_names('mass_exact')
    assert len(names)==13 and len(names)*80==1040
    assert names[0]=='dense' and not any(n.startswith(('contribution_', 'qk_', 'mass_value_')) for n in names)
    assert len(final_condition_names(None))==9
    with pytest.raises(ValueError):final_condition_names('unselected_method')


def fixture_setup():
    rows = [dict(id=f'a{i}',prompt_hash=f'pa{i}',benchmark='aime26',task='aime',calibration=True) for i in range(6)]
    rows += [dict(id=f'l{t}{i}',prompt_hash=f'pl{t}{i}',benchmark='longbench',task=str(t),calibration=True)
        for t in range(5) for i in range(2)]
    return dict(calibration=rows, final=rows[:6]+[
        dict(id='heldout',prompt_hash='heldout_hash',benchmark='longbench',calibration=False)])


def test_screen_reuses_only_calibration_and_all_five_tasks():
    setup=fixture_setup()
    assert len(screen_rows(setup))==16


@pytest.mark.parametrize('key',['id','prompt_hash'])
def test_final_leakage_rejected(key):
    setup=fixture_setup();setup['calibration'][6][key]=setup['final'][-1][key]
    with pytest.raises(AssertionError):screen_rows(setup)


def test_task_imbalance_rejected():
    setup=fixture_setup();setup['calibration'][-1]['task']='0'
    with pytest.raises(AssertionError):screen_rows(setup)


def scores():
    return [dict(benchmark=b,condition=f'{m}_s{t}',count=6 if b=='aime26' else 10,
        accuracy=.5,dense_accuracy=.6,overall_physical_sparsity=t/100,overall_denominator_mass=.9)
        for m in CANDIDATES for b in ('aime26','longbench') for t in (50,75)]


def test_selection_balances_benchmarks_and_deterministic_ties():
    rows=scores()
    assert select_candidate(rows)[0]=='mass_exact'
    rows[-1]['accuracy']=.6
    assert select_candidate(rows)[0]=='mass_value'


def test_actual_sparsity_breaks_accuracy_tie():
    rows=scores();rows[-1]['overall_physical_sparsity']=.8
    assert select_candidate(rows)[0]=='mass_value'


def test_failed_incomplete_candidate_cannot_win():
    rows=scores();rows[-1]['accuracy']=1.;rows[-1]['count']=9
    assert select_candidate(rows)[0]=='mass_exact'
    assert select_candidate([])==(None,[])


def test_cache_links_are_idempotent_but_never_redirected(tmp_path):
    source=tmp_path/'source';source.mkdir()
    link=tmp_path/'link';link_input(link,source);link_input(link,source)
    assert link.resolve()==source
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(RuntimeError):link_input(link,other)
    file=tmp_path/'occupied';file.write_text('preserve')
    with pytest.raises(RuntimeError):link_input(file,source)
    assert file.read_text()=='preserve'


def test_final_cache_lookup_only_for_shared_aime_calibration(tmp_path):
    from experiments.diffusion_gemma_value_aware.run import shard_path
    name='mass_exact_s50'
    row=dict(id='aime26/2',benchmark='aime26',calibration=True)
    path=shard_path(tmp_path,'final',name,row['id'])
    path.parent.mkdir(parents=True);path.write_text('{}')
    assert matching_sources(tmp_path,row,name,{'config':{}})==[path]
    row['calibration']=False
    assert matching_sources(tmp_path,row,name,{'config':{}})==[]
    row.update(benchmark='longbench',calibration=True)
    assert matching_sources(tmp_path,row,name,{'config':{}})==[]
