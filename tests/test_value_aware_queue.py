"""Queued follow-ups preserve explicit, bounded development selection."""
import sys
import pytest
from experiments.diffusion_gemma_value_aware.queue import stage_command


def test_bounded_development_command():
    names=['blasst_original','mass_exact','zero_pv']
    assert stage_command('development',names,[.5])==[
        sys.executable,'-u','-m','experiments.diffusion_gemma_value_aware.launch',
        'development','--methods',*names,'--targets','0.5']


@pytest.mark.parametrize('names,targets',[(None,None),(['mass_exact'],None),(None,[.5]),(['mass_exact'],[.6])])
def test_no_implicit_or_unsupported_development_sweep(names,targets):
    with pytest.raises(ValueError):stage_command('development',names,targets)


def test_refinement_stage_unchanged_and_rejects_ignored_selections():
    assert stage_command('refinement-screen')[-1]=='refinement-screen'
    with pytest.raises(ValueError):stage_command('refinement-screen',['mass_exact'],[.5])
    with pytest.raises(ValueError):stage_command('unknown')


def test_explicit_calibration_followup_command():
    command=stage_command('calibrate',['blasst_aggressive','mass_exact'],[.25,.9],3)
    assert command[4:]==['calibrate','--methods','blasst_aggressive','mass_exact',
        '--targets','0.25','0.9','--max-rounds','3']


@pytest.mark.parametrize('rounds',[None,0,-1])
def test_calibration_queue_requires_bounded_rounds(rounds):
    with pytest.raises(ValueError,match='positive round limit'):
        stage_command('calibrate',['mass_exact'],[.25],rounds)


def test_other_stages_do_not_silently_ignore_round_limits():
    with pytest.raises(ValueError):stage_command('development',['mass_exact'],[.5],3)
    with pytest.raises(ValueError):stage_command('refinement-screen',max_rounds=3)
