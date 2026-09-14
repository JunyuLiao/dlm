import math
import numpy as np
import pytest

from experiments.diffusion_gemma_value_aware_followup.diagnostics import (
    finite_delta,delta_statistics,selected_rows,comparisons,risk_changes)


def test_value_signal_offsets_and_mandatory_tiles_are_explicit():
    a=np.array([np.inf,1.+math.log(2),2.+math.log(2)])
    b=np.array([np.inf,1.,2.])
    delta,excluded=finite_delta(a,b,math.log(2))
    assert excluded==1 and np.allclose(delta,0.)
    with pytest.raises(ValueError,match='support'):
        finite_delta(a,[0.,1.,2.])
    with pytest.raises(ValueError,match='unaligned'):
        finite_delta(a,[1.,2.])


def test_signal_distribution_weights_physical_candidates_not_samples():
    result=delta_statistics([np.zeros(100),np.ones(1)],3)
    assert result['finite_physical_candidates']==101
    assert result['mean']==pytest.approx(1/101)
    assert result['fraction_abs_below_0p001']==pytest.approx(100/101)
    assert result['excluded_nonfinite_candidates']==3


def test_diagnostic_selection_refuses_final_rows():
    configs={n:dict(pooling='rms') for n in ('value','mass_value','risk')}
    with pytest.raises(ValueError,match='cannot use final'):
        selected_rows([dict(split='final')],configs)


def test_budget_mismatch_is_not_treated_as_a_value_improvement():
    common=dict(benchmark='longbench_v2',split='development',attention_type='overall',target=.9,mass=.9)
    rows=[dict(common,name='value',relative_error=.1,physical_sparsity=.8),
        dict(common,name='no_value_control',relative_error=.2,physical_sparsity=.9)]
    result=comparisons(rows)[0]
    assert not result['exact_budget_match'] and result['relative_error_change'] is None
    rows[0]['physical_sparsity']=.9
    result=comparisons(rows)[0]
    assert result['exact_budget_match'] and result['relative_error_change']==pytest.approx(-.5)


def test_risk_signal_diagnosis_never_reads_final_files(tmp_path):
    with pytest.raises(ValueError,match='calibration only'):
        risk_changes(tmp_path,dict(calibration=[dict(split='final')]),{})
