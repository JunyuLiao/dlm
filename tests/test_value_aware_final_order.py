"""Baseline-first execution must survive canonical JSON key sorting."""
import json
import pytest
from experiments.diffusion_gemma_value_aware.evaluate import final_condition_order


def conditions():
    return json.loads(json.dumps({
        'dense':{'config':{}},
        'blasst_original_s25':{'config':{'method':'blasst'}},
        'mass_exact_s25':{'config':{'method':'mass','mode':'exact'}},
    },sort_keys=True))


def test_dense_precedes_alphabetically_earlier_conditions():
    assert final_condition_order(conditions())==[
        'dense','blasst_original_s25','mass_exact_s25']


def test_sparse_resume_still_reuses_dense_first():
    assert final_condition_order(conditions(),['mass_exact_s25'])==['dense','mass_exact_s25']
    assert final_condition_order(conditions(),['dense'])==['dense']


@pytest.mark.parametrize('selection',[[],['typo'],['dense','typo']])
def test_invalid_selection_fails_before_model_loading(selection):
    with pytest.raises(ValueError):final_condition_order(conditions(),selection)


@pytest.mark.parametrize('invalid',[{}, {'dense':{'config':{'method':'blasst'}}}])
def test_missing_or_modified_dense_fails(invalid):
    with pytest.raises(ValueError):final_condition_order(invalid)
