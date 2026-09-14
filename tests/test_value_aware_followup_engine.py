import copy

import pytest

from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware.refinements import source_sha as refinement_sha
from experiments.diffusion_gemma_value_aware.ranking_guards import source_sha as guard_sha


def example():
    row=dict(id='lb2/test',benchmark='longbench_v2',prompt_hash='prompt',seed=42,generation_budget=128)
    records=[dict(layer=layer,head=head,step=0,probe='execution',
        attention_type='global' if layer%6==5 else 'local',eligible=2,skipped=0,pv_omitted=0,
        mass_sum=1,rows=1,error_sq=0,dense_sq=1,softmax_skipped=0,compensated=0,
        prefix_eligible=1,canvas_eligible=1,boundary_eligible=0,
        prefix_skipped=0,canvas_skipped=0,boundary_skipped=0)
        for layer in range(30) for head in range(16)]
    data=dict(row,fingerprint='fp',config={},thresholds=None,screen=False,records=records,
        finite_calls=30,termination_reason='eos',completion_tokens=[1,2])
    return row,data


def test_complete_dense_cache_passes_shared_raw_audit():
    row,data=example();check_result(data,row,'fp',{},None)


@pytest.mark.parametrize('field,value',[
    ('fingerprint','wrong'),('seed',99),('prompt_hash','wrong'),('generation_budget',32),
    ('thresholds',{'local':0.}),('screen',True),
])
def test_incompatible_cached_generations_are_rejected(field,value):
    row,data=example();data[field]=value
    with pytest.raises(ValueError):check_result(data,row,'fp',{},None)


def test_dense_joint_screen_provenance_is_audited_without_changing_data():
    row,data=example();data.update(screen=True,refinement_sha256=refinement_sha(),ranking_guard_sha256=guard_sha())
    before=copy.deepcopy(data)
    check_result(data,row,'fp',{},None,screen=True)
    assert data==before
    data['ranking_guard_sha256']='bad'
    with pytest.raises(ValueError,match='observer source'):
        check_result(data,row,'fp',{},None,screen=True)


def test_physical_counts_and_layer_classification_are_not_trusted():
    row,data=example();data['records'][0]['skipped']=3
    with pytest.raises(ValueError,match='work counts'):
        check_result(data,row,'fp',{},None)
    row,data=example();data['records'][0]['attention_type']='global'
    with pytest.raises(ValueError,match='layer classification'):
        check_result(data,row,'fp',{},None)
