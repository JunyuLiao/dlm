from experiments.value_direction_hopper.query_adaptive_guardrail_report import (
    _bootstrap, _token_agreement,
)


def test_task_stratified_paired_interval_keeps_examples_together():
    left=[];right=[]
    for task in range(13):
        for number in range(10):
            identifier=f'task{task}/sample{number}'
            left.append(dict(id=identifier,task=str(task),score=1.,steps=5))
            right.append(dict(id=identifier,task=str(task),score=0.,steps=4))
    assert _bootstrap(left,right,'score',draws=100)==[1.,1.]
    assert _bootstrap(left,right,'steps',draws=100)==[1.,1.]


def test_token_agreement_counts_extra_positions_and_post_divergence_matches():
    sparse=[dict(id='a',completion_tokens=[1,9,3,4]),
            dict(id='b',completion_tokens=[5])]
    dense={'a':[1,2,3], 'b':[5,6]}
    # a: positions 1 and 3 match, fourth is extra; b: one match, one missing.
    assert _token_agreement(sparse,dense)==(3/6,0.)
