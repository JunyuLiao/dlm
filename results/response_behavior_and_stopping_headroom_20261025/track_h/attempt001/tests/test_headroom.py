import importlib.util
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from census import census,stopping_class
from bank import matching,build
from measure import compare_to_reference


def row(step,**kw):
    return dict(commit=False,native_step=step,native_history_ready=True,stable=True,confidence=False,converged=False,cap_reached=False,**kw)


def test_anchor_requires_later_canvas_and_never_forces_last_call():
    trace=[row(1),row(2),dict(commit=True),row(1),row(2),row(3)]
    assert census(trace)['first_eligible_anchor']==dict(canvas=1,after_completed_calls=2)
    assert census([row(1),row(2)])['first_eligible_anchor'] is None
    assert census([dict(commit=True),row(47),row(48)])['first_eligible_anchor'] is None


def test_history_not_ready_is_separate_from_both_predicates():
    x=row(1);x.update(native_history_ready=False,stable=False)
    assert stopping_class(x)=='history_not_ready'
    x.update(native_history_ready=True);assert stopping_class(x)=='both_fail'


def test_directional_matching_keeps_physical_target_independent_of_motion():
    assert matching(dict(current_error=.02,deleted_mass=.1),dict(current_error=.0205,deleted_mass=.105))
    assert not matching(dict(current_error=.02,deleted_mass=.1),dict(current_error=.03,deleted_mass=.105))


def test_bank_is_current_only_and_preserves_native_rng():
    g=torch.Generator().manual_seed(7)
    q=torch.randn((256,16,6),generator=g)*.01;k=torch.randn((512,2,6),generator=g)*.01
    v=torch.ones((512,2,6))+torch.randn((512,2,6),generator=g)*.01
    state=torch.get_rng_state().clone();b,p,w=build(q,k,v,256,.1)
    assert torch.equal(state,torch.get_rng_state())
    assert all(x['drop_count']==1 for x in b['units']) and w['QK_rectangles']==32
    assert all(len(x['all_biases'])==4 for x in p)
    for selection in b['selections'].values():
        assert all(len(x['tiles'])==1 and max(x['tiles'])<4 for x in selection['units'])


def test_signed_reference_margin_handles_ties_and_cancellation():
    tied=torch.ones((1,8))
    reference=dict(winner=torch.tensor([0]),margin=torch.zeros(1))
    assert compare_to_reference(tied,reference)['reference_winner_margin_by_position']==[0.]
    z=torch.tensor([[2.,2.5,0.],[0.,3.,2.]])
    reference=dict(winner=torch.tensor([0,1]),margin=torch.tensor([2.,1.]))
    out=compare_to_reference(z,reference)
    assert out['reference_winner_crossings']==1
    assert out['worst_margin_erosion']==2.5
    assert out['signed_margin_change_by_position']==[-2.5,0.]


def test_candidate_bias_matches_explicit_current_masked_attention():
    g=torch.Generator().manual_seed(19)
    q=torch.randn((256,16,6),generator=g)*.02;k=torch.randn((512,2,6),generator=g)*.02
    v=torch.randn((512,2,6),generator=g)
    public,private,_=build(q,k,v,256,.1)
    for unit,stored in zip(public['units'],private):
        h,qb=unit['head'],unit['qb'];z=q[qb*128:(qb+1)*128,h]@k[:,h//8].T*.1
        dense=z.softmax(-1)@v[:,h//8]
        for i,candidate in enumerate(unit['candidates']):
            masked=z.clone()
            for tile in candidate['tiles']:masked[:,tile*64:(tile+1)*64]=-torch.inf
            assert torch.allclose(stored['all_biases'][i],masked.softmax(-1)@v[:,h//8]-dense,atol=1e-6)


def test_anchor_cannot_stitch_calls_across_canvas_commits():
    assert census([dict(commit=True),row(2),dict(commit=True),row(2)])['first_eligible_anchor'] is None
    from decide import complete_endpoint
    assert not complete_endpoint(dict(remaining_calls=3),dict(pulse=dict(remaining_calls=None)))
    assert not complete_endpoint(dict(remaining_calls=None),dict(pulse=dict(remaining_calls=3)))
    assert complete_endpoint(dict(remaining_calls=3),dict(pulse=dict(remaining_calls=2)))


def test_cap_termination_is_not_counted_as_an_additional_confidence_delay():
    x=row(48);x.update(cap_reached=True,converged=True)
    assert stopping_class(x)=='cap_only_termination'
    result=census([dict(commit=True),x,dict(commit=True)])
    assert result['counts'].get('confidence_only_delay',0)==0
    assert result['counts']['cap_only_termination']==1
    x['confidence']=True
    assert stopping_class(x)=='adaptive_and_cap'
