import pytest
from scripts.v31_value_replicated_study import CONTROL,BEFORE,AFTER,schedule,drift,clustered,paired_values


def row(n=10,active=False):
    return dict(output_hash=str(n),denoise_forwards=n,canvases=2,output_tokens=400,rng_seed=42,
        receipts=dict(adapter=dict(value_refinement_work=dict(corrected_calls=int(active)))))


def test_warmups_excluded_and_each_replicate_reverses_same_candidate_order():
    modes=['full_swap8','residual','residual_direction_guard','full_swap8_direction_guard']
    cells=[dict(index=i,seed=42) for i in range(4)];warm=[dict(index=10,seed=42)]
    one=schedule(cells,modes,warm,1);two=schedule(cells,modes,warm,2)
    for rows,arms,groups,count in (one,two):
        assert count==4 and len(rows)==28 and len(arms)==29
        assert arms[0]==CONTROL and arms[1:5]==modes
        assert rows[:4]==warm*4
        for cell,g in zip(cells,groups):
            assert rows[g['start']:g['end']]==[cell]*6
            assert arms[g['start']+1:g['end']+1]==g['arms']
            assert g['labels'][0]==BEFORE and g['labels'][-1]==AFTER
    for a,b in zip(one[2],two[2]):assert a['arms'][1:-1]==b['arms'][1:-1][::-1]


def test_drift_is_preserved_even_with_equal_public_hashes():
    rows=[row(),row(11,True),row()]
    text=[dict(completion=x) for x in ('control','candidate','control')]
    assert drift(rows,text,[CONTROL,'residual',CONTROL])==[]
    text[-1]['completion']='changed'
    assert drift(rows,text,[CONTROL,'residual',CONTROL])==['control decoded text drift']
    rows[1]=row();text[-1]['completion']='control'
    assert drift(rows,text,[CONTROL,'residual',CONTROL])==['inactive candidate decoded text drift: residual']


def test_pairing_averages_both_bookends_and_engines_without_best_of_two():
    scores={};rows={}
    for rep,(a,b,v) in enumerate(((0,1,1),(0,0,0)),1):
        for label,correct,n in ((BEFORE,a,10),(AFTER,b,14),('residual',v,9+rep)):
            scores[f'r{rep}_{label}']={'k':correct};rows[f'r{rep}_{label}']={'k':row(n)}
    values,reps=paired_values(scores,rows,'residual')
    assert values['k']==[.25,-1.5,0,0]
    assert reps[1]['k'][0]==.5 and reps[2]['k'][0]==0


def test_cluster_bootstrap_rejects_unequal_weights_and_retains_constant_effect():
    values={'a':[2,3],'b':[2,3],'c':[2,3],'d':[2,3]}
    result=clustered(values,dict(a=0,b=0,c=1,d=1))
    assert result['delta']==[2,3] and result['ci95']==[[2,2],[3,3]]
    with pytest.raises(RuntimeError):clustered(values,dict(a=0,b=0,c=0,d=1))
