from scripts.v31_value_warmup_factor import CONTROL,measured_group


def record(active=False):
    return dict(index=10,panel_seed=42,output_hash='same',denoise_forwards=20,
                canvases=2,output_tokens=400,rng_seed=42,
                receipts=dict(adapter=dict(value_refinement_work=dict(corrected_calls=int(active)))))


def test_active_candidate_text_change_is_allowed_but_bookend_drift_is_retained():
    rows=[record(),record(True),record()]
    private=[dict(completion=s) for s in ('control','candidate','control')]
    assert measured_group(rows,private,[CONTROL,'residual',CONTROL])['errors']==[]
    private[-1]['completion']='changed control'
    assert measured_group(rows,private,[CONTROL,'residual',CONTROL])['errors']==['control decoded text drift']


def test_inactive_candidate_text_drift_is_retained_even_if_public_hash_matches():
    rows=[record(),record(),record()]
    private=[dict(completion=s) for s in ('control','changed','control')]
    result=measured_group(rows,private,[CONTROL,'residual',CONTROL])
    assert result['errors']==['inactive candidate decoded text drift: residual']
    assert 'completion' not in str(result)
