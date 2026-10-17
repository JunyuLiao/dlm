from scripts.v31_value_e2e_study import schedule,validate,CONTROL,MODES
from scripts.v31_value_e2e_report import bootstrap
import numpy as np


def test_schedule_pairs_every_question_and_reverses_order():
    cells=[dict(index=i,seed=42) for i in range(5)]
    a,arms,groups,warm=schedule(cells,cells[:2],'shared','clean')
    b,barms,bgroups,bwarm=schedule(cells,cells[:2],'shared','audit')
    assert warm==bwarm==4 and len(a)==len(b)==24 and len(arms)==25
    assert arms[0]==CONTROL
    for i,(g,h) in enumerate(zip(groups,bgroups)):
        assert a[g['start']:g['end']]==[cells[i]]*4
        assert g['arms'][0]==g['arms'][-1]==CONTROL
        assert g['arms'][1:3]==h['arms'][1:3][::-1]
        assert set(g['labels'])=={'control_before','control_after',*MODES}


def test_clean_validation_does_not_invent_work_audit():
    c=dict(dataset='d',index=0,seed=42)
    row=dict(dataset='d',index=0,panel_seed=42,repeat=0,cuda_graph_captures=0,block_size=32,chunk=16384,
        max_model_len=141312,fix_51994=True,fa4_local_fix=True,value_clean_timing=True,
        timer_policy='request_and_prefill_boundaries',value_audit=False,local_kv_budget=None,
        denoise_forwards=2,canvases=1,output_tokens=5,engine_steps=4,prefill_steps=1,sampler_calls=3,
        prefill_s=2.,decode_s=3.,wall_s=5.,receipts=None)
    assert not validate(row,c,'dense_full_fix51994','clean')
    assert 'measured graph capture' in validate(dict(row,cuda_graph_captures=1),c,'dense_full_fix51994','clean')
    assert 'boundary time sum' in validate(dict(row,wall_s=4.9),c,'dense_full_fix51994','clean')
    assert 'input/seed' in validate(dict(row,panel_seed=1),c,'dense_full_fix51994','clean')


def test_paired_speedup_and_quality_use_same_trials():
    a=np.ones((8,7));a[:,0]=50;a[:,4:6]=10
    b=a.copy();b[:,0]=75;b[:,4:6]=5
    r=bootstrap(a,b,draws=100)
    assert r['estimate'][0]==25
    np.testing.assert_allclose(r['estimate'][7:],[2,2,2,2])
    np.testing.assert_allclose(r['ci95'][7:],[[2,2]]*4)
