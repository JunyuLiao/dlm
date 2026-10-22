import json

import pytest

from scripts import optimized_refresh_followup as followup
from scripts.optimized_refresh_report import paired_report


def test_physical_counts_include_first_canvas_and_skip_absent_held(tmp_path):
    rows=[{'receipts':{'adapter':{
        'global_eligible_tiles':1000,'global_kept_tiles':550,
        'value_phase_tiles':{'held':{'eligible_tiles':800,'kept_tiles':400}},
        'value_canvas_phase_tiles':{'2':{'held':{'eligible_tiles':800,'kept_tiles':400}},
                                    '1':{'refresh':{'eligible_tiles':200,'kept_tiles':150}}}}}},
        {'receipts':{'adapter':{
        'global_eligible_tiles':100,'global_kept_tiles':100,'value_phase_tiles':{},
        'value_canvas_phase_tiles':{'1':{'initial':{'eligible_tiles':100,'kept_tiles':100}}}}}}]
    (tmp_path/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    work=followup.physical(tmp_path)
    assert work['global_eligible_tiles']==1100 and work['global_kept_tiles']==650
    assert work['held_sparsity']==.5
    assert work['first_kept_tiles']==250 and work['first_eligible_tiles']==300


def test_math_calibration_requires_both_held_and_whole_request(tmp_path,monkeypatch):
    protocol={'threshold_policy':{'candidates':[.05,.1],'refinements':2,'objective':'held and whole physical sparsity'}}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    cells=tmp_path/'private/schedules/math500/calibration.json';cells.parent.mkdir(parents=True)
    cells.write_text('[{},{}]')
    calls=[]
    def job(*args):calls.append(args[-1]);return args[-1]
    def work(tau):
        return dict(global_eligible_tiles=1000,global_kept_tiles=550 if tau==.1 else 610,
                    held_eligible_tiles=800,held_kept_tiles=440,first_eligible_tiles=500,
                    first_kept_tiles=300,global_sparsity=.45 if tau==.1 else .39,held_sparsity=.45)
    monkeypatch.setattr(followup.base,'run_job',job);monkeypatch.setattr(followup,'physical',work)
    assert followup.calibrate(tmp_path,'math500')==.1
    assert calls==[.05,.1]


def test_ruler_uncertainty_preserves_fractional_credit_and_task_clusters(tmp_path):
    keys=[f'ruler|{i}|42|0' for i in range(4)]
    scores={'v31_optimized':dict(zip(keys,[100.,50.,25.,0.])),
            'v2_online_optimized':dict(zip(keys,[100.,50.5,25.5,0.]))}
    rows=[dict(dataset='ruler',index=i,panel_seed=42,repeat=0,wall_s=1.,decode_s=.9,denoise_forwards=5) for i in range(4)]
    paired_report(tmp_path,scores,{arm:rows for arm in scores},{},task_of={0:'a',1:'a',2:'b',3:'b'})
    result=json.loads((tmp_path/'paired_vs_v31.json').read_text())['v2_online_optimized']
    assert result['accuracy_difference_pp']==.25
    assert result['problem_clusters']==2 and result['cluster_kind']=='RULER task'
