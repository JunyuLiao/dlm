import json
from pathlib import Path
from scripts import v2_execution_followup as campaign

def test_ruler_calibration_targets_allstep_not_held(tmp_path,monkeypatch):
    study=tmp_path;cells=study/'private/schedules/ruler128k/calibration.json';cells.parent.mkdir(parents=True);cells.write_text('[{}]')
    protocol={'threshold_policy':{'candidates':[.01,.005],'refinements':0}}
    (study/'protocol.json').write_text(json.dumps(protocol));calls=[]
    def job(*args):calls.append(args[-1]);return Path(str(args[-1]))
    def physical(leaf):
        s=.8 if str(leaf)=='.01' or str(leaf)=='0.01' else .45
        return dict(global_sparsity=s,held_sparsity=.9,held_eligible_tiles=100,held_kept_tiles=10)
    monkeypatch.setattr(campaign.base,'run_job',job);monkeypatch.setattr(campaign.previous,'physical',physical)
    assert campaign.calibrate(study,'ruler128k')==.005
    assert calls==[.01,.005]
    receipt=json.loads((study/'calibration_held/ruler128k/selected_threshold.json').read_text())
    assert receipt['global_sparsity']==.45 and receipt['held_sparsity']==.9 and not receipt['accuracy_used']

def test_longbench_fixed_threshold_has_no_calibration_forwards(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'fixed_v2_threshold':.021}))
    def fail(*args):raise AssertionError('calibration model forward')
    monkeypatch.setattr(campaign.base,'run_job',fail)
    assert campaign.calibrate(tmp_path,'longbench_v2')==.021

def test_requested_matrix_retains_exact_budgets_and_longbench_arms():
    assert campaign.PANELS['aime26_k1920'][1]==1920
    assert campaign.PANELS['math500_k384'][1]==384
    assert campaign.PANELS['longbench_v2'][1:3]==(8192,['v31_optimized','v2_online_optimized'])
