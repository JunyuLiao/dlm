import json

import pytest

from scripts import optimized_refresh_native_campaign as campaign


@pytest.mark.parametrize('suite,threshold',[('humaneval',0.5600716206066294),('longbench_v2',0.021)])
def test_fixed_threshold_does_not_launch_calibration(tmp_path,monkeypatch,suite,threshold):
    protocol={'fixed_v2_thresholds':{suite:{'threshold':threshold,'source':'frozen user decision'}}}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    def forbidden(*args,**kwargs):
        pytest.fail('fixed workload must not run calibration forwards')
    monkeypatch.setattr(campaign,'run_job',forbidden)
    first=campaign.calibrate_held(tmp_path,suite)
    second=campaign.calibrate_held(tmp_path,suite)
    assert first==second
    assert first['threshold']==threshold
    assert first['calibration_cells']==0
    assert first['held_sparsity'] is None
    assert first['status']=='fixed_threshold_by_user'


def test_qualification_reuse_rejects_changed_worker(tmp_path,monkeypatch):
    prior=tmp_path/'prior';prior.mkdir();(prior/'protocol.json').write_text('{}')
    shared=dict(runtime={'runtime':'same'},kernel_sha256='same',model_inventory_sha256='same',suites={},input_sha256={})
    old=dict(shared,execution_sha256={'scripts/optimized_refresh_worker.py':'old'})
    new=dict(shared,execution_sha256={'scripts/optimized_refresh_worker.py':'changed'},
             qualification_reuse={'study':str(prior),'protocol_sha256':campaign.sha(prior/'protocol.json')})
    monkeypatch.setattr(campaign,'verify',lambda p:old if p==prior else new)
    with pytest.raises(RuntimeError,match='execution identity mismatch'):
        campaign.qualify(tmp_path)


@pytest.mark.parametrize('suite,in_range',[('aime26',True),('math500',False)])
def test_calibration_reuses_points_and_extends_only_when_needed(tmp_path,monkeypatch,suite,in_range):
    previous=tmp_path/'previous.json'
    prior={'suite':suite,'protocol_sha256':'past','candidates':{'0.8':{
        'calls':10,'eligible_tiles':1000,'kept_tiles':550 if in_range else 680,
        'sparsity':.45 if in_range else .32}}}
    previous.write_text(json.dumps(prior))
    protocol={'qualification_reuse':{'protocol_sha256':'past'},
              'calibration_reuse':{suite:{'source':'previous.json','sha256':campaign.sha(previous)}},
              'calibration':{'candidate_thresholds':[1.6], 'refinements':8}}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    cells=tmp_path/f'private/schedules/{suite}/calibration.json';cells.parent.mkdir(parents=True)
    cells.write_text(json.dumps([{}]*4))
    calls=[]
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    def job(*args):
        calls.append(args[-1]);return tmp_path/'measurement'
    monkeypatch.setattr(campaign,'run_job',job)
    monkeypatch.setattr(campaign,'held_work',lambda p:dict(calls=10,eligible_tiles=1000,kept_tiles=550,sparsity=.45))
    result=campaign.calibrate_held(tmp_path,suite)
    assert result['in_target_range']
    assert result['candidates']['0.8']==prior['candidates']['0.8']
    assert calls==([] if in_range else [1.6])
