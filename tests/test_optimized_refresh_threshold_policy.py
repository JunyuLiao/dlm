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
