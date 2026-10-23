import json
import pytest
from scripts.v2_execution_results import build,publication_scope

def test_publication_rejects_partial_campaign(tmp_path):
    campaign=tmp_path/'campaign';campaign.mkdir()
    with pytest.raises(RuntimeError,match='campaign incomplete'):
        build(campaign,tmp_path/'report')
    assert not (tmp_path/'report').exists()

def test_scope_change_excludes_only_cancelled_panel(tmp_path):
    from scripts import optimized_refresh_campaign as base
    order=['aime26_k1920','math500_k384','ruler128k_s45','longbench_v2']
    base.write(tmp_path/'protocol.json',dict(order=order,panels={}))
    scope=dict(original_protocol_sha256=base.sha(tmp_path/'protocol.json'),original_order=order,active_order=order[:3],cancelled_panels=order[3:])
    base.write(tmp_path/'USER_SCOPE_CHANGE_20261023.json',scope)
    for panel in order[:3]:base.write(tmp_path/panel/'final_complete.json',dict(status='complete'))
    for name in ['PAUSED_USER_REQUEST.json','NOT_RUN_USER_CANCELLED.json']:base.write(tmp_path/'longbench_v2'/name,{})
    _,active,receipt=publication_scope(tmp_path)
    assert active==order[:3] and receipt['original_campaign_complete'] is False
    (tmp_path/'longbench_v2/runs').mkdir()
    with pytest.raises(RuntimeError,match='unexpectedly started'):publication_scope(tmp_path)

def test_publication_rejects_missing_panel_marker(tmp_path):
    campaign=tmp_path/'campaign';campaign.mkdir()
    (campaign/'final_complete.json').write_text('{}')
    (campaign/'protocol.json').write_text(json.dumps(dict(order=['longbench_v2'],panels={'longbench_v2':'bad'})))
    # The verifier must reject absent execution/protocol evidence before it
    # creates any output directory or a completion marker.
    with pytest.raises((RuntimeError,FileNotFoundError)):
        build(campaign,tmp_path/'report')
    assert not (tmp_path/'report').exists()
