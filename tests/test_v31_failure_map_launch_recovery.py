import json
import pytest
from scripts.v31_failure_map_launch_recovery import prepare


def test_no_reuse(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
    with pytest.raises(RuntimeError,match='new sibling'):prepare(old,new)


@pytest.mark.parametrize('records,returncode',[(1,2),(0,1)])
def test_only_zero_record_path_failure_can_continue(tmp_path,records,returncode):
    old=tmp_path/'old';old.mkdir()
    (old/'state.json').write_text(json.dumps(dict(status='failed',phase='screen')))
    leaf=old/'screen/family/attempt001';leaf.mkdir(parents=True)
    (leaf/'receipt.json').write_text(json.dumps(dict(records=records,private_records=0,returncode=returncode,timed_out=False)))
    with pytest.raises(RuntimeError,match='not a zero-record'):prepare(old,tmp_path/'new')
    assert not (tmp_path/'new').exists()
