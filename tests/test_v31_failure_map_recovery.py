import json
import pytest
from scripts.v31_failure_map_qualification_recovery import recover


@pytest.mark.parametrize("status,phase",[("running","qualification"),("failed","qualification"),("budget_limited_partial","screen"),("complete_bounded_diagnostic","repair")])
def test_recovery_cannot_resume_active_failed_or_scientific_phases(tmp_path,status,phase):
    old=tmp_path/'old';old.mkdir()
    (old/'state.json').write_text(json.dumps(dict(status=status,phase=phase)))
    with pytest.raises(RuntimeError,match="only uncompleted"):
        recover(old,tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_recovery_never_reuses_an_attempt(tmp_path):
    new=tmp_path/'new';new.mkdir()
    with pytest.raises(RuntimeError,match="new continuation"):
        recover(tmp_path/'old',new)
