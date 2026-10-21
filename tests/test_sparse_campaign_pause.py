"""The user-paused benchmark must not be resumed through the mutable controller."""
import json
import pytest

def test_paused_attempt_rejects_any_new_worker_before_reading_source(tmp_path):
 from scripts.optimized_refresh_campaign import run_job
 (tmp_path/'PAUSED_USER_REQUEST.json').write_text(json.dumps(dict(do_not_resume_attempt=True)))
 with pytest.raises(RuntimeError,match='Explicitly paused attempt'):
  run_job(tmp_path,'humaneval','v31_optimized','audit',tmp_path/'absent.json','runs')

def test_old_campaign_and_preparation_are_disabled_without_qualification(tmp_path):
 from scripts.optimized_refresh_campaign import run,prepare
 with pytest.raises(RuntimeError,match='relaunch is gated'):run(tmp_path)
 with pytest.raises(RuntimeError,match='preparation is disabled'):prepare(tmp_path/'new')
