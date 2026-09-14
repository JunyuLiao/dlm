"""The automatic handoff cannot bypass selection/audit or hide an incomplete run."""
import json
from types import SimpleNamespace
import pytest
from experiments.diffusion_gemma_value_aware import finalize
from experiments.diffusion_gemma_value_aware.queue import stage_command


def fixtures(root,missing=False):
    (root/'candidate_decision.json').write_text(json.dumps(dict(
        selected_methods=['mass_exact','mass','mass_value'],heldout_used=False)))
    for name in ('blasst_original','blasst_aggressive','mass_exact','mass','mass_value'):
        for target in (25,50,75,90):
            for benchmark in ('aime26','longbench'):
                if missing and (name,target,benchmark)==('mass_exact',25,'aime26'):continue
                path=root/'verified_policies'/benchmark/f'{name}_s{target}.json'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')


def simulate(tmp_path,monkeypatch,codes,complete,missing=False):
    fixtures(tmp_path,missing)
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs))
        if command[3].endswith('.report'):
            (tmp_path/'audit.json').write_text(json.dumps(dict(complete=complete,expected=3760,completed=3760 if complete else 80)))
        return SimpleNamespace(returncode=codes[len(calls)-1])
    monkeypatch.setattr(finalize.subprocess,'run',run)
    result=finalize.primary_handoff(['mass_exact','mass','mass_value'],tmp_path)
    return result,calls,json.loads((tmp_path/'primary_handoff_terminal.json').read_text())


def test_failed_freeze_never_launches_gpu(tmp_path,monkeypatch):
    result,calls,status=simulate(tmp_path,monkeypatch,[1],False)
    assert result==1 and len(calls)==1 and not status['complete']
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']==''


def test_complete_handoff_runs_existing_gates_in_order(tmp_path,monkeypatch):
    result,calls,status=simulate(tmp_path,monkeypatch,[0,0,0],True)
    assert result==0 and status['complete']
    assert [c[0][3].rsplit('.',1)[-1] for c in calls]==['evaluate','launch','report']
    assert calls[1][0][-1]=='run-final' and 'env' not in calls[1][1]
    assert calls[2][1]['env']['CUDA_VISIBLE_DEVICES']==''


@pytest.mark.parametrize('codes,complete',[([0,0,0],False),([0,1,0],False),([0,0,1],True)])
def test_report_audit_not_exit_code_decides_completeness(tmp_path,monkeypatch,codes,complete):
    result,calls,status=simulate(tmp_path,monkeypatch,codes,complete)
    assert result==1 and len(calls)==3 and not status['complete']


def test_final_queue_requires_explicit_candidates_and_full_target_scope():
    assert stage_command('finalize',['mass_exact'])[3:]==[
        'experiments.diffusion_gemma_value_aware.finalize','--methods','mass_exact']
    for methods,targets,rounds in [(None,None,None),(['mass_exact'],[.5],None),(['mass_exact'],None,3)]:
        with pytest.raises(ValueError):stage_command('finalize',methods,targets,rounds)


def test_missing_policy_gets_one_bounded_cache_reusing_recovery(tmp_path,monkeypatch):
    result,calls,status=simulate(tmp_path,monkeypatch,[0,0,0,0],True,missing=True)
    assert result==0 and len(calls)==4 and status['recovery_methods']==['mass_exact']
    assert calls[0][0][4:]==['calibrate','--methods','mass_exact','--targets','0.25','--max-rounds','3']
    assert calls[1][0][4]=='freeze'


def test_failed_recovery_cannot_launch_final(tmp_path,monkeypatch):
    result,calls,status=simulate(tmp_path,monkeypatch,[1],False,missing=True)
    assert result==1 and len(calls)==1 and not status['complete']


def test_handoff_rejects_changed_selection_before_any_gpu_work(tmp_path):
    fixtures(tmp_path)
    with pytest.raises(ValueError,match='candidate decision'):
        finalize.missing_policy_recovery(tmp_path,['aligned'])
