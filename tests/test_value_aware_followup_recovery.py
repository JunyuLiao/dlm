import json
import pytest

from experiments.diffusion_gemma_value_aware_followup import recover as r


def test_missing_calibration_methods_are_selected_without_reading_final_scores(tmp_path):
    for benchmark in r.BENCHMARKS:
        folder = tmp_path/'verified_policies'/benchmark
        folder.mkdir(parents=True)
        for name in r.METHODS:
            if (benchmark, name) != ('longbench_v2', 'aligned'):
                (folder/f'{name}_s50.json').write_text('existing policy audited by normal pipeline')
    assert r.missing_methods(tmp_path) == ['aligned']


def test_recovery_reuses_normal_stages_and_continues_after_calibration_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'missing_methods', lambda _: ['aligned'])
    calls = []
    def supervise(root, stage, extra):
        calls.append((stage, extra))
        if stage == 'calibrate':
            raise SystemExit(1)
        assert r.jobs.worker_args is r.stage_command
    monkeypatch.setattr(r.jobs, 'supervise', supervise)
    out = r.run_recovery(tmp_path)
    assert calls == [('calibrate', ['--methods', 'aligned', '--targets', '0.5']), ('development', [])]
    assert len(out['failures']) == 1 and out['remaining_missing'] == ['aligned']
    assert 'recovery_calibration' in (tmp_path/'failures.jsonl').read_text()


def test_complete_policies_do_not_rerun_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'missing_methods', lambda _: [])
    calls = []
    monkeypatch.setattr(r.jobs, 'supervise', lambda root, stage, extra: calls.append(stage))
    assert r.run_recovery(tmp_path)['failures'] == []
    assert calls == ['development']


def test_recovery_rejects_a_stale_or_unrelated_dependency(tmp_path, monkeypatch):
    (tmp_path/'queued_workflow.json').write_text(json.dumps(dict(pid=10)))
    monkeypatch.setattr(r, 'process_identity', lambda _: None)
    with pytest.raises(ValueError, match='actual live queue'):
        r.launch(tmp_path, 10)
    monkeypatch.setattr(r, 'process_identity', lambda _: dict(pid=10, state='S', start_ticks='1', command='unrelated'))
    with pytest.raises(ValueError, match='actual live queue'):
        r.launch(tmp_path, 10)
