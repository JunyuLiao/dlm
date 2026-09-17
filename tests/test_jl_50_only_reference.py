from pathlib import Path
import json
from experiments import diffusion_gemma_jl_50_only_reference as study


def test_scope_and_resumed_worker_cannot_schedule75(tmp_path, monkeypatch):
    monkeypatch.setattr(study.fb, 'prepare', lambda root: dict(conditions=['old'], revision='test'))
    monkeypatch.setattr(study.fb, 'execution', lambda root: dict(fingerprint='test', sources={}))
    folder = study.control(tmp_path); folder.mkdir()
    (folder/'tests.xml').write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    (tmp_path/'user_scope_50_only.json').write_text(json.dumps(dict(allowed_target_sparsities=[.5], first_finish='full_centered_s50')))
    (tmp_path/'execution_contract.json').write_text('{}')
    (tmp_path/'accepted_round2').mkdir()
    (tmp_path/'accepted_round2/extension_contract.json').write_text('{"sources": {}}')
    (tmp_path/'final_configs').mkdir()
    for label in study.CONDITIONS:
        (tmp_path/'final_configs'/f'{label}.json').write_text('{}')
    setup, contract, schedule = study.scheduling_contract(tmp_path, freeze=True)
    assert setup['targets'] == [.5] and len(setup['conditions']) == 5
    assert schedule['expected_with_baselines'] == 400 and schedule['labels'] == ['full_centered_s50']
    def worker(root):
        assert study.rf.LABELS == ('full_centered_s50',)
        assert study.rf.control(root) == folder
        (folder/'terminal.json').write_text(json.dumps(dict(complete=True, completed=400, expected=720)))
    monkeypatch.setattr(study.rf, 'work', worker)
    study.execute(tmp_path, 'work')
    assert study.read(folder/'terminal.json')['expected'] == 400
    assert study.rf.LABELS == ('full_centered_s50', 'full_centered_s75')


def test_report_view_filters75_without_modifying_originals(tmp_path):
    root = tmp_path/'source'; root.mkdir()
    (root/'setup.json').write_text('{}')
    (root/'shared_state_index.json').write_text('[]')
    for label in study.CONDITIONS:
        path = root/'final_configs'/f'{label}.json'; path.parent.mkdir(exist_ok=True); path.write_text('{}')
        stage = 'dense' if label == 'dense' else 'final'; (root/stage/label).mkdir(parents=True)
        if label != 'dense':
            for benchmark in ('aime26', 'longbench_v2'):
                p = root/'policies'/benchmark/f'{label}.json'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    index = []
    for target in (.5, .75):
        p = root/f'diagnostic{target}.json'
        p.write_text(json.dumps(dict(identity=dict(name='full_centered', target=target))))
        index.append(dict(path=str(p), sha256=study.sha(p.read_bytes())))
    (root/'shared_diagnostics_index.json').write_text(json.dumps(index))
    view = study.report_view(root, {}, dict(fingerprint='test'))
    assert len(study.read(view/'shared_diagnostics_index.json')) == 1
    assert study.read(view/'scope.json')['expected'] == 400
    assert len(study.read(root/'shared_diagnostics_index.json')) == 2
    assert (view/'final/full_centered_s50').is_symlink()
    assert not (view/'final/full_centered_s75').exists()


def test_scoped_report400_completion_and_calibration_description(tmp_path):
    for completed in (399, 400):
        audit = dict(completed=completed, missing=[] if completed == 400 else ['missing'], violations=[])
        study.write_report(tmp_path, dict(revision='test'), [], [], [], [], audit, [])
        assert audit['expected'] == 400 and audit['complete'] == (completed == 400)
        text = (tmp_path/'report.md').read_text()
        assert '/400 audited outputs:80 reference outputs plus320 reused baselines' in text
        assert 'Both50% and75%' not in text and '/720' not in text
        assert 'at2048 AIME/4096 LongBench' in text
        assert ('Status: COMPLETE' in text) == (completed == 400)
