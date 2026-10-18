"""CPU-only tests of cohort selection, schedule binding and hard budgets."""
import copy
import json
import math
import pytest
from scripts import v31_refresh_diagnostic_study as study


def cell(index, length, dataset='lb'):
    return dict(dataset=dataset, index=index, id=str(index), seed=42, prompt_tokens=length)


def test_cohort_uses_metadata_and_distinct_tasks():
    lb = [cell(i * 3 + j, length + j) for i, length in enumerate((20000, 40000, 80000, 110000)) for j in range(3)]
    ruler = [cell(i, 130000, 'ruler') for i in range(12)]
    meta = [dict(task='variant' + str(i), task_base='task' + str(i // 2)) for i in range(12)]
    first = study.choose_cohorts(lb, ruler, meta)
    second = study.choose_cohorts(list(reversed(lb)), list(reversed(ruler)), meta)
    assert first == second
    assert len(first['longbench_v2']) == 8
    assert len({meta[c['index']]['task_base'] for c in first['ruler128k']}) == 4
    assert [c['prompt_tokens'] for c in first['headroom']] == [40002, 110002]
    altered = [dict(c, previous_accuracy=0, denoise_forwards=999) for c in lb]
    assert [c['index'] for c in study.choose_cohorts(altered, ruler, meta)['longbench_v2']] == [c['index'] for c in first['longbench_v2']]


def test_factorial_schedule_exact_coverage_and_warmup_alignment():
    pool = [cell(i, 40000) for i in range(8)]
    cells, arms = study.factorial_schedule(pool, 0)
    assert len(cells) == 100 and len(arms) == 101
    assert arms[0]['policy'] == 'control' and arms[0]['warmup']
    assert [a['policy'] for a in arms[1:5]] == list(study.FACTORIAL)
    assert all(a['warmup'] for a in arms[1:5])
    assert sum(not a['warmup'] for a in arms[1:]) == 96
    for offset in range(4, len(cells), 6):
        group = arms[offset + 1:offset + 7]
        assert group[0]['label'] == 'control_before' and group[-1]['label'] == 'control_after'
        assert {a['policy'] for a in group[1:-1]} == set(study.FACTORIAL)
        assert len({(c['index'], c['seed']) for c in cells[offset:offset + 6]}) == 1
    qcells, qarms = study.factorial_schedule(pool[:1], 0, (42,))
    assert len(qcells) == 10 and len(qarms) == 11


def test_headroom_repetitions_and_control_policy():
    pool = [cell(1, 40000), cell(2, 110000)]
    cells, arms = study.headroom_schedule(pool, 'shared', True)
    assert len(cells) == 10 and sum(not a['warmup'] for a in arms[1:]) == 8
    assert {a['policy'] for a in arms} == {'control', 'fixed16'}
    assert {a['label'] for a in arms if not a['warmup']} == {'dynamic', 'fixed16'}
    dense_cells, dense_arms = study.headroom_schedule(pool, 'dense_full_fix51994', True)
    assert len(dense_cells) == 6 and len(dense_arms) == 7
    assert {a['policy'] for a in dense_arms} == {'control'}


def test_campaign_and_phase_budget_includes_failed_or_orphaned_workers():
    reservations = [dict(phase='qualification', reserved_s=900., elapsed_s=100., status='failed'),
                    dict(phase='headroom', reserved_s=1000., elapsed_s=None, status='started')]
    assert study.available_seconds('qualification', reservations) == 800.
    assert study.available_seconds('headroom', reservations) == 500.
    assert study.available_seconds('factorial', reservations) == 3900.
    reservations.append(dict(phase='factorial', reserved_s=3900., elapsed_s=3900.))
    assert study.available_seconds('factorial', reservations) == 0.
    assert sum(study.charged_seconds(reservations).values()) == 5000.
    with pytest.raises(ValueError):
        study.validate_budget(dict(study.BUDGET, reserve=901.), 7200.)
    with pytest.raises(ValueError):
        study.validate_budget(dict(study.BUDGET), 7201.)
    with pytest.raises(ValueError):
        study.charged_seconds([dict(phase='qualification', reserved_s=100, elapsed_s=math.nan)])


def test_only_final_partial_json_line_is_recoverable(tmp_path):
    path = tmp_path / 'records.jsonl'
    path.write_text('{"ok": 1}\n{"ok":')
    assert study.complete_jsonl(path, allow_truncated=True) == [{'ok': 1}]
    with pytest.raises(ValueError):
        study.complete_jsonl(path)
    path.write_text('{"ok":\n{"ok": 1}\n')
    with pytest.raises(ValueError):
        study.complete_jsonl(path, allow_truncated=True)


def test_missing_audit_fields_are_not_zero_evidence():
    c = cell(1, 40000)
    row = dict(dataset='lb', index=1, panel_seed=42, rng_seed=study.request_seed(31, c, 0),
               cuda_graph_captures=0, fix_51994=True, fa4_local_fix=True, value_clean_timing=False,
               denoise_forwards=2, canvases=1, sampler_calls=3, engine_steps=5, prefill_steps=2,
               block_size=32, chunk=16384, max_model_len=141312)
    errors = study.validate_row(row, c, study.entry('control', 'control_before'), 0,
        dict(family='shared', pass_name='audit'), dict(block=32, chunk=16384, max_len=141312, mage_k=8192))
    assert 'LOCAL scope' in errors and 'GLOBAL layer count' in errors
    assert 'missing refresh diagnostic' in errors and 'refresh schedule identity' in errors


def test_qualification_separates_structural_from_causal_gate():
    labels = list(study.FACTORIAL) + ['control_before', 'control_after']
    rows = [dict(label=label, output_hash='a', denoise_forwards=20, canvases=1, output_tokens=100,
                 diagnostic_receipt=dict(exposed=True, prefix_signature='prefix')) for label in labels]
    assert study.qualification_summary(rows, [])['causal_status'] == 'qualified'
    rows[0]['diagnostic_receipt']['prefix_signature'] = 'different'
    result = study.qualification_summary(rows, [])
    assert result['structural_status'] == 'passed' and result['causal_status'] == 'unresolved'
    assert result['factorial_prefix_status'] == 'mismatched'
    rows[0]['diagnostic_receipt']['prefix_signature'] = 'prefix'
    rows[0]['output_hash'] = 'changed'
    assert study.qualification_summary(rows, [])['dense_fresh_and_aa_exact_parity'] is False
    assert study.qualification_summary(rows[:-1], [])['structural_status'] == 'failed'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))


@pytest.fixture
def recovery_case(tmp_path):
    parent, successor = tmp_path / 'campaign/attempt001', tmp_path / 'campaign/attempt002'
    parent.mkdir(parents=True); successor.mkdir()
    source = {}
    for rel, content in [('scripts/v31_refresh_diagnostic_study.py', 'original coordinator'),
                         ('scripts/v31_refresh_headroom_profile.py', 'original profiler'),
                         ('scripts/v31_refresh_diagnostic_worker.py', 'unchanged native worker')]:
        path = parent / 'private/execution' / rel
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        source[rel] = study.base.sha(path)
    plans = []
    for engine in range(1, 7):
        phase = 'qualification' if engine == 1 else 'headroom'
        folder = f'{phase}/engine{engine:03d}/attempt001'
        for root in (parent, successor):
            for name in ('cells', 'arms'):
                write_json(root / folder / f'private/{name}.json', [dict(engine=engine)])
        plans.append(dict(engine=engine, phase=phase, suite='longbench_v2', family='shared',
                          pass_name='audit' if engine == 1 else 'clean' if engine < 6 else 'profile',
                          repeat=0, folder=folder, request_count=2, measured_count=1,
                          **{name + '_sha256': study.base.sha(parent / folder / f'private/{name}.json')
                             for name in ('cells', 'arms')}))
    original = dict(source_commit='old-source', plans=plans, execution_fingerprints=source,
                    budgets_seconds=dict(study.BUDGET), total_gpu_seconds=7200.,
                    minimum_engine_start_seconds=180., cohorts={'longbench_v2': [{'index': 9}]},
                    frozen_native={'canvas': 256}, runtime={'gpu': 'same-h100'}, input_fingerprints={})
    write_json(parent / 'protocol.json', original)
    original_hash = study.base.sha(parent / 'protocol.json')
    reservations = [dict(attempt=parent.name, engine=i, phase='qualification' if i == 1 else 'headroom',
                         reserved_s=100., elapsed_s=10., status='finished', returncode=int(i == 6))
                    for i in range(1, 7)]
    write_json(parent.parent / 'gpu_budget.json', dict(schema='refresh_gpu_reservations_v1', reservations=reservations))
    write_json(parent / 'state.json', dict(status='failed', completed_engines=[1, 2, 3, 4, 5],
        failure={'engine': 6, 'status': 'failed'}, cumulative_gpu_seconds=60., protocol_sha256=original_hash))
    for plan in plans[:5]:
        leaf = parent / plan['folder']
        marker = dict(status='complete', errors=[], returncode=0, timed_out=False,
                      protocol_sha256=original_hash, source_commit='old-source', records=2,
                      private_records=2, measured_records=1, planned_measured=1)
        for name in ('records.jsonl', 'private/generation.private.jsonl', 'scored.json'):
            write_json(leaf / name, {'original': plan['engine']})
        write_json(leaf / 'config.json', dict(plan, protocol_sha256=original_hash, source_commit='old-source'))
        write_json(leaf / 'receipt.json', marker)
        members = ['records.jsonl', 'private/generation.private.jsonl', 'scored.json', 'config.json', 'receipt.json']
        if plan['phase'] == 'qualification':
            write_json(leaf / 'qualification.json', {'causal_status': 'qualified'})
            members.append('qualification.json')
        marker['files_sha256'] = {name: study.base.sha(leaf / name) for name in members}
        write_json(leaf / 'attempt_complete.json', marker)
    protocol = copy.deepcopy(original)
    protocol['source_commit'] = 'new-source'
    protocol['execution_fingerprints'].update({study.RECOVERY_SPEC: 'addendum-hash',
        'scripts/v31_refresh_headroom_profile.py': 'fixed-profiler-hash',
        'scripts/v31_refresh_diagnostic_study.py': 'successor-coordinator-hash'})
    return parent, successor, protocol


def test_successor_references_original_artifacts_without_copying_or_relabeling(recovery_case):
    parent, successor, protocol = recovery_case
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    study.bind_successor(successor, protocol, parent)
    assert study.validate_references(successor, protocol) == [1, 2, 3, 4, 5]
    assert protocol['budgets_seconds'] == study.RECOVERY_BUDGET
    assert 'source_reference' not in protocol['plans'][5]
    for plan in protocol['plans'][:5]:
        reference = plan['source_reference']
        assert reference['source_commit'] == 'old-source'
        assert reference['protocol_sha256'] == study.base.sha(parent / 'protocol.json')
        assert study.plan_artifact_leaf(successor, plan) == parent / plan['folder']
        assert not (successor / plan['folder'] / 'records.jsonl').exists()
    assert before == {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    reservations = study.base.read(parent.parent / 'gpu_budget.json')['reservations']
    assert study.available_seconds('headroom', reservations, protocol['budgets_seconds']) == 1750.
    assert study.available_seconds('factorial', reservations, protocol['budgets_seconds']) == 3900.


@pytest.mark.parametrize('change', ['native_source', 'cohort', 'schedule', 'runtime'])
def test_successor_rejects_changes_outside_recovery_scope(recovery_case, change):
    parent, successor, protocol = recovery_case
    if change == 'native_source':
        protocol['execution_fingerprints']['scripts/v31_refresh_diagnostic_worker.py'] = 'changed'
    elif change == 'cohort':
        protocol['cohorts']['longbench_v2'][0]['index'] = 10
    elif change == 'schedule':
        protocol['plans'][0]['cells_sha256'] = 'changed'
    else:
        protocol['runtime']['gpu'] = 'different'
    with pytest.raises(RuntimeError):
        study.bind_successor(successor, protocol, parent)


@pytest.mark.parametrize('artifact', ['records.jsonl', 'attempt_complete.json', 'private/cells.json'])
def test_successor_revalidates_every_referenced_artifact(recovery_case, artifact):
    parent, successor, protocol = recovery_case
    study.bind_successor(successor, protocol, parent)
    path = parent / protocol['plans'][0]['folder'] / artifact
    path.write_text(path.read_text() + '\n')
    with pytest.raises(RuntimeError):
        study.validate_references(successor, protocol)


def test_successor_cannot_forget_failed_gpu_work_or_reference_failure(recovery_case):
    parent, successor, protocol = recovery_case
    study.bind_successor(successor, protocol, parent)
    ledger = parent.parent / 'gpu_budget.json'
    data = study.base.read(ledger); data['reservations'].pop(); write_json(ledger, data)
    with pytest.raises(RuntimeError, match='reservations'):
        study.validate_references(successor, protocol)


def test_successor_rejects_reference_path_escape(recovery_case):
    parent, successor, protocol = recovery_case
    study.bind_successor(successor, protocol, parent)
    plan = copy.deepcopy(protocol['plans'][0]); plan['source_reference']['attempt'] = '../elsewhere'
    with pytest.raises(RuntimeError):
        study.plan_artifact_leaf(successor, plan)


def test_successor_run_counts_verified_references_without_launch(recovery_case, tmp_path, monkeypatch):
    parent, successor, protocol = recovery_case
    study.bind_successor(successor, protocol, parent)
    write_json(successor / 'protocol.json', protocol)
    (tmp_path / 'results').mkdir()
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'verify', lambda path, p, **kwargs: study.validate_references(path, p))
    def unexpected(*args, **kwargs):
        raise AssertionError('referenced engines must not launch')
    monkeypatch.setattr(study, 'require_idle', unexpected)
    monkeypatch.setattr(study.subprocess, 'Popen', unexpected)
    ledger = parent.parent / 'gpu_budget.json'; before = ledger.read_bytes()
    result = study.run(successor, stop_after='qualification')
    assert result['status'] == 'phase_stop'
    assert result['completed_engines'] == result['referenced_engines'] == [1, 2, 3, 4, 5]
    assert result['cumulative_gpu_seconds'] == 60.
    assert ledger.read_bytes() == before
