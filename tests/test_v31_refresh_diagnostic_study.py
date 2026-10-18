"""CPU-only tests of cohort selection, schedule binding and hard budgets."""
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
