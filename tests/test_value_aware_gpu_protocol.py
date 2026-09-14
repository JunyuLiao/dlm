import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.diffusion_gemma_value_aware_gpu.protocol import select_additions, QUOTAS
from experiments.diffusion_gemma_value_aware_gpu.workflow import semantic_policy
from experiments.diffusion_gemma_value_aware_gpu.report import summarize
from experiments.diffusion_gemma_value_aware.report_metrics import aggregate


def samples():
    items = []
    previous = dict(final=[], calibration=[], development=[])
    for domain in QUOTAS:
        for difficulty in ('easy','hard'):
            for i in range(14):
                identity = f'{domain}/{difficulty}/{i}'
                items.append(dict(_id=identity, domain=domain, difficulty=difficulty,
                                  length='short' if i < 9 else 'medium', answer='unused'))
                row = dict(id=identity, source_id=identity, benchmark='longbench_v2', task=domain, difficulty=difficulty)
                if i == 0: previous['calibration'].append(row)
                elif i <= 5: previous['final'].append(row)
                elif i == 6: previous['development'].append(row)
    return items, previous


def test_sample_expansion_is_score_blind_deterministic_and_disjoint():
    items, previous = samples()
    chosen = select_additions(items, previous)
    assert len(chosen) == 20
    assert chosen == select_additions(list(reversed(items)), previous)
    changed = [dict(r, answer='different', prediction='different', cache_available=True) for r in items]
    assert [r['_id'] for r in chosen] == [r['_id'] for r in select_additions(changed, previous)]
    old = {r['source_id'] for rows in previous.values() for r in rows}
    assert not old & {r['_id'] for r in chosen}


def test_same_threshold_annotations_may_alias_but_not_changed_rules():
    config = dict(method='blasst')
    a = {k:dict(log_threshold=0., cap_one=True, unattainable=True, source='50') for k in ('local','global')}
    b = {k:dict(log_threshold=0., source='75') for k in ('local','global')}
    assert semantic_policy(config, a) == semantic_policy(config, b)
    b['local']['log_threshold'] = .1
    assert semantic_policy(config, a) != semantic_policy(config, b)


def test_v2_micro_macro_and_physical_count_weighting_are_distinct():
    rows = []
    for task, count, score, eligible, skipped in [('a', 2, 1., 100, 90), ('b', 1, 0., 10, 5)]:
        for i in range(count):
            stats = aggregate([dict(eligible=eligible, skipped=skipped, rows=2, mass_sum=1.5, error_sq=1., dense_sq=4.)])
            rows.append(dict(id=f'{task}/{i}', task=task, accuracy=score, dense_accuracy=1.,
                matching=2, compared=5, exact_match=False, unparsed_answer=False,
                aggregates={k:stats for k in ('overall','global','local')}))
    result = summarize(rows)
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['equal_task_macro'] == .5
    assert result['overall_physical_sparsity'] == pytest.approx(185/210)
    assert result['token_agreement'] == .4


def test_full_work_cannot_pass_a_failed_cuda_gate(tmp_path, monkeypatch):
    from experiments.diffusion_gemma_value_aware_gpu import workflow
    import dllm.models
    monkeypatch.setattr(workflow, 'prepare', lambda root:{})
    monkeypatch.setattr(workflow, 'execution', lambda root, **kw:{})
    monkeypatch.setattr(dllm.models, 'create_adapter', lambda *a, **k:SimpleNamespace(load=lambda:None))
    def failed(*args): raise ValueError('CUDA validation failed')
    monkeypatch.setattr(workflow, 'smoke', failed)
    monkeypatch.setattr(workflow, 'prepare_policies', lambda *args:pytest.fail('Calibration started before validation'))
    with pytest.raises(ValueError, match='CUDA validation failed'):
        workflow.work(tmp_path)
