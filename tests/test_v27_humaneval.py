import copy
import hashlib
import json

import pytest

from scripts import v27_humaneval as he


def test_prepare_keeps_gold_out_of_generation():
    task = dict(task_id='HumanEval/0', prompt='def f(x):\n    """Toy specification."""\n',
                entry_point='f', test='def check(candidate):\n    assert candidate(2) == 3',
                canonical_solution='    return x + 1')
    rows, gold = he.prepare_rows([task], lambda text: [1, 2, 3])
    assert rows[0]['generation_budget'] == 8192 and rows[0]['thinking'] is True
    assert rows[0]['prompt_hash'] == hashlib.sha256(rows[0]['prompt'].encode()).hexdigest()
    assert not set(rows[0]) & {'test', 'entry_point', 'canonical_solution', 'gold'}
    assert 'canonical_solution' not in gold['humaneval/0']
    assert 'def check' not in rows[0]['prompt']
    with pytest.raises(ValueError):
        he.prepare_rows([task, task], lambda text: [])


@pytest.mark.parametrize('raw,reason', [
    ('analysis only ```python\ndef f(x): return x\n```', 'no_final_response'),
    ('<channel|>```python\ndef g(x): return x\n```', 'missing_entry_point'),
    ('<channel|>```python\ndef f(:\n```', 'syntax_error'),
    ('<channel|>```java\nwhatever\n```', 'no_python_fence'),
])
def test_invalid_extraction(raw, reason):
    assert he.extract_code(raw, 'f') == (None, reason)


def test_extractor_never_chooses_later_code_by_test_success():
    raw = '<channel|>```python\ndef f(x): return 0\n```\n```python\ndef f(x): return x\n```<turn|>'
    assert he.extract_code(raw, 'f') == ('def f(x): return 0', 'parsed')
    assert he.extract_code('<channel|>def f(x): return x<eos>', 'f')[1] == 'parsed'


def test_contract_rejects_task_or_sandbox_drift():
    c = he.contract('a' * 64, 'b' * 64)
    he.validate_contract(c)
    for field, value in [('thinking', False), ('generation_budget', 2048), ('test_wall_timeout_s', 30),
                         ('dataset_revision', 'master'), ('sandbox', 'unrestricted')]:
        bad = copy.deepcopy(c)
        bad[field] = value
        with pytest.raises(ValueError):
            he.validate_contract(bad)


def test_humaneval_freeze_six_seeds_and_same_host_cells(tmp_path):
    from test_v27_panel import _setup, _row
    from scripts.v21_freeze_panel import freeze_v27
    from scripts.v21_run import validate_protocol, check_task_row
    from scripts.v27_datasets import base_task, runtime_base_task
    arms = {'D_fa4_allkept': {'kind': 'dense_fa4_allkept'},
            'M3': {'kind': 'method', 'parent': 'M3_R3_A8_current_output'}}
    sp, op, bp, pool = _setup(tmp_path, arms, warm=False, seeds=(404, 505, 606, 707, 808, 909))
    spec = json.loads(sp.read_text())
    spec['ids'] = {'humaneval': ['humaneval/0', 'humaneval/7']}
    spec['extra_gold_sha256'] = {'humaneval': 'c' * 64}
    spec['task_contracts'] = {'humaneval': he.contract('a' * 64, 'b' * 64)}
    sp.write_text(json.dumps(spec))
    (pool / 'humaneval_pool_manifest.json').write_text(json.dumps([_row(i) for i in spec['ids']['humaneval']]))
    frozen = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    assert len(frozen['schedule']) == 2 * 6 * 2
    assert base_task('humaneval') == 'humaneval'
    assert runtime_base_task('humaneval') == 'longbench_v2'
    for block in frozen['block_assignments']:
        assert len({e['host'] for e in frozen['schedule'] if e['block'] == int(block)}) == 1
    check_task_row('humaneval', dict(thinking=True, generation_budget=8192))
    frozen['task_contracts']['humaneval']['generation_budget'] = 2048
    with pytest.raises(ValueError):
        validate_protocol(frozen)
