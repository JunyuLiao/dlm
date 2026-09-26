"""v15 CPU tests: LongBench-v2 scorer contract, panel selection rule, schedule/resume/score/aggregate.

No GPU, no model weights, no real gold: synthetic label pairs only.
"""
import json

import pytest

from scripts import v15_longbench_task as task
from scripts.v13_seed_runs import execution_key, plan_schedule, run_schedule, validate_resume

ARMS = ('B8_P', 'CVM_T', 'D_native', 'T_G_original', 'T_P')


# ------------------------------------------------------------------ scorer
@pytest.mark.parametrize('raw,gold,term,task_ok,strict_ok,parsed', [
    ('<|channel>thought\nmaybe \\boxed{C}<channel|>Reasoning.\nAnswer: \\boxed{B}<turn|>', 'B', 'eos', True, True, True),
    ('<|channel>thought\nmaybe \\boxed{C}<channel|>Reasoning.\nAnswer: \\boxed{B}<turn|>', 'C', 'eos', False, False, True),  # EOS-wrong; thought not mined
    ('<|channel>thought\nI think \\boxed{A} ...', 'A', 'length', False, False, False),            # cap inside thought: no final channel
    ('<|channel>thought\nx<channel|>(A) fails, (C) fits.\nAnswer: \\boxed{C}', 'C', 'length', True, False, True),  # valid final choice at cap
    ('<|channel>thought\nx<channel|>Answer: \\boxed{E}', 'A', 'eos', False, False, False),         # malformed choice
    ('<|channel>thought\nx<channel|>A or D? The best is D.', 'D', 'eos', False, False, False),      # no box/format: no arbitrary letter search
    ('<|channel>thought\nx<channel|>Answer: D', 'D', 'eos', True, True, True),                     # NeMo 'Answer: X' fallback
])
def test_scorer_contract(raw, gold, term, task_ok, strict_ok, parsed):
    (row,) = task.score([raw], [gold], [term])
    assert row['task_correct'] is task_ok and row['strict_correct'] is strict_ok and row['parsed'] is parsed


def test_scorer_rejects_non_choice_gold():
    with pytest.raises(ValueError):
        task.score(['<channel|>Answer: \\boxed{A}'], ['7'], ['eos'])


# ------------------------------------------------------------------ selection rule
def test_round_robin_selection_is_deterministic_and_balanced():
    from scripts.v15_build_protocol import select
    pool = [dict(source_id=f'id{d}{b}{k}', domain=f'D{d}', bin=b, prompt_tokens_n=1) for d in range(3)
            for b in ('10-15K', '15-20K') for k in range(3)]
    a, b = select(pool, n=8), select(list(reversed(pool)), n=8)
    assert [r['source_id'] for r in a] == [r['source_id'] for r in b]
    assert {(r['domain'], r['bin']) for r in a[:6]} == {(f'D{d}', x) for d in range(3) for x in ('10-15K', '15-20K')}
    assert len(select(pool[:2], n=12)) == 2                   # shortfall is returned, never padded


# ------------------------------------------------------------------ schedule / resume / score / aggregate
def fake_run(entry, gold):
    letter = gold[entry['id']] if (entry['arm'] != 'D_native' or entry['seed'] == 17) else 'A'
    raw = f'<|channel>thought\nx<channel|>Answer: \\boxed{{{letter}}}<turn|>'
    calls = 20 + ARMS.index(entry['arm'])
    return dict(ok=True, id=entry['id'], seed=entry['seed'], generation_seed=entry['seed'], api_wall_s=float(calls),
                decoder_calls=calls, canvases=2, output_tokens=300, termination='eos', completion_token_hash=f'h{entry["cell_id"]}',
                per_canvas_calls=[10, calls - 10], triton_misses=0, new_shared_objects=[]), dict(raw_completion=raw, seed=entry['seed'], id=entry['id'])


def test_synthetic_schedule_save_resume_score_aggregate(tmp_path):
    from scripts.v15_summarize import load_records, summarize
    ids, seeds = ['longbench_v2/q1', 'longbench_v2/q2'], [17, 29]
    gold = {'longbench_v2/q1': 'B', 'longbench_v2/q2': 'C'}
    hashes = {a: f'hash{a}' for a in ARMS}
    schedule = plan_schedule('v15_test', 'rev', hashes, ids, seeds, 5)
    assert len(schedule) == 2 * 2 * 5 * 2
    ledger, receipts = [], {}

    def execute(row, seed, config, entry):
        rec, receipt = fake_run(entry, gold)
        return dict(record=rec, receipt=receipt)

    def save(path, obj):
        receipts[str(path)] = obj
    rows = {i: dict(id=i) for i in ids}
    kw = dict(execute_one=execute, rows=rows, configs={a: {} for a in ARMS}, ledger_append=ledger.append,
              private=tmp_path, save_receipt=save)
    assert run_schedule(schedule, done=set(), max_executions=13, **kw) == 'stopped'
    identity = dict(protocol_id='v15_test', model_revision='rev', arm_hashes=hashes, source_hashes={}, private_root='p')
    events = [dict(event='start', **identity)] + ledger
    done = validate_resume(events, identity, schedule)
    assert len(done) == 13
    assert run_schedule(schedule, done=done, **kw) == 'complete'
    assert len([e for e in ledger if e['event'] == 'run']) == 40
    protocol = dict(protocol_id='v15_test', schedule=schedule, ids=ids, seeds=seeds, arms={a: {} for a in ARMS},
                    bins={'longbench_v2/q1': '10-15K', 'longbench_v2/q2': '15-20K'})
    summary, cells = summarize(protocol, ledger, gold, lambda r: receipts[r['private_receipt']], boot=200)
    assert summary['meta']['complete'] and summary['per_arm']['CVM_T']['task_correct'] == 4
    assert summary['per_arm']['D_native']['task_correct'] == 2
    p = summary['pairs']['CVM_T/B8_P']
    assert p['decomposition_check'] and abs(p['summed_time_ratio'] - 21 / 20) < 1e-12
    # duplicate, foreign (old AIME) and mis-seeded records are refused, never last-record-wins
    with pytest.raises(ValueError):
        load_records(ledger + [dict(ledger[0])], schedule)
    with pytest.raises(ValueError):
        load_records(ledger + [dict(ledger[0], execution_key='aime:attempt0:0')], schedule)
    with pytest.raises(ValueError):
        load_records([dict(ledger[0], seed=42)], schedule)
    with pytest.raises(RuntimeError):
        validate_resume(events + [dict(event='start', **dict(identity, arm_hashes={}))], identity, schedule)


def test_warm_row_with_new_shared_object_is_rejected():
    from scripts.v15_seed_runs import warm_acceptance
    a0 = dict(ok=True, completion_token_hash='h', per_canvas_calls=[1], termination='eos')
    assert warm_acceptance(a0, dict(a0, triton_misses=0, new_shared_objects=[]))['accepted']
    assert not warm_acceptance(a0, dict(a0, triton_misses=0, new_shared_objects=['/x.so']))['accepted']
    assert not warm_acceptance(a0, dict(a0, triton_misses=1, new_shared_objects=[]))['accepted']
