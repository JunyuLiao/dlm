"""Toy CPU tests for the v31 official-protocol rule and scorers (no real data, no torch, no pytest fixtures).

Run: python -m pytest tests/test_v31_official_scorers.py  or  python scripts/v31_run_tests.py tests/test_v31_official_scorers.py
(cwd = repo root). The RULER metric functions come from the pinned checkout when RULER_ROOT is set, else from a
stand-in with the same string_match_all / string_match_part definitions.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

import v31_official as op  # noqa: E402

STAND_IN_METRICS = '''
def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

TASKS = {'niah': {'metric_fn': string_match_all}, 'variable_tracking': {'metric_fn': string_match_all},
         'common_words_extraction': {'metric_fn': string_match_all}, 'freq_words_extraction': {'metric_fn': string_match_all},
         'qa': {'metric_fn': string_match_part}}
'''


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def rec(ds, index, completion, finish='stop', seed=1, repeat=0, id_=None):
    return dict(dataset=ds, index=index, panel_seed=seed, repeat=repeat, id=id_ or f'{ds}/{index}', completion=completion,
                finish_reason=finish)


def records_of(label, recs):
    return [(label, op.cell_key(r), r) for r in recs]


def ruler_metrics():
    root = os.environ.get('RULER_ROOT')
    if root:
        return op.ruler_metrics(root)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'scripts/eval/synthetic'
        path.mkdir(parents=True)
        (path / 'constants.py').write_text(STAND_IN_METRICS)
        return op.ruler_metrics(d)

# ---------------------------------------------------------------- shared rule


def test_duplicate_cell_keys_raise():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'x_armA.private.jsonl'
        r = rec('ds', 0, 'a')
        f.write_text(json.dumps(r) + '\n' + json.dumps(dict(r, completion='b')) + '\n')
        raises(ValueError, op.read_private, [str(f)])
        g = Path(d) / 'y_armA.private.jsonl'
        f.write_text(json.dumps(r) + '\n')
        g.write_text(json.dumps(r) + '\n')
        raises(ValueError, op.read_private, [str(f), str(g)])           # same arm, same cell, two files
        h = Path(d) / 'z_armB.private.jsonl'
        h.write_text(json.dumps(r) + '\n')
        assert [label for label, _, _ in op.read_private([str(f), str(h)])] == ['armA', 'armB']
    table = {}
    op.put(table, 'a', 'k', 1)
    raises(ValueError, op.put, table, 'a', 'k', 2)
    assert op.finish_class('stop') == op.finish_class('eos') == 'finished' and op.finish_class('length') == 'capped'


def test_manifest_index_must_match_id():
    rows = [dict(id='ds/a'), dict(id='ds/b')]
    assert op.check_cell(rows, 'ds', dict(rec('ds', 1, ''), id='ds/b'))['id'] == 'ds/b'
    raises(ValueError, op.check_cell, rows, 'ds', dict(rec('ds', 1, ''), id='ds/a'))


def test_final_response_and_aime_extraction_compiled_from_protocol_sources():
    final_response, source = op.load_final_response()
    assert source['path'].endswith('protocol.py') and len(source['sha256']) == 64
    assert final_response('<|channel>thought\n<channel|>Answer<turn|>junk', False) == 'Answer'
    assert final_response(' 123, 456.<turn|>', False) == '123, 456.'            # continuation after a prefilled prefix
    assert final_response('<|channel>thought\nstill thinking', True) == ''
    assert final_response('<|channel>thought\nx\n<channel|>A<channel|> B <eos>', True) == 'B'
    numeric_score, _ = op.aime_numeric_score()
    assert numeric_score('so \\boxed{70}', '70')['correct']
    assert not numeric_score('so \\boxed{71}', '70')['correct']
    assert numeric_score('The answer is: 12', '12')['correct']
    assert numeric_score('no digits', '12')['extracted'] is None


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FakeAdapter:
    tokenizer = FakeTokenizer()

    def encode_prompt(self, text, extra):
        assert extra == {'thinking': False}
        return [2] + [ord(c) for c in f'U[{text}]M']


def test_ruler_prompt_puts_prefix_after_the_model_turn_opener():
    ids, n_user = op.ruler_prompt_ids(FakeAdapter(), 'context?', ' The answer is', '<O>')
    assert ids[:n_user] == FakeAdapter().encode_prompt('context?', {'thinking': False})
    assert ''.join(map(chr, ids[n_user:])) == '<O> The answer is'           # verbatim prefix, leading space kept
    assert ''.join(map(chr, ids[1:n_user])) == 'U[context?]M'                 # the user turn holds no prefix
    assert op.RESPONSE_OPENER['diffusion_gemma'] == '<|channel>thought\n<channel|>'


def test_ruler_official_aggregate_weights_tasks_equally_per_length():
    scores = {'L1|0|1|0': 100.0, 'L1|1|1|0': 0.0, 'L1|2|1|0': 0.0, 'L1|3|1|0': 50.0,   # task a: 3 cells, task b: 1 cell
              'L2|0|1|0': 80.0, 'L2|1|1|0': 40.0}
    task = {'L1|0|1|0': 'a', 'L1|1|1|0': 'a', 'L1|2|1|0': 'a', 'L1|3|1|0': 'cwe', 'L2|0|1|0': 'a', 'L2|1|1|0': 'cwe'}
    agg = op.ruler_official_aggregate(scores, task.get, lambda k: k.split('|')[0])
    assert abs(agg['L1']['avg'] - (100 / 3 + 50) / 2) < 1e-9                  # not the cell mean 37.5
    assert agg['L1']['cwe'] == 50 and abs(agg['L1']['without_cwe'] - 100 / 3) < 1e-9
    assert agg['L2']['avg'] == 60 and abs(agg['_overall'] - ((100 / 3 + 50) / 2 + 60) / 2) < 1e-9


def test_longbench_official_pieces():
    assert op.lb_extract_answer('**The correct answer is (B)**') == 'B'
    assert op.lb_extract_answer('The correct answer is C.') == 'C'
    assert op.lb_extract_answer('the correct answer is (B)') is None             # case-sensitive, as pred.py
    assert op.lb_extract_answer('Answer: B') is None
    ids = list(range(120001))
    cut = op.middle_truncate(ids)
    assert len(cut) == 120000 and cut[:60000] == ids[:60000] and cut[60000:] == ids[-60000:]
    assert op.middle_truncate(list(range(120000))) == list(range(120000))
    item = dict(context=' doc $Q$ ', question=' q? ', choice_A=' a', choice_B='b ', choice_C='c', choice_D='d')
    filled = op.lb_fill('<$DOC$>[$Q$]($C_A$|$C_B$|$C_C$|$C_D$)', item)
    assert filled == '<doc q?>[q?](a|b|c|d)'                                  # stripped fields, pred.py replace order
    rows = ([dict(judge=True, difficulty='easy', length='short')] * 2 + [dict(judge=False, difficulty='hard', length='long')]
            + [dict(judge=True, difficulty='hard', length='medium')])
    assert op.lb_result(rows) == dict(Overall=75.0, Easy=100.0, Hard=50.0, Short=100.0, Medium=100.0, Long=0.0)
    assert op.lb_result([dict(judge=True, difficulty='easy', length='short')])['Hard'] is None
    assert op.lb_result([dict(judge=True, difficulty='easy', length='short')] + [dict(judge=False, difficulty='easy', length='short')] * 2)['Overall'] == 33.3

# ---------------------------------------------------------------- RULER scorer


def test_ruler_scorer_official_primary_strict_secondary():
    import v31_score_ruler as sr
    rows = {'r32': [dict(id='r32/0', task='cwe', task_base='common_words_extraction', thinking=False),
                    dict(id='r32/1', task='vt', task_base='variable_tracking', thinking=False),
                    dict(id='r32/2', task='qa_1', task_base='qa', thinking=False)]}
    gold = {'r32': {'r32/0': [f'w{i}' for i in range(10)], 'r32/1': ['VAR AAAAA', 'VAR BBBBB'], 'r32/2': ['Paris', 'paris city']}}
    final_response, _ = op.load_final_response()
    recs = records_of('arm', [
        rec('r32', 0, ' w0, w1, w2, w3, w4, w5, w6.<turn|>'),                     # 7 of 10 words -> 70 (partial credit)
        rec('r32', 1, 'VAR AAAAA VAR BBBBB', finish='length'),                   # complete but capped -> 100 primary
        rec('r32', 2, '<|channel>thought\n<channel|> PARIS<turn|>'),            # an own empty thought block is cut
    ])
    official, secondary, diag = sr.score(recs, ruler_metrics(), {}, rows.get, gold.get, final_response)
    o = official['arm']
    assert o['r32|0|1|0'] == 70.0 and o['r32|1|1|0'] == 100.0 and o['r32|2|1|0'] == 100.0
    s = secondary[sr.STRICT]['arm']
    assert s == {'r32|0|1|0': False, 'r32|1|1|0': False, 'r32|2|1|0': True}    # strict needs 100 AND a finish
    assert diag['arm']['capped'] == 1 and diag['arm']['cells'] == 3 and diag['arm']['nulls'] == 0
    summary = sr.summarize(official, diag, rows.get)['arm']
    assert abs(summary['official']['r32']['avg'] - 90.0) < 1e-9 and summary['official']['_overall'] == summary['official']['r32']['avg']
    raises(ValueError, sr.score, recs + recs[:1], ruler_metrics(), {}, rows.get, gold.get, final_response)
    bad = records_of('arm', [dict(rec('r32', 0, 'x'), id='r32/2')])
    raises(ValueError, sr.score, bad, ruler_metrics(), {}, rows.get, gold.get, final_response)

# ---------------------------------------------------------------- AIME / MRCR / GraphWalks / LongBench


def test_aime_scorer_exact_without_finish_requirement_and_avg_at_k():
    import v31_score_aime as sa
    rows = {'aime26': [dict(id='aime26/1', source_id='1', thinking=True), dict(id='aime26/2', source_id='2', thinking=True)]}
    answers = {'1': '70', '2': '12'}
    final_response, _ = op.load_final_response()
    numeric_score, _ = op.aime_numeric_score()
    recs = records_of('arm', [
        rec('aime26', 0, '<|channel>thought\n...<channel|>\\boxed{70}', finish='length', id_='aime26/1'),
        rec('aime26', 0, '<|channel>thought\n\\boxed{70} still thinking', seed=2, id_='aime26/1'),   # unfinished thought
        rec('aime26', 1, '<|channel>t<channel|>The answer is 12<turn|>', id_='aime26/2'),
        rec('aime26', 1, '<|channel>t<channel|>no number', seed=2, id_='aime26/2'),
    ])
    official, secondary, diag = sa.score(recs, answers, rows.get, final_response, numeric_score)
    assert official['arm'] == {'aime26|0|1|0': True, 'aime26|0|2|0': False, 'aime26|1|1|0': True, 'aime26|1|2|0': False}
    assert secondary[sa.SECONDARY]['arm']['aime26|0|1|0'] is False                # capped: secondary only
    d = diag['arm']
    assert d['capped'] == 1 and d['no_final_response'] == 1 and d['unparsed'] == 2
    s = sa.summarize(official, diag)['arm']
    assert s['avg_at_k'] == 50.0 and s['k_per_problem'] == [2] and s['accuracy_by_panel_seed'] == {'1': 100.0, '2': 0.0}


def test_mrcr_scorer_ratio_primary():
    import v31_score_mrcr as sm
    rows = {'m32': [dict(id='m32/0', thinking=False), dict(id='m32/1', thinking=False)]}
    gold = {'m32': {'m32/0': dict(answer='RNDhello world', random_string_to_prepend='RND'),
                    'm32/1': dict(answer='RNDabc', random_string_to_prepend='RND')}}
    final_response, _ = op.load_final_response()
    recs = records_of('arm', [rec('m32', 0, '<|channel>thought\n<channel|>RNDhello world<turn|>', finish='length'),
                              rec('m32', 1, 'abc<turn|>')])
    official, secondary, diag = sm.score(recs, rows.get, gold.get, final_response)
    assert official['arm'] == {'m32|0|1|0': 1.0, 'm32|1|1|0': 0.0}
    assert secondary[sm.SECONDARY]['arm'] == {'m32|0|1|0': False, 'm32|1|1|0': False}
    assert diag['arm']['missing_prefix'] == 1 and diag['arm']['capped'] == 1
    s = sm.summarize(official, diag)['arm']
    assert s['bins']['m32'] == dict(mean_ratio=0.5, cells=2) and s['mean_over_bins'] == 0.5
    assert abs(sm.grade('RNDhello word', 'RNDhello world', 'RND') - 2 * 10 / 21) < 1e-9


def test_graphwalks_scorer_f1_primary_per_bin_per_type():
    import v31_score_graphwalks as sg
    rows = {'g22': [dict(id='g22/0', problem_type='bfs', thinking=True), dict(id='g22/1', problem_type='parents', thinking=True)]}
    gold = {'g22': {'g22/0': ['a', 'b'], 'g22/1': ['c']}}
    final_response, _ = op.load_final_response()
    recs = records_of('arm', [rec('g22', 0, '<|channel>t<channel|>reasoning\nFinal Answer: [a, x]<turn|>', finish='length'),
                              rec('g22', 1, '<|channel>t<channel|>Final Answer: [c]\nthanks')])
    official, secondary, diag, types = sg.score(recs, rows.get, gold.get, final_response)
    assert official['arm'] == {'g22|0|1|0': 0.5, 'g22|1|1|0': 0.0}            # last line lacks 'Final Answer:' -> 0
    assert diag['arm']['unparsed'] == 1 and diag['arm']['capped'] == 1
    s = sg.summarize(official, diag, types)['arm']
    assert s['bins']['g22']['bfs']['mean_f1'] == 0.5 and s['bins']['g22']['parents']['mean_f1'] == 0.0
    assert s['bins']['g22']['all'] == dict(mean_f1=0.25, cells=2)


def test_longbench_official_scorer_runs_and_columns():
    import v31_score_longbench_official as sl
    rows = {'lb': [dict(id='lb/0', difficulty='easy', length='short', thinking=False, truncated=False),
                   dict(id='lb/1', difficulty='hard', length='long', thinking=False, truncated=True)]}
    gold = {'lb': {'lb/0': 'B', 'lb/1': 'C'}}
    final_response, _ = op.load_final_response()
    recs = records_of('arm', [rec('lb', 0, '<|channel>thought\n<channel|>The correct answer is (B)<turn|>', finish='length'),
                              rec('lb', 1, 'I think C'),
                              rec('lb', 0, 'The correct answer is (A)', seed=2),
                              rec('lb', 1, 'The correct answer is C', seed=2)])
    official, secondary, diag, meta = sl.score(recs, rows.get, gold.get, final_response)
    assert official['arm'] == {'lb|0|1|0': True, 'lb|1|1|0': False, 'lb|0|2|0': False, 'lb|1|2|0': True}
    assert secondary[sl.SECONDARY]['arm']['lb|0|1|0'] is False
    d = diag[('arm', 'lb')]
    assert d['unparsed'] == 1 and d['capped'] == 1 and d['truncated_items'] == 2
    s = sl.summarize(official, diag, meta)['arm']['lb']
    assert s['runs']['seed1_repeat0'] == dict(Overall=50.0, Easy=100.0, Hard=0.0, Short=100.0, Medium=None, Long=0.0, items=2)
    assert s['mean_over_runs']['Overall'] == 50.0 and 'Medium' not in s['mean_over_runs']

# ---------------------------------------------------------------- HumanEval scorer plumbing (sandbox: test_v27_humaneval)


def test_humaneval_scorer_pass_without_finish_requirement():
    import types
    import v31_score_humaneval as sh
    he = types.SimpleNamespace(
        extract_code=lambda raw, ep: (raw, 'parsed') if raw.startswith('def') else (None, 'no_final_response'),
        program_for=lambda problem, code: code, run_test=lambda program: (program.endswith('ok'), ''))
    gold = {'humaneval/0': dict(entry_point='f'), 'humaneval/1': dict(entry_point='g')}
    recs = records_of('arm', [rec('humaneval', 0, 'def f ok', finish='length', id_='humaneval/0'),
                              rec('humaneval', 0, 'def f bad', seed=2, id_='humaneval/0'),
                              rec('humaneval', 1, 'nothing', id_='humaneval/1')])
    official, secondary, diag = sh.score(recs, gold, he, workers=2)
    assert official['arm'] == {'humaneval|0|1|0': True, 'humaneval|0|2|0': False, 'humaneval|1|1|0': False}
    assert secondary[sh.SECONDARY]['arm']['humaneval|0|1|0'] is False
    id_of = {(l, k): r['id'] for l, k, r in recs}
    s = sh.summarize(official, diag, id_of)['arm']
    assert s['pass_at_1'] == 25.0 and s['samples_per_task'] == [1, 2] and s['extraction_no_final_response'] == 1

# ---------------------------------------------------------------- RULER official table


def test_ruler_table_task_weighted_and_strict_only_in_secondary_file():
    import v31_ruler_official_table as tb
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'L1_rowinfo.json').write_text(json.dumps([dict(id=f'L1/{i}', task=t) for i, t in enumerate(['a', 'a', 'a', 'cwe'])]))
        (d / 'L2_rowinfo.json').write_text(json.dumps([dict(id=f'L2/{i}', task=t) for i, t in enumerate(['a', 'cwe'])]))
        keys = ['L1|0|1|0', 'L1|1|1|0', 'L1|2|1|0', 'L1|3|1|0', 'L2|0|1|0', 'L2|1|1|0']
        ref = dict(zip(keys, [100.0, 0.0, 0.0, 50.0, 80.0, 40.0]))
        arm = dict(zip(keys, [100.0, 100.0, 0.0, 70.0, 80.0, 40.0]))
        (d / 'off.json').write_text(json.dumps(dict(ref=ref, armx=arm)))
        (d / 'strict.json').write_text(json.dumps(dict(ref={k: v == 100 for k, v in ref.items()},
                                                       armx={k: v == 100 for k, v in arm.items()})))
        out = io.StringIO()
        argv = sys.argv
        sys.argv = ['x', str(d), 'ref', str(d / 'off.json'), '--secondary', str(d / 'strict.json'), '--secondary-out', str(d / 's.md')]
        try:
            with contextlib.redirect_stdout(out):
                tb.main()
        finally:
            sys.argv = argv
        table = [x for x in out.getvalue().splitlines() if x.startswith('|')]
        assert table and not any('strict' in x.lower() for x in table)          # the boolean is not in the main table
        cols = [c.strip() for c in next(x for x in table if x.startswith('| armx |')).strip('|').split('|')]
        # L1: a = (100+100+0)/3, cwe = 70 -> (66.67 + 70) / 2; L2 unchanged 60; diff = ((66.67+70)/2 - (33.33+50)/2) / 2
        assert cols[1] == '68.3' and cols[2] == '60.0' and cols[3] == '64.2'
        expect = ((200 / 3 + 70) / 2 - (100 / 3 + 50) / 2) / 2
        assert cols[6].startswith(f'{expect:+.2f}')
        secondary = (d / 's.md').read_text()
        assert 'strict all-correct' in secondary and '| armx | 2 | 0 | 2/6 | 0/1 | 1.000 |' in secondary
        assert '| ref | 1 | 0 | 1/6 | - | - |' in secondary
