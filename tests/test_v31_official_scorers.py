"""Toy CPU tests for the v31 official-protocol rule, scorers and comparison tools (no real data, no torch, no fixtures).

Run: python -m pytest tests/test_v31_official_scorers.py  or  python scripts/v31_run_tests.py tests/test_v31_official_scorers.py
(cwd = repo root). The RULER metric functions come from the pinned checkout when RULER_ROOT is set, else from a
stand-in with the same string_match_all / string_match_part definitions.
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import types
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
MAN_SHA = 'a' * 64


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    raise AssertionError(f'{getattr(fn, "__name__", fn)} did not raise {exc.__name__}')


def ruler_metrics():
    root = os.environ.get('RULER_ROOT')
    if root:
        return op.ruler_metrics(root)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'scripts/eval/synthetic'
        path.mkdir(parents=True)
        (path / 'constants.py').write_text(STAND_IN_METRICS)
        return op.ruler_metrics(d)


def make_pool(directory, dataset, rows, max_model_len=4096, manifest_sha=MAN_SHA):
    """A v31_rowinfo_v2 file for toy rows (each: id + fields); prompt_sha256 = sha of a toy token list per row."""
    out = []
    for i, r in enumerate(rows):
        out.append(dict(r, index=i, generation_budget=r.get('generation_budget', 64), prompt_token_count=10,
                        prompt_sha256=op.token_ids_sha256([i, 1, 2])))
    Path(directory, f'{dataset}_rowinfo.json').write_text(json.dumps(dict(schema=op.ROWINFO_SCHEMA, dataset=dataset,
                                                                          manifest_sha256=manifest_sha,
                                                                          max_model_len=max_model_len, rows=out)))


def rec(ds, index, completion, finish='stop', seed=1, repeat=0, budget=64, id_=None, **extra):
    r = dict(dataset=ds, index=index, panel_seed=seed, repeat=repeat, id=id_ or f'{ds}/{index}', completion=completion,
             finish_reason=finish, manifest_sha256=MAN_SHA, prompt_sha256=op.token_ids_sha256([index, 1, 2]), budget=budget,
             rng_seed=100 + index, prompt_tokens=10, max_model_len=4096, chunk=16384, block_size=32)
    r.update(extra)
    return r


def records_of(label, recs):
    return [(label, op.cell_key(r), r) for r in recs]

# ---------------------------------------------------------------- shared rule: keys, binding, plan, comparability


def test_duplicate_cell_keys_raise():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'x_armA.private.jsonl'
        r = rec('ds', 0, 'a')
        f.write_text(json.dumps(r) + '\n' + json.dumps(dict(r, completion='b')) + '\n')
        raises(ValueError, op.read_private, [str(f)])
        g = Path(d) / 'y_armA.private.jsonl'
        f.write_text(json.dumps(r) + '\n')
        g.write_text(json.dumps(r) + '\n')
        raises(ValueError, op.read_private, [str(f), str(g)])
    table = {}
    op.put(table, 'a', 'k', 1)
    raises(ValueError, op.put, table, 'a', 'k', 2)
    assert op.finish_class('stop') == op.finish_class('eos') == 'finished' and op.finish_class('length') == 'capped'
    assert op.label_of('/x/tag_dense_default.private.jsonl') == op.label_of('tag_dense_default.jsonl') == 'dense_default'


def test_prompt_sha256_formula_equals_the_bench():
    src = (ROOT / 'scripts/v31_vllm_paired_bench.py').read_text()
    assert 'prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()' in src
    import hashlib
    ids = [2, 105, 2364, 107]
    assert op.token_ids_sha256(ids) == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    for field in ('MAX_MODEL_LEN', 'max_model_len_source', 'manifest_sha256=man_sha', 'rng_seed=seed, prompt_tokens=len(ids), **bind'):
        assert field in src


def test_bind_refuses_other_pool_prompt_budget_and_unbound_records():
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'ds', [dict(id='ds/0'), dict(id='ds/1')])
        pools = op.Pools(d)
        row, b, ok = op.bind(rec('ds', 1, 'x'), pools)
        assert ok and row['id'] == 'ds/1' and b['rng_seed'] == 101 and b['max_model_len'] == 4096
        raises(ValueError, op.bind, rec('ds', 1, 'x', id_='ds/0'), pools)                      # id at index
        raises(ValueError, op.bind, rec('ds', 1, 'x', manifest_sha256='b' * 64), pools)        # another manifest
        raises(ValueError, op.bind, rec('ds', 1, 'x', prompt_sha256='c' * 64), pools)          # another prompt
        raises(ValueError, op.bind, rec('ds', 1, 'x', budget=65), pools)                      # another budget
        legacy = {k: v for k, v in rec('ds', 1, 'x').items() if k not in op.BINDING_FIELDS}
        raises(ValueError, op.bind, legacy, pools)
        assert op.bind(legacy, pools, legacy=True)[2] is False


def test_plan_intersection_and_coverage():
    with tempfile.TemporaryDirectory() as d:
        cells = Path(d) / 'cells.json'
        cells.write_text(json.dumps([dict(dataset='ds', id=f'ds/{i}', index=i, seed=1) for i in range(3)]))
        plan = op.planned_cells([cells], repeats=2)
        assert len(plan) == 6 and 'ds|2|1|1' in plan
        plan = op.planned_cells([cells])
        full = records_of('a', [rec('ds', i, '') for i in range(3)])
        part = records_of('b', [rec('ds', i, '') for i in range(2)])
        raises(ValueError, op.select_cells, full + part, plan)                                 # b misses a cell
        kept, cov = op.select_cells(full + part, plan, allow_missing=True)
        assert {k for _, k, _ in kept} == {'ds|0|1|0', 'ds|1|1|0'}
        assert cov['a'] == dict(planned=3, present=3, missing=0, dropped=1, scored=2)
        assert cov['b'] == dict(planned=3, present=2, missing=1, dropped=0, scored=2)
        stray = records_of('a', [rec('ds', 7, '')])
        raises(ValueError, op.select_cells, full + stray, plan, True)                          # outside the plan
        raises(ValueError, op.select_cells, records_of('a', [rec('ds', 0, '', id_='other')]), plan, True)


def test_check_comparable():
    b = {f: 1 for f in op.COMPARE_FIELDS}
    binding = {'x': {'k': dict(b)}, 'y': {'k': dict(b)}}
    assert op.check_comparable(binding, ['x', 'y'], ['k']) == 1
    for f in op.COMPARE_FIELDS:
        bad = {'x': {'k': dict(b)}, 'y': {'k': dict(b, **{f: 2})}}
        raises(ValueError, op.check_comparable, bad, ['x', 'y'], ['k'])
    old = {'x': {'k': dict(b, prompt_sha256=None)}, 'y': {'k': dict(b, prompt_sha256=None)}}
    raises(ValueError, op.check_comparable, old, ['x', 'y'], ['k'])
    assert op.check_comparable(old, ['x', 'y'], ['k'], legacy=True) == 1

# ---------------------------------------------------------------- answer text, RULER / LongBench pieces


def test_answer_text_variants_and_protocol_parity():
    final_response, source = op.load_final_response()
    assert source['file'].endswith('protocol.py') and '/' not in source['file'][:1] and len(source['sha256']) == 64
    rng = random.Random(3)
    alphabet = ['a', ' ', '\n', '<channel|>', '<|channel>thought\n', '<turn|>', '<eos>', '<|endoftext|>', 'B']
    for _ in range(3000):
        raw = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        for thinking in (False, True):
            assert op.answer_unstripped(raw, thinking).strip() == final_response(raw, thinking)
    assert op.answer_unstripped('<|channel>thought\n<channel|> RND x \n<turn|>', False) == ' RND x \n'
    assert op.cut_at_end(' 12, <|channel>x<channel|>34<turn|>tail') == ' 12, <|channel>x<channel|>34'
    numeric_score, src = op.aime_numeric_score()
    assert numeric_score('so \\boxed{70}', '70')['correct'] and not numeric_score('\\boxed{71}', '70')['correct']
    assert numeric_score('no digits', '12')['extracted'] is None and src['module'].endswith('aime30.protocol')


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
    assert ''.join(map(chr, ids[n_user:])) == '<O> The answer is' and ''.join(map(chr, ids[1:n_user])) == 'U[context?]M'
    ids, n_user = op.ruler_prompt_ids(FakeAdapter(), 'context?', ' The answer is', '')          # the no-opener ablation
    assert ''.join(map(chr, ids[n_user:])) == ' The answer is'
    assert op.RESPONSE_OPENER['diffusion_gemma'] == '<|channel>thought\n<channel|>' and len(op.RULER_TASKS) == 13


def test_ruler_sample_score_unrounded_and_aggregate():
    m = ruler_metrics()
    assert abs(op.ruler_sample_score(m['freq_words_extraction'], 'a b', ['a', 'b', 'c']) - 200 / 3) < 1e-9   # not 66.67
    assert op.ruler_sample_score(m['qa'], 'Paris', ['paris', 'x']) == 100.0
    fake = lambda preds, refs: 0.0
    fake.__name__ = 'string_match_all'
    raises(AssertionError, op.ruler_sample_score, fake, 'a', ['a'])
    scores = {'L1|0|1|0': 100.0, 'L1|1|1|0': 0.0, 'L1|2|1|0': 0.0, 'L1|3|1|0': 50.0, 'L2|0|1|0': 80.0, 'L2|1|1|0': 40.0}
    task = {'L1|0|1|0': 'a', 'L1|1|1|0': 'a', 'L1|2|1|0': 'a', 'L1|3|1|0': 'cwe', 'L2|0|1|0': 'a', 'L2|1|1|0': 'cwe'}
    agg = op.ruler_official_aggregate(scores, task.get, lambda k: k.split('|')[0])
    assert abs(agg['L1']['avg'] - (100 / 3 + 50) / 2) < 1e-9 and abs(agg['_overall'] - ((100 / 3 + 50) / 2 + 60) / 2) < 1e-9


def test_longbench_official_pieces():
    assert op.lb_extract_answer('**The correct answer is (B)**') == 'B' and op.lb_extract_answer('The correct answer is C.') == 'C'
    assert op.lb_extract_answer('the correct answer is (B)') is None and op.lb_extract_answer('Answer: B') is None
    ids = list(range(120001))
    cut = op.middle_truncate(ids)
    assert len(cut) == 120000 and cut[:60000] == ids[:60000] and cut[60000:] == ids[-60000:]
    item = dict(context=' doc $Q$ ', question=' q? ', choice_A=' a', choice_B='b ', choice_C='c', choice_D='d')
    assert op.lb_fill('<$DOC$>[$Q$]($C_A$|$C_B$|$C_C$|$C_D$)', item) == '<doc q?>[q?](a|b|c|d)'
    rows = [dict(judge=True, difficulty='easy', length='short')] + [dict(judge=False, difficulty='easy', length='short')] * 2
    assert op.lb_result(rows)['Overall'] == 33.3 and abs(op.lb_result(rows, digits=None)['Overall'] - 100 / 3) < 1e-12
    assert op.lb_result([dict(judge=True, difficulty='easy', length='short')])['Hard'] is None

# ---------------------------------------------------------------- scorers


def test_ruler_scorer_prefilled_cut_and_strict_secondary():
    import v31_score_ruler as sr
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'r', [dict(id='r/0', task='cwe', task_base='common_words_extraction', thinking=False, prefilled=True),
                           dict(id='r/1', task='vt', task_base='variable_tracking', thinking=False, prefilled=True),
                           dict(id='r/2', task='qa_1', task_base='qa', thinking=False, prefilled=False)])
        gold = {'r': {'r/0': [f'w{i}' for i in range(10)], 'r/1': ['VAR AAAAA', 'VAR BBBBB'], 'r/2': ['Paris', 'paris city']}}
        final_response, _ = op.load_final_response()
        recs = records_of('arm', [
            rec('r', 0, ' w0, w1, w2,<channel|> w3, w4, w5, w6.<turn|>'),       # prefilled: no split at <channel|>
            rec('r', 1, 'VAR AAAAA VAR BBBBB', finish='length'),                 # complete but capped -> 100 primary
            rec('r', 2, '<|channel>thought\n<channel|> PARIS<turn|>')])         # older pool: final_response
        official, sec, binding, diag = sr.score(recs, ruler_metrics(), {}, op.Pools(d, sr.FIELDS), gold.get, final_response)
        assert official['arm'] == {'r|0|1|0': 70.0, 'r|1|1|0': 100.0, 'r|2|1|0': 100.0}
        assert sec[sr.STRICT]['arm'] == {'r|0|1|0': False, 'r|1|1|0': False, 'r|2|1|0': True}
        assert diag['arm']['capped'] == 1 and diag['arm']['verified'] == 3 and binding['arm']['r|1|1|0']['budget'] == 64
        s = sr.summarize(official, diag, op.Pools(d, sr.FIELDS))['arm']
        assert abs(s['official']['r']['avg'] - 90.0) < 1e-9
        raises(ValueError, sr.score, recs + recs[:1], ruler_metrics(), {}, op.Pools(d, sr.FIELDS), gold.get, final_response)


def test_aime_scorer_exact_without_finish_requirement_and_avg_at_k():
    import v31_score_aime as sa
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'aime', [dict(id='aime/1', source_id='1', thinking=True), dict(id='aime/2', source_id='2', thinking=True)])
        final_response, _ = op.load_final_response()
        numeric_score, _ = op.aime_numeric_score()
        recs = records_of('arm', [rec('aime', 0, '<|channel>thought\n...<channel|>\\boxed{70}', finish='length', id_='aime/1'),
                                  rec('aime', 0, '<|channel>thought\n\\boxed{70} still thinking', seed=2, id_='aime/1'),
                                  rec('aime', 1, '<|channel>t<channel|>The answer is 12<turn|>', id_='aime/2'),
                                  rec('aime', 1, '<|channel>t<channel|>no number', seed=2, id_='aime/2')])
        official, sec, _, diag = sa.score(recs, {'1': '70', '2': '12'}, op.Pools(d, ('source_id', 'thinking')), final_response,
                                          numeric_score)
        assert official['arm'] == {'aime|0|1|0': True, 'aime|0|2|0': False, 'aime|1|1|0': True, 'aime|1|2|0': False}
        assert sec[sa.SECONDARY]['arm']['aime|0|1|0'] is False and diag['arm']['no_final_response'] == 1
        s = sa.summarize(official, diag)['arm']
        assert s['avg_at_k'] == 50.0 and s['k_per_problem'] == [2] and s['accuracy_by_panel_seed'] == {'1': 100.0, '2': 0.0}


def test_mrcr_scorer_grades_the_unstripped_response():
    import v31_score_mrcr as sm
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'm', [dict(id='m/0', thinking=False), dict(id='m/1', thinking=False)])
        gold = {'m': {'m/0': dict(answer='RNDhello world', random_string_to_prepend='RND'),
                      'm/1': dict(answer='RNDabc', random_string_to_prepend='RND')}}
        recs = records_of('arm', [rec('m', 0, '<|channel>thought\n<channel|>RNDhello world<turn|>', finish='length'),
                                  rec('m', 1, '<|channel>thought\n<channel|> RNDabc<turn|>')])   # leading space: README gives 0
        official, sec, _, diag = sm.score(recs, op.Pools(d, ('thinking',)), gold.get)
        assert official['arm'] == {'m|0|1|0': 1.0, 'm|1|1|0': 0.0} and diag['arm']['missing_prefix'] == 1
        assert sec[sm.SECONDARY]['arm'] == {'m|0|1|0': False, 'm|1|1|0': False}
        assert sm.summarize(official, diag)['arm']['bins']['m'] == dict(mean_ratio=0.5, cells=2)


def test_graphwalks_scorer_unstripped_last_line():
    import v31_score_graphwalks as sg
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'g', [dict(id='g/0', problem_type='bfs', thinking=True), dict(id='g/1', problem_type='parents', thinking=True),
                           dict(id='g/2', problem_type='bfs', thinking=True)])
        gold = {'g': {'g/0': ['a', 'b'], 'g/1': ['c'], 'g/2': ['d']}}
        recs = records_of('arm', [rec('g', 0, '<|channel>t<channel|>reasoning\nFinal Answer: [a, x]<turn|>', finish='length'),
                                  rec('g', 1, '<|channel>t<channel|>Final Answer: [c]\nthanks'),
                                  rec('g', 2, '<|channel>t<channel|>Final Answer: [d]\n<turn|>')])   # trailing newline: unparsed
        official, _, _, diag, types = sg.score(recs, op.Pools(d, ('problem_type', 'thinking')), gold.get)
        assert official['arm'] == {'g|0|1|0': 0.5, 'g|1|1|0': 0.0, 'g|2|1|0': 0.0} and diag['arm']['unparsed'] == 2
        s = sg.summarize(official, diag, types)['arm']
        assert s['bins']['g']['bfs']['mean_f1'] == 0.25 and s['bins']['g']['all']['cells'] == 3


def test_longbench_official_scorer_runs_unrounded_mean():
    import v31_score_longbench_official as sl
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'lb', [dict(id=f'lb/{i}', difficulty='easy', length='short', thinking=False, truncated=i == 2) for i in range(3)])
        gold = {'lb': {'lb/0': 'B', 'lb/1': 'C', 'lb/2': 'A'}}
        final_response, _ = op.load_final_response()
        recs = records_of('arm', [rec('lb', 0, 'The correct answer is (B)', finish='length'), rec('lb', 1, 'I think C'),
                                  rec('lb', 2, 'The correct answer is (B)'), rec('lb', 0, 'The correct answer is (B)', seed=2),
                                  rec('lb', 1, 'The correct answer is C', seed=2), rec('lb', 2, 'The correct answer is (B)', seed=2)])
        official, sec, _, diag, meta = sl.score(recs, op.Pools(d, sl.FIELDS), gold.get, final_response)
        assert sum(official['arm'].values()) == 3 and sec[sl.SECONDARY]['arm']['lb|0|1|0'] is False
        assert diag[('arm', 'lb')]['unparsed'] == 1 and diag[('arm', 'lb')]['truncated_items'] == 2
        s = sl.summarize(official, diag, meta)['arm']['lb']
        assert s['runs']['seed1_repeat0']['Overall'] == 33.3 and s['runs']['seed2_repeat0']['Overall'] == 66.7
        assert s['mean_over_runs']['Overall'] == 50.0                       # mean of 33.33.. and 66.66.. (rounded: 50.0)


def test_humaneval_scorer_timeouts_rerun_serially():
    import v31_score_humaneval as sh
    calls = []

    def run_test(program):
        calls.append(program)
        if program.endswith('slow') and calls.count(program) == 1:
            return False, 'timeout'
        return program.endswith(('ok', 'slow')), 'passed'
    he = types.SimpleNamespace(extract_code=lambda raw, ep: (raw, 'parsed') if raw.startswith('def') else (None, 'no_final_response'),
                               program_for=lambda problem, code: code, run_test=run_test)
    with tempfile.TemporaryDirectory() as d:
        make_pool(d, 'humaneval', [dict(id='humaneval/0', thinking=True), dict(id='humaneval/1', thinking=True)])
        gold = {'humaneval/0': dict(entry_point='f'), 'humaneval/1': dict(entry_point='g')}
        recs = records_of('arm', [rec('humaneval', 0, 'def f ok', finish='length', id_='humaneval/0'),
                                  rec('humaneval', 0, 'def f slow', seed=2, id_='humaneval/0'),
                                  rec('humaneval', 1, 'nothing', id_='humaneval/1')])
        official, sec, _, diag, timeouts = sh.score(recs, gold, he, op.Pools(d, ('thinking',)), workers=2)
        assert official['arm'] == {'humaneval|0|1|0': True, 'humaneval|0|2|0': True, 'humaneval|1|1|0': False}
        assert timeouts == 1 and diag['arm']['timeouts_after_rerun'] == 0 and sec[sh.SECONDARY]['arm']['humaneval|0|1|0'] is False
        s = sh.summarize(official, diag, {(l, k): r['id'] for l, k, r in recs})['arm']
        assert s['pass_at_1'] == 50.0 and s['samples_per_task'] == [1, 2] and s['verified'] == 3

# ---------------------------------------------------------------- comparison tools


def run_main(module, argv):
    out, old = io.StringIO(), sys.argv
    sys.argv = ['x', *map(str, argv)]
    try:
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = old
    return out.getvalue()


def paired_fixture(d, kind_rows, values, n_seeds=1, mismatch=None):
    """rowinfo for dataset 'ds', cells (seeds 1..n), official values {label: {key: v}} and a comparable binding."""
    make_pool(d, 'ds', kind_rows)
    cells = [dict(dataset='ds', id=r['id'], index=i, seed=s) for s in range(1, n_seeds + 1) for i, r in enumerate(kind_rows)]
    Path(d, 'cells.json').write_text(json.dumps(cells))
    Path(d, 'off.json').write_text(json.dumps(values))
    b = {f: 1 for f in op.COMPARE_FIELDS}
    binding = {l: {k: dict(b) for k in v} for l, v in values.items()}
    if mismatch:
        label, key, field = mismatch
        binding[label][key][field] = 2
    Path(d, 'bind.json').write_text(json.dumps(binding))


def test_paired_tool_boolean_mcnemar_and_breakdown():
    import v31_paired_official as pt
    with tempfile.TemporaryDirectory() as d:
        rows = [dict(id=f'ds/{i}', difficulty='easy' if i < 2 else 'hard', length='short') for i in range(4)]
        ref = {f'ds|{i}|1|0': v for i, v in enumerate([True, True, False, False])}
        arm = {f'ds|{i}|1|0': v for i, v in enumerate([True, False, True, True])}
        paired_fixture(d, rows, dict(ref=ref, arm=arm))
        text = run_main(pt, ['longbench', d, 'ref', Path(d, 'off.json'), Path(d, 'bind.json'), '--cells', Path(d, 'cells.json')])
        line = next(x for x in text.splitlines() if x.startswith('| arm | ds |'))
        cols = [c.strip() for c in line.strip('|').split('|')]
        assert cols[2] == '75.0' and cols[3] == '50.0' and cols[4].startswith('+25.00') and cols[5].startswith('2/1')
        assert 'planned 4 present 4 dropped 0 scored 4' in text
        bd = next(x for x in text.splitlines() if x.startswith('| arm | ds | 75.0 (+25.0)'))
        assert '| 50.0 (-50.0) | 100.0 (+100.0) | 75.0 (+25.0) | - | - |' in bd   # Easy, Hard, Short, Medium, Long
        paired_fixture(d, rows, dict(ref=ref, arm=arm), mismatch=('arm', 'ds|1|1|0', 'max_model_len'))
        raises(ValueError, run_main, pt, ['longbench', d, 'ref', Path(d, 'off.json'), Path(d, 'bind.json'), '--cells',
                                          Path(d, 'cells.json')])
        paired_fixture(d, rows, dict(ref=ref, arm={k: v for k, v in arm.items() if k != 'ds|3|1|0'}))
        raises(ValueError, run_main, pt, ['longbench', d, 'ref', Path(d, 'off.json'), Path(d, 'bind.json'), '--cells',
                                          Path(d, 'cells.json')])
        text = run_main(pt, ['longbench', d, 'ref', Path(d, 'off.json'), Path(d, 'bind.json'), '--cells', Path(d, 'cells.json'),
                             '--allow-missing', '--only', 'nothing'])
        assert 'ref: planned 4 present 4 dropped 1 scored 3' in text           # --only does not change the intersection


def test_paired_tool_aime_nested_seeds_and_continuous_kinds():
    import v31_paired_official as pt
    with tempfile.TemporaryDirectory() as d:
        rows = [dict(id=f'ds/{i}', problem_type='bfs' if i else 'parents') for i in range(2)]
        ref = {f'ds|{i}|{s}|0': False for i in range(2) for s in (1, 2, 3, 4)}
        arm = {f'ds|{i}|{s}|0': (i == 0 and s <= 2) for i in range(2) for s in (1, 2, 3, 4)}
        paired_fixture(d, rows, dict(ref=ref, arm=arm), n_seeds=4)
        text = run_main(pt, ['aime', d, 'ref', Path(d, 'off.json'), Path(d, 'bind.json'), '--cells', Path(d, 'cells.json')])
        cols = [c.strip() for c in next(x for x in text.splitlines() if x.startswith('| arm | ds |')).strip('|').split('|')]
        assert cols[2] == '25.0' and cols[4].startswith('+25.00') and cols[5].startswith('2/0') and cols[6:] == ['8', '2']
        f1_ref = {f'ds|{i}|1|0': 0.5 for i in range(2)}
        f1_arm = {'ds|0|1|0': 1.0, 'ds|1|1|0': 0.25}
        paired_fixture(d, rows, dict(ref=f1_ref, arm=f1_arm))
        text = run_main(pt, ['graphwalks', d, 'ref', Path(d, 'off.json'), Path(d, 'bind.json'), '--cells', Path(d, 'cells.json')])
        assert '| arm | ds | 0.6250 | 0.5000 | +0.1250' in text and '| arm | ds bfs | 0.2500 | 0.5000 | -0.2500' in text
        assert '1/1 (p=1.000)' in text


def test_ruler_table_thirteen_tasks_and_binding():
    import v31_ruler_official_table as tb
    with tempfile.TemporaryDirectory() as d:
        tasks = list(op.RULER_TASKS)
        for ln in ('L1', 'L2'):
            make_pool(d, ln, [dict(id=f'{ln}/{i}', task=t) for i, t in enumerate(tasks + ['cwe'])])
        cells = [dict(dataset=ln, id=f'{ln}/{i}', index=i, seed=1) for ln in ('L1', 'L2') for i in range(14)]
        Path(d, 'cells.json').write_text(json.dumps(cells))
        keys = [f"{c['dataset']}|{c['index']}|1|0" for c in cells]
        ref = {k: 50.0 for k in keys}
        arm = {k: (100.0 if k == 'L1|13|1|0' else 50.0) for k in keys}                     # one of two cwe cells of L1
        Path(d, 'off.json').write_text(json.dumps(dict(ref=ref, armx=arm)))
        b = {f: 1 for f in op.COMPARE_FIELDS}
        Path(d, 'bind.json').write_text(json.dumps({l: {k: dict(b) for k in keys} for l in ('ref', 'armx')}))
        Path(d, 'strict.json').write_text(json.dumps({l: {k: v == 100 for k, v in t.items()} for l, t in (('ref', ref), ('armx', arm))}))
        argv = [d, 'ref', Path(d, 'off.json'), '--binding', Path(d, 'bind.json'), '--cells', Path(d, 'cells.json'),
                '--secondary', Path(d, 'strict.json'), '--secondary-out', Path(d, 's.md')]
        text = run_main(tb, argv)
        table = [x for x in text.splitlines() if x.startswith('|')]
        assert not any('strict' in x.lower() for x in table) and '13 tasks per length asserted' in text
        cols = [c.strip() for c in next(x for x in table if x.startswith('| armx |')).strip('|').split('|')]
        # L1 cwe mean (100 + 50) / 2 = 75 -> L1 avg = (12 * 50 + 75) / 13; overall = (that + 50) / 2
        assert cols[1] == f'{(12 * 50 + 75) / 13:.1f}' and cols[3] == f'{((12 * 50 + 75) / 13 + 50) / 2:.1f}'
        assert cols[6].startswith(f'{(25 / 13) / 2:+.2f}') and '| armx | 1 | 0 | 1/28 | 0/1 | 1.000 |' in Path(d, 's.md').read_text()
        missing_task = {l: {k: v for k, v in t.items() if not k.startswith('L2|8|')} for l, t in (('ref', ref), ('armx', arm))}
        Path(d, 'off.json').write_text(json.dumps(missing_task))
        raises(ValueError, run_main, tb, argv[:5] + argv[5:7] + ['--allow-missing'])        # vt missing at L2: refused
        Path(d, 'off.json').write_text(json.dumps(dict(ref=ref, armx=arm)))
        bad = {l: {k: dict(b, chunk=2 if (l == 'armx' and k == keys[0]) else 1) for k in keys} for l in ('ref', 'armx')}
        Path(d, 'bind.json').write_text(json.dumps(bad))
        raises(ValueError, run_main, tb, argv[:7])


def test_perf_breakdown_refuses_different_settings():
    import v31_perf_breakdown as pb
    base = dict(dataset='ds', index=0, panel_seed=1, repeat=0, rng_seed=5, budget=64, prompt_tokens=10, max_model_len=4096,
                chunk=16384, block_size=32)
    pb.check_pairs('a', 'r', [(dict(base), dict(base))])
    for f in ('rng_seed', 'budget', 'prompt_tokens', 'max_model_len', 'chunk', 'block_size'):
        raises(ValueError, pb.check_pairs, 'a', 'r', [(dict(base, **{f: 7}), dict(base))])
    raises(ValueError, pb.check_pairs, 'a', 'r', [(dict(base, prompt_sha256='x'), dict(base))])


def test_cap_rate_from_public_records():
    import v31_cap_rate as cr
    with tempfile.TemporaryDirectory() as d:
        recs = [dict(dataset='g90', index=i, repeat=0, finish_reason='length' if i < 3 else 'stop', output_tokens=100 * (i + 1),
                     budget=16384) for i in range(4)] + [dict(dataset='g90', index=0, repeat=-1, finish_reason='stop',
                                                              output_tokens=1, budget=16384)]
        f = Path(d, 'pilot_dense_default_fix.jsonl')
        f.write_text(''.join(json.dumps(r) + '\n' for r in recs))
        text = run_main(cr, [f])
        assert '| dense_default_fix | g90 | 4 | 3 | 0.750 | 16384 | 250 / 400 / 400 |' in text
