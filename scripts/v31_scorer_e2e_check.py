"""End-to-end check of the v31 official scorers on the REAL pool files with SYNTHETIC completions (CPU only).

For each official pool it writes private synthetic completion files carrying the binding fields of the bench (they
contain gold text: keep PRIVATE_DIR private), runs the scorer CLI on them with the planned cells and checks the
aggregates:
  perfect          the gold answer in the official answer format, finish 'stop'      -> metric at its maximum
  cappedperfect    the same text, finish 'length' (no end token)                      -> primary unchanged, secondary False
  empty            an empty final response                                             -> metric 0, counted as null / unparsed
and the refusals: a record whose prompt_sha256 / budget / manifest_sha256 differs from the pool, a missing planned cell
(without --allow-missing) and a record outside the plan must each make the scorer fail. Then the paired tool / RULER
table run on the same outputs. Pools: pools_v31_official/{ruler_v33ofc, ruler_v33noop, longbench_v2_ofc, aime26_b32k,
graphwalks_b16k, mrcr_ofc}. Prints aggregates only.
usage: python v31_scorer_e2e_check.py PRIVATE_DIR   (cwd = a deployment holding experiments/, PYTHONPATH=src:.)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path('/media/volume/dllm-1/dyh/pools_v31_official')
RULER_ROOT = '/media/volume/dllm-1/dyh/ruler_scoring/ruler_long_v27/ruler_repo'
AIME = '/home/exouser/.cache/huggingface/hub/datasets--math-ai--aime26/snapshots/79037aebdb6580008fb960d17cb21fd3099083e3/aime2026.jsonl'
LB = ('longbench_v2_0shot', 'longbench_v2_0shot_think', 'longbench_v2_cot_think')


def cells_of(path):
    return json.loads(Path(path).read_text())


def rowinfo(pool_dir, ds):
    return json.loads((pool_dir / f'{ds}_rowinfo.json').read_text())


def records(pool_dir, cells, body, thinking, finish, tamper=None):
    infos = {}
    out = []
    for c in cells:
        info = infos.setdefault(c['dataset'], rowinfo(pool_dir, c['dataset']))
        row = info['rows'][c['index']]
        head = '<|channel>thought\nsynthetic\n<channel|>' if thinking else ''
        r = dict(dataset=c['dataset'], index=c['index'], panel_seed=c['seed'], repeat=0, id=c['id'],
                 completion=head + body(c) + ('<turn|>' if finish == 'stop' else ''), finish_reason=finish,
                 manifest_sha256=info['manifest_sha256'], prompt_sha256=row['prompt_sha256'], budget=row['generation_budget'],
                 rng_seed=1000 + c['index'], prompt_tokens=row['prompt_token_count'], max_model_len=info['max_model_len'],
                 chunk=16384, block_size=32)
        if tamper:
            tamper(r)
        out.append(r)
    return out


def write(path, recs):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in recs))
    return str(path)


def run(script, *args, expect_fail=False):
    out = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], capture_output=True, text=True)
    if expect_fail:
        if out.returncode == 0:
            raise RuntimeError(f'{script} accepted input it must refuse')
        return out.stderr.strip().splitlines()[-1][:160]
    if out.returncode:
        raise RuntimeError(f'{script} failed:\n{out.stderr[-3000:]}')
    return out.stdout


def suite(name, out, pool_dir, cells_file, body, thinking, scorer, fixed):
    """Write perfect / cappedperfect / empty arms, score them, check the three refusals; returns the summary."""
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    cells = cells_of(cells_file)
    files = [write(d / 'e2e_perfect.private.jsonl', records(pool_dir, cells, body, thinking, 'stop')),
             write(d / 'e2e_cappedperfect.private.jsonl', records(pool_dir, cells, body, thinking, 'length')),
             write(d / 'e2e_empty.private.jsonl', records(pool_dir, cells, lambda c: '', thinking, 'stop'))]
    run(scorer, d / 'scores', *fixed, '--cells', cells_file, *files)
    refusals = {}
    for what, tamper in (('prompt_sha256', lambda r: r.update(prompt_sha256='0' * 64)),
                         ('budget', lambda r: r.update(budget=r['budget'] + 1)),
                         ('manifest_sha256', lambda r: r.update(manifest_sha256='1' * 64))):
        bad = write(d / 'e2e_bad.private.jsonl', records(pool_dir, cells[:1], body, thinking, 'stop', tamper))
        refusals[what] = run(scorer, d / 'bad', *fixed, '--cells', cells_file, bad, '--allow-missing', expect_fail=True)
    partial = write(d / 'e2e_partial.private.jsonl', records(pool_dir, cells[:-1], body, thinking, 'stop'))
    refusals['missing_cell'] = run(scorer, d / 'bad', *fixed, '--cells', cells_file, files[0], partial, expect_fail=True)
    stray = dict(cells[0], index=cells[0]['index'] + 100000)
    refusals['unplanned_cell'] = run(scorer, d / 'bad', *fixed, '--cells', cells_file,
                                     write(d / 'e2e_stray.private.jsonl', records(pool_dir, cells[:1], body, thinking, 'stop')
                                           + [dict(records(pool_dir, cells[:1], body, thinking, 'stop')[0], index=stray['index'])]),
                                     '--allow-missing', expect_fail=True)
    for f in ('e2e_bad', 'e2e_partial', 'e2e_stray'):
        (d / f'{f}.private.jsonl').unlink()
    return json.loads((d / 'scores.summary.json').read_text()), refusals


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    results, ok = {}, True
    # RULER (prefilled pools: the completion is the continuation after the answer prefix)
    for pool, cells_name in (('ruler_v33ofc', 'cells_ruler_v33ofc.json'), ('ruler_v33noop', 'cells_ruler_v33noop.json')):
        pd = W / pool
        gold = {}
        for f in pd.glob('*_gold.json'):
            gold.update(json.loads(f.read_text()))
        s, ref = suite(pool, out, pd, pd / cells_name, lambda c: ' ' + ', '.join(gold[c['id']]) + '.', False,
                       'v31_score_ruler.py', (RULER_ROOT, pd, pd))
        results[pool] = dict(arms={k: dict(overall=v['official']['_overall'], cells=v['cells'], capped=v.get('capped', 0),
                                           nulls=v['nulls'], verified=v.get('verified', 0)) for k, v in s['arms'].items()},
                             refusals=ref)
        a = s['arms']
        ok &= a['perfect']['official']['_overall'] == 100 and a['cappedperfect']['official']['_overall'] == 100 \
            and a['empty']['official']['_overall'] == 0
    # RULER table on the ofc outputs (13 tasks per length, comparable settings)
    e = out / 'ruler_v33ofc'
    table = run('v31_ruler_official_table.py', W / 'ruler_v33ofc', 'perfect', e / 'scores.official.json', '--binding',
                e / 'scores.binding.json', '--cells', W / 'ruler_v33ofc/cells_ruler_v33ofc.json',
                '--secondary', e / 'scores.strict_all_correct_finished.json', '--secondary-out', e / 'strict_table.md')
    results['ruler_table_rows'] = [x for x in table.splitlines() if x.startswith('| ') and 'official' not in x]
    # LongBench-v2
    for ds in LB:
        pd = W / 'longbench_v2_ofc'
        g = json.loads((pd / f'{ds}_gold.json').read_text())
        s, ref = suite(ds, out, pd, pd / f'cells_{ds}.json', lambda c: f"The correct answer is ({g[c['id']]})",
                       rowinfo(pd, ds)['rows'][0]['thinking'], 'v31_score_longbench_official.py', (pd, pd))
        results[ds] = dict(arms={k: dict(v[ds]['runs']['seed1_repeat0'], unparsed=v[ds]['unparsed']) for k, v in s['arms'].items()},
                           refusals=ref)
        r = results[ds]['arms']
        ok &= r['perfect']['Overall'] == 100.0 and r['cappedperfect']['Overall'] == 100.0 and r['empty']['Overall'] == 0.0
        e = out / ds
        results[ds]['paired'] = [x for x in run('v31_paired_official.py', 'longbench', pd, 'perfect', e / 'scores.official.json',
                                                 e / 'scores.binding.json', '--cells', pd / f'cells_{ds}.json').splitlines()
                                 if x.startswith('| empty')]
    # GraphWalks
    pd = W / 'graphwalks_b16k'
    g = {}
    for f in pd.glob('*_gold.json'):
        g.update(json.loads(f.read_text()))
    s, ref = suite('graphwalks_b16k', out, pd, pd / 'cells_graphwalks_b16k.json', lambda c: 'Final Answer: [' + ', '.join(g[c['id']]) + ']',
                   True, 'v31_score_graphwalks.py', (pd, pd))
    results['graphwalks_b16k'] = dict(arms={k: {b: v['bins'][b]['all']['mean_f1'] for b in v['bins']} for k, v in s['arms'].items()},
                                      refusals=ref)
    ok &= all(x == 1.0 for x in results['graphwalks_b16k']['arms']['perfect'].values())
    ok &= all(x == 0.0 for x in results['graphwalks_b16k']['arms']['empty'].values())
    # MRCR (README bins)
    pd = W / 'mrcr_ofc'
    g = {}
    for f in pd.glob('*_gold.json'):
        g.update(json.loads(f.read_text()))
    s, ref = suite('mrcr_ofc', out, pd, pd / 'cells_mrcr_ofc.json', lambda c: g[c['id']]['answer'], False, 'v31_score_mrcr.py', (pd, pd))
    results['mrcr_ofc'] = dict(arms={k: {b: round(x['mean_ratio'], 6) for b, x in v['bins'].items()} for k, v in s['arms'].items()},
                               refusals=ref)
    ok &= all(x == 1.0 for x in results['mrcr_ofc']['arms']['perfect'].values())
    ok &= all(x == 0.0 for x in results['mrcr_ofc']['arms']['empty'].values())
    # AIME26, seeds 1-4
    pd = W / 'aime26_b32k'
    answers = {str(json.loads(x)['id']): str(json.loads(x)['answer']) for x in Path(AIME).read_text().splitlines() if x.strip()}
    src = {r['id']: r['source_id'] for r in rowinfo(pd, 'aime26_b32k')['rows']}
    s, ref = suite('aime26_b32k', out, pd, pd / 'cells_aime26_b32k.json',
                   lambda c: 'So the answer is \\boxed{' + answers[str(src[c['id']])] + '}', True, 'v31_score_aime.py', (pd, AIME))
    results['aime26_b32k'] = dict(arms={k: dict(avg_at_k=v['avg_at_k'], k=v['k_per_problem'], cells=v['cells']) for k, v in s['arms'].items()},
                                  refusals=ref)
    ok &= results['aime26_b32k']['arms']['perfect']['avg_at_k'] == 100 and results['aime26_b32k']['arms']['empty']['avg_at_k'] == 0
    e = out / 'aime26_b32k'
    results['aime26_b32k']['paired'] = [x for x in run('v31_paired_official.py', 'aime', pd, 'perfect', e / 'scores.official.json',
                                                         e / 'scores.binding.json', '--cells', pd / 'cells_aime26_b32k.json').splitlines()
                                        if x.startswith('| empty')]
    print(json.dumps(results, indent=1, sort_keys=True))
    print('ALL_EXPECTED' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
