"""End-to-end check of the v31 official scorers on the REAL pool files with SYNTHETIC completions (CPU only).

For each official pool it writes private synthetic completion files (they contain gold text: keep PRIVATE_DIR
private), runs the scorer CLI on them and checks the aggregates:
  perfect          the gold answer in the official answer format, finish 'stop'      -> metric at its maximum
  capped_perfect   the same text, finish 'length' (no end token)                      -> primary unchanged, secondary False
  empty            an empty final response                                             -> metric 0, counted as null / unparsed
Pools: pools_v31_official/ruler_v33 (completion = continuation after the prefilled answer prefix), longbench_v2 (all
three variants), aime26 (seeds 1-4), graphwalks; pools_v31/mrcr. Prints aggregates only.
usage: python v31_scorer_e2e_check.py PRIVATE_DIR   (cwd = a deployment holding experiments/, PYTHONPATH=src:.)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path('/media/volume/dllm-1/dyh/pools_v31_official')
MRCR = Path('/media/volume/dllm-1/dyh/pools_v31/mrcr')
RULER_ROOT = '/media/volume/dllm-1/dyh/ruler_scoring/ruler_long_v27/ruler_repo'
AIME = '/home/exouser/.cache/huggingface/hub/datasets--math-ai--aime26/snapshots/79037aebdb6580008fb960d17cb21fd3099083e3/aime2026.jsonl'


def cells(path):
    return json.loads(Path(path).read_text())


def write(out_dir, tag, label, recs):
    path = out_dir / f'{tag}_{label}.private.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in recs))
    return str(path)


def make(out_dir, tag, cell_list, answer_text, thinking):
    files = []
    for label, finish, empty in (('perfect', 'stop', False), ('cappedperfect', 'length', False), ('empty', 'stop', True)):
        recs = []
        for c in cell_list:
            body = '' if empty else answer_text(c)
            head = '<|channel>thought\nsynthetic\n<channel|>' if thinking else ''
            tail = '<turn|>' if finish == 'stop' else ''
            recs.append(dict(dataset=c['dataset'], index=c['index'], panel_seed=c['seed'], repeat=0, id=c['id'],
                             completion=head + body + tail, finish_reason=finish))
        files.append(write(out_dir, tag, label, recs))
    return files


def run(script, prefix, *args):
    out = subprocess.run([sys.executable, str(HERE / script), str(prefix), *map(str, args)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f'{script} failed:\n{out.stderr[-3000:]}')
    return json.loads(Path(f'{prefix}.summary.json').read_text())['arms']


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    results, ok = {}, True
    # RULER: the continuation after the prefilled answer prefix
    gold = {}
    for ds in ('ruler32k_v33', 'ruler64k_v33', 'ruler128k_v33'):
        gold.update(json.loads((W / 'ruler_v33' / f'{ds}_gold.json').read_text()))
    (out / 'ruler').mkdir(exist_ok=True)
    files = make(out / 'ruler', 'e2e', cells(W / 'ruler_v33/cells_ruler_v33.json'), lambda c: ' ' + ', '.join(gold[c['id']]) + '.', False)
    s = run('v31_score_ruler.py', out / 'ruler/scores', RULER_ROOT, W / 'ruler_v33', W / 'ruler_v33', *files)
    results['ruler'] = {k: dict(overall=v['official']['_overall'], cells=v['cells'], capped=v.get('capped', 0), nulls=v['nulls'])
                        for k, v in s.items()}
    strict = json.loads((out / 'ruler/scores.strict_all_correct_finished.json').read_text())
    ok &= (s['perfect']['official']['_overall'] == 100 and s['cappedperfect']['official']['_overall'] == 100
           and s['empty']['official']['_overall'] == 0 and all(strict['perfect'].values()) and not any(strict['cappedperfect'].values()))
    # LongBench-v2 official pools
    for ds in ('longbench_v2_0shot', 'longbench_v2_0shot_think', 'longbench_v2_cot_think'):
        g = json.loads((W / 'longbench_v2' / f'{ds}_gold.json').read_text())
        info = json.loads((W / 'longbench_v2' / f'{ds}_rowinfo.json').read_text())
        (out / ds).mkdir(exist_ok=True)
        files = make(out / ds, 'e2e', cells(W / f'longbench_v2/cells_{ds}.json'), lambda c: f"The correct answer is ({g[c['id']]})",
                     info[0]['thinking'])
        s = run('v31_score_longbench_official.py', out / ds / 'scores', W / 'longbench_v2', W / 'longbench_v2', *files)
        results[ds] = {k: dict(v[ds]['runs']['seed1_repeat0'], unparsed=v[ds]['unparsed'], capped=v[ds].get('capped', 0))
                       for k, v in s.items()}
        ok &= (all(x == 100.0 for c, x in results[ds]['perfect'].items() if c in ('Overall', 'Easy', 'Hard', 'Short', 'Medium', 'Long'))
               and results[ds]['cappedperfect']['Overall'] == 100.0 and results[ds]['empty']['Overall'] == 0.0
               and results[ds]['empty']['unparsed'] == 503)
    # GraphWalks
    g = {}
    for ds in ('graphwalks_22k', 'graphwalks_45k', 'graphwalks_90k'):
        g.update(json.loads((W / 'graphwalks' / f'{ds}_gold.json').read_text()))
    (out / 'graphwalks').mkdir(exist_ok=True)
    files = make(out / 'graphwalks', 'e2e', cells(W / 'graphwalks/cells_graphwalks.json'),
                 lambda c: 'Final Answer: [' + ', '.join(g[c['id']]) + ']', True)
    s = run('v31_score_graphwalks.py', out / 'graphwalks/scores', W / 'graphwalks', W / 'graphwalks', *files)
    results['graphwalks'] = {k: {b: v['bins'][b]['all']['mean_f1'] for b in v['bins']} for k, v in s.items()}
    ok &= (all(x == 1.0 for x in results['graphwalks']['perfect'].values()) and all(x == 0.0 for x in results['graphwalks']['empty'].values()))
    # MRCR (pools_v31 as is)
    g = {}
    for ds in ('mrcr2_32k', 'mrcr2_64k', 'mrcr2_128k'):
        g.update(json.loads((MRCR / f'{ds}_gold.json').read_text()))
    (out / 'mrcr').mkdir(exist_ok=True)
    files = make(out / 'mrcr', 'e2e', cells(MRCR / 'cells_mrcr.json'), lambda c: g[c['id']]['answer'], False)
    s = run('v31_score_mrcr.py', out / 'mrcr/scores', MRCR, MRCR, *files)
    results['mrcr'] = {k: {b: round(v['bins'][b]['mean_ratio'], 6) for b in v['bins']} for k, v in s.items()}
    ok &= all(x >= 0.99 for x in results['mrcr']['perfect'].values()) and all(x == 0.0 for x in results['mrcr']['empty'].values())
    # AIME26, seeds 1-4
    answers = {str(json.loads(x)['id']): str(json.loads(x)['answer']) for x in Path(AIME).read_text().splitlines() if x.strip()}
    rows = {r['id']: r for r in json.loads((W / 'aime26/aime26_rowinfo.json').read_text())}
    (out / 'aime26').mkdir(exist_ok=True)
    files = make(out / 'aime26', 'e2e', cells(W / 'aime26/cells_aime26.json'),
                 lambda c: 'So the answer is \\boxed{' + answers[str(rows[c['id']]['source_id'])] + '}', True)
    s = run('v31_score_aime.py', out / 'aime26/scores', W / 'aime26', AIME, *files)
    results['aime26'] = {k: dict(avg_at_k=v['avg_at_k'], k=v['k_per_problem'], cells=v['cells'], capped=v.get('capped', 0)) for k, v in s.items()}
    ok &= results['aime26']['perfect']['avg_at_k'] == 100 and results['aime26']['empty']['avg_at_k'] == 0
    print(json.dumps(results, indent=1, sort_keys=True))
    print('ALL_EXPECTED' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
