"""Exploration / hold-out split of the official LongBench-v2 0shot_think pool (thinking on, 16384-token budget).

Algorithm search (screening panels) may only use the EXPLORATION cells; the hold-out cells are reserved for the
pre-registered confirmation of the final algorithm and are not run before it. Eligible for exploration: rows with
prompt_token_count >= 32768 (where the sparse path is active at every GLOBAL layer). The exploration set is a seeded
sample stratified by LongBench-v2's own length class (short / medium / long), proportional to the eligible counts
(largest remainder). The hold-out set is every other row of the pool (eligible or not). CPU only, stdlib only.

usage: python v31_lbt_split.py POOL_DIR OUT_DIR [--n 160] [--seed 20261004]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET = 'longbench_v2_0shot_think'
MIN_PROMPT = 32768
CLASSES = ('short', 'medium', 'long')


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def allocate(counts, n):
    """Largest-remainder proportional allocation of n over the classes."""
    total = sum(counts.values())
    raw = {c: n * counts[c] / total for c in CLASSES}
    out = {c: int(raw[c]) for c in CLASSES}
    for c in sorted(CLASSES, key=lambda c: (-(raw[c] - out[c]), CLASSES.index(c)))[:n - sum(out.values())]:
        out[c] += 1
    return out


def split(rows, n, seed):
    eligible = {c: sorted(r['index'] for r in rows if r['length'] == c and r['prompt_token_count'] >= MIN_PROMPT)
                for c in CLASSES}
    take = allocate({c: len(v) for c, v in eligible.items()}, n)
    rng = random.Random(seed)
    explore = sorted(i for c in CLASSES for i in rng.sample(eligible[c], take[c]))
    return explore, {c: len(v) for c, v in eligible.items()}, take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pool')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=160)
    ap.add_argument('--seed', type=int, default=20261004)
    a = ap.parse_args()
    pool, out = Path(a.pool), Path(a.out)
    info = json.loads((pool / f'{DATASET}_rowinfo.json').read_text())
    cells_src = pool / f'cells_{DATASET}.json'
    cells = json.loads(cells_src.read_text())
    rows = info['rows']
    if [r['index'] for r in rows] != list(range(len(rows))) or [c['index'] for c in cells] != list(range(len(rows))):
        raise AssertionError('rowinfo rows and cells must list every index in order')
    if any(c['id'] != r['id'] for c, r in zip(cells, rows)):
        raise AssertionError('cells and rowinfo disagree on ids')
    explore, eligible, take = split(rows, a.n, a.seed)
    keep = set(explore)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, sel in (('cells_lbt_explore.json', [c for c in cells if c['index'] in keep]),
                      ('cells_lbt_holdout.json', [c for c in cells if c['index'] not in keep])):
        (out / name).write_text(json.dumps(sel, indent=1) + '\n')
        files[name] = dict(cells=len(sel), sha256=sha_file(out / name))
    split_doc = dict(
        schema='v31_lbt_split_v1', dataset=DATASET, rule=__doc__.split('\n\n')[1].replace('\n', ' '),
        seed=a.seed, n=a.n, min_prompt_tokens=MIN_PROMPT, eligible=eligible, explore_per_class=take,
        explore_indices=explore, source=dict(cells=cells_src.name, cells_sha256=sha_file(cells_src),
                                             rowinfo_sha256=sha_file(pool / f'{DATASET}_rowinfo.json'),
                                             manifest_sha256=info['manifest_sha256'], max_model_len=info['max_model_len']),
        outputs=files)
    (out / 'lbt_split.json').write_text(json.dumps(split_doc, indent=1) + '\n')
    print(json.dumps({k: split_doc[k] for k in ('eligible', 'explore_per_class', 'outputs')}))


if __name__ == '__main__':
    main()
