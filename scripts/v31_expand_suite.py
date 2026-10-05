"""Expansion suite S2: more items of the same four official long-context pools for the arms that survived S1
screening, so their accuracy and speed rest on larger samples. S2 lists only items NOT in S1 (the S1 cells keep their
records; analyses pool S1 + S2 per arm), seeds 1-2 like S1:

  RULER v34ofc (exploration pool)  items p0002-p0005 of each of the 13 tasks at 32K / 64K / 128K      (156 items)
  LongBench-v2 0shot_think         every EXPLORATION-split item not in S1 (160 - 32; hold-out untouched) (128 items)
  MRCR 2-needle ofc                a seeded sample of 8 more items per bin, outside S1                  (24 items)
  GraphWalks b32k                  a seeded sample of 8 more items per bin, outside S1                  (24 items)

S1 + S2 then cover RULER 6 of 15 items per task and length, LongBench all 160 exploration items, MRCR 16 of 24 and
GraphWalks 12 of 24 items per bin. The remaining RULER v34 items, the RULER v33 pool, the 343 LongBench hold-out items
and the MRCR / GraphWalks items in neither suite stay for the pre-registered confirmation; s2_suite.json records
them. CPU only, stdlib only.

usage: python v31_expand_suite.py POOLS_DIR LBT_SPLIT_DIR S1_DIR OUT_DIR [--seeds 1,2] [--sample-seed 20261005]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

RULER_ITEMS = ('p0002', 'p0003', 'p0004', 'p0005')
MRCR_MORE, GW_MORE = 8, 8
EOL = chr(10)
PARA = EOL * 2


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(c):
    return c['dataset'], c['index']


def per_bin_more(cells, used, k, rng):
    out = []
    for ds in sorted({c['dataset'] for c in cells}):
        free = [c for c in cells if c['dataset'] == ds and key(c) not in used]
        out += rng.sample(free, k)
    return sorted(out, key=key)


def with_seeds(items, seeds):
    return [dict(c, seed=s) for s in seeds for c in items]


def by_dataset(items):
    return {ds: sorted(c['index'] for c in items if c['dataset'] == ds) for ds in sorted({c['dataset'] for c in items})}


def build(load, s1, seeds, sample_seed):
    """load: part -> pool cells (LongBench: the exploration split); s1: part -> S1 cells. Returns (s2, rest)."""
    rng = random.Random(sample_seed)
    used = {k: {key(c) for c in v} for k, v in s1.items()}
    s2 = dict(ruler=sorted([c for c in load['ruler'] if c['id'].rsplit('_', 1)[1] in RULER_ITEMS], key=key),
              lbt=sorted([c for c in load['lbt'] if key(c) not in used['lbt']], key=key),
              mrcr=per_bin_more(load['mrcr'], used['mrcr'], MRCR_MORE, rng),
              gw=per_bin_more(load['gw'], used['gw'], GW_MORE, rng))
    for k, its in s2.items():
        assert not ({key(c) for c in its} & used[k]), k                 # S2 is disjoint from S1
    rest = {k: [c for c in load[k] if key(c) not in used[k] | {key(x) for x in s2[k]}] for k in s2}
    return s2, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pools')
    ap.add_argument('lbt_split')
    ap.add_argument('s1')
    ap.add_argument('out')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--sample-seed', type=int, default=20261005)
    a = ap.parse_args()
    pools, split_dir, s1_dir, out = Path(a.pools), Path(a.lbt_split), Path(a.s1), Path(a.out)
    seeds = [int(s) for s in a.seeds.split(',')]
    src = dict(ruler=pools / 'ruler_v34ofc' / 'cells_ruler_v34ofc.json',
               lbt=split_dir / 'cells_lbt_explore.json',
               mrcr=pools / 'mrcr_ofc' / 'cells_mrcr_ofc.json',
               gw=pools / 'graphwalks_b32k' / 'cells_graphwalks_b32k.json')
    load = {k: json.loads(p.read_text()) for k, p in src.items()}
    s1 = {}
    for k in src:
        cells = json.loads((s1_dir / f'cells_sc_{k}.json').read_text())
        s1[k] = list({key(c): c for c in cells}.values())
    s2, rest = build(load, s1, seeds, a.sample_seed)
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema='v31_expand_suite_v1', rule=__doc__.split(PARA)[1], seeds=seeds, sample_seed=a.sample_seed,
               s1_suite_sha256=sha_file(s1_dir / 'sc_suite.json'), parts={})
    for k, its in s2.items():
        name = f'cells_s2_{k}.json'
        (out / name).write_text(json.dumps(with_seeds(its, seeds), indent=1) + EOL)
        doc['parts'][k] = dict(source=src[k].name, source_sha256=sha_file(src[k]), items=len(its),
                               cells=len(its) * len(seeds), file=name, sha256=sha_file(out / name),
                               items_by_dataset=by_dataset(its), s1_items=len(s1[k]),
                               left_for_confirmation=len(rest[k]), left_by_dataset=by_dataset(rest[k]))
    (out / 's2_suite.json').write_text(json.dumps(doc, indent=1) + EOL)
    print(json.dumps({k: (v['items'], v['cells'], v['left_for_confirmation']) for k, v in doc['parts'].items()}))


if __name__ == '__main__':
    main()
