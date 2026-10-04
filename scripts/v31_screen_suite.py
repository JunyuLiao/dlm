"""Screening suite S1: small seeded subsets of the four official long-context pools, each item run under several
sampler seeds, so every candidate algorithm is screened on all four task families before any is expanded.

  RULER v34ofc (exploration pool)  items p0000 + p0001 of each of the 13 tasks at 32K / 64K / 128K      (78 items)
  LongBench-v2 0shot_think         a seeded, length-class-stratified sample of the EXPLORATION split   (32 items)
                                   (lbt_split.json; the hold-out split is never screened)
  MRCR 2-needle ofc                a seeded sample of 8 items per bin (32K / 64K / 128K)               (24 items)
  GraphWalks b32k                  a seeded sample of 4 items per bin (22K / 45K / 90K)                (12 items)
Short-prompt part S1s (--short; thinking on, 8192-token budget, pool manifests_ae, pinned max_model_len 13312: the
sparse path acts on the model's own reasoning once the context passes the token budget):
  AIME26                           all 30 problems (too few to split; a confirmation uses fresh seeds)  (30 items)
  HumanEval                        a seeded sample of 40 of the 164 problems; the other 124 are held out (40 items)

MRCR and GraphWalks have no separate exploration pool: their screening items are recorded here and a later
confirmation uses only the complementary items of each bin. Every item is listed once per seed (cell key dataset |
index | seed); every arm of a cell runs on the same host. CPU only, stdlib only.

usage: python v31_screen_suite.py POOLS_DIR LBT_SPLIT_DIR OUT_DIR [--seeds 1,2] [--sample-seed 20261004]
       python v31_screen_suite.py --short AE_MANIFEST_DIR HE_CELLS OUT_DIR [--seeds 1,2] [--sample-seed 20261004]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

RULER_ITEMS = ('p0000', 'p0001')
LBT_N = {'short': 6, 'medium': 18, 'long': 8}
MRCR_PER_BIN, GW_PER_BIN = 8, 4
HE_N = 40
EOL = chr(10)
PARA = EOL * 2


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ruler_items(cells):
    return [c for c in cells if c['id'].rsplit('_', 1)[1] in RULER_ITEMS]


def per_bin_sample(cells, k, rng):
    out = []
    for ds in sorted({c['dataset'] for c in cells}):
        out += rng.sample([c for c in cells if c['dataset'] == ds], k)
    return sorted(out, key=lambda c: (c['dataset'], c['index']))


def lbt_items(cells, length_of, rng):
    out = []
    for cls, k in LBT_N.items():
        out += rng.sample([c for c in cells if length_of[c['index']] == cls], k)
    return sorted(out, key=lambda c: c['index'])


def with_seeds(items, seeds):
    return [dict(c, seed=s) for s in seeds for c in items]


def short_main(a):
    man_dir, he_src, out = Path(a.pools), Path(a.lbt_split), Path(a.out)
    seeds = [int(s) for s in a.seeds.split(',')]
    rng = random.Random(a.sample_seed)
    man = man_dir / 'aime26_generation_manifest.json'
    rows = json.loads(man.read_text())
    aime = [dict(dataset='aime26', id=r['id'], index=i, prompt_tokens=r['prompt_token_count'], seed=1)
            for i, r in enumerate(rows)]
    he_all = json.loads(he_src.read_text())
    he = sorted(rng.sample(he_all, HE_N), key=lambda c: c['index'])
    keep = {c['index'] for c in he}
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema='v31_screen_suite_short_v1', rule=__doc__.split(PARA)[1], seeds=seeds,
               sample_seed=a.sample_seed, parts={})
    for k, its, src in (('aime', aime, man), ('he', he, he_src)):
        name = f'cells_sc_{k}.json'
        (out / name).write_text(json.dumps(with_seeds(its, seeds), indent=1) + EOL)
        doc['parts'][k] = dict(source=src.name, source_sha256=sha_file(src), items=len(its), cells=len(its) * len(seeds),
                               file=name, sha256=sha_file(out / name), indices=sorted(c['index'] for c in its))
    (out / 'cells_he_holdout.json').write_text(json.dumps([c for c in he_all if c['index'] not in keep], indent=1) + EOL)
    doc['he_holdout'] = dict(file='cells_he_holdout.json', cells=len(he_all) - len(keep),
                             sha256=sha_file(out / 'cells_he_holdout.json'))
    (out / 'sc_suite_short.json').write_text(json.dumps(doc, indent=1) + EOL)
    print(json.dumps({k: (v['items'], v['cells']) for k, v in doc['parts'].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short', action='store_true')
    ap.add_argument('pools')
    ap.add_argument('lbt_split')
    ap.add_argument('out')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--sample-seed', type=int, default=20261004)
    a = ap.parse_args()
    if a.short:
        return short_main(a)
    pools, split_dir, out = Path(a.pools), Path(a.lbt_split), Path(a.out)
    seeds = [int(s) for s in a.seeds.split(',')]
    rng = random.Random(a.sample_seed)
    src = dict(ruler=pools / 'ruler_v34ofc' / 'cells_ruler_v34ofc.json',
               lbt=split_dir / 'cells_lbt_explore.json',
               mrcr=pools / 'mrcr_ofc' / 'cells_mrcr_ofc.json',
               gw=pools / 'graphwalks_b32k' / 'cells_graphwalks_b32k.json')
    load = {k: json.loads(p.read_text()) for k, p in src.items()}
    rows = json.loads((pools / 'longbench_v2_ofc' / 'longbench_v2_0shot_think_rowinfo.json').read_text())['rows']
    length_of = {r['index']: r['length'] for r in rows}
    items = dict(ruler=ruler_items(load['ruler']),
                 lbt=lbt_items(load['lbt'], length_of, rng),
                 mrcr=per_bin_sample(load['mrcr'], MRCR_PER_BIN, rng),
                 gw=per_bin_sample(load['gw'], GW_PER_BIN, rng))
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema='v31_screen_suite_v1', rule=__doc__.split('\n\n')[1], seeds=seeds, sample_seed=a.sample_seed,
               parts={})
    for k, its in items.items():
        name = f'cells_sc_{k}.json'
        (out / name).write_text(json.dumps(with_seeds(its, seeds), indent=1) + '\n')
        doc['parts'][k] = dict(source=str(src[k].name), source_sha256=sha_file(src[k]), items=len(its),
                               cells=len(its) * len(seeds), file=name, sha256=sha_file(out / name),
                               items_by_dataset={ds: sorted(c['index'] for c in its if c['dataset'] == ds)
                                                 for ds in sorted({c['dataset'] for c in its})})
    (out / 'sc_suite.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: (v['items'], v['cells']) for k, v in doc['parts'].items()}))


if __name__ == '__main__':
    main()
