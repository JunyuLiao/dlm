"""Offline query-row grouping analysis on dumped need matrices (``scripts/v27_need_dump.py``); CPU only.

The FA4 hd512 kernel runs 64-row query tiles; a tile loads every KV64 tile that ANY of its 64 rows needs, so the
kernel work of a grouping is sum over groups of |union of the members' need sets| (counted in 64-row tile units:
a 128-row group counts twice). Every grouping below has the same number of 64-row tiles (64 per decision), so the
kernel launches the same grid. Groupings:
  q128           per head, natural rows 0-127 / 128-255 (the executed M3 map; checked against the router)
  q64            per head, natural 64-row tiles (v27 q64)
  q64r           per head and 128-row block, rows sorted by need count (v27 q64r)
  head_count     per head, all 256 rows sorted by need count
  head_set       per head, rows sorted by the need bit-string (descending lexicographic; chw/value_aware's set key)
  head_greedy    per head, greedy clustering (seed = heaviest row, add the row adding the fewest new tiles)
  gqa_pack       per KV head: 8 positions x 8 query heads per tile (FA4 pack-GQA layout, a fixed permutation)
  gqa_count      per KV head: all 2048 (head, position) rows sorted by need count
  gqa_set        per KV head: rows sorted by the need bit-string
  gqa_greedy     per KV head: greedy clustering over the 2048 rows
  per_row        sum of per-row needs / 64 (no 64-row tiling can go below this)
Cross-head groupings run as an FA4 MHA call with the KV heads as heads and 2048 gathered query rows.
usage: python -m scripts.v27_regroup_offline OUT.json DUMP_DIR [GREEDY_MAX_PER_BIN]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(packed):
    return int(POP[packed].sum())


def union_cost(packed, groups):
    return sum(popcount(np.bitwise_or.reduce(packed[g], axis=0)) * (len(g) // 64) for g in groups)


def chunks(order, size=64):
    return [order[i:i + size] for i in range(0, len(order), size)]


def set_order(packed):
    # descending lexicographic on the bit-string, first tile most significant (np.lexsort: last key is primary)
    keys = [255 - packed[:, j] for j in range(packed.shape[1] - 1, -1, -1)]
    return np.lexsort(keys)


def count_order(packed):
    return np.argsort(POP[packed].sum(1), kind='stable')


def greedy_groups(packed, size=64):
    rem = list(range(packed.shape[0]))
    counts = POP[packed].sum(1)
    groups = []
    while rem:
        idx = np.array(rem)
        seed = idx[np.argmax(counts[idx])]
        members, union = [seed], packed[seed].copy()
        left = idx[idx != seed]
        while len(members) < size and len(left):
            added = POP[packed[left] & ~union].sum(1)
            pick = int(np.argmin(added))
            members.append(left[pick])
            union |= packed[left[pick]]
            left = np.delete(left, pick)
        groups.append(np.array(members))
        chosen = set(members)
        rem = [r for r in rem if r not in chosen]
    return groups


def analyse(path, greedy):
    d = np.load(path)
    need = d['need']                                   # [H, 256, nbytes]
    h = need.shape[0]
    kept128 = d['kept128']                             # [H, 2, nbytes]
    res = defaultdict(int)
    router = 0
    for i in range(h):
        n = need[i]
        res['q128'] += union_cost(n, [np.arange(0, 128), np.arange(128, 256)])
        router += popcount(kept128[i]) * 2
        res['q64'] += union_cost(n, chunks(np.arange(256)))
        q64r = []
        for blk in (0, 1):
            rows = np.arange(blk * 128, blk * 128 + 128)
            q64r += chunks(rows[count_order(n[rows])])
        res['q64r'] += union_cost(n, q64r)
        res['head_count'] += union_cost(n, chunks(count_order(n)))
        res['head_set'] += union_cost(n, chunks(set_order(n)))
        res['head_greedy'] += union_cost(n, greedy_groups(n))
        res['per_row'] += POP[n].sum() / 64
    for g in range(2):                                 # KV groups: query heads 8g .. 8g+7
        heads = need[8 * g:8 * g + 8]                  # [8, 256, nbytes]
        flat = heads.reshape(8 * 256, -1)              # row = head * 256 + position
        pack = np.arange(8 * 256).reshape(8, 256).T.reshape(-1)    # position-major, head-minor
        res['gqa_pack'] += union_cost(flat, chunks(pack))
        res['gqa_count'] += union_cost(flat, chunks(count_order(flat)))
        res['gqa_set'] += union_cost(flat, chunks(set_order(flat)))
        if greedy:
            res['gqa_greedy'] += union_cost(flat, greedy_groups(flat))
    res['router_check'] = int(res['q128'] == router)
    res['pt'] = int(d['pt'])
    return dict(res)


def main():
    out, dump = Path(sys.argv[1]), Path(sys.argv[2])
    greedy_max = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    files = sorted(dump.glob('*.npz'))
    per_bin, rows, seen = defaultdict(lambda: defaultdict(float)), [], defaultdict(int)
    for f in files:
        b = f.name.split('_')[0]
        greedy = seen[b] < greedy_max
        seen[b] += 1
        r = analyse(f, greedy)
        r.update(file=f.name, bin=b, greedy=greedy)
        rows.append(r)
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ('pt',):
                per_bin[b][k] += v
                if greedy and k == 'q64':
                    per_bin[b]['q64_greedy_subset'] += v      # gqa_greedy itself exists only on this subset
        print(f.name, {k: r[k] for k in ('q128', 'q64', 'q64r', 'head_set', 'gqa_pack', 'gqa_set', 'per_row')}, flush=True)
    summary = {}
    for b, t in per_bin.items():
        base = t['q64']
        s = {k: round(t[k] / base, 4) for k in ('q128', 'q64', 'q64r', 'head_count', 'head_set', 'head_greedy',
                                                 'gqa_pack', 'gqa_count', 'gqa_set', 'per_row')}
        if t.get('q64_greedy_subset'):
            s['gqa_greedy'] = round(t['gqa_greedy'] / t['q64_greedy_subset'], 4)
            s['greedy_decisions'] = min(seen[b], greedy_max)
        s['decisions'] = seen[b]
        s['router_check'] = f"{int(t['router_check'])}/{seen[b]}"
        summary[b] = s
    json.dump(dict(relative_to_q64=summary, decisions=rows), open(out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
