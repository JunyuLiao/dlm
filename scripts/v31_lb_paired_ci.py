"""Paired per-cell ratios (geometric mean, 95% bootstrap CI over cells) of S/N, N/C, per-canvas decode and W against a
dense reference, pooled over hosts. usage: python lb_ci.py REF_LABEL host=GLOB [host=GLOB ...] [--only a,b]"""
import collections
import glob
import json
import math
import os
import random
import sys

rng = random.Random(7)
args = sys.argv[1:]
only = None
if '--only' in args:
    i = args.index('--only')
    only = args[i + 1].split(',')
    del args[i:i + 2]
ref, specs = args[0], args[1:]
d = collections.defaultdict(dict)
for spec in specs:
    host, pat = spec.split('=', 1)
    for f in glob.glob(pat):
        lab = os.path.basename(f).split('_', 1)[1].replace('.jsonl', '')
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if r.get('repeat', 0) < 0 or not r.get('denoise_forwards'):
                continue
            d[lab][(host, r['dataset'], r['index'], r['panel_seed'])] = r


def ci(vals):
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(2000))
    return '%.3f [%.3f, %.3f]' % (math.exp(m), math.exp(bs[50]), math.exp(bs[1949]))


for ds in sorted({k[1] for k in d[ref]}):
    print(f'\n### {ds} (reference {ref})\n')
    print('| arm | cells | S/N | N/C | per-canvas decode (N/C x S/N) | W |')
    print('|---|---|---|---|---|---|')
    for lab in sorted(d):
        if lab == ref or (only and not any(o in lab for o in only)):
            continue
        keys = [k for k in d[ref] if k[1] == ds and k in d[lab]]
        if not keys:
            continue
        g = collections.defaultdict(list)
        for k in keys:
            a, r = d[lab][k], d[ref][k]
            sn = (a['decode_s'] / a['denoise_forwards']) / (r['decode_s'] / r['denoise_forwards'])
            nc = (a['denoise_forwards'] / a['canvases']) / (r['denoise_forwards'] / r['canvases'])
            g['sn'].append(math.log(sn))
            g['nc'].append(math.log(nc))
            g['pc'].append(math.log(sn * nc))
            g['w'].append(math.log(a['wall_s'] / r['wall_s']))
        print(f"| {lab} | {len(keys)} | {ci(g['sn'])} | {ci(g['nc'])} | {ci(g['pc'])} | {ci(g['w'])} |")
