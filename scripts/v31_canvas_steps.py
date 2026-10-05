import collections
import glob
import json
import os

P = 'E:/dlm/v31_private/paired'
d = collections.defaultdict(list)
for f in glob.glob(P + '/*/m_*.jsonl'):
    lab = os.path.basename(f).split('_', 1)[1].rsplit('.', 1)[0]
    for line in open(f):
        r = json.loads(line)
        t = (r.get('receipts') or {}).get('trace')
        if t:
            d[(lab, r['dataset'][-3:])] += t['canvas_steps']
for (lab, ds), s in sorted(d.items()):
    s.sort()
    n = len(s)
    q = lambda p: s[min(n - 1, int(p * n))]
    beyond = sum(x - 19 for x in s if x >= 20) / sum(s)
    print(f"{lab[:36]:36s} {ds} n={n:4d} p50={q(.5)} p90={q(.9)} p95={q(.95)} p99={q(.99)} max={s[-1]} "
          f">=16:{sum(x >= 16 for x in s) / n:.2f} >=20:{sum(x >= 20 for x in s) / n:.2f} "
          f">=30:{sum(x >= 30 for x in s) / n:.3f} share_of_steps_after_19={beyond:.3f}")
