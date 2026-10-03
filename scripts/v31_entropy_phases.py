"""Where do sparsity-induced extra denoising steps occur? (panel m, TRACE=1 receipts)

Every denoising step of a canvas is assigned to a phase by the canvas mean token entropy the sampler sees after it
(the quantity compared with the confidence threshold c, 0.005 in this deployment):
  early  >= 1.0        mid  [0.1, 1.0)        tail  [c, 0.1)   (almost converged, not yet below c)
  conv   < c           (the converging step; a canvas ends right after it, or at the step cap)
Per arm and length bin, against the paired dense trace (native arm): steps per canvas in each phase (pooled over
cells, geometric means of per-cell ratios are not used because phases can be empty), and canvases that hit the cap.
usage: python v31_entropy_phases.py REF_LABEL FILE.jsonl [...]   (files whose records carry receipts.trace)
"""
import collections
import json
import sys
from pathlib import Path

PHASES = ('early', 'mid', 'tail', 'conv')


def phase(e, c):
    return 'early' if e >= 1.0 else 'mid' if e >= 0.1 else 'tail' if e >= c else 'conv'


def main():
    ref, files = sys.argv[1], sys.argv[2:]
    rows = collections.defaultdict(dict)
    for f in files:
        label = Path(f).stem.split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            t = (r.get('receipts') or {}).get('trace')
            if t:
                rows[(r['dataset'][-3:], r['index'], r['panel_seed'])][label] = t
    labels = sorted({l for v in rows.values() for l in v})
    print('steps per canvas by phase, each arm next to the dense trace on the SAME cells; cap = canvases >= 40 steps')
    print(f"{'arm':40s} bin cells canvases  early   mid   tail  conv  total  cap")

    def pooled(lab, ks):
        count = collections.Counter()
        canvases = cap = 0
        for k in ks:
            t = rows[k][lab]
            c = t['confidence_threshold']
            for ent in t['canvas_entropy']:
                canvases += 1
                cap += len(ent) >= 40
                for e in ent:
                    count[phase(e, c)] += 1
        return {p: count[p] / max(1, canvases) for p in PHASES}, canvases, cap

    for ds in ('32k', '64k'):
        for lab in [l for l in labels if l != ref]:
            ks = [k for k, v in rows.items() if k[0] == ds and lab in v and ref in v]
            if not ks:
                continue
            for name in (ref, lab):
                per, canvases, cap = pooled(name, ks)
                tag = '  (dense, same cells)' if name == ref else ''
                print(f"{(name + tag)[:40]:40s} {ds} {len(ks):4d} {canvases:8d} "
                      + ' '.join(f'{per[p]:5.2f}' for p in PHASES) + f" {sum(per.values()):6.2f} {cap:4d}")


if __name__ == '__main__':
    main()
