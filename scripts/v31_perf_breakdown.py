"""Where the time goes: paired decomposition of the end-to-end request time of each arm against a reference arm.

Per request: W = prefill + S (decode), S = N x (S/N) with N denoising forwards, N = C x (N/C) with C canvases.
For every (arm, dataset), over the cells both arms ran (same cell, seed, repeat; one record per cell), the table gives
geometric means of the paired ratios arm / reference, and the split of log(W ratio) into
  prefill   log((P + S_ref) / (P_ref + S_ref))                 -- the prefill change alone
  decode    log((P + S) / (P + S_ref))                         -- the decode change given the arm's prefill
and of log(S ratio) into canvases C, steps per canvas N/C and per-step cost S/N (these three add up exactly).
Reference medians (seconds, forwards) give the scale. Public records only; no text.
Comparability: every paired cell must have identical rng_seed, budget, prompt_tokens, max_model_len, chunk, block_size
(and prompt / manifest sha256 when recorded) in both arms; otherwise the tool refuses (COMPARE).
usage: python v31_perf_breakdown.py REF_LABEL RECORDS.jsonl [...] [--datasets a,b]
  labels from file names <tag>_<label>.jsonl
"""
import collections
import json
import math
import statistics
import sys
from pathlib import Path

COMPARE = ('rng_seed', 'budget', 'prompt_tokens', 'max_model_len', 'chunk', 'block_size', 'prompt_sha256', 'manifest_sha256')


def check_pairs(label, ref, pairs):
    """Refuse a pair whose run settings differ (a missing field must be missing in both arms)."""
    for a, r in pairs:
        bad = [f for f in COMPARE if a.get(f) != r.get(f)]
        if bad:
            raise ValueError(f"{label} vs {ref}, cell {a['dataset']}|{a['index']}|{a['panel_seed']}|{a['repeat']}: "
                             f'different {bad}; refusing to compare')


def main():
    args = sys.argv[1:]
    only = None
    if '--datasets' in args:
        i = args.index('--datasets')
        only = set(args[i + 1].split(','))
        del args[i:i + 2]
    ref, files = args[0], args[1:]
    recs = collections.defaultdict(dict)
    for f in files:
        label = Path(f).name.split('.jsonl')[0].split('_', 1)[1]
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            if r.get('repeat', 0) < 0 or not r.get('denoise_forwards') or (only and r['dataset'] not in only):
                continue
            recs[label][(r['dataset'], r['index'], r['panel_seed'], r['repeat'])] = r
    base = recs.pop(ref)
    datasets = sorted({k[0] for k in base})
    cols = ['arm', 'cells', 'W', 'prefill', 'decode S', 'N', 'C', 'N/C', 'S/N', 'logW: prefill + decode',
            'logS: C + N/C + S/N']
    for ds in datasets:
        b = {k: v for k, v in base.items() if k[0] == ds}
        print(f'\n### {ds}: reference {ref} medians: W {statistics.median(r["wall_s"] for r in b.values()):.1f} s, '
              f'prefill {statistics.median(r["prefill_s"] for r in b.values()):.1f} s, '
              f'decode {statistics.median(r["decode_s"] for r in b.values()):.1f} s, '
              f'N {statistics.median(r["denoise_forwards"] for r in b.values()):.0f}, '
              f'C {statistics.median(r["canvases"] for r in b.values()):.0f}, '
              f'N/C {statistics.median(r["denoise_forwards"] / r["canvases"] for r in b.values()):.1f}, '
              f'S/N {1000 * statistics.median(r["decode_s"] / r["denoise_forwards"] for r in b.values()):.1f} ms\n')
        print('| ' + ' | '.join(cols) + ' |')
        print('|' + '---|' * len(cols))
        for label in sorted(recs):
            pairs = [(recs[label][k], b[k]) for k in b if k in recs[label]]
            if not pairs:
                continue
            check_pairs(label, ref, pairs)
            g = collections.defaultdict(list)
            for a, r in pairs:
                P, S, Pr, Sr = a['prefill_s'], a['decode_s'], r['prefill_s'], r['decode_s']
                g['W'].append(math.log(a['wall_s'] / r['wall_s']))
                g['P'].append(math.log(P / Pr))
                g['S'].append(math.log(S / Sr))
                g['N'].append(math.log(a['denoise_forwards'] / r['denoise_forwards']))
                g['C'].append(math.log(a['canvases'] / r['canvases']))
                g['NC'].append(math.log((a['denoise_forwards'] / a['canvases']) / (r['denoise_forwards'] / r['canvases'])))
                g['SN'].append(math.log((S / a['denoise_forwards']) / (Sr / r['denoise_forwards'])))
                g['wp'].append(math.log((P + Sr) / (Pr + Sr)))
                g['wd'].append(math.log((P + S) / (P + Sr)))
            m = {k: sum(v) / len(v) for k, v in g.items()}
            e = {k: math.exp(v) for k, v in m.items()}
            row = [label, str(len(pairs)), f"{e['W']:.3f}", f"{e['P']:.3f}", f"{e['S']:.3f}", f"{e['N']:.3f}",
                   f"{e['C']:.3f}", f"{e['NC']:.3f}", f"{e['SN']:.3f}",
                   f"{m['wp']:+.3f} {m['wd']:+.3f}", f"{m['C']:+.3f} {m['NC']:+.3f} {m['SN']:+.3f}"]
            print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
