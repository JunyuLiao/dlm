"""AIME target-sparsity panel: per-arm realized sparsity (by construction), steps, tokens, speed and accuracy.

usage: python aime_sparsity_table.py <scored.csv or -> <ledger> [<ledger> ...]
Pairs are (id, seed) cells; every arm of a cell ran on one host. Pairs in which either run captured new CUDA
graphs while timed are excluded from the speed ratios (not from accuracy).
Realized sparsity counts GLOBAL key tiles (64 keys) skipped over all GLOBAL calls of a request: canvas tiles
(256 keys) are always kept, the first call of each canvas is dense, and a held call skips floor((1-keep) x prefix
tiles) (topk: keep = 1 - target; SparseD: keep = 1 - target). It is an upper-bound-free construction count, not
telemetry.
"""
import csv, json, math, random, sys, collections

POOL = {r['id']: r['prompt_token_count'] for r in
        json.load(open('E:/dlm/v27_private/pool/aime26_pool_manifest.json', encoding='utf-8'))}
TARGET = {'topk30': .3, 'topk40': .4, 'topk50': .5, 'topk70': .7, 'topk80': .8,
          's30_': .3, 's40_': .4, 's50_': .5, 's70_': .7, 's80_': .8}


def target_of(arm):
    for k, v in TARGET.items():
        if k in arm:
            return v
    return None


def realized(e, target):
    if target is None:
        return None
    prompt = POOL[e['id']]
    skipped = total = 0
    for c, n in enumerate(e.get('per_canvas_calls') or []):
        prefix_tiles = math.ceil((prompt + 256 * c) / 64)
        tiles = prefix_tiles + 4
        total += n * tiles
        skipped += (n - 1) * math.floor(target * prefix_tiles)
    return skipped / total if total else None


def geo(x):
    return math.exp(sum(map(math.log, x)) / len(x))


def ci(byq, reps=2000):
    qs = list(byq)
    random.seed(7)
    bs = sorted(geo([r for q in (random.choice(qs) for _ in qs) for r in byq[q]]) for _ in range(reps))
    return bs[int(.025 * reps)], bs[int(.975 * reps) - 1]


def main():
    scored, ledgers = sys.argv[1], sys.argv[2:]
    quality = {}
    if scored != '-':
        for r in csv.DictReader(open(scored, encoding='utf-8')):
            if r.get('first_status') == 'success':
                quality[(r['id'], str(r['seed']), r['arm'])] = r['strict_correct'] == 'True'
    cells = collections.defaultdict(dict)
    for path in ledgers:
        for line in open(path, encoding='utf-8'):
            e = json.loads(line) if line.strip() else {}
            if e.get('event') != 'run' or not e.get('ok') or e.get('role', 'attempt0') != 'attempt0':
                continue
            pe = e.get('phase_evidence') or {}
            cells[(e['id'], str(e['seed']))][e['arm']] = dict(
                W=e.get('api_wall_s') or e['outer_wall_s'], S=pe.get('prefill_end_to_finish_gpu_s'),
                N=e['decoder_calls'], C=e['canvases'], T=e['output_tokens'], ng=bool(e.get('substrate_new_graphs')),
                sp=realized(e, target_of(e['arm'])), ok=quality.get((e['id'], str(e['seed']), e['arm'])))
    base = 'D_fa4_allkept'
    arms = sorted({a for v in cells.values() for a in v}, key=lambda a: (a != base, a))
    print('| arm | cells | realized sparsity (GLOBAL tiles) | correct (dense) | +/- | W [CI] | steps N [CI] | steps/canvas | tokens T | per-step S/N [CI] |')
    print('|---|---:|---:|---|---|---|---|---|---|---|')
    for a in arms:
        pairs = [(k, v[a], v[base]) for k, v in cells.items() if a in v and base in v]
        clean = [(k, x, y) for k, x, y in pairs if not (x['ng'] or y['ng'])]
        out = {}
        for name, f in (('W', lambda x, y: x['W'] / y['W']), ('N', lambda x, y: x['N'] / y['N']),
                        ('NC', lambda x, y: (x['N'] / x['C']) / (y['N'] / y['C'])), ('T', lambda x, y: x['T'] / y['T']),
                        ('SN', lambda x, y: (x['S'] / x['N']) / (y['S'] / y['N']))):
            byq = collections.defaultdict(list)
            for k, x, y in clean:
                byq[k[0]].append(f(x, y))
            flat = [r for v in byq.values() for r in v]
            lo, hi = ci(byq)
            out[name] = f'{geo(flat):.3f} [{lo:.3f}, {hi:.3f}]'
        sps = [x['sp'] for _, x, _ in pairs if x['sp'] is not None]
        sp = f'{100 * sum(sps) / len(sps):.0f}%' if sps else ('0%' if a == base else 'n/a')
        cor = sum(bool(x['ok']) for _, x, _ in pairs)
        bcor = sum(bool(y['ok']) for _, _, y in pairs)
        plus = sum(bool(x['ok']) and not y['ok'] for _, x, y in pairs)
        minus = sum(not x['ok'] and bool(y['ok']) for _, x, y in pairs)
        print(f"| {a} | {len(pairs)} ({len(clean)} timed) | {sp} | {cor} ({bcor}) | +{plus}/-{minus} | {out['W']} | "
              f"{out['N']} | {out['NC'].split(' ')[0]} | {out['T'].split(' ')[0]} | {out['SN']} |")


if __name__ == '__main__':
    main()
