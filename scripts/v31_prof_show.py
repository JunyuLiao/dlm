import collections, glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + '/*.prof.jsonl')):
    print('##', f.split('/')[-1])
    for i, l in enumerate(open(f)):
        r = json.loads(l)
        if i == 0:
            continue
        sl = r['step_list']
        print(f" req{i}: steps {r['steps']} step_ms {r['step_ms_total']} global {r['global_ms_total']} hook {r['hook_ms_total']}")
        for k, v in r['calls_by_kind'].items():
            print(f"    call {k[:40]:40s} n={v['n']:4d} ms={v['ms_mean']:.3f} sparse={v['sparse_ms_mean']:.3f} kv={v['kv_ms_mean']:.3f} kept={v['kept_mean']}")
        c = collections.Counter(tuple(s[3]) for s in sl)
        for kk, n in c.most_common():
            ms = sorted(s[0] for s in sl if tuple(s[3]) == kk)
            g = [s[1] for s in sl if tuple(s[3]) == kk]
            h = [s[2] for s in sl if tuple(s[3]) == kk]
            print(f"    step {str(kk)[:50]:50s} n={n:3d} median={ms[len(ms)//2]:.2f} global={sum(g)/n:.2f} hook={sum(h)/n:.3f}")
