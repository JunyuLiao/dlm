"""RULER arms side by side under RULER's OFFICIAL score, on their common cells (aggregates only).

Per length: the per-sample metric (0-100, partial credit; the .official.json of scripts/v31_score_ruler.py) averaged per
task, then the unweighted mean over that length's tasks -- every task weighs the same however many cells it holds, so
unbalanced pools aggregate as RULER does; overall = the mean over lengths (v31_official.ruler_official_aggregate). The cwe
and w/o-cwe columns follow the same per-length rule. Paired difference to the reference arm = the same weighted contrast
of per-cell differences (a cell weighs 1 / (lengths x tasks of its length x cells of its length-task stratum), so it
equals official(arm) - official(ref)), with a stratified bootstrap 95% CI (cells resampled within each length x task
stratum, 4000 draws); better / worse cells with a two-sided sign test; GLOBAL work from the public records.
The strict all-correct boolean is not part of this table: --secondary STRICT.json --secondary-out FILE.md writes its
counts (per length, total, lost / gained vs the reference, exact McNemar p) to a separate, secondary file.
usage:
  python v31_ruler_official_table.py MANIFEST_DIR REF_LABEL OFFICIAL.json [--secondary STRICT.json --secondary-out FILE.md]
      [--alias NEW=A+B ...] [--only SUBSTR,...] [--public PUBLIC.jsonl ...]
"""
import collections
import json
import random
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402
from v31_ruler_compare import compute, mcnemar  # noqa: E402


def sign_p(pos, neg):
    n = pos + neg
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def take(args, flag, many=False):
    if flag not in args:
        return [] if many else None
    i = args.index(flag)
    if many:
        out = args[i + 1:]
        del args[i:]
        return out
    value = args[i + 1]
    del args[i:i + 2]
    return value


def merge_aliases(table, aliases):
    for new, parts in aliases.items():
        if all(p in table for p in parts):
            merged = {}
            for p in parts:
                for k, v in table.pop(p).items():
                    if k in merged:
                        raise ValueError(f'alias {new}: cell {k} in more than one part')
                    merged[k] = v
            table[new] = merged


def strata_of(keys, task):
    """{(length, task): [cell keys]} and the official cell weights 1 / (L * T_length * n_stratum)."""
    strata = collections.defaultdict(list)
    for k in sorted(keys):
        strata[(k.split('|')[0], task(k))].append(k)
    lengths = {ln for ln, _ in strata}
    tasks_per_length = collections.Counter(ln for ln, _ in strata)
    return [(ks, 1.0 / (len(lengths) * tasks_per_length[ln])) for (ln, _), ks in sorted(strata.items())]


def contrast(strata, d):
    return sum(w * statistics.mean(d[k] for k in ks) for ks, w in strata)


def bootstrap(strata, d, reps=4000, seed=7):
    rng = random.Random(seed)
    vals = [([d[k] for k in ks], w) for ks, w in strata]
    boots = sorted(sum(w * sum(rng.choice(v) for _ in v) / len(v) for v, w in vals) for _ in range(reps))
    return boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]


def main():
    args = sys.argv[1:]
    pub = take(args, '--public', many=True)
    aliases = {}
    while '--alias' in args:
        new, parts = take(args, '--alias').split('=')
        aliases[new] = parts.split('+')
    only = take(args, '--only')
    only = only.split(',') if only else None
    secondary, secondary_out = take(args, '--secondary'), take(args, '--secondary-out')
    if bool(secondary) != bool(secondary_out):
        raise SystemExit('--secondary and --secondary-out go together')
    man_dir, ref, raw_path = Path(args[0]), args[1], args[2]
    raw = json.loads(Path(raw_path).read_text())
    merge_aliases(raw, aliases)
    labels = [l for l in raw if l == ref or only is None or any(s in l for s in only)]
    common = set.intersection(*(set(raw[l]) for l in labels))
    owner = {p: new for new, parts in aliases.items() for p in parts}
    work = collections.defaultdict(dict)
    for f in pub:
        label = owner.get(Path(f).name.split('.jsonl')[0].split('_', 1)[1], Path(f).name.split('.jsonl')[0].split('_', 1)[1])
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            key = op.cell_key(r)
            c = compute(r)
            if key in common and c is not None:
                work[label][key] = c
    lengths = sorted({k.split('|')[0] for k in common})
    task_of = {ds: [r['task'] for r in op.load_rows(man_dir, ds, ('id', 'task'))] for ds in lengths}
    task = lambda k: task_of[k.split('|')[0]][int(k.split('|')[1])]
    length = lambda k: k.split('|')[0]
    strata = strata_of(common, task)
    agg = {l: op.ruler_official_aggregate({k: raw[l][k] for k in common}, task, length) for l in labels}
    tasks_by_length = {ln: len(agg[ref][ln]['tasks']) for ln in lengths}
    print(f'common cells: {len(common)}; reference {ref}; tasks per length {tasks_by_length}; '
          f'per length = unweighted mean over tasks of the task means; overall = mean over lengths\n')
    head = (['arm'] + [f'{ln} official' for ln in lengths] +
            ['official', 'cwe', 'w/o cwe', 'official diff vs ref [95% CI]', 'better/worse (p)', 'work'])
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for label in [ref] + sorted(l for l in labels if l != ref):
        a = agg[label]
        cwe = [a[ln]['cwe'] for ln in lengths if a[ln]['cwe'] is not None]
        rest = [a[ln]['without_cwe'] for ln in lengths if a[ln]['without_cwe'] is not None]
        row = ([label] + [f"{a[ln]['avg']:.1f}" for ln in lengths] +
               [f"{a['_overall']:.1f}", f'{statistics.mean(cwe):.1f}' if cwe else '-', f'{statistics.mean(rest):.1f}' if rest else '-'])
        if label == ref:
            row += ['-', '-']
        else:
            d = {k: raw[label][k] - raw[ref][k] for k in common}
            lo, hi = bootstrap(strata, d)
            pos, neg = sum(x > 0 for x in d.values()), sum(x < 0 for x in d.values())
            row += [f'{contrast(strata, d):+.2f} [{lo:+.2f}, {hi:+.2f}]', f'{pos}/{neg} ({sign_p(pos, neg):.3f})']
        w = work.get(label)
        row.append(f'{sum(x[2] for x in w.values()) / sum(x[3] for x in w.values()):.3f}' if w and len(w) == len(common) else '-')
        print('| ' + ' | '.join(row) + ' |')
    if secondary:
        strict = json.loads(Path(secondary).read_text())
        merge_aliases(strict, aliases)
        lines = [f'Secondary: strict all-correct (score = 100 and a stop / eos finish; not RULER\'s metric) on the same '
                 f'{len(common)} common cells; reference {ref}.', '',
                 '| ' + ' | '.join(['arm'] + lengths + ['total', 'lost/gained vs ref', 'McNemar p']) + ' |',
                 '|' + '---|' * (len(lengths) + 4)]
        for label in [ref] + sorted(l for l in labels if l != ref):
            s, r = strict[label], strict[ref]
            row = [label] + [str(sum(bool(s[k]) for k in common if length(k) == ln)) for ln in lengths]
            row.append(f'{sum(bool(s[k]) for k in common)}/{len(common)}')
            lost = sum(bool(r[k]) and not s[k] for k in common)
            gained = sum(bool(s[k]) and not r[k] for k in common)
            row += ['-', '-'] if label == ref else [f'{lost}/{gained}', f'{mcnemar(lost, gained):.3f}']
            lines.append('| ' + ' | '.join(row) + ' |')
        Path(secondary_out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'\nsecondary strict all-correct table -> {secondary_out}')


if __name__ == '__main__':
    main()
