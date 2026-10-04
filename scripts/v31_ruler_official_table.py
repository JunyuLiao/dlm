"""RULER arms side by side under RULER's OFFICIAL score, on one explicit set of common cells (aggregates only).

Inputs: the .official.json and .binding.json of scripts/v31_score_ruler.py and the planned cells.
- Cells: one explicit intersection over ALL arms in the file (computed before --only, so the selection never changes it);
  a cell outside the plan raises; a planned cell missing in an arm raises unless --allow-missing; per arm planned /
  present / dropped is printed; every length must keep all 13 RULER tasks (asserted).
- Comparability: per cell every arm must have run with identical rng_seed, budget, prompt_tokens, max_model_len, chunk,
  block_size, prompt and manifest sha256 (v31_official.check_comparable), else the table refuses.
- Per length: the per-sample metric (0-100, partial credit, unrounded) averaged per task, then the unweighted mean over
  that length's 13 tasks -- every task weighs the same however many cells it holds (unbalanced pools aggregate as RULER
  does); overall = the mean over lengths (v31_official.ruler_official_aggregate); cwe and w/o-cwe follow the same rule.
- Paired difference to the reference = the same weighted contrast of per-cell differences (a cell weighs
  1 / (lengths x tasks of its length x cells of its length-task stratum), so it equals official(arm) - official(ref)),
  95% CI by a stratified bootstrap (cells resampled within each length x task stratum, 4000 draws); better / worse cells
  with an exact sign test; GLOBAL work from the public records.
The strict all-correct boolean is not part of this table: --secondary STRICT.json --secondary-out FILE.md writes its
counts (per length, total, lost / gained vs the reference, exact McNemar p) to a separate, secondary file.
usage:
  python v31_ruler_official_table.py MANIFEST_DIR REF_LABEL OFFICIAL.json --binding BINDING.json --cells CELLS.json
      [--cells ...] [--repeats N] [--allow-missing] [--legacy-unbound] [--secondary STRICT.json --secondary-out FILE.md]
      [--alias NEW=A+B ...] [--only SUBSTR,...] [--public PUBLIC.jsonl ...]
"""
import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402
from v31_ruler_compare import compute, mcnemar  # noqa: E402


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
    """[(cell keys of one length x task stratum, official cell weight 1 / (L * T_length * n_stratum))]."""
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('manifest_dir')
    p.add_argument('ref')
    p.add_argument('official')
    p.add_argument('--binding', required=True)
    p.add_argument('--cells', action='append', required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--allow-missing', action='store_true')
    p.add_argument('--legacy-unbound', action='store_true')
    p.add_argument('--secondary')
    p.add_argument('--secondary-out')
    p.add_argument('--alias', action='append', default=[])
    p.add_argument('--only')
    p.add_argument('--public', nargs='*', default=[])
    a = p.parse_args()
    if bool(a.secondary) != bool(a.secondary_out):
        raise SystemExit('--secondary and --secondary-out go together')
    aliases = {x.split('=')[0]: x.split('=')[1].split('+') for x in a.alias}
    raw, binding = json.loads(Path(a.official).read_text()), json.loads(Path(a.binding).read_text())
    merge_aliases(raw, aliases)
    merge_aliases(binding, aliases)
    if a.ref not in raw:
        raise SystemExit(f'reference {a.ref} not in {a.official}')
    labels = sorted(raw)
    plan = op.planned_cells(a.cells, a.repeats)
    common, coverage = op.common_cells({l: raw[l] for l in labels}, plan, a.allow_missing)
    op.check_comparable(binding, labels, common, a.legacy_unbound)
    shown = [a.ref] + [l for l in labels if l != a.ref and (a.only is None or any(s in l for s in a.only.split(',')))]
    lengths = sorted({k.split('|')[0] for k in common}, key=op.natural_key)
    task_of = {ds: [r['task'] for r in op.load_pool(a.manifest_dir, ds, ('task',))[0]] for ds in lengths}
    task = lambda k: task_of[k.split('|')[0]][int(k.split('|')[1])]
    length = lambda k: k.split('|')[0]
    for ln in lengths:
        tasks = {task(k) for k in common if length(k) == ln}
        if tasks != set(op.RULER_TASKS):
            raise ValueError(f'{ln}: the common cells hold {len(tasks)} of the 13 RULER tasks; refusing an official mean')
    owner = {p_: new for new, parts in aliases.items() for p_ in parts}
    work = collections.defaultdict(dict)
    for f in a.public:
        label = owner.get(op.label_of(f), op.label_of(f))
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            key = op.cell_key(r)
            c = compute(r)
            if key in common and c is not None:
                work[label][key] = c
    strata = strata_of(common, task)
    agg = {l: op.ruler_official_aggregate({k: raw[l][k] for k in common}, task, length) for l in shown}
    print(f'common cells: {len(common)} (13 tasks per length asserted); reference {a.ref}; per length = unweighted mean over '
          f'tasks of the task means; overall = mean over lengths; comparable settings checked on every cell')
    for line in op.coverage_lines(coverage):
        print('coverage: ' + line)
    print()
    head = (['arm'] + [f'{ln} official' for ln in lengths] +
            ['official', 'cwe', 'w/o cwe', 'official diff vs ref [95% CI]', 'better/worse (p)', 'work'])
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for label in shown:
        g = agg[label]
        cwe = [g[ln]['cwe'] for ln in lengths if g[ln]['cwe'] is not None]
        rest = [g[ln]['without_cwe'] for ln in lengths if g[ln]['without_cwe'] is not None]
        row = ([label] + [f"{g[ln]['avg']:.1f}" for ln in lengths] +
               [f"{g['_overall']:.1f}", f'{statistics.mean(cwe):.1f}' if cwe else '-', f'{statistics.mean(rest):.1f}' if rest else '-'])
        if label == a.ref:
            row += ['-', '-']
        else:
            d = {k: raw[label][k] - raw[a.ref][k] for k in common}
            lo, hi = bootstrap(strata, d)
            pos, neg = sum(x > 0 for x in d.values()), sum(x < 0 for x in d.values())
            row += [f'{contrast(strata, d):+.2f} [{lo:+.2f}, {hi:+.2f}]', f'{pos}/{neg} ({op.binomial_two_sided(pos, neg):.3f})']
        w = work.get(label)
        row.append(f'{sum(x[2] for x in w.values()) / sum(x[3] for x in w.values()):.3f}' if w and len(w) == len(common) else '-')
        print('| ' + ' | '.join(row) + ' |')
    if a.secondary:
        strict = json.loads(Path(a.secondary).read_text())
        merge_aliases(strict, aliases)
        lines = [f'Secondary: strict all-correct (score = 100 and a stop / eos finish; not RULER\'s metric) on the same '
                 f'{len(common)} common cells; reference {a.ref}.', '',
                 '| ' + ' | '.join(['arm'] + lengths + ['total', 'lost/gained vs ref', 'McNemar p']) + ' |',
                 '|' + '---|' * (len(lengths) + 4)]
        for label in shown:
            s, r = strict[label], strict[a.ref]
            row = [label] + [str(sum(bool(s[k]) for k in common if length(k) == ln)) for ln in lengths]
            row.append(f'{sum(bool(s[k]) for k in common)}/{len(common)}')
            lost = sum(bool(r[k]) and not s[k] for k in common)
            gained = sum(bool(s[k]) and not r[k] for k in common)
            row += ['-', '-'] if label == a.ref else [f'{lost}/{gained}', f'{mcnemar(lost, gained):.3f}']
            lines.append('| ' + ' | '.join(row) + ' |')
        Path(a.secondary_out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'\nsecondary strict all-correct table -> {a.secondary_out}')


if __name__ == '__main__':
    main()
