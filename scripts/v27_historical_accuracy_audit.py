"""Audit paired historical accuracy without publishing private scored CSV rows.

This consumes existing qualified scores; it does not rescore or independently
qualify worker provenance. Each (dataset, private item id, seed) must have exactly
one selected arm and reference. Pool files only when their seeds do not overlap.
Resample whole questions, retaining every seed observation. The cell McNemar
number is exploratory, not a question-cluster test or a noninferiority result.
Only aggregate statistics and recognized public dataset labels are emitted.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path


PUBLIC_DATASETS = frozenset(('longbench_v2_32k', 'longbench_v2_64k',
                             'longbench_v2_96k', 'aime26', 'ruler32k_r17',
                             'ruler64k_r17', 'ruler92k_r17'))


def read_pairs(paths, arm, base):
    """Return private pairing keys in memory only; never include them in output."""
    if not paths or not arm or not base or arm == base:
        raise ValueError('distinct arms and at least one scored input are required')
    cells = defaultdict(dict)
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            required = {'dataset', 'id', 'seed', 'arm', 'first_status',
                        'quality_eligible', 'scored_first', 'strict_correct'}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError('scored input lacks required columns')
            for row in reader:
                selected = row['arm']
                if selected not in (arm, base):
                    continue
                if (row['first_status'] != 'success' or row['quality_eligible'] != 'True'
                        or row['scored_first'] != 'True'
                        or row['strict_correct'] not in ('True', 'False')):
                    raise ValueError('selected row is not a qualified strict score')
                if any(not row[k] for k in ('dataset', 'id', 'seed')):
                    raise ValueError('selected row lacks pairing identity')
                try:
                    seed = int(row['seed'])
                except ValueError:
                    raise ValueError('selected row seed must be an integer') from None
                key = (row['dataset'], row['id'], seed)
                if selected in cells[key]:
                    raise ValueError('duplicate selected execution, including across input files')
                cells[key][selected] = row
    if not cells:
        raise ValueError('no selected paired executions')
    pairs = defaultdict(list)
    for (dataset, item, seed), entries in sorted(cells.items()):
        if set(entries) != {arm, base}:
            raise ValueError('selected execution lacks its paired reference')
        a, b = entries[arm], entries[base]
        # Historical cell_id identifies the arm execution, so it intentionally
        # differs across arms. Pair by dataset/item/seed, then verify host/GPU.
        for field in ('host', 'gpu_uuid'):
            if field in a or field in b:
                if not a.get(field) or a.get(field) != b.get(field):
                    raise ValueError('paired execution identity differs')
        pairs[dataset].append((item, seed, a['strict_correct'] == 'True',
                               b['strict_correct'] == 'True'))
    return pairs


def exploratory_mcnemar(arm_only, base_only):
    n = arm_only + base_only
    if not n:
        return 1.0
    k = min(arm_only, base_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def item_bootstrap_ci(groups, reps=20000, seed=7):
    """Cell-weighted difference; resample items with all their observations."""
    if not groups or any(not group for group in groups) or reps < 40:
        raise ValueError('nonempty item clusters and at least 40 bootstrap draws required')
    totals = [(sum(group), len(group)) for group in groups]
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        selected = [rng.choice(totals) for _ in totals]
        samples.append(sum(x[0] for x in selected) / sum(x[1] for x in selected))
    samples.sort()
    return [samples[math.floor(.025 * reps)], samples[math.ceil(.975 * reps) - 1]]


def exact_item_sign_flip(groups, max_items=16):
    """Flip each complete item difference, testing symmetric item contrasts.

    This is a distinct paired symmetry test, not McNemar. No Monte Carlo fallback
    is silently substituted for an exact test when the item limit is exceeded.
    """
    if not groups or any(not group for group in groups) or not 1 <= max_items <= 20:
        raise ValueError('nonempty clusters and an exact item limit in 1..20 required')
    if len(groups) > max_items:
        return None
    totals = [sum(group) for group in groups]
    observed = abs(sum(totals))
    nonzero = [value for value in totals if value]
    extreme = sum(abs(sum(value * sign for value, sign in zip(nonzero, signs))) >= observed
                  for signs in itertools.product((-1, 1), repeat=len(nonzero)))
    return extreme / 2 ** len(nonzero)


def summarize(pairs, bootstrap_reps=20000, bootstrap_seed=7, exact_max_items=16):
    if not pairs:
        raise ValueError('no paired datasets')
    rows = []
    for ordinal, dataset in enumerate(sorted(pairs), 1):
        records = pairs[dataset]
        if not records:
            raise ValueError('empty paired dataset')
        groups = defaultdict(list)
        for item, _, a, b in records:
            groups[item].append(int(a) - int(b))
        clusters = [groups[item] for item in sorted(groups)]
        counts = [len(group) for group in clusters]
        arm_correct = sum(a for _, _, a, _ in records)
        base_correct = sum(b for _, _, _, b in records)
        arm_only = sum(a and not b for _, _, a, b in records)
        base_only = sum(b and not a for _, _, a, b in records)
        p = exact_item_sign_flip(clusters, exact_max_items)
        rows.append(dict(dataset=dataset if dataset in PUBLIC_DATASETS else f'anonymous_dataset_{ordinal:03d}',
                         cells=len(records), items=len(groups),
                         observations_per_item_min=min(counts), observations_per_item_max=max(counts),
                         arm_correct=arm_correct, base_correct=base_correct,
                         arm_only=arm_only, base_only=base_only,
                         accuracy_difference=(arm_correct - base_correct) / len(records),
                         accuracy_difference_item_cluster_ci95=item_bootstrap_ci(
                             clusters, bootstrap_reps, bootstrap_seed),
                         item_positive=sum(sum(group) > 0 for group in clusters),
                         item_negative=sum(sum(group) < 0 for group in clusters),
                         item_zero=sum(sum(group) == 0 for group in clusters),
                         exact_item_sign_flip_p=p,
                         exact_item_sign_flip_assignments=2 ** len(groups) if p is not None else None,
                         exact_item_sign_flip_status='exact' if p is not None else 'item_limit_exceeded',
                         exploratory_cell_mcnemar_exact_p=exploratory_mcnemar(arm_only, base_only)))
    return dict(schema='v27_historical_accuracy_audit_v1',
                bootstrap=dict(seed=bootstrap_seed, draws=bootstrap_reps,
                               method='item-cluster percentile 95%; all observations retained; cell-weighted difference',
                               percentile_indices='floor(0.025*draws), ceil(0.975*draws)-1'),
                exact_sign_flip=dict(max_items=exact_max_items,
                                     assumption='independent symmetric paired item contrasts under the null'),
                interpretation='Existing qualified strict scores only. Cell McNemar is exploratory. '
                               'Repeated seeds are not independent question clusters. '
                               'These tests do not establish accuracy noninferiority.',
                datasets=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scored', action='append', required=True, type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--bootstrap-reps', type=int, default=20000)
    parser.add_argument('--bootstrap-seed', type=int, default=7)
    parser.add_argument('--exact-max-items', type=int, default=16)
    args = parser.parse_args(argv)
    try:
        if args.out and args.out.resolve() in {path.resolve() for path in args.scored}:
            raise ValueError('aggregate output cannot overwrite a scored input')
        result = summarize(read_pairs(args.scored, args.arm, args.base),
                           args.bootstrap_reps, args.bootstrap_seed, args.exact_max_items)
    except (ValueError, OSError, csv.Error):
        # Do not echo private paths, item identities or raw malformed fields.
        parser.exit(2, 'Historical accuracy audit failed input or statistical validation.\n')
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')
    return result


if __name__ == '__main__':
    main()
