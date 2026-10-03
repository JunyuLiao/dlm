"""Offline audit for the uniform-threshold temporal-prior proposal.

This does not replay attention masks.  It consumes the archived per-step
diagnostics and estimates the coefficient schedule that ``T_prior`` would
produce from the causal sampler signals.  The output is intentionally a
trajectory diagnostic, not a sparsity or quality result: routing must still be
run on a GPU with one local/global threshold pair.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _rows(path, condition):
    grouped = defaultdict(list)
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('condition') == condition:
                grouped[row['id']].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row['iteration'])
    return grouped


def summarize(path, condition, *, beta=3., gamma=.5, canvas_size=256):
    """Summarize raw-T and proposed T_prior coefficients by call index.

    ``renoised`` and ``argmax_flips`` are aggregate diagnostics in the archive,
    so the proposed coefficient is an expectation-level proxy.  It is useful
    for checking the direction and scale of the prior, but cannot validate the
    resulting block masks.
    """
    grouped = _rows(path, condition)
    buckets = defaultdict(lambda: dict(n=0, raw_sum=0., prior_sum=0.,
                                        renoised_sum=0., flips_sum=0.))
    for records in grouped.values():
        trajectory = 1.
        temporal = 0.
        for record in records:
            iteration = int(record['iteration'])
            renoised = float(record.get('renoised', 0.))
            accepted = float(record.get('accepted', 0.))
            flip_count = float(record.get('argmax_flips') or 0.)
            total = max(accepted + renoised, 1.)
            item = buckets[iteration]
            item['n'] += 1
            item['raw_sum'] += 1. + beta * temporal
            item['prior_sum'] += 1. + beta * (
                trajectory + (1. - trajectory) * temporal)
            item['renoised_sum'] += renoised / total
            item['flips_sum'] += flip_count / max(canvas_size, 1)
            trajectory = gamma * trajectory + (1. - gamma) * (renoised / total)
            if iteration > 1:
                temporal = gamma * temporal + (1. - gamma) * (
                    flip_count / max(canvas_size, 1))
    result = []
    for iteration in sorted(buckets):
        item = buckets[iteration]
        n = item['n']
        result.append(dict(condition=condition, iteration=iteration, n=n,
            raw_coefficient=item['raw_sum'] / n,
            prior_coefficient=item['prior_sum'] / n,
            renoised_fraction=item['renoised_sum'] / n,
            flip_fraction=item['flips_sum'] / n))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('per_step', type=Path)
    parser.add_argument('--condition', default='T_s70')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beta', type=float, default=3.)
    parser.add_argument('--gamma', type=float, default=.5)
    args = parser.parse_args()
    rows = summarize(args.per_step, args.condition, beta=args.beta,
                     gamma=args.gamma)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ['condition', 'iteration', 'n',
                                 'raw_coefficient', 'prior_coefficient',
                                 'renoised_fraction', 'flip_fraction'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(condition=args.condition, rows=len(rows),
                          output=str(args.output))))


if __name__ == '__main__':
    main()
