"""CPU-only alias2 load screen on private v27_need_dump NPZ snapshots.

Natural q64 is the baseline. The candidate exchanges a few rows between the
two Q64 groups of each original Q128 block, optimizing maximum/tail CTA load
rather than sorting by need count. Heads and Q128 blocks never mix. Every
group still has 64 rows, and its support is the union of all member-row needs.

For each group's cnt kept KV64 tiles, the paged alias2 consumer produces two
CTAs with floor(cnt/2) and ceil(cnt/2) tiles. The reported balanced_proxy_cost
is max(max_CTA_load, total_tiles / sm_count), an idealized load lower bound in
tile units, NOT GPU time. It excludes permutation, selector, list-building,
memory locality, launch and merge costs. Old dumps contain prefix needs only;
tail work is excluded unless --tail-tiles supplies an explicit uniform guess.

No GPU/model imports. CLI output is anonymous aggregate statistics only:
python -m scripts.v28_regroup_screen PRIVATE_DUMP_DIR NEW_REPORT.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np


POP = np.array([i.bit_count() for i in range(256)], dtype=np.int32)


@dataclass(frozen=True)
class Options:
    sm_count: int = 132
    max_total_increase: float = 0.01
    min_proxy_gain: float = 0.05
    max_moved_fraction: float = 0.125
    max_swaps_per_head: int = 4
    candidates_per_group: int = 8
    tail_tiles: int = 0

    def __post_init__(self):
        for key in ('sm_count', 'max_swaps_per_head', 'candidates_per_group', 'tail_tiles'):
            value = getattr(self, key)
            if type(value) is not int or value < (1 if key in ('sm_count', 'candidates_per_group') else 0):
                raise ValueError(f'invalid {key}')
        for key, upper in (('max_total_increase', 0.1), ('min_proxy_gain', 1.0),
                           ('max_moved_fraction', 1.0)):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f'invalid {key}')


def validate_need(need):
    need = np.asarray(need)
    if need.dtype != np.bool_ or need.ndim != 3:
        raise ValueError('need must be a bool [heads, queries, prefix_tiles] array')
    h, q, pt = need.shape
    if h < 1 or q < 128 or q % 128 or pt < 1:
        raise ValueError('only complete Q128 blocks are supported; no padded query rows')
    return need


def natural_order(need):
    h, q, _ = validate_need(need).shape
    return np.broadcast_to(np.arange(q), (h, q)).copy()


def validate_order(need, order):
    h, q, _ = validate_need(need).shape
    order = np.asarray(order)
    if order.shape != (h, q) or not np.issubdtype(order.dtype, np.integer):
        raise ValueError('order must contain row indices separately for each head')
    expected = np.arange(q)
    if not np.all(np.sort(order, axis=1) == expected):
        raise ValueError('order must be a per-head bijection')
    if not np.all(order // 128 == expected[None, :] // 128):
        raise ValueError('rows cannot leave their original Q128 block')
    return order


def group_support(need, order):
    """Union support [H,Q/64,PT]; scatter output rows back by the same order."""
    need = validate_need(need)
    order = validate_order(need, order)
    h, q, pt = need.shape
    reordered = need[np.arange(h)[:, None], order]
    return reordered.reshape(h, q // 64, 64, pt).any(axis=2)


def _p95(values):
    values = np.asarray(values).reshape(-1)
    # Nearest-rank p95, with an explicit convention independent of NumPy version.
    return int(np.sort(values)[math.ceil(0.95 * len(values)) - 1])


def alias2_loads(group_counts):
    counts = np.asarray(group_counts)
    if not np.issubdtype(counts.dtype, np.integer) or counts.size == 0 or np.any(counts < 0):
        raise ValueError('group_counts must be nonnegative integer tile counts')
    # Match adapter._split's lo[i] = floor(cnt*i/2), batch-alias-major order.
    return np.stack((counts // 2, counts - counts // 2), axis=0).reshape(-1)


def cost_from_counts(group_counts, sm_count=132):
    if type(sm_count) is not int or sm_count < 1:
        raise ValueError('invalid sm_count')
    loads = alias2_loads(group_counts)
    total, maximum = int(loads.sum()), int(loads.max())
    mean = total / len(loads)
    return dict(total_tiles=total, cta_count=len(loads), max_cta_tiles=maximum,
                p95_cta_tiles=_p95(loads), mean_cta_tiles=mean,
                max_over_mean=maximum / mean if mean else 0.0,
                balanced_proxy_cost=max(float(maximum), total / sm_count))


def _counts(packed, order):
    h, q, width = packed.shape
    rows = packed[np.arange(h)[:, None], order].reshape(h, q // 64, 64, width)
    return POP[np.bitwise_or.reduce(rows, axis=2)].sum(axis=-1)


def _objective(counts):
    loads = alias2_loads(counts)
    return int(loads.max()), _p95(loads), int(loads.sum())


def optimize_local_swaps(need, options=Options()):
    """Bounded deterministic local search; never import/call the old sort-only candidate."""
    need = validate_need(need)
    order = natural_order(need)
    packed = np.packbits(need, axis=-1)
    h, q, _ = need.shape
    counts = _counts(packed, order)
    baseline_total = int(counts.sum()) + h * (q // 64) * options.tail_tiles
    limit = baseline_total * (1 + options.max_total_increase)
    moved_limit = math.floor(q * options.max_moved_fraction)
    swaps = 0
    for head in range(h):
        for _ in range(options.max_swaps_per_head):
            before = _objective(counts[head] + options.tail_tiles)
            best = None
            for block in range(q // 128):
                g0, g1 = 2 * block, 2 * block + 1
                positions = []
                for group in (g0, g1):
                    start = group * 64
                    # One representative per distinct bit-string; cap work.
                    _, unique = np.unique(packed[head, order[head, start:start + 64]],
                                          axis=0, return_index=True)
                    positions.append(start + unique[:options.candidates_per_group])
                for a in positions[0]:
                    for b in positions[1]:
                        trial = order[head].copy()
                        trial[a], trial[b] = trial[b], trial[a]
                        if np.count_nonzero(trial != np.arange(q)) > moved_limit:
                            continue
                        changed = []
                        for group in (g0, g1):
                            union = np.bitwise_or.reduce(packed[head, trial[group*64:(group+1)*64]], axis=0)
                            changed.append(int(POP[union].sum()))
                        trial_counts = counts[head].copy()
                        trial_counts[[g0, g1]] = changed
                        total = int(counts.sum()) + h * (q // 64) * options.tail_tiles
                        total += int(trial_counts.sum() - counts[head].sum())
                        if total > limit + 1e-12:
                            continue
                        objective = _objective(trial_counts + options.tail_tiles)
                        if objective < before and (best is None or objective < best[0]):
                            best = objective, trial, trial_counts
            if best is None:
                break
            _, order[head], counts[head] = best
            swaps += 1
    validate_order(need, order)
    return order, swaps


def gate_candidate(baseline, candidate, moved_rows, total_rows, options=Options()):
    """A conservative proxy gate, not a performance or accuracy acceptance test."""
    if candidate['cta_count'] != baseline['cta_count']:
        raise ValueError('candidate changes the fixed CTA count')
    reasons = []
    if candidate['total_tiles'] > baseline['total_tiles'] * (1 + options.max_total_increase) + 1e-12:
        reasons.append('total_work_increase')
    if candidate['max_cta_tiles'] > baseline['max_cta_tiles']:
        reasons.append('maximum_regression')
    if candidate['p95_cta_tiles'] > baseline['p95_cta_tiles']:
        reasons.append('p95_regression')
    if moved_rows <= 0 or moved_rows > total_rows * options.max_moved_fraction:
        reasons.append('permutation_budget')
    base = baseline['balanced_proxy_cost']
    gain = 1 - candidate['balanced_proxy_cost'] / base if base > 0 else 0.0
    if gain <= 0 or gain + 1e-12 < options.min_proxy_gain:
        reasons.append('insufficient_proxy_gain')
    return not reasons, reasons


def screen_need(need, options=Options()):
    need = validate_need(need)
    baseline_order = natural_order(need)
    candidate_order, swaps = optimize_local_swaps(need, options)
    orders = {'q64': baseline_order, 'local_swap': candidate_order}
    counts = {key: group_support(need, value).sum(-1) + options.tail_tiles
              for key, value in orders.items()}
    costs = {key: cost_from_counts(value, options.sm_count) for key, value in counts.items()}
    moved = int(np.count_nonzero(candidate_order != baseline_order))
    accepted, reasons = gate_candidate(costs['q64'], costs['local_swap'], moved,
                                      need.shape[0] * need.shape[1], options)
    orders['gated'] = candidate_order if accepted else baseline_order
    counts['gated'] = counts['local_swap'] if accepted else counts['q64']
    costs['gated'] = costs['local_swap'] if accepted else costs['q64']
    return dict(costs=costs, loads={key: alias2_loads(value) for key, value in counts.items()},
                accepted=accepted, reasons=reasons, swaps=swaps, moved_rows=moved,
                orders=orders)


def _unpack_checked(packed, pt, expected_shape):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.shape != expected_shape:
        raise ValueError('invalid packed support dtype or shape')
    unpacked = np.unpackbits(packed, axis=-1)
    if unpacked[..., pt:].any():
        raise ValueError('nonzero packed padding bits')
    return unpacked[..., :pt].astype(bool)


def load_need(path):
    """Load original dump format, fail on padding or router support disagreement."""
    with np.load(path, allow_pickle=False) as data:
        if not {'need', 'pt', 'kept128'} <= set(data.files):
            raise ValueError('snapshot requires need, pt and kept128')
        pt_raw = np.asarray(data['pt'])
        if pt_raw.ndim != 0 or not np.issubdtype(pt_raw.dtype, np.integer) or int(pt_raw) < 1:
            raise ValueError('pt must be a positive integer scalar')
        pt = int(pt_raw)
        shape = data['need'].shape
        if len(shape) != 3 or shape[-1] != (pt + 7) // 8:
            raise ValueError('packed need extent disagrees with pt')
        h, q, _ = shape
        need = validate_need(_unpack_checked(data['need'], pt, shape))
        kept128 = _unpack_checked(data['kept128'], pt, (h, q // 128, shape[-1]))
        if not np.array_equal(kept128, need.reshape(h, q // 128, 128, pt).any(2)):
            raise ValueError('kept128 does not match the original row needs')
    return need


def aggregate_results(results, options=Options()):
    if not results:
        raise ValueError('no snapshots to screen')
    variants = {}
    for name in ('q64', 'local_swap', 'gated'):
        loads = np.concatenate([r['loads'][name] for r in results])
        total, count = int(loads.sum()), len(loads)
        proxy_sum = sum(r['costs'][name]['balanced_proxy_cost'] for r in results)
        variants[name] = dict(total_tiles=total, cta_count=count, max_cta_tiles=int(loads.max()),
                              p95_cta_tiles=_p95(loads), mean_cta_tiles=total / count,
                              balanced_proxy_cost_sum=proxy_sum,
                              balanced_proxy_cost_mean=proxy_sum / len(results))
    base = variants['q64']['balanced_proxy_cost_sum']
    for values in variants.values():
        values['proxy_ratio_to_q64'] = values['balanced_proxy_cost_sum'] / base if base else None
    return dict(schema='v28_alias2_regroup_screen_v1', options=asdict(options),
                measurement='CPU tile-load proxy only; not GPU time or end-to-end speedup',
                limitations=['prefix-only need snapshots', 'uniform tail assumption only',
                             'permutation/selector/list/launch/merge/memory costs excluded',
                             'fixed Q64 groups and alias2 CTA count; Q128/head isolation'],
                snapshots=len(results), accepted=sum(r['accepted'] for r in results),
                rejection_reasons=dict(Counter(reason for r in results for reason in r['reasons'])),
                local_swaps=sum(r['swaps'] for r in results),
                moved_rows=sum(r['moved_rows'] for r in results), variants=variants)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output', help='new JSON report, or - for anonymous stdout')
    for key, default, kind in (('sm_count', 132, int), ('max_total_increase', .01, float),
                              ('min_proxy_gain', .05, float), ('max_moved_fraction', .125, float),
                              ('max_swaps_per_head', 4, int), ('candidates_per_group', 8, int),
                              ('tail_tiles', 0, int)):
        parser.add_argument('--' + key.replace('_', '-'), type=kind, default=default)
    args = parser.parse_args(argv)
    options = Options(**{key: getattr(args, key) for key in asdict(Options())})
    if args.output != '-' and Path(args.output).exists():
        raise FileExistsError('refusing to overwrite an existing report')
    files = sorted(args.input_dir.glob('*.npz'))
    report = aggregate_results([screen_need(load_need(path), options) for path in files], options)
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output == '-':
        print(encoded, end='')
    else:
        with Path(args.output).open('x', encoding='utf-8') as out:
            out.write(encoded)
    return report


if __name__ == '__main__':
    main()
