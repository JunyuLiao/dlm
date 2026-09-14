"""Read-only tuning diagnostics: Gaussian shape, mass bounds, worst-row risk.

No generation or policy selection occurs here. Per-bucket quantile averages
are explicitly NOT pooled quantiles; only exported raw moments can be pooled.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from experiments.diffusion_gemma_value_aware.report import csv_write, table
from experiments.diffusion_gemma_value_aware.operators import BETAS
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from .analysis import screen_bundle
from .engine import contract
from .policies import inputs
from .protocol import ROOT, prepare, sha


def pool_moments(buckets):
    """The observer's `skew` is E[z^3], not its central standardized moment."""
    n = sum(r['count'] for r in buckets)
    if n <= 0 or any(r['count'] <= 0 for r in buckets):
        raise ValueError('positive proxy observation counts required')
    def mean(fn):
        return sum(r['count'] * fn(r) for r in buckets) / n
    m1 = mean(lambda r: r['mean'])
    m2 = mean(lambda r: r['std']**2 + r['mean']**2)
    m3 = mean(lambda r: r['skew'])
    m4 = mean(lambda r: r['fourth_moment'])
    if not all(math.isfinite(x) for x in (m1, m2, m3, m4)):
        raise ValueError('nonfinite proxy moments')
    variance = max(0., m2 - m1*m1)
    third = m3 - 3*m1*m2 + 2*m1**3
    fourth = m4 - 4*m1*m3 + 6*m1*m1*m2 - 3*m1**4
    return dict(count=n, buckets=len(buckets), mean=m1, std=math.sqrt(variance),
        central_skew=third / variance**1.5 if variance > 1e-12 else None,
        excess_kurtosis=fourth / variance**2 - 3 if variance > 1e-12 else None,
        weighting='eligible physical candidate counts; reconstructed raw moments')


def quantile_summary(buckets):
    """Describe saved bucket quantiles without inventing a global percentile."""
    if not buckets or any(r['count'] <= 0 for r in buckets):
        raise ValueError('nonempty positive-count quantile buckets required')
    n = sum(r['count'] for r in buckets)
    result = dict(observations=n, buckets=len(buckets))
    for key in ('p10', 'p50', 'p90', 'p99'):
        values = [r[key] for r in buckets]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite saved quantile')
        result[f'count_weighted_bucket_{key}'] = sum(r['count']*r[key] for r in buckets)/n
    result['min_bucket_p50'] = min(r['p50'] for r in buckets)
    result['max_bucket_p50'] = max(r['p50'] for r in buckets)
    result['interpretation'] = 'count-weighted averages of bucket log-quantiles, NOT pooled quantiles'
    return result


def guarded_sol_rows(summary, raw_counts):
    result = []
    for row in summary:
        if row['split'] not in ('calibration', 'development'):
            raise ValueError('final examples cannot enter tuning diagnostics')
        parts = row['probe'].split('/')
        if len(parts) != 4 or parts[0] != 'guarded_sol':
            continue
        _, proxy, mode, target = parts
        if proxy not in ('plain', 'value') or mode not in ('gaussian', 'topk'):
            raise ValueError('unknown guarded Sol screen')
        target = int(target[1:])/100.
        if target not in BETAS:
            raise ValueError('unknown analytic Gaussian target')
        counts = raw_counts[row['benchmark'], row['split'], row['probe'], row['attention_type']]
        if any(counts[k] != row[k] for k in ('eligible', 'skipped', 'rows')):
            raise ValueError('guarded Sol summary differs from raw counters')
        result.append(dict(benchmark=row['benchmark'], split=row['split'],
            attention_type=row['attention_type'], proxy=proxy, mode=mode,
            target=target, beta=BETAS[target] if mode == 'gaussian' else None,
            actual=row['physical_sparsity'], mass=row['mass'], relative_error=row['relative_error'],
            rescued_row_fraction=counts['rescued_rows']/max(counts['rows'], 1),
            eligible=row['eligible'], skipped=row['skipped'],
            scope='shared dense-state masks with signal-only nonempty repair; NOT sparse-generation results'))
    return result


def build(root):
    setup = prepare(root)
    execution = contract(root)
    inputs(root, execution)
    imports = json.loads((root/'imported_sources.json').read_text())
    expected = json.loads((root/'screen_analysis_audit.json').read_text())['sources']
    sources = {}
    proxy_buckets = []
    estimator_buckets = []
    sol_counts = defaultdict(lambda: defaultdict(float))
    for row in setup['calibration'] + setup['development']:
        bundle = screen_bundle(root, row, execution, imports)
        for index, (data, _, source) in enumerate(bundle):
            key = row['id'] + ('/supplement' if index else '')
            if source != expected[key]:
                raise ValueError('screen source differs from completed analysis')
            for path_key, digest_key in (('path', 'sha256'), ('arrays_path', 'arrays_sha256')):
                sources[source[path_key]] = source[digest_key]
            identity = dict(id=row['id'], benchmark=row['benchmark'], split=row['split'])
            # JointScreen does not export Gaussian moments. The supplemental
            # observer exports these once; exact-mass rescue does not change z.
            if index:
                proxy_buckets.extend(dict(identity, **r) for r in data['distributions'])
                continue
            for record in data['records']:
                if record['probe'].startswith('guarded_sol/'):
                    for kind in ('overall', record['attention_type']):
                        counts = sol_counts[row['benchmark'], row['split'], record['probe'], kind]
                        for name in ('eligible', 'skipped', 'rows', 'rescued_rows'):
                            counts[name] += record[name]
            for bucket, record in enumerate(data['refinement_diagnostics']):
                common = dict(identity, bucket=bucket,
                    **{k: record[k] for k in ('layer', 'step', 'attention_type', 'previous_output_available')})
                measurements = [('mass_bound_log_overestimate',
                    record['mass_bound_log_overestimate_count'],
                    record['mass_bound_log_overestimate_quantiles'])]
                measurements += [(f'{r["method"]}_log_worst_over_mean', r['count'],
                    r['log_worst_over_mean_quantiles']) for r in record['row_inflation']]
                for metric, count, values in measurements:
                    if count == 0:
                        if values:
                            raise ValueError('empty estimator bucket has quantiles')
                        continue
                    if len(values) != 4:
                        raise ValueError('missing estimator quantiles')
                    estimator_buckets.append(dict(common, metric=metric, count=count,
                        **dict(zip(('p10', 'p50', 'p90', 'p99'), values))))
    groups = defaultdict(list)
    for row in proxy_buckets:
        groups[row['benchmark'], row['split'], row['attention_type'], row['value_aware']].append(row)
    proxy = [dict(benchmark=b, split=s, attention_type=k, value_aware=v, **pool_moments(g))
        for (b, s, k, v), g in sorted(groups.items())]
    groups.clear()
    for row in estimator_buckets:
        groups[row['benchmark'], row['split'], row['attention_type'], row['metric']].append(row)
    estimator = [dict(benchmark=b, split=s, attention_type=k, metric=m, **quantile_summary(g))
        for (b, s, k, m), g in sorted(groups.items())]
    sol = guarded_sol_rows(json.loads((root/'screen_summary.json').read_text()), sol_counts)
    if not proxy or not estimator or not sol:
        raise ValueError('missing required tuning distributions')
    dest = root/'distribution_diagnostics'
    dest.mkdir(exist_ok=True)
    datasets = dict(proxy_moments=proxy, proxy_buckets=proxy_buckets,
        estimator_quantile_summaries=estimator, estimator_buckets=estimator_buckets,
        guarded_sol=sol)
    for name, rows in datasets.items():
        _write(dest/f'{name}.json', rows)
        csv_write(dest/f'{name}.csv', rows)
    text = '# Distribution and estimator checks — tuning data only\n\n'
    text += 'These are sampled shared-dense-state diagnostics, not end-to-end benchmark results. '
    text += 'Calibration and development remain separate. No threshold, pooling choice or algorithm is changed.\n\n'
    text += '## Gaussian proxy shape\n\n'
    text += 'Eligible-candidate-weighted raw moments are combined before calculating central skew and excess kurtosis. '
    text += 'A Gaussian has skew 0 and excess kurtosis 0; standardized mean/variance alone do not establish Gaussianity. '
    text += 'These dependent observations are descriptive, not independent-sample normality tests.\n\n'
    text += table(proxy, ['benchmark', 'split', 'attention_type', 'value_aware', 'count', 'mean', 'std', 'central_skew', 'excess_kurtosis']) + '\n\n'
    text += '## Gaussian target versus achieved tile skipping\n\n'
    text += 'Counts include nonempty repair and structural/boundary effects. β is never calibrated. '
    text += 'These corrected masks restore tiles using only the declared proxy, not dense attention mass. '
    text += 'Top-k at the same proxy is included in the CSV to separate proxy quality from distribution-based budget selection.\n\n'
    text += table([r for r in sol if r['mode'] == 'gaussian'],
        ['benchmark', 'split', 'attention_type', 'proxy', 'target', 'beta', 'actual', 'mass', 'rescued_row_fraction']) + '\n\n'
    text += '## Mass-bound looseness and worst-row inflation\n\n'
    text += 'Saved buckets contain log-ratio quantiles, not full samples. The summary below is the observation-count-weighted average '
    text += 'of each bucket’s median. It is **not the median of the pooled distribution**. Full per-layer/step buckets and p10/p90/p99 '
    text += 'summaries are exported. Mass-bound ratios compare the count×exp(max) estimate with exact online mass on the same dense '
    text += 'preceding state, excluding absent preceding support. Worst/mean ratios compare per-query risks within physical128-query tiles.\n\n'
    text += table(estimator, ['benchmark', 'split', 'attention_type', 'metric', 'observations',
        'count_weighted_bucket_p50', 'min_bucket_p50', 'max_bucket_p50']) + '\n\n'
    text += 'Large positive mass-bound ratios support testing exact online mass; worst-row inflation identifies a possible aggregation '
    text += 'bottleneck but does not establish that relaxing the all-row safeguard is safe. Current candidate execution keeps that safeguard.\n'
    (dest/'report.md').write_text(text)
    for name in ('screen_summary.json', 'screen_analysis_audit.json'):
        path = root/name
        sources[str(path)] = sha(path.read_bytes())
    _write(dest/'audit.json', dict(complete=True, heldout_used=False,
        fingerprint=execution['fingerprint'], sources=sources,
        code_sha256=sha(Path(__file__).read_bytes()), counts={k: len(v) for k, v in datasets.items()},
        scope='read-only calibration/development diagnosis; not final accuracy'))
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    data = build(args.output)
    print(json.dumps({k: len(v) for k, v in data.items()}))


if __name__ == '__main__':
    main()
