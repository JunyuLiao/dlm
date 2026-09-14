"""Audit and expose all proposed methods, without rewriting previous results.

These are prior AIME6 / LongBench-v1 calibration results, NOT the new final
AIME30 / LongBench-v2 comparison. Reuse the original scorers, aggregation and
raw calibration-provenance verifier; never select a policy using final scores.
"""
import argparse
import json
from pathlib import Path

from experiments.diffusion_gemma_value_aware.protocol import (
    ROOT as PREVIOUS, fingerprint, prepare, score, sha,
)
from experiments.diffusion_gemma_value_aware.policy_audit import (
    point_for, verify_measurement,
)
from experiments.diffusion_gemma_value_aware.report import (
    inspect_output, summary, csv_write, table,
)
from experiments.diffusion_gemma_value_aware.execution import provenance
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write

ROOT = Path('results/diffusion_gemma_value_aware_followup')
METHODS = ('blasst_original', 'blasst_aggressive', 'value', 'mass',
    'mass_value', 'risk', 'aligned', 'centered', 'compensate', 'zero_pv', 'mass_exact')


def audited_policy_group(previous, setup, name, benchmark, target=.5):
    path = previous/'verified_policies'/benchmark/f'{name}_s{round(target*100)}.json'
    policy = json.loads(path.read_text())
    if (policy['name'], policy['benchmark'], policy['target']) != (name, benchmark, target):
        raise ValueError('policy path/identity mismatch')
    fp = fingerprint(previous)
    if policy['fingerprint'] != fp:
        raise ValueError('previous scientific sources changed')
    point = point_for(policy)
    stage = point.get('verification_stage', policy.get('verification_stage', 'calibration'))
    if stage not in ('calibration', 'original_boundary_repair'):
        raise ValueError('only calibration sources may support policy reuse')
    rows = [r for r in setup['calibration'] if r['benchmark'] == benchmark]
    sources = {str(path): sha(path.read_bytes())}
    outputs, measured = [], []
    for row in rows:
        source = shard_path(previous, stage, point['source_condition'], row['id'])
        dense_source = shard_path(previous, 'screen', 'dense', row['id'])
        out = json.loads(source.read_text())
        dense = json.loads(dense_source.read_text())
        for p, d, config in ((source, out, policy['config']), (dense_source, dense, {})):
            condition = dict(config=config, thresholds={benchmark: d['thresholds']}, **provenance(config))
            _, _, errors = inspect_output(d, row, fp, condition)
            if errors:
                raise ValueError(f'{p}: {errors}')
            sources[str(p)] = sha(p.read_bytes())
        if dense['thresholds'] is not None:
            raise ValueError('dense reference has routing thresholds')
        outputs.append(out)
        matching, compared = token_counts(dense['completion_tokens'], out['completion_tokens'])
        measured.append(dict(id=row['id'], task=row['task'],
            accuracy=score(row, out['prediction']), dense_accuracy=score(row, dense['prediction']),
            matching=matching, compared=compared,
            exact_match=dense['completion_tokens'] == out['completion_tokens'],
            records=[r for r in out['records'] if r['probe'] == 'execution'],
            termination_reason=out['termination_reason'], output_length=len(out['completion_tokens'])))
    verify_measurement(policy, outputs, rows)
    return policy, measured, sources


def build(previous=PREVIOUS, root=ROOT):
    setup = prepare(previous)
    rows, sources = [], {}
    for benchmark in ('aime26', 'longbench'):
        for method in METHODS:
            policy, group, evidence = audited_policy_group(previous, setup, method, benchmark)
            sources.update(evidence)
            rows.append(dict(benchmark=benchmark, split='prior_calibration_only',
                method=method, target=.5, config=policy['config'], thresholds=policy['policy'],
                target_metric='pv_omission' if method in ('compensate', 'zero_pv') else 'physical_sparsity',
                **summary(group)))
    _write(root/'history'/'proposed_methods.json', rows)
    csv_write(root/'history'/'proposed_methods.csv', rows)
    _write(root/'history'/'audit.json', dict(complete=True, groups=len(rows),
        fingerprint=fingerprint(previous), sources=sources,
        scope='Prior calibration6 AIME26 and calibration10 LongBench v1 only; not new final evaluation'))
    brief = [dict(benchmark=r['benchmark'], method=r['method'], n=r['count'],
        pooling=r['config'].get('pooling', 'rms'), score=100*r['accuracy'],
        dense_score=100*r['dense_accuracy'], physical=100*r['overall_physical_sparsity'],
        pv_omission=100*r['overall_pv_omission'], mass=100*r['overall_mass'],
        output_error=r['overall_relative_error']) for r in rows]
    text = '# Prior proposed-method evidence — calibration only\n\n'
    text += ('All rows use the existing 50% target and independently calibrated '
        'local/global thresholds. AIME rows cover six tuning questions, not all30. '
        'LongBench rows are the previous v1 calibration10, **not v2**. These '
        'results must not be substituted for the requested new final comparison.\n\n')
    text += table(brief, ['benchmark', 'method', 'pooling', 'n', 'score', 'dense_score',
        'physical', 'pv_omission', 'mass', 'output_error']) + '\n\n'
    text += ('Percentages except output error. Compensation/zero-PV preserve the '
        'denominator and have zero physical deletion. Their target is PV omission; '
        'the mass column is exact-PV mass, not preserved denominator mass. '
        'Pooling omissions use the original Config default RMS; centered/compensation '
        'decisions use vectors/radii instead of a pooled magnitude. Source hashes '
        'and raw-shard policy checks are in audit.json. No new thresholds were fitted.\n')
    (root/'history'/'report.md').write_text(text)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, default=PREVIOUS)
    p.add_argument('--output', type=Path, default=ROOT)
    args = p.parse_args()
    rows = build(args.previous, args.output)
    print(json.dumps(dict(completed_groups=len(rows), scope='prior calibration only')))


if __name__ == '__main__':
    main()
