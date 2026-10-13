"""One-factor paired contrasts from a validated completed native study.

No scoring, generation, source-study mutation, or unbalanced main-effect fit.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INTERVALS = (1, 2, 4, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_name(mode, progress, interval):
    return f'{mode}_global_{"progress" if progress else "no_progress"}_dense_s{interval}'


def factorial():
    return [dict(arm=arm_name(mode, progress, interval), mode=mode,
                 progress_dense=progress, interval=interval)
            for progress in (True, False) for mode in ('raw', 'stable')
            for interval in INTERVALS]


def cell_key(row):
    seed = row['panel_seed'] if 'panel_seed' in row else row['seed']
    return f'{row["dataset"]}|{row["index"]}|{seed}|{row.get("repeat", 0)}'


def contrast(candidate, reference, score, refscore):
    if set(candidate) != set(reference) or set(score) != set(candidate) or set(refscore) != set(candidate):
        raise ValueError('a paired contrast requires identical cell coverage')
    clusters = defaultdict(list)
    for key, row in candidate.items():
        clusters[row['index']].append(key)
    ids = sorted(clusters)
    acc = np.array([sum(int(score[k]) - int(refscore[k]) for k in clusters[i]) for i in ids])
    sizes = np.array([len(clusters[i]) for i in ids])
    gen = np.random.default_rng(51994)
    samples = gen.integers(len(ids), size=(10000, len(ids)))
    delta = 100 * acc[samples].sum(1) / sizes[samples].sum(1)
    result = dict(cells=len(candidate), problem_clusters=len(ids),
                  wins=sum(score[k] and not refscore[k] for k in candidate),
                  losses=sum(refscore[k] and not score[k] for k in candidate),
                  correct_delta=int(acc.sum()), accuracy_delta_pp=100 * acc.sum() / sizes.sum(),
                  accuracy_delta_ci95_pp=np.quantile(delta, [.025, .975]).tolist())
    for label, value in [('N', lambda r: r['denoise_forwards']),
                         ('NC', lambda r: r['denoise_forwards'] / max(1, r['canvases'])),
                         ('T', lambda r: r['output_tokens'])]:
        a = np.array([sum(value(candidate[k]) for k in clusters[i]) for i in ids], dtype=float)
        b = np.array([sum(value(reference[k]) for k in clusters[i]) for i in ids], dtype=float)
        ratios = a[samples].sum(1) / b[samples].sum(1)
        result[label + '_ratio'] = float(a.sum() / b.sum())
        result[label + '_ratio_ci95'] = np.quantile(ratios, [.025, .975]).tolist()
    return result


def load_native(study):
    final = json.loads((study / 'final_complete.json').read_text())
    if final['status'] != 'complete' or not final['official_scoring_complete']:
        raise RuntimeError('source study is incomplete')
    identity_file = study / 'specs/execution_identity.json'
    if sha(identity_file) != final['source_identity_sha256']:
        raise RuntimeError('completed source identity changed')
    for relative, digest in json.loads(identity_file.read_text())['source_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f'validated generation source changed: {relative}')
    completion = json.loads((study / 'native_generation_complete.json').read_text())
    expected = {cell_key(r) for r in json.loads((study / 'private/cells_native_panel154.json').read_text())}
    reports = {(r['arm'], r['workload']): r for r in json.loads((study / 'reports/native_workload_results.json').read_text())}
    rows, scores, hashes = {}, {}, {}
    for entry in completion['shards']:
        leaf = study / entry['path']
        for name, digest in entry['artifacts_sha256'].items():
            if sha(leaf / name) != digest:
                raise RuntimeError('native artifact identity changed')
        receipt = json.loads((leaf / 'receipt.json').read_text())
        if receipt['status'] != 'complete' or receipt['validation_errors']:
            raise RuntimeError('unclean native receipt')
        target = rows.setdefault(entry['arm'], {})
        for line in (leaf / 'records.jsonl').read_text().splitlines():
            row = json.loads(line)
            key = cell_key(row)
            if key in target:
                raise RuntimeError('duplicate native cell')
            target[key] = row
    for arm in completion['arms']:
        if set(rows[arm]) != expected:
            raise RuntimeError('native cell coverage changed')
        scores[arm] = {}
        for dataset, prefix in [('longbench_v2_0shot_think', 'longbench'), ('aime26', 'aime')]:
            path = study / 'scoring' / arm / (prefix + '.official.json')
            sc = next(iter(json.loads(path.read_text()).values()))
            planned = {k for k in expected if k.startswith(dataset + '|')}
            if set(sc) != planned or any(type(v) is not bool for v in sc.values()):
                raise RuntimeError('official score coverage/schema mismatch')
            if sum(sc.values()) != reports[arm, dataset]['correct']:
                raise RuntimeError('official scores disagree with published native report')
            scores[arm][dataset] = sc
            hashes[str(path.relative_to(study))] = sha(path)
    for name in ('final_complete.json', 'native_generation_complete.json', 'reports/native_workload_results.json', 'reports/fairness_audit.json'):
        hashes[name] = sha(study / name)
    if hashes['reports/fairness_audit.json'] != final['fairness_audit_sha256']:
        raise RuntimeError('source fairness identity changed')
    return rows, scores, reports, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-study', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    source, study = args.source_study.resolve(), args.study.resolve()
    output = study / 'reports/native_factor_contrasts.json'
    if output.exists():
        raise RuntimeError('preserve completed factor analysis; choose a new attempt')
    rows, scores, reports, hashes = load_native(source)
    contrasts = []
    def add(factor, candidate, reference, held):
        if candidate not in rows or reference not in rows:
            return
        for ds in ('longbench_v2_0shot_think', 'aime26'):
            a = {k: r for k, r in rows[candidate].items() if r['dataset'] == ds}
            b = {k: r for k, r in rows[reference].items() if r['dataset'] == ds}
            contrasts.append(dict(factor=factor, candidate=candidate, reference=reference,
                                 held_fixed=held, workload=ds,
                                 **contrast(a, b, scores[candidate][ds], scores[reference][ds])))
    for progress in (True, False):
        for interval in INTERVALS:
            add('stability', arm_name('stable', progress, interval), arm_name('raw', progress, interval),
                dict(progress_dense=progress, interval=interval))
    for mode in ('raw', 'stable'):
        for interval in INTERVALS:
            add('progress', arm_name(mode, True, interval), arm_name(mode, False, interval),
                dict(mode=mode, interval=interval))
        for progress in (True, False):
            for i, reference_interval in enumerate(INTERVALS):
                for interval in INTERVALS[i + 1:]:
                    add('interval', arm_name(mode, progress, interval), arm_name(mode, progress, reference_interval),
                        dict(mode=mode, progress_dense=progress))
    add('progress_without_cheap_refresh', 'v31_progress_dense_control', 'v31_no_progress_dense_hold', dict(interval=0, updates_enabled=False))
    for entry in factorial():
        add('cheap_refresh_vs_matched_hold', entry['arm'],
            'v31_progress_dense_control' if entry['progress_dense'] else 'v31_no_progress_dense_hold',
            dict(progress_dense=entry['progress_dense']))
    grid = [dict(**r, native_available=r['arm'] in rows,
                 native_results=[{k: reports[r['arm'], ds][k] for k in ('workload', 'correct', 'cells', 'N_mean', 'NC_mean', 'output_tokens_mean')}
                                 for ds in ('longbench_v2_0shot_think', 'aime26')] if r['arm'] in rows else []) for r in factorial()]
    result = dict(schema='v31_paired_factor_contrasts_v1', status='complete',
                  source_study=str(source.relative_to(ROOT)), source_artifacts_sha256=hashes,
                  analysis_source_sha256=sha(__file__), bootstrap_seed=51994, bootstrap_replicates=10000,
                  data_status='previously examined 64-problem LB development subset; 30 AIME problems x 3 seeds',
                  balanced_factorial=False, progress_with_cheap_refresh_identifiable=False,
                  factorial=grid, contrasts=contrasts,
                  interpretation='Only available one-factor contrasts; no unbalanced pooled main effects or post-hoc promotion gates.')
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status='complete', factorial_cells=len(grid), native_available=sum(r['native_available'] for r in grid), contrasts=len(contrasts))))


if __name__ == '__main__':
    main()
