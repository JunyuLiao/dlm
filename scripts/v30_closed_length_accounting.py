"""CPU-only accounting of closed evidence; never read completion ledgers.

No source changes, new inference, confidence intervals, or causal attribution.
Private join keys stay in memory. Public output contains aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

V18_PIN = '8704cd072582dbb766762f955152bc5533c61986'
V28_PIN = 'dcfdb8730efff131f92d0a8bbbab514ef1eedf18'
BINS = ('longbench_v2_32k', 'longbench_v2_64k', 'longbench_v2_96k')
METRICS = ('W', 'S', 'N', 'SN')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def geomean(values):
    values = list(values)
    if not values or any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('geometric mean requires nonempty finite positive values')
    return math.exp(statistics.mean(math.log(x) for x in values))


def metrics(row):
    w, s, n = (row[k] for k in ('wall_s', 'decode_span_s', 'denoise_forward_count'))
    geomean((w, s, n))
    return dict(W=w, S=s, N=n, SN=s/n)


def keyed(rows):
    result = {}
    for row in rows:
        key = (row['dataset'], row['index'], row['repeat'])
        if key in result:
            raise ValueError('duplicate private cell key')
        result[key] = row
    return result


def paired(candidate, reference, require_equal=False):
    """Intersect explicitly; preserve subset size instead of dividing unlike summaries."""
    a, b = keyed(candidate), keyed(reference)
    if require_equal and set(a) != set(b):
        raise ValueError('paired inventories differ')
    keys = set(a) & set(b)
    if not keys:
        raise ValueError('no paired cells')
    grouped = defaultdict(list)
    for key in sorted(keys):
        left, right = a[key], b[key]
        for field in ('host', 'gpu_uuid', 'engine_seed', 'deploy_commit', 'protocol_id'):
            if left[field] != right[field]:
                raise ValueError('paired execution identity differs')
        grouped[key[0]].append((left, right))
    output = []
    for dataset, pairs in sorted(grouped.items()):
        ratios = {m: geomean(metrics(a)[m]/metrics(b)[m] for a, b in pairs) for m in METRICS}
        absolute = {}
        for label, index in (('candidate', 0), ('reference', 1)):
            rows = [pair[index] for pair in pairs]
            absolute[label] = dict(
                N_mean=statistics.mean(r['denoise_forward_count'] for r in rows),
                C_mean=statistics.mean(r['commit_forward_count'] for r in rows),
                output_tokens_mean=statistics.mean(r['output_tokens'] for r in rows),
                C_over_N_geomean=geomean(r['commit_forward_count']/r['denoise_forward_count'] for r in rows),
                **{m+'_geomean': geomean(metrics(r)[m] for r in rows) for m in METRICS})
        output.append(dict(dataset=dataset, cells=len(pairs), questions=len({p[0]['index'] for p in pairs}),
                           source='raw_same_cell_pairs', ratios=ratios, absolute=absolute))
    return output


def load_v18(root):
    root = Path(root)
    arms = defaultdict(list)
    for block in (0, 1):
        for arm in ('dense', 'method', 'native', 'allkept'):
            directory = root / f'b{block}_{arm}'
            terminal = read_json(directory/'terminal.json')
            rows = [json.loads(line) for line in (directory/'records.jsonl').read_text(encoding='utf-8').splitlines()]
            if terminal.get('complete') is not True or terminal['expected_timed'] != len(rows) or terminal['completed_timed'] != len(rows):
                raise ValueError('V18 worker incomplete or row coverage differs')
            for row in rows:
                if row['deploy_commit'] != V18_PIN or row['protocol_id'] != 'v27_vllm_lb_v18b_20261002' or row['arm'] != arm:
                    raise ValueError('V18 source/protocol/arm differs')
                if row['denoise_forward_count'] != row['scheduler_denoise_forward_count'] + row['speculative_unused_denoising']:
                    raise ValueError('actual/scheduler N receipt differs')
                if not math.isclose(row['wall_s'], row['prefill_s'] + row['decode_span_s'], abs_tol=1e-8):
                    raise ValueError('request span identity differs')
                if row['graph_captures_timed'] != 0:
                    raise ValueError('timed graph capture found')
            arms[arm].extend(rows)
    if {k: len(v) for k, v in arms.items()} != dict(dense=236, method=236, native=48, allkept=48):
        raise ValueError('V18 frozen arm coverage differs')
    comparisons = []
    for a, b in (('method', 'dense'), ('method', 'native'), ('method', 'allkept'), ('native', 'dense'), ('allkept', 'native')):
        for item in paired(arms[a], arms[b], require_equal=a == 'method' and b == 'dense'):
            comparisons.append(dict(candidate=a, reference=b, **item))
    published = read_json(root/'summary/summary.json')
    for item in comparisons:
        matches = [r for r in published['comparisons'] if (r['dataset'], r['arm'], r['base']) == (item['dataset'], item['candidate'], item['reference'])]
        if matches and (len(matches) != 1 or matches[0]['cells'] != item['cells'] or any(not math.isclose(matches[0][m], item['ratios'][m], rel_tol=1e-12) for m in METRICS)):
            raise ValueError('raw paired values differ from archived published summary')
    method = arms['method']
    sparse_coverage = dict(requests=len(method), telemetry_modes=sorted({r['receipts']['method']['telemetry'] for r in method}),
                           per_tile_statistics_receipts=sorted({r['receipts']['method']['per_tile_statistics'] for r in method}),
                           actual_sparsity_available=False)
    return dict(source_commit=V18_PIN, protocol_id='v27_vllm_lb_v18b_20261002',
                coverage={k: len(v) for k, v in arms.items()}, comparisons=comparisons,
                sparsity_coverage=sparse_coverage)


def load_v28(summary_path, receipts_path):
    summary, receipt = read_json(summary_path), read_json(receipts_path)
    if receipt['generation_source_commit'] != V28_PIN or receipt['closed_workers'] != 24 or receipt['failed_workers'] != 0 or receipt['timed_requests'] != 288:
        raise ValueError('V28 frozen closed inventory differs')
    if receipt['immutable_generation_sources_verified'] is not True or receipt['graph_captures_timed'] != 0:
        raise ValueError('V28 source/capture receipt differs')
    variants = {v['variant']: {d['dataset']: d for d in v['by_dataset']} for v in summary['variants']}
    if len(variants) != 6 or any(set(ds) != set(BINS) or any(d['requests'] != 16 for d in ds.values()) for ds in variants.values()):
        raise ValueError('V28 dataset coverage differs')
    rows = []
    for comparison in summary['comparisons']:
        for d in comparison['by_dataset']:
            rows.append(dict(candidate=d['candidate'], reference=d['reference'], dataset=d['dataset'],
                             cells=d['requests'], questions=d['question_clusters'], source='published_paired_summary_only',
                             ratios={m: d[m]['ratio'] for m in METRICS}))
    # Same complete 16-cell inventories: ratio of geometric means equals paired
    # geometric mean. This is an algebraic point estimate, not new pairing/CI proof.
    for a, b in (('main_legacy', 'native'), ('main_release', 'native'), ('q64_release', 'native')):
        for dataset in BINS:
            rows.append(dict(candidate=a, reference=b, dataset=dataset, cells=16, questions=2,
                             source='same_complete_inventory_summary_geomean_ratio_no_new_ci',
                             ratios={m: variants[a][dataset][m]['geomean']/variants[b][dataset][m]['geomean'] for m in METRICS}))
    return dict(source_commit=V28_PIN, scorer_commit=receipt['scorer_source_commit'], raw_rows_locally_available=False,
                coverage=dict(workers=24, timed_requests=288, requests_per_variant=48, questions_per_bin=2),
                comparisons=rows, actual_sparsity_available=False,
                sparsity_note='No actual tile density retained in the supplied summary artifacts; unknown, not zero.')


def load_hf(summary_path):
    with Path(summary_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    arm = 'M3_R6_A64_fused_dp_async_m1ln2_c0_fa4'
    selected = [r for r in rows if r['arm'] == arm and r['base'] == 'D_fa4_allkept']
    if {r['dataset'] for r in selected} != set(BINS) or len(selected) != 3:
        raise ValueError('HF E14 summary coverage differs')
    return dict(protocol_id='v27_lb_q64c_e14_46bf7b5a765b7204', deploy_label='v27_e14_0fc74fc',
                source_commit='0fc74fcbfe7b9e58b6cf6061f7581f49585f6403',
                baseline='HF D_fa4_allkept num_splits=1; not current vLLM dynamic-causal baseline',
                comparisons=[dict(candidate=arm, reference=r['base'], dataset=r['dataset'], cells=int(r['cells']),
                                  source='published_paired_summary_rounded_4dp', ratios={m: float(r[m]) for m in METRICS}) for r in selected],
                actual_sparsity_available=False, sparsity_note='Input paired summary has no tile-density measurements.')


def load_v29(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8 or any(r['formal_performance_claim_allowed'] != 'False' for r in rows):
        raise ValueError('V29 diagnostic inventory differs')
    pins = dict(cost32k_002='39e08c521d15b458a1e4c19e98de42a312ff0c25',
                cost32k_events003='54163f4e7c8766aaa4187807420b3cf3167dba81')
    if {(r['campaign'], r['arm']) for r in rows} != {(c, a) for c in pins for a in ('dense', 'native', 'allkept', 'method')} or any(r['source_commit'] != pins[r['campaign']] or r['engine_seed'] != '28001' for r in rows):
        raise ValueError('V29 source/cell scope differs')
    return dict(scope='32K singleton profiled diagnostics only; no 64K/96K data; not formal speed evidence',
                requests=[dict(campaign=r['campaign'], arm=r['arm'], source_commit=r['source_commit'],
                               requested_graph=r['requested_graph'], N=int(r['N']), C=int(r['C']),
                               output_tokens=int(r['output_tokens']), W=float(r['W_profiled_s']),
                               S=float(r['S_profiled_s']), SN_ms=float(r['S_over_N_profiled_ms'])) for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v18-root', 'v28-summary', 'v28-receipts', 'hf-summary', 'v29-requests', 'output'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('refuse to overwrite existing accounting output')
    result = dict(schema='v30_closed_length_accounting_v1', ratio_definition='candidate/reference; paired geometric means unless explicitly summary-derived',
                  causal_claims=False, V18b=load_v18(args.v18_root), V28=load_v28(args.v28_summary, args.v28_receipts),
                  HF_E14=load_hf(args.hf_summary), V29_diagnostics=load_v29(args.v29_requests))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True, V18_rows=568, V28_scope='summary-only', V29_scope='32K-profiled-singletons')))


if __name__ == '__main__':
    main()
