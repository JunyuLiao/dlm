"""Reduce the v22 frozen-offset Q16 validation and native attention-share receipts.

Stdlib only. Selection is NOT redone here: the threshold was frozen from the
two LongBench call-3 calibration states. The matched-work error ratio is a
descriptive log-linear interpolation between measured offsets, never a
measured point. Zero-attention oracle logits are invalid; only elapsed time is
used, and LOCAL/all-zero oracles are flagged because zeroed attention changes
MoE routing and therefore expert load.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def _median(values):
    values = sorted(values)
    n = len(values)
    return None if not n else (values[n//2] if n % 2 else (values[n//2-1]+values[n//2])/2)


def interpolate_error_at_equal_work(points):
    """points: [(work_ratio, error_ratio)] in offset order; None if not bracketed."""
    for (w1, e1), (w2, e2) in zip(points, points[1:]):
        if w1 > 0 and w2 > 0 and w1 != w2 and (w1-1)*(w2-1) <= 0:
            t = -math.log(w1)/(math.log(w2)-math.log(w1))
            return e1+t*(e2-e1)
    return None


def reduce(receipts):
    rows, shares = [], []
    commits = set()
    for host, path in receipts.items():
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('completeness', {}).get('status') != 'complete':
            raise ValueError(f'{host}: validation receipt is not complete')
        if data.get('screen') != 'frozen_q16_offsets_validation_and_attention_share':
            raise ValueError(f'{host}: not a validation receipt')
        commits.add(data['source_commit'])
        for key, group in data['groups'].items():
            for idx in group['selected_calls']:
                screen = group['calls'][str(idx)]['layers']['5']['geometry']['comparison'][
                    'matched_q16_calibration']
                if screen['offsets'] != [0.0, -0.25, -0.5, -1.0, -2.0]:
                    raise ValueError('offsets differ from the frozen contract')
                base = screen['coarse_baseline']
                bw = base['retained_legal_pairs']
                be = base['output_vs_full_fp32']['relative_l2']
                if not bw or not be:
                    raise ValueError(f'{host} {key} call {idx}: zero coarse work or error')
                points = [(c['retained_legal_pairs']/bw, c['output_vs_full_fp32']['relative_l2']/be)
                          for c in screen['q16_candidates']]
                selected = screen['q16_candidates'][3]  # frozen rule result: offset -1.0
                assert selected['offset'] == -1.0
                rows.append(dict(
                    host=host, dataset=group['dataset'], group=key, call=idx,
                    role=('calibration' if group['dataset'] == 'longbench_v2' and idx == 3 else 'validation'),
                    keys=screen['shape']['keys'],
                    coarse_kept_fraction=bw/screen['denominator_legal_pairs'],
                    coarse_relative_l2=be,
                    coarse_row_p99=base['output_relative_l2_per_row']['p99'],
                    selected_offset=-1.0, selected_work_ratio=points[3][0],
                    selected_error_ratio=points[3][1],
                    selected_kept_fraction=selected['retained_legal_pairs']/screen['denominator_legal_pairs'],
                    selected_row_p99=selected['output_relative_l2_per_row']['p99'],
                    selected_row_max=selected['output_relative_l2_per_row']['max'],
                    selected_removed_mass_p99=selected['removed_mass_per_row']['p99'],
                    coarse_removed_mass_p99=base['removed_mass_per_row']['p99'],
                    frontier=';'.join(f'{w:.4f}/{e:.4f}' for w, e in points),
                    descriptive_error_ratio_at_equal_work=interpolate_error_at_equal_work(points)))
                share = group['attention_share'][str(idx)]['summary']
                native = share['native']
                shares.append(dict(
                    host=host, dataset=group['dataset'], group=key, call=idx,
                    keys=screen['shape']['keys'],
                    native_forward_ms=native['median_event_ms'],
                    global_attention_ms=native['GLOBAL_attention_in_forward_median_ms'],
                    local_attention_ms=native['LOCAL_attention_in_forward_median_ms'],
                    global_share=native['GLOBAL_attention_in_forward_median_ms']/native['median_event_ms'],
                    local_share=native['LOCAL_attention_in_forward_median_ms']/native['median_event_ms'],
                    no_global_ratio=share['no_global_attention']['ratio_to_native'],
                    no_local_ratio_moe_confounded=share['no_local_attention']['ratio_to_native'],
                    no_attention_ratio_moe_confounded=share['no_attention']['ratio_to_native']))
    if len(commits) != 1:
        raise ValueError(f'receipts come from different source commits: {commits}')
    summary = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        v = [r for r in rows if r['dataset'] == dataset and r['role'] == 'validation']
        s = [r for r in shares if r['dataset'] == dataset]
        interp = [r['descriptive_error_ratio_at_equal_work'] for r in v
                  if r['descriptive_error_ratio_at_equal_work'] is not None]
        summary[dataset] = dict(
            validation_states=len(v),
            median_coarse_kept_fraction=_median([r['coarse_kept_fraction'] for r in v]),
            median_selected_kept_fraction=_median([r['selected_kept_fraction'] for r in v]),
            median_selected_work_ratio=_median([r['selected_work_ratio'] for r in v]),
            median_selected_error_ratio=_median([r['selected_error_ratio'] for r in v]),
            states_selected_less_work_and_error=sum(r['selected_work_ratio'] < 1 and r['selected_error_ratio'] < 1 for r in v),
            median_descriptive_error_ratio_at_equal_work=_median(interp),
            range_descriptive_error_ratio_at_equal_work=[min(interp), max(interp)] if interp else None,
            share_states=len(s),
            median_global_share=_median([r['global_share'] for r in s]),
            median_local_share=_median([r['local_share'] for r in s]),
            median_no_global_ratio=_median([r['no_global_ratio'] for r in s]),
            range_no_global_ratio=[min(r['no_global_ratio'] for r in s), max(r['no_global_ratio'] for r in s)])
    return dict(schema='v22_q16_validation_reduce_v1', quality_eligible=False,
                source_commit=commits.pop(), rows=rows, attention_share=shares, summary=summary,
                note=('Frozen threshold offset -1.0 was chosen on the two LB call-3 states; other '
                      'states validate it without reselection. Error is FP32 attention-output '
                      'relative L2 against full attention, not task quality. Zero-attention oracle '
                      'logits are invalid; LOCAL/all-zero oracle ratios are MoE-routing confounded.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', action='append', required=True, help='host=path')
    parser.add_argument('--out-json', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, required=True)
    args = parser.parse_args(argv)
    for out in (args.out_json, args.out_csv):
        if out.exists():
            raise SystemExit(f'refusing to overwrite {out}')
    receipts = dict(item.split('=', 1) for item in args.receipt)
    result = reduce(receipts)
    result['input_sha256'] = {host: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for host, p in receipts.items()}
    result['reducer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out_json.write_text(json.dumps(result, indent=1)+'\n', encoding='utf-8')
    with args.out_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]) + [
            k for k in result['attention_share'][0] if k not in result['rows'][0]])
        writer.writeheader()
        shares = {(r['host'], r['group'], r['call']): r for r in result['attention_share']}
        for row in result['rows']:
            writer.writerow(row | shares[(row['host'], row['group'], row['call'])])
    print(json.dumps(result['summary'], indent=1))


if __name__ == '__main__':
    main()
