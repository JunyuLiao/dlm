"""Aggregate completed native EOS and exact-work support diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    root,out=a.root,a.out;out.mkdir(parents=True,exist_ok=False)
    paths=[root/'attempt007/receipt.json',root/'attempt008/metrics.json',
           root/'attempt008/oracle.json',root/'attempt009/receipt.json']
    native,metrics,oracle,matched=[load(x) for x in paths]
    errors=[]
    for attempt,receipt_name in [('attempt007','receipt.json'),('attempt009','receipt.json')]:
        marker=load(root/attempt/'final_complete.json')
        if marker['status']!='complete' or marker['receipt_sha256']!=sha(root/attempt/receipt_name):
            errors.append('invalid final marker '+attempt)
    marker=load(root/'attempt008/final_complete.json')
    if marker['metrics_sha256']!=sha(paths[1]) or marker['oracle_sha256']!=sha(paths[2]):errors.append('operator hashes')
    if len(native['results'])!=16 or not native['observer_output_and_count_parity']:errors.append('native coverage/parity')
    for r in native['results']:
        if not all(r[k] for k in ['matches_saved_output','matches_saved_counts','observer_matches_clean_bookends']):errors.append('native mismatch')
        if len(r['steps'])!=r['counts']['denoising_forwards']:errors.append('trace count mismatch')
    if len(oracle)!=160:errors.append('operator coverage')
    if sum(r['independent_fp32_cpu_vs_cuda_v2_map_mismatches'] for r in oracle):errors.append('selector oracle mismatch')
    if sum(r['fresh_v2_vs_actual_map_mismatches'] for r in oracle):errors.append('fresh actual mismatch')
    if len(matched['results'])!=12:errors.append('matched-count coverage')
    for r in matched['results']:
        if not r['per_head_tile_counts_identical']:errors.append('budget mismatch')
        if r['route']=='saved_v2_map' and r['max_abs_logit_delta']!=0:errors.append('saved-map logit mismatch')
    groups=defaultdict(list)
    for r in metrics:groups[(r['label'],r['method'])].append(r)
    summaries={}
    for (label,method),rows in groups.items():
        summaries[label+'/'+method]=dict(cases=len(rows),
            retained_mass=st.mean(r['retained_mass']['mean'] for r in rows),
            error_over_value_rms=st.mean(r['error_over_value_rms']['mean'] for r in rows),
            error_over_dense_output=st.mean(r['relative_error_over_dense_output']['mean'] for r in rows),
            kept_tiles=sum(r['kept'] for r in rows),eligible_tiles=sum(r['eligible'] for r in rows),
            first_tile_keep_fraction=st.mean(r['tile_keep_fraction'][0] for r in rows),
            first_and_mandatory_only_heads=sum(r['first_and_mandatory_only_heads'] for r in rows),
            heads=len(rows)*16,iou_with_v2=st.mean(r['iou_with_actual_v2'] for r in rows))
    events=[]
    eos_methods=defaultdict(list)
    for r in native['results']:
        if 'same_state' not in r:continue
        first=next(x for x in r['steps'] if x['eos_writes'])
        for route in r['same_state']:eos_methods[route['route']].append(route)
        matched_rows=[x for x in matched['results'] if x['id']==r['id']]
        events.append(dict(id=r['id'],output_tokens=r['output_tokens'],first_write=first,
                           terminal=r['steps'][-1],same_state=r['same_state'],matched_count=matched_rows))
    for r in matched['results']:eos_methods[r['route']].append(r)
    eos_summaries={}
    for name,rows in eos_methods.items():
        eos_summaries[name]=dict(positions=sum(len(r['eos_argmax']) for r in rows),
            eos_argmaxes=sum(sum(r['eos_argmax']) for r in rows),
            high_confidence_eos=sum(e and p>.5 for r in rows for e,p in zip(r['eos_argmax'],r['eos_probability'])),
            mean_eos_probability=st.mean(p for r in rows for p in r['eos_probability']))
    summary=dict(status='pass' if not errors else 'fail',errors=errors,
                 cohort='HumanEval first four canonical previously examined items, seed42; diagnostic only',
                 observed_native_requests=16,native_bookend_requests=32,same_state_forwards=24,
                 matched_count_forwards=12,operator_snapshot_layers=160,
                 operator=summaries,eos=eos_summaries,events=events,
                 oracle=dict(logmass_max_abs=max(r['logmass_max_abs'] for r in oracle),
                     projected_mean_max_abs=max(r['projected_mean_max_abs'] for r in oracle),
                     map_mismatches=sum(r['independent_fp32_cpu_vs_cuda_v2_map_mismatches'] for r in oracle),
                     projected_full_v_iou=st.mean(r['projected_vs_full_v_iou'] for r in oracle),
                     consumer_fp32_max_relative_l2=max(r['consumer_vs_fp32_relative_l2'] for r in metrics if r['method']=='actual_v2')),
                 inputs=[dict(path=str(x),sha256=sha(x)) for x in paths])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(status=summary['status'],errors=errors,
                             summary_sha256=sha(out/'summary.json'),report_source_sha256=sha(__file__)),indent=2)+'\n')
    if errors:raise RuntimeError(errors)
    (out/'final_complete.json').write_text(json.dumps(dict(status='complete',receipt_sha256=sha(out/'receipt.json')),indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],eos=eos_summaries,oracle=summary['oracle']),indent=2))


if __name__=='__main__':main()
