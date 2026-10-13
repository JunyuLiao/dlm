"""Read-only 30-minute H100 monitor and final sanitized campaign publication.

It never resumes unrelated queues or modifies generation/selection settings.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time


def read(p):return json.loads(Path(p).read_text())

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def publish_complete(root,repo):
    final=read(root/'final_complete.json');protocol=read(root/'protocol.json')
    summaries=[read(p) for p in sorted((root/'reports').glob('*/*/summary.json'))]
    curve=read(root/'curve.json');curve['status']='complete'
    (root/'curve.json').write_text(json.dumps(curve,indent=2)+'\n')
    with (root/'phase_accounting.csv').open('w',newline='') as f:
        fields=['suite','arm','threshold','phase','calls','eligible_tiles','kept_tiles','skipped_tiles','kept_fraction','sparsity']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s in summaries:
            for phase,a in s['work']['phases'].items():
                w.writerow(dict(suite=s['suite'],arm=s['arm'],threshold=s['threshold'],phase=phase,**a))
    with (root/'cumulative_canvas_work.csv').open('w',newline='') as f:
        fields=['suite','arm','threshold','canvas_prefix','items','eligible_tiles','kept_tiles','sparsity','mean','median','p10','p90']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s in summaries:
            for a in s['work']['cumulative_common_prefix']:
                d=a['item_kept_distribution'];w.writerow({k:v for k,v in dict(suite=s['suite'],arm=s['arm'],threshold=s['threshold'],**a,**d).items() if k in fields})
    # Matched early prefixes use actual denoising ordinals and a common visible
    # canvas cohort. Lifecycle IDs (which include prefill) are never the axis.
    early_rows=[]
    for suite in ['longbench_v2','aime26','math500']:
        cs=next(x for x in summaries if x['suite']==suite and x['arm']=='current_v31_control')
        def load_records(summary):
            out={}
            for rel in summary['attempts']:
                for line in (root/rel/'records.jsonl').open():
                    r=json.loads(line)
                    k=f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
                    m=r['receipts']['adapter'].get('value_canvas_phase_tiles')
                    if not m:return None
                    ranked=[{'kept':sum(p['kept_tiles'] for p in m[cid].values()),
                             'eligible':sum(p['eligible_tiles'] for p in m[cid].values())}
                            for cid in sorted(m,key=int)]
                    out[k]={'visible_C':r['canvases'],'ranked':ranked}
            return out
        c=load_records(cs)
        if c is None:continue
        for s in summaries:
            if s['suite']!=suite or s['threshold'] is None:continue
            v=load_records(s)
            if v is None:continue
            for prefix in [1,2,4,8,12,16,20]:
                keys=[k for k in c if c[k]['visible_C']>=prefix and v[k]['visible_C']>=prefix
                      and len(c[k]['ranked'])>=prefix and len(v[k]['ranked'])>=prefix]
                if not keys:continue
                ck=[sum(x['kept'] for x in c[k]['ranked'][:prefix]) for k in keys]
                vk=[sum(x['kept'] for x in v[k]['ranked'][:prefix]) for k in keys]
                ce=sum(sum(x['eligible'] for x in c[k]['ranked'][:prefix]) for k in keys)
                ve=sum(sum(x['eligible'] for x in v[k]['ranked'][:prefix]) for k in keys)
                import numpy as np
                diff=np.asarray(vk)-np.asarray(ck)
                early_rows.append(dict(suite=suite,arm=s['arm'],threshold=s['threshold'],canvas_prefix=prefix,
                    common_items=len(keys),control_eligible=ce,candidate_eligible=ve,control_kept=sum(ck),candidate_kept=sum(vk),
                    control_sparsity=1-sum(ck)/ce,candidate_sparsity=1-sum(vk)/ve,
                    relative_kept_delta=(sum(vk)-sum(ck))/sum(ck),
                    item_delta_median=float(np.median(diff)),item_delta_p10=float(np.quantile(diff,.1)),
                    item_delta_p90=float(np.quantile(diff,.9))))
    (root/'early_work_matched_cohorts.json').write_text(json.dumps(early_rows,indent=2)+'\n')
    if early_rows:
        with (root/'early_work_matched_cohorts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(early_rows[0]));w.writeheader();w.writerows(early_rows)
    with (root/'paired_comparisons.csv').open('w',newline='') as f:
        fields=['suite','arm','threshold','delta_pp','problem_clusters','ci95_low_pp','ci95_high_pp','mcnemar_exact_p','output_hash_mismatches','output_token_count_mismatches','denoising_call_mismatches']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for s in summaries:
            p=s.get('paired_vs_v31')
            if p:
                w.writerow({k:v for k,v in dict(suite=s['suite'],arm=s['arm'],threshold=s['threshold'],ci95_low_pp=p['problem_cluster_bootstrap_ci95_pp'][0],ci95_high_pp=p['problem_cluster_bootstrap_ci95_pp'][1],**p).items() if k in fields})
    relative=str(root.relative_to(repo));best=final['best_shared_threshold']
    lines=['',f'## V2 local threshold sweep completed — {protocol["date"]}','',
        f'Campaign `{relative}` completed all 503 LongBench-v2 items, 30 AIME26 problems',
        '(seeds 42/43/44), and the fixed 100-item MATH500 subset for seven uniform',
        'V2 thresholds plus fresh matched V31/FULL/PIECEWISE/all-kept controls.',
        f'Frozen source commit `{protocol["source_commit"]}`; private execution bytes',
        'and runtime/input hashes are in `protocol.json`. Clean timing remains paused.',
        f'Best shared tested threshold: `{best}`. This is exploratory full-panel tuning,',
        'not a held-out confirmation or proof of a global optimum. Native FA4, GLOBAL-only',
        'routing, sampler, and exact V2 formula are unchanged. Absolute FA4 kept tiles',
        'and selector discovery/selection work are reported separately.','']
    for suite in ['longbench_v2','aime26','math500']:
        ctrl=next(x for x in summaries if x['suite']==suite and x['arm']=='current_v31_control')
        win=final['workload_winners'][suite]
        label='v2_tau_'+format(win['threshold'],'.6f').rstrip('0').replace('.','p')
        cand=next(x for x in summaries if x['suite']==suite and x['arm']==label)
        p=cand['paired_vs_v31'];a=cand['work'];c=ctrl['work']
        lines.append(f"- {suite}: workload-best tau={win['threshold']}, {cand['correct']}/{cand['items']} vs V31 {ctrl['correct']}/{ctrl['items']}; N/problem {a['mean_N']:.2f} vs {c['mean_N']:.2f}; kept-work ratio {cand['global_kept_ratio_vs_v31']:.4f}; paired delta {p['delta_pp']:+.3f} pp, problem-bootstrap CI {p['problem_cluster_bootstrap_ci95_pp']}.")
    lines+=['','Done: qualification, all frozen thresholds/controls, midpoint refinement,',
            'complete bound scoring, source/fairness receipts and sanitized accounting.',
            'Running: none. Next: independent confirmation of a chosen accuracy/work',
            'configuration before clean timing; no existing paused campaign is resumed.','']
    text='\n'.join(lines)
    # Record only when every scorer and final marker have completed.
    for path in ['HANDOFF.md','docs/DECISIONS.md','docs/RESULTS_LEDGER.md']:
        p=repo/path
        if f'Campaign `{relative}` completed' not in p.read_text():
            with p.open('a') as f:f.write(text)
    (root/'documentation_complete.json').write_text(json.dumps({'status':'complete','files':['HANDOFF.md','docs/DECISIONS.md','docs/RESULTS_LEDGER.md']},indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--pid',type=int,required=True);ap.add_argument('--interval',type=int,default=1800)
    args=ap.parse_args();root=args.study.resolve();repo=Path(__file__).resolve().parents[1]
    while True:
        state=read(root/'state.json') if (root/'state.json').exists() else {}
        attempt=root/state.get('attempt','unused')
        records=attempt/'records.jsonl'
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
        item=dict(time_unix=time.time(),coordinator_alive=alive(args.pid),state=state,
            current_records=sum(1 for _ in records.open()) if records.exists() else 0,
            completed_attempts=len(list(root.glob('runs/*/*/*/attempt*/attempt_complete.json'))),
            qualified_attempts=len(list(root.glob('qualification/*/*/*/attempt*/attempt_complete.json'))),
            completed_summaries=len(list((root/'reports').glob('*/*/final_complete.json'))),
            gpu_apps=apps.stdout.strip(),gpu_query_returncode=apps.returncode,
            final_complete=(root/'final_complete.json').exists())
        with (root/'monitor.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
        if item['final_complete']:
            publish_complete(root,repo);return
        if not item['coordinator_alive']:
            (root/'monitor_alert.json').write_text(json.dumps({'status':'needs_resume','last_state':state,'partial_attempt_retained':True},indent=2)+'\n');return
        time.sleep(args.interval)

if __name__=='__main__':main()
