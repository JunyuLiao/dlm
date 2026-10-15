"""Read-only outcome/trajectory decomposition after completed native scoring.

Outcome-conditioned cohorts explain an observed aggregate, not a deployable
oracle selector or an unbiased causal comparison. Unconditional clustered
intervals remain in the frozen native development report.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from scripts import v31_value_decision_replay as io
from scripts import v31_v2_local_threshold_sweep as base
from scripts.v31_value_native_study import CONTROLS


def cohort_name(suite,control,candidate):
    if suite=='ruler128k':
        if candidate>control:return 'candidate_score_higher'
        if candidate<control:return 'control_score_higher'
        return 'both_full_credit' if control==100 else 'equal_partial_score'
    if control and candidate:return 'both_correct'
    if candidate:return 'candidate_only_correct'
    if control:return 'control_only_correct'
    return 'both_wrong'


def load_arm(study,suite,arm):
    dest=study/f'reports/{suite}/{arm}'
    done=io.read(dest/'final_complete.json')
    if done['summary_sha256']!=io.sha(dest/'summary.json'):raise RuntimeError('completed native summary changed')
    score=study/f'scoring/{suite}/{arm}/official.official.json'
    if done['official_sha256']!=io.sha(score):raise RuntimeError('official scores changed')
    summary=io.read(dest/'summary.json');rows=[]
    for rel in summary['attempts']:
        leaf=study/rel;marker=io.read(leaf/'attempt_complete.json')
        if marker['records_sha256']!=io.sha(leaf/'records.jsonl'):raise RuntimeError('completed trajectory records changed')
        rows.extend(base.jsonl(leaf/'records.jsonl'))
    records={base.key(r):r for r in rows};values=io.read(score)[arm]
    if len(records)!=len(rows) or records.keys()!=values.keys():raise RuntimeError('trajectory/score cohort mismatch')
    return records,values


def run(study):
    done=io.read(study/'final_complete.json');protocol=io.read(study/'protocol.json')
    if done['protocol_sha256']!=io.sha(study/'protocol.json'):raise RuntimeError('native protocol changed')
    dest=study/'outcome_trajectory_analysis'
    if dest.exists():
        marker=io.read(dest/'analysis_complete.json')
        if marker['summary_sha256']!=io.sha(dest/'summary.json'):raise RuntimeError('completed postmortem changed')
        return
    result=dict(status='complete',native_complete_sha256=io.sha(study/'final_complete.json'),comparisons=[],
        scope='descriptive outcome-conditioned problem-trials; repeated seeds are not independent; no oracle selection or causal claim',
        timing='paused; N reductions do not establish speedup')
    lines=['# Native outcome and trajectory decomposition','',result['scope']+'.', '',
        'Unconditional paired clustered intervals are in the development report. This table checks whether call changes occur on successful answers, changed answers, or failures. RULER retains fractional scores.', '',
        '| Workload | Variant | Outcome cohort | Trials | Control N/trial | Variant N/trial | Delta total N | Delta canvases/trial | Delta output tokens/trial |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for suite in protocol['suites']:
        control,cs=load_arm(study,suite,'current_v31_control')
        for arm in protocol['arms']:
            if arm in CONTROLS:continue
            candidate,vs=load_arm(study,suite,arm)
            if candidate.keys()!=control.keys():raise RuntimeError('paired cohorts differ')
            groups=defaultdict(list)
            for key in control:groups[cohort_name(suite,float(cs[key]),float(vs[key]))].append(key)
            delta_sum=0
            for name,keys in sorted(groups.items()):
                n=len(keys);cn=sum(control[k]['denoise_forwards'] for k in keys);vn=sum(candidate[k]['denoise_forwards'] for k in keys)
                delta_sum+=vn-cn
                r=dict(suite=suite,arm=arm,cohort=name,trials=n,control_N=cn,candidate_N=vn,delta_N=vn-cn,
                    control_N_per_trial=cn/n,candidate_N_per_trial=vn/n,
                    delta_C_per_trial=sum(candidate[k]['canvases']-control[k]['canvases'] for k in keys)/n,
                    delta_T_per_trial=sum(candidate[k]['output_tokens']-control[k]['output_tokens'] for k in keys)/n,
                    control_C=sum(control[k]['canvases'] for k in keys),candidate_C=sum(candidate[k]['canvases'] for k in keys),
                    output_hash_mismatches=sum(control[k]['output_hash']!=candidate[k]['output_hash'] for k in keys))
                r['control_N_per_C']=cn/r['control_C'];r['candidate_N_per_C']=vn/r['candidate_C']
                result['comparisons'].append(r)
                lines.append(f"| {suite} | {arm} | {name} | {n} | {cn/n:.3f} | {vn/n:.3f} | {vn-cn:+d} | {r['delta_C_per_trial']:+.3f} | {r['delta_T_per_trial']:+.1f} |")
            if sum(map(len,groups.values()))!=len(control):raise RuntimeError('outcome partition does not conserve cells')
            if delta_sum!=sum(candidate[k]['denoise_forwards']-control[k]['denoise_forwards'] for k in control):
                raise RuntimeError('outcome decomposition does not conserve native calls')
    io.write(dest/'summary.json',result,True)
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    io.write(dest/'analysis_complete.json',dict(status='complete',summary_sha256=io.sha(dest/'summary.json'),
        native_complete_sha256=io.sha(study/'final_complete.json')),True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    run(p.parse_args().study.resolve())
