"""Separate actual CUDA activity from profiler annotation spans, CPU only."""
import argparse
import json
from pathlib import Path
from scripts.v31_value_pause_analysis import read, sha, require


def union_ms(events):
    total=0.;end=-float('inf')
    for a,b in sorted((e['ts'],e['ts']+e['dur']) for e in events):
        total+=max(0.,b-max(end,a));end=max(end,b)
    return total/1000


def trace(path):
    events=read(path)['traceEvents']
    activity=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e]
    kernels=[e for e in activity if e['cat']=='kernel']
    cpu=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('stage:')]
    gpu=[e for e in events if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('stage:')]
    stages={}
    for e in cpu:
        g=next((x for x in gpu if x['name']==e['name']),None)
        inside=[x for x in activity if g and x['ts']>=g['ts']-.1 and x['ts']+x['dur']<=g['ts']+g['dur']+.1]
        stages[e['name']]=dict(cpu_scope_ms=e['dur']/1000,gpu_scope_span_ms=g['dur']/1000 if g else 0,
                              gpu_activity_union_ms=union_ms(inside),activity_events=len(inside))
    span=(max(x['ts']+x['dur'] for x in activity)-min(x['ts'] for x in activity))/1000
    return dict(trace_sha256=sha(path),kernels=len(kernels),copies_memsets=len(activity)-len(kernels),
                kernel_union_ms=union_ms(kernels),activity_union_ms=union_ms(activity),activity_span_ms=span,
                activity_gap_ms=span-union_ms(activity),stages=stages,
                interpretation='Chrome kernel/copy/memset categories only. Excludes gpu_user_annotation spans, which include gaps and double count activity in raw profiler event totals. Instrumented timing, not production timing.')


def run(root):
    output=root/'cost_analysis/attempt001';require(not output.exists(),'preserve attempt')
    profile=root/'cost_profile/attempt003';data=read(profile/'summary.json')
    cpu=read(root/'cpu_analysis/attempt001/summary.json')
    result=dict(status='complete_cost_diagnostic',profiles={},break_even={})
    for r in data['results']:
        req=r['request']
        result['profiles'][req]=dict(prefix=r['prefix'],numerator_bytes=r['numerator_bytes'],
            warmed_ms={k:r[k]['host_ms'] for k in ('moments','selection','reference_selection','dense_selected_reductions','graph_selection')},
            moments=trace(profile/f'private/{req}_moments.json'),selection=trace(profile/f'private/{req}_selection.json'),
            graph_parity=r['graph_parity'])
    c=cpu['clean']['summaries']['v31_bookend_mean'];a=cpu['audit']['summaries']['v31_bookend_mean']
    for method in ('full_swap8','residual_direction_guard'):
        m=cpu['clean']['summaries'][method];audit=cpu['audit']['summaries'][method];cost=cpu['value_cost_audit'][method]
        s0=c['S']/c['N'];s1=m['S']/m['N']
        result['break_even'][method]=dict(current_N_ratio=m['N']/c['N'],current_generation_ratio=m['S']/c['S'],current_request_ratio=m['W']/c['W'],
            amortized_step_ms=[s0*1000,s1*1000],
            needed_N_reduction_at_observed_amortized_cost=1-s0/s1,
            zero_value_observation_audit_proxy_S=audit['S']-cost['value_s_per_problem'],
            zero_value_observation_audit_proxy_delta_S=audit['S']-cost['value_s_per_problem']-a['S'],
            interpretation='Required N change holds amortized cost and prefill mix fixed; not a controlled forward price. Removing audit event elapsed from audit S is an optimistic accounting counterfactual, not measured clean latency or a formal implementation lower bound.')
    old=cpu['old']['comparisons']['full_swap8'];old_anchor=20.515625
    result['old_swap_savings_at_current_control_step_cost']=dict(
        calls_saved=old['control']['N']-old['candidate']['N'],
        saved_s=(old['control']['N']-old['candidate']['N'])*c['S']/c['N'],
        old_anchors_per_problem=old_anchor,
        maximum_extra_ms_per_anchor=(old['control']['N']-old['candidate']['N'])*c['S']/c['N']/old_anchor*1000,
        interpretation='Cross-study illustrative break-even: assumes old trajectory saving, current control amortized S/N and old observation count; not an observed speedup.')
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (output/'final_complete.json').write_text(json.dumps(dict(status='complete',source_sha256=sha(__file__),summary_sha256=sha(output/'summary.json'),
        profile_summary_sha256=sha(profile/'summary.json'),cpu_summary_sha256=sha(root/'cpu_analysis/attempt001/summary.json')),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
