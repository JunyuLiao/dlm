"""Frozen question-clustered lifecycle contrasts; no outcome-driven tuning."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,math,itertools
import numpy as np
from reuse_runtime import ROOT,save,sha
from reuse_selectors import SELECTORS,full_policy_regressions
from worker import cid

INPUTS={}
def read(p):
    INPUTS[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def kept(r):return r['work']['overall_eligible_rectangles']-r['work']['overall_skipped_rectangles']
def geom(xs):return math.exp(float(np.mean(np.log(xs)))) if xs else None
def stats(values,family=1):
    if not values:return None
    byq=defaultdict(list)
    for q,v in values:byq[q].append(v)
    qq={q:float(np.mean(v)) for q,v in byq.items()};a=np.array(list(qq.values()));rng=np.random.default_rng(2601026)
    boot=a[rng.integers(0,len(a),(20000,len(a)))].mean(1);alpha=.05/family
    return dict(arithmetic_equal_question_mean=float(a.mean()),geometric_paired_mean=geom([v for q,v in values]) if all(v>0 for q,v in values) else None,question_values=qq,question_clusters=len(qq),CI=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),confidence=1-alpha)
def table(rs):
    return dict(cells=len(rs),questions=len({r['id'] for r in rs}),correct=sum(r['correct'] for r in rs),N=sum(r['N'] for r in rs),C=sum(r['C'] for r in rs),T=sum(r['T'] for r in rs),pooled_N_per_C=sum(r['N'] for r in rs)/sum(r['C'] for r in rs),caps=sum(r['output_capped'] for r in rs),missing_final=sum(r['no_final_response'] for r in rs),extraction_failures=sum(not r['answer_parsed'] for r in rs),GLOBAL_eligible=sum(r['work']['global_eligible_tiles'] for r in rs),GLOBAL_kept=sum(r['work']['global_kept_tiles'] for r in rs),GLOBAL_skipped=sum(r['work']['global_skipped_tiles'] for r in rs),LOCAL_eligible=sum(r['work']['native_local_eligible_rectangles'] for r in rs),LOCAL_skipped=0,overall_kept=sum(kept(r) for r in rs),clean_latency_qualified=False)
def main():
    out=ROOT/'analysis001';out.mkdir(exist_ok=False)
    assert read(ROOT/'runs/treatments/final_complete.json')['passed'];manifest=read(ROOT/'branch_manifest.json');records={};refs={};dense={};pairs=[];tables=[];quality=[];factorial=[];decisions=[];physical=[];shadow=[]
    for p in (ROOT/'branch_records').glob('*.json'):
        r=read(p);assert p.stem==r['branch_identity'];records[p.stem]=r
        assert r['startup_model_calls_in_request']==r['extra_inside_execute_Python_forwards']==0
        assert sum(x['skipped'] for x in r['consumer_physical'])==r['work']['overall_skipped_rectangles']
        assert all(x['LOCAL_skipped']==x['protected_skipped']==0 and x['current_QKV'] for x in r['consumer_physical'])
        physical.append(dict(identity=p.stem,id=r['id'],seed=r['seed'],GLOBAL_eligible=r['work']['global_eligible_tiles'],GLOBAL_kept=r['work']['global_kept_tiles'],GLOBAL_skipped=r['work']['global_skipped_tiles'],LOCAL_eligible=r['work']['native_local_eligible_rectangles'],LOCAL_skipped=0,overall_denominator=r['work']['overall_eligible_rectangles'],operator_diagnostic_work=r['operator_diagnostic_work'],refreshes=r['refresh_calls'],held_calls=r['held_calls'],native_forwards=r['model_forwards'],uncommitted_tail=r['uncommitted_tail'],audit_W=r['W'],audit_S=r['S'],audit_S_per_N=r['S_per_N'],clean_policy_timing=False))
        for e in r['events']:
            if e.get('intervention') and e.get('shadow'):shadow.append(dict(identity=p.stem,id=r['id'],seed=r['seed'],**e['shadow'],guard_now=e['measurement']['feasible'],error=e['measurement']['current_error'],deleted_mass=e['measurement']['deleted_mass']))
    assert len(records)==manifest['unique_branches']
    for c in manifest['cells']:
        key=(c['id'],c['seed']);refs[key]=read(ROOT/f'runs/references/{cid(c,c["seed"])}.json')
        for arm in ['dense_full','piecewise']:dense[key,arm]=read(ROOT/f'runs/{arm}/{cid(c,c["seed"])}.json')
    labels={}
    for c in manifest['cells']:
        for l in c['labels']:labels[(c['id'],c['seed'],l['fraction'],l['selector'],l['lifecycle'])]=records[l['branch']] if l['branch'] else refs[c['id'],c['seed']]
    def policy(c,f,s,l):return labels[c['id'],c['seed'],f,s,l]
    def remain(r):return r['local_endpoint']['remaining_calls'] if r['local_endpoint'] else None
    for arm in ['A','dense_full','piecewise']:
        rs=list(refs.values()) if arm=='A' else [dense[k,arm] for k in refs]
        for task in ['all','aime26','humaneval','longbench_v2']:tables.append(dict(policy=arm,benchmark=task,**table([r for r in rs if task=='all' or r['benchmark']==task])))
    for f,s,l in itertools.product([.05,.1],SELECTORS,['SA','AS','held2','refresh2','matched_change2','held4','refresh4']):
        rs=[policy(c,f,s,l) for c in manifest['cells']]
        for task in ['all','aime26','humaneval','longbench_v2']:tables.append(dict(policy=s,fraction=f,lifecycle=l,benchmark=task,**table([r for r in rs if task=='all' or r['benchmark']==task])))
        for comparison in ['A','dense_full','piecewise','current_error','support_sham']:
            bs=[refs[c['id'],c['seed']] if comparison=='A' else dense[(c['id'],c['seed']),comparison] if comparison in ['dense_full','piecewise'] else policy(c,f,comparison,l) for c in manifest['cells']]
            transitions=Counter();reg=[]
            for r,b in zip(rs,bs):
                transitions[f'{int(b["correct"])}->{int(r["correct"])}']+=1
                if b['correct'] and not r['correct']:reg.append(dict(id=r['id'],seed=r['seed'],exposed=r['intervention_calls']>0))
            quality.append(dict(selector=s,fraction=f,lifecycle=l,versus=comparison,transitions=dict(transitions),regressions=reg,N=stats([(r['id'],r['N']/b['N']) for r,b in zip(rs,bs)]),N_per_C=stats([(r['id'],r['N_per_C']/b['N_per_C']) for r,b in zip(rs,bs)]),C=stats([(r['id'],r['C']/b['C']) for r,b in zip(rs,bs)]),T=stats([(r['id'],r['T']/b['T']) for r,b in zip(rs,bs)]),pooled_N_ratio=sum(r['N'] for r in rs)/sum(b['N'] for b in bs),quality_policy_denominator=len(rs)))
    for f,s,horizon in itertools.product([.05,.1],SELECTORS,[2,4]):
        local=[];local_A=[];ident_local=[];correct_local=[];byq=defaultdict(list);helds=[];refreshes=[];As=[];CE=[];SH=[];reg=[];distinct=[]
        for c in manifest['cells']:
            key=(c['id'],c['seed']);a=refs[key];h=policy(c,f,s,f'held{horizon}');r=policy(c,f,s,f'refresh{horizon}');ce=policy(c,f,'current_error',f'held{horizon}');sh=policy(c,f,'support_sham',f'held{horizon}');helds.append(h);refreshes.append(r);As.append(a);CE.append(ce);SH.append(sh)
            rh,rr,ra=remain(h),remain(r),remain(a);ratio=rh/rr if rh is not None and rr is not None else 1.;aratio=rh/ra if rh is not None and ra is not None else 1.
            comp=[e for e in r['events'] if e.get('second_or_later_comparison')];ident=any(e['identifying'] for e in comp);meaningful=any(e['meaningful'] for e in comp)
            firsth=next((e for e in h['events'] if e.get('intervention') and e['offset']==0),None);firstr=next((e for e in r['events'] if e.get('intervention') and e['offset']==0),None)
            common=None
            if firsth and firstr:
                assert firsth['actual_support_sha256']==firstr['actual_support_sha256'] and firsth['actual_output_sha256']==firstr['actual_output_sha256']
                t=str(c['anchor']['after_completed_calls']+1);common=h['window_native_states'][t]==r['window_native_states'][t];assert common,'held/refresh first pulse state differs'
            if c['anchor']:
                t=str(c['anchor']['after_completed_calls']+1);base_state=a['window_native_states'].get(t);hs=h['window_native_states'].get(t)
                if base_state and hs:assert all(base_state[x]==hs[x] for x in ['cpu_rng','cuda_rng']),'aligned first-call draw order differs'
            distinct_choice=all(next((e['actual_support_sha256'] for e in h['events'] if e.get('intervention')),None)!=next((e['actual_support_sha256'] for e in z['events'] if e.get('intervention')),None) for z in [ce,sh]) if firsth else False
            entry=dict(id=c['id'],seed=c['seed'],benchmark=c['benchmark'],selector=s,fraction=f,horizon=horizon,held_remaining=rh,refresh_remaining=rr,A_remaining=ra,ratio=ratio,ratio_A=aratio,identifying=ident,meaningful_current_comparison=meaningful,first_pulse_state_equal=common,held_exposure=h['intervention_calls'],refresh_exposure=r['intervention_calls'],local_gain_vs_refresh=(rr-rh) if rh is not None and rr is not None else None,local_gain_vs_A=(ra-rh) if rh is not None and ra is not None else None,held_correct=h['correct'],refresh_correct=r['correct'],reference_correct=a['correct'],distinct_vs_CE_sham=distinct_choice,initial_missing=c['anchor'] is None)
            pairs.append(entry);byq[c['id']].append(entry);local.append((c['id'],ratio));local_A.append((c['id'],aratio))
            if ident:ident_local.append((c['id'],ratio))
            if a['correct']:correct_local.append((c['id'],ratio))
            controls=[a,dense[key,'dense_full'],dense[key,'piecewise'],ce,sh]
            for n,b in zip(['A','FULL','PW','held_CE','held_sham'],controls):
                if b['correct'] and not h['correct']:reg.append(dict(id=c['id'],seed=c['seed'],versus=n,exposed=h['intervention_calls']>0))
        repeated=[q for q,cs in byq.items() if len(cs)==2 and all(x['identifying'] and x['ratio']<1 for x in cs)]
        adverseA=[q for q,cs in byq.items() if len(cs)==2 and all(x['ratio_A']>1 for x in cs)]
        mean=stats(local,32);reuse_gate=mean['arithmetic_equal_question_mean']<=.9 and len(repeated)>=3 and len({q.split('/')[0] for q in repeated})>=2 and stats(local_A)['arithmetic_equal_question_mean']<=.95 and not adverseA
        qlegacy=[]
        for q,cs in byq.items():
            if len(cs)!=2 or not all(x['distinct_vs_CE_sham'] for x in cs):continue
            cc=[c for c in manifest['cells'] if c['id']==q]
            if all(remain(policy(c,f,s,f'held{horizon}'))<remain(policy(c,f,'current_error',f'held{horizon}')) and remain(policy(c,f,s,f'held{horizon}'))<remain(policy(c,f,'support_sham',f'held{horizon}')) for c in cc):qlegacy.append(q)
        crat={}
        for n,bs in [('CE',CE),('sham',SH)]:crat[n]=stats([(h['id'],remain(h)/remain(b) if remain(h) is not None and remain(b) is not None else 1.) for h,b in zip(helds,bs)])
        selector_gate=len(qlegacy)>=3 and len({q.split('/')[0] for q in qlegacy})>=2 and all(v['arithmetic_equal_question_mean']<=.9 for v in crat.values())
        nrat={n:geom([h['N']/b['N'] for h,b in zip(helds,bs)]) for n,bs in [('A',As),('refresh',refreshes)]};cratio={n:geom([h['C']/b['C'] for h,b in zip(helds,bs)]) for n,bs in [('A',As),('refresh',refreshes)]}
        workpass=all(sum(kept(h) for h in helds)<=sum(kept(b) for b in bs) for bs in [As,refreshes]);convergence=all(v<=1 for v in nrat.values()) and workpass
        fewer=all(v<=.95 for v in nrat.values()) and all(v<=.95 for v in cratio.values()) and workpass and all(sum(h['output_capped'] or h['no_final_response'] for h in helds)<=sum(b['output_capped'] or b['no_final_response'] for b in bs) for bs in [As,refreshes])
        complete=not reg and (convergence or fewer)
        eligible=s in ['current_error','mass'] or s in ['readout_margin','response_qpast','margin_weighted'] and selector_gate
        decisions.append(dict(selector=s,fraction=f,horizon=horizon,engineering=True,reuse_gate=reuse_gate,selector_gate=selector_gate,complete_behavior_gate=complete,advance=eligible and reuse_gate and complete,paired_local_ratio=mean,paired_local_ratio_A=stats(local_A),identifying_ratio=stats(ident_local),correct_reference_ratio=stats(correct_local),paired_repeated_benefit_questions=repeated,repeated_A_damage=adverseA,distinct_selection_questions=qlegacy,held_vs_CE_sham=crat,complete_geometric_N=nrat,complete_geometric_C=cratio,work_gate=workpass,convergence_route=convergence,fewer_canvas_route=fewer,quality_regressions=reg,aliases_or_nonidentifying=sum(not p['identifying'] for p in pairs[-len(manifest['cells']):])))
    for c in manifest['cells']:
        for f,s in itertools.product([.05,.1],SELECTORS):
            a=refs[c['id'],c['seed']];aa=remain(a);sa=remain(policy(c,f,s,'SA'));ass=remain(policy(c,f,s,'AS'));ss=remain(policy(c,f,s,'held2'));h4=remain(policy(c,f,s,'held4'))
            factorial.append(dict(id=c['id'],seed=c['seed'],selector=s,fraction=f,AA=aa,SA=sa,AS=ass,SS=ss,held4=h4,SA_minus_AA=sa-aa if sa is not None and aa is not None else None,AS_minus_AA=ass-aa if ass is not None and aa is not None else None,SS_minus_SA=ss-sa if ss is not None and sa is not None else None,interaction_SS_SA_AS_AA=ss-sa-ass+aa if all(v is not None for v in [aa,sa,ass,ss]) else None,duration4_minus1=h4-sa if h4 is not None and sa is not None else None))
    selected=sorted([d for d in decisions if d['advance']],key=lambda d:(d['paired_local_ratio']['arithmetic_equal_question_mean'],d['complete_geometric_N']['A'],SELECTORS.index(d['selector']),d['fraction'],d['horizon']))
    for n,v in [('tables.json',tables),('quality_transitions.json',quality),('reuse_pairs.json',pairs),('factorial.json',factorial),('physical_work.json',physical),('shadow_certificate.json',shadow),('method_decisions.json',decisions),('input_hashes.json',INPUTS)]:save(out/n,v)
    summary=dict(completed=True,unique_branches=len(records),scientific_labels=manifest['scientific_labels'],questions=6,seeds=[42,43],any_configuration_advances=bool(selected),selected_finalist=selected[0] if selected else None,clean_policy_latency_qualified=False,baseline_tables=tables[:12],shadow_observations=len(shadow),shadow_failures=sum(not s['safe'] for s in shadow),held_guard_violations=sum(not s['guard_now'] for s in shadow),all_mixed_and_negative_configurations_retained=True)
    save(out/'summary.json',summary);save(out/'final_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),branch_manifest_sha256=sha(ROOT/'branch_manifest.json'),summary_sha256=sha(out/'summary.json'),scope='complete frozen development diagnostic; conditional confirmation only if gates pass'))
    print(json.dumps(dict(unique_branches=len(records),any_configuration_advances=bool(selected),shadow_failures=summary['shadow_failures'])))
if __name__=='__main__':main()
