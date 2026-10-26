"""Derived, input-hashed handoff tables/figures; no source or outcome mutation."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;S=R.parent;E=S/'experiment/attempt006';P=S/'profiling/attempt002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def cluster(values):
    x=np.asarray([np.mean(v) for v in values.values()],dtype=float)
    if not len(x):return None
    rng=np.random.default_rng(20261026);bs=x[rng.integers(0,len(x),(20000,len(x)))].mean(1)
    means={q:float(np.mean(v)) for q,v in values.items()}
    return dict(mean=float(x.mean()),CI95=np.quantile(bs,[.025,.975]).tolist(),question_clusters=len(x),seeds_are_not_independent_questions=True,leave_one_question_out=[dict(excluded=q,mean=float(np.mean([v for other,v in means.items() if other!=q]))) for q in means] if len(means)>1 else [])
def summarize(records):
    return dict(requests=len(records),questions=len({v['id'] for v in records}),correct=sum(v['correct'] for v in records),N=sum(v['N'] for v in records),C=sum(v['C'] for v in records),T=sum(v['T'] for v in records),pooled_N_per_C=sum(v['N'] for v in records)/sum(v['C'] for v in records),output_caps=sum(v['output_capped'] for v in records),canvas_cap_events=sum(not x['commit'] and x['cap_reached'] for v in records for x in v['native_trace']),missing_final=sum(v['no_final_response'] for v in records),extraction_failures=sum(not v['answer_parsed'] for v in records))
def main():
    assert load(E/'analysis001/final_complete.json')['passed'];inputs={str(p.relative_to(S)):sha(p) for p in [E/'analysis001/final_complete.json',E/'analysis001/method_decisions.json',E/'analysis001/branches.json',E/'branch_manifest.json',P/'manifest_identity.json',E/'numerical_qualification.json']};decisions=load(E/'analysis001/method_decisions.json');branches=load(E/'analysis001/branches.json');plan=load(E/'branch_manifest.json');rows=load(P/'manifest_identity.json')['identities'];development={r['id'] for r in rows if r['partition']=='development'};profiles={};refs={};treatments={};allwork=[]
    for name,h in plan['input_hashes'].items():assert sha(E/name)==h;inputs['experiment/'+name]=h
    for name,h in load(E/'numerical_qualification.json')['input_hashes'].items():assert sha(E/name)==h;inputs['experiment/'+name]=h
    for p in (P/'runs/profile').glob('*/audit*.json'):
        v=load(p)
        if v['id'] in development:profiles[(v['id'],v['seed'],v['arm'])]=v;inputs['profiling/'+str(p.relative_to(P))]=sha(p)
    for p in (E/'runs/references').glob('*seed*.json'):
        v=load(p);refs[(v['id'],v['seed'])]=v;inputs['experiment/'+str(p.relative_to(E))]=sha(p)
    for p in (E/'runs/treatments').glob('*/s*.json'):
        v=load(p);treatments[v['branch_identity']]=v;inputs['experiment/'+str(p.relative_to(E))]=sha(p)
        assert v['startup_model_calls_in_request']==v['extra_inside_execute_Python_forwards']==0
        w=v['work'];skipped=sum(x['skipped'] for x in v['consumer_physical']);assert skipped==w['global_skipped_tiles']==w['overall_skipped_rectangles'];assert w['global_kept_tiles']==w['global_eligible_tiles']-skipped;assert math.isclose(w['overall_rectangle_sparsity'],skipped/w['overall_eligible_rectangles']);assert sum(x['skipped']>0 for x in v['consumer_physical'])==1;assert all(x['LOCAL_skipped']==x['protected_skipped']==0 for x in v['consumer_physical'])
        allwork.append(dict(identity=v['branch_identity'],GLOBAL_eligible=w['global_eligible_tiles'],GLOBAL_kept=w['global_kept_tiles'],GLOBAL_skipped=skipped,LOCAL_eligible=w['native_local_eligible_rectangles'],LOCAL_kept=w['native_local_eligible_rectangles'],LOCAL_skipped=0,overall_denominator=w['overall_eligible_rectangles'],overall_sparsity=skipped/w['overall_eligible_rectangles'],diagnostic_work=v['operator_diagnostic_work'],native_uncommitted_tail=v['uncommitted_tail'],model_forwards=v['model_forwards'],audit_W=v['W'],audit_S=v['S'],audit_S_per_N=v['S_per_N'],clean_online_timing=False))
    sys.path.insert(0,str(E/'code'));from pulse_runtime import core_trace_equal
    prefix_checks=[];native_stops=[]
    for cell in plan['cells']:
        ref=refs[(cell['id'],cell['seed'])]
        for b in cell['branches']:
            v=treatments[b['identity']];before=lambda r:r['canvas']<b['canvas'] or (r['canvas']==b['canvas'] and not r['commit'] and r['native_step']<b['step'])
            assert core_trace_equal([r for r in v['native_trace'] if before(r)],[r for r in ref['native_trace'] if before(r)])
            assert v['initial_cpu_rng']==ref['initial_cpu_rng'] and v['initial_cuda_rng']==ref['initial_cuda_rng']
            assert v['local_endpoints'][b['stratum']]['common_state_certified']
            prefix_checks.append(dict(identity=b['identity'],native_prefix_trace_equal=True,request_entry_RNG_equal=True,pre_pulse_active_state_and_site_QKV_qualified=True,arbitrary_restoration_qualified=False))
            def event(record,step):return next((x for x in record['events'] if x.get('stratum')==b['stratum'] and x.get('step')==step),None)
            actual=event(v,b['step']);reference=event(ref,b['step']);assert actual is not None and reference is not None
            rng={k:actual['native_state'][k]==reference['native_state'][k] for k in ['cpu_rng','cuda_rng']};assert all(rng.values()),'Immediate aligned-call RNG differs'
            motion={k:actual['native_state'][k]['byte_sha256']!=reference['native_state'][k]['byte_sha256'] for k in ['stochastic_canvas','native_winner','self_condition','history','confidence','converged']}
            def stop(record):
                row=record['local_endpoints'][b['stratum']]['terminal'];adaptive=row['history_ready'] and row['stable'] and row['confidence'];return dict(reason='both' if adaptive and row['cap_reached'] else 'adaptive' if adaptive else 'cap',native_step=row['native_step'],confidence=row['confidence'],stable=row['stable'],history_ready=row['history_ready'],entropy_slack=row['entropy_slack'])
            native_stops.append(dict(identity=b['identity'],reference=stop(ref),treatment=stop(v),immediate_aligned_call_RNG_checks=rng,immediate_state_changed=motion,next_same_canvas_available=v['local_endpoints'][b['stratum']]['next_same_canvas_available'],acceptance_count_qualification=False))
    tables=[]
    for task in ['aime26','humaneval','longbench_v2']:
        for arm in ['corrected_dense_full','dense_piecewise','matched_allkept_l5_l17_l29']:
            rs=[v for key,v in profiles.items() if v['benchmark']==task and key[2]==arm];tables.append(dict(benchmark=task,arm=arm,**summarize(rs)))
    paired=[]
    for d in decisions:
        for task in ['all','aime26','humaneval','longbench_v2']:
            for family in ['current_error','support_sham','readout_margin']:
                selected=[]
                for c in d['cells']:
                    if task!='all' and c['benchmark']!=task:continue
                    identity=c['choices'].get(family);v=treatments[identity] if identity else refs[(c['id'],c['seed'])];selected.append(v)
                tables.append(dict(benchmark=task,arm=family,anchor=d['anchor'],fraction=d['fraction'],**summarize(selected)))
            for baseline in ['corrected_dense_full','dense_piecewise','matched_allkept_l5_l17_l29','current_error','support_sham']:
                byquality=defaultdict(list);byn=defaultdict(list);bync=defaultdict(list);byt=defaultdict(list);trans=Counter();regressions=[]
                for c in d['cells']:
                    if task!='all' and c['benchmark']!=task:continue
                    rid=c['choices'].get('readout_margin');v=treatments[rid] if rid else refs[(c['id'],c['seed'])]
                    if baseline in ['current_error','support_sham']:
                        bid=c['choices'].get(baseline);b=treatments[bid] if bid else refs[(c['id'],c['seed'])]
                    else:b=profiles[(c['id'],c['seed'],baseline)]
                    byquality[c['id']].append(int(v['correct'])-int(b['correct']));byn[c['id']].append(v['N']/b['N']);bync[c['id']].append((v['N']/v['C'])/(b['N']/b['C']));byt[c['id']].append(v['T']/b['T']);trans[f'{int(b["correct"])}->{int(v["correct"])}']+=1
                    if b['correct'] and not v['correct']:regressions.append(dict(id=c['id'],seed=c['seed']))
                paired.append(dict(anchor=d['anchor'],fraction=d['fraction'],benchmark=task,proposed='readout_margin',baseline=baseline,quality_difference=cluster(byquality),N_ratio=cluster(byn),paired_N_per_C_ratio=cluster(bync),T_ratio=cluster(byt),quality_transitions=dict(trans),regressions=regressions))
    matched_contrasts=[];outcome_lookup={r['identity']:r for r in branches}
    for cell in plan['cells']:
        bybank=defaultdict(list)
        for b in cell['branches']:
            if b['layer']==29:bybank[(b['bank_relative'],b['drop_count'])].append(b)
        for (name,drop),bs in bybank.items():
            site=load(E/name);bank=next(z for z in site['banks'] if z.get('drop_count')==drop and 'alias_of' not in z)
            for i,a in enumerate(bs):
                for b in bs[i+1:]:
                    ca=bank['candidates'][a['candidate_index']];cb=bank['candidates'][b['candidate_index']]
                    if abs(ca['current_error']-cb['current_error'])/max(ca['current_error'],cb['current_error'],1e-6)>.05 or abs(ca['deleted_mass']-cb['deleted_mass'])>.01:continue
                    ra=outcome_lookup[a['identity']];rb=outcome_lookup[b['identity']]
                    matched_contrasts.append(dict(id=cell['id'],seed=cell['seed'],benchmark=cell['benchmark'],anchor=a['stratum'],drop_count=drop,supports=[a['support_sha256'],b['support_sha256']],families=[a['families'],b['families']],local_gain_difference=ra['local_gain']-rb['local_gain'],projected_gain_difference=ca['projected_entropy_weighted_gain']-cb['projected_entropy_weighted_gain'],complete_quality_difference=int(ra['correct'])-int(rb['correct']),equal_physical_work=True,error_mass_matched=True,candidate_pairs_not_independent_questions=True))
    distributions=[]
    for key,rs in group(branches).items():distributions.append(dict(benchmark=key[0],layer=key[1],anchor=key[2],fraction=key[3],unique_branches=len(rs),questions=len({r['id'] for r in rs}),local_outcomes=dict(Counter(r['local_outcome'] for r in rs)),local_gain_distribution=dict(Counter(str(r['local_gain']) for r in rs)),correct_to_wrong=sum(r['reference_correct'] and not r['correct'] for r in rs),wrong_to_correct=sum(not r['reference_correct'] and r['correct'] for r in rs),completed_canvas_equal=sum(r['completed_canvas_equal'] for r in rs)))
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for ax,task in zip(axes,['aime26','humaneval','longbench_v2']):
        rs=[r for r in branches if r['benchmark']==task and r['layer']==29];counts=Counter(r['local_outcome'] for r in rs);ax.bar(['Fewer','Same','More'],[counts[k] for k in ['favorable','neutral','adverse']],color=['#2a788e','#999999','#ce5b5b']);ax.set_title(task);ax.set_ylabel('Unique candidate branches')
    fig.suptitle('Single-pulse discovery: local calls, not independent task samples');fig.tight_layout();fig.savefig(R/'local_call_distributions.png',dpi=180);fig.savefig(R/'local_call_distributions.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,task in zip(axes,['aime26','humaneval','longbench_v2']):
        cell=next(c for c in plan['cells'] if c['benchmark']==task and c['seed']==42 and c['branches']);b=cell['branches'][0];a=refs[(cell['id'],cell['seed'])];s=treatments[b['identity']];after=a['anchors'][b['stratum']]['after_completed_calls']
        for name,v,color in [('A',a,'#555555'),('S',s,'#2a788e')]:
            trace=[r for r in v['native_trace'] if not r['commit'] and r['canvas']==b['canvas'] and r['native_step']>=after];x=[r['native_step']-after for r in trace];y=[max(r['entropy_mean'],1e-12) for r in trace];ax.semilogy(x,y,'.-',color=color,label=f'{name}: R={v["local_endpoints"][b["stratum"]]["remaining_calls"]}, correct={v["correct"]}')
            stable=[(xx,yy) for xx,yy,r in zip(x,y,trace) if r['history_ready'] and r['stable']]
            if stable:ax.scatter(*zip(*stable),facecolors='none',edgecolors=color,s=35)
        ax.axhline(.005,linestyle='--',color='black',linewidth=.8);ax.set_title(task);ax.set_xlabel('Calls after common anchor');ax.set_ylabel('Whole-canvas entropy');ax.legend(fontsize=7)
    fig.suptitle('Frozen perturbative sentinels; open circles indicate native stability');fig.tight_layout();fig.savefig(R/'sentinel_native_trajectories.png',dpi=180);fig.savefig(R/'sentinel_native_trajectories.svg');plt.close(fig)
    allocations=[];forwards=[]
    for base in [P,E]:
        for p in (base/'launches').glob('*/exit_receipt.json'):v=load(p);inputs[str(p.relative_to(S))]=sha(p);allocations.append(dict(source=str(p.relative_to(S)),**v))
        for p in (base/'runs').glob('*/final_complete.json'):
            v=load(p);forwards.append(dict(source=str(p.relative_to(S)),**v))
        for p in (base/'runs').glob('*/*/final_complete.json'):
            v=load(p);forwards.append(dict(source=str(p.relative_to(S)),**v))
    for base in [S/'experiment/attempt003',S/'experiment/attempt004',S/'experiment/attempt005']:
        for p in (base/'launches').glob('*/exit_receipt.json'):
            v=load(p);inputs[str(p.relative_to(S))]=sha(p);allocations.append(dict(source=str(p.relative_to(S)),**v))
    failure=S/'experiment/attempt003/runs/references/failure.json';v=load(failure);inputs[str(failure.relative_to(S))]=sha(failure);forwards.append(dict(source=str(failure.relative_to(S)),failed_attempt=True,**v))
    failure=S/'experiment/attempt005/runs/treatments/failure.json';v=load(failure);inputs[str(failure.relative_to(S))]=sha(failure);forwards.append(dict(source=str(failure.relative_to(S)),failed_attempt=True,**v))
    save(R/'engineering_prefix_checks.json',prefix_checks)
    save(R/'native_stop_and_state.json',native_stops)
    save(R/'tables.json',tables);save(R/'paired_quality_trajectory.json',paired);save(R/'matched_candidate_contrasts.json',matched_contrasts);save(R/'candidate_distributions.json',distributions);save(R/'physical_work.json',allwork);save(R/'resources.json',dict(allocations=allocations,allocated_GPU_worker_seconds=sum(v['allocated_seconds'] for v in allocations if v['GPU']),explicit_forward_receipts=forwards,total_explicit_model_forwards=sum(v['engine_execute_forwards']+v['startup_model_calls_outside_execute']+v['extra_model_calls_inside_execute'] for v in forwards),no_campaign_ceiling=True,clean_timing_qualified=False));save(R/'input_hashes.json',inputs);save(R/'final_complete.json',dict(passed=True,report_source_sha256=sha(Path(__file__)),experiment_protocol_sha256=sha(E/'protocol.json'),analysis_summary_sha256=sha(E/'analysis001/summary.json'),paired_summary_sha256=sha(R/'paired_quality_trajectory.json'),no_speedup_claim=True))
def group(records):
    result=defaultdict(list)
    for r in records:
        for fraction in r['fractions']:result[(r['benchmark'],r['layer'],r['anchor'],fraction)].append(r)
    return result
if __name__=='__main__':main()
