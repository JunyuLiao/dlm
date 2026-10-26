"""Complete development tables/plots and reconciled canonical resource accounting."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,csv,os
import numpy as np
S=Path(__file__).resolve().parent;R=S/'attempt002';OUT=S/'report001';INPUTS={}
def read(p):INPUTS[str(p.relative_to(S))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def save(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def resources():
    workers=[];requests=[]
    for attempt,jobs in [('attempt001',['dense_full','piecewise','references']),('attempt002',['treatments']),('attempt004',['qualification'])]:
        for job in jobs:
            root=S/attempt;end=read(root/f'launches/{job}/exit.json');assert end.get('returncode')==0
            total=read(root/f'runs/{job}/final_complete.json');assert total['passed']
            rr=[]
            for p in (root/f'runs/{job}').rglob('*.json'):
                v=read(p)
                if 'model_forwards' in v and 'id' in v:rr.append(v)
            assert sum(r['model_forwards'] for r in rr)+total['initialization_engine_forwards']==total['engine_execute_forwards']
            count=total['engine_execute_forwards']+total['startup_model_calls_outside_execute']+total['extra_model_calls_inside_execute']
            workers.append(dict(attempt=attempt,job=job,requests=len(rr),allocated_GPU_worker_seconds=end.get('allocated_GPU_worker_seconds',end.get('allocated_seconds')),explicit_model_forwards=count,engine_execute_forwards=total['engine_execute_forwards'],startup_model_calls_outside_execute=total['startup_model_calls_outside_execute'],initialization_execute_forwards=total['initialization_engine_forwards'],extra_inside_execute=total['extra_model_calls_inside_execute']))
            requests.extend(rr)
    return dict(workers=workers,complete_native_requests=len(requests),explicit_model_forwards=sum(w['explicit_model_forwards'] for w in workers),allocated_GPU_worker_seconds=sum(w['allocated_GPU_worker_seconds'] for w in workers),allocated_GPU_worker_hours=sum(w['allocated_GPU_worker_seconds'] for w in workers)/3600,one_worker_at_a_time=True,source_compatible_copied_controls_counted_once=True,includes_startup_warmup_qualification_scoring_wait_and_cleanup=True,parent_poll_resolution_seconds=10,CPU_reader_failures=2,failed_GPU_workers=0,no_campaign_resource_ceiling=True,clean_online_latency_qualified=False)
def main():
    OUT.mkdir(exist_ok=False);assert read(R/'analysis003/final_complete.json')['passed'];dec=read(R/'analysis003/method_decisions.json');tables=read(R/'analysis003/tables.json');pairs=read(R/'analysis003/reuse_pairs.json');quality=read(R/'analysis003/quality_transitions.json');shadow=read(R/'analysis003/shadow_certificate.json');plan=read(R/'branch_manifest.json');rsrc=resources();save('resources.json',rsrc)
    fields=sorted(set(k for r in tables for k in r))
    with (OUT/'all_policy_tables.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(tables)
    go=[d for d in dec if d['advance']];counts=Counter('saving' if p['local_gain_vs_refresh'] and p['local_gain_vs_refresh']>0 else 'regression' if p['local_gain_vs_refresh'] and p['local_gain_vs_refresh']<0 else 'tie' for p in pairs if p['held_remaining'] is not None)
    summary=dict(completed=True,unique_native_development_branches=plan['unique_branches'],scientific_labels=plan['scientific_labels'],independent_development_questions=6,paired_seeds=[42,43],policy_cells_per_configuration=12,exposed_initial_seed_cells=sum(bool(c['branches']) for c in plan['cells']),initial_fallback_seed_cells=sum(not bool(c['branches']) for c in plan['cells']),primary_hold_refresh_configurations=len(dec),configurations_passing_reuse_gate=sum(d['reuse_gate'] for d in dec),configurations_passing_selector_gate=sum(d['selector_gate'] for d in dec),configurations_passing_complete_behavior=sum(d['complete_behavior_gate'] for d in dec),configurations_advancing=len(go),descriptive_dependent_label_local_outcomes=dict(counts),quality_regressions_all_config_comparators=sum(len(q['regressions']) for q in quality),no_noninferiority_claim=True,clean_online_latency_qualified=False,resources=rsrc)
    save('summary.json',summary)
    os.environ['MPLCONFIGDIR']=str(S/'private/mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['current_error','mass','support_sham','readout_margin','response_qpast','margin_weighted','signed_high','signed_low'];labels=['Current error','Mass','Support sham','Previous-pair readout','Q-past Response','Margin-weighted','Signed high','Signed low']
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True)
    for ax,(fraction,horizon) in zip(axes.flat,[(.05,2),(.1,2),(.05,4),(.1,4)]):
        dd=[next(d for d in dec if d['fraction']==fraction and d['horizon']==horizon and d['selector']==n) for n in names];y=np.arange(8);x=np.array([d['paired_local_ratio']['arithmetic_equal_question_mean'] for d in dd]);lo=np.array([d['paired_local_ratio']['CI'][0] for d in dd]);hi=np.array([d['paired_local_ratio']['CI'][1] for d in dd]);ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt='o',capsize=3);ax.axvline(1,color='black',ls='--');ax.axvline(.9,color='green',ls=':');ax.set_yticks(y,labels);ax.set_title(f'{fraction:.0%} nominal deletion, {horizon}-call window');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.supxlabel('Arithmetic equal-question remaining-call ratio: held / real refresh (lower is better)');fig.suptitle('Six development question clusters; family-adjusted descriptive intervals');fig.tight_layout();fig.savefig(OUT/'held_vs_refresh_local.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for n,label in zip(names,labels):
        dd=[d for d in dec if d['selector']==n];ax.scatter([d['paired_local_ratio']['arithmetic_equal_question_mean'] for d in dd],[d['complete_geometric_N']['refresh'] for d in dd],label=label,s=50)
    ax.axvline(1,color='black',ls='--');ax.axhline(1,color='black',ls='--');ax.set_xlabel('Local remaining calls: arithmetic question ratio held / refresh');ax.set_ylabel('Complete N: geometric paired ratio held / refresh');ax.set_title('Local effects and complete-request effects are separate');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'local_vs_complete_calls.png',dpi=180);plt.close(fig)
    branches={b['identity']:b for c in plan['cells'] for b in c['branches']};certificate=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,fraction in zip(axes,[.05,.1]):
        byage=defaultdict(list)
        for row in shadow:
            b=branches[row['identity']]
            if any(l['selector']=='current_error' and l['lifecycle']=='held4' and l['fraction']==fraction for l in b['labels']):byage[row['age']].append(row)
        ages=[];lower=[];actual=[]
        for age,rows in sorted(byage.items()):
            byq=defaultdict(list)
            for row in rows:byq[row['id']].append(row)
            l=float(np.mean([np.mean([r['lower_bound_min'] for r in v]) for v in byq.values()]));a=float(np.mean([np.mean([r['actual_retained_mass_min'] for r in v]) for v in byq.values()]));ages.append(age);lower.append(l);actual.append(a);certificate.append(dict(fraction=fraction,age=age,questions=len(byq),observations=len(rows),mean_question_lower_bound=l,mean_question_actual_retained_mass=a))
        ax.plot(ages,lower,'o-',label='Shadow lower bound');ax.plot(ages,actual,'s-',label='Actual retained mass');ax.axhline(.8,color='black',ls='--');ax.set_title(f'CE held4, {fraction:.0%} nominal deletion');ax.set_xlabel('Hold age');ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
    axes[0].set_ylabel('Equal-question mean minimum retained mass');axes[1].legend(fontsize=8);fig.suptitle('Certificate is a shadow measurement; it never refreshes held maps');fig.tight_layout();fig.savefig(OUT/'certificate_shadow.png',dpi=180);plt.close(fig);save('certificate_question_summaries.json',certificate)
    lines=['# Controlled reuse development report','',f"All {plan['unique_branches']} frozen distinct requests completed native decoding and official scoring. There are six question clusters and two paired seeds; three initial seed cells fall back to A.",'',f"Reuse gate passes: {summary['configurations_passing_reuse_gate']}/{len(dec)}. Finalists: {len(go)}. Clean online latency remains unqualified.",'','Ratios below are arithmetic equal-question local held/refresh and geometric paired complete N held/refresh. Pooled totals are separately labeled in CSV/JSON.','', '| Selector | Budget | Window | Local held/refresh | Complete N held/refresh | Repeated-benefit questions | Quality regressions | Advance |','|---|---:|---:|---:|---:|---:|---:|---|']
    for d in dec:lines.append(f"| {d['selector']} | {d['fraction']:.0%} | {d['horizon']} | {d['paired_local_ratio']['arithmetic_equal_question_mean']:.4f} | {d['complete_geometric_N']['refresh']:.4f} | {len(d['paired_repeated_benefit_questions'])} | {len(d['quality_regressions'])} | {d['advance']} |")
    lines+=['','All configurations, losses, aliases, caps, missing final responses and infeasible fallbacks are retained. Absence of observed quality loss does not establish noninferiority. Diagnostic implementations and precomputed initial supports cannot establish clean online latency.','',f"Resources: {rsrc['complete_native_requests']} complete requests, {rsrc['explicit_model_forwards']} explicit model forwards, {rsrc['allocated_GPU_worker_hours']:.6f} allocated H100-worker hours including startup, warmups, qualification and cleanup. Copied controls count once."]
    with (OUT/'summary.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    save('input_hashes.json',INPUTS);save('final_complete.json',dict(passed=True,analysis_identity='attempt002/analysis003',resources_sha256=hashlib.sha256((OUT/'resources.json').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),conditional_confirmation_launched=False))
    print(json.dumps(summary))
if __name__=='__main__':main()
