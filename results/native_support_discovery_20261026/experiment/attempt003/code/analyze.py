"""Question-clustered discovery outcomes; oracle opportunity is not selector success."""
from pathlib import Path
from collections import Counter,defaultdict
import itertools,json,math
import numpy as np
from pulse_runtime import ROOT,PROFILE,save,sha,core_trace_equal
from op_bank import matched
def load(p):return json.loads(p.read_text())
def geometric(values):return math.exp(sum(math.log(v) for v in values)/len(values)) if values else None
def bootstrap(values):
    if not values:return None
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(20261026);v=x[rng.integers(0,len(x),(20000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),question_clusters=len(x),interval95=np.quantile(v,[.025,.975]).tolist(),descriptive_small_sample=True)
def main():
    assert load(ROOT/'runs/treatments/final_complete.json')['passed'];plan=load(ROOT/'branch_manifest.json');out=ROOT/'analysis001';out.mkdir(exist_ok=False);inputs={'branch_manifest.json':sha(ROOT/'branch_manifest.json')};branchrows=[];groups=[];configuration=defaultdict(list);qout=defaultdict(list);missing=[]
    for name,h in plan['input_hashes'].items():assert sha(ROOT/name)==h;inputs[name]=h
    assert load(ROOT/'numerical_qualification.json')['passed'];inputs['numerical_qualification.json']=sha(ROOT/'numerical_qualification.json')
    quals=list((ROOT/'runs/treatments').glob('*/qualification.json'))
    for p in quals:assert load(p)['passed'];inputs[str(p.relative_to(ROOT))]=sha(p)
    treated_tasks={c['benchmark'] for c in plan['cells'] if c['branches']}
    assert len(quals)==len(treated_tasks),'Every treated task requires SAME-policy qualification'
    profiles={}
    for arm in ['corrected_dense_full','dense_piecewise','matched_allkept_l5_l17_l29']:
        for p in (PROFILE/'runs/profile'/arm).glob('audit*.json'):
            r=load(p);profiles[(r['id'],r['seed'],arm)]=(p,r)
    for cell in plan['cells']:
        cid=__import__('study_worker').cell_id(cell,cell['seed']);rp=ROOT/'runs/references'/(cid+'.json');ref=load(rp);inputs[str(rp.relative_to(ROOT))]=sha(rp);records={}
        for p in (ROOT/'runs/treatments'/cid).glob('s*.json'):
            v=load(p);records[v['branch_identity']]=v;inputs[str(p.relative_to(ROOT))]=sha(p)
        for b in cell['branches']:
            assert b['identity'] in records;v=records[b['identity']];a=ref['local_endpoints'][b['stratum']];s=v['local_endpoints'][b['stratum']];gain=a['remaining_calls']-s['remaining_calls'];controls={}
            for arm in ['corrected_dense_full','dense_piecewise','matched_allkept_l5_l17_l29']:
                p,c=profiles[(cell['id'],cell['seed'],arm)];inputs['profiling/'+str(p.relative_to(PROFILE))]=sha(p)
                controls[arm]=dict(correct_to_wrong=c['correct'] and not v['correct'],wrong_to_correct=not c['correct'] and v['correct'],N_ratio=v['N']/c['N'],C_ratio=v['C']/c['C'],T_ratio=v['T']/c['T'])
            KV=[]
            for key,value in v['committed_KV_where_measured'].items():
                base=ref['committed_KV_where_measured'].get(key)
                # Early reference captures early layers separately.
                if base is None:
                    ep=ROOT/'runs/early'/(cid+'.json')
                    if ep.exists():base=load(ep)['committed_KV_where_measured'].get(key)
                KV.append(dict(scope=key,reference_available=base is not None,byte_equal=base==value if base else None,numeric_equal=all(base[k]['numeric_sha256']==value[k]['numeric_sha256'] for k in ['K','V']) if base else None))
            rec=dict(id=cell['id'],seed=cell['seed'],benchmark=cell['benchmark'],question_stratum=cell['stratum'],identity=b['identity'],layer=b['layer'],anchor=b['stratum'],fractions=b['fractions'],drop_count=b['drop_count'],families=b['families'],support_sha256=b['support_sha256'],current_error=b['current_error'],deleted_mass=b['deleted_mass'],reference_remaining=a['remaining_calls'],remaining=s['remaining_calls'],local_gain=gain,local_outcome='favorable' if gain>0 else 'adverse' if gain<0 else 'neutral',terminal_cap=s['terminal']['cap_reached'],reference_terminal_cap=a['terminal']['cap_reached'],completed_canvas_equal=s['commit']==a['commit'],next_same_canvas_available=s['next_same_canvas_available'],committed_KV_where_measured=KV,correct=v['correct'],reference_correct=ref['correct'],controls=controls,N=v['N'],C=v['C'],T=v['T'],N_per_C=v['N_per_C'],reference_N=ref['N'],reference_C=ref['C'],reference_T=ref['T'],output_capped=v['output_capped'],missing_final=v['no_final_response'],extraction_failed=not v['answer_parsed'],full_output_equal=v['output_sha256']==ref['output_sha256'],final_RNG_equal=v['final_cuda_rng']==ref['final_cuda_rng'],work=v['work'],reference_work=ref['work'],operator_diagnostic_work=v['operator_diagnostic_work'],source_record_sha256=sha(ROOT/f'runs/treatments/{cid}/s{cell["branches"].index(b):03d}.json'))
            branchrows.append(rec);qout[(cell['id'],b['layer'],b['stratum'],b['drop_count'])].append(rec)
            source_site=load(ROOT/b['bank_relative'])
            if b['layer']==29:
                source_bank=next(z for z in source_site['banks'] if z.get('drop_count')==b['drop_count'] and 'alias_of' not in z);candidate=source_bank['candidates'][b['candidate_index']]
            else:candidate=source_site['banks'][b['candidate_index']]
            rec['prechoice_operator_features']={k:candidate.get(k) for k in ['bias_Frobenius','bias_relative_Frobenius','bias_entropy_concentration','projected_entropy_weighted_gain','projected_stable_crossings','prefix_only_deleted_mass_max','actual_vs_FP32_cosine','roundoff_or_tiny_bias']}
        for name in ['confidence','instability']:
            for fraction in [.05,.1]:
                bs=[b for b in cell['branches'] if b['layer']==29 and b['stratum']==name and fraction in b['fractions']];choices={}
                for family in ['current_error','support_sham','readout_margin']:
                    hits=[b for b in bs if family in b['families']];assert len(hits)<=1
                    if hits:choices[family]=next(r for r in branchrows if r['identity']==hits[0]['identity'])
                if len(choices)<3:
                    missing.append(dict(id=cell['id'],seed=cell['seed'],anchor=name,fraction=fraction,reason='missing_anchor_or_feasible_bank',policy_neutral_ratio=1))
                    configuration[(name,fraction)].append(dict(id=cell['id'],seed=cell['seed'],benchmark=cell['benchmark'],available=False,ratios={'current_error':1.,'support_sham':1.},choices={}));continue
                r=choices['readout_margin'];ce=choices['current_error'];sham=choices['support_sham'];bankpath=ROOT/next(b['bank_relative'] for b in bs if 'readout_margin' in b['families']);site=load(bankpath);bank=next(k for k in site['banks'] if k.get('drop_count')==r['drop_count'] and 'alias_of' not in k);private_path=ROOT/bank['private_relative'];assert sha(private_path)==bank['private_sha256'];inputs[bank['private_relative']]=bank['private_sha256'];priv=__import__('torch').load(private_path,map_location='cpu',weights_only=False)
                def meaningful(other):
                    i=bank['selections']['readout_margin'];j=bank['selections'][other];a=bank['candidates'][i];b=bank['candidates'][j];x=priv['candidates'][i]['actual_bias'];y=priv['candidates'][j]['actual_bias'];sep=float((x-y).norm())/max(float(x.norm()),float(y.norm()),1e-12)
                    return matched(a,b) and a['bias_direction_available'] and b['bias_direction_available'] and sep>=.25
                available=dict(id=cell['id'],seed=cell['seed'],benchmark=cell['benchmark'],available=True,identifying=meaningful('current_error') and meaningful('support_sham'),ratios={family:r['remaining']/choices[family]['remaining'] for family in ['current_error','support_sham']},choices={f:c['identity'] for f,c in choices.items()},readout=r,ce=ce,sham=sham)
                configuration[(name,fraction)].append(available)
    decisions=[]
    for (anchor,fraction),cells in configuration.items():
        byq=defaultdict(list)
        for c in cells:byq[c['id']].append(c)
        ratios={f:[sum(c['ratios'][f] for c in cs)/2 for cs in byq.values()] for f in ['current_error','support_sham']};identq=[q for q,cs in byq.items() if len(cs)==2 and all(c.get('identifying',False) for c in cs)]
        leadq=[q for q,cs in byq.items() if len(cs)==2 and all(c['available'] and c['ratios']['current_error']<1 and c['ratios']['support_sham']<1 for c in cs)]
        opp=[];safe=[]
        for q,cs in byq.items():
            ok=[];good=[]
            for c in cs:
                candidates=[r for r in branchrows if r['id']==q and r['seed']==c['seed'] and r['layer']==29 and r['anchor']==anchor and fraction in r['fractions']]
                ok.append(any(r['local_gain']>=1 for r in candidates));good.append(any(r['local_gain']>=1 and r['correct'] and r['reference_correct'] for r in candidates))
            if len(cs)==2 and all(ok):opp.append(q)
            if len(cs)==2 and all(good):safe.append(q)
        tasks={c['benchmark'] for c in cells if c['id'] in opp};itasks={c['benchmark'] for c in cells if c['id'] in identq};exposed=[c for c in cells if c['available']];quality_reg=[]
        for c in exposed:
            r=c['readout'];quality_reg.extend(dict(id=c['id'],seed=c['seed'],versus=arm) for arm,v in r['controls'].items() if v['correct_to_wrong'])
            for family,key in [('current_error','ce'),('support_sham','sham')]:
                if c[key]['correct'] and not r['correct']:quality_reg.append(dict(id=c['id'],seed=c['seed'],versus=family))
        compute={f:{metric:geometric([(c['readout'][metric]/c[key][metric]) if c['available'] else 1 for c in cells]) for metric in ['N','C']} for f,key in [('current_error','ce'),('support_sham','sham')]}
        def kept(r):return r['work']['overall_eligible_rectangles']-r['work']['overall_skipped_rectangles']
        workgate=all(sum(kept(c['readout'])-kept(c[key]) for c in exposed)<=0 for key in ['ce','sham'])
        gates=dict(engineering=True,opportunity=len(opp)>=3 and len(tasks)>=2 and len(safe)>=2,prediction=len(identq)>=4 and len(itasks)>=2 and all(np.mean(x)<=.90 for x in ratios.values()) and len(leadq)>=2,complete_quality_computation=not quality_reg and all(any(v<1 for v in x.values()) for x in compute.values()) and workgate and any(c['readout']['correct'] for c in exposed),clean_latency=False)
        decisions.append(dict(anchor=anchor,fraction=fraction,gates=gates,advance=all(gates[k] for k in ['engineering','opportunity','prediction','complete_quality_computation']),identifying_questions=identq,paired_leading_questions=leadq,oracle_opportunity_questions=opp,correct_reference_safe_oracle_questions=safe,prediction_ratios={f:bootstrap(x) for f,x in ratios.items()},quality_regressions=quality_reg,complete_request_geometric_ratios=compute,work_gate=workgate,cells=[{k:v for k,v in c.items() if k not in ['readout','ce','sham']} for c in cells]))
    assert len(branchrows)==plan['unique_branches']
    properties=property_analysis(branchrows)
    summary=dict(completed=True,unique_branches=len(branchrows),independent_development_questions=len({c['id'] for c in plan['cells']}),outcomes=dict(Counter(r['local_outcome'] for r in branchrows)),by_task_layer_anchor_budget=[dict(benchmark=key[0],layer=key[1],anchor=key[2],drop_count=key[3],branches=len(rs),outcomes=dict(Counter(r['local_outcome'] for r in rs))) for key,rs in group_records(branchrows).items()],missing=missing,clean_timing_qualified=False,any_rule_advances=any(d['advance'] for d in decisions),scope='development single pulses; no reuse or clean speedup claim')
    save(out/'branches.json',branchrows);save(out/'property_analysis.json',properties);save(out/'method_decisions.json',decisions);save(out/'summary.json',summary);save(out/'input_hashes.json',inputs);save(out/'final_complete.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),branch_manifest_sha256=sha(ROOT/'branch_manifest.json'),summary_sha256=sha(out/'summary.json')))
    print(json.dumps(summary))
def group_records(records):
    groups=defaultdict(list)
    for r in records:groups[(r['benchmark'],r['layer'],r['anchor'],r['drop_count'])].append(r)
    return groups
def property_analysis(records):
    from scipy.stats import spearmanr
    groups=defaultdict(list);features=['current_error','deleted_mass','bias_Frobenius','bias_relative_Frobenius','bias_entropy_concentration','projected_entropy_weighted_gain','projected_stable_crossings'];perstate=[];byquestion=defaultdict(list)
    for r in records:groups[(r['id'],r['seed'],r['layer'],r['anchor'],r['drop_count'])].append(r)
    for key,rs in groups.items():
        for f in features:
            points=[(r.get(f,r['prechoice_operator_features'].get(f)),r['local_gain']) for r in rs];points=[(x,y) for x,y in points if x is not None]
            corr=None
            if len(points)>=4 and len({x for x,y in points})>1 and len({y for x,y in points})>1:corr=float(spearmanr([x for x,y in points],[y for x,y in points]).statistic)
            perstate.append(dict(id=key[0],seed=key[1],layer=key[2],anchor=key[3],drop_count=key[4],feature=f,candidates=len(points),Spearman=corr))
            if corr is not None:byquestion[(key[0],f)].append(corr)
    summaries={f:bootstrap([float(np.mean(v)) for (q,feature),v in byquestion.items() if feature==f]) for f in features}
    return dict(descriptive_only=True,per_state=perstate,question_clustered_correlations=summaries,availability={'current_error':'extra_current_operator','deleted_mass':'extra_current_operator','bias_vectors':'extra_current_operator','readout_projection':'past_native_pair_and_current_weights_plus_extra_current_operator','entropy_after_pulse':'outcome_only','future_stability':'outcome_only','best_bank_support':'outcome_selected_oracle_not_predictor'},learned_controller=False,selection_does_not_use_future=True)
if __name__=='__main__':main()
