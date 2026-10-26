"""Existing-control census; native event order, never CPU emission intervals."""
from pathlib import Path
from collections import Counter,defaultdict
import copy,hashlib,json
import numpy as np

LEAF=Path(__file__).resolve().parents[1]
REPO=LEAF.parents[3]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    assert p.is_relative_to(LEAF),'Writer escaped new census leaf'
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)

def parse(record,study):
    rows=record['native_trace'];canvases=[];current=[];ordinary=0
    for raw in rows:
        row=copy.deepcopy(raw)
        if row['commit']:
            assert current and row['native_step']==0
            if 'canvas' in row:
                assert row['canvas']==len(canvases)+(1 if study=='H' else 0)
            assert current[-1]['converged']
            canvases.append(current);current=[]
            continue
        assert row['native_step']==len(current)+1
        if 'canvas' in row:assert row['canvas']==len(canvases)
        assert row['native_step']<=48
        ready=row['native_history_ready'] if study=='R' else row['history_ready']
        assert all(k in row for k in ['stable','confidence','converged','cap_reached'])
        adaptive=bool(ready and row['stable'] and row['confidence'])
        assert row['cap_reached']==(row['native_step']>=48)
        assert row['converged']==(adaptive or row['cap_reached'])
        assert not current or not current[-1]['converged']
        row['history_ready']=ready
        row['entropy_slack_canonical']=.005-row['entropy_mean'] if 'entropy_mean' in row else None
        if study=='R':
            assert abs(row['entropy_slack']+row['entropy_slack_canonical'])<1e-12
        row['winner_hash']=row.get('native_winner_sha256',(row.get('fingerprint') or {}).get('native_winner'))
        row['predicate_delay']='cap' if row['cap_reached'] else 'none' if adaptive else 'confidence' if ready and row['stable'] else 'stability' if row['confidence'] else 'both'
        current.append(row);ordinary+=1
    assert ordinary==record['N'] and len(canvases)==record['C']
    assert (record['T']+255)//256==record['C']
    assert len(current)==1 and current[0]['native_step']==1,'Preserve actual uncommitted tail; unexpected schema'
    return canvases,current

def describe_canvas(calls,index):
    terminal=calls[-1];full=all(x['winner_hash'] is not None for x in calls)
    settled=None
    if full:
        settled=len(calls)
        while settled>1 and calls[settled-2]['winner_hash']==terminal['winner_hash']:settled-=1
        for a,b in zip(calls,calls[1:]):
            if b['history_ready']:assert b['stable']==(a['winner_hash']==b['winner_hash'])
    opportunities={}
    for rule in ['L1','L2']:
        found=[];previous=False
        for row in calls:
            eligible=row['history_ready'] and row['stable'] and not row['confidence'] and not row['converged']
            t=row['native_step']
            if index>=1 and t<=46 and eligible and (rule=='L1' or previous):
                found.append(dict(t=t,U=max(0,len(calls)-t-1),suffix='unknown' if not full else 'terminal' if t>=settled else 'transient',entropy_slack=row['entropy_slack_canonical']))
            previous=eligible
        opportunities[rule]=found
    adaptive=terminal['history_ready'] and terminal['stable'] and terminal['confidence']
    return dict(canvas=index,m=len(calls),termination='both' if adaptive and terminal['cap_reached'] else 'adaptive' if adaptive else 'cap',
        full_winner_history=full,winner_calls=sum(x['winner_hash'] is not None for x in calls),entropy_calls=sum(x['entropy_slack_canonical'] is not None for x in calls),
        terminal_suffix_start=settled,verified_confidence_only_calls=sum(full and x['native_step']>=settled and x['predicate_delay']=='confidence' for x in calls),
        predicate_counts=dict(Counter(x['predicate_delay'] for x in calls)),opportunities=opportunities)

def main():
    protocol=json.loads((LEAF/'protocol.json').read_text());binding=json.loads((LEAF/'binding.json').read_text())
    assert binding['protocol_sha256']==sha(LEAF/'protocol.json')
    for p,h in protocol['inputs'].items():assert sha(REPO/p)==h,p
    for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
    manifest=json.loads((LEAF/'input_manifest.json').read_text());assert len(manifest['records'])==368
    quality_path=REPO/'results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/analysis001/summary.json'
    quality=json.loads(quality_path.read_text())['rows'];qmap={(v['id'],v['seed'],v['arm']):v for v in quality}
    assert len(qmap)==len(quality)
    meta={}
    for study,p in [('R','results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001/manifest_identity.json'),('H','results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001/manifest_identity.json'),('C','results/commit_boundary_single_pulse_20261026/panel/attempt004/roster_identity.json')]:
        meta[study]={x['id']:x for x in json.loads((REPO/p).read_text())['identities']}
    selected={x['id']:x['stratum'] for x in json.loads((REPO/'results/commit_boundary_single_pulse_20261026/panel/attempt004/manifest_identity.json').read_text())['identities']}
    roots={'R':REPO/'results/response_behavior_and_stopping_headroom_20261025/track_r/attempt001','H':REPO/'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001','C':REPO/'results/commit_boundary_single_pulse_20261026/panel/attempt004'}
    requests=[];canvas_rows=[];identities=set()
    for source in manifest['records']:
        study=source['study'];v=json.loads((REPO/source['path']).read_text());consumer='FULL' if 'full' in source['arm'].lower() else 'A'
        key=(study,consumer,v['id'],v['seed']);assert key not in identities;identities.add(key)
        assert (v['id'],v['seed'])==(source['id'],source['seed'])
        assert v['protocol_sha256']==sha(roots[study]/'protocol.json')
        m=meta[study][v['id']]
        assert v['model_forwards']==(m['prompt_token_count']+16383)//16384+v['N']+v['C']
        if study=='R':
            q=qmap[(v['id'],v['seed'],v['arm'])]
            assert all(v[k]==q[k] for k in ['N','C','T','output_sha256'])
            correct=q['correct'];no_final=q['no_final_response'];parsed=q['quality_detail']['parsed'];cap=q['output_cap']
        else:correct=v['correct'];no_final=v['no_final_response'];parsed=v['answer_parsed'];cap=v['T']==m['generation_budget']
        assert cap==(v['T']==m['generation_budget'])
        canvases,tail=parse(v,study);descriptions=[describe_canvas(cs,i) for i,cs in enumerate(canvases)]
        first={};all_opps={}
        for rule in ['L1','L2']:
            ops=[dict(canvas=d['canvas'],**o) for d in descriptions for o in d['opportunities'][rule]]
            first[rule]=ops[0] if ops else None
            all_opps[rule]=dict(count=len(ops),terminal=sum(x['suffix']=='terminal' for x in ops),transient=sum(x['suffix']=='transient' for x in ops),unknown=sum(x['suffix']=='unknown' for x in ops),hindsight_sum_canvas_first_U=sum(d['opportunities'][rule][0]['U'] for d in descriptions if d['opportunities'][rule]))
        req=dict(study=study,consumer=consumer,raw_arm=source['arm'],protocol_sha256=v['protocol_sha256'],id=v['id'],seed=v['seed'],benchmark=source['benchmark'],N=v['N'],C=v['C'],T=v['T'],uncommitted_tail=len(tail),
            correct=correct,output_capped=cap,no_final_response=no_final,answer_parsed=parsed,domain=m.get('domain'),context_tokens=m['prompt_token_count'],length_band=m.get('length_band'),
            stratum=selected.get(v['id'],'profiling_only') if study=='C' else m.get('stratum',m.get('split')),source_path=source['path'],source_sha256=source['sha256'],
            full_winner_history=all(d['full_winner_history'] for d in descriptions),committed_calls=sum(d['m'] for d in descriptions),verified_confidence_only_calls=sum(d['verified_confidence_only_calls'] for d in descriptions),
            first=first,all_opportunities=all_opps,metadata=m)
        requests.append(req)
        for d in descriptions:canvas_rows.append({k:req[k] for k in ['study','consumer','id','seed','benchmark','correct','output_capped','no_final_response','stratum']}|d)
    counterparts={(r['study'],r['consumer'],r['id'],r['seed']):r for r in requests}
    for r in requests:r['both_controls_correct']=r['correct'] and counterparts[(r['study'],'A' if r['consumer']=='FULL' else 'FULL',r['id'],r['seed'])]['correct']
    rng=np.random.default_rng(20261026);summaries=[]
    for study,task,consumer in sorted({(r['study'],r['benchmark'],r['consumer']) for r in requests}):
        for subset in ['all','correct','incorrect','correct_uncapped_final','output_capped','missing_final','both_controls_correct']:
            group=[r for r in requests if (r['study'],r['benchmark'],r['consumer'])==(study,task,consumer)]
            check={'all':lambda r:True,'correct':lambda r:r['correct'],'incorrect':lambda r:not r['correct'],'correct_uncapped_final':lambda r:r['correct'] and not r['output_capped'] and not r['no_final_response'],'output_capped':lambda r:r['output_capped'],'missing_final':lambda r:r['no_final_response'],'both_controls_correct':lambda r:r['both_controls_correct']}[subset]
            group=[r for r in group if check(r)];ids={r['id'] for r in group};cs=[c for c in canvas_rows if c['study']==study and c['consumer']==consumer and (c['id'],c['seed']) in {(r['id'],r['seed']) for r in group}]
            rules={}
            for rule in ['L1','L2']:
                flags=[r for r in group if r['first'][rule] is not None]
                verified=[r for r in group if r['full_winner_history']]
                eq=defaultdict(list)
                for r in verified:
                    op=r['first'][rule];eq[r['id']].append(int(op is not None and op['suffix']=='terminal' and op['U']>=2))
                vals=np.array([np.mean(x) for x in eq.values()]);interval=None
                if len(vals):
                    boot=vals[rng.integers(0,len(vals),size=(20000,len(vals)))].mean(1);interval=dict(point=float(vals.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),question_clusters=len(vals))
                rules[rule]=dict(first_opportunity_requests=len(flags),first_U=[r['first'][rule]['U'] for r in flags],first_terminal=sum(r['first'][rule]['suffix']=='terminal' for r in flags),first_transient=sum(r['first'][rule]['suffix']=='transient' for r in flags),first_unknown=sum(r['first'][rule]['suffix']=='unknown' for r in flags),verified_U2_equal_question=interval,verified_request_denominator=len(verified),all_opportunity_count=sum(r['all_opportunities'][rule]['count'] for r in group))
            summaries.append(dict(study=study,benchmark=task,consumer=consumer,subset=subset,requests=len(group),questions=len(ids),both_seed_questions=sum({r['seed'] for r in group if r['id']==identity}=={42,43} for identity in ids),N=sum(r['N'] for r in group),C=sum(r['C'] for r in group),committed_calls=sum(r['committed_calls'] for r in group),uncommitted_tail=sum(r['uncommitted_tail'] for r in group),full_winner_requests=sum(r['full_winner_history'] for r in group),verified_confidence_only_calls=sum(r['verified_confidence_only_calls'] for r in group),termination=dict(Counter(c['termination'] for c in cs)),rules=rules))
    gates=[]
    for study in ['R','C','H']:
        for consumer in ['FULL','A']:
            for sensitivity in ['correct','both_controls_correct']:
                for rule in ['L1','L2']:
                    q=defaultdict(set);tasks={}
                    for r in requests:
                        if r['study']!=study or r['consumer']!=consumer or not r[sensitivity] or not r['full_winner_history']:continue
                        op=r['first'][rule]
                        if op is not None and op['suffix']=='terminal' and op['U']>=2:q[r['id']].add(r['seed']);tasks[r['id']]=r['benchmark']
                    replicated=[identity for identity,seeds in q.items() if seeds=={42,43}];passed=len(q)>=6 and len(set(tasks.values()))>=2 and len(replicated)>=3
                    gates.append(dict(study=study,consumer=consumer,sensitivity=sensitivity,rule=rule,qualifying_questions=len(q),tasks=sorted(set(tasks.values())),both_seed_questions=replicated,question_seeds={k:sorted(v) for k,v in q.items()},historical_gate_passed=passed,underidentified_no_full_winners=study!='R'))
    out=LEAF/'analysis001';out.mkdir(exist_ok=False)
    save(out/'requests.json',requests);save(out/'canvases.json',canvas_rows);save(out/'summary.json',dict(aggregates=summaries,historical_gates=gates,records=len(requests),independent_questions=len({r['id'] for r in requests}),protocol_sha256=sha(LEAF/'protocol.json'),no_new_model_or_operator_calls=True,old_gate_not_veto_on_new_authorized_general_study=True))
    save(LEAF/'final_complete.json',dict(passed=True,protocol_sha256=sha(LEAF/'protocol.json'),source_checkpoint=binding['source_checkpoint'],requests=len(requests),committed_canvases=len(canvas_rows),input_manifest_sha256=sha(LEAF/'input_manifest.json'),model_forwards=0,operator_forwards=0,GPU_allocated_seconds=0))
    print(json.dumps(dict(records=len(requests),canvases=len(canvas_rows),independent_questions=len({r['id'] for r in requests}),historical_gates=[g for g in gates if g['sensitivity']=='correct'])))
if __name__=='__main__':main()
