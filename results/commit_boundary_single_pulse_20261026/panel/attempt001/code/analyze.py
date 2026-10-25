"""Protocol-identified, question-clustered descriptive end-to-end readout."""
from pathlib import Path
import json,hashlib,math
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def main():
 out=ROOT/'analysis001';out.mkdir(exist_ok=False);p=json.loads((ROOT/'protocol.json').read_text());manifest=json.loads((ROOT/'manifest_identity.json').read_text());identities=manifest['identities'];inputs={};rows=[];by={}
 def read(path):inputs[str(path.relative_to(ROOT))]=sha(path);return json.loads(path.read_text())
 for arm in p['arms']:
  leaf=ROOT/'runs/panel'/arm;assert read(leaf/'final_complete.json')['passed']
  for i in range(24):
   a=read(leaf/f'audit{i:03d}.json');c=read(leaf/f'clean{i:03d}.json');assert read(leaf/f'pair{i:03d}.json')['passed']
   assert (a['id'],a['seed'])==(identities[i//2]['id'],[42,43][i%2]);assert (a['id'],a['seed'],arm) not in by
   by[(a['id'],a['seed'],arm)]=(a,c);r={k:a[k] for k in ['id','seed','benchmark','stratum','arm','N','C','T','N_per_C','correct','answer_parsed','no_final_response','output_capped','anchor','remaining_calls','committed_canvases','uncommitted_tail']}
   r.update(W=c['W'] if c['timing_qualified'] else None,S=c['S'] if c['timing_qualified'] else None,S_per_N=c['S_per_N'] if c['timing_qualified'] else None,timing_qualified=c['timing_qualified'],pulse_calls=a['runtime']['pulse_calls'],abstention=a['runtime']['abstention'],support_sha256=(a['runtime']['selected_support'] or {}).get('support_sha256'),work=a['work'],discovery=a['runtime']['diagnostic_work'],native_canvas_caps=sum(x.get('cap_reached',False) for x in a['native_trace'] if not x['commit']),output_sha256=a['output_sha256']);rows.append(r)
 assert len(rows)==168
 stats=[];comparisons=[];rng=np.random.default_rng(20261026)
 for task in ['aime26','humaneval','longbench_v2']:
  for stratum in ['all','targeted','coverage']:
   group=[r for r in rows if r['benchmark']==task and (stratum=='all' or r['stratum']==stratum)]
   for arm in p['arms']:
    g=[r for r in group if r['arm']==arm];counter=Counter()
    for r in g:
     for scope in ['global','local']:
      counter[scope+'_eligible']+=r['work'][scope+'_eligible_tiles'];counter[scope+'_kept']+=r['work'][scope+'_kept_tiles']
     for k,v in r['discovery'].items():counter['discovery_'+k]+=v
    stats.append(dict(task=task,stratum=stratum,arm=arm,independent_questions=len({r['id'] for r in g}),seed_cells=len(g),correct=sum(r['correct'] for r in g),N=sum(r['N'] for r in g),C=sum(r['C'] for r in g),T=sum(r['T'] for r in g),mean_N_per_C=float(np.mean([r['N_per_C'] for r in g])),mean_clean_W=float(np.mean([r['W'] for r in g if r['W'] is not None])) if any(r['W'] is not None for r in g) else None,exposed=sum(r['pulse_calls'] for r in g),output_caps=sum(r['output_capped'] for r in g),missing_final=sum(r['no_final_response'] for r in g),extraction_failures=sum(not r['answer_parsed'] for r in g),native_canvas_caps=sum(r['native_canvas_caps'] for r in g),physical=dict(counter)))
   for arm in p['arms'][3:]:
    for ref in ['corrected_dense_full','matched_allkept_layer29','current_error_single_pulse','support_sham_single_pulse']:
     if arm==ref:continue
     for cohort in ['policy_all','actually_exposed','common_eligible_four_pulse_policies']:
      cells=[r for r in group if r['arm']==arm]
      if cohort=='actually_exposed':cells=[r for r in cells if r['pulse_calls']]
      if cohort=='common_eligible_four_pulse_policies':cells=[r for r in cells if all(by[(r['id'],r['seed'],ar)][0]['runtime']['pulse_calls']==1 for ar in p['arms'][3:])]
      clusters=defaultdict(list);transitions=[]
      for s in cells:
       a,c=by[(s['id'],s['seed'],arm)];b,d=by[(s['id'],s['seed'],ref)];v={k:math.log(a[k]/b[k]) for k in ['N','C','T','N_per_C']}
       for k in ['W','S','S_per_N']:v[k]=math.log(c[k]/d[k]) if c['timing_qualified'] and d['timing_qualified'] else None
       v['quality_difference']=int(a['correct'])-int(b['correct']);clusters[s['id']].append(v)
       transitions.append(dict(id=s['id'],seed=s['seed'],reference_correct=b['correct'],method_correct=a['correct'],reference_output_capped=b['output_capped'],method_output_capped=a['output_capped'],reference_missing_final=b['no_final_response'],method_missing_final=a['no_final_response'],reference_N=b['N'],method_N=a['N'],method_remaining=a['remaining_calls'],reference_remaining=b['remaining_calls']))
      estimates={}
      for endpoint in ['N','C','T','N_per_C','W','S','S_per_N','quality_difference']:
       vals=[np.mean([z[endpoint] for z in g if z[endpoint] is not None]) for g in clusters.values() if any(z[endpoint] is not None for z in g)]
       if not vals:estimates[endpoint]=None;continue
       values=np.array(vals);boot=values[rng.integers(0,len(values),size=(20000,len(values)))].mean(1);f=(lambda x:float(x)) if endpoint=='quality_difference' else (lambda x:float(np.exp(x)))
       estimates[endpoint]=dict(point=f(values.mean()),CI95=[f(np.quantile(boot,.025)),f(np.quantile(boot,.975))],question_clusters=len(vals))
      comparisons.append(dict(task=task,stratum=stratum,cohort=cohort,method=arm,reference=ref,cells=len(cells),questions=len(clusters),estimates=estimates,correct_to_wrong=sum(t['reference_correct'] and not t['method_correct'] for t in transitions),wrong_to_correct=sum(not t['reference_correct'] and t['method_correct'] for t in transitions),transitions=transitions))
 aliases=[];mechanisms=[]
 for identity in identities:
  for seed in [42,43]:
   support_groups=defaultdict(list)
   aa=by[(identity['id'],seed,'matched_allkept_layer29')][0]
   for arm in p['arms'][3:]:
    x=by[(identity['id'],seed,arm)][0];s=x['runtime']['selected_support']
    if s:support_groups[s['support_sha256']].append(arm)
    if x['runtime']['pulse_calls']:
     same_anchor=x['pre_intervention_native_state']==aa['pre_intervention_native_state'];assert same_anchor
     terminal=x['target_terminal'];at=aa['target_terminal'];mechanisms.append(dict(id=x['id'],seed=seed,arm=arm,remaining=x['remaining_calls'],A_remaining=aa['remaining_calls'],same_prestate=True,
      terminal_argmax_equal=terminal['fingerprint']['native_winner']==at['fingerprint']['native_winner'] if terminal and at else None,
      output_equal=x['output_sha256']==aa['output_sha256'],N=x['N'],A_N=aa['N'],correct=x['correct'],A_correct=aa['correct'],directional_contrast=(x['runtime']['candidate_bank'] or {}).get('directional_contrast')))
   aliases.append(dict(id=identity['id'],seed=seed,support_groups=dict(support_groups)))
 result=dict(rows=rows,aggregates=stats,comparisons=comparisons,support_aliases=aliases,mechanisms=mechanisms,question_clusters_per_task=4,seed_cells_per_task=8,uncertainty='Descriptive question-cluster bootstrap; no noninferiority claim',protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'manifest_identity.json'))
 save(out/'summary.json',result);save(out/'input_hashes.json',inputs)
 lines=['# Small online single-pulse end-to-end panel','', 'Twelve independent questions, two seeds; targeted and coverage strata are separate. All168 audit/clean pairs qualify. Single-pulse policies retain abstentions and selection overhead. No Stage B/C or production promotion.','', '|Task / stratum|Arm|Correct/cells|N|C|T|Mean clean W|Exposed|','|---|---|---:|---:|---:|---:|---:|---:|']
 for x in stats:lines.append(f"|{x['task']} / {x['stratum']}|{x['arm']}|{x['correct']}/{x['seed_cells']}|{x['N']}|{x['C']}|{x['T']}|{x['mean_clean_W']}|{x['exposed']}|")
 lines+=['','Every correctness transition and question-cluster interval is in summary.json. No-pulse equality is not evidence that deletion preserved quality. Correctness regressions, output caps, missing final answers and remaining-call versus complete-work differences must be considered together.'];(ROOT/'summary.md').write_text('\n'.join(lines)+'\n');save(ROOT/'readout_frozen.json',dict(passed=True,summary_sha256=sha(out/'summary.json'),inputs_sha256=sha(out/'input_hashes.json')))
if __name__=='__main__':main()
