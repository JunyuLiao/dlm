"""CPU-only supplemental interpretation; reads completed frozen report/rows only."""
import collections, hashlib, json, math, pathlib, statistics, sys
P=pathlib.Path(sys.argv[1]); O=pathlib.Path(sys.argv[2])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x): return statistics.mean(x) if x else None
def quant(x,p):
 x=sorted(x)
 if not x:return None
 k=(len(x)-1)*p;lo=int(k);hi=min(lo+1,len(x)-1)
 return x[lo]+(x[hi]-x[lo])*(k-lo)
def write(name,data):
 with (O/name).open('x') as f:
  if isinstance(data,str):f.write(data)
  else:json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
protocol=json.loads((P/'protocol.json').read_text()); summary=json.loads((O/'summary.json').read_text())
# All planned factorial leaves, including incomplete ones, must be explicitly accounted for.
plans=[p for p in protocol['plans'] if p['phase']=='factorial']; rows=[];bindings=[]
for p in plans:
 leaf=P/p['folder']; marker=leaf/'attempt_complete.json'; score=leaf/'scored.json'
 if not marker.exists() or not score.exists():raise RuntimeError(f'Not terminal complete: {leaf}')
 rr=json.loads(score.read_text());rows+=rr
 bindings.append({'engine':p['engine'],'suite':p['suite'],'repeat':p['repeat'],'source_commit':rr[0]['source_commit'],'protocol_sha256':rr[0]['protocol_sha256'],'marker_path':str(marker),'marker_sha256':sha(marker),'scored_path':str(score),'scored_sha256':sha(score),'rows':len(rr)})
groups=collections.defaultdict(dict)
for r in rows:groups[(r['suite'],r['index'],r['panel_seed'],r['engine'])][r['label']]=r
F=('dense_fresh','dense_hold','sparse_fresh','sparse_hold'); L=F+('control_before','control_after')
assert all(set(v)==set(L) for v in groups.values())

def noise(gg):
 aa=[v['control_after']['denoise_forwards']-v['control_before']['denoise_forwards'] for _,v in gg]
 sa=[v['control_after']['score']-v['control_before']['score'] for _,v in gg]
 changes=[]
 for k,v in gg:
  a,b=v['control_before'],v['control_after']; changed=[m for m in ('output_hash','denoise_forwards','canvases','output_tokens','score','finish_reason') if a[m]!=b[m]]
  if changed:changes.append({'index':k[1],'panel_seed':k[2],'engine':k[3],'changed_fields':changed,'before':{m:a[m] for m in changed},'after':{m:b[m] for m in changed},'prefix_metadata_equal':a['diagnostic_receipt']['prefix_metadata']==b['diagnostic_receipt']['prefix_metadata'],'anchor_changed_fields':[m for m in set(a['diagnostic_receipt'].get('control_anchor',{}))|set(b['diagnostic_receipt'].get('control_anchor',{})) if a['diagnostic_receipt'].get('control_anchor',{}).get(m)!=b['diagnostic_receipt'].get('control_anchor',{}).get(m)]})
 return {'groups':len(gg),'signed_mean_N_after_minus_before':mean(aa),'mean_absolute_N':mean(list(map(abs,aa))),'p95_absolute_N':quant(list(map(abs,aa)),.95),'max_absolute_N':max(map(abs,aa),default=None),'mean_absolute_score':mean(list(map(abs,sa))),'score_change_count':sum(x!=0 for x in sa),'hash_change_count':sum(v['control_before']['output_hash']!=v['control_after']['output_hash'] for _,v in gg),'dense_fresh_hash_differs_before':sum(v['dense_fresh']['output_hash']!=v['control_before']['output_hash'] for _,v in gg),'dense_fresh_hash_differs_after':sum(v['dense_fresh']['output_hash']!=v['control_after']['output_hash'] for _,v in gg),'changes':changes}

specs={
 'current_sparse_minus_dense':('sparse_minus_dense',1,{'dense_fresh':.5,'dense_hold':.5}),
 'future_hold_minus_fresh':('fresh_minus_hold',-1,{'dense_fresh':.5,'sparse_fresh':.5}),
 'current_sparse_minus_dense_given_fresh':('sparse_minus_dense_fresh',1,{'dense_fresh':1}),
 'current_sparse_minus_dense_given_hold':('sparse_minus_dense_hold',1,{'dense_hold':1}),
 'future_hold_minus_fresh_given_dense':('fresh_minus_hold_dense',-1,{'dense_fresh':1}),
 'future_hold_minus_fresh_given_sparse':('fresh_minus_hold_sparse',-1,{'sparse_fresh':1}),
 'interaction':('interaction',1,{})}

def contrast_table(suite,gg):
 out={};s=summary['suites'][suite];aa=noise(gg)
 for name,(key,sign,weights) in specs.items():
  c=s['contrasts'][key]; delta=sign*c['means']['denoise_forwards'];base=sum(weights[l]*s['arms'][l]['means']['denoise_forwards'] for l in weights) if weights else None
  lo=[sign*x['means']['denoise_forwards'] for x in c['leave_one_question_out']]; ci=sorted(sign*x for x in c['bootstrap']['ci95']['denoise_forwards'])
  subdivisions={}
  for groupkey in ('by_seed','by_engine'):
   subdivisions[groupkey]={}
   for k,z in c[groupkey].items():
    sel=[v for kk,v in gg if str(kk[2 if groupkey=='by_seed' else 3])==k];base_sub=mean([sum(w*v[l]['denoise_forwards'] for l,w in weights.items()) for v in sel]) if weights else None
    d=sign*z['means']['denoise_forwards'];subdivisions[groupkey][k]={'mean_delta_N':d,'reference_mean_N':base_sub,'relative_delta_percent':100*d/base_sub if base_sub else None,'mean_score_delta':sign*z['means']['score']}
  out[name]={'report_contrast':key,'orientation_multiplier':sign,'mean_delta_N':delta,'reference_definition':weights,'reference_mean_N':base,'relative_delta_percent':100*delta/base if base else None,'N_ci95_question_clustered':ci,'mean_score_delta':sign*c['means']['score'],'score_ci95_question_clustered':sorted(sign*x for x in c['bootstrap']['ci95']['score']),'leave_one_question_out_N_range':[min(lo),max(lo)],'leave_one_question_out_sign_changes':min(lo)<0<max(lo),'N_tails_report_orientation':c['tails']['denoise_forwards'],'absolute_mean_delta_over_mean_absolute_AA_N':abs(delta)/aa['mean_absolute_N'] if aa['mean_absolute_N'] else None,'absolute_mean_delta_over_max_absolute_AA_N':abs(delta)/aa['max_absolute_N'] if aa['max_absolute_N'] else None,'noise_ratio_caveat':'Descriptive comparisons only; max A/A is not a noise bound, and no noise subtraction or causal qualification is implied.',**subdivisions}
 return out

pairs={'current_given_fresh':('sparse_fresh','dense_fresh'),'current_given_hold':('sparse_hold','dense_hold'),'future_given_dense':('dense_hold','dense_fresh'),'future_given_sparse':('sparse_hold','sparse_fresh')}
def flips(gg):
 out={}
 for name,(a,b) in pairs.items():
  by=collections.defaultdict(list)
  for k,v in gg:by[(k[1],k[2])].append({'engine':k[3],'delta_score':v[a]['score']-v[b]['score']})
  kinds={}
  for direction in ('adverse','beneficial'):
   vals=[{'index':q,'panel_seed':se,'repeats':z} for (q,se),z in sorted(by.items()) if len(z)==2 and all(x['delta_score']<0 if direction=='adverse' else x['delta_score']>0 for x in z)]
   counts=collections.Counter(v['index'] for v in vals)
   kinds[direction]={'question_seed_cases':len(vals),'distinct_questions':len(counts),'questions_both_seeds':[q for q,n in counts.items() if n==2],'cases':vals}
  out[name]={'left':a,'right':b,**kinds,'note':'Same score-change direction in both engine repeats; repeats do not provide independent accuracy observations. On RULER these are task-score changes, not binary correct/incorrect flips.'}
 return out

suite_reports={}
for suite in sorted({k[0] for k in groups}):
 gg=[(k,v) for k,v in groups.items() if k[0]==suite];engines=sorted({k[3] for k,_ in gg})
 rep={'paired_groups':len(gg),'distinct_questions':len({k[1] for k,_ in gg}),'seeds':sorted({k[2] for k,_ in gg}),'engines':engines,'all_six_recorded_prefix_match_groups':sum(len({v[l]['diagnostic_receipt']['prefix_signature'] for l in L})==1 for _,v in gg),'all_four_exposed_complete_scope_groups':sum(all(v[l]['diagnostic_receipt']['exposed'] and v[l]['diagnostic_receipt']['complete_scope'] and not v[l]['diagnostic_receipt']['errors'] for l in F) for _,v in gg),'aa_pooled':noise(gg),'aa_by_engine':{str(e):noise([(k,v) for k,v in gg if k[3]==e]) for e in engines},'contrasts':contrast_table(suite,gg),'repeated_score_changes':flips(gg)}
 repeat=collections.defaultdict(list)
 for k,v in gg:
  for label,r in v.items():repeat[(k[1],k[2],label)].append(r)
 changes=[]
 for (q,se,label),rr in sorted(repeat.items()):
  if len(rr)!=2:continue
  a,b=sorted(rr,key=lambda x:x['engine']);fields=[m for m in ('output_hash','denoise_forwards','canvases','output_tokens','score','finish_reason') if a[m]!=b[m]]
  if fields:changes.append({'index':q,'panel_seed':se,'label':label,'engines':[a['engine'],b['engine']],'changed_fields':fields,'before':{m:a[m] for m in fields},'after':{m:b[m] for m in fields},'full_recorded_prefix_metadata_equal':a['diagnostic_receipt']['prefix_metadata']==b['diagnostic_receipt']['prefix_metadata'],'event_list_equal':a['diagnostic_receipt']['events']==b['diagnostic_receipt']['events'],'recorded_post_event_entries_equal':a['refresh_post_event_states']==b['refresh_post_event_states'],'full_state_scope':'Full self-conditioning is hashed only at the first post-event entry; later compact agreement cannot localize earliest hidden-state divergence. Tensor fingerprints bind contiguous bytes, not layout/stride/dtype/alignment or all consumer inputs.'})
 rep['same_arm_engine_repeat_changes']=changes
 decomposition={}
 for label,(a,b) in pairs.items():
  cats={}
  for cat in ('improved','unchanged','worsened'):
   selected=[]
   for k,v in gg:
    ds=v[a]['score']-v[b]['score']; actual='improved' if ds>0 else 'worsened' if ds<0 else 'unchanged'
    if actual==cat:selected.append((k,v))
   nd=[v[a]['denoise_forwards']-v[b]['denoise_forwards'] for k,v in selected]
   cats[cat]={'paired_engine_groups':len(selected),'question_seed_cases':len({(k[1],k[2]) for k,v in selected}),'distinct_questions':len({k[1] for k,v in selected}),'mean_delta_N':mean(nd),'sum_delta_N':sum(nd),'mean_delta_score':mean([v[a]['score']-v[b]['score'] for k,v in selected]),'cases':[{'index':k[1],'seed':k[2],'engine':k[3],'delta_N':v[a]['denoise_forwards']-v[b]['denoise_forwards'],'delta_score':v[a]['score']-v[b]['score']} for k,v in selected]}
  decomposition[label]={'left':a,'right':b,'score_change_categories':cats,'note':'Post-hoc explanation, not a filtered treatment estimate. All rows remain in headline contrasts. More calls may accompany better answers, so N alone is not a quality-matched harm or a deployment objective.'}
 rep['score_change_vs_call_change_decomposition']=decomposition
 first={}
 for a,b in [('dense_fresh','dense_hold'),('sparse_fresh','sparse_hold'),('dense_fresh','sparse_fresh'),('dense_hold','sparse_hold')]:
  stats=collections.Counter();diff=collections.Counter()
  for _,v in gg:
   x,y=v[a]['refresh_post_event_states'],v[b]['refresh_post_event_states']
   if not x or not y:stats['missing']+=1;continue
   stats['compared']+=1
   fields=set(x[0])|set(y[0]);dd=[f for f in fields if x[0].get(f)!=y[0].get(f)]
   stats['exact_recorded_first_entry_matches' if not dd else 'recorded_first_entry_differs']+=1;diff.update(dd)
  first[a+'_vs_'+b]={'counts':dict(stats),'field_difference_counts':dict(diff)}
 rep['first_post_event_factor_separation']=first
 rep['all_finish_reasons']=dict(collections.Counter(r['finish_reason'] for k,v in gg for r in v.values()))
 suite_reports[suite]=rep

result={'status':'complete_observed_factorial_supplement','scope':'Bounded first-progress-event mechanism diagnostic; audit time is not deployable latency. No cheap selector is tested.','bindings':bindings,'report_summary_sha256':sha(O/'summary.json'),'protocol_sha256':sha(P/'protocol.json'),'source_identity':sorted({r['source_commit'] for r in rows}),'suites':suite_reports,'decision':{'coverage':'Eight LB questions, both seeds and two engine repeats satisfy the minimum exposure/coverage condition if all complete scope counts equal32. RULER is reported separately.','structural_vs_causal':'Frozen summary passed prefix/integrity gates describe recorded checks; they do not override the A/A comparability gate or imply cloned full-engine state.','causal_promotion':'unresolved','reason':'Observed A/A N/hash drift and same-arm engine-repeat drift prevent an unconditional clean causal or exact reproducibility claim. Current-output conditional effects also fail seed and leave-one-question-out stability. The frozen rule did not specify a statistical noise estimator; reporting mean-noise and maximum excursions avoids silently choosing a permissive interpretation.','mechanism_triage':'Future-support preservation under sparse current output is the stronger follow-up hypothesis: repeated adverse score changes, positive N cost of holding across seeds/engines/leave-one-question-out. It is not a qualified deployable selector or a proof of universal dense-output necessity.','deployable_headroom':'No >=10% request-latency benefit established by this factorial audit. See separate clean headroom/profile review; its audit trajectories differ from clean and critical-path transfer is unresolved.','required_next_step':'Resolve repeatability and qualify a useful regime before promoting selector engineering or serving-speed claims. Retain the inherited V31 and fixed refresh as baselines.'},'limitations':['Previously examined bounded prompts; eight LB question clusters do not estimate full-benchmark accuracy.','Engine repeats remain in question-cluster bootstrap and are not independent accuracy samples.','All opposite-direction and tail cases are retained; no nonsignificance implies noninferiority.','CURRENT output uses the common old mask; FUTURE map is exact fresh versus held. This is not fresh-mask sparse output at the same event.','First-event intervention is not a test of replacing every refresh, initialization, or commit.','The frozen summary uses generic full-state prefix wording; only the measured recorded prefix/first-GLOBAL fingerprint scope is justified.','RULER scorer values are partial-credit task scores on0–1 scale; .875 is not necessarily7/8 fully correct.','No numerical wall-time comparison from factorial instrumented rows is interpreted as speedup.']}
write('mechanism_review.json',result)
lines=['The bounded factorial diagnostic is complete. Its implementation and recorded-prefix checks pass, but clean causal promotion remains unresolved under the frozen A/A rule. Current-output effects are heterogeneous; preserving future support with sparse current output is the stronger provisional mechanism signal.','', 'This supplement qualifies the frozen summary: its prefix gate means matched measured prefix/first-GLOBAL fingerprints, not a cloned full-engine checkpoint. It does not override the A/A/no-op comparability condition.','']
for suite,s in suite_reports.items():
 lines += [f'**{suite}**: {s["paired_groups"]} paired groups, {s["distinct_questions"]} questions, seeds {s["seeds"]}, engines {s["engines"]}. All-six recorded prefixes match in {s["all_six_recorded_prefix_match_groups"]} groups; all-four exposed/complete scope in {s["all_four_exposed_complete_scope_groups"]}.','']
 a=s['aa_pooled'];lines += [f'A/A absolute N difference: mean {a["mean_absolute_N"]:.5g}, p95 {a["p95_absolute_N"]:.5g}, max {a["max_absolute_N"]}; {a["hash_change_count"]} output-hash disagreements and {a["score_change_count"]} score disagreements. Mean absolute N by engine: '+', '.join(f'{e}: {z["mean_absolute_N"]:.5g}' for e,z in s['aa_by_engine'].items())+'. These observed excursions are descriptive, not bounds on unknown nondeterminism.','', '| Contrast (positive means more calls after removal) | Mean ΔN | Explicit reference N | Relative ΔN | Question-cluster 95% interval | LOO ΔN range |','|---|---:|---:|---:|---|---|']
 for name,c in s['contrasts'].items():
  if name=='interaction':continue
  lo,hi=c['N_ci95_question_clustered'];ll,lh=c['leave_one_question_out_N_range'];lines.append(f'| {name} | {c["mean_delta_N"]:.4f} | {c["reference_mean_N"]:.4f} | {c["relative_delta_percent"]:+.2f}% | [{lo:.3f}, {hi:.3f}] | [{ll:.3f}, {lh:.3f}] |')
 lines += ['', 'The main current-output reference averages DF and DH; the main future-support reference averages DF and SF. Conditional references are the right-hand baseline named in the JSON. The intervals are descriptive and do not establish noninferiority.','', '| Repeated score change | Adverse question/seed cases (distinct questions) | Beneficial question/seed cases (distinct questions) |','|---|---:|---:|']
 for name,c in s['repeated_score_changes'].items():lines.append(f'| {name} | {c["adverse"]["question_seed_cases"]} ({c["adverse"]["distinct_questions"]}) | {c["beneficial"]["question_seed_cases"]} ({c["beneficial"]["distinct_questions"]}) |')
 lines+=['','A case counts once only when its score-change direction repeats in both engines. Engine repeats are technical repetitions, not independent accuracy observations; RULER changes can be partial credit. All case identities, per-seed/per-engine effects, repeat mismatches, and signed tail contributions remain in the JSON.','']
for suite,s in suite_reports.items():
 lines += [f'**Post-hoc score/call decomposition: {suite}**', '', '| Contrast | Score change | Engine groups (question/seed cases) | Mean ΔN |', '|---|---|---:|---:|']
 for label in ('current_given_fresh','future_given_dense','future_given_sparse'):
  for cat,z in s['score_change_vs_call_change_decomposition'][label]['score_change_categories'].items():
   v='n/a' if z['mean_delta_N'] is None else f'{z["mean_delta_N"]:+.3f}'
   lines.append(f'| {label} | {cat} | {z["paired_engine_groups"]} ({z["question_seed_cases"]}) | {v} |')
 lines += ['', 'These are descriptive outcome strata, not filtered causal estimates. Technical repeats are not independent answer observations. Every group remains in the headline comparisons.', '']
lines+=['For LongBench, SF−DF has opposite seed signs and changes sign when Q77 is omitted; SH−DH also has a leave-one-question-out sign reversal. Pooled current-output effects exceed average A/A variation but are smaller than the largest observed A/A excursion. Same-arm sparse-fresh engine-repeat drift adds a33-call difference with unchanged score. These facts prevent a clean output-dominant conclusion.','', 'Holding future support has a more stable N penalty, especially with sparse current output, and repeats adverse as well as beneficial score changes. Nevertheless, the dense-current future effect is nearly zero in one seed, substantial tails remain, and A/A drift qualifies the overall causal claim. The defensible decision is provisional mechanism triage, not a gate pass or a new method claim.','', 'All factorial timings include diagnostic work. No >=10% incremental request-latency benefit is established; profile-to-clean transfer remains unresolved in the separate headroom review. This single-event study does not test a cheap selector, replacement of every refresh, or initialization/commit removal.','', 'More denoising calls are not automatically harmful at different quality. For example, Q77 sparse-fresh extra calls accompany beneficial answer changes. The post-hoc score-change/call-change decomposition retains every row in headline effects and must not be treated as a filtered causal estimate.', '', 'Full self-conditioning was recorded only at the first post-event state. Equality of later compact entries does not prove equality of full hidden states or locate the earliest divergence.']
write('mechanism_review.md','\n'.join(lines)+'\n')
write('mechanism_review_analysis.py',pathlib.Path(__file__).read_text())
receipt={'status':'complete','kind':'CPU-only supplemental analysis','input_protocol_sha256':sha(P/'protocol.json'),'frozen_summary_sha256':sha(O/'summary.json'),'analysis_program_sha256':sha(pathlib.Path(__file__)),'factorial_inputs':bindings,'outputs':{x:sha(O/x) for x in ('mechanism_review.json','mechanism_review.md','mechanism_review_analysis.py')}}
write('mechanism_review_receipt.json',receipt)
print(json.dumps({'rows':len(rows),'groups':len(groups),'outputs':receipt['outputs'],'receipt_sha256':sha(O/'mechanism_review_receipt.json')},indent=2))
