"""Align branch-specific native phases, distinguish bytes from active numbers."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1];B=R/'boundary/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def main():
 assert json.loads((B/'execution_complete.json').read_text())['passed'];out=B/'analysis001';out.mkdir(exist_ok=False);inputs={};cases=[];control_checks=[]
 def read(p):inputs[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
 def eq(a,b,key):
  if key not in a or key not in b:return None
  x,y=a[key],b[key]
  if x is None or y is None:return x==y
  return dict(byte_equal=x['byte_sha256']==y['byte_sha256'],numerical_equal=x['numerical_sha256']==y['numerical_sha256'],method_nonfinite=sum(x.get(k,0) for k in ['nan','positive_inf','negative_inf']),reference_nonfinite=sum(y.get(k,0) for k in ['nan','positive_inf','negative_inf']),method_negative_zero=x.get('negative_zero'),reference_negative_zero=y.get('negative_zero'))
 def phases(x):
  events=x['boundary_events'];target=x['anchor']['canvas']
  ordinary=[e for e in events if e['phase']=='ordinary_sampler_exit' and e['canvas']==target]
  def one(name):
   z=[e for e in events if e['phase']==name];assert len(z)==1,(x['label'],name,len(z));return z[0]
  assert ordinary
  return dict(terminal=ordinary[-1],commit_model=one('native_commit_model_entry'),commit_sampler_entry=one('commit_sampler_entry'),initialization=one('commit_sampler_exit_and_next_initialization'),next_model=one('next_canvas_first_model_input'),next_sampler=one('next_canvas_first_sampler_exit'))
 for idx in [0,1,3]:
  control=read(B/f'executions/references/cell{idx:03d}_A_observed/receipt.json');cp=phases(control)
  q=read(B/f'executions/references/cell{idx:03d}_qualification.json');assert q['passed'];control_checks.append(q)
  for path in sorted((B/'executions/development').glob(f'cell{idx:03d}_*/receipt.json')):
   s=read(path)
   if not s['runtime']['pulse_calls']:continue
   q=read(B/f'executions/development/{s["label"]}_qualification.json');assert q['passed'];sp=phases(s);comparisons={}
   for name in cp:
    a,b=sp[name],cp[name];fields={k:eq(a,b,k) for k in ['input_ids','inputs_embeds','self_condition','stochastic_canvas','argmax_canvas','emitted_canvas','valid_canvas_length'] if k in a or k in b}
    states={k:eq(a.get('states',{}),b.get('states',{}),k) for k in set(a.get('states',{}))|set(b.get('states',{}))}
    layers={}
    for layer in sorted(set(a.get('layers',{}))|set(b.get('layers',{})),key=int):
     aa=a['layers'][layer];bb=b['layers'][layer];layers[layer]={k:eq(aa,bb,k) for k in ['query','current_key','current_value','committed_K','committed_V','output']}
     layers[layer]['same_active_lengths']=(aa['active_key_start'],aa['active_key_end'])==(bb['active_key_start'],bb['active_key_end'])
    comparisons[name]=dict(method_canvas=a['canvas'],reference_canvas=b['canvas'],method_N=a['N'],reference_N=b['N'],rng_cpu_equal=a['rng']['cpu']==b['rng']['cpu'],rng_cuda_equal=a['rng']['cuda']==b['rng']['cuda'],fields=fields,states=states,layers=layers,
      cumulative_output_prefix_equal=a.get('cumulative_output_prefix_sha256')==b.get('cumulative_output_prefix_sha256') if name=='initialization' else None)
   cm=comparisons['commit_model'];init=comparisons['initialization'];terminal=comparisons['terminal'];committed_equal=all(v[k]['numerical_equal'] for v in cm['layers'].values() for k in ['committed_K','committed_V'])
   assert len(cm['layers'])==30 and len(comparisons['next_model']['layers'])==30
   cases.append(dict(id=s['id'],seed=s['seed'],label=s['label'],old_output_reproduced=True,remaining=s['remaining_calls'],A_remaining=control['remaining_calls'],N=s['N'],A_N=control['N'],T=s['T'],A_T=control['T'],correct=s['correct'],A_correct=control['correct'],
     terminal_argmax_numerically_equal=terminal['states']['argmax_canvas']['numerical_equal'],commit_input_numerically_equal=cm['fields']['input_ids']['numerical_equal'],
     committed_cache_all30_layers_numerically_equal=committed_equal,cumulative_emitted_prefix_equal=init['cumulative_output_prefix_equal'],
     initialization_canvas_equal=init['states']['stochastic_canvas']['numerical_equal'],initialization_SC_numerically_equal=init['states']['self_condition']['numerical_equal'],
     initialization_RNG_equal=init['rng_cuda_equal'],whole_output_equal=s['output_sha256']==control['output_sha256'],comparisons=comparisons))
 assert len(cases)==10
 save(out/'summary.json',dict(cases=cases,qualified_old_requests=10,qualified_common_states=3,controls=control_checks,
  interpretation='Compare post-native-commit active caches and numerical reset state before attributing next-input differences to initialization/RNG position. RNG is not independently manipulated and is not claimed as the sole mediator.',
  phase_semantics='commit_model records model-entry state and per-layer active cache/output AFTER that native layer executes; next_model cache includes its current stochastic canvas tail. Never call inactive allocation capacity committed state.',
  signed_zero='Canonicalized CPU copies only; nonfinite statistics separately explicit. Native tensors untouched.',no_extra_forwards=True,no_RNG_realignment=True))
 save(out/'input_hashes.json',inputs)
 lines=['# Exact replay commit-boundary diagnostic','', '|ID / seed|Support|Remaining A→S|Terminal argmax equal|Commit cache numerical equal (30layers)|Next init canvas equal|Init CUDA RNG equal|Full output equal|','|---|---|---|---|---|---|---|---|']
 for x in cases:lines.append(f"|{x['id']} / {x['seed']}|{x['label']}|{x['A_remaining']}→{x['remaining']}|{x['terminal_argmax_numerically_equal']}|{x['committed_cache_all30_layers_numerically_equal']}|{x['initialization_canvas_equal']}|{x['initialization_RNG_equal']}|{x['whole_output_equal']}|")
 lines+=['','All requests finish natively and reproduce historical output,N,C,T,RNG/forward fingerprints. Same-target tokens are not task correctness. Complete conditional phase differences and byte/numerical/nonfinite comparisons are in summary.json. No restoration, RNG compensation, extra model forward or new selector.'];(B/'summary.md').write_text('\n'.join(lines)+'\n');save(B/'final_complete.json',dict(passed=True,protocol_sha256=sha(B/'protocol.json'),summary_sha256=sha(out/'summary.json'),request_scope='exact three historical H states; ten distinct supports'))
if __name__=='__main__':main()
