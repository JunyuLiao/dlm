import hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];REPO=R.parents[1];OLD=R.parent/'response_behavior_and_stopping_headroom_20261025';H=OLD/'track_h/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def main():
 inputs={};rows=[]
 def read(p):inputs[str(p.relative_to(REPO))]=sha(p);return json.loads(p.read_text())
 protocol=read(H/'protocol.json');assert all(sha(REPO/p)==h for p,h in protocol['sources'].items())
 assert read(H/'final_complete.json')['passed']
 for p in sorted((H/'executions/development').glob('cell*/receipt.json')):
  s=read(p)
  if not s['runtime']['pulse_calls']:continue
  index=int(s['label'][4:7]);rp=H/f'executions/references/cell{index:03d}_A_observed/receipt.json';a=read(rp)
  def terminal(x):return [z for z in x['native_trace'] if not z['commit'] and z['canvas']==x['anchor']['canvas']][-1]
  st,at=terminal(s),terminal(a)
  pv=p.parent/'private/native_reference.pt';av=rp.parent/'private/native_reference.pt'
  for v,rec in [(pv,s),(av,a)]:assert sha(v)==rec['private_native_reference_sha256'];inputs[str(v.relative_to(REPO))]=sha(v)
  sp=torch.load(pv,weights_only=True);ap=torch.load(av,weights_only=True)
  equal=torch.equal(sp[st['native_step']]['winner'],ap[at['native_step']]['winner'])
  assert equal
  rows.append(dict(id=s['id'],seed=s['seed'],label=s['label'],source=str(p.relative_to(REPO)),source_sha256=sha(p),reference_source=str(rp.relative_to(REPO)),reference_sha256=sha(rp),
   remaining=s['remaining_calls'],reference_remaining=a['remaining_calls'],earlier=s['remaining_calls']<a['remaining_calls'],terminal_native_argmax_equal=equal,
   whole_output_equal=s['output_sha256']==a['output_sha256'],final_cpu_rng_equal=s['final_cpu_rng']==a['final_cpu_rng'],final_cuda_rng_equal=s['final_cuda_rng']==a['final_cuda_rng'],
   terminal_adaptive=st['converged'] and st['confidence'] and st['stable'],terminal_cap=st['cap_reached'],reference_terminal_cap=at['cap_reached'],
   correct=s['correct'],reference_correct=a['correct'],answer_parsed=s['answer_parsed'],no_final_response=s['no_final_response'],output_capped=s['T']==16384,
   N=s['N'],C=s['C'],T=s['T'],terminal_SC_byte_equal=st['native_state_fingerprint']['self_condition']==at['native_state_fingerprint']['self_condition'],
   terminal_SC_numerical_equality='unavailable from old hashes; signed zeros are possible; new boundary observer compares numerical copies',
   deleting_GQA_max=max(x['error'] for x in s['runtime']['GQA_qualification'] if x['deleting'])))
 assert len(rows)==10 and sum(x['earlier'] for x in rows)==5
 assert all(x['whole_output_equal']!=x['earlier'] for x in rows)
 corrections=OLD/'track_r/attempt001/closeout002/physical_work_corrections.json';d=read(corrections);assert d['corrected_rows']==420
 # Read full named inputs for independent hash binding; no historical mutation.
 for name in ['summary.md','COMMANDS.md','closeout001/verification.json','closeout001/method_decisions.json','track_r/attempt001/analysis001/summary.json','track_h/attempt001/analysis001/summary.json','track_h/attempt001/closeout001/native_pulse_effects.json']:
  p=OLD/name;inputs[str(p.relative_to(REPO))]=sha(p)
 checks=[x['error'] for p in (H/'executions').glob('**/receipt.json') for x in json.loads(p.read_text()).get('runtime',{}).get('GQA_qualification',[])]
 result=dict(cases=rows,unique_deleting_requests=10,earlier_stops=5,unchanged_stops=5,terminal_argmax_equal=10,
  canonical_entropy_slack='confidence_threshold - mean_entropy = .005 - entropy; positive means below threshold; R legacy sign is opposite',
  canonical_R_work_source=str(corrections.relative_to(REPO)),stale_R_receipts=420,prior_aggregates_unchanged=True,
  deleting_GQA_max=max(x['deleting_GQA_max'] for x in rows),all_checks_GQA_max=max(checks),
  old_gate_limitations=['omitted complete-request work','no correct-reference coverage required','quality regression gate used A/CE/sham rather than FULL/PW'],
  historical_failed_gate_preserved=True,RNG_position_hypothesis_not_exclusive=True)
 out=Path(__file__).parent;save(out/'summary.json',result);save(out/'input_hashes.json',inputs)
 lines=['# CPU addendum: verified historical single pulses','', 'All ten terminal argmax canvases agree. Five earlier-stop requests change complete outputs; five unchanged-stop requests reproduce complete outputs. This is consistent with, but does not isolate, native RNG-position mediation. Old hashes do not resolve signed-zero SC differences.','', '|ID / seed|Support|A remaining|S remaining|Output equal|CUDA RNG equal|Adaptive / capped|Correct / parsed|','|---|---|---:|---:|---|---|---|---|']
 for x in rows:lines.append(f"|{x['id']} / {x['seed']}|{x['label']}|{x['reference_remaining']}|{x['remaining']}|{x['whole_output_equal']}|{x['final_cuda_rng_equal']}|{x['terminal_adaptive']} / {x['terminal_cap']}|{x['correct']} / {x['answer_parsed']}|")
 lines+=['',f"Deleting GQA maximum {result['deleting_GQA_max']:.9f}; all-control maximum {result['all_checks_GQA_max']:.9f}. Canonical slack =0.005−entropy. The canonical420-row work correction is bound by input SHA256; raw stale fields stay historical. Historical failed gate stays failed; its quality/work omissions are not reused."]
 (out/'summary.md').write_text('\n'.join(lines)+'\n');save(out/'final_complete.json',dict(passed=True,summary_sha256=sha(out/'summary.json'),inputs_sha256=sha(out/'input_hashes.json'),no_model_forwards=True))
 print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
