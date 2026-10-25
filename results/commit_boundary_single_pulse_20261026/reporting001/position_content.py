"""Private winner comparisons, public position metrics and descriptive text counts."""
from pathlib import Path
import hashlib,json
from collections import defaultdict
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];C=R/'panel/attempt003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert json.loads((C/'execution_complete.json').read_text())['passed'];out=C/'diagnostics001';out.mkdir(exist_ok=False);inputs={};cells=[];content=[]
 def read(p):inputs[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
 protocol=read(C/'protocol.json');manifest=read(C/'manifest_identity.json')
 def private(row,arm,i):
  p=C/f'private/generations/panel/{arm}/audit{i:03d}.pt';assert sha(p)==row['private_reference_sha256'];inputs[str(p.relative_to(R))]=sha(p);return torch.load(p,weights_only=True)
 records={}
 for arm in protocol['arms']:
  for i in range(24):records[(arm,i)]=read(C/f'runs/panel/{arm}/audit{i:03d}.json')
 for i in range(24):
  a=records[('matched_allkept_layer29',i)];ap=private(a,'matched_allkept_layer29',i);banks=[]
  for arm in protocol['arms'][3:]:
   s=records[(arm,i)];rt=s['runtime']
   if rt['candidate_bank'] is not None:banks.append((arm,rt['candidate_bank']))
   if not rt['pulse_calls']:continue
   sp=private(s,arm,i);anchor=s['anchor'];canvas=anchor['canvas'];step=anchor['after_completed_calls']+1;effects=[]
   for label,native_step in [('instantaneous_pulse',step),('next_ordinary_same_canvas_propagation',step+1)]:
    sv=next((v for v in s['native_trace'] if not v['commit'] and v['canvas']==canvas and v['native_step']==native_step),None);av=next((v for v in a['native_trace'] if not v['commit'] and v['canvas']==canvas and v['native_step']==native_step),None)
    if sv is None or av is None:effects.append(dict(phase=label,available=False,method_call_available=sv is not None,A_call_available=av is not None,reason='native branch stops/canvas commits; never substitute next canvas or force a call'));continue
    key=(canvas,native_step);winner_flips=int((sp[key]['winner']!=ap[key]['winner']).sum());de=np.array(sv['entropy_by_position'])-np.array(av['entropy_by_position'])
    effects.append(dict(phase=label,available=True,native_step=native_step,actual_native_winner_changes=winner_flips,entropy_delta_by_position=de.tolist(),entropy_delta_quantiles=np.quantile(de,[0,.05,.5,.95,1]).tolist(),native_confidence=sv['confidence'],A_native_confidence=av['confidence'],native_stable=sv['stable'],A_native_stable=av['stable'],native_history_ready=sv['history_ready'],A_native_history_ready=av['history_ready'],entropy_slack=.005-sv['entropy_mean'],A_entropy_slack=.005-av['entropy_mean'],signed_reference_winner_margin_change_by_position=sv.get('signed_margin_change_by_position'),worst_margin_erosion=sv.get('worst_margin_erosion'),mean_absolute_margin_change=sv.get('mean_absolute_margin_change'),margin_quantiles=sv.get('signed_margin_change_quantiles'),reference_winner_crossings=sv.get('reference_winner_crossings'),stochastic_canvas_byte_changed=sv['fingerprint']['canvas']!=av['fingerprint']['canvas'],SC_byte_changed=sv['fingerprint']['self_condition']!=av['fingerprint']['self_condition'],SC_numerical_difference_not_established_by_bytes=True,history_byte_changed=sv['fingerprint']['history']!=av['fingerprint']['history']))
   bank=rt['candidate_bank'];cells.append(dict(id=s['id'],seed=s['seed'],benchmark=s['benchmark'],stratum=s['stratum'],arm=arm,anchor=anchor,remaining=s['remaining_calls'],A_remaining=a['remaining_calls'],N=s['N'],A_N=a['N'],C=s['C'],A_C=a['C'],T=s['T'],A_T=a['T'],correct=s['correct'],A_correct=a['correct'],effects=effects,directional_contrast=bank['directional_contrast'],CE_vs_sham_matched=True if bank['feasible_common'] else None))
  if banks:assert all(b==banks[0][1] for _,b in banks),'current-state bank differs across scoring policies'
 for task in ['aime26','humaneval','longbench_v2']:
  for stratum in ['all','targeted','coverage']:
   for arm in protocol['arms']:
    group=[v for (ar,_),v in records.items() if ar==arm and v['benchmark']==task and (stratum=='all' or v['stratum']==stratum)]
    keys=['characters','whitespace_words','duplicate_line_fraction','repeated_8gram_fraction','revision_vocabulary_count','repeated_function_definitions']
    content.append(dict(task=task,stratum=stratum,arm=arm,questions=len({v['id'] for v in group}),seed_cells=len(group),metrics={k:float(np.mean([v['text_metrics'][k] for v in group])) for k in keys},warning='Descriptive text properties; not correctness of reasoning, native early-answer time, or reduced distraction.'))
 with (out/'summary.json').open('x') as f:json.dump(dict(exposed_cells=cells,content_aggregates=content,reference_agreement_is_not_task_correctness=True,no_AA_SA_AS_SS_or_reuse_inference=True,entropy_scalar_is_recomputed_from_actual_returned_scaled_logits_native_confidence_flag_authoritative=True),f,indent=2)
 with (out/'input_hashes.json').open('x') as f:json.dump(inputs,f,indent=2,sort_keys=True)
 print('Position and content readout complete; exposed cells',len(cells))
if __name__=='__main__':main()
