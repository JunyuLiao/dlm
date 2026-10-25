"""Independent consumer/selector-only checks after the complete panel."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];C=R/'panel/attempt004'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert json.loads((C/'execution_complete.json').read_text())['passed'];out=C/'fairness001';out.mkdir(exist_ok=False)
 protocol=json.loads((C/'protocol.json').read_text());manifest=json.loads((C/'manifest_identity.json').read_text());inputs={};cells=[]
 def read(p):inputs[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
 for i in range(24):
  records={arm:read(C/f'runs/panel/{arm}/audit{i:03d}.json') for arm in protocol['arms']};a=records['matched_allkept_layer29'];methods=[]
  for arm in protocol['arms'][3:]:
   s=records[arm];rt=s['runtime'];checks=dict(shared_anchor=s['anchor']==a['anchor'],shared_prestate=s['pre_intervention_native_state']==a['pre_intervention_native_state'],shared_current_QKV=rt['current_operand_route']==a['runtime']['current_operand_route'],LOCAL_dense=s['work']['native_local_skipped_tiles']==0)
   physical=rt['physical_calls'];skipped=sum(x['skipped'] for x in physical)
   checks.update(physical_deleted_matches=s['work']['global_skipped_tiles']==skipped,single_pulse=sum(x['skipped']>0 for x in physical)==rt['pulse_calls'],protected_dense=all(x['protected_skipped']==0 and x['LOCAL_skipped']==0 and x['current_QKV'] for x in physical))
   if not rt['pulse_calls']:checks['abstention_is_exact_A']=all(s[k]==a[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards'])
   else:
    selected=rt['selected_support'];rows=[x for x in physical if x['skipped']>0];checks['actual_support_exact']=rows[0]['actual_deleted_support']==selected['units'];checks['four_units_equal_budget']=len({len(u['tiles']) for u in selected['units']})==1 and len(selected['units'])==4
   methods.append(dict(arm=arm,passed=all(checks.values()),checks=checks))
  full=records['corrected_dense_full'];pw=records['dense_piecewise'];dense_comparison={k:full[k]==pw[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards']}
  cells.append(dict(id=a['id'],seed=a['seed'],methods=methods,FULL_PIECEWISE_comparison=dense_comparison))
 passed=all(m['passed'] for x in cells for m in x['methods'])
 with (out/'summary.json').open('x') as f:json.dump(dict(passed=passed,cells=cells,protocol_sha256=sha(C/'protocol.json'),manifest_sha256=sha(C/'manifest_identity.json'),canonical_entropy_slack='.005 - mean entropy',FULL_PIECEWISE_treated_as_separate_controls=True),f,indent=2)
 with (out/'input_hashes.json').open('x') as f:json.dump(inputs,f,indent=2,sort_keys=True)
 assert passed,'selector-only fairness failed; preserve affected comparisons, do not promote'
 print('Independent fairness passed:24 paired item-seed cells')
if __name__=='__main__':main()
