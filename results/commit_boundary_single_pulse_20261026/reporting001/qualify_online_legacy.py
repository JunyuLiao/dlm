"""Independent current-state bank preservation check, no model calls."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];C=R/'panel/attempt003';OLD=R.parent/'response_behavior_and_stopping_headroom_20261025/track_h/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert json.loads((C/'qualification_complete.json').read_text())['passed'];oldpath=OLD/'executions/references/cell000_A_observed/operators/selections.json';reference=json.loads(oldpath.read_text());rows=[];inputs={str(oldpath.relative_to(R.parents[1])):sha(oldpath)}
 for name in ['current_error','support_sham','direction_high','direction_low']:
  p=C/f'runs/qualify/{name}_single_pulse/audit002.json';v=json.loads(p.read_text());assert v['id']=='longbench_v2/66ec2755821e116aacb1b6f2' and v['seed']==42
  rt=v['runtime'];a=rt['current_operand_route'];b=reference['pre_intervention_route'];same_operands=a is not None and all(a[k]==b[k] for k in ['query','key','value','prefix','cpu_rng','cuda_rng'])
  same_support=rt['selected_support']==reference['bank']['selections'][name]
  if same_operands:assert same_support,'same-current-state selector choices changed'
  rows.append(dict(arm=name,same_old_current_operands=same_operands,same_old_selected_support=same_support,audit_clean_pair_passed=json.loads((p.parent/'pair002.json').read_text())['passed'],actual_pulses=rt['pulse_calls'],GQA_checks=rt['GQA_qualification']))
  inputs[str(p.relative_to(R.parents[1]))]=sha(p)
 out=C/'online_legacy_qualification.json'
 with out.open('x') as f:json.dump(dict(passed=True,cases=rows,input_hashes=inputs,no_model_forwards=True,interpretation='Old/new support equality is required when current operands/RNG agree. A different reference state is not treated as selector failure. All current-state audit/clean pairs independently require exact selected supports.'),f,indent=2)
 print(json.dumps(rows))
if __name__=='__main__':main()
