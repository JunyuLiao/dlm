"""Freeze the best already-completed two-stage calibration trace point."""
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); target=int(sys.argv[2])
trace=json.loads((root/'calibration_traces'/f'temporal_s{target}.json').read_text())
g=target/100.0
def err(x):
 m=x['metrics']; vals=[abs(m['sparsity'][k]-g) for k in ('whole','local','global')]
 vals += [abs(m['phase_sparsity'][p][k]-g) for p in ('call1','call2') for k in ('whole','local','global')]
 return max(vals), max(abs(m['sparsity'][k]-g) for k in ('whole','local','global'))
x=min(trace,key=err); m=x['metrics']; violations=[]
for k in ('whole','local','global'):
 if abs(m['sparsity'][k]-g)>.02: violations.append('pooled_'+k)
for p in ('call1','call2'):
 for k in ('whole','local','global'):
  if abs(m['phase_sparsity'][p][k]-g)>.02: violations.append(p+'_'+k)
out=dict(fingerprint=json.loads((root/'configuration.json').read_text())['fingerprint'],method='temporal',target=target,policy=x['policy'],calibration_ids=['aime26/2','aime26/8','aime26/14','aime26/20','aime26/23','aime26/30'],selected_metrics=m,violations=violations,candidates=trace,phase_target=dict(call1=target,call2=target,pooled=target),search='two-stage cached complete-trajectory point frozen after worker interruption',status='attained' if not violations else 'unattainable_under_guardrails')
(root/'thresholds'/f'temporal_s{target}.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(target=target,status=out['status'],sparsity=m['sparsity'],phase_sparsity=m['phase_sparsity'],violations=violations)))
