from pathlib import Path
import json,time
R=Path(__file__).resolve().parents[1];B=R/'boundary/attempt001';C=R/'panel/attempt003'
def read(p):return json.loads(p.read_text()) if p.exists() else None
b=read(B/'private/controller_heartbeat.json') or {};bh=read(B/'private/heartbeat.json') or {};c=read(C/'private/controller_heartbeat.json') or {};ch=read(C/'private/heartbeat.json') or {}
value=dict(unix=time.time(),boundary=dict(job=b.get('job'),complete_requests=len(list((B/'executions').glob('**/receipt.json'))),qualified_states=sum(read(p)['passed'] for p in (B/'executions/references').glob('*qualification.json')),qualified_unique_pulses=sum(read(p)['passed'] for p in (B/'executions/development').glob('*qualification.json')),engine_executions=bh.get('forwards'),worker_startup_model_calls=bh.get('startup'),completed=(B/'execution_complete.json').exists()),panel=dict(job=c.get('job','waiting_for_boundary'),audit_requests=len(list((C/'runs').glob('**/audit*.json'))),qualified_pairs=len(list((C/'runs').glob('**/pair*.json'))),cohort_frozen=(C/'manifest_binding.json').exists(),engineering_qualified=(C/'qualification_complete.json').exists(),engine_executions=ch.get('forwards'),completed=(C/'execution_complete.json').exists()),readout=read(R/'reporting_execution001/progress.json'))
failures={}
for name,p in [('boundary',B/'controller_stopped.json'),('panel',C/'controller_stopped.json'),('CPU_readout',R/'reporting_execution001/failure.json')]:
 if p.exists():failures[name]=read(p)
value.update(failures=failures,fully_finished=(R/'reporting_execution001/final_complete.json').exists())
print(json.dumps(value,sort_keys=True))
