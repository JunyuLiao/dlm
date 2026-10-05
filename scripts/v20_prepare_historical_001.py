import json, os, pathlib, subprocess, sys
h=json.loads(pathlib.Path(sys.argv[1]).read_text()); root=pathlib.Path(h['root'])
worker=root/'deploy/cp3_00a2c4d'; py=h['python']; env=dict(os.environ,**h['env'])
assert h['host'] in ('149.165.151.254','149.165.159.64')
limit=4000
assert json.loads((root/'primary/remainder_001.stage.json').read_text())['status']=='complete'
initial=json.loads((root/'primary/remainder_001.json').read_text())
assert initial['execution']['run_events']==308 and initial['execution']['all_runs_ok']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
protocol=worker/'results/fan_m1_m3_multidataset_20260927/frozen_protocol.json'; binding=root/'generation/binding_001.json'
preflight="import sys;from pathlib import Path;from scripts.v20_run import validate_inputs,stage_entries;p,*_=validate_inputs(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5],stage='historical');e=stage_entries(p,sys.argv[4],sys.argv[5],'historical');assert len(e)==50 and len(set(x['block'] for x in e))==25 and min(x['block'] for x in e)>=0;print('CPU_binding_historical50_passed')"
subprocess.run([py,'-c',preflight,str(protocol),str(binding),str(root/'private_inputs'),h['host'],h['uuid']],cwd=worker,env=env,check=True)
stage='historical_001';out=root/'primary';private=root/'private_historical_001';ledger=out/(stage+'.ledger.jsonl')
assert not private.exists() and not ledger.exists()
argv=[py,'-m','scripts.v20_run','--protocol',str(protocol),'--binding',str(binding),'--manifests-dir',str(root/'private_inputs'),'--private',str(private),'--ledger',str(ledger),'--lock',str(out/(stage+'.worker.lock')),'--host',h['host'],'--timeout','900','--stage','historical','--deadline-epoch','1790537280','--gpu-budget-s',str(limit-100),'--remaining-requests','50','--block-guard-s','900']
config=out/(stage+'.wrapper_config.json')
c=dict(schema='v20_request_stage_v1',stage_name=stage,mode='request',argv=argv,private_dir=str(private),ledger=str(ledger),output_dir=str(out),cwd=str(worker))
with config.open('x') as f:json.dump(c,f,indent=2)
launch=dict(cwd=str(worker),env=h['env'],launch_marker=str(out/(stage+'.launch.json')),log=str(out/(stage+'.log')),command=[py,'-m','scripts.v20_stage','--receipt',str(out/(stage+'.stage.json')),'--deadline','1790537280','--max-seconds',str(limit),'--gpu-uuid',h['uuid'],'--',py,str(root/'deploy/cp5_05f9947/scripts/v20_request_stage.py'),'--config',str(config)])
with (out/(stage+'.launch_config.json')).open('x') as f:json.dump(launch,f,indent=2)
print(json.dumps(dict(status='CPU_prepared_not_launched',host=h['host'],stage=stage,max_seconds=limit,planned_executions=50)))
