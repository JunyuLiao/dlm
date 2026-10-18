"""CPU-only deterministic synthesis; raw snapshots and traces remain private."""
from collections import defaultdict
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
CAMPAIGN=ROOT/'results/v31_refresh_repeatability_20261018'
PARENT=CAMPAIGN/'attempt001'
REPLAY=CAMPAIGN/'replay_corrected/attempt002'
OUT=Path(__file__).parent

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean_cfg(d):
 if isinstance(d,dict):return {k:clean_cfg(v) for k,v in d.items() if k!='compiled_object'}
 if isinstance(d,list):return [clean_cfg(v) for v in d]
 return d

probe=read(PARENT/'capture/probe.json')['captures']
engines=[read(REPLAY/f'engine{i:03d}/analysis/summary.json') for i in (1,2)]
num=[]
for i,capture in enumerate(probe):
 a,b=[e['reports'][i] for e in engines]
 checks=[]
 for e in (a,b):
  checks.extend(r for trial in e['auto'] for r in trial['vs_first'].values())
  checks.extend(r for trial in e['auto'] for r in trial['vs_capture'].values())
 cfgs=[clean_cfg(trial['config']) for e in (a,b) for trial in e['auto']]
 cross=[x['hashes']==y['hashes'] for x,y in zip(a['auto'],b['auto'])]
 num.append(dict(snapshot=capture['file'],event=capture['event'],auto_comparisons=len(checks),
  all_comparisons_exact=all(r['exact'] for r in checks),max_abs=max(r.get('max_abs',0) for r in checks),
  config_matches_capture=all(x==clean_cfg(capture['config']) for x in cfgs),
  cross_process_hashes_match=all(cross),input_and_rng_checks=all(e['input_hashes_match'] and e['rng_unchanged'] for e in (a,b)),
  native_capture_rng_unchanged=capture['rng_unchanged'],conditional_split1_repetitions=sum(len(e['conditional_split1']) for e in (a,b))))

attr=read(PARENT/'nvtx/attribution.json')
clean=rows(PARENT/'clean/records.jsonl');traced=rows(PARENT/'nvtx/records.jsonl')
metadata={r['id']:r for r in read(PARENT/'nvtx/probe.json')['scopes']}
counts=attr['attribution']['per_range_activity_counts']
missing_attention=[i for i,m in metadata.items() if m['request']>=3 and ('/global_' in m['label'] or m['label'].endswith('/local_attention')) and int(counts.get(str(i),0))==0]
logs=(PARENT/'nvtx/private/worker.log').read_text(errors='replace').splitlines()
completions=[i for i,line in enumerate(logs) if line.startswith('{"arm":')]
measured_jit=[i+1 for i,line in enumerate(logs) if 'JIT compilation during inference:' in line and len(completions)>=2 and i>completions[1]]
metrics=[]
phase_rows=attr['attribution']['rows']
buffer_rows=read(OUT/'kv_buffer_attribution.json')['buffer_rows']
for idx,(c,t) in enumerate(zip(clean,traced),1):
 if idx<3:continue
 s={r['label']:r for r in phase_rows if r['request']==idx}
 def total(labels,key):return sum(s.get(label,{}).get(key,0) for label in labels)/1e9
 progress=['denoise/global_refresh']
 dense=['denoise/global_refresh','denoise/global_initial','denoise/global_bootstrap_dense','commit/global_dense']
 def costs(labels):
  cpu=total(labels,'cpu_ns');gpu=total(labels,'gpu_sum_ns')
  return dict(cpu_s=cpu,gpu_activity_sum_s=gpu,optimistic_fixed_trajectory_deletion_fraction=(cpu+gpu)/c['wall_s'])
 outside_buffers=[r for r in buffer_rows if r['request']==idx and r['label'] in ('kv_buffer_under_held','kv_buffer_under_carried')]
 extra_buffer_s=sum(r['cpu_ns']+r['gpu_sum_ns'] for r in outside_buffers)/1e9
 metrics.append(dict(request=idx,index=c['index'],policy=c['receipts']['adapter']['refresh_policy'],
  clean_W=c['wall_s'],clean_S=c['decode_s'],N=c['denoise_forwards'],C=c['canvases'],T=c['output_tokens'],
  traced_W=t['wall_s'],progress=costs(progress),all_dense_global_decode=costs(dense),
  held_carried_buffer_cpu_plus_gpu_s=extra_buffer_s,
  dense_plus_buffer_deletion_fraction=costs(dense)['optimistic_fixed_trajectory_deletion_fraction']+extra_buffer_s/c['wall_s'],
  phases={label:dict(calls=r['calls'],cpu_s=r['cpu_ns']/1e9,gpu_activity_sum_s=r['gpu_sum_ns']/1e9,gpu_union_s=r['gpu_union_ns']/1e9) for label,r in s.items()}))

parity=all(read(PARENT/'nvtx'/name)['status']=='matched' for name in ('parity_original.json','parity_new_clean.json'))
qualified=(parity and attr['request_segmentation']['segmentation_status']=='qualified' and not missing_attention
 and not measured_jit and attr['attribution']['ambiguous_launches']==0
 and all(not x['native_counter_mismatches'] for x in attr['scope_counts']))
ledger=read(CAMPAIGN/'gpu_budget.json')['reservations'];budget=defaultdict(float)
for r in ledger:
 assert r['elapsed_s'] is not None,'active reservation'
 budget[r['phase']]+=r['elapsed_s']
result=dict(status='complete_cpu_synthesis',numerical=num,
 numerical_sample_gate_passed=len(num)==6 and all(n['all_comparisons_exact'] and n['config_matches_capture'] and n['cross_process_hashes_match'] and n['input_and_rng_checks'] and n['native_capture_rng_unchanged'] for n in num),
 numerical_scope='three unique input states, two native captures each, sixteen auto replays per snapshot in each of two processes; not a replay of the old failing engine history',
 attribution_gate_passed=qualified,missing_measured_attention_activity=missing_attention,measured_jit_warning_lines=measured_jit,
 runtime_metrics=metrics,reservation_seconds=dict(budget),total_reservation_seconds=sum(budget.values()),cap_seconds=2700.,
 assumptions='Deletion screen sums inclusive parent CPU duration and associated GPU activity duration; deliberately double counts overlap, assumes unchanged future trajectory, and is not a measured speedup.',
 further_selector_launched=False)
inputs=[PARENT/'protocol.json',PARENT/'protocol_description_correction.json',PARENT/'capture/probe.json',PARENT/'capture/historical_hash_comparison.json',
 REPLAY/'protocol.json',OUT/'kv_buffer_attribution.json',PARENT/'nvtx/attribution.json',PARENT/'clean/records.jsonl',PARENT/'nvtx/records.jsonl',CAMPAIGN/'gpu_budget.json']
inputs += [REPLAY/f'engine{i:03d}/{f}' for i in (1,2) for f in ('receipt.json','bootstrap_receipt.json','analysis/summary.json')]
result['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in inputs}
result['analysis_source_sha256']=sha(Path(__file__))
path=OUT/'summary.json'
with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ('status','numerical_sample_gate_passed','attribution_gate_passed','total_reservation_seconds')}))
