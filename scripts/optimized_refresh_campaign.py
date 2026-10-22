"""Frozen, resumable multi-suite evaluation of optimized online V2 and V31."""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path('/home/exouser/ljy/dlm')
PYTHON='/home/exouser/dyh/v31_env_20261005/bin/python'
MODEL=Path('/media/volume/new-dllm/dlm-migrated-20260921/cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b')
DEFAULT=ROOT/'results/optimized_refresh_benchmarks_20261021/attempt001'
SUITES=dict(humaneval=dict(dataset='humaneval',count=164,seeds=[42],mage_k=128,max_len=9216,block=64,chunk=4096,memory=.80,prior_tau=.5600716206066294),
 aime26=dict(dataset='aime26',count=90,seeds=[42,43,44],mage_k=128,max_len=9216,block=64,chunk=4096,memory=.80,prior_tau=.021),
 math500=dict(dataset='math500',count=500,seeds=[42],mage_k=128,max_len=16384,block=64,chunk=4096,memory=.80,prior_tau=.021),
 longbench_v2=dict(dataset='longbench_v2_0shot_think',count=503,seeds=[42],mage_k=8192,max_len=141312,block=32,chunk=16384,memory=.85,prior_tau=.021))
ARMS=['v2_online_optimized','v31_optimized','dense_full_fix51994','dense_piecewise','allkept_fa4']
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');tmp.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def prepare(study):
 raise RuntimeError('Legacy full-campaign preparation is disabled by the sparse-execution-first handoff. A new qualified protocol with per-workload held-tile calibration is required.')

def _prepare_legacy(study):
 if study.exists():raise RuntimeError('new attempt required')
 tests=ROOT/'results/optimized_refresh_benchmarks_20261021/preflight/attempt001/tests.xml'
 tree=ET.parse(tests).getroot();assert len(list(tree.iter('testcase')))>=21 and not any(x.tag in ('failure','error','skipped') for x in tree.iter())
 study.mkdir(parents=True);ex=study/'private/execution'
 for folder in ['scripts','experiments','src']:
  shutil.copytree(ROOT/folder,ex/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.png','*.pdf','*.pt','*.npz'))
 libsource=ROOT/'results/v2_online_warpspec_20261021/qualified_source/attempt002/7300f103369c804d';shutil.copytree(libsource,study/'private/kernel')
 shutil.copyfile(ROOT/'docs/OPTIMIZED_REFRESH_BENCHMARKS_20261021.md',study/'spec.md')
 old=ROOT/'results/v31_v2_local_threshold_sweep_20261014_attempt002/private/pools'
 for suite,s in SUITES.items():
  pool=study/f'private/pools/{suite}';pool.mkdir(parents=True)
  ds=s['dataset']
  if suite in ('aime26','longbench_v2'):
   for f in (old/suite).iterdir():shutil.copyfile(f,pool/f.name)
  elif suite=='humaneval':
   source=ROOT/'results/humaneval_value_observe_no_overlap_k128_20261019/attempt001/private/pools/thinking_on'
   for f in source.iterdir():shutil.copyfile(f,pool/f.name)
  else:
   from dllm.models import create_adapter
   adapter=create_adapter('diffusion_gemma',str(MODEL),device='cpu',precision='float32').load_tokenizer()
   raw=Path('/home/exouser/chw/dlm_test/math500.jsonl')
   data=sorted([json.loads(x) for x in raw.read_text().splitlines()],key=lambda x:x['unique_id']);assert len(data)==500
   ms=[];rows=[]
   for i,item in enumerate(data):
    prompt='Solve the following math problem. Show your reasoning and put the final answer in \\boxed{}.\n\n'+item['problem']
    tokens=adapter.encode_prompt(prompt,{'thinking':True});rid=f'math500_{i:04d}'
    ms.append(dict(id=rid,source_id=item['unique_id'],index=i,benchmark='math500',task='math500',thinking=True,prompt=prompt,prompt_tokens=tokens,prompt_token_count=len(tokens),generation_budget=4096))
    rows.append(dict(id=rid,index=i,source_id=item['unique_id'],thinking=True,generation_budget=4096,prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),gold=dict(answer=item['answer'])))
   write(pool/(ds+'_generation_manifest.json'),ms)
   write(pool/(ds+'_rowinfo.json'),dict(schema='v31_rowinfo_v2',dataset=ds,manifest_sha256=sha(pool/(ds+'_generation_manifest.json')),max_model_len=s['max_len'],rows=rows))
   write(study/'math_source_receipt.json',dict(source_sha256=sha(raw),questions=len(ms)))
  ms=read(pool/(ds+'_generation_manifest.json'))
  assert all(x['thinking'] for x in ms)
  cells=[dict(dataset=ds,id=m['id'],index=i,prompt_tokens=m['prompt_tokens'],seed=seed) for seed in s['seeds'] for i,m in enumerate(ms)]
  assert len(cells)==s['count']
  folder=study/f'private/schedules/{suite}';write(folder/'cells.json',cells)
  # Sixteen approximately equal-size shards bound recovery cost.
  shards=math.ceil(len(cells)/32)
  for i in range(shards):write(folder/f'shard{i+1:03d}.json',cells[len(cells)*i//shards:len(cells)*(i+1)//shards])
  ordered=sorted(range(len(ms)),key=lambda i:(len(ms[i]['prompt_tokens']),i))
  # Four deterministic prompt-length strata, all AIME seeds represented later.
  chosen=[ordered[min(len(ordered)-1,(2*j+1)*len(ordered)//8)] for j in range(4)]
  cal=[dict(dataset=ds,id=ms[i]['id'],index=i,prompt_tokens=ms[i]['prompt_tokens'],seed=42) for i in chosen]
  write(folder/'calibration.json',cal);write(folder/'smoke.json',cal[:1])
  cp=study/f'private/calibration_pools/{suite}';cp.mkdir(parents=True)
  cms=[dict(m,generation_budget=256) for m in ms];write(cp/(ds+'_generation_manifest.json'),cms)
 fingerprints={str(p.relative_to(ex)):sha(p) for p in ex.rglob('*') if p.is_file()}
 inputs={str(p.relative_to(study)):sha(p) for folder in ['pools','schedules','calibration_pools'] for p in (study/'private'/folder).rglob('*') if p.is_file()}
 protocol=dict(schema='optimized_refresh_native_v1',created_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),suites=SUITES,arms=ARMS,execution_sha256=fingerprints,input_sha256=inputs,kernel_sha256=sha(study/'private/kernel/v2_online_sm90.so'),spec_sha256=sha(study/'spec.md'),preflight_sha256=sha(tests),native=dict(canvas=256,max_denoise_steps=48,entropy_bound=.1,confidence=.005,stability=1,temperature=[.8,.4],thinking=True,seed_base=31),calibration=dict(first_canvas_diagnostic_output_cap=256,main_output_limits_unchanged=True,strata=4,initial_candidates='prior,prior/2,prior/4,prior*2,prior*4,0',refinements=2,objective='absolute physical GLOBAL full-PV kept tiles in first canvas; tie kept-fraction',target_absolute_relative=.02,target_fraction_absolute=.0025),clean='request/prefill synchronized boundaries; audit separately',user_authorization='2026-10-21 explicit all-suite evaluation with current optimized kernels; prior dense gate superseded')
 from scripts.v31_v2_local_threshold_sweep import runtime_identity
 protocol['runtime']=runtime_identity()
 write(study/'protocol.json',protocol);write(study/'state.json',dict(status='prepared',updated_utc=utc()))
 print(json.dumps(dict(event='prepared',study=str(study),suite_cells={s:x['count'] for s,x in SUITES.items()})),flush=True)
 return protocol

def verify(study):
 p=read(study/'protocol.json');ex=study/'private/execution'
 for rel,h in p['execution_sha256'].items():
  if sha(ex/rel)!=h:raise RuntimeError('execution drift '+rel)
 for rel,h in p['input_sha256'].items():
  if sha(study/rel)!=h:raise RuntimeError('input drift '+rel)
 if sha(study/'private/kernel/v2_online_sm90.so')!=p['kernel_sha256']:raise RuntimeError('kernel drift')
 return p

def environment(study,suite,arm,pass_name,tau,leaf):
 from scripts.v31_v2_local_threshold_sweep import env_for,SUITES as inherited
 inherited[suite]=SUITES[suite]
 env=env_for(study,suite,'current_v31_control')
 env.update(STUDY_ARM=arm,VALUE_AUDIT=str(int(pass_name=='audit')),VALUE_CLEAN_TIMING=str(int(pass_name=='clean')),VALUE_BOUNDARY_TIMING=str(int(pass_name=='clean')),TRACE=str(int(pass_name=='audit')),REPEATS='1',PYTHONHASHSEED='0',TORCHINDUCTOR_DETERMINISTIC='1',FROZEN_SOURCE_SHA256=sha(study/'protocol.json'),OPTIMIZED_REFRESH_METHOD='v2' if arm.startswith('v2') else 'v31' if arm.startswith('v31') else 'dense',OPTIMIZED_REFRESH_LIBRARY=str(study/'private/kernel/v2_online_sm90.so'),OPTIMIZED_REFRESH_OBSERVER_DIR=str(study/'private/cache/optimized_observer'),OPTIMIZED_REFRESH_BASE_WORKER=str(study/'private/execution/scripts/v31_humaneval_cfgdet_base_worker.py'),OPTIMIZED_KERNEL_INVENTORY=str(leaf/'private/kernel_inventory.json'),RESIDENT_SPARSE_EXECUTION='1',RESIDENT_SPARSE_GRAPH_BUCKETS='0',RESIDENT_V31_PAGED_REFRESH='1')
 if arm.startswith('v2'):env.update(VALUE_SELECTOR='value_v2_online_preserve_mass',VALUE_THRESHOLD=str(tau))
 env['OPTIMIZED_WARM_EACH_INPUT']='1'
 env['RESIDENT_ALLKEPT_EXECUTION']='1'
 return env

def validate(rows,suite,arm,pass_name,planned):
 errors=[];expected={(x['index'],x['seed']) for x in planned}
 if len(rows)!=len(planned) or {(x['index'],x['panel_seed']) for x in rows}!=expected:errors.append('coverage')
 for x in rows:
  a=((x.get('receipts') or {}).get('adapter') or {})
  if x['cuda_graph_captures']!=0:errors.append('request graph capture')
  if x['denoise_forwards']<=0 or x['engine_steps']!=x['sampler_calls']+x['prefill_steps']:errors.append('native calls')
  if x['value_audit']!=(pass_name=='audit') or x['value_clean_timing']!=(pass_name=='clean'):errors.append('pass flags')
  if a.get('order_errors',0):errors.append('lifecycle order')
  if pass_name=='clean' and (x.get('request_jit_event_count')!=0 or x.get('request_jit_events')!=[]):errors.append('timed JIT event in universal counter')
  if pass_name=='audit':
   for key in ['global_eligible_tiles','global_kept_tiles','native_local_skipped_tiles']:
    if key not in a:errors.append('missing '+key)
   if a.get('native_local_skipped_tiles')!=0:errors.append('LOCAL skip')
   if a.get('global_calls')!=5*x['denoise_forwards']:errors.append('GLOBAL scope')
   if a.get('native_local_calls_audited')!=25*x['denoise_forwards']:errors.append('LOCAL scope')
   if a.get('value_nonfinite_logits_calls',0) or a.get('value_nonfinite_attention_calls',0):errors.append('nonfinite')
  if arm.startswith(('v2','v31')):
   if not a.get('optimized_refresh_calls') or a['optimized_refresh_extra_model_forwards']!=0:errors.append('missing optimized path')
   if pass_name=='audit' and a['optimized_refresh_budget_errors']!=0:errors.append('budget')
   if not a.get('resident_sparse_execution') or a.get('resident_sparse_extra_forwards')!=0:errors.append('missing resident execution')
   if a.get('resident_sparse_held_kv_copy_calls')!=0 or a.get('resident_sparse_held_selector_calls')!=0 or a.get('resident_sparse_held_list_builds')!=0:errors.append('held-path overhead')
   if not a.get('resident_sparse_held_calls'):errors.append('no held calls')
   if pass_name=='clean' and (a.get('resident_sparse_request_triton_compiles',0)!=0 or bool(a.get('resident_sparse_request_jit_events'))):errors.append('timed JIT event')
   if pass_name=='clean' and x.get('cuda_graph_captures')!=0:errors.append('timed CUDA capture')
   if a.get('optimized_refresh_method')!=('v2' if arm.startswith('v2') else 'v31'):errors.append('refresh method identity')
   if arm.startswith('v2'):
    if a.get('optimized_refresh_library_sha256') is None:errors.append('missing V2 library receipt')
    options=a.get('optimized_refresh_build_options') or {}
    if not options.get('interior_mask') or not options.get('bulk3d'):errors.append('wrong V2 kernel build')
  if arm.startswith('dense') or arm=='allkept_fa4':
   if pass_name=='audit' and a['global_kept_tiles']!=a['global_eligible_tiles']:errors.append('dense work')
  if arm=='allkept_fa4' and (not a.get('resident_allkept_execution') or a.get('resident_allkept_extra_forwards')!=0):errors.append('missing resident all-kept control')
 return sorted(set(errors))

def run_job(study,suite,arm,pass_name,cells,stage,tau=None):
 if (study/'PAUSED_USER_REQUEST.json').exists():raise RuntimeError('Explicitly paused attempt is immutable; create a new qualified frozen attempt.')
 p=verify(study)
 from scripts.v31_v2_local_threshold_sweep import runtime_identity
 if runtime_identity()!=p['runtime']:raise RuntimeError('runtime/host/GPU identity drift')
 label=arm+(('_tau'+format(tau,'.12g').replace('.','p')) if tau is not None else '')
 parent=study/stage/suite/label/pass_name/cells.stem;parent.mkdir(parents=True,exist_ok=True)
 for leaf in sorted(parent.glob('attempt???')):
  if (leaf/'attempt_complete.json').exists():return leaf
 leaf=parent/f'attempt{1+max([int(x.name[-3:]) for x in parent.glob("attempt???")],default=0):03d}';leaf.mkdir();(leaf/'private').mkdir()
 env=environment(study,suite,arm,pass_name,tau,leaf)
 calibration=stage.startswith('calibration');pool=study/f'private/{"calibration_pools" if calibration else "pools"}/{suite}'
 arm_arg='mage' if arm.startswith(('v2','v31')) else 'allkept' if arm=='allkept_fa4' else 'dense';cg='default' if arm=='dense_full_fix51994' else 'PIECEWISE'
 command=[PYTHON,str(study/'private/execution/scripts/optimized_refresh_worker.py'),str(MODEL),str(pool),str(cells),str(leaf/'records.jsonl'),str(leaf/'private/generation.private.jsonl'),arm_arg,cg]
 write(leaf/'config.json',dict(suite=suite,arm=arm,pass_name=pass_name,tau=tau,protocol_sha256=sha(study/'protocol.json'),cells_sha256=sha(cells),command=command))
 write(study/'state.json',dict(status='running',stage=stage,suite=suite,arm=arm,pass_name=pass_name,tau=tau,active_attempt=str(leaf.relative_to(study)),updated_utc=utc()))
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
 if apps.strip():raise RuntimeError('GPU occupied before worker: '+apps)
 print(json.dumps(dict(event='launch',stage=stage,suite=suite,arm=arm,pass_name=pass_name,tau=tau,leaf=str(leaf.relative_to(study)))),flush=True)
 started=time.monotonic()
 with (leaf/'private/worker.log').open('w') as log:
  proc=subprocess.Popen(command,cwd=study/'private/execution',env=env,stdout=log,stderr=subprocess.STDOUT)
  write(leaf/'process.json',dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
  last_progress=time.monotonic()
  while proc.poll() is None:
   time.sleep(5)
   if time.monotonic()-last_progress>=60:
    last_progress=time.monotonic()
    write(study/'heartbeat.json',dict(utc=utc(),pid=proc.pid,active_attempt=str(leaf.relative_to(study)),records=sum(1 for _ in (leaf/'records.jsonl').open()) if (leaf/'records.jsonl').exists() else 0))
 elapsed=time.monotonic()-started
 after_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
 if after_apps.strip():raise RuntimeError('foreign client appeared during worker; attempt withheld: '+after_apps)
 if proc.returncode:
  write(leaf/'failed_attempt.json',dict(returncode=proc.returncode,elapsed_seconds=elapsed));raise RuntimeError('worker failed '+str(leaf))
 rows=[json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines()]
 errors=validate(rows,suite,arm,pass_name,read(cells))
 write(leaf/'receipt.json',dict(validation_errors=errors,records=len(rows),worker_reservation_seconds=elapsed,protocol_sha256=sha(study/'protocol.json')))
 if errors:raise RuntimeError('invalid records '+str(errors)+' '+str(leaf))
 write(leaf/'attempt_complete.json',dict(status='complete',records=len(rows),utc=utc(),protocol_sha256=sha(study/'protocol.json')))
 print(json.dumps(dict(event='complete',suite=suite,arm=arm,pass_name=pass_name,records=len(rows),elapsed=elapsed)),flush=True)
 return leaf

def work(leaf):
 rows=[json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines()];e=k=0
 for x in rows:
  phases=x['receipts']['adapter']['value_canvas_phase_tiles'];first=phases[min(phases,key=int)]
  e+=sum(v['eligible_tiles'] for v in first.values());k+=sum(v['kept_tiles'] for v in first.values())
 return dict(eligible=e,kept=k,kept_fraction=k/e)

def calibrate(study,suite):
 dest=study/f'calibration/{suite}/selected_threshold.json'
 if dest.exists():return read(dest)['threshold']
 cells=study/f'private/schedules/{suite}/calibration.json';s=SUITES[suite]
 ref=work(run_job(study,suite,'v31_optimized','audit',cells,'calibration_control'))
 points={};prior=s['prior_tau']
 def measure(t):
  if t not in points:points[t]=work(run_job(study,suite,'v2_online_optimized','audit',cells,'calibration_candidates',t))
 def key(t):return(abs(points[t]['kept']/ref['kept']-1),abs(points[t]['kept_fraction']-ref['kept_fraction']),t)
 def matched(t):return abs(points[t]['kept']/ref['kept']-1)<=.02 and abs(points[t]['kept_fraction']-ref['kept_fraction'])<=.0025
 measure(prior)
 if not matched(prior):
  for t in [prior/2,prior/4,prior*2,prior*4,0.]:measure(t)
  for iteration in range(2):
   best=min(points,key=key)
   if matched(best):break
   ordered=sorted(points);j=ordered.index(best);intervals=[(ordered[i],ordered[i+1]) for i in [j-1,j] if 0<=i<len(ordered)-1]
   for a,b in intervals:measure(math.sqrt(a*b) if a else b/2)
 best=min(points,key=key)
 write(dest,dict(status='frozen_work_only_calibration',threshold=best,prior_threshold=prior,prior_reused=best==prior,reference=ref,candidates={str(t):v for t,v in points.items()},matched=matched(best),absolute_work_ratio=points[best]['kept']/ref['kept'],fraction_delta_pp=100*(points[best]['kept_fraction']-ref['kept_fraction']),accuracy_used=False,first_canvas_diagnostic_cap=256,protocol_sha256=sha(study/'protocol.json')))
 return best

def run(study):
 if (study/'PAUSED_USER_REQUEST.json').exists():raise RuntimeError('Explicitly paused benchmark attempt must not resume.')
 raise RuntimeError('Full benchmark relaunch is gated on the new sparse-execution qualification and workload-specific held-tile calibration. This legacy first-canvas coordinator is disabled.')

def _run_legacy(study):
 verify(study)
 for suite in SUITES:
  tau=calibrate(study,suite)
  smoke=study/f'private/schedules/{suite}/smoke.json'
  for arm in ARMS:
   audit=run_job(study,suite,arm,'audit',smoke,'qualification',tau if arm.startswith('v2') else None)
   clean=run_job(study,suite,arm,'clean',smoke,'qualification',tau if arm.startswith('v2') else None)
   a=read_lines(audit/'records.jsonl');b=read_lines(clean/'records.jsonl')
   if [(x['output_hash'],x['denoise_forwards']) for x in a]!=[(x['output_hash'],x['denoise_forwards']) for x in b]:raise RuntimeError('audit/clean trajectory mismatch '+suite+' '+arm)
  for cells in sorted((study/f'private/schedules/{suite}').glob('shard???.json')):
   for arm in ARMS:
    for part in ['audit','clean']:run_job(study,suite,arm,part,cells,'runs',tau if arm.startswith('v2') else None)
  score_suite(study,suite)
 write(study/'generation_complete.json',dict(status='complete',utc=utc(),protocol_sha256=sha(study/'protocol.json')))
 from scripts.optimized_refresh_report import campaign_report
 campaign_report(study)

def read_lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def score_suite(study,suite):
 # Stage shared sanitized records and private completions under method labels.
 out=study/f'analysis/{suite}/attempt001'
 if (out/'analysis_complete.json').exists():return
 out.mkdir(parents=True,exist_ok=True);pool=study/f'private/pools/{suite}'
 from scripts import v31_official as op
 from scripts import v31_score_aime as aime,v31_score_math500_bound as mat,v31_score_longbench_official as lb,v31_score_humaneval as he
 records=[];clean={};audit={}
 for arm in ARMS:
  found=[]
  for p in (study/f'runs/{suite}').glob(arm+'*/audit/shard???/attempt???'):
   if (p/'attempt_complete.json').exists():found.append(p)
  rows=[];generations=[]
  for p in sorted(found):rows+=read_lines(p/'records.jsonl');generations+=read_lines(p/'private/generation.private.jsonl')
  if len(rows)!=SUITES[suite]['count']:raise RuntimeError('incomplete audit '+suite+' '+arm)
  audit[arm]=rows
  cr=[]
  for p in sorted((study/f'runs/{suite}').glob(arm+'*/clean/shard???/attempt???')):
   if (p/'attempt_complete.json').exists():cr+=read_lines(p/'records.jsonl')
  if len(cr)!=len(rows):raise RuntimeError('incomplete clean')
  clean[arm]=cr
  paired={op.cell_key(x):(x['output_hash'],x['denoise_forwards']) for x in rows}
  if any(paired[op.cell_key(x)]!=(x['output_hash'],x['denoise_forwards']) for x in cr):raise RuntimeError('audit/clean full panel mismatch '+suite+' '+arm)
  for g in generations:records.append((arm,op.cell_key(g),g))
 from scripts import v31_score_ruler as ruler
 fields=('source_id','thinking') if suite=='aime26' else ('thinking','gold') if suite=='math500' else lb.FIELDS if suite=='longbench_v2' else ruler.FIELDS if suite=='ruler128k' else ('thinking',)
 pools=op.Pools(pool,fields);final,_=op.load_final_response()
 if suite=='aime26':
  answers={str(x['id']):str(x['answer']) for x in read_lines(pool/'aime2026_answers.jsonl')};numeric,_=op.aime_numeric_score();scores,secondary,binding,diag=aime.score(records,answers,pools,final,numeric);scoring=aime.summarize(scores,diag)
 elif suite=='longbench_v2':
  ds=SUITES[suite]['dataset'];gold=read(pool/(ds+'_gold.json'));scores,secondary,binding,diag,meta=lb.score(records,pools,lambda ds:gold,final);scoring=lb.summarize(scores,diag,meta)
 elif suite=='humaneval':
  from scripts import v27_humaneval as harness
  harness.sandbox_preflight();gold=read(pool/'gold.json');scores,secondary,binding,diag,timeouts=he.score(records,gold,harness,pools,workers=4);scoring=he.summarize(scores,diag,{(a,k):x['id'] for a,k,x in records})
 elif suite=='ruler128k':
  ruler_root=Path(read(study/'private/scorer_location.json')['ruler'])
  identity=op.ruler_verify_checkout(ruler_root)
  if identity!=read(study/'protocol.json')['ruler_scorer']:raise RuntimeError('RULER scorer identity drift')
  ds=SUITES[suite]['dataset'];gold=read(pool/(ds+'_gold.json'))
  scores,secondary,binding,diag=ruler.score(records,op.ruler_metrics(ruler_root),op.ruler_task_bases(ruler_root),pools,lambda ds:gold,final)
  scoring=ruler.summarize(scores,diag,pools)
 else:
  verifier=mat.make_verifier();scores={};diag={}
  for arm,k,x in records:
   row,b,valid=op.bind(x,pools);ok,parsed,error=mat.judge(verifier,row['gold']['answer'],final(x['completion'],row['thinking']));scores.setdefault(arm,{})[k]=ok
  scoring={arm:dict(correct=sum(s.values()),cells=len(s),accuracy=sum(s.values())/len(s)) for arm,s in scores.items()}
 write(out/'official.json',scores);write(out/'scoring.json',scoring)
 if suite=='ruler128k':write(out/'binding.json',binding)
 summary={}
 for arm,rows in audit.items():
  cr=clean[arm];ad=[x['receipts']['adapter'] for x in rows];n=sum(x['denoise_forwards'] for x in cr);c=sum(x['canvases'] for x in cr);s=sum(x['decode_s'] for x in cr);w=sum(x['wall_s'] for x in cr)
  ga=sum(x['global_eligible_tiles'] for x in ad);gk=sum(x['global_kept_tiles'] for x in ad);local=sum(x['native_local_eligible_rectangles'] for x in ad)
  first_e=first_k=0
  for a in ad:
   phases=a['value_canvas_phase_tiles'];f=phases[min(phases,key=int)];first_e+=sum(v['eligible_tiles'] for v in f.values());first_k+=sum(v['kept_tiles'] for v in f.values())
  refresh=sum(x.get('optimized_refresh_calls',0) for x in ad);refreshms=sum(x.get('optimized_refresh_gpu_ms') or 0 for x in ad)
  phase_work={}
  for a in ad:
   for phase,row in a.get('value_phase_tiles',{}).items():
    dst=phase_work.setdefault(phase,dict(calls=0,eligible_tiles=0,kept_tiles=0))
    for key in dst:dst[key]+=row.get(key,0)
  for phase,row in phase_work.items():row['sparsity']=1-row['kept_tiles']/row['eligible_tiles'] if row['eligible_tiles'] else None
  resident_phase={}
  for a in ad:
   for phase,row in a.get('resident_sparse_phase_timing',{}).items():
    dst=resident_phase.setdefault(phase,dict(calls=0,gpu_ms=0.))
    dst['calls']+=row.get('calls',0);dst['gpu_ms']+=row.get('gpu_ms',0.)
  for phase,row in resident_phase.items():row['mean_gpu_ms']=row['gpu_ms']/row['calls'] if row['calls'] else None
  summary[arm]=dict(cells=len(cr),correct=sum(scores[arm].values()),accuracy=sum(scores[arm].values())/len(cr),N=n,C=c,N_per_C=n/c,T=sum(x['output_tokens'] for x in cr),S=s,W=w,S_per_N=s/n,mean_W=w/len(cr),global_eligible_tiles=ga,global_kept_tiles=gk,global_sparsity=1-gk/ga,local_eligible_rectangles=local,local_skipped_tiles=0,overall_eligible_rectangles=ga+local,overall_skipped_rectangles=ga-gk,overall_sparsity=(ga-gk)/(ga+local),first_canvas_eligible=first_e,first_canvas_kept=first_k,first_canvas_sparsity=1-first_k/first_e,phase_work=phase_work,refresh_calls=refresh,mean_refresh_gpu_ms=refreshms/refresh if refresh else None,resident_phase_timing=resident_phase)
  if suite=='ruler128k':
   summary[arm].update(correct=sum(v>=100-1.e-9 for v in scores[arm].values()),accuracy=scoring[arm]['official']['_overall']/100,official_score=scoring[arm]['official']['_overall'],per_task=scoring[arm]['official'][SUITES[suite]['dataset']]['tasks'])
 ref=summary['v31_optimized'];v2=summary['v2_online_optimized']
 write(out/'first_canvas_work.json',dict(absolute_kept_ratio=v2['first_canvas_kept']/ref['first_canvas_kept'],kept_fraction_delta_pp=100*(ref['first_canvas_sparsity']-v2['first_canvas_sparsity']),matched_absolute=abs(v2['first_canvas_kept']/ref['first_canvas_kept']-1)<=.02,calibration_transductive=True,total_flops_matched=False))
 write(out/'summary.json',summary)
 from scripts.optimized_refresh_report import paired_report,write_suite_markdown
 task_of={x['index']:x['task'] for x in read(pool/(SUITES[suite]['dataset']+'_rowinfo.json'))['rows']} if suite=='ruler128k' else None
 paired_report(out,scores,clean,audit,task_of=task_of)
 write_suite_markdown(out,suite,summary,read(study/f'calibration_held/{suite}/selected_threshold.json'))
 write(out/'analysis_complete.json',dict(status='complete',utc=utc(),protocol_sha256=sha(study/'protocol.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','verify','job']);p.add_argument('--study',type=Path,default=DEFAULT);p.add_argument('--suite');p.add_argument('--arm',default='v31_optimized');p.add_argument('--pass-name',default='audit');p.add_argument('--tau',type=float);a=p.parse_args();study=a.study.resolve()
 if a.action=='prepare':prepare(study)
 elif a.action=='verify':print(verify(study)['source_commit'])
 elif a.action=='job':run_job(study,a.suite,a.arm,a.pass_name,study/f'private/schedules/{a.suite}/smoke.json','qualification',a.tau)
 else:
  try:run(study)
  except Exception as error:
   previous=read(study/'state.json') if (study/'state.json').exists() else {}
   write(study/'state.json',dict(previous,status='failed',error=str(error),updated_utc=utc()))
   raise
