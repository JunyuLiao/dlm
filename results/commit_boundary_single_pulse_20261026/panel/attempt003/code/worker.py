"""Actual fresh online pulse policies; audit and synchronized clean requests."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import torch
from runtime import Runtime
from measure import summarize,digest,compare_to_reference
from text_metrics import metrics
REPO=Path(__file__).resolve().parents[5];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
START=time.time();LIVE={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['profile','qualify','panel'],required=True);ap.add_argument('--arm',required=True);a=ap.parse_args();root=a.root.resolve()
 protocol=json.loads((root/'protocol.json').read_text());paths=json.loads((root/'private/runtime_paths.json').read_text())
 for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
 out=root/'runs'/a.phase/a.arm;out.mkdir(parents=True,exist_ok=False)
 if a.phase=='profile':rows=json.loads((root/'private/roster.json').read_text());cells=[(x,42) for x in rows]
 elif a.phase=='qualify':
  rows=json.loads((root/'private/roster.json').read_text());old=json.loads((REPO/'results/response_behavior_and_stopping_headroom_20261025/track_h/attempt001/private/manifest.json').read_text());cells=[(next(x for x in rows if x['benchmark']==b),42) for b in ['aime26','humaneval']]+[(old[0],42)]
 else:
  assert json.loads((root/'qualification_complete.json').read_text())['passed']
  assert sha(root/'manifest_identity.json')==json.loads((root/'manifest_binding.json').read_text())['manifest_sha256']
  assert sha(root/'private/manifest.json')==json.loads((root/'manifest_identity.json').read_text())['private_manifest_sha256']
  rows=json.loads((root/'private/manifest.json').read_text());cells=[(x,s) for x in rows for s in [42,43]]
 from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
 from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994,apply_fa4_local_fix
 import torch._inductor.config as cfg
 cfg.deterministic=True;cfg.benchmark_combo_kernel=False;cfg.combo_kernels=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 install_compilation_policy();apply_fix_51994();apply_fa4_local_fix(build_dir=root/'runtime/fa4_local_fix')
 from vllm import LLM,SamplingParams
 from vllm.inputs import TokensPrompt
 from vllm.model_executor.models import diffusion_gemma as dg
 from vllm.v1.worker.gpu.model_runner import GPUModelRunner
 from vllm.utils import jit_monitor
 from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
 from scripts import v31_official as official,v27_humaneval as he
 final,_=official.load_final_response();numeric,_=official.aime_numeric_score();gold={x['task_id']:x for x in json.loads(Path(paths['gold_path']).read_text()).values()};he.sandbox_preflight()
 state=dict(N=0,C=0,bound=False,audit=False,all_forwards=0,startup_forwards=0,executing=False,trace=[],consecutive=0,captures=0,jit=0,anchor_state=None)
 rt=Runtime(state);rt.install();audit=NativeGeometryAudit(['full_attention' if i in [5,11,17,23,29] else 'sliding_attention' for i in range(30)],16);audit.bound=False;audit.install();LIVE.update(state=state,out=out)
 execute0=GPUModelRunner.execute_model;model0=dg.DiffusionGemmaForConditionalGeneration.forward;sample0=dg._compiled_sample_step;cap0=torch.cuda.CUDAGraph.capture_begin;jit0=jit_monitor._handle_jit_event
 def execute(runner,*args,**kw):
  state['all_forwards']+=1;state['executing']=True
  if state['all_forwards']%64==0 and state['audit']:(root/'private/heartbeat.json').write_text(json.dumps(dict(phase=a.phase,arm=a.arm,unix=time.time(),forwards=state['all_forwards'])))
  try:return execute0(runner,*args,**kw)
  finally:state['executing']=False
 def model(m,*args,**kw):
  if not state['executing']:state['startup_forwards']+=1
  return model0(m,*args,**kw)
 def capture(g,*args,**kw):state['captures']+=1;return cap0(g,*args,**kw)
 def jit(**kw):
  if state['bound']:state['jit']+=1
  return jit0(**kw)
 def sample(*args,**kw):
  if not state['bound']:return sample0(*args,**kw)
  if not state['audit'] and not a.arm.endswith('_single_pulse'):
   state['clean_native_calls']+=1;return sample0(*args,**kw)
  commit=bool(args[8][args[1]].item());result=sample0(*args,**kw)
  if commit:
   if state['audit']:state['trace'].append(dict(commit=True,canvas=state['C'],native_step=0,emitted_sha256=digest(args[15][args[2]].reshape(-1)),rng_cpu=digest(torch.get_rng_state()),rng_cuda=digest(torch.cuda.get_rng_state())))
   state['C']+=1;state['consecutive']=0;return result
  state['N']+=1;slot=args[1];length=int(args[14][slot].item());history=args[13][slot];step=int(args[7][slot].item());ready=length>=kw['ST'];stable=bool((history==history[:,:1]).all());conf=bool(args[9][slot].item());conv=bool(args[8][slot].item())
  qualifies=state['C']>=1 and ready and stable and not conf and not conv;state['consecutive']=state['consecutive']+1 if qualifies else 0
  trigger=rt.anchor is None and state['consecutive']>=2 and step<=46
  if trigger:rt.anchor=dict(canvas=state['C'],after_completed_calls=step)
  if state['audit']:
   cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();row=dict(commit=False,canvas=state['C'],native_step=step,history_ready=ready,stable=stable,confidence=conf,converged=conv,cap_reached=step>=48)
   if trigger or (rt.anchor and state['C']==rt.anchor['canvas']):
    measured,fp,private=summarize(result,args,kw);row.update(measured);row['fingerprint']=fp
    state['private_reference'][(state['C'],step)]=private
    if (state['C'],step) in state['reference']:row.update(compare_to_reference(result,state['reference'][(state['C'],step)]))
    if trigger:state['anchor_state']=dict(fingerprint=fp,N=state['N'],C=state['C'])
   state['trace'].append(row)
   assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
  return result
 GPUModelRunner.execute_model=execute;dg.DiffusionGemmaForConditionalGeneration.forward=model;dg._compiled_sample_step=sample;torch.cuda.CUDAGraph.capture_begin=capture;jit_monitor._handle_jit_event=jit
 mode='FULL_AND_PIECEWISE' if a.arm=='corrected_dense_full' else 'PIECEWISE'
 llm=LLM(model=paths['model_path'],dtype='bfloat16',max_model_len=262144,max_num_seqs=1,max_num_batched_tokens=16384,enable_chunked_prefill=True,gpu_memory_utilization=.80,enable_prefix_caching=False,trust_remote_code=False,seed=0,block_size=32,disable_log_stats=True,compilation_config={'cudagraph_mode':mode})
 engine=llm.llm_engine;tokenizer=llm.get_tokenizer();initial=state['all_forwards']
 def request(row,seed,label,is_audit):
  state.update(N=0,C=0,bound=True,audit=is_audit,trace=[],consecutive=0,anchor_state=None,private_reference={},reference={},clean_native_calls=0)
  reference_path=root/'private/generations'/a.phase/'matched_allkept_layer29'/(label+'.pt')
  if is_audit and label.startswith('audit') and a.arm.endswith('_single_pulse') and reference_path.exists():state['reference']=torch.load(reference_path,weights_only=True)
  rt.begin(a.arm,is_audit);audit.begin_request();audit.bound=is_audit
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.synchronize();before=state['all_forwards'];cap=state['captures'];jc=state['jit'];start=time.perf_counter();steps=0;prefill=None;output=None
  engine.add_request(label,TokensPrompt(prompt_token_ids=row['prompt_tokens']),SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
  while engine.has_unfinished_requests():
   values=engine.step();steps+=1
   if steps==math.ceil(len(row['prompt_tokens'])/16384):torch.cuda.synchronize();prefill=time.perf_counter()
   for value in values:
    if value.finished:output=value
  torch.cuda.synchronize();end=time.perf_counter();state['bound']=False;rt.active=False;audit.bound=False
  assert output is not None and steps==state['all_forwards']-before
  value=output.outputs[0];T=len(value.token_ids)
  if not is_audit and not a.arm.endswith('_single_pulse'):
   state['C']=math.ceil(T/256);state['N']=state['clean_native_calls']-state['C']
  assert math.ceil(T/256)==state['C'] and steps==math.ceil(len(row['prompt_tokens'])/16384)+state['N']+state['C']
  cpu_rng=digest(torch.get_rng_state());gpu_rng=digest(torch.cuda.get_rng_state());tokens=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=True)
  if hasattr(tokens,'keys'):tokens=tokens['input_ids']
  assert list(tokens)==row['prompt_tokens']
  text=tokenizer.decode(value.token_ids,skip_special_tokens=False);answer=final(text,True);detail={};b=row['benchmark']
  if b=='aime26':v=numeric(answer,row['expected']);correct=v['correct'];detail['parsed']=v['extracted'] is not None
  elif b=='longbench_v2':parsed=official.lb_extract_answer(answer);correct=parsed==row['expected'];detail['parsed']=parsed is not None
  else:
   problem=gold[row['source_id']];code,reason=he.extract_code(text,problem['entry_point']);correct,outcome=he.run_test(he.program_for(problem,code)) if code is not None else (False,'not_run');detail.update(parsed=code is not None,extraction=reason,outcome=outcome)
  runtime=rt.receipt();work=audit.receipt() if is_audit else None;trace=state['trace'];anchor=rt.anchor;target=[x for x in trace if not x['commit'] and anchor and x['canvas']==anchor['canvas']];terminal=target[-1] if target else None
  if work is not None:
   skipped=sum(x['skipped'] for x in runtime['physical_calls']);work['global_kept_tiles']-=skipped;work['global_skipped_tiles']=skipped;work['overall_skipped_rectangles']=skipped;work['overall_rectangle_sparsity']=skipped/work['overall_eligible_rectangles'];work['native_dense_phase_geometry_baseline']=work.pop('value_phase_tiles')
  record=dict(id=row['id'],seed=seed,benchmark=b,stratum=row.get('stratum','qualification'),arm=a.arm,label=label,audit=is_audit,N=state['N'],C=state['C'],T=T,N_per_C=state['N']/state['C'],model_forwards=steps,output_sha256=hashlib.sha256(json.dumps(list(value.token_ids)).encode()).hexdigest(),final_cpu_rng=cpu_rng,final_cuda_rng=gpu_rng,
   native_trace=trace,anchor=anchor,pre_intervention_native_state=state['anchor_state'],remaining_calls=sum(x['native_step']>anchor['after_completed_calls'] for x in target) if anchor and state['C']>anchor['canvas'] else None,
   target_terminal=terminal,committed_canvases=state['C'],uncommitted_tail=sum(not x['commit'] and x['canvas']==state['C'] for x in trace) if is_audit else None,
   runtime=runtime,work=work,correct=bool(correct),answer_parsed=detail['parsed'],no_final_response=not answer,quality_detail=detail,text_metrics=metrics(text),finish_reason=value.finish_reason,output_capped=T==row['generation_budget'],
   W=end-start,S=end-prefill,S_per_N=(end-prefill)/state['N'],new_captures=state['captures']-cap,new_jit=state['jit']-jc,timing_qualified=not is_audit and state['captures']==cap and state['jit']==jc,protocol_sha256=sha(root/'protocol.json'),selection_online=True)
  private=root/'private/generations'/a.phase/a.arm/(label+'.json');save(private,dict(completion=text,output_sha256=record['output_sha256']));record['private_generation_sha256']=sha(private)
  if is_audit:
   torch.save(state['private_reference'],private.with_suffix('.pt'));record['private_reference_sha256']=sha(private.with_suffix('.pt'))
  save(out/(label+'.json'),record)
  print(json.dumps({k:record[k] for k in ['id','seed','arm','label','N','C','T','correct','anchor','remaining_calls','timing_qualified']}),flush=True);return record
 request(cells[0][0],42,'warm',True)
 for i,(row,seed) in enumerate(cells):
  ar=request(row,seed,f'audit{i:03d}',True)
  if a.phase=='profile':continue
  cl=request(row,seed,f'clean{i:03d}',False);keys=['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards']
  if a.arm.endswith('_single_pulse'):keys.append('anchor')
  checks={k:ar[k]==cl[k] for k in keys};checks['selected_support']=ar['runtime']['selected_support']==cl['runtime']['selected_support'];checks['exposure']=ar['runtime']['pulse_calls']==cl['runtime']['pulse_calls']
  save(out/f'pair{i:03d}.json',dict(passed=all(checks.values()),checks=checks,audit_sha256=sha(out/f'audit{i:03d}.json'),clean_sha256=sha(out/f'clean{i:03d}.json'),timing_valid=cl['timing_qualified']))
  assert all(checks.values()),'audit/clean online policy qualification failed'
 for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
 save(out/'final_complete.json',dict(passed=True,cells=len(cells),model_executions=state['all_forwards'],startup_forwards=state['startup_forwards'],initialization_engine_forwards=initial,allocated_worker_seconds=time.time()-START,protocol_sha256=sha(root/'protocol.json')))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if LIVE:save(LIVE['out']/'failure.json',dict(passed=False,error_class=type(e).__name__,reason=str(e),model_executions=LIVE['state']['all_forwards'],startup_forwards=LIVE['state']['startup_forwards'],allocated_seconds=time.time()-START))
  raise
