"""Shared fresh-request harness; no restoration or hidden/suffix forward."""
from pathlib import Path
import hashlib,json,math,os,sys,time
import torch
from shared_consumer import CommonConsumer,fingerprint
from observations import observe

REPO=Path(__file__).resolve().parents[5];STUDY=REPO/'results/native_support_discovery_20261026'
sys.path[:0]=[str(REPO),str(REPO/'scripts')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    assert p.resolve().is_relative_to(STUDY.resolve()),'Output escaped new study'
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)

def anchors(trace,prompt_tokens):
    found={}
    for row in trace:
        if row['commit']:continue
        t=row['native_step'];prefix=prompt_tokens+row['canvas']*256
        eligible=row['history_ready'] and not row['confidence'] and not row['converged'] and t<=46 and prefix//64>=4
        name='confidence' if row['stable'] else 'instability'
        if eligible and name not in found and (name=='confidence' or t>=3):found[name]=dict(canvas=row['canvas'],after_completed_calls=t,prefix=prefix)
    for name,a in found.items():
        cs=[x for x in trace if not x['commit'] and x['canvas']==a['canvas']]
        terminal=cs[-1];a.update(reference_remaining=sum(x['native_step']>a['after_completed_calls'] for x in cs),optimistic_U=max(0,terminal['native_step']-a['after_completed_calls']-1),reference_terminal_step=terminal['native_step'],reference_cap=terminal['cap_reached'])
    return found

class NativeHarness:
    def __init__(self,root,arm,extension=None):
        self.root=Path(root).resolve();assert self.root.is_relative_to(STUDY.resolve());self.arm=arm;self.extension=extension;self.started=time.time()
        self.paths=json.loads((STUDY/'private/runtime_paths.json').read_text());self.protocol=json.loads((self.root/'protocol.json').read_text())
        self.verify_sources()
        from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
        from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994,apply_fa4_local_fix
        import torch._inductor.config as cfg
        cfg.deterministic=True;cfg.benchmark_combo_kernel=False;cfg.combo_kernels=False
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        install_compilation_policy();apply_fix_51994();apply_fa4_local_fix(build_dir=self.root/'runtime/fa4_local_fix')
        from vllm import LLM
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.utils import jit_monitor
        from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
        self.state=dict(bound=False,audit=False,N=0,C=0,step=0,executions=0,startup=0,python_model_calls=0,executing=False,current_execute_Python_calls=0,extra_inside_execute=0,captures=0,jit=0,trace=[],private_calls={},prior=None)
        state=self.state;self.consumer=CommonConsumer(state);self.consumer.install()
        self.geometry=NativeGeometryAudit(['full_attention' if i in [5,11,17,23,29] else 'sliding_attention' for i in range(30)],16);self.geometry.bound=False;self.geometry.install()
        execute0=GPUModelRunner.execute_model;model0=dg.DiffusionGemmaForConditionalGeneration.forward;sample0=dg._compiled_sample_step;capture0=torch.cuda.CUDAGraph.capture_begin;jit0=jit_monitor._handle_jit_event
        def execute(runner,*args,**kw):
            state['executions']+=1;state['executing']=True;state['current_execute_Python_calls']=0
            if state['executions']%64==0:
                (self.root/'private/heartbeat.json').parent.mkdir(parents=True,exist_ok=True)
                (self.root/'private/heartbeat.json').write_text(json.dumps(dict(unix=time.time(),executions=state['executions'],startup=state['startup'],extra_inside_execute=state['extra_inside_execute'],label=state.get('label'))))
            try:return execute0(runner,*args,**kw)
            finally:
                state['extra_inside_execute']+=max(0,state['current_execute_Python_calls']-1);state['executing']=False
        def model(m,*args,**kw):
            state['python_model_calls']+=1;self.consumer.model=m
            if state['executing']:state['current_execute_Python_calls']+=1
            else:state['startup']+=1
            if self.extension is not None and state['bound']:self.extension.model_entry(m,args,kw)
            return model0(m,*args,**kw)
        def sample(*args,**kw):
            if not state['bound']:return sample0(*args,**kw)
            committing=bool(args[8][args[1]].item());result=sample0(*args,**kw)
            if committing:
                row=dict(commit=True,canvas=state['C'],native_step=0,emitted_canvas_sha256=fingerprint(args[15][args[2]].reshape(-1))) if state['audit'] else None
                if self.extension is not None:self.extension.after_sample(result,args,kw,True,row,None)
                if state['audit']:state['trace'].append(row)
                state['C']+=1;state['step']=0;return result
            state['N']+=1;state['step']=int(args[7][args[1]].item());row=None;private=None
            need_prior=self.extension is not None and getattr(self.extension,'needs_prior',False)
            if state['audit'] or need_prior:
                cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();ref=None;positions=False
                if self.extension is not None:ref,positions=self.extension.reference_for(state['C'],state['step'])
                row,private=observe(result,args,kw,reference=ref,positions=positions,minimal=not state['audit'])
                state['prior']=private
                if state['audit']:
                    row.update(commit=False,canvas=state['C']);state['trace'].append(row);state['private_calls'][(state['C'],state['step'])]=private
                if self.extension is not None:self.extension.after_sample(result,args,kw,False,row,private)
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'observer/operator RNG consumption'
            elif self.extension is not None:self.extension.after_sample(result,args,kw,False,None,None)
            return result
        def capture(graph,*args,**kw):state['captures']+=1;return capture0(graph,*args,**kw)
        def jit(**kw):
            if state['bound']:state['jit']+=1
            return jit0(**kw)
        GPUModelRunner.execute_model=execute;dg.DiffusionGemmaForConditionalGeneration.forward=model;dg._compiled_sample_step=sample;torch.cuda.CUDAGraph.capture_begin=capture;jit_monitor._handle_jit_event=jit
        mode='FULL_AND_PIECEWISE' if arm=='corrected_dense_full' else 'PIECEWISE'
        self.llm=LLM(model=self.paths['model_path'],dtype='bfloat16',max_model_len=262144,max_num_seqs=1,max_num_batched_tokens=16384,enable_chunked_prefill=True,gpu_memory_utilization=.80,enable_prefix_caching=False,trust_remote_code=False,seed=0,block_size=32,disable_log_stats=True,compilation_config={'cudagraph_mode':mode})
        self.engine=self.llm.llm_engine;self.tokenizer=self.llm.get_tokenizer();self.initial_executions=state['executions']
        from scripts import v31_official as official,v27_humaneval as he
        self.official=official;self.he=he;self.final,_=official.load_final_response();self.numeric,_=official.aime_numeric_score();self.gold={x['task_id']:x for x in json.loads(Path(self.paths['gold_path']).read_text()).values()};he.sandbox_preflight()
    def verify_sources(self):
        for p,h in self.protocol['sources'].items():assert sha(REPO/p)==h,p
        for p,h in self.protocol['scorer_sources'].items():assert sha(REPO/p)==h,p
        package=Path(self.paths['python']).parents[1]/'lib/python3.12/site-packages/vllm'
        for p,h in self.protocol['runtime_source_pins'].items():assert sha(package/p)==h,p
    def request(self,row,seed,label,audit=True,GQA=False):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        state=self.state;state.update(bound=True,audit=audit,N=0,C=0,step=0,trace=[],private_calls={},prior=None,label=label)
        self.consumer.begin(common=self.arm not in ['corrected_dense_full','dense_piecewise']);self.geometry.begin_request();self.geometry.bound=audit
        if GQA and self.consumer.common:self.consumer.gqa_pending=set([5,17,29])
        if self.extension is not None:self.extension.begin(self,row,seed,label,audit)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.synchronize()
        initial_cpu=fingerprint(torch.get_rng_state());initial_cuda=fingerprint(torch.cuda.get_rng_state())
        before=state['executions'];startup=state['startup'];extra=state['extra_inside_execute'];captures=state['captures'];jit=state['jit'];start=time.perf_counter();prefill=None;steps=0;output=None
        self.engine.add_request(label,TokensPrompt(prompt_token_ids=row['prompt_tokens']),SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
        while self.engine.has_unfinished_requests():
            values=self.engine.step();steps+=1
            if steps==math.ceil(len(row['prompt_tokens'])/16384):torch.cuda.synchronize();prefill=time.perf_counter()
            for value in values:
                if value.finished:output=value
        torch.cuda.synchronize();end=time.perf_counter();state['bound']=False;self.geometry.bound=False
        assert output is not None and steps==state['executions']-before
        value=output.outputs[0];T=len(value.token_ids)
        assert math.ceil(T/256)==state['C']
        assert steps==math.ceil(len(row['prompt_tokens'])/16384)+state['N']+state['C']
        cpu=fingerprint(torch.get_rng_state());gpu=fingerprint(torch.cuda.get_rng_state())
        tokens=self.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        if hasattr(tokens,'keys'):tokens=tokens['input_ids']
        assert list(tokens)==row['prompt_tokens'],'native prompt tokenization changed'
        text=self.tokenizer.decode(value.token_ids,skip_special_tokens=False);answer=self.final(text,True);detail={};task=row['benchmark']
        if task=='aime26':v=self.numeric(answer,row['expected']);correct=v['correct'];detail['parsed']=v['extracted'] is not None
        elif task=='longbench_v2':parsed=self.official.lb_extract_answer(answer);correct=parsed==row['expected'];detail['parsed']=parsed is not None
        else:
            problem=self.gold[row['source_id']];code,reason=self.he.extract_code(text,problem['entry_point']);correct,outcome=self.he.run_test(self.he.program_for(problem,code)) if code is not None else (False,'not_run');detail.update(parsed=code is not None,extraction=reason,outcome=outcome)
        work=self.geometry.receipt() if audit else None
        if work:
            skipped=sum(x['skipped'] for x in self.consumer.records);work['global_kept_tiles']-=skipped;work['global_skipped_tiles']=skipped;work['overall_skipped_rectangles']=skipped;work['overall_rectangle_sparsity']=skipped/work['overall_eligible_rectangles'];work['native_dense_phase_geometry_baseline']=work.pop('value_phase_tiles')
        record=dict(id=row['id'],seed=seed,benchmark=task,arm=self.arm,label=label,audit=audit,partition=row.get('partition'),stratum=row.get('stratum',row.get('roster_stratum')),
            N=state['N'],C=state['C'],T=T,N_per_C=state['N']/state['C'],model_forwards=steps,startup_model_calls_in_request=state['startup']-startup,extra_inside_execute_Python_forwards=state['extra_inside_execute']-extra,
            output_sha256=hashlib.sha256(json.dumps(list(value.token_ids)).encode()).hexdigest(),initial_cpu_rng=initial_cpu,initial_cuda_rng=initial_cuda,final_cpu_rng=cpu,final_cuda_rng=gpu,
            correct=bool(correct),answer_parsed=detail['parsed'],no_final_response=not answer,quality_detail=detail,output_capped=T==row['generation_budget'],finish_reason=value.finish_reason,
            native_trace=state['trace'],anchors=anchors(state['trace'],len(row['prompt_tokens'])) if audit else None,committed_canvases=state['C'],uncommitted_tail=sum(not x['commit'] and x['canvas']==state['C'] for x in state['trace']) if audit else None,
            current_QKV=True,LOCAL_dense=True,consumer_physical=self.consumer.records,GQA_qualification=self.consumer.gqa,diagnostic_work=self.consumer.extra_work,work=work,
            W=end-start,S=end-prefill,S_per_N=(end-prefill)/state['N'],new_captures=state['captures']-captures,new_jit=state['jit']-jit,clean_timing_qualified=not audit and state['captures']==captures and state['jit']==jit,
            protocol_sha256=sha(self.root/'protocol.json'),prompt_tokenization_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),prompt_hash=row['prompt_hash'])
        if self.extension is not None:record.update(self.extension.finish(record))
        private=self.root/'private/generations'/label
        save(private.with_suffix('.json'),dict(completion=text,output_sha256=record['output_sha256']))
        record['private_generation_sha256']=sha(private.with_suffix('.json'))
        if audit:
            private.parent.mkdir(parents=True,exist_ok=True)
            with private.with_suffix('.pt').open('xb') as f:torch.save(state['private_calls'],f)
            record['private_reference_sha256']=sha(private.with_suffix('.pt'))
        self.last_record=record
        print(json.dumps({k:record[k] for k in ['id','seed','arm','label','N','C','T','correct']}),flush=True)
        return record
    def totals(self):
        return dict(engine_execute_forwards=self.state['executions'],startup_model_calls_outside_execute=self.state['startup'],extra_model_calls_inside_execute=self.state['extra_inside_execute'],python_model_forward_invocations=self.state['python_model_calls'],initialization_engine_forwards=self.initial_executions,worker_body_seconds=time.time()-self.started)
