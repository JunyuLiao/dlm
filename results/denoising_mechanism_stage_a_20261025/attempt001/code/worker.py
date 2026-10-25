"""Ordinary fresh native requests; no restore, hidden forward or stop override."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from runtime import Runtime,digest,save
from operators import anchors

REPO=Path('/home/exouser/ljy/dlm')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--job',choices=['smoke','full','a1'],required=True);args=ap.parse_args()
    root=args.root.resolve();protocol=json.loads((root/'protocol.json').read_text())
    paths=json.loads((root/'private/runtime_paths.json').read_text())
    for p,h in protocol['sources'].items():assert sha(REPO/p)==h,'source drift '+p
    for p,h in json.loads((root/'private/external_source_paths.json').read_text()).items():assert sha(p)==h,'runtime drift'
    out=root/'executions'/args.job;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'private/manifest.json').read_text());assert sha(root/'private/manifest.json')==protocol['private_manifest_sha256']
    if args.job=='smoke':manifest=[next(r for r in manifest if r['id']=='humaneval/HumanEval/124')]
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994,apply_fa4_local_fix
    import torch._inductor.config as cfg
    cfg.deterministic=True;cfg.benchmark_combo_kernel=False;cfg.combo_kernels=False
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    policy=install_compilation_policy();apply_fix_51994();apply_fa4_local_fix(build_dir=root/'runtime/fa4_local_fix')
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models import diffusion_gemma as dg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.utils import jit_monitor
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    from scripts import v31_official as official
    from scripts import v27_humaneval as he
    final,_=official.load_final_response();numeric,_=official.aime_numeric_score()
    gold_path=Path(paths['gold_path'])
    gold={r['task_id']:r for r in json.loads(gold_path.read_text()).values()};he.sandbox_preflight()
    audit=NativeGeometryAudit(['full_attention' if i in [5,11,17,23,29] else 'sliding_attention' for i in range(30)],16);audit.bound=False;audit.install()
    rt=Runtime();rt.install()
    state=dict(all_forwards=0,startup_forwards=0,executing=False,bound=False,N=0,C=0,trace=[],fingerprints=[],captures=0,jit=0,canvas=0,measure=True)
    original_execute=GPUModelRunner.execute_model;original_model=dg.DiffusionGemmaForConditionalGeneration.forward
    original_sample=dg._compiled_sample_step;original_capture=torch.cuda.CUDAGraph.capture_begin;original_jit=jit_monitor._handle_jit_event
    def execute(runner,*a,**kw):
        state['all_forwards']+=1;state['executing']=True
        if state['all_forwards']%32==0:
            (root/'private'/('heartbeat_'+args.job+'.json')).write_text(json.dumps(dict(unix=time.time(),forwards=state['all_forwards'])))
        try:return original_execute(runner,*a,**kw)
        finally:state['executing']=False
    def model(m,*a,**kw):
        if not state['executing']:state['startup_forwards']+=1
        return original_model(m,*a,**kw)
    def capture(g,*a,**kw):state['captures']+=1;return original_capture(g,*a,**kw)
    def jit(**kw):
        if state['bound']:state['jit']+=1
        return original_jit(**kw)
    def sample(*a,**kw):
        if not state['bound']:return original_sample(*a,**kw)
        committing=bool(a[8][a[1]].item());result=original_sample(*a,**kw)
        cpu_rng=torch.get_rng_state().clone();rng=torch.cuda.get_rng_state().clone()
        if committing:state['C']+=1;rt.first_canvas=False;state['canvas']+=1
        else:state['N']+=1
        row=dict(commit=committing,canvas=state['canvas'],native_step=int(a[7][a[1]].item()))
        if state['measure'] and (state['canvas']==0 or (committing and state['canvas']==1)):
            slot=a[1];history=a[13][slot];ready=int(a[14][slot].item())>=kw['ST']
            if not committing:
                assert torch.equal(result.argmax(-1).reshape_as(a[6][slot]),a[6][slot]),'native winner mismatch'
                ent=[];marg=[]
                for chunk in result.reshape(-1,result.shape[-1]).split(16):
                    lp=chunk.float().log_softmax(-1);ent.append((-(lp.exp()*lp).sum(-1)).cpu())
                    top=chunk.float().topk(2,dim=-1).values;marg.append((top[:,0]-top[:,1]).cpu())
                entropy=torch.cat(ent);margin=torch.cat(marg)
                row.update(entropy_mean=float(entropy.mean()),entropy_by_position=entropy.tolist(),margin_by_position=margin.tolist(),
                    stable=bool((history==history[:,:1]).all()),history_ready=ready,confidence=bool(a[9][slot].item()),converged=bool(a[8][slot].item()),
                    cap=int(a[7][slot].item())>=48)
            snap={name:digest(a[index][slot]) for name,index in [('canvas',5),('winner',6),('self_condition',10)]}
            active_history=history[:,:min(int(a[14][slot].item()),history.shape[1])]
            snap.update(logits=digest(result),history=digest(active_history),cpu_rng=digest(cpu_rng),cuda_rng=digest(rng),
                        native_step=row['native_step'],confidence=bool(a[9][slot].item()),converged=bool(a[8][slot].item()))
            state['fingerprints'].append(snap)
        state['trace'].append(row)
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(rng,torch.cuda.get_rng_state()),'observer RNG changed'
        return result
    GPUModelRunner.execute_model=execute;dg.DiffusionGemmaForConditionalGeneration.forward=model
    dg._compiled_sample_step=sample;torch.cuda.CUDAGraph.capture_begin=capture;jit_monitor._handle_jit_event=jit
    mode='FULL_AND_PIECEWISE' if args.job=='full' else 'PIECEWISE'
    llm=LLM(model=paths['model_path'],dtype='bfloat16',max_model_len=262144,max_num_seqs=1,max_num_batched_tokens=16384,
        enable_chunked_prefill=True,gpu_memory_utilization=.80,enable_prefix_caching=False,trust_remote_code=False,
        seed=0,block_size=32,disable_log_stats=True,compilation_config={'cudagraph_mode':mode})
    engine=llm.llm_engine;tokenizer=llm.get_tokenizer();initial=state['all_forwards']
    def request(row,seed,label,common,observe,landmarks=(),measure=True):
        folder=out/label;folder.mkdir(parents=True,exist_ok=False)
        rt.begin(folder/'operators',list(landmarks),observe,common)
        state.update(bound=True,N=0,C=0,canvas=0,trace=[],fingerprints=[],jit=0,measure=measure)
        audit.begin_request();audit.bound=True
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.synchronize()
        start=time.perf_counter();before=state['all_forwards'];captures=state['captures'];steps=0;prefill=None;output=None
        engine.add_request(label,TokensPrompt(prompt_token_ids=row['prompt_tokens']),SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
        while engine.has_unfinished_requests():
            values=engine.step();steps+=1
            if steps==math.ceil(len(row['prompt_tokens'])/16384):torch.cuda.synchronize();prefill=time.perf_counter()
            for x in values:
                if x.finished:output=x
        torch.cuda.synchronize();end=time.perf_counter();state['bound']=False;rt.active=False
        assert output is not None and state['all_forwards']-before==steps,'extra model forward/missing output'
        v=output.outputs[0];T=len(v.token_ids);C=math.ceil(T/256)
        assert C==state['C'] and steps==math.ceil(len(row['prompt_tokens'])/16384)+state['N']+C
        exact=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=row['thinking'])
        if hasattr(exact,'keys'):exact=exact['input_ids']
        assert list(exact)==row['prompt_tokens'],'tokenization changed'
        text=tokenizer.decode(v.token_ids,skip_special_tokens=False);response=final(text,row['thinking']);detail={}
        if row['benchmark']=='aime26':
            quality=numeric(response,row['expected']);correct=quality['correct'];detail['parsed']=quality['extracted'] is not None
        elif row['benchmark']=='humaneval':
            problem=gold[row['source_id']];code,reason=he.extract_code(text,problem['entry_point'])
            correct,outcome=he.run_test(he.program_for(problem,code)) if code is not None else (False,'not_run')
            detail.update(extraction=reason,outcome=outcome)
        else:
            answer=official.lb_extract_answer(response);correct=answer==row['expected'];detail['parsed']=answer is not None
        work=audit.receipt();runtime=rt.receipt()
        # This A1 stage executes only all-kept consumers; future A2 needs its
        # own source binding and per-call work subtraction before launch.
        assert all(x['skipped']==0 for x in runtime['physical_calls'])
        first=sum(not x['commit'] and x['canvas']==0 for x in state['trace'])
        result=dict(id=row['id'],seed=seed,split=row['split'],benchmark=row['benchmark'],label=label,common_consumer=common,
            operator_observer=observe,N=state['N'],C=C,T=T,N_per_C=state['N']/C,first_canvas_calls=first,
            model_forwards=steps,output_sha256=hashlib.sha256(json.dumps(list(v.token_ids)).encode()).hexdigest(),
            final_cpu_rng=digest(torch.get_rng_state()),final_cuda_rng=digest(torch.cuda.get_rng_state()),
            native_trace=state['trace'],first_canvas_fingerprints=state['fingerprints'],runtime=runtime,work=work,
            correct=bool(correct),no_final_response=not response,quality_detail=detail,finish_reason=v.finish_reason,
            W=end-start,S=end-prefill,S_per_N=(end-prefill)/state['N'],timing_qualified=False,
            captures=state['captures']-captures,jit=state['jit'],protocol_sha256=sha(root/'protocol.json'))
        save(folder/'receipt.json',result)
        private=root/'private/generations'/args.job/label;private.parent.mkdir(parents=True,exist_ok=True)
        save(private.with_suffix('.json'),dict(completion=text))
        print(json.dumps(dict(label=label,id=row['id'],seed=seed,N=result['N'],C=C,T=T,first_canvas=first,correct=bool(correct))),flush=True)
        return result
    cells=[(row,seed) for split in ['development','confirmation'] for row in manifest if row['split']==split for seed in row['seeds']]
    if args.job=='smoke':cells=cells[:1]
    for index,(row,seed) in enumerate(cells):
        if index==0:request(row,seed,'warm',args.job!='full',False,measure=False)
        if args.job=='full':request(row,seed,f'cell{index:03d}_native_full',False,False)
        else:
            # Prospective dense profile is recorded for every frozen identity;
            # no identity is replaced using its measured difficulty.
            request(row,seed,f'cell{index:03d}_piecewise',False,False)
            plain=request(row,seed,f'cell{index:03d}_allkept_unobserved',True,False,measure=False)
            reference=request(row,seed,f'cell{index:03d}_allkept_reference',True,False)
            targets=anchors(reference['first_canvas_calls'])
            observed=request(row,seed,f'cell{index:03d}_allkept_observed',True,True,targets)
            fields=['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards']
            checks={k:plain[k]==reference[k]==observed[k] for k in fields}
            checks['first_canvas_sampler_state_logits_rng']=reference['first_canvas_fingerprints']==observed['first_canvas_fingerprints']
            checks['current_QKV_routes']=plain['runtime']['route_fingerprints']==reference['runtime']['route_fingerprints']==observed['runtime']['route_fingerprints']
            save(out/f'cell{index:03d}_qualification.json',dict(passed=all(checks.values()),checks=checks,anchors=targets,id=row['id'],seed=seed))
            assert all(checks.values()),'fresh request observer engineering parity'
            del plain,reference,observed
    for p,h in protocol['sources'].items():assert sha(REPO/p)==h,'source drift at end '+p
    save(out/'final_complete.json',dict(passed=True,cells=len(cells),job=args.job,model_executions=state['all_forwards'],
        startup_forwards=state['startup_forwards'],initialization_engine_forwards=initial,protocol_sha256=sha(root/'protocol.json'),
        native_sampler_unchanged=True,A2_executed=False,allocation_seconds=time.time()-STARTED))


if __name__=='__main__':
    STARTED=time.time()
    try:main()
    except BaseException as error:
        ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--job');args=ap.parse_args()
        save(args.root/'executions'/args.job/'failure.json',dict(error_class=type(error).__name__,reason=str(error),allocated_seconds=time.time()-STARTED))
        raise
