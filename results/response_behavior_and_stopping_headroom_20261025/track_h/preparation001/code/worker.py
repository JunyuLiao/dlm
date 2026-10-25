"""Fresh full native requests at a later bottleneck; no restoration or hold."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from runtime import Runtime,save
from measure import summarize,compare_to_reference,digest

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
LIVE={}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--job',choices=['full','references','development','confirmation'],required=True)
    args=ap.parse_args();root=args.root.resolve();study=root.parents[1]
    assert (study/'track_r/attempt001/readout_frozen.json').exists(),'R readout must precede H'
    protocol=json.loads((root/'protocol.json').read_text());paths=json.loads((root/'private/runtime_paths.json').read_text())
    assert sha(study/'track_r/attempt001/readout_frozen.json')==protocol['R_readout_sha256']
    for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
    if args.job=='confirmation':assert json.loads((root/'development_decision.json').read_text())['confirmation_authorized']
    out=root/'executions'/args.job;out.mkdir(parents=True,exist_ok=False)
    raw=json.loads((root/'private/manifest.json').read_text());assert sha(root/'private/manifest.json')==protocol['private_manifest_sha256']
    all_cells=[(r,s) for split in ['development','confirmation'] for r in raw if r['split']==split for s in [42,43]]
    cells=[x for x in all_cells if args.job not in ['development','confirmation'] or x[0]['split']==args.job]
    if args.job in ['development','confirmation']:
        available=[];unavailable_reasons=[]
        for row,seed in cells:
            idx=next(i for i,(r,s) in enumerate(all_cells) if (r['id'],s)==(row['id'],seed))
            path=root/f'executions/references/cell{idx:03d}_A_observed/operators/selections.json'
            control=path.parent.parent/'receipt.json'
            bank=json.loads(path.read_text())['bank'] if path.exists() else None
            ended=json.loads(control.read_text())['remaining_calls'] is not None
            available.append(bool(bank and bank['qualified'] and ended))
            unavailable_reasons.append('no_qualifying_anchor' if bank is None else 'target_canvas_not_committed' if not ended else bank['reason'])
        if not any(available):
            for i,(row,seed) in enumerate(cells):save(out/f'cell{i:03d}_abstention.json',dict(id=row['id'],seed=seed,reason=unavailable_reasons[i]))
            save(out/'final_complete.json',dict(passed=True,job=args.job,item_seed_cells=len(cells),model_executions=0,startup_forwards=0,initialization_engine_forwards=0,allocated_worker_seconds=time.time()-STARTED,protocol_sha256=sha(root/'protocol.json'),no_model_initialized=True))
            return
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
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    from scripts import v31_official as official
    final,_=official.load_final_response()
    state=dict(N=0,C=0,all_forwards=0,startup_forwards=0,executing=False,bound=False,trace=[],consecutive=0,anchor_state=None,reference={},capture=True)
    rt=Runtime(state);rt.install()
    audit=NativeGeometryAudit(['full_attention' if i in [5,11,17,23,29] else 'sliding_attention' for i in range(30)],16)
    audit.bound=False;audit.install();LIVE.update(state=state,out=out)
    original_execute=GPUModelRunner.execute_model;original_model=dg.DiffusionGemmaForConditionalGeneration.forward
    original_sample=dg._compiled_sample_step
    def execute(runner,*a,**kw):
        state['all_forwards']+=1;state['executing']=True
        if state['all_forwards']%32==0:
            (root/'private/heartbeat.json').write_text(json.dumps(dict(job=args.job,unix=time.time(),forwards=state['all_forwards'],startup=state['startup_forwards'])))
        try:return original_execute(runner,*a,**kw)
        finally:state['executing']=False
    def model(m,*a,**kw):
        if not state['executing']:state['startup_forwards']+=1
        return original_model(m,*a,**kw)
    def sample(*a,**kw):
        if not state['bound']:return original_sample(*a,**kw)
        commit=bool(a[8][a[1]].item());result=original_sample(*a,**kw)
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        if commit:
            state['C']+=1;state['consecutive']=0
            state['trace'].append(dict(commit=True,canvas=state['C'],native_step=0));return result
        state['N']+=1;slot=a[1];history=a[13][slot];length=int(a[14][slot].item());step=int(a[7][slot].item())
        stable=bool((history==history[:,:1]).all());ready=length>=kw['ST']
        confidence=bool(a[9][slot].item());converged=bool(a[8][slot].item())
        qualifying=state['C']>=1 and ready and stable and not confidence and not converged
        state['consecutive']=state['consecutive']+1 if qualifying else 0
        row=dict(commit=False,canvas=state['C'],native_step=step,stable=stable,history_ready=ready,confidence=confidence,converged=converged,cap_reached=step>=48)
        trigger=rt.anchor is None and state['consecutive']>=2 and step<=46
        target_window=rt.anchor is not None and state['C']==rt.anchor['canvas'] and step>rt.anchor['after_completed_calls']
        if state['capture'] and (trigger or target_window):
            measured,fingerprint,private=summarize(result,a,kw);row.update(measured)
        elif trigger or target_window:
            fingerprint={name:digest(a[index][slot]) for name,index in [('canvas',5),('native_winner',6),('self_condition',10)]}
            fingerprint.update(logits=digest(result),history=digest(history[:,:min(length,history.shape[1])]),cpu_rng=digest(cpu),cuda_rng=digest(gpu),native_step=step,history_length=length,confidence=confidence,converged=converged)
            private=None
        else:fingerprint=None;private=None
        if fingerprint is not None:row['native_state_fingerprint']=fingerprint
        if trigger:
            rt.anchor=dict(canvas=state['C'],after_completed_calls=step)
            state['anchor_state']=dict(fingerprint=fingerprint,N=state['N'],C=state['C'])
            if state['expected_anchor'] is not None:assert state['anchor_state']==state['expected_anchor'],'native anchor state/RNG/counters mismatch'
        if rt.anchor is not None and state['C']==rt.anchor['canvas'] and step>rt.anchor['after_completed_calls']:
            if private is not None:
                state['private_reference'][step]=private
                row['reference_metrics_available']=step in state['reference']
                if step in state['reference']:row.update(compare_to_reference(result,state['reference'][step]))
        state['trace'].append(row)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state()),'observer RNG changed'
        return result
    GPUModelRunner.execute_model=execute;dg.DiffusionGemmaForConditionalGeneration.forward=model;dg._compiled_sample_step=sample
    mode='FULL_AND_PIECEWISE' if args.job=='full' else 'PIECEWISE'
    llm=LLM(model=paths['model_path'],dtype='bfloat16',max_model_len=262144,max_num_seqs=1,max_num_batched_tokens=16384,
        enable_chunked_prefill=True,gpu_memory_utilization=.80,enable_prefix_caching=False,trust_remote_code=False,seed=0,block_size=32,disable_log_stats=True,compilation_config={'cudagraph_mode':mode})
    engine=llm.llm_engine;tokenizer=llm.get_tokenizer();initial=state['all_forwards']
    def request(row,seed,label,common,capture,observe=False,selection=None,reference=None,expected_anchor=None,expected_route=None):
        folder=out/label;folder.mkdir(parents=True,exist_ok=False)
        state.update(N=0,C=0,bound=True,trace=[],consecutive=0,anchor_state=None,capture=capture,reference=reference or {},expected_anchor=expected_anchor,private_reference={})
        rt.begin(folder/'operators',common,observe,selection,expected_route)
        audit.begin_request();audit.bound=True
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.synchronize()
        start=time.perf_counter();before=state['all_forwards'];steps=0;prefill=None;output=None
        engine.add_request(label,TokensPrompt(prompt_token_ids=row['prompt_tokens']),SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
        while engine.has_unfinished_requests():
            values=engine.step();steps+=1
            if steps==math.ceil(len(row['prompt_tokens'])/16384):torch.cuda.synchronize();prefill=time.perf_counter()
            for value in values:
                if value.finished:output=value
        torch.cuda.synchronize();end=time.perf_counter();state['bound']=False;rt.active=False
        assert output is not None and steps==state['all_forwards']-before
        value=output.outputs[0];T=len(value.token_ids);assert math.ceil(T/256)==state['C']
        assert steps==math.ceil(len(row['prompt_tokens'])/16384)+state['N']+state['C']
        tokens=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        if hasattr(tokens,'keys'):tokens=tokens['input_ids']
        assert list(tokens)==row['prompt_tokens'],'prompt tokenization drift'
        text=tokenizer.decode(value.token_ids,skip_special_tokens=False);answer=final(text,True);parsed=official.lb_extract_answer(answer)
        runtime=rt.receipt();work=audit.receipt();skipped=sum(x['skipped'] for x in runtime['physical_calls'])
        work['global_kept_tiles']-=skipped;work['global_skipped_tiles']=skipped;work['overall_skipped_rectangles']=skipped
        work['overall_rectangle_sparsity']=skipped/work['overall_eligible_rectangles']
        legacy=work.pop('value_phase_tiles');work['native_dense_phase_geometry_baseline']=legacy
        target=runtime['physical_calls'];phases={}
        for phase in ['allkept','one_pulse']:
            calls=[x for x in target if x['phase']==phase]
            phases['layer29_'+phase]=dict(calls=len(calls),eligible_tiles=sum(x['eligible'] for x in calls),kept_tiles=sum(x['kept'] for x in calls),skipped_tiles=sum(x['skipped'] for x in calls))
        other=work['global_eligible_tiles']-sum(x['eligible'] for x in target)
        phases['native_GLOBAL_dense']=dict(calls=work['global_calls']-len(target),eligible_tiles=other,kept_tiles=other,skipped_tiles=0)
        assert sum(x['kept_tiles'] for x in phases.values())==work['global_kept_tiles']
        work['value_phase_tiles']=phases
        if selection is not None:assert rt.pulse_calls==1,'missing or repeated pulse'
        else:assert skipped==0
        anchor=rt.anchor
        remaining=sum(not x['commit'] and anchor is not None and x['canvas']==anchor['canvas'] and x['native_step']>anchor['after_completed_calls'] for x in state['trace'])
        target_terminated=anchor is not None and state['C']>anchor['canvas']
        record=dict(id=row['id'],seed=seed,split=row['split'],label=label,N=state['N'],C=state['C'],T=T,N_per_C=state['N']/state['C'],
            model_forwards=steps,output_sha256=hashlib.sha256(json.dumps(list(value.token_ids)).encode()).hexdigest(),final_cpu_rng=digest(torch.get_rng_state()),final_cuda_rng=digest(torch.cuda.get_rng_state()),
            native_window_fingerprints=[dict(canvas=x['canvas'],native_step=x['native_step'],fingerprint=x['native_state_fingerprint']) for x in state['trace'] if 'native_state_fingerprint' in x],
            anchor=anchor,pre_intervention_native_state=state['anchor_state'],remaining_calls=remaining if target_terminated else None,target_canvas_terminated=target_terminated,execution_mode=mode,native_trace=state['trace'],runtime=runtime,work=work,
            correct=parsed==row['expected'],answer_parsed=parsed is not None,no_final_response=not answer,finish_reason=value.finish_reason,
            W=end-start,S=end-prefill,S_per_N=(end-prefill)/state['N'],timing_qualified=False,protocol_sha256=sha(root/'protocol.json'))
        private=root/'private/generations'/args.job/label;private.parent.mkdir(parents=True,exist_ok=True)
        save(private.with_suffix('.json'),dict(completion=text))
        privatefolder=folder/'private';privatefolder.mkdir(exist_ok=True);torch.save(state['private_reference'],privatefolder/'native_reference.pt')
        record['private_native_reference_sha256']=sha(privatefolder/'native_reference.pt')
        save(folder/'receipt.json',record)
        print(json.dumps({k:record[k] for k in ['label','id','seed','N','C','T','anchor','remaining_calls','correct']}),flush=True)
        return record
    first=next(x for x in raw if x['split']=='development');request(first,42,'warm',args.job!='full',False)
    for index,(row,seed) in enumerate(cells):
        if args.job=='full':request(row,seed,f'cell{index:03d}_full',False,True);continue
        if args.job=='references':
            request(row,seed,f'cell{index:03d}_piecewise',False,True)
            plain=request(row,seed,f'cell{index:03d}_A_unobserved',True,False)
            observed=request(row,seed,f'cell{index:03d}_A_observed',True,True,True)
            repeated=request(row,seed,f'cell{index:03d}_A_repeat',True,True)
            checks={k:plain[k]==observed[k]==repeated[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','model_forwards','anchor','pre_intervention_native_state','native_window_fingerprints']}
            checks['current_QKV_routes']=plain['runtime']['current_operand_routes']==observed['runtime']['current_operand_routes']==repeated['runtime']['current_operand_routes']
            save(out/f'cell{index:03d}_qualification.json',dict(passed=all(checks.values()),id=row['id'],seed=seed,checks=checks,anchor=observed['anchor']))
            assert all(checks.values()),'later-canvas fresh observer equality failed'
        else:
            reference_index=next(i for i,(r,s) in enumerate(all_cells) if (r['id'],s)==(row['id'],seed))
            folder=root/f'executions/references/cell{reference_index:03d}_A_observed'
            control=json.loads((folder/'receipt.json').read_text());path=folder/'operators/selections.json'
            bank=json.loads(path.read_text()) if path.exists() else None
            if bank is None or not bank['bank']['qualified'] or control['remaining_calls'] is None:
                save(out/f'cell{index:03d}_abstention.json',dict(id=row['id'],seed=seed,reason='no_qualifying_anchor' if bank is None else 'target_canvas_not_committed' if control['remaining_calls'] is None else bank['bank']['reason']))
                continue
            assert sha(folder/'private/native_reference.pt')==control['private_native_reference_sha256']
            reference=torch.load(folder/'private/native_reference.pt',weights_only=True);seen={}
            for name in ['current_error','support_sham','direction_high','direction_low']:
                selection=bank['bank']['selections'][name];support_hash=selection['support_sha256']
                if support_hash in seen:
                    save(out/f'cell{index:03d}_{name}_alias.json',dict(id=row['id'],seed=seed,alias=seen[support_hash],support_sha256=support_hash));continue
                request(row,seed,f'cell{index:03d}_{name}',True,True,selection=selection,reference=reference,
                    expected_anchor=control['pre_intervention_native_state'],expected_route=bank['pre_intervention_route'])
                seen[support_hash]=f'cell{index:03d}_{name}'
            after=request(row,seed,f'cell{index:03d}_A_after',True,True,expected_anchor=control['pre_intervention_native_state'],expected_route=bank['pre_intervention_route'])
            assert all(after[k]==control[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','pre_intervention_native_state','native_window_fingerprints']),'post-pulse all-kept contamination'
    for p,h in protocol['sources'].items():assert sha(REPO/p)==h
    save(out/'final_complete.json',dict(passed=True,job=args.job,item_seed_cells=len(cells),model_executions=state['all_forwards'],startup_forwards=state['startup_forwards'],
        initialization_engine_forwards=initial,allocated_worker_seconds=time.time()-STARTED,protocol_sha256=sha(root/'protocol.json'),one_pulse=True,native_sampler_unchanged=True))


if __name__=='__main__':
    STARTED=time.time()
    try:main()
    except BaseException as error:
        if LIVE:save(LIVE['out']/'failure.json',dict(error_class=type(error).__name__,reason=str(error),model_executions=LIVE['state']['all_forwards'],startup_forwards=LIVE['state']['startup_forwards'],allocated_seconds=time.time()-STARTED))
        raise
