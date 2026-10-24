"""Bounded first-canvas AA/SA/AS/SS diagnostic; never a production selector."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    for p,h in protocol['sources'].items():
        if sha(p)!=h:raise RuntimeError('source drift '+p)
    if sha(root/'private/tokenized_manifest.json')!=protocol['private_manifest_sha256']:raise RuntimeError('manifest drift')
    out=root/'execution001';out.mkdir(exist_ok=False)
    resource=dict(started_unix=time.time(),forwards=0,runtime_forwards=0,startup_forwards=0,engine_steps=0,captures=0,jit=0)
    deadline=float(os.environ['PULSEHOLD_DEADLINE_UNIX'])
    prior_forwards=int(os.environ.get('PULSEHOLD_PRIOR_FORWARDS','0'))
    def tick(startup=False):
        if time.time()>=deadline-30 or resource['forwards']+prior_forwards>=protocol['resource']['forwards']:
            raise RuntimeError('RESOURCE_CEILING')
        resource['forwards']+=1
        resource['startup_forwards' if startup else 'runtime_forwards']+=1
        (root/'private/live_resource.json').write_text(json.dumps(resource))
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994,apply_fa4_local_fix
    import torch._inductor.config as cfg
    cfg.deterministic=True;cfg.benchmark_combo_kernel=False;cfg.combo_kernels=False
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    policy=install_compilation_policy();apply_fix_51994();fix=apply_fa4_local_fix(build_dir=root/'runtime/fa4_local_fix')
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models import diffusion_gemma as dg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from production_observer import ProductionObserver,digest,rng_state,assert_rng_equal
    from state_restore import Snapshot
    from runtime import Runtime
    from pulsehold import first_canvas_anchors,select_finalists
    from native_measurements import capture_native_sample,align_branch,decision_score
    from support_bank import candidate_bank,matched_pairs,apply_guards
    observer=ProductionObserver();rt=Runtime();rt.install();runner_ref={};executing={'yes':False};measure={'on':False,'keep_logits':False}
    events=[];trace=[];fingerprints=[];winner_history=[]
    original_execute=GPUModelRunner.execute_model
    def execute(runner,*a,**kw):
        runner_ref['runner']=runner;tick();executing['yes']=True
        try:return original_execute(runner,*a,**kw)
        finally:executing['yes']=False
    GPUModelRunner.execute_model=execute
    model_forward=dg.DiffusionGemmaForConditionalGeneration.forward
    def counted_model(model,*a,**kw):
        if not executing['yes']:tick(startup=True)
        return model_forward(model,*a,**kw)
    dg.DiffusionGemmaForConditionalGeneration.forward=counted_model
    capture0=torch.cuda.CUDAGraph.capture_begin
    def capture(graph,*a,**kw):resource['captures']+=1;return capture0(graph,*a,**kw)
    torch.cuda.CUDAGraph.capture_begin=capture
    from vllm.utils import jit_monitor
    jit0=jit_monitor._handle_jit_event
    def jit(**kw):resource['jit']+=1;return jit0(**kw)
    jit_monitor._handle_jit_event=jit
    sample0=dg._compiled_sample_step
    def sample(*a,**kw):
        result=observer.sample(sample0,*a,**kw)
        rt.canvas=observer.commit_calls
        if observer.mode!='off':
            fingerprints.append(dict(scaled_sha256=digest(result),state=observer.fingerprints[-1]))
        if measure['on']:
            rng=rng_state()
            _,row=capture_native_sample(lambda *unused,**ignored:result,*a,**kw)
            if not row['is_commit']:
                if not row['winner_measurement']['native_winner_equal']:raise RuntimeError('native winner mismatch')
                logits=row['scaled_logits']
                lp=logits.float().log_softmax(-1);ent=-(lp.exp()*lp).sum(-1)
                row['entropy_mean']=float(ent.mean());row['entropy_slack']=float(ent.mean())-kw['confidence_threshold']
                winner_history.append(row['native_winners'].detach().cpu().clone())
                public=dict(native_step=row['native_step'],confidence=row['native_confidence'],converged=row['native_commit_pending'],
                    history_len=row['native_history_len'],entropy_mean=row['entropy_mean'],entropy_slack=row['entropy_slack'],
                    winner_measurement=row['winner_measurement'],acceptance_measurement=row['acceptance_measurement'],state_hashes=row['state_hashes'])
                trace.append(public)
                if measure['keep_logits'] and len(events)<2:
                    row['scaled_logits']=logits.detach().cpu().clone()
                    for name in ['native_winners','native_history','sampled_canvas','canvas_after','self_condition_after']:
                        row[name]=row[name].detach().cpu().clone()
                    events.append(row)
            assert_rng_equal(rng)
        return result
    dg._compiled_sample_step=sample
    rows=json.loads((root/'private/tokenized_manifest.json').read_text());byid={x['id']:x for x in rows}
    llm=LLM(model=protocol['model_path'],dtype='bfloat16',max_model_len=12288,max_num_seqs=1,max_num_batched_tokens=16384,
        enable_chunked_prefill=True,gpu_memory_utilization=.85,enable_prefix_caching=False,trust_remote_code=False,
        seed=0,block_size=32,disable_log_stats=True,compilation_config={'cudagraph_mode':'PIECEWISE'})
    engine=llm.llm_engine;core=engine.engine_core.engine_core;runner=runner_ref['runner'];tokenizer=llm.get_tokenizer()
    def step():
        resource['engine_steps']+=1
        outputs=engine.step();torch.cuda.synchronize();return outputs
    def add(row,label,mode='minimal'):
        rt.reset();rt.enabled=True;observer.begin(mode);events.clear();trace.clear();fingerprints.clear();winner_history.clear()
        torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
        engine.add_request(label,TokensPrompt(prompt_token_ids=row['prompt_tokens']),SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
    def until_commit():
        f0=resource['runtime_forwards'];s0=resource['engine_steps'];commits=observer.commit_calls;n0=observer.denoise_calls
        while engine.has_unfinished_requests() and observer.commit_calls==commits:step()
        return dict(forwards=resource['runtime_forwards']-f0,steps=resource['engine_steps']-s0,committed=observer.commit_calls>commits,
            remaining_denoise=observer.denoise_calls-n0,trace=list(trace),fingerprint_sha256=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest(),
            rng_cpu=digest(torch.get_rng_state()),rng_cuda=digest(torch.cuda.get_rng_state()),physical=list(rt.records),
            global_eligible_tiles=rt.global_eligible,global_skipped_tiles=sum(x['skipped'] for x in rt.records),
            global_kept_tiles=rt.global_eligible-sum(x['skipped'] for x in rt.records),
            local_eligible_tiles=rt.local_eligible,local_kept_tiles=rt.local_eligible,local_skipped_tiles=0,
            work_denominator='actual native denoise metadata Q128xKV64 rectangles; not CUDA CTAs')
    def cleanup(label):
        if engine.has_unfinished_requests():engine.abort_request([label]);step()
    def full_finish():
        final=None;f0=resource['runtime_forwards']
        while engine.has_unfinished_requests():
            for value in step():
                if value.finished:final=value
        if final is None:raise RuntimeError('missing final native output')
        value=final.outputs[0]
        return dict(output_tokens=len(value.token_ids),output_sha256=hashlib.sha256(json.dumps(list(value.token_ids)).encode()).hexdigest(),
            finish_reason=value.finish_reason,stop_reason=value.stop_reason,forwards=resource['runtime_forwards']-f0,
            native_fingerprints_sha256=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest(),
            rng_cpu=digest(torch.get_rng_state()),rng_cuda=digest(torch.cuda.get_rng_state()))
    # Warmup is an explicit first-canvas native request under the original cap.
    warm=byid[protocol['development'][1]['id']];add(warm,'warm');measure['on']=False;until_commit();cleanup('warm')
    item_results=[]
    for item_index,identity in enumerate(protocol['development']):
        row=dict(byid[identity['id']]);row['seed']=identity['seed'];label='development'+str(item_index)
        expected=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],add_generation_prompt=True,enable_thinking=row['thinking'],tokenize=True)
        if hasattr(expected,'keys'):expected=expected['input_ids']
        if expected and isinstance(expected[0],list):expected=expected[0]
        if list(expected)!=row['prompt_tokens']:raise RuntimeError('prompt tokenization mismatch')
        add(row,label);measure.update(on=True,keep_logits=False)
        initial=until_commit();L=len(trace);anchors=first_canvas_anchors(L);cleanup(label)
        itemdir=out/('item'+str(item_index).zfill(3));write(itemdir/'reference_landmarks.json',dict(id=row['id'],seed=row['seed'],L=L,anchors=list(anchors),native_reference=initial))
        anchor_results=[]
        for anchor in anchors:
            label='anchor'+str(item_index)+'_'+str(anchor)
            add(row,label);measure.update(on=True,keep_logits=False);rt.capture=True
            while observer.denoise_calls<anchor and observer.commit_calls==0:step()
            if observer.denoise_calls!=anchor or observer.commit_calls:raise RuntimeError('anchor unavailable after reference')
            if len(winner_history)<2:raise RuntimeError('no previous two native predictions')
            protected=(winner_history[-1]==winner_history[-2]).clone()
            old_frame=rt.frames[-1];output_states=list(engine.output_processor.request_states.values())
            if any(s.detokenizer.token_ids for s in output_states):raise RuntimeError('emitted output before anchor')
            rt.capture=False
            roots={name:getattr(runner,name) for name in ('req_states','input_buffers','block_tables','model_state','sampler','draft_tokens_handler','kv_caches','attn_groups')}
            roots.update(scheduler=core.scheduler,output_processor=engine.output_processor,observer=observer)
            shared=list(runner.model.parameters())+[runner.model,engine.output_processor.tokenizer,core.structured_output_manager,runner.sampler.embed_weight,runner.sampler.normalizer]
            snap=Snapshot(roots,shared=shared);folder=itemdir/('anchor'+str(anchor).zfill(3));write(folder/'snapshot.json',snap.verify())
            def restore():
                import tokenizers
                snap.restore();check=snap.verify()
                if not check['passed']:raise RuntimeError('restore storage/metadata mismatch')
                for s in output_states:s.detokenizer.stream=tokenizers.decoders.DecodeStream(ids=row['prompt_tokens'],skip_special_tokens=s.detokenizer.skip_special_tokens)
                events.clear();trace.clear();fingerprints.clear();winner_history.clear()
                rt.calls=anchor;rt.canvas=0;rt.anchor=anchor;rt.records=[];rt.frames=[];rt.capture=False
                rt.global_eligible=0;rt.local_eligible=0
                for name in ('req_states','input_buffers','block_tables','model_state','sampler','draft_tokens_handler','kv_caches','attn_groups'):
                    if getattr(runner,name) is not roots[name]:raise RuntimeError('runner root identity changed')
                return check
            def run(consumer='native',schedule='AA',support=None,one=False,complete=False,measurement=True):
                restore();rt.consumer=consumer;rt.schedule=schedule;rt.support=support;measure.update(on=measurement,keep_logits=True)
                if one:step();meta=dict(forwards=1,steps=1,trace=list(trace),physical=list(rt.records))
                elif complete:meta=full_finish()
                else:meta=until_commit()
                if meta.get('steps') is not None and meta['forwards']!=meta['steps']:raise RuntimeError('extra forward')
                return list(events),meta
            # Unrestored native continuation then two restores through first commit.
            events.clear();trace.clear();fingerprints.clear();winner_history.clear();measure.update(on=True,keep_logits=True)
            rt.consumer='native';rt.anchor=anchor
            unrestored=until_commit()
            unrestored_full=None
            if row['id']=='humaneval/HumanEval/124':
                unrestored_full=full_finish()
                unrestored_full['forwards']+=unrestored['forwards']
            native_events,native_meta=run()
            _,native_repeat=run()
            parity_keys=['fingerprint_sha256','remaining_denoise','rng_cpu','rng_cuda','forwards','steps','committed']
            if any(unrestored[k]!=native_meta[k] or native_meta[k]!=native_repeat[k] for k in parity_keys):raise RuntimeError('native first-canvas restore parity')
            _,uninstrumented=run(measurement=False)
            if any(native_meta[k]!=uninstrumented[k] for k in parity_keys):raise RuntimeError('new native measurement changed continuation')
            # Full-request parity on representative HE124 anchors only, before any full outcomes.
            full_qualification=None
            if row['id']=='humaneval/HumanEval/124':
                _,f1=run(complete=True);_,f2=run(complete=True)
                if f1!=f2 or f1!=unrestored_full:raise RuntimeError('full-request restore parity')
                full_qualification=dict(passed=True,unrestored_reference=unrestored_full,branches=[f1,f2],scope='HE124 corresponding anchor only')
            first,first_meta=run(one=True);rt.capture=True
            restore();rt.consumer='native';rt.capture=True;measure.update(on=True,keep_logits=True);step()
            frame=rt.frames[-1];rt.capture=False
            aa_events,aa_meta=run(consumer='allkept')
            _,aa_repeat=run(consumer='allkept')
            if any(aa_meta[k]!=aa_repeat[k] for k in parity_keys):raise RuntimeError('allkept first-canvas parity')
            q,k,v,prefix,scale=frame
            oldq,oldk,oldv,oldprefix,_=old_frame
            if prefix!=oldprefix:raise RuntimeError('response crosses canvas')
            units=[]
            for h in [0,8]:
                for qb in [0,1]:units.append((oldq[qb*128:(qb+1)*128,h],oldk[:,h//8],oldv[:,h//8]))
            candidates,bank_receipt=candidate_bank(q,k,v,prefix,dict(kind='single_delete',scale=scale),previous_qkv=units,seed=protocol['diagnostic_seed'],max_candidates=12)
            private_vectors={}
            for candidate in candidates:
                private_vectors[candidate['support_hash']]={}
                for unit in candidate['units']:
                    if 'response_components' in unit:
                        private_vectors[candidate['support_hash']][unit['unit']]=unit['response_components'].pop('_private_signed_vectors')
            tensor_path=root/'private'/('signed_response_'+str(item_index)+'_'+str(anchor)+'.pt')
            torch.save(private_vectors,tensor_path)
            bank_receipt['signed_vectors_private_sha256']=sha(tensor_path)
            del private_vectors
            guard=apply_guards(candidates);pairs=matched_pairs(candidates)
            # First pulses only; second-call outcomes remain unread until choices are immutable.
            decisions={};first_metrics={}
            for candidate in candidates:
                if not candidate['feasible']:continue
                support=torch.tensor(candidate['support'],dtype=torch.bool)
                pulse,meta=run(consumer='allkept',schedule='SA',support=support,one=True)
                decisions[candidate['support_hash']]=decision_score(aa_events[0],pulse[0],protected)
                first_metrics[candidate['support_hash']]=align_branch([aa_events[0]],pulse)
            choices=select_finalists(candidates,decisions)
            selected=set(choices.get('distinct_selected_hashes',[]))
            # Bound to four score winners; mass retained as metadata/control choice.
            selected={choices['selections'][name] for name in ['current_error','response','decision','sham'] if choices['selections'][name] is not None}
            write(folder/'candidate_bank.json',dict(candidates=candidates,receipt=bank_receipt,guards=guard,matched_pairs=pairs,first_pulse_decisions=decisions,first_pulse_metrics=first_metrics))
            write(folder/'selections.json',dict(**choices,finalist_hashes=sorted(selected),protocol_sha256=sha(root/'protocol.json')))
            controls=dict(native=unrestored,native_repeat=native_repeat,new_measurement_disabled=uninstrumented,measurement_parity=True,gqa_qualification=rt.gqa_receipt,
                allkept=aa_meta,allkept_repeat=aa_repeat,full_qualification=full_qualification,native_first_canvas_parity=True,
                allkept_first_canvas_parity=True,allkept_vs_native=align_branch(native_events,aa_events))
            # Predeclared most perturbative operator candidate tests post-treatment restoration.
            if candidates:
                perturb=max(candidates,key=lambda x:(x['current_error'],x['support_hash']))
                run(consumer='allkept',schedule='SS',support=torch.tensor(perturb['support'],dtype=torch.bool))
                _,post_native=run();_,post_aa=run(consumer='allkept')
                if any(post_native[x]!=native_meta[x] or post_aa[x]!=aa_meta[x] for x in parity_keys):raise RuntimeError('post-perturbation control contamination')
                controls.update(perturbation_hash=perturb['support_hash'],post_native=post_native,post_allkept=post_aa)
            write(folder/'controls.json',controls)
            results=[]
            if guard['identifying']:
                for candidate in sorted(candidates,key=lambda x:x['support_hash']):
                    if candidate['support_hash'] not in selected:continue
                    for schedule in ['SA','AS','SS']:
                        ev,meta=run(consumer='allkept',schedule=schedule,support=torch.tensor(candidate['support'],dtype=torch.bool))
                        rec=dict(support_hash=candidate['support_hash'],schedule=schedule,execution=meta,versus_AA=align_branch(aa_events,ev),AA_remaining=aa_meta['remaining_denoise'])
                        results.append(rec);write(folder/('branch'+str(len(results)).zfill(3)+'.json'),rec)
            write(folder/'receipt.json',dict(id=row['id'],seed=row['seed'],anchor=anchor,L=L,feasible=guard['feasible_count'],matched_pairs=sum(x['matched'] for x in pairs),response_pairs=sum(x['response_identifying_pair'] for x in pairs),distinct_finalists=len(selected),factorial_branches=len(results),identifying=guard['identifying'],measurement='actual native winners/state; device reconstructed acceptance only'))
            anchor_results.append(dict(anchor=anchor,feasible=guard['feasible_count'],distinct_finalists=len(selected),factorial_branches=len(results)))
            restore();cleanup(label);del snap,roots
        item_results.append(dict(id=row['id'],seed=row['seed'],L=L,anchors=anchor_results))
    for p,h in protocol['sources'].items():
        if sha(p)!=h:raise RuntimeError('source drift at completion '+p)
    write(out/'runtime.json',dict(compilation_policy=policy,local_fix_sha256=sha(fix),gpu=torch.cuda.get_device_name(),resource=resource))
    write(out/'final_complete.json',dict(passed=True,items=item_results,protocol_sha256=sha(root/'protocol.json'),resource=resource,scope='two-short-item development pulse/hold leaf; confirmation/scored outcomes deferred'))
    print(json.dumps(dict(passed=True,items=item_results,resource=resource)),flush=True)
if __name__=='__main__':main()
