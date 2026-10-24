"""Matched native two-call forks with selector-only current-Q/K/V support."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

def stats(logits, temp, previous):
    x=logits.float()/temp
    entropy=[];margin=[];argmax=[]
    for chunk in x.split(16):
        lp=chunk.log_softmax(-1)
        entropy.append(-(lp.exp()*lp).sum(-1))
        top=chunk.topk(2,dim=-1)
        margin.append(top.values[:,0]-top.values[:,1]);argmax.append(top.indices[:,0])
    entropy=torch.cat(entropy);margin=torch.cat(margin);argmax=torch.cat(argmax)
    h,idx=entropy.sort()
    accepted=torch.zeros_like(entropy,dtype=torch.bool)
    accepted[idx]=h.cumsum(0)-h.cummax(0).values<=.1
    return dict(entropy=entropy,margin=margin,argmax=argmax,accepted=accepted,
        stable=argmax==previous,confident=bool(entropy.mean()<.005))

def compare(candidate, reference):
    out=[]
    for c,r in zip(candidate,reference):
        out.append(dict(logit_l2=float((c['logits']-r['logits']).norm()),
            logit_max=float((c['logits']-r['logits']).abs().max()),
            top1_changed=int((c['argmax']!=r['argmax']).sum()),
            accepted_changed=int((c['accepted']!=r['accepted']).sum()),
            stable_changed=int((c['stable']!=r['stable']).sum()),
            confident_changed=c['confident']!=r['confident'],
            entropy_delta=float(c['entropy'].mean()-r['entropy'].mean()),
            margin_delta=float((c['margin']-r['margin']).mean()),
            self_condition_drift=float((c['sc']-r['sc']).norm()),
            native_stop_changed=c['converged']!=r['converged'],
            candidate_stop=c['converged'],reference_stop=r['converged'],
            correction_to_reference=int(((c['argmax']==r['argmax']) & (c['previous']!=r['argmax'])).sum()),
            break_from_reference=int(((c['argmax']!=r['argmax']) & (c['previous']==r['argmax'])).sum())))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--group',choices=['smoke','short','long'],required=True)
    args=parser.parse_args();root=args.root.resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    for p,h in protocol['sources'].items():
        if sha(p)!=h:raise RuntimeError('source drift: '+p)
    if sha(root/'private/tokenized_manifest.json') != protocol['private_manifest_sha256']:raise RuntimeError('private manifest drift')
    if sha(protocol['restore_gate']) != protocol['restore_gate_sha256']:raise RuntimeError('restore receipt drift')
    gate=json.loads(Path(protocol['restore_gate']).read_text())
    if not gate['passed']:raise RuntimeError('missing native restore qualification')
    if args.group!='smoke':
        smoke=json.loads((root/'smoke/final_complete.json').read_text())
        if not smoke['passed']:raise RuntimeError('support consumer smoke failed')
    out=root/args.group;out.mkdir(exist_ok=False)
    import torch._inductor.config as cfg
    cfg.deterministic=True;cfg.benchmark_combo_kernel=False;cfg.combo_kernels=False
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994,apply_fa4_local_fix
    policy=install_compilation_policy();apply_fix_51994()
    fix=apply_fa4_local_fix(build_dir=root/'runtime/fa4_local_fix')
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models import diffusion_gemma as dg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from production_observer import ProductionObserver,digest
    from support_oracle import Runtime,bank
    from state_restore import Snapshot
    rt=Runtime();rt.install()
    observer=ProductionObserver();counter=dict(bound=False,forwards=0,captures=0,jit=0)
    holder={};events=[];event_active={'enabled':False}
    from vllm.utils import jit_monitor
    native_jit=jit_monitor._handle_jit_event
    def jit(**kw):
        if counter['bound']:counter['jit']+=1
        return native_jit(**kw)
    jit_monitor._handle_jit_event=jit
    execute=GPUModelRunner.execute_model
    def counted(runner,*a,**kw):
        holder['runner']=runner
        if counter['bound']:counter['forwards']+=1
        return execute(runner,*a,**kw)
    GPUModelRunner.execute_model=counted
    capture=torch.cuda.CUDAGraph.capture_begin
    def captured(graph,*a,**kw):
        if counter['bound']:counter['captures']+=1
        return capture(graph,*a,**kw)
    torch.cuda.CUDAGraph.capture_begin=captured
    native=dg._compiled_sample_step
    def sample(*a,**kw):
        denoise=not bool(a[8][a[1]].item())
        previous=a[6][a[1]].detach().cpu().flatten().clone()
        logits=a[0].detach().float().cpu().clone() if counter['bound'] and denoise and event_active['enabled'] else None
        before_step=int(a[7][a[1]].item())
        result=observer.sample(native,*a,**kw)
        if logits is not None:
            temp=.4+.4*max(1,48-before_step)/48
            values=stats(logits,temp,previous)
            values.update(logits=logits,previous=previous,
                sc=a[10][a[1]].detach().float().cpu().clone(),
                converged=bool(a[8][a[1]].item()),
                native_state=observer.fingerprints[-1])
            events.append(values)
        return result
    dg._compiled_sample_step=sample
    pools=json.loads((root/'private/tokenized_manifest.json').read_text())
    by_id={r['id']:r for r in pools}
    ids=[protocol['smoke_item']] if args.group=='smoke' else protocol[args.group+'_ids']
    started=time.monotonic()
    llm=LLM(model=protocol['model_path'],dtype='bfloat16',
        max_model_len=202752 if args.group=='long' else 12288,
        max_num_seqs=1,max_num_batched_tokens=16384,enable_chunked_prefill=True,
        gpu_memory_utilization=.85,enable_prefix_caching=False,trust_remote_code=False,
        seed=0,block_size=32,disable_log_stats=True,
        compilation_config={'cudagraph_mode':'PIECEWISE'})
    engine=llm.llm_engine;core=engine.engine_core.engine_core;runner=holder['runner']
    tokenizer=llm.get_tokenizer();item_results=[]
    def add(row,label,observed):
        rt.begin();rt.bound=observed;rt.capture_enabled=observed
        observer.begin('minimal' if observed else 'off');events.clear()
        torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
        engine.add_request(label,TokensPrompt(prompt_token_ids=row['prompt_tokens']),
            SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
    def step():
        result=engine.step();torch.cuda.synchronize();return result
    def finish():
        last=None;steps=0
        while engine.has_unfinished_requests():
            for value in step():
                if value.finished:last=value
            steps+=1
        if last is None:raise RuntimeError('missing native completion')
        completion=last.outputs[0]
        return dict(steps=steps,output_tokens=len(completion.token_ids),
            output_sha256=hashlib.sha256(json.dumps(list(completion.token_ids)).encode()).hexdigest(),
            finish_reason=completion.finish_reason,N=observer.denoise_calls,C=observer.commit_calls)
    add(by_id[ids[0]],'warm',False);warm=finish()
    for item_index,item_id in enumerate(ids):
        row=by_id[item_id]
        expected=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],add_generation_prompt=True,enable_thinking=row['thinking'],tokenize=True)
        if hasattr(expected,'keys'):expected=expected['input_ids']
        if expected and isinstance(expected[0],list):expected=expected[0]
        if list(expected)!=row['prompt_tokens']:raise RuntimeError('prompt tokenization changed')
        counter['bound']=True;add(row,'item'+str(item_index),True)
        frames=[]
        anchor=protocol['anchor_after_denoising_calls']
        while observer.denoise_calls<anchor:
            step()
            if rt.captured is not None:
                frames.append(rt.captured);rt.captured=None
        previous=frames[-1]
        output_states=list(engine.output_processor.request_states.values())
        if any(s.detokenizer.token_ids for s in output_states):raise RuntimeError('anchor has emitted output')
        roots={name:getattr(runner,name) for name in ('req_states','input_buffers','block_tables','model_state','sampler','draft_tokens_handler','kv_caches','attn_groups')}
        roots.update(scheduler=core.scheduler,output_processor=engine.output_processor,observer=observer)
        shared=list(runner.model.parameters())+[runner.model,engine.output_processor.tokenizer,core.structured_output_manager,runner.sampler.embed_weight,runner.sampler.normalizer]
        snapshot=Snapshot(roots,shared=shared)
        folder=out/('item'+str(item_index).zfill(3));folder.mkdir()
        write(folder/'snapshot.json',snapshot.verify())
        def restore():
            import tokenizers
            snapshot.restore()
            for s in output_states:
                s.detokenizer.stream=tokenizers.decoders.DecodeStream(ids=row['prompt_tokens'],skip_special_tokens=s.detokenizer.skip_special_tokens)
            events.clear();rt.calls=anchor;rt.records=[];rt.captured=None
            rt.target_call=anchor+1;rt.duration=2;rt.capture_enabled=False
        def branch(label,mask=None,duration=2,capture_frame=False):
            restore();rt.arm=label;rt.mask=mask;rt.duration=duration;rt.capture_enabled=capture_frame
            event_active['enabled']=True
            f0=counter['forwards'];steps=0
            while len(events)<2 and engine.has_unfinished_requests():step();steps+=1
            if len(events)!=2:raise RuntimeError('insufficient native next-state events')
            if counter['forwards']-f0!=steps:raise RuntimeError('hidden forward')
            return list(events),dict(steps=steps,forwards=counter['forwards']-f0,physical=rt.records,
                rng_cpu=digest(torch.get_rng_state()),rng_cuda=digest(torch.cuda.get_rng_state()))
        dense,dense_meta=branch('native',capture_frame=True)
        current=rt.captured
        if current is None:raise RuntimeError('PIECEWISE capture path did not execute')
        # Capture the intervention state (first next call) in a separate exact
        # one-step replay, so the support does not accidentally use call t+1.
        restore();rt.arm='native';rt.capture_enabled=True;event_active['enabled']=True;step();current=rt.captured
        replay_first=events[0]
        if not torch.equal(replay_first['logits'],dense[0]['logits']):raise RuntimeError('same-support first-call parity')
        repeat,repeat_meta=branch('native')
        if any(not torch.equal(a['logits'],b['logits']) or a['native_state']!=b['native_state'] for a,b in zip(dense,repeat)):raise RuntimeError('PIECEWISE restore parity failed')
        allkept,allkept_meta=branch('allkept')
        if not allkept_meta['physical'] or any(x['skipped_tiles'] for x in allkept_meta['physical']):raise RuntimeError('allkept consumer receipt missing')
        write(folder/'controls.json',dict(dense=dense_meta,repeated=repeat_meta,allkept=allkept_meta,allkept_vs_native=compare(allkept,dense),restore_native_equal=True))
        q,k,v,prefix,scale=current
        pq,pk,pv,oldprefix,_=previous
        if prefix!=oldprefix:raise RuntimeError('canvas transition invalidates response anchor')
        bank_rows=[]
        device='cuda'
        q,k,v=[x.to(device=device,dtype=torch.float64) for x in (q,k,v)]
        old=(pq,pk,pv)
        seen=set()
        for fraction in protocol['budget_fractions']:
            for candidate in bank(q,k,v,old,prefix,fraction,scale):
                if candidate['support_sha256'] in seen:continue
                seen.add(candidate['support_sha256'])
                mask=candidate.pop('mask').cpu()
                native_events,metadata=branch('sparse',mask=mask)
                if metadata['rng_cpu']!=dense_meta['rng_cpu'] or metadata['rng_cuda']!=dense_meta['rng_cuda']:raise RuntimeError('selector changed RNG order')
                record=dict(candidate,requested_fraction=fraction,native_vs_allkept=compare(native_events,allkept),
                    native_vs_dense=compare(native_events,dense),execution=metadata)
                bank_rows.append(record)
                write(folder/('candidate'+str(len(bank_rows)).zfill(3)+'.json'),record)
        # Finish exactly one restored dense request so engine cleanup and EOS
        # remain native; diagnostic branches are explicitly bounded at two calls.
        restore();rt.arm='native';rt.capture_enabled=False;event_active['enabled']=False;final=finish()
        item=dict(id=item_id,seed=row['seed'],prefix=prefix,layer=29,head=0,query_block=0,
            candidate_records=len(bank_rows),dense_final=final,restore_equal=True,
            paired_task_accuracy=None,scope='two-call native diagnostic branches; no sparse task-score or full-trajectory claim')
        write(folder/'receipt.json',item);item_results.append(item)
        del snapshot,roots,dense,repeat,allkept,events[:],q,k,v
        counter['bound']=False
        print(json.dumps(item),flush=True)
    write(out/'runtime.json',dict(compilation_policy=policy,local_fix_sha256=sha(fix),gpu=torch.cuda.get_device_name()))
    for p,h in protocol['sources'].items():
        if sha(p)!=h:raise RuntimeError('source drift during execution: '+p)
    write(out/'final_complete.json',dict(passed=True,protocol_sha256=sha(root/'protocol.json'),items=item_results,
        cuda_captures=counter['captures'],jit_events=counter['jit'],explicit_native_forwards=counter['forwards'],
        resource_seconds=time.monotonic()-started,scope='bounded native selector-only diagnostics, not benchmark promotion'))

if __name__=='__main__':
    main()
