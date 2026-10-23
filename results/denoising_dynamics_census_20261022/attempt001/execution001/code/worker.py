"""Frozen dense-only production smoke/census; never enables a selector."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['smoke','census'],required=True)
    parser.add_argument('--group',choices=['short','long'])
    args=parser.parse_args();root=args.root.resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    for path,digest_ in protocol['sources'].items():
        if sha(path)!=digest_: raise RuntimeError('source fingerprint drift: '+path)
    if args.stage=='census':
        gate=json.loads((root/'smoke/final_complete.json').read_text())
        if not gate['passed'] or gate['protocol_sha256']!=sha(root/'protocol.json'):
            raise RuntimeError('dense observer smoke gate not satisfied')
        if not args.group: raise ValueError('census requires a group')
    output=root/('smoke' if args.stage=='smoke' else 'census_'+args.group)
    output.mkdir(exist_ok=False)
    write(output/'run_config.json',dict(stage=args.stage,group=args.group,
        protocol_sha256=sha(root/'protocol.json'),diagnostic_only=True))
    import torch
    import torch._inductor.config as cfg
    cfg.deterministic=True;cfg.benchmark_combo_kernel=False;cfg.combo_kernels=False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994,apply_fa4_local_fix
    policy=install_compilation_policy()
    apply_fix_51994()
    local_fix=apply_fa4_local_fix(build_dir=root/'runtime/fa4_local_fix')
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models import diffusion_gemma as dg
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    from transformers import AutoConfig
    from production_observer import ProductionObserver,digest
    inventory=dict(execute_model=0,captures=0,bound=False)
    original_execute=GPUModelRunner.execute_model
    def execute(self,*a,**kw):
        if inventory['bound']:inventory['execute_model']+=1
        return original_execute(self,*a,**kw)
    GPUModelRunner.execute_model=execute
    original_capture=torch.cuda.CUDAGraph.capture_begin
    def capture(self,*a,**kw):
        inventory['captures']+=1
        return original_capture(self,*a,**kw)
    torch.cuda.CUDAGraph.capture_begin=capture
    observer=ProductionObserver();sample=dg._compiled_sample_step
    dg._compiled_sample_step=lambda *a,**kw:observer.sample(sample,*a,**kw)
    config=AutoConfig.from_pretrained(protocol['model_path'])
    text=getattr(config,'text_config',config)
    geometry=NativeGeometryAudit(text.layer_types,text.num_attention_heads);geometry.install()
    pools=json.loads((root/'private/tokenized_manifest.json').read_text())
    rows={row['id']:row for row in pools}
    if args.stage=='smoke':
        row=rows[protocol['smoke_item']]
        schedule=[('warm','off',row),('off_before','off',row),
                  ('minimal_before','minimal',row),('rich','rich',row),
                  ('minimal_after','minimal',row),('off_after','off',row)]
        max_len=protocol['short_max_model_len']
    else:
        selected=[rows[i] for i in protocol['eligible_ids']
            if (rows[i]['benchmark']=='longbench_v2')==(args.group=='long')]
        if not selected:raise RuntimeError('no runnable items')
        schedule=[('warm','off',selected[0])]+[(r['id'],'rich',r) for r in selected]
        max_len=protocol['long_max_model_len'] if args.group=='long' else protocol['short_max_model_len']
    started=time.monotonic()
    llm=LLM(model=protocol['model_path'],dtype='bfloat16',max_model_len=max_len,
        max_num_seqs=1,max_num_batched_tokens=16384,enable_chunked_prefill=True,
        gpu_memory_utilization=.85,enable_prefix_caching=False,trust_remote_code=False,
        seed=0,block_size=32)
    engine=llm.llm_engine;tokenizer=llm.get_tokenizer()
    write(output/'runtime.json',dict(torch=torch.__version__,vllm=vllm.__version__,
        gpu=torch.cuda.get_device_name(),compilation_policy=policy,
        resolved_cudagraph_mode=str(engine.vllm_config.compilation_config.cudagraph_mode),
        local_fix_sha256=sha(local_fix),max_model_len=max_len,
        native_sampler_source_sha256=sha(dg.__file__)))
    records={};fingerprints={}
    for index,(label,mode,row) in enumerate(schedule):
        expected=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
            add_generation_prompt=True,enable_thinking=row['thinking'],tokenize=True)
        if list(expected)!=row['prompt_tokens']:raise RuntimeError('prompt tokenization changed')
        observer.begin(mode)
        if mode=='rich':geometry.begin_request()
        else:geometry.bound=False
        inventory['execute_model']=0;capture_start=inventory['captures']
        torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
        torch.cuda.synchronize();inventory['bound']=True
        request_start=time.monotonic();steps=0;final=None
        engine.add_request(str(index),TokensPrompt(prompt_token_ids=row['prompt_tokens']),
            SamplingParams(max_tokens=row['generation_budget'],skip_special_tokens=False))
        while engine.has_unfinished_requests():
            outs=engine.step();steps+=1
            for out in outs:
                if out.finished:final=out
        torch.cuda.synchronize();inventory['bound']=False
        if final is None:raise RuntimeError('native request has no final output')
        completion=final.outputs[0]
        rng=dict(cpu=digest(torch.get_rng_state()),cuda=digest(torch.cuda.get_rng_state()))
        record=dict(id=row['id'],label=label,mode=mode,seed=row['seed'],
            prompt_token_count=len(row['prompt_tokens']),
            prompt_tokenization_sha256=hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest(),
            prompt_hash=row['prompt_hash'],output_tokens=len(completion.token_ids),
            output_sha256=hashlib.sha256(json.dumps(list(completion.token_ids)).encode()).hexdigest(),
            finish_reason=completion.finish_reason,stop_reason=completion.stop_reason,
            sampler_calls=observer.sampler_calls,engine_steps=steps,
            model_execute_calls=inventory['execute_model'],
            new_cuda_captures=inventory['captures']-capture_start,final_rng=rng,
            instrumented_request_seconds=time.monotonic()-request_start,
            clean_timing=False,trajectory_summary=observer.summary() if mode=='rich' else None,
            geometry=geometry.receipt() if mode=='rich' else None)
        folder=output/('request'+str(index).zfill(3))
        write(folder/'record.json',record)
        if mode!='off':
            write(folder/'state_hashes.json',observer.fingerprints)
            fingerprints[label]=observer.fingerprints
        if mode=='rich':write(folder/'trace.json',observer.records)
        write(folder/'private/generation.json',dict(id=row['id'],completion=tokenizer.decode(completion.token_ids,skip_special_tokens=False)))
        records[label]=record
        print(json.dumps(dict(label=label,id=row['id'],sampler_calls=observer.sampler_calls,
            output_tokens=len(completion.token_ids),seconds=record['instrumented_request_seconds'])),flush=True)
    if args.stage=='smoke':
        fields=('output_sha256','output_tokens','finish_reason','stop_reason','sampler_calls',
            'engine_steps','model_execute_calls','final_rng','prompt_tokenization_sha256')
        baseline=records['off_before'];differences={}
        for label,rec in records.items():
            if label=='warm':continue
            changed=[f for f in fields if rec[f]!=baseline[f]]
            if changed:differences[label]=changed
        def effective(trace):
            trace=json.loads(json.dumps(trace))
            for entry in trace:
                if entry['before']['step']==[0]:
                    entry['before'].pop('argmax_sha256',None)
            return trace
        same_states=(effective(fingerprints['minimal_before'])==effective(fingerprints['rich'])
                     ==effective(fingerprints['minimal_after']))
        rich=records['rich']
        passed=(not differences and same_states and
            rich['trajectory_summary']['native_predicate_mismatches']==0 and
            all(r['model_execute_calls']==r['engine_steps'] for k,r in records.items() if k!='warm'))
        receipt=dict(passed=passed,differences=differences,per_call_native_state_hashes_equal=same_states,
            zero_extra_forwards=all(r['model_execute_calls']==baseline['model_execute_calls'] for k,r in records.items() if k!='warm'),
            final_native_rng_equal=all(r['final_rng']==baseline['final_rng'] for k,r in records.items() if k!='warm'),
            rich_observer_rng_checks=rich['trajectory_summary']['observer_rng_checks'],
            full_cache_fork_restore_validated=False,
            limitation='Native per-call state and repeat-output invariance only; full KV/layout fork restore is a separate Phase 2 gate.')
        write(output/'receipt.json',receipt)
        if not passed:raise RuntimeError('dense observer parity gate failed; census forbidden')
    else:
        receipt=dict(passed=all(r['trajectory_summary']['native_predicate_mismatches']==0
            for k,r in records.items() if k!='warm'),
            completed_ids=[r['id'] for k,r in records.items() if k!='warm'])
    write(output/'final_complete.json',dict(protocol_sha256=sha(root/'protocol.json'),
        **receipt,resource_reservation_seconds=time.monotonic()-started,
        scope='one-item observer smoke' if args.stage=='smoke' else 'eligible '+args.group+' dense diagnostic census',
        full30_complete=False))

if __name__=='__main__':main()
