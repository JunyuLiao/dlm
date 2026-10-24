"""Qualify in-place native continuation; all executed forwards are counted."""
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
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root/'protocol.json').read_text())
    for p, h in protocol['sources'].items():
        if sha(p) != h:
            raise RuntimeError('source drift: '+p)
    if sha(root/'private/tokenized_manifest.json') != protocol['private_manifest_sha256']:
        raise RuntimeError('private manifest drift')
    out = root/'qualification'
    out.mkdir(exist_ok=False)
    import torch._inductor.config as cfg
    cfg.deterministic = True
    cfg.benchmark_combo_kernel = False
    cfg.combo_kernels = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994, apply_fa4_local_fix
    policy = install_compilation_policy()
    apply_fix_51994()
    local_fix = apply_fa4_local_fix(build_dir=root/'runtime/fa4_local_fix')
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models import diffusion_gemma as dg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from production_observer import ProductionObserver, digest
    from state_restore import Snapshot
    observer = ProductionObserver()
    inventory = dict(bound=False, forwards=0, captures=0)
    runner_ref = {}
    original_execute = GPUModelRunner.execute_model
    def execute(runner, *a, **kw):
        runner_ref['runner'] = runner
        if inventory['bound']:
            inventory['forwards'] += 1
        return original_execute(runner, *a, **kw)
    GPUModelRunner.execute_model = execute
    original_capture = torch.cuda.CUDAGraph.capture_begin
    def capture(graph, *a, **kw):
        if inventory['bound']:
            inventory['captures'] += 1
        return original_capture(graph, *a, **kw)
    torch.cuda.CUDAGraph.capture_begin = capture
    original_sample = dg._compiled_sample_step
    logit_hashes = []
    def sample(*a, **kw):
        if inventory['bound']:
            logit_hashes.append(digest(a[0]))
        return observer.sample(original_sample, *a, **kw)
    dg._compiled_sample_step = sample
    started = time.monotonic()
    llm = LLM(model=protocol['model_path'], dtype='bfloat16', max_model_len=12288,
        max_num_seqs=1, max_num_batched_tokens=16384, enable_chunked_prefill=True,
        gpu_memory_utilization=.85, enable_prefix_caching=False, trust_remote_code=False,
        seed=0, block_size=32, disable_log_stats=True)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    runner = runner_ref['runner']
    pools = json.loads((root/'private/tokenized_manifest.json').read_text())
    row = next(r for r in pools if r['id']==protocol['smoke_item'])
    tokenizer = llm.get_tokenizer()
    expected = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
        add_generation_prompt=True, enable_thinking=row['thinking'], tokenize=True)
    if hasattr(expected, 'keys'):
        expected = expected['input_ids']
    if expected and isinstance(expected[0], list):
        expected = expected[0]
    if list(expected) != row['prompt_tokens']:
        raise RuntimeError('prompt tokenization mismatch')
    def add(label, mode):
        observer.begin(mode)
        torch.manual_seed(row['seed'])
        torch.cuda.manual_seed_all(row['seed'])
        engine.add_request(label, TokensPrompt(prompt_token_ids=row['prompt_tokens']),
            SamplingParams(max_tokens=row['generation_budget'], skip_special_tokens=False))
    def step():
        outputs = engine.step()
        torch.cuda.synchronize()
        return outputs
    def finish():
        final = None
        steps = 0
        while engine.has_unfinished_requests():
            for value in step():
                if value.finished:
                    final = value
            steps += 1
        if final is None:
            raise RuntimeError('no native final output')
        completion = final.outputs[0]
        return dict(output_sha256=hashlib.sha256(json.dumps(list(completion.token_ids)).encode()).hexdigest(),
            output_tokens=len(completion.token_ids), finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason, steps=steps, final_rng_cpu=digest(torch.get_rng_state()),
            final_rng_cuda=digest(torch.cuda.get_rng_state()))
    add('warm', 'off')
    warm = finish()
    inventory['bound'] = True
    add('restore_gate', 'minimal')
    initial_steps = 0
    while observer.denoise_calls < protocol['anchor_after_denoising_calls']:
        if not engine.has_unfinished_requests():
            raise RuntimeError('native request ended before anchor')
        step()
        initial_steps += 1
    if runner.execute_model_state is not None:
        raise RuntimeError('not a complete native engine-step boundary')
    # The DecodeStream is opaque. The frozen first-canvas anchor has emitted no
    # tokens, so reconstruct it with its exact original prompt after restoration.
    import tokenizers
    output_states = list(engine.output_processor.request_states.values())
    for state in output_states:
        if state.detokenizer.token_ids:
            raise RuntimeError('qualification requires no emitted tokens at anchor')
    runner_fields = ('req_states','input_buffers','block_tables','model_state',
        'sampler','draft_tokens_handler','kv_caches','attn_groups')
    roots = {name:getattr(runner,name) for name in runner_fields}
    roots.update(scheduler=core.scheduler, output_processor=engine.output_processor, observer=observer)
    shared = list(runner.model.parameters()) + [runner.model, engine.output_processor.tokenizer]
    shared.extend([core.structured_output_manager, runner.sampler.embed_weight, runner.sampler.normalizer])
    snapshot = Snapshot(roots, shared=shared)
    write(out/'snapshot_inventory.json', snapshot.verify())
    anchor_fingerprints = len(observer.fingerprints)
    anchor_logits = len(logit_hashes)
    branches = []
    reference_states = None
    reference_logits = None
    for label in ('dense_reference','dense_restored','dense_restored_repeat'):
        if branches:
            snapshot.restore()
            check = snapshot.verify()
            write(out/(label+'_restore.json'), check)
            if not check['passed']:
                raise RuntimeError('storage restoration mismatch')
            for state in output_states:
                state.detokenizer.stream = tokenizers.decoders.DecodeStream(
                    ids=row['prompt_tokens'], skip_special_tokens=state.detokenizer.skip_special_tokens)
        del logit_hashes[anchor_logits:]
        f0 = inventory['forwards']
        result = finish()
        states = observer.fingerprints[anchor_fingerprints:]
        logits = logit_hashes[anchor_logits:]
        result.update(label=label, forwards=inventory['forwards']-f0,
            sampler_calls=observer.sampler_calls, denoising_calls=observer.denoise_calls,
            commit_calls=observer.commit_calls, native_state_sha256=hashlib.sha256(json.dumps(states,sort_keys=True).encode()).hexdigest(),
            logits_sha256=hashlib.sha256(json.dumps(logits).encode()).hexdigest())
        write(out/(label+'.json'), result)
        if reference_states is None:
            reference_states, reference_logits = states, list(logits)
        elif states != reference_states or logits != reference_logits:
            raise RuntimeError('restored native continuation differs')
        branches.append(result)
        print(json.dumps(result), flush=True)
    native_fields = set(branches[0])-{'label'}
    passed = all(all(r[k]==branches[0][k] for k in native_fields) for r in branches)
    passed = passed and inventory['captures']==0 and all(r['forwards']==r['steps'] for r in branches)
    receipt = dict(passed=passed, protocol_sha256=sha(root/'protocol.json'),
        branches=branches, initial_native_steps=initial_steps, warm=warm,
        zero_hidden_forwards=all(r['forwards']==r['steps'] for r in branches),
        request_cuda_captures=inventory['captures'], full_request_continuation_equal=passed,
        sampler_changed=False, state_restore_scope='first-canvas complete engine-step boundary; all KV storage, scheduler/request/sampler state and RNG in place; opaque detokenizer reconstructed at zero emitted tokens',
        snapshot=snapshot.verify(), resource_seconds=time.monotonic()-started,
        compilation_policy=policy, local_fix_sha256=sha(local_fix))
    write(out/'receipt.json', receipt)
    if not passed:
        raise RuntimeError('native restore gate failed')
    write(out/'final_complete.json', dict(passed=True, protocol_sha256=sha(root/'protocol.json'),
        receipt_sha256=sha(out/'receipt.json'), scope='native dense restoration qualification only'))

if __name__=='__main__':
    main()
