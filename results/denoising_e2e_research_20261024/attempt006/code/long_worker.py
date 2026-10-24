"""Complete native requests, separate audit/clean passes, no restoration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'attempt003/code'))

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
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--suite', choices=['short', 'long'], required=True)
    ap.add_argument('--stage', choices=['smoke', 'panel'], required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / 'e2e_protocol.json').read_text())
    binding_path=Path(__file__).resolve().parents[1]/'execution_binding.json'
    binding=json.loads(binding_path.read_text())
    if sha(Path(__file__))!=binding['worker_sha256'] or sha(root/'e2e_protocol.json')!=binding['parent_protocol_sha256']:
        raise RuntimeError('long execution binding drift')
    heartbeat=root/'private'/('long_heartbeat_'+args.arm+'.json')
    heartbeat.parent.mkdir(parents=True,exist_ok=True)
    heartbeat.write_text(json.dumps(dict(phase='startup',unix=time.time())))
    for p, h in protocol['sources'].items():
        if sha(p) != h:
            raise RuntimeError('source drift: ' + p)
    leaf = root / 'runs' / args.stage / args.suite / args.arm
    leaf.mkdir(parents=True, exist_ok=False)
    raw = json.loads((root / 'private/tokenized_manifest.json').read_text())
    by_id = {r['id']: r for r in raw}
    identities = protocol[args.stage][args.suite]
    rows = [dict(by_id[x['id']], seed=x['seed']) for x in identities]
    mode = 'FULL_AND_PIECEWISE' if args.arm == 'dense_full_fix51994' else 'PIECEWISE'
    from scripts.v31_humaneval_repro_policy_worker import install_compilation_policy
    from scripts.v31_humaneval_cfgdet_base_worker import apply_fix_51994, apply_fa4_local_fix, NativeEmissionIntervals
    import torch._inductor.config as cfg
    cfg.deterministic = True
    cfg.benchmark_combo_kernel = False
    cfg.combo_kernels = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = install_compilation_policy()
    apply_fix_51994()
    fix = apply_fa4_local_fix(build_dir=root / 'runtime/fa4_local_fix')
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.models import diffusion_gemma as dg
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.utils import jit_monitor
    from experiments.numerical_qk_reuse.v31_value_native_audit import NativeGeometryAudit
    from online_runtime import Runtime
    online = args.arm not in {'dense_full_fix51994', 'dense_piecewise', 'v31_optimized'}
    runtime = Runtime(args.arm, False) if online else None
    if runtime:
        runtime.install()
    adapter = None
    if args.arm == 'v31_optimized':
        from transformers import AutoConfig
        from experiments.numerical_qk_reuse import vllm_adapter
        from experiments.numerical_qk_reuse.optimized_refresh_runtime import install
        text = AutoConfig.from_pretrained(protocol['model_path']).text_config
        adapter = vllm_adapter.VllmMethodAdapter(text.layer_types, arm='mage', profile=False,
            mage_k=128 if args.suite == 'short' else 8192,
            mage_select_step=1, mage_granularity='qblock_max', mage_carry_first=True,
            mage_reselect_trigger=[.15], mage_trigger_signal='settle', mage_sticky=1.386,
            logit_stats='fused', dp_build='chunked', observe_backend='fa4', mage_select='fa4',
            kv_copy_backend='triton', merge_backend='triton', value_audit=False,
            value_clean_timing=True)
        vllm_adapter.install_vllm_patches(adapter)
        install(adapter, method='v31', library='', observer_directory=str(root / 'runtime/optimized_observer'), schedule=2)
        from experiments.numerical_qk_reuse.resident_execution_runtime import install as resident_install
        resident_install(adapter, graph_buckets=False, v31_paged_refresh=True)
    audit = NativeGeometryAudit(['full_attention' if i in [5, 11, 17, 23, 29] else 'sliding_attention' for i in range(30)], 16)
    audit.bound = False
    audit.install()
    state = dict(bound=False, audit=False, calls=0, trace=[], captures=0, jit=0,
                 all_forwards=0, engine_steps=0, startup_forwards=0, executing=False)
    execute0 = GPUModelRunner.execute_model

    def execute(runner, *a, **kw):
        state['all_forwards'] += 1
        if state['all_forwards']%64==0:
            heartbeat.write_text(json.dumps(dict(phase='native_execution',unix=time.time(),engine_forwards=state['all_forwards'])))
        state['executing'] = True
        try:
            return execute0(runner, *a, **kw)
        finally:
            state['executing'] = False

    GPUModelRunner.execute_model = execute
    model0 = dg.DiffusionGemmaForConditionalGeneration.forward

    def model_forward(model, *a, **kw):
        if not state['executing']:
            state['startup_forwards'] += 1
        return model0(model, *a, **kw)

    dg.DiffusionGemmaForConditionalGeneration.forward = model_forward
    cap0 = torch.cuda.CUDAGraph.capture_begin

    def capture(graph, *a, **kw):
        state['captures'] += 1
        return cap0(graph, *a, **kw)

    torch.cuda.CUDAGraph.capture_begin = capture
    jit0 = jit_monitor._handle_jit_event

    def jit(**kw):
        if state['bound']:
            state['jit'] += 1
        return jit0(**kw)

    jit_monitor._handle_jit_event = jit
    sample0 = dg._compiled_sample_step

    def sample(*a, **kw):
        if not state['bound']:
            return sample0(*a, **kw)
        state['calls'] += 1
        committing = bool(a[8][a[1]].item()) if state['audit'] or runtime else False
        result = sample0(*a, **kw)
        rng = torch.cuda.get_rng_state().clone() if state['audit'] else None
        if runtime:
            runtime.native_sample(result, a, committing)
        if state['audit']:
            slot = a[1]
            row = dict(commit=committing, native_step=int(a[7][slot].item()),
                confidence=bool(a[9][slot].item()), converged=bool(a[8][slot].item()))
            if not committing:
                lp = result.float().log_softmax(-1)
                entropy = -(lp.exp() * lp).sum(-1)
                history = a[13][slot]
                stable_positions = (history == history[:, :1]).all(dim=1)
                native_winners = a[6][slot]
                if not torch.equal(result.argmax(-1).reshape_as(native_winners), native_winners):
                    raise RuntimeError('actual native argmax does not match returned scaled logits')
                row.update(entropy_mean=float(entropy.mean()), entropy_slack=float(entropy.mean()) - .005,
                    stable=bool(stable_positions.all()), stable_position_fraction=float(stable_positions.float().mean()),
                    confident_position_fraction=float((entropy < .005).float().mean()),
                    native_history_length=int(a[14][slot].item()),
                    native_history_ready=int(a[14][slot].item()) >= kw['ST'],
                    native_winner_sha256=hashlib.sha256(a[6][slot].detach().cpu().numpy().tobytes()).hexdigest(),
                    cap_reached=int(a[7][slot].item()) >= 48)
            state['trace'].append(row)
            if not torch.equal(rng, torch.cuda.get_rng_state()):
                raise RuntimeError('observer consumed native CUDA RNG')
        return result

    dg._compiled_sample_step = sample
    settings = protocol['runtime_settings'][args.suite]
    llm = LLM(model=protocol['model_path'], dtype='bfloat16', max_model_len=settings['max_model_len'],
        max_num_seqs=1, max_num_batched_tokens=16384, enable_chunked_prefill=True,
        gpu_memory_utilization=.85, enable_prefix_caching=False, trust_remote_code=False,
        seed=0, block_size=32, disable_log_stats=True, compilation_config={'cudagraph_mode': mode})
    engine = llm.llm_engine
    tokenizer = llm.get_tokenizer()
    initial_forwards = state['all_forwards']

    def request(row, label, measured, audit_pass):
        if runtime:
            runtime.begin(); runtime.audit = audit_pass
        if adapter:
            adapter.value_audit = audit_pass
            adapter.value_clean_timing = not audit_pass
            adapter.profile = audit_pass
            adapter.begin_request()
        audit.begin_request(); audit.bound = audit_pass
        state.update(bound=True, audit=audit_pass, calls=0, trace=[], jit=0)
        torch.manual_seed(row['seed']); torch.cuda.manual_seed_all(row['seed'])
        torch.cuda.synchronize()
        captures, forwards = state['captures'], state['all_forwards']
        start = time.perf_counter()
        engine.add_request(label, TokensPrompt(prompt_token_ids=row['prompt_tokens']),
            SamplingParams(max_tokens=row['generation_budget'], skip_special_tokens=False))
        count = 0
        prefill_end = None
        final = None
        intervals = NativeEmissionIntervals()
        while engine.has_unfinished_requests():
            outputs = engine.step()
            count += 1; state['engine_steps'] += 1
            # Accounting stop only if native request/output bounds are violated.
            if count>math.ceil(len(row['prompt_tokens'])/16384)+(math.ceil(row['generation_budget']/256)+1)*49+2:
                raise RuntimeError('native request execution bound violated')
            if count == math.ceil(len(row['prompt_tokens']) / 16384):
                torch.cuda.synchronize(); prefill_end = time.perf_counter()
            for output in outputs:
                if output.outputs:
                    intervals.observe(len(output.outputs[0].token_ids), state['calls'])
                if output.finished:
                    final = output
        torch.cuda.synchronize(); end = time.perf_counter(); state['bound'] = False
        if final is None or prefill_end is None:
            raise RuntimeError('missing full native completion/prefill boundary')
        value = final.outputs[0]
        T = len(value.token_ids); C = math.ceil(T / 256); N = state['calls'] - C
        if state['all_forwards'] - forwards != count:
            raise RuntimeError('extra model execution')
        work = audit.receipt() if audit_pass else None
        if work and runtime:
            work['global_kept_tiles'] -= runtime.skipped
            work['global_skipped_tiles'] = runtime.skipped
            work['overall_skipped_rectangles'] = runtime.skipped
        if work and adapter:
            actual = adapter.end_request()['adapter']
            work['adapter_receipt'] = actual
            if actual['global_eligible_tiles'] != work['global_eligible_tiles']:
                raise RuntimeError('V31 physical eligible denominator mismatch')
            work['global_kept_tiles'] = actual['global_kept_tiles']
            work['global_skipped_tiles'] = actual['global_eligible_tiles'] - actual['global_kept_tiles']
            work['overall_skipped_rectangles'] = work['global_skipped_tiles']
        rec = dict(id=row['id'], benchmark=row['benchmark'], seed=row['seed'], arm=args.arm,
            pass_name='audit' if audit_pass else 'clean', prompt_hash=row['prompt_hash'],
            prompt_tokenization_sha256=hashlib.sha256(json.dumps(row['prompt_tokens']).encode()).hexdigest(),
            protocol_sha256=sha(root / 'e2e_protocol.json'), execution_binding_sha256=sha(binding_path), generation_budget=row['generation_budget'],
            N=N, C=C, N_per_C=N/C if C else None, T=T, model_forwards=count,
            native_interval_steps=intervals.receipt(state['calls'], T, C),
            S=end-prefill_end, W=end-start, S_per_N=(end-prefill_end)/N if N else None,
            timing_valid=not audit_pass and state['captures'] == captures and state['jit'] == 0,
            cuda_capture_events=state['captures']-captures, request_jit_events=state['jit'],
            output_sha256=hashlib.sha256(json.dumps(list(value.token_ids)).encode()).hexdigest(),
            finish_reason=value.finish_reason,
            stop_reason='native_stop_token' if isinstance(value.stop_reason, int) else value.stop_reason,
            native_trace=state['trace'] if audit_pass else None, work=work,
            selector=runtime.receipt() if runtime and audit_pass else None)
        if measured:
            write(leaf / (label + '.json'), rec)
            write(root / 'private/generations' / args.stage / args.suite / args.arm / (label + '.json'),
                  dict(id=row['id'], seed=row['seed'], output_sha256=rec['output_sha256'], completion=tokenizer.decode(value.token_ids, skip_special_tokens=False)))
            print(json.dumps({k: rec[k] for k in ['id', 'seed', 'arm', 'pass_name', 'N', 'C', 'T', 'W', 'timing_valid']}), flush=True)
        return rec

    for index, row in enumerate(rows):
        expected = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])],
            tokenize=True, add_generation_prompt=True, enable_thinking=row['thinking'])
        if hasattr(expected, 'keys'):
            expected = expected['input_ids']
        if expected and isinstance(expected[0], list):
            expected = expected[0]
        if expected != row['prompt_tokens']:
            raise RuntimeError('prompt tokenization drift')
        # The complete untimed audit is the shape-specific warm request for
        # clean timing. Also exercise one independent full warm request first.
        if index == 0:
            request(row, f'warm{index:03d}', False, False)
        a = request(row, f'audit{index:03d}', True, True)
        b = request(row, f'clean{index:03d}', True, False)
        if any(a[k] != b[k] for k in ['output_sha256', 'N', 'C', 'T']):
            raise RuntimeError('audit/clean native trajectory mismatch')
        # A shape-specific warm request preceded the clean pass; new capture/JIT
        # makes timing unavailable, while accuracy/trajectory remain interpretable.
        write(leaf / f'cell{index:03d}_receipt.json', dict(passed=True, id=row['id'], seed=row['seed'],
            audit_clean_match=True, clean_timing_valid=b['timing_valid']))
        (root / 'private/e2e_live_resource.json').write_text(json.dumps(dict(arm=args.arm,
            suite=args.suite, stage=args.stage, completed_cells=index+1,
            engine_forwards=state['all_forwards'], startup_forwards=state['startup_forwards'],
            allocated_worker_seconds=time.time()-worker_started)))
    for p, h in protocol['sources'].items():
        if sha(p) != h:
            raise RuntimeError('source drift at completion: ' + p)
    write(leaf / 'final_complete.json', dict(passed=True, cells=len(rows), arm=args.arm,
        protocol_sha256=sha(root / 'e2e_protocol.json'), execution_binding_sha256=sha(binding_path), model_executions=state['all_forwards'],
        engine_initialization_executions=initial_forwards, startup_model_forwards=state['startup_forwards'], native_sampler_unchanged=True,
        local_fix_sha256=sha(fix), compilation_policy=policy))


if __name__ == '__main__':
    worker_started = time.time()
    main()
