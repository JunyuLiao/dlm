"""Seed-paired vLLM panel worker: dense vs the method inside vLLM 0.30.0's native DiffusionGemma.

Why: vLLM's diffusion sampler draws its initial canvas and per-step Gumbel noise from the default torch CUDA
generator, and vLLM rejects per-request seeds for diffusion models. Without control, one request's denoising
trajectory varies several-fold between runs (V29: 99 vs 374 forwards for one 32K request), so arms cannot be
compared request by request. Here the default torch CPU/CUDA generators are reseeded before every request with
seed = f(dataset, index, panel seed, repeat), identical for every arm. Verified: same mode + same seed gives
token-identical outputs (results/v31_20261003/graphmode001). The method's own randomness uses private generators
(projection bank on CPU, State generator), so it never shifts the sampler's noise.

Arms (one engine per process): dense | native | allkept | method | mage (MAGE port, budget MAGE_K tokens); cudagraph mode per process (CG='default' keeps
vLLM's default FULL+PIECEWISE, 'PIECEWISE' the matched mode). Adapter arms require PIECEWISE.

Per request (public record, no text): prompt tokens, output tokens, sampler calls, canvases C (= commits =
ceil(output / 256)), denoising forwards N = sampler calls - C, prefill / decode / wall time (each engine step is
synchronized), per-step decode times summary, finish reason, adapter + method receipts. Private: raw completion with
special tokens (for the v15 final-channel LongBench scorer), id, finish reason.
usage: python v31_vllm_paired_bench.py MODEL MANIFEST_DIR CELLS_JSON OUT_JSONL PRIVATE_JSONL ARM CG [CONFIG_JSON]
  env: FIX_51994=1 (backport the upstream FULL-graph causal-buffer fix), LOGIT_STATS=fused (one-pass sampler-hook
       statistics, v31_logit_stats; default legacy torch ops), DP_BUILD=chunked (parallel dense-prefix build,
       v31_dp_chunked), OBSERVE=fa4 (FA4 in-kernel observation for compact-mu configs, v31_fa4_observe),
       MAGE_SELECT=fa4 (MAGE selection statistics from the FA4 observation), KV_COPY=triton / MERGE=triton (V30 one-kernel
       paged K/V refresh and alias-split LSE merge), TRACE=1 (per-canvas step counts and mean-entropy trajectories in the
       public record), REPEATS (default 1), MEM (0.85), BLOCK (32), CHUNK (16384), LIMIT, DATASETS (comma list), SEED_BASE (31),
       V27_ADAPTER_DIR (overlay holding vllm_adapter.py), SHARD=k/K (take cells k, k+K, ...)
"""
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
CANVAS = 256


def request_seed(base, cell, repeat):
    key = f"{base}|{cell['dataset']}|{cell.get('index')}|{cell['seed']}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') & 0x7FFFFFFF


def main():
    model_dir, manifest_dir, cells_path, out_path, private_path, arm, cg = sys.argv[1:8]
    config_path = sys.argv[8] if len(sys.argv) > 8 else None
    if arm not in ('dense', 'native', 'allkept', 'method', 'mage'):
        raise ValueError(arm)
    if arm != 'dense' and cg != 'PIECEWISE':
        raise ValueError('adapter arms need CG=PIECEWISE')
    cells = json.loads(Path(cells_path).read_text())
    if os.environ.get('DATASETS'):
        keep = set(os.environ['DATASETS'].split(','))
        cells = [c for c in cells if c['dataset'] in keep]
    if os.environ.get('SHARD'):
        k, K = (int(x) for x in os.environ['SHARD'].split('/'))
        cells = cells[k::K]
    if os.environ.get('LIMIT'):
        cells = cells[:int(os.environ['LIMIT'])]
    repeats = int(os.environ.get('REPEATS', '1'))
    seed_base = int(os.environ.get('SEED_BASE', '31'))
    rows = {}
    for ds in {c['dataset'] for c in cells}:
        for r in json.loads((Path(manifest_dir) / f'{ds}_generation_manifest.json').read_text()):
            rows[(ds, r['id'])] = r
    import torch
    import vllm
    import vllm.model_executor.models.diffusion_gemma as dg
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    fix_51994 = os.environ.get('FIX_51994') == '1'
    if fix_51994:
        apply_fix_51994()

    adapter, config, adapter_sha = None, None, None
    if arm != 'dense':
        import experiments.numerical_qk_reuse as pkg
        if os.environ.get('V27_ADAPTER_DIR'):
            pkg.__path__.append(os.environ['V27_ADAPTER_DIR'])
        from experiments.numerical_qk_reuse import vllm_adapter
        adapter_sha = hashlib.sha256(Path(vllm_adapter.__file__).read_bytes()).hexdigest()
        text = AutoConfig.from_pretrained(model_dir)
        text = getattr(text, 'text_config', text)
        if arm == 'method':
            config = json.loads(Path(config_path).read_text())
        adapter = vllm_adapter.VllmMethodAdapter(text.layer_types, config=config,
                                                 condition=None if config is None else config['condition'], arm=arm,
                                                 mage_k=int(os.environ.get('MAGE_K', '1024')),
                                                 logit_stats=os.environ.get('LOGIT_STATS', 'legacy'),
                                                 dp_build=os.environ.get('DP_BUILD', 'legacy'),
                                                 observe_backend=os.environ.get('OBSERVE', 'triton'),
                                                 mage_select=os.environ.get('MAGE_SELECT', 'torch'),
                                                 kv_copy_backend=os.environ.get('KV_COPY', 'torch'),
                                                 merge_backend=os.environ.get('MERGE', 'torch'),
                                                 trace_canvas=os.environ.get('TRACE') == '1')
        vllm_adapter.install_vllm_patches(adapter)
    counter = dict(calls=0)
    inner = dg._compiled_sample_step                    # (already wrapped by the adapter for adapter arms)

    def counting(*args, **kwargs):
        counter['calls'] += 1
        return inner(*args, **kwargs)
    dg._compiled_sample_step = counting

    longest = max(len(rows[(c['dataset'], c['id'])]['prompt_tokens']) + int(rows[(c['dataset'], c['id'])]['generation_budget'])
                  for c in cells)
    max_len = min(262144, ((longest + 4096) // 1024 + 1) * 1024)
    chunk = int(os.environ.get('CHUNK', '16384'))
    kw = dict(model=model_dir, dtype='bfloat16', max_model_len=max_len, max_num_seqs=1, max_num_batched_tokens=chunk,
              enable_chunked_prefill=True, gpu_memory_utilization=float(os.environ.get('MEM', '0.85')),
              enable_prefix_caching=False, trust_remote_code=False, seed=0, block_size=int(os.environ.get('BLOCK', '32')))
    if cg != 'default':
        kw['compilation_config'] = {'cudagraph_mode': cg}
    llm = LLM(**kw)
    engine = llm.llm_engine
    tok = llm.get_tokenizer()
    meta = dict(schema='v31_vllm_paired_v1', arm=arm, cudagraph_mode=cg, vllm=vllm.__version__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), max_model_len=max_len, chunk=chunk, block_size=kw['block_size'],
                gpu_memory_utilization=kw['gpu_memory_utilization'], seed_base=seed_base, adapter_sha256=adapter_sha,
                method_fingerprint=None if config is None else config.get('fingerprint'), fix_51994=fix_51994,
                mage_k=int(os.environ.get('MAGE_K', '1024')) if arm == 'mage' else None,
                mage_select=os.environ.get('MAGE_SELECT', 'torch') if arm == 'mage' else None,
                kv_copy_backend=os.environ.get('KV_COPY', 'torch') if arm != 'dense' else None,
                merge_backend=os.environ.get('MERGE', 'torch') if arm != 'dense' else None,
                logit_stats=os.environ.get('LOGIT_STATS', 'legacy') if arm == 'method' else None,
                dp_build=os.environ.get('DP_BUILD', 'legacy') if arm == 'method' else None,
                observe_backend=os.environ.get('OBSERVE', 'triton') if arm == 'method' else None)
    out = open(out_path, 'a', encoding='utf-8')
    priv = open(private_path, 'a', encoding='utf-8')
    schedule = [(True, cells[0], -1)] + [(False, c, r) for r in range(repeats) for c in cells]
    for warm, cell, rep in schedule:
        row = rows[(cell['dataset'], cell['id'])]
        ids = list(row['prompt_tokens'])
        params = SamplingParams(max_tokens=int(row['generation_budget']), skip_special_tokens=False)
        seed = request_seed(seed_base, cell, max(rep, 0))
        if adapter is not None:
            adapter.begin_request()
        counter['calls'] = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.add_request(f"{'w' if warm else 'r'}{rep}-{cell['dataset']}-{cell.get('index')}-{cell['seed']}",
                           TokensPrompt(prompt_token_ids=ids), params)
        steps, final = [], None
        while engine.has_unfinished_requests():
            a = time.perf_counter()
            outs = engine.step()
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - a)
            for o in outs:
                if o.finished:
                    final = o
        wall = time.perf_counter() - start
        receipts = adapter.end_request() if adapter is not None else None
        if warm:
            continue
        o = final.outputs[0]
        n_out = len(o.token_ids)
        canvases = math.ceil(n_out / CANVAS)
        k = -(-len(ids) // chunk)                          # prefill engine steps (one chunk each)
        decode = steps[k:]
        rec = dict(meta, dataset=cell['dataset'], index=cell.get('index'), panel_seed=cell['seed'], repeat=rep,
                   rng_seed=seed, prompt_tokens=len(ids), budget=int(row['generation_budget']), output_tokens=n_out,
                   sampler_calls=counter['calls'], canvases=canvases, denoise_forwards=counter['calls'] - canvases,
                   engine_steps=len(steps), prefill_steps=k, prefill_s=round(sum(steps[:k]), 5),
                   decode_s=round(sum(decode), 5), wall_s=round(wall, 5),
                   step_median_ms=round(1000 * statistics.median(decode), 3) if decode else None,
                   finish_reason=o.finish_reason,
                   output_hash=hashlib.sha256(json.dumps(list(o.token_ids)).encode()).hexdigest()[:16],
                   receipts=None if receipts is None else dict(adapter=receipts.get('adapter'),
                                                               method=_slim(receipts.get('method')),
                                                               trace=receipts.get('trace')))
        out.write(json.dumps(rec, default=str) + '\n')
        out.flush()
        priv.write(json.dumps(dict(arm=arm, cudagraph_mode=cg, dataset=cell['dataset'], index=cell.get('index'),
                                   panel_seed=cell['seed'], repeat=rep, id=cell['id'], finish_reason=o.finish_reason,
                                   completion=tok.decode(o.token_ids, skip_special_tokens=False))) + '\n')
        priv.flush()
        print(json.dumps({k2: rec[k2] for k2 in ('arm', 'cudagraph_mode', 'dataset', 'index', 'panel_seed', 'repeat',
                                                 'output_tokens', 'canvases', 'denoise_forwards', 'wall_s',
                                                 'step_median_ms')}), flush=True)


def _slim(m):
    if not m:
        return m
    keep = ('effective_method', 'carried_first_calls', 'fused_observations', 'dp_routes', 'held_decision_calls',
            'bootstrap_dense_calls', 'preqk_consumer_calls', 'attention_calls', 'layer_native_calls', 'fresh_fused_calls',
            'fa4_list_builds', 'async_observation_routes', 'protected_routes', 'v30_sensitivity', 'cgate')
    return {k: m.get(k) for k in keep if k in m}


def apply_fix_51994():
    """Backport of vLLM PR #51994 (merged 2026-09-30, not in 0.30.0): DiffusionGemma's per-request causal buffer was
    bool, so FlashAttentionMetadataBuilder.build() cast it out of place to int32 on every call; FULL CUDA graphs bound
    the capture-time copy and replayed a frozen causal/bidirectional flag. Allocating the buffer as int32 makes
    prepare_attn's slice assignment an in-place update that captured graphs see (the upstream fix also turns the
    builder's silent cast into an error; behaviour with an int32 buffer is identical)."""
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    cls = dg.DiffusionGemmaModelState
    if getattr(cls, '_v31_fix_51994', False):
        return
    init = cls.__init__

    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._causal_buf = torch.zeros(self._causal_buf.shape[0], dtype=torch.int32, device=self._causal_buf.device)
    cls.__init__ = patched
    cls._v31_fix_51994 = True


if __name__ == '__main__':
    main()
