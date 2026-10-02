"""The method inside vLLM 0.30.0's native DiffusionGemma vs vLLM's own dense serving, same engine and settings.

Arms (one per process, so each gets a fresh engine):
  dense    vLLM unchanged (its FA4 dense GLOBAL attention); cudagraph mode from VLLM_BENCH_CG (default: vLLM's own)
  allkept  the vLLM adapter's K/V buffers + FA4 all-kept on every GLOBAL decoder call (adapter cost, no skipping)
  method   the unchanged method core through experiments/numerical_qk_reuse/vllm_adapter.py with a frozen v21
           effective config (CONFIG_JSON, e.g. a panel's host fragment config of the main arm)
The adapter arms require cudagraph mode PIECEWISE (attention runs eagerly every step); measure dense in both modes.

Batch 1, bf16, chunked prefill, `--block-size` from VLLM_BENCH_BLOCK (32 gives GLOBAL 64-token pages), memory from
VLLM_BENCH_MEM. Every engine step is timed (synchronized). Public output: token counts, timings, receipts (adapter call
counts, the core's counters incl. effective_method). Completions go only to the private file VLLM_BENCH_PRIVATE.
vLLM rejects per-request seeds for diffusion models, so cells are matched by item, not by trajectory.
usage: python v27_vllm_method_bench.py MODEL_DIR MANIFEST_DIR CELLS_JSON OUT_JSONL ARM [CONFIG_JSON] [DATASETS]
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')


def main():
    model_dir, manifest_dir, cells_path, out_path, arm = sys.argv[1:6]
    config_path = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] not in ('', '-') else None
    datasets = set(sys.argv[7].split(',')) if len(sys.argv) > 7 else None
    if arm not in ('dense', 'allkept', 'method'):
        raise ValueError(arm)
    cells = json.loads(Path(cells_path).read_text())
    if datasets:
        cells = [c for c in cells if c['dataset'] in datasets]
    if os.environ.get('VLLM_BENCH_LIMIT'):
        cells = cells[:int(os.environ['VLLM_BENCH_LIMIT'])]
    chunk = int(os.environ.get('VLLM_BENCH_BATCHED', '16384'))
    mem = float(os.environ.get('VLLM_BENCH_MEM', '0.85'))
    block = int(os.environ['VLLM_BENCH_BLOCK']) if os.environ.get('VLLM_BENCH_BLOCK') else None
    cg = os.environ.get('VLLM_BENCH_CG') or None
    if arm != 'dense' and cg != 'PIECEWISE':
        raise ValueError('adapter arms need VLLM_BENCH_CG=PIECEWISE')
    private = os.environ.get('VLLM_BENCH_PRIVATE')
    rows = {}
    for dataset in {c['dataset'] for c in cells}:
        for r in json.loads((Path(manifest_dir) / f'{dataset}_generation_manifest.json').read_text()):
            rows[(dataset, r['id'])] = r
    longest = max(len(rows[(c['dataset'], c['id'])]['prompt_tokens']) + int(rows[(c['dataset'], c['id'])]['generation_budget'])
                  for c in cells)
    import torch
    import vllm
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    adapter = None
    config = None
    adapter_sha = None
    if arm != 'dense':
        # The core is imported from a panel deploy (cwd, PYTHONPATH=src:.) so its frozen source hashes validate; the
        # adapter (not a validated source) may live in an overlay directory appended to the package path.
        import hashlib
        import experiments.numerical_qk_reuse as pkg
        if os.environ.get('V27_ADAPTER_DIR'):
            pkg.__path__.append(os.environ['V27_ADAPTER_DIR'])
        from experiments.numerical_qk_reuse import vllm_adapter
        adapter_sha = hashlib.sha256(Path(vllm_adapter.__file__).read_bytes()).hexdigest()
        hf = AutoConfig.from_pretrained(model_dir)
        text = getattr(hf, 'text_config', hf)
        if arm == 'method':
            config = json.loads(Path(config_path).read_text())
        adapter = vllm_adapter.VllmMethodAdapter(text.layer_types, config=config,
                                                 condition=None if config is None else config['condition'], arm=arm)
        vllm_adapter.install_vllm_patches(adapter)
    max_len = min(262144, ((longest + 4096) // 1024 + 1) * 1024)
    kw = dict(model=model_dir, dtype='bfloat16', max_model_len=max_len, max_num_seqs=1, max_num_batched_tokens=chunk,
              enable_chunked_prefill=True, gpu_memory_utilization=mem, enable_prefix_caching=False,
              trust_remote_code=False, seed=0)
    if block:
        kw['block_size'] = block
    if cg:
        kw['compilation_config'] = {'cudagraph_mode': cg}
    llm = LLM(**kw)
    engine = llm.llm_engine
    out = open(out_path, 'a', encoding='utf-8')
    priv = open(private, 'a', encoding='utf-8') if private else None
    meta = dict(arm=arm, vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                max_model_len=max_len, chunk=chunk, gpu_memory_utilization=mem, block_size=block or 'default',
                cudagraph_mode=cg or 'default', adapter_sha256=adapter_sha,
                method_fingerprint=None if config is None else config.get('fingerprint'))
    for warm, cell in [(True, cells[0])] + [(False, c) for c in cells]:
        row = rows[(cell['dataset'], cell['id'])]
        ids = list(row['prompt_tokens'])
        params = SamplingParams(max_tokens=int(row['generation_budget']))
        rid = f"{'warm' if warm else 'run'}-{cell['dataset']}-{cell['seed']}-{abs(hash(cell['id'])) % 10**8}"
        if adapter is not None:
            adapter.begin_request()
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.add_request(rid, TokensPrompt(prompt_token_ids=ids), params)
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
        k = -(-len(ids) // chunk)
        decode = steps[k:]
        rec = dict(meta, dataset=cell['dataset'], index=cell.get('index'), panel_seed=cell['seed'], seed_applied=False,
                   prompt_tokens=len(ids), budget=int(row['generation_budget']),
                   output_tokens=len(final.outputs[0].token_ids) if final else None,
                   engine_steps=len(steps), prefill_steps=k, prefill_s=round(sum(steps[:k]), 4), wall_s=round(wall, 4),
                   decode_span_s=round(sum(decode), 4),
                   step_median_ms=round(1000 * statistics.median(decode), 3) if decode else None,
                   step_mean_ms=round(1000 * statistics.mean(decode), 3) if decode else None,
                   finish_reason=final.outputs[0].finish_reason if final else None,
                   receipts=receipts)
        out.write(json.dumps(rec, default=str) + '\n')
        out.flush()
        if priv is not None and final is not None:
            priv.write(json.dumps(dict(arm=arm, dataset=cell['dataset'], id=cell['id'], seed=cell['seed'],
                                       completion=final.outputs[0].text)) + '\n')
            priv.flush()
        print(json.dumps({k2: v for k2, v in rec.items() if k2 != 'receipts'}), flush=True)


if __name__ == '__main__':
    main()
