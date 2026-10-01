"""v27 diagnostic (not a panel, timing only): how the GLOBAL-attention share of a DiffusionGemma denoising step, and
the per-step saving of FA4 block-sparse execution, scale with the serving batch size.

A real request of the dense control (D_fa4_allkept) of a bound v27 run dir is decoded eagerly. At one decoder-call
index its exact call arguments are captured and replicated to batch B (canvas inputs, self-conditioning logits,
position ids, masks and every encoder-cache layer copied B times). For each B the whole decoder forward is captured
in one CUDA graph and replayed (CUDA-event median), with the five GLOBAL layers' attention as
  dense      -- FA4 through its block-sparse interface with every tile kept (= D_fa4_allkept),
  keepX      -- the same FA4 kernel with a seeded synthetic keep map of fraction X of the prefix tiles (first tile and
                the canvas tiles always kept; outputs discarded -- TIMING ONLY, no quality meaning),
  canvasonly -- only the first and the canvas tiles kept (prefix attention ~ removed),
plus the per-step sampler work of the official step on the B x 256 x vocab logits, and the per-call K/V concat that the
eager path still pays (piecewise_v3+ removes it; reported separately so it can be subtracted).
The selection overhead of M1/M2/M3 is NOT included: rows are pure block-sparse execution.
usage: python v27_batch_step_bench.py RUN_DIR HOST UUID SEED STAGE ITEM_INDEX CALL_INDEX BATCHES KEEPS OUT_JSONL
       e.g. ... 404 longbench_v2_64k 0 6 1,2,4,8 0.12,0.2 out.jsonl
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import sys
from pathlib import Path


def main(argv=None):
    argv = argv or sys.argv[1:]
    run, host, uuid, seed, stage = Path(argv[0]), argv[1], argv[2], int(argv[3]), argv[4]
    item, call_index = int(argv[5]), int(argv[6])
    batches = [int(x) for x in argv[7].split(',')]
    keeps = [float(x) for x in argv[8].split(',')]
    out = open(argv[9], 'a', encoding='utf-8')
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       host, uuid, stage=stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    fwd_fa4 = v27_fa4.load()
    BST = v27_fa4._BST
    dense_original = v27_fa4.dense
    mode = dict(keep='dense', keys=None)
    lists = {}

    def maps(b, heads, qb, kt, keep, device):
        key = (b, heads, qb, kt, keep)
        if key not in lists:
            canvas_tiles = 4                                       # 256 canvas keys = the last 4 KV64 tiles
            if keep == 'dense':
                kept = torch.ones(b, heads, qb, kt, dtype=torch.bool, device=device)
            else:
                g = torch.Generator(device=device).manual_seed(0)
                p = 0.0 if keep == 'canvasonly' else float(keep)
                kept = torch.rand(b, heads, qb, kt, device=device, generator=g) < p
                kept[..., 0] = True
                kept[..., kt - canvas_tiles:] = True
            order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True).to(torch.int32)
            zeros = torch.zeros((b, heads, qb), device=device, dtype=torch.int32)
            lists[key] = (BST(mask_block_cnt=zeros, mask_block_idx=torch.zeros((b, heads, qb, 1), device=device,
                                                                                dtype=torch.int32),
                              full_block_cnt=kept.sum(-1).to(torch.int32).contiguous(),
                              full_block_idx=order.contiguous(), block_size=(128, 64)),
                          float(kept[..., :kt - canvas_tiles].float().mean()))
        return lists[key]

    def attention(q, k, v, scale):
        # batch-capable FA4 call (the module's own wrappers assume batch 1)
        mode['keys'] = k.shape[2]
        b, heads = q.shape[0], q.shape[1]
        bst, _ = maps(b, heads, -(-q.shape[2] // 128), -(-k.shape[2] // 64), mode['keep'], q.device)
        return fwd_fa4(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scale, causal=False,
                       block_sparse_tensors=bst)[0]

    def timed_graph(fn, n=15):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        del g
        torch.cuda.synchronize()
        return round(statistics.median(times), 3)

    def batched(x, b):
        if isinstance(x, torch.Tensor):
            return x.expand(b, *x.shape[1:]).contiguous() if x.dim() > 0 and x.shape[0] == 1 else x
        if isinstance(x, dict):
            return {key: batched(val, b) for key, val in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(batched(val, b) for val in x)
        return x

    def batched_cache(cache, b):
        new = copy.copy(cache)
        new.layers = []
        for layer in cache.layers:
            nl = copy.copy(layer)
            nl.keys, nl.values = batched(layer.keys, b), batched(layer.values, b)
            new.layers.append(nl)
        return new

    def sampler_work(logits):
        # the official step's per-step work on the logits (temperature, fp32 softmax, multinomial, argmax,
        # entropy-bound acceptance sort/cumsum, stop-criterion entropy)
        x = logits.float() / 0.75
        probs = torch.softmax(x, dim=-1)
        draw = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
        top = x.argmax(-1)
        logp = torch.log_softmax(x, -1)
        ent = -(logp.exp() * logp).sum(-1)
        srt, idx = torch.sort(ent, dim=-1)
        acc = torch.cumsum(srt, -1) - srt <= 0.1
        return draw, top, ent.mean(-1), acc, idx

    model_forward = type(adapter.model).forward
    calls = [0]

    def forward(self, *a, **k):
        index = calls[0]
        calls[0] += 1
        if index == call_index:
            pkv = k.get('past_key_values')
            for b in batches:
                row = dict(batch=b, call_index=index, stage=stage, item_index=item)
                try:
                    kb = {key: (batched_cache(val, b) if key == 'past_key_values' else batched(val, b))
                          for key, val in k.items()}
                    ab = batched(a, b)
                    for keep in ['dense'] + keeps + ['canvasonly']:
                        mode['keep'] = keep
                        row[f'forward_{keep}'] = timed_graph(lambda: model_forward(self, *ab, **kb))
                        if keep != 'dense':
                            match = [val[1] for key, val in lists.items() if key[0] == b and key[-1] == keep]
                            row[f'kept_prefix_{keep}'] = round(match[-1], 4) if match else None
                    mode['keep'] = 'dense'
                    try:
                        logits = torch.randn(b, 256, adapter.model.config.get_text_config().vocab_size,
                                             device='cuda', dtype=torch.bfloat16)
                        row['sampler'] = timed_graph(lambda: sampler_work(logits))
                        del logits
                    except Exception as exc:
                        row['sampler_error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
                    # the per-call K/V concat the eager decoder still performs (all 30 layers)
                    cat_k = [(layer.keys, layer.values) for layer in kb['past_key_values'].layers]
                    canvas_k = [torch.empty((b, kk.shape[1], 256, kk.shape[3]), device='cuda', dtype=kk.dtype)
                                for kk, _ in cat_k]

                    def concat():
                        for (kk, vv), ck in zip(cat_k, canvas_k):
                            torch.cat([kk, ck], dim=2)
                            torch.cat([vv, ck], dim=2)
                    row['kv_concat'] = timed_graph(concat)
                    row['keys'] = mode['keys']
                    del kb, ab, cat_k, canvas_k
                except Exception as exc:   # record (e.g. OOM at large batch) and keep going
                    row['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
                    mode['keep'] = 'dense'
                lists.clear()
                torch.cuda.empty_cache()
                out.write(json.dumps(row) + '\n')
                out.flush()
                print(json.dumps(row), flush=True)
        return model_forward(self, *a, **k)

    v27_fa4.dense = attention
    type(adapter.model).forward = forward
    try:
        config = configs[stage]['D_fa4_allkept']
        row = rows[protocol['ids'][stage][item]]
        with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel='fa4'):
            receipt = _one(adapter, row, seed, config)
        print(json.dumps(dict(done=True, torch=torch.__version__, calls=receipt['total_decoder_calls'])), flush=True)
    finally:
        type(adapter.model).forward = model_forward
        v27_fa4.dense = dense_original


if __name__ == '__main__':
    main()
