"""Official-serving dense check for DiffusionGemma: vLLM's native implementation (FA4 on SM90 for diffusion models),
batch 1, bf16, on exactly the token ids, budgets and seeds of frozen panel cells, so its per-step and request times
can be set against our dense control (D_fa4_allkept) on the same cells and host.

Runs vLLM in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) and times every engine step. With one request in flight an
engine step is one model call: step 0 is the prompt prefill (max_num_batched_tokens covers the whole prompt), the rest
are canvas denoising steps and canvas commits. No prompt or output text is written; only token counts and timings.
usage: python v27_vllm_dense_bench.py MODEL_DIR MANIFEST_DIR CELLS_JSON OUT_JSONL
  CELLS_JSON: [{"dataset": ..., "id": ..., "index": ..., "seed": ...}, ...] (kept private; ids never written out)
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')


def main():
    model_dir, manifest_dir, cells_path, out_path = sys.argv[1:5]
    cells = json.loads(Path(cells_path).read_text())
    rows = {}
    for dataset in {c['dataset'] for c in cells}:
        for r in json.loads((Path(manifest_dir) / f'{dataset}_generation_manifest.json').read_text()):
            rows[(dataset, r['id'])] = r
    longest = max(len(rows[(c['dataset'], c['id'])]['prompt_tokens']) + int(rows[(c['dataset'], c['id'])]['generation_budget'])
                  for c in cells)
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    max_len = min(262144, ((longest + 4096) // 1024 + 1) * 1024)
    llm = LLM(model=model_dir, dtype='bfloat16', max_model_len=max_len, max_num_seqs=1,
              max_num_batched_tokens=max_len, gpu_memory_utilization=0.88, enable_prefix_caching=False,
              trust_remote_code=False, seed=0)
    engine = llm.llm_engine
    out = open(out_path, 'a', encoding='utf-8')
    meta = dict(vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(), max_model_len=max_len)
    for warm, cell in [(True, cells[0])] + [(False, c) for c in cells]:
        row = rows[(cell['dataset'], cell['id'])]
        ids = list(row['prompt_tokens'])
        params = SamplingParams(max_tokens=int(row['generation_budget']), seed=int(cell['seed']))
        rid = f"{'warm' if warm else 'run'}-{cell['dataset']}-{cell['seed']}-{abs(hash(cell['id'])) % 10**8}"
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
        if warm:
            continue
        decode = steps[1:]
        rec = dict(meta, dataset=cell['dataset'], index=cell.get('index'), seed=cell['seed'],
                   prompt_tokens=len(ids), budget=int(row['generation_budget']),
                   output_tokens=len(final.outputs[0].token_ids) if final else None,
                   engine_steps=len(steps), prefill_s=round(steps[0], 4), wall_s=round(wall, 4),
                   decode_span_s=round(sum(decode), 4),
                   step_median_ms=round(1000 * statistics.median(decode), 3) if decode else None,
                   step_mean_ms=round(1000 * statistics.mean(decode), 3) if decode else None,
                   finish_reason=final.outputs[0].finish_reason if final else None)
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
