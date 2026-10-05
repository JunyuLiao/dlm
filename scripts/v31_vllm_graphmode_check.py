"""Does vLLM's cudagraph mode change DiffusionGemma's denoising numerics? (dense only, no adapter)

Every arm that runs in PIECEWISE mode (native hooks, all-kept, main) took 20-30% fewer denoising forwards than
vLLM's default dense (FULL decode graphs) in the V18b/V28 panels, on large samples. Before any dense reference is
chosen, check with controlled randomness:
  - before every request the default torch CPU/CUDA generators are reseeded with a per-(cell, repeat) seed; the
    official sampler draws its Gumbel noise and initial canvas from them, so equal seeds give equal noise;
  - each cell runs twice with the SAME seed in one engine (determinism within a mode);
  - the wrapped sampler records, for every sampler call, the argmax tokens' hash and the mean token entropy of the
    temperature-scaled logits, and dumps the first denoising call's scaled logits of each request (private, bf16).
Run once per mode (default and PIECEWISE) and compare the dumps with v31_compare_graphmode.py.
Public output: counts, hashes, entropies, timings. Private: logits dumps and completion hashes only (no text).
usage: python v31_vllm_graphmode_check.py MODEL MANIFEST_DIR CELLS_JSON OUT_JSONL CG DUMP_DIR [DATASETS]
  CG: 'default', 'PIECEWISE' or 'eager' (enforce_eager=True, no CUDA graphs: the reference)
  env DUMP_CALLS=1,2,3 (sampler calls to dump), DUMP_ONLY=<index>_<seed> (dump only that cell)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')


def main():
    model_dir, manifest_dir, cells_path, out_path, cg, dump_dir = sys.argv[1:7]
    datasets = set(sys.argv[7].split(',')) if len(sys.argv) > 7 else None
    cells = json.loads(Path(cells_path).read_text())
    if datasets:
        cells = [c for c in cells if c['dataset'] in datasets]
    cells = cells[:int(os.environ.get('LIMIT', '6'))]
    rows = {}
    for ds in {c['dataset'] for c in cells}:
        for r in json.loads((Path(manifest_dir) / f'{ds}_generation_manifest.json').read_text()):
            rows[(ds, r['id'])] = r
    import torch
    import vllm
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    fix_51994 = os.environ.get('FIX_51994') == '1'
    if fix_51994:
        apply_fix_51994()

    trace = dict(calls=[], dump=None)
    original = dg._compiled_sample_step

    def sample_step(*args, **kwargs):
        scaled = original(*args, **kwargs)
        x = scaled.float()
        top = x.argmax(-1)
        logp = torch.log_softmax(x, -1)
        ent = float(-(logp.exp() * logp).sum(-1).mean())
        h = hashlib.sha256(top.cpu().numpy().tobytes()).hexdigest()[:16]
        trace['calls'].append((h, round(ent, 5)))
        if trace['dump'] is not None and len(trace['calls']) in trace['dump'][1]:
            torch.save(scaled.detach().to(torch.bfloat16).cpu(), f"{trace['dump'][0]}_call{len(trace['calls'])}.pt")
        return scaled
    dg._compiled_sample_step = sample_step

    longest = max(len(rows[(c['dataset'], c['id'])]['prompt_tokens']) + int(rows[(c['dataset'], c['id'])]['generation_budget'])
                  for c in cells)
    max_len = min(262144, ((longest + 4096) // 1024 + 1) * 1024)
    kw = dict(model=model_dir, dtype='bfloat16', max_model_len=max_len, max_num_seqs=1,
              max_num_batched_tokens=int(os.environ.get('CHUNK', '16384')), enable_chunked_prefill=True,
              gpu_memory_utilization=float(os.environ.get('MEM', '0.85')), enable_prefix_caching=False,
              trust_remote_code=False, seed=0, block_size=32)
    if cg == 'eager':
        kw['enforce_eager'] = True
    elif cg != 'default':
        kw['compilation_config'] = {'cudagraph_mode': cg}
    llm = LLM(**kw)
    engine = llm.llm_engine
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    out = open(out_path, 'a', encoding='utf-8')
    schedule = [(True, cells[0], 0)] + [(False, c, rep) for c in cells for rep in (0, 1)]
    for ordinal, (warm, cell, rep) in enumerate(schedule):
        row = rows[(cell['dataset'], cell['id'])]
        seed = 7_000_003 * (int(cell.get('index', 0)) + 1) + 1009 * int(cell['seed'])
        trace['calls'] = []
        # dump the first sampler call that follows the prompt (the first denoising call) of each timed repeat-0 request
        calls = tuple(int(x) for x in os.environ.get('DUMP_CALLS', '1').split(','))
        only = os.environ.get('DUMP_ONLY')
        tag = f"{cell.get('index')}_{cell['seed']}"
        trace['dump'] = None if warm or rep or (only and tag != only) else (
            str(Path(dump_dir) / f"{cg}{'_fix' if fix_51994 else ''}_{cell['dataset']}_{tag}"), calls)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.add_request(f'r{ordinal}', TokensPrompt(prompt_token_ids=list(row['prompt_tokens'])),
                           SamplingParams(max_tokens=int(row['generation_budget'])))
        final = None
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    final = o
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if warm:
            continue
        toks = list(final.outputs[0].token_ids)
        rec = dict(cudagraph_mode=cg, fix_51994=fix_51994, dataset=cell['dataset'], index=cell.get('index'), panel_seed=cell['seed'],
                   repeat=rep, rng_seed=seed, sampler_calls=len(trace['calls']), output_tokens=len(toks),
                   output_hash=hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16], wall_s=round(wall, 3),
                   first_calls=trace['calls'][:12], finish_reason=final.outputs[0].finish_reason,
                   vllm=vllm.__version__, torch=torch.__version__)
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != 'first_calls'}), flush=True)


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
