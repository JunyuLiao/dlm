"""vLLM port probe (no timing claims): for DiffusionGemma in vLLM 0.30.0, record at every FA4 dense attention call the KV
page size and whether the request's block table is one contiguous run of physical pages (then a contiguous K/V view
of the cache could feed the tested contiguous block-sparse path). Eager mode so the wrapper can inspect tensors; page
allocation is the same. Two sequential requests (the second reuses freed pages). No text is written.
usage: python v27_vllm_block_table_probe.py MODEL_DIR MANIFEST_DIR CELLS_JSON OUT_JSON
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')


def main():
    model_dir, manifest_dir, cells_path, out_path = sys.argv[1:5]
    cells = json.loads(Path(cells_path).read_text())
    picks = [next(c for c in cells if c['dataset'] == 'longbench_v2_32k'), next(c for c in cells if c['dataset'] == 'longbench_v2_64k')]
    rows = {}
    for c in picks:
        for r in json.loads((Path(manifest_dir) / f"{c['dataset']}_generation_manifest.json").read_text()):
            if r['id'] == c['id']:
                rows[c['dataset']] = r
    import torch
    import vllm.v1.attention.backends.flash_attn as fa
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    orig = fa._FA4_DENSE_ATTENTION_KERNEL
    seen = Counter()
    state = {'request': None}

    def wrapped(*args, **kw):
        bt, kc, su, q = kw.get('block_table'), kw.get('k'), kw.get('seqused_k'), kw.get('q')
        if bt is not None and su is not None and q is not None:
            page = int(kc.shape[1])
            n = int((int(su[0].item()) + page - 1) // page)
            row = bt[0, :n]
            contig = bool(((row[1:] - row[:-1]) == 1).all().item()) if n > 1 else True
            seen[(state['request'], page, contig, 'canvas' if q.shape[0] <= 256 else 'prefill')] += 1
        return orig(*args, **kw)

    fa._FA4_DENSE_ATTENTION_KERNEL = wrapped
    llm = LLM(model=model_dir, dtype='bfloat16', max_model_len=81920, max_num_seqs=1, max_num_batched_tokens=16384,
              enable_chunked_prefill=True, gpu_memory_utilization=0.92, enable_prefix_caching=False, enforce_eager=True)
    for name, row in rows.items():
        state['request'] = name
        llm.generate([TokensPrompt(prompt_token_ids=list(row['prompt_tokens']))], SamplingParams(max_tokens=600),
                     use_tqdm=False)
    out = [dict(request=k[0], page_size=k[1], block_table_contiguous=k[2], call=k[3], calls=v) for k, v in sorted(seen.items(), key=str)]
    json.dump(dict(rows=out), open(out_path, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
