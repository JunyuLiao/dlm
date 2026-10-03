"""Component qualification: real prefix supports, synthetic QKV, actual vLLM strides.

This is not a request/accuracy benchmark. Natural Q64 changes support versus
Q128; each is checked against its OWN masked FP32 reference. Timings include
the existing adapter's alias-list lookup, page-table/used allocation and LSE
merge, but exclude selector, KV copies, and model work. Publish aggregates only.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import time


def selected_paths(directory, per_bin):
    from scripts.v28_regroup_screen import load_need
    if per_bin < 1:
        raise ValueError('per_bin must be positive')
    bins = {}
    for path in sorted(Path(directory).glob('*.npz')):
        need = load_need(path)
        bins.setdefault(need.shape[-1], []).append(path)
    if not bins:
        raise ValueError('no validated snapshots')
    return [p for _, paths in sorted(bins.items()) for p in paths[:per_bin]]


def support(need, rows):
    import numpy as np
    if rows not in (64, 128) or need.shape[1] % rows:
        raise ValueError('complete Q64/Q128 groups required')
    h, q, pt = need.shape
    prefix = need.reshape(h, q // rows, rows, pt).any(2)
    # Real dumps cover prefix only; every canvas tile is explicitly kept.
    return np.concatenate((prefix, np.ones((h, q // rows, q // 64), dtype=bool)), axis=-1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('private_need_dir')
    p.add_argument('new_report')
    p.add_argument('--per-bin', type=int, default=2)
    p.add_argument('--repeats', type=int, default=30)
    a = p.parse_args(argv)
    dest = Path(a.new_report)
    if dest.exists() or a.repeats < 4:
        raise ValueError('new report and at least four repeats required')
    paths = selected_paths(a.private_need_dir, a.per_bin)
    import numpy as np
    import torch
    from experiments.numerical_qk_reuse import v27_fa4 as fa
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    from scripts.v28_regroup_screen import load_need
    fwd = fa.load()
    started = time.monotonic()
    g = torch.Generator(device='cuda').manual_seed(2802)
    records = []
    for path in paths:
        need = load_need(path)
        h, qn, pt = need.shape
        if h != 16 or qn != 256:
            raise ValueError('this qualification pins Gemma GLOBAL shape')
        hk, d, page = 2, 512, 64
        nk = pt * 64 + qn
        pages = nk // page
        # Native cache is [physical page, HK, page token, K+V head dim].
        cache = torch.randn(pages, hk, page, 2*d, device='cuda', dtype=torch.bfloat16, generator=g)
        kc, vc = cache.transpose(1, 2).split(d, dim=-1)
        table = torch.randperm(pages, generator=g, device='cuda').to(torch.int32)
        k = kc[table.long()].reshape(nk, hk, d)
        v = vc[table.long()].reshape(nk, hk, d)
        query = torch.randn(1, h, qn, d, device='cuda', dtype=torch.bfloat16, generator=g)
        adapter = VllmMethodAdapter(['full_attention'], arm='allkept', lifecycle='request_clear')
        adapter.paged = dict(k=kc, v=vc, table=table, nk=nk)
        lists, masks = {}, {}
        for rows in (128, 64):
            masks[rows] = torch.from_numpy(support(need, rows)).to('cuda')
            lists[rows] = fa.block_sparse_tensors(masks[rows][None], q_block=rows)
        scale = d ** -0.5
        # k/v supply shape to adapter dispatch; paged buffers supply real data.
        kvk, kvv = k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        def run(rows):
            return adapter.sparse_lists(None, query, kvk, kvv, lists[rows], scale)
        errors = {}
        for rows in (128, 64):
            got = run(rows)[0]
            max_abs, max_ref = 0., 0.
            for head in range(h):
                kh = k[:, head // (h // hk)].float()
                vh = v[:, head // (h // hk)].float()
                scores = query[0, head].float() @ kh.T * scale
                mask = masks[rows][head].repeat_interleave(rows, 0).repeat_interleave(64, 1)
                ref = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vh
                max_abs = max(max_abs, (got[:, head].float()-ref).abs().max().item())
                max_ref = max(max_ref, ref.abs().max().item())
            errors[str(rows)] = dict(max_abs=max_abs, relative_max=max_abs/max_ref)
            if max_abs/max_ref > 0.02:
                raise RuntimeError('paged alias2 masked reference qualification failed')
        used = torch.tensor([nk], dtype=torch.int32, device='cuda')
        dynamic = torch.tensor([False], device='cuda')
        def dense():
            return fwd(query.transpose(1, 2), kc, vc, softmax_scale=scale, causal=True,
                       dynamic_causal=dynamic, page_table=table[None], seqused_k=used, num_splits=0)[0]
        funcs = {'native_dense_component':dense, 'q128_alias2':lambda:run(128), 'q64_alias2':lambda:run(64)}
        for fn in funcs.values():
            for _ in range(8): fn()
        torch.cuda.synchronize()
        builds_before = adapter.calls['split_list_builds']
        samples = {name:[] for name in funcs}
        walls = {name:[] for name in funcs}
        for repeat in range(a.repeats):
            names = list(funcs)
            names = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in names:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record(); funcs[name](); end.record(); end.synchronize()
                walls[name].append((time.perf_counter()-wall)*1000)
                samples[name].append(start.elapsed_time(end))
        if adapter.calls['split_list_builds'] != builds_before:
            raise RuntimeError('timed region rebuilt held lists')
        rec = dict(prefix_tokens=pt*64, samples=a.repeats, errors=errors,
                   gpu_ms={k:statistics.median(v) for k,v in samples.items()},
                   synchronized_wall_ms={k:statistics.median(v) for k,v in walls.items()},
                   counters=dict(adapter.calls), actual_k_stride=list(kc.stride()),
                   effective_method='actual_paged_alias2_q64_vs_q128',
                   required_support_covered=True, random_physical_page_order=True)
        rec['q64_over_q128_gpu'] = rec['gpu_ms']['q64_alias2']/rec['gpu_ms']['q128_alias2']
        records.append(rec)
        print(json.dumps(rec), flush=True)
        del query, cache, kc, vc, k, v, kvk, kvv, adapter, masks, lists, got, kh, vh, scores, mask, ref
    torch.cuda.synchronize()
    report = dict(schema='v28_alias2_q64_component_v1', records=records,
                  reserved_gpu_seconds=time.monotonic()-started,
                  q64_over_q128_geomean=math.exp(statistics.mean(math.log(x['q64_over_q128_gpu']) for x in records)),
                  scope='synthetic QKV, real prefix need supports, full canvas kept; component only',
                  exclusions=['selector', 'KV copies', 'request lifecycle', 'model', 'accuracy'],
                  cuda_graphs='none requested; eager warmed kernel benchmark')
    dest.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
