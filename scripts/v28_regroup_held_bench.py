"""Held-order regroup ablation: synthetic QKV, real historical prefix needs.

This is an amortization diagnostic, not online-method or request performance.
Offline search, selector and first-use map/list construction are outside timing.
The total candidate includes Q gather, paged alias2 consumer and output scatter.
Only anonymous aggregates are saved; no filenames, orders or need bits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from scripts.v28_regroup_screen import (Options, group_support, load_need,
                                       natural_order, screen_need, validate_order)


SEARCH = Options(max_swaps_per_head=12, candidates_per_group=16)


def nominal_bin(need):
    tokens = need.shape[-1] * 64 + need.shape[1]
    return min((32768, 65536, 98304), key=lambda value: abs(tokens - value))


def select_accepted(directory, options=SEARCH):
    """At most one accepted snapshot per nearest historical nominal length bin."""
    bins = {}
    for path in sorted(Path(directory).glob('*.npz')):
        need = load_need(path)
        bins.setdefault(nominal_bin(need), []).append((path, need))
    if not bins:
        raise ValueError('no validated snapshots')
    selected = []
    for length, candidates in sorted(bins.items()):
        for _, need in candidates:
            result = screen_need(need, options)
            if result['accepted']:
                selected.append((length, need, result))
                break
    return selected


def support_for_order(need, order):
    prefix = group_support(need, order)
    return np.concatenate((prefix, np.ones((*prefix.shape[:2], need.shape[1] // 64), dtype=bool)), -1)


def original_row_support(need, order):
    """Candidate union mask in original row order, including the full canvas."""
    validate_order(need, order)
    expanded = np.repeat(support_for_order(need, order), 64, axis=1)
    restored = np.empty_like(expanded)
    restored[np.arange(need.shape[0])[:, None], order] = expanded
    return restored


def gather_query(query, order):
    import torch
    token_major = query.transpose(1, 2)
    index = order.T[None, :, :, None].expand_as(token_major)
    return torch.gather(token_major, 1, index).transpose(1, 2)


def scatter_output(output, order):
    import torch
    index = order.T[None, :, :, None].expand_as(output)
    return torch.empty_like(output).scatter_(1, index, output)


def break_even(natural_ms, consumer_ms, permutation_ms):
    if not all(math.isfinite(x) and x >= 0 for x in (natural_ms, consumer_ms, permutation_ms)):
        raise ValueError('invalid timing')
    saving = natural_ms - consumer_ms
    return dict(consumer_saving_ms=saving, gather_scatter_ms=permutation_ms,
                optimistic_margin_ms=saving-permutation_ms,
                optimistic_break_even=saving > permutation_ms)


def qualified_error(max_abs, max_ref, tolerance=.02):
    if (not all(math.isfinite(x) for x in (max_abs, max_ref, tolerance))
            or max_abs < 0 or max_ref <= 0 or tolerance <= 0
            or max_abs/max_ref > tolerance):
        raise RuntimeError('held regroup masked reference qualification failed')
    return dict(max_abs=max_abs, relative_max=max_abs/max_ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_need_dir')
    parser.add_argument('new_report')
    parser.add_argument('--repeats', type=int, default=32)
    args = parser.parse_args(argv)
    dest = Path(args.new_report)
    if dest.exists() or args.repeats < 4:
        raise ValueError('new report and at least four repeats required')
    selected = select_accepted(args.private_need_dir)
    if not selected:
        raise ValueError('no candidate passed the fixed more-search gate')

    import torch
    from experiments.numerical_qk_reuse import v27_fa4 as fa
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    # All reference matmuls must use IEEE FP32, irrespective of prior process settings.
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    fwd = fa.load()
    started = time.monotonic()
    generator = torch.Generator(device='cuda').manual_seed(2803)
    records = []
    for length, need, screened in selected:
        h, qn, pt = need.shape
        if h != 16 or qn != 256:
            raise ValueError('qualification pins Gemma GLOBAL shape')
        hk, d, page = 2, 512, 64
        nk = pt * page + qn
        cache = torch.randn(nk // page, hk, page, 2*d, device='cuda', dtype=torch.bfloat16, generator=generator)
        kc, vc = cache.transpose(1, 2).split(d, -1)
        table = torch.randperm(nk // page, device='cuda', generator=generator).to(torch.int32)
        k = kc[table.long()].reshape(nk, hk, d)
        v = vc[table.long()].reshape(nk, hk, d)
        query = torch.randn(qn, h, d, device='cuda', dtype=torch.bfloat16, generator=generator).transpose(0, 1)[None]
        orders = {'natural_q64': natural_order(need), 'held': screened['orders']['gated']}
        order_gpu = torch.from_numpy(orders['held']).to(device='cuda', dtype=torch.long)
        lists = {name: fa.block_sparse_tensors(torch.from_numpy(support_for_order(need, order))[None].to('cuda'), q_block=64)
                 for name, order in orders.items()}
        adapter = VllmMethodAdapter(['full_attention'], arm='allkept', lifecycle='request_clear')
        adapter.paged = dict(k=kc, v=vc, table=table, nk=nk)
        kvk, kvv, scale = k.transpose(0, 1)[None], v.transpose(0, 1)[None], d ** -0.5

        def consume(q, name):
            return adapter.sparse_lists(None, q, kvk, kvv, lists[name], scale)

        def held_total():
            return scatter_output(consume(gather_query(query, order_gpu), 'held'), order_gpu)

        errors = {}
        for name in orders:
            got = consume(query, name) if name == 'natural_q64' else held_total()
            if not torch.isfinite(got).all().item():
                raise RuntimeError('nonfinite consumer output')
            mask = torch.from_numpy(original_row_support(need, orders[name])).to('cuda').repeat_interleave(page, -1)
            max_abs, max_ref = 0., 0.
            for head in range(h):
                scores = query[0, head].float() @ k[:, head // (h // hk)].float().T * scale
                ref = scores.masked_fill(~mask[head], -torch.inf).softmax(-1) @ v[:, head // (h // hk)].float()
                if not torch.isfinite(ref).all().item():
                    raise RuntimeError('nonfinite masked reference')
                max_abs = max(max_abs, (got[0, :, head].float()-ref).abs().max().item())
                max_ref = max(max_ref, ref.abs().max().item())
            errors[name] = qualified_error(max_abs, max_ref)
        held_query = gather_query(query, order_gpu)
        output_template = consume(query, 'natural_q64')

        def permutation_only():
            # Standalone allocations/kernels, without the consumer; optimistic
            # estimate only, since cache interactions prevent additive timing.
            return gather_query(query, order_gpu), scatter_output(output_template, order_gpu)

        funcs = dict(natural_q64=lambda: consume(query, 'natural_q64'),
                     held_total=held_total, held_consumer_only=lambda: consume(held_query, 'held'),
                     permutation_only=permutation_only)
        for fn in funcs.values():
            for _ in range(8):
                fn()
        torch.cuda.synchronize()
        builds_before = adapter.calls['split_list_builds']
        samples, walls = {name: [] for name in funcs}, {name: [] for name in funcs}
        for repeat in range(args.repeats):
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
        medians = {name: statistics.median(values) for name, values in samples.items()}
        records.append(dict(nominal_length_bin=length, prefix_tokens=pt*page, samples=args.repeats,
                            errors=errors, gpu_ms=medians,
                            synchronized_wall_ms={name: statistics.median(values) for name, values in walls.items()},
                            held_total_over_natural=medians['held_total']/medians['natural_q64'],
                            optimistic_break_even=break_even(medians['natural_q64'], medians['held_consumer_only'], medians['permutation_only']),
                            counters=dict(adapter.calls), actual_k_stride=list(kc.stride()),
                            actual_q_token_major_stride=list(query.transpose(1, 2).stride()),
                            moved_rows=screened['moved_rows'], required_prefix_support_covered=True,
                            random_physical_page_order=True))
        del query, cache, kc, vc, k, v, kvk, kvv, lists, adapter, held_query, output_template, got, scores, ref, mask, order_gpu
    torch.cuda.synchronize()
    report = dict(schema='v28_regroup_held_component_v1', search_options=asdict(SEARCH), records=records,
                  reserved_gpu_seconds=time.monotonic()-started,
                  held_total_over_natural_geomean=math.exp(statistics.mean(math.log(r['held_total_over_natural']) for r in records)),
                  scope='accepted historical prefix supports; synthetic QKV; full canvas kept; held order only',
                  interpretation='amortization upper-limit diagnostic, not online performance or accuracy',
                  exclusions=['offline search', 'selector', 'first-use keep-map/list and alias split construction',
                              'KV copies', 'model', 'request lifecycle', 'accuracy'],
                  permutation_cost='standalone gather/scatter estimate; not necessarily additive with consumer',
                  fp32_reference='IEEE FP32; TF32 explicitly disabled', cuda_graphs='none requested; eager warmed benchmark')
    with dest.open('x', encoding='utf8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    return report


if __name__ == '__main__':
    main()
