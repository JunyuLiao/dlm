"""Q64 alias S1/2/4 component sweep, with one real-need state per nominal bin.

Synthetic QKV use native token-major Q/interleaved paged KV strides and random
physical pages. All S share one Q64 mask and retain the adapter's generic merge,
including S1. Held-map timings include table/used allocations and LSE merge;
selector, first-use list construction, KV copies and model work are excluded.
This is not accuracy, online-method or request-speed evidence. Output is anonymous.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from scripts.v28_alias2_q64_bench import selected_paths, support
from scripts.v28_regroup_held_bench import nominal_bin, qualified_error
from scripts.v28_regroup_screen import load_need


SPLITS = (1, 2, 4)


def make_adapters(factory, paged):
    """Separate caches are mandatory: adapter cache keys omit the split count."""
    adapters = {}
    for splits in SPLITS:
        adapter = factory(['full_attention'], arm='allkept', lifecycle='request_clear')
        if adapter._split_cache:
            raise ValueError('sweep requires fresh split caches')
        adapter.splits = splits
        adapter.paged = paged
        adapters[splits] = adapter
    if len({id(adapter) for adapter in adapters.values()}) != len(SPLITS):
        raise ValueError('each split count requires an independent adapter')
    return adapters


def validate_split_batch(split, splits):
    if split.full_block_cnt.shape[0] != splits or split.full_block_idx.shape[0] != splits:
        raise RuntimeError('cached split batch does not match requested alias count')


def rotated_splits(repeat):
    offset = repeat % len(SPLITS)
    return SPLITS[offset:] + SPLITS[:offset]


def assert_held_builds(adapters, before):
    if any(adapter.calls['split_list_builds'] != before[splits] for splits, adapter in adapters.items()):
        raise RuntimeError('timed region rebuilt held split lists')


def timing_ratios(medians):
    if set(medians) != set(SPLITS) or not all(math.isfinite(value) and value > 0 for value in medians.values()):
        raise ValueError('invalid S1/2/4 timings')
    return {str(splits): medians[splits]/medians[2] for splits in SPLITS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_need_dir')
    parser.add_argument('new_report')
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args(argv)
    dest = Path(args.new_report)
    if dest.exists() or args.repeats < 6:
        raise ValueError('new report and at least six repeats required')
    paths = selected_paths(args.private_need_dir, per_bin=1)
    states = [(nominal_bin(need), need) for need in map(load_need, paths)]
    if sorted(length for length, _ in states) != [32768, 65536, 98304]:
        raise ValueError('all three nominal length bins are required')

    import torch
    from experiments.numerical_qk_reuse import v27_fa4 as fa
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    fa.load()
    started = time.monotonic()
    generator = torch.Generator(device='cuda').manual_seed(2804)
    records = []
    for length, need in states:
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
        mask = torch.from_numpy(support(need, 64)).to('cuda')
        lists = fa.block_sparse_tensors(mask[None], q_block=64)
        adapters = make_adapters(VllmMethodAdapter, dict(k=kc, v=vc, table=table, nk=nk))
        for splits, adapter in adapters.items():
            split = adapter._split(lists)
            validate_split_batch(split, splits)
            if not torch.equal(split.full_block_cnt.sum(0), lists.full_block_cnt[0]):
                raise RuntimeError('split counts do not partition the fixed support')
        kvk, kvv, scale = k.transpose(0, 1)[None], v.transpose(0, 1)[None], d ** -0.5

        def run(splits):
            return adapters[splits].sparse_lists(None, query, kvk, kvv, lists, scale)

        # All splits use the same FP32 masked reference; no change to selection.
        outputs = {splits: run(splits) for splits in SPLITS}
        if not all(torch.isfinite(output).all().item() for output in outputs.values()):
            raise RuntimeError('nonfinite paged alias output')
        maximum = {splits: 0. for splits in SPLITS}
        max_ref = 0.
        for head in range(h):
            scores = query[0, head].float() @ k[:, head // (h // hk)].float().T * scale
            row_mask = mask[head].repeat_interleave(64, 0).repeat_interleave(page, -1)
            ref = scores.masked_fill(~row_mask, -torch.inf).softmax(-1) @ v[:, head // (h // hk)].float()
            if not torch.isfinite(ref).all().item():
                raise RuntimeError('nonfinite shared masked reference')
            max_ref = max(max_ref, ref.abs().max().item())
            for splits in SPLITS:
                maximum[splits] = max(maximum[splits], (outputs[splits][0, :, head].float()-ref).abs().max().item())
        errors = {str(splits): qualified_error(maximum[splits], max_ref) for splits in SPLITS}
        del outputs
        for splits in SPLITS:
            for _ in range(8):
                run(splits)
        torch.cuda.synchronize()
        builds_before = {splits: adapter.calls['split_list_builds'] for splits, adapter in adapters.items()}
        samples, walls = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
        for repeat in range(args.repeats):
            for splits in rotated_splits(repeat):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record(); run(splits); end.record(); end.synchronize()
                walls[splits].append((time.perf_counter()-wall)*1000)
                samples[splits].append(start.elapsed_time(end))
        assert_held_builds(adapters, builds_before)
        medians = {splits: statistics.median(values) for splits, values in samples.items()}
        records.append(dict(nominal_length_bin=length, prefix_tokens=pt*page, samples=args.repeats,
                            errors=errors, gpu_ms={str(s): value for s, value in medians.items()},
                            synchronized_wall_ms={str(s): statistics.median(values) for s, values in walls.items()},
                            ratio_to_alias2=timing_ratios(medians),
                            counters={str(s): dict(a.calls) for s, a in adapters.items()},
                            actual_k_stride=list(kc.stride()),
                            actual_q_token_major_stride=list(query.transpose(1, 2).stride()),
                            fixed_q64_support=True, required_prefix_support_covered=True,
                            random_physical_page_order=True, generic_merge_for_all_splits=True))
        del query, cache, kc, vc, k, v, kvk, kvv, mask, lists, adapters, adapter, split, scores, row_mask, ref
    torch.cuda.synchronize()
    report = dict(schema='v28_q64_alias_sweep_component_v1', alias_counts=list(SPLITS), records=records,
                  reserved_gpu_seconds=time.monotonic()-started,
                  geomean_ratio_to_alias2={str(s): math.exp(statistics.mean(math.log(r['ratio_to_alias2'][str(s)]) for r in records)) for s in SPLITS},
                  scope='one historical real-need state per nominal bin; synthetic QKV; full canvas kept; component only',
                  exclusions=['selector', 'first-use keep-map/list and alias split construction', 'KV copies',
                              'request lifecycle', 'model', 'accuracy'],
                  s1_execution='adapter generic softmax/LSE merge retained; no direct-output optimization',
                  fp32_reference='one shared IEEE FP32 masked reference; TF32 explicitly disabled',
                  cuda_graphs='none requested; eager warmed benchmark')
    with dest.open('x', encoding='utf8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    return report


if __name__ == '__main__':
    main()
