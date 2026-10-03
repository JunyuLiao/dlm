"""Fixed held-order permutation diagnostic, never a quality/request benchmark.

permutation mode needs only torch/Triton; held mode reuses the unchanged Q64
support and alias2 consumer. Offline search and first construction are excluded.
Each kernel is qualified against torch before any timed use. No peer code used.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from scripts import v28_regroup_held_bench as held


def validate_permutation(order, heads, rows):
    order = np.asarray(order)
    if (heads <= 0 or rows <= 0 or order.shape != (heads, rows)
            or order.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(rows), order.shape))):
        raise ValueError('per-head row permutation must be a bijection')
    return order


def inverse_order(order):
    order = np.asarray(order)
    if order.ndim != 2:
        raise ValueError('per-head row permutation must be two-dimensional')
    validate_permutation(order, *order.shape)
    return np.argsort(order, axis=1)


def linear_addresses(offsets, order, rows, heads, dim, source_strides, scatter=False):
    """CPU address audit, including final partial launch blocks; not GPU time."""
    validate_permutation(order, heads, rows)
    if dim <= 0 or len(source_strides) != 3 or any(s <= 0 for s in source_strides):
        raise ValueError('positive dimensions and source strides required')
    offsets = np.asarray(offsets, dtype=np.int64)
    valid = (offsets >= 0) & (offsets < rows*heads*dim)
    safe = np.where(valid, offsets, 0)
    d, h, r = safe % dim, (safe // dim) % heads, safe // (dim*heads)
    mapped = order[h, r]
    sq, sh, sd = source_strides
    source = (r if scatter else mapped)*sq + h*sh + d*sd
    target = (mapped*heads*dim+h*dim+d) if scatter else safe
    return source, target, valid


def _kernel():
    # Lazy imports keep all CPU tests independent of torch/Triton.
    # tl is a lazy module global: JIT name resolution need not inspect closures.
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def permute(X, Order, Y, Q: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                SQ: tl.constexpr, SH: tl.constexpr, SD: tl.constexpr,
                SCATTER: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        valid = i < Q*H*D
        d = i % D
        h = (i // D) % H
        r = i // (D*H)
        mapped = tl.load(Order+h*Q+r, mask=valid, other=0)
        if SCATTER:
            src = r*SQ+h*SH+d*SD
            dst = mapped*H*D+h*D+d
        else:
            src = mapped*SQ+h*SH+d*SD
            dst = i
        value = tl.load(X+src, mask=valid, other=0)
        tl.store(Y+dst, value, mask=valid)

    return permute


class TritonPermutation:
    """One immutable held order; qualification is deliberately outside timing."""

    def __init__(self, order_gpu):
        import torch
        if (order_gpu.ndim != 2 or order_gpu.dtype != torch.long
                or not order_gpu.is_cuda or not order_gpu.is_contiguous()):
            raise ValueError('contiguous CUDA int64 order required')
        self.heads, self.rows = order_gpu.shape
        validate_permutation(order_gpu.cpu().numpy(), self.heads, self.rows)
        self.order = order_gpu  # Own the tensor; slot/pointer reuse is not identity.
        self.kernel = _kernel()
        self.gather_qualified = self.scatter_qualified = False

    def _run(self, token_major, scatter):
        import torch
        import triton
        if (token_major.ndim != 4 or token_major.shape[:3] != (1, self.rows, self.heads)
                or token_major.shape[3] <= 0 or not token_major.is_cuda
                or token_major.device != self.order.device
                or token_major.dtype != torch.bfloat16 or any(s <= 0 for s in token_major.stride())):
            raise ValueError('single-batch CUDA BF16 token-major geometry required')
        output = torch.empty(tuple(token_major.shape), device=token_major.device, dtype=token_major.dtype)
        q, h, d = token_major.shape[1:]
        self.kernel[(triton.cdiv(q*h*d, 1024),)](token_major, self.order, output,
            q, h, d, *token_major.stride()[1:], scatter, 1024, num_warps=4)
        return output

    @staticmethod
    def _exact(got, expected):
        import torch
        if (not torch.isfinite(got).all().item()
                or not torch.isfinite(expected).all().item() or not torch.equal(got, expected)):
            raise RuntimeError('Triton permutation exact torch oracle failed before timing')

    def gather(self, query):
        token_major = query.transpose(1, 2)
        got = self._run(token_major, False).transpose(1, 2)
        if not self.gather_qualified:
            self._exact(got, held.gather_query(query, self.order))
            self._exact(self._run(got.transpose(1, 2), True), token_major)
            self.gather_qualified = True
        return got

    def scatter(self, output):
        got = self._run(output, True)
        if not self.scatter_qualified:
            self._exact(got, held.scatter_output(output, self.order))
            self.scatter_qualified = True
        return got

    def require_qualified(self):
        if not (self.gather_qualified and self.scatter_qualified):
            raise RuntimeError('both permutation kernels must qualify before timing')


def interpretation(mode):
    return ('same-host standalone permutation cost; do not add to a different-host consumer time'
            if mode == 'permutation' else
            'held-order amortization diagnostic, not online performance or model accuracy')


def _time_functions(torch, funcs, repeats):
    for fn in funcs.values():
        for _ in range(8):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in funcs}
    for repeat in range(repeats):
        names = list(funcs)
        names = names[repeat % len(names):]+names[:repeat % len(names)]
        for name in names:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); funcs[name](); end.record(); end.synchronize()
            samples[name].append(begin.elapsed_time(end))
    return {name: statistics.median(values) for name, values in samples.items()}


def permutation_report(directory, repeats):
    selected = held.select_accepted(directory)
    if len(selected) != 3 or {x[0] for x in selected} != {32768, 65536, 98304}:
        raise ValueError('the three original held nominal bins are required')
    import torch
    started = time.monotonic()
    generator = torch.Generator(device='cuda').manual_seed(2863)
    records = []
    for length, need, screened in selected:
        h, q, _ = need.shape
        if (h, q) != (16, 256):
            raise ValueError('qualification pins Gemma GLOBAL shape')
        order = torch.from_numpy(screened['orders']['gated']).to(device='cuda', dtype=torch.long)
        backend = TritonPermutation(order)
        query = torch.randn(q, h, 512, device='cuda', dtype=torch.bfloat16, generator=generator).transpose(0, 1)[None]
        output = torch.randn(1, q, h, 512, device='cuda', dtype=torch.bfloat16, generator=generator)
        backend.gather(query); backend.scatter(output); backend.require_qualified()
        funcs = dict(torch_permutation=lambda: (held.gather_query(query, order), held.scatter_output(output, order)),
                     triton_permutation=lambda: (backend.gather(query), backend.scatter(output)))
        medians = _time_functions(torch, funcs, repeats)
        records.append(dict(nominal_length_bin=length, samples=repeats, gpu_ms=medians,
                            triton_over_torch=medians['triton_permutation']/medians['torch_permutation'],
                            exact_permutation_oracle=True, native_q_stride=list(query.stride()),
                            moved_rows=screened['moved_rows']))
    torch.cuda.synchronize()
    return dict(schema='v28_regroup_triton_permutation_v1', records=records,
                reserved_gpu_seconds=time.monotonic()-started,
                triton_over_torch_geomean=math.exp(statistics.mean(math.log(r['triton_over_torch']) for r in records)),
                interpretation=interpretation('permutation'), quality_evaluated=False,
                scope='same fixed held selections; synthetic native-stride BF16 Q and token-major output',
                exclusions=['offline search', 'first construction', 'consumer', 'selector', 'model', 'request', 'accuracy'],
                cuda_graphs='none requested; eager warmed; rotated same-host comparison')


def held_report(directory, repeats):
    # Reuse selection/support/reference helpers. All six routes share this
    # process, query, physical pages and alias2 adapter, and rotate together.
    selected = held.select_accepted(directory)
    if len(selected) != 3 or {x[0] for x in selected} != {32768, 65536, 98304}:
        raise ValueError('the three original held nominal bins are required')
    import torch
    from experiments.numerical_qk_reuse import v27_fa4 as fa
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    fa.load()
    started = time.monotonic()
    generator = torch.Generator(device='cuda').manual_seed(2803)
    records = []
    for length, need, screened in selected:
        h, qn, pt = need.shape
        if (h, qn) != (16, 256):
            raise ValueError('qualification pins Gemma GLOBAL shape')
        hk, d, page = 2, 512, 64
        nk = pt*page+qn
        cache = torch.randn(nk//page, hk, page, 2*d, device='cuda', dtype=torch.bfloat16, generator=generator)
        kc, vc = cache.transpose(1, 2).split(d, -1)
        table = torch.randperm(nk//page, device='cuda', generator=generator).to(torch.int32)
        k = kc[table.long()].reshape(nk, hk, d)
        v = vc[table.long()].reshape(nk, hk, d)
        query = torch.randn(qn, h, d, device='cuda', dtype=torch.bfloat16, generator=generator).transpose(0, 1)[None]
        orders = dict(natural=held.natural_order(need), held=screened['orders']['gated'])
        order = torch.from_numpy(orders['held']).to(device='cuda', dtype=torch.long)
        backend = TritonPermutation(order)
        lists = {name: fa.block_sparse_tensors(torch.from_numpy(held.support_for_order(need, value))[None].to('cuda'), q_block=64)
                 for name, value in orders.items()}
        adapter = VllmMethodAdapter(['full_attention'], arm='allkept', lifecycle='request_clear')
        if adapter.splits != 2:
            raise RuntimeError('held benchmark requires unchanged alias2')
        adapter.paged = dict(k=kc, v=vc, table=table, nk=nk)
        kvk, kvv, scale = k.transpose(0, 1)[None], v.transpose(0, 1)[None], d**-.5

        def consume(q, name):
            return adapter.sparse_lists(None, q, kvk, kvv, lists[name], scale)

        def torch_total():
            return held.scatter_output(consume(held.gather_query(query, order), 'held'), order)

        def triton_total():
            return backend.scatter(consume(backend.gather(query), 'held'))

        errors = {}
        # Full consumer oracles are separate from exact permutation qualification.
        for name, fn, mask_name in (('natural_q64', lambda: consume(query, 'natural'), 'natural'),
                                   ('held_torch', torch_total, 'held'), ('held_triton', triton_total, 'held')):
            got = fn()
            if not torch.isfinite(got).all().item():
                raise RuntimeError('nonfinite consumer output')
            mask = torch.from_numpy(held.original_row_support(need, orders[mask_name])).to('cuda').repeat_interleave(page, -1)
            max_abs, max_ref = 0., 0.
            for head in range(h):
                scores = query[0, head].float() @ k[:, head//(h//hk)].float().T*scale
                ref = scores.masked_fill(~mask[head], -torch.inf).softmax(-1) @ v[:, head//(h//hk)].float()
                if not torch.isfinite(ref).all().item():
                    raise RuntimeError('nonfinite IEEE FP32 masked reference')
                max_abs = max(max_abs, (got[0, :, head].float()-ref).abs().max().item())
                max_ref = max(max_ref, ref.abs().max().item())
            errors[name] = held.qualified_error(max_abs, max_ref)
        held_query = backend.gather(query)
        output_template = consume(query, 'natural')
        backend._exact(backend.scatter(output_template), held.scatter_output(output_template, order))
        backend.require_qualified()
        funcs = dict(natural_q64=lambda: consume(query, 'natural'), held_torch=torch_total, held_triton=triton_total,
                     held_consumer_only=lambda: consume(held_query, 'held'),
                     torch_permutation=lambda: (held.gather_query(query, order), held.scatter_output(output_template, order)),
                     triton_permutation=lambda: (backend.gather(query), backend.scatter(output_template)))
        builds_before = adapter.calls['split_list_builds']
        medians = _time_functions(torch, funcs, repeats)
        if adapter.calls['split_list_builds'] != builds_before:
            raise RuntimeError('timed/warm region rebuilt qualified held lists')
        records.append(dict(nominal_length_bin=length, prefix_tokens=pt*page, samples=repeats, errors=errors,
                            gpu_ms=medians, exact_permutation_oracle=True,
                            triton_over_torch=medians['held_triton']/medians['held_torch'],
                            held_triton_over_natural=medians['held_triton']/medians['natural_q64'],
                            optimistic_break_even={name: held.break_even(medians['natural_q64'], medians['held_consumer_only'], medians[name+'_permutation'])
                                                   for name in ('torch', 'triton')},
                            counters=dict(adapter.calls), actual_k_stride=list(kc.stride()),
                            actual_q_token_major_stride=list(query.transpose(1, 2).stride()),
                            required_prefix_support_covered=True, random_physical_page_order=True,
                            moved_rows=screened['moved_rows']))
        del query, cache, kc, vc, k, v, kvk, kvv, lists, adapter, backend, held_query, output_template, got, scores, ref, mask, order
    torch.cuda.synchronize()
    return dict(schema='v28_regroup_triton_held_v1', records=records, search_options=asdict(held.SEARCH),
                reserved_gpu_seconds=time.monotonic()-started,
                triton_over_torch_geomean=math.exp(statistics.mean(math.log(r['triton_over_torch']) for r in records)),
                held_triton_over_natural_geomean=math.exp(statistics.mean(math.log(r['held_triton_over_natural']) for r in records)),
                permutation_backend='triton_pointwise_token_major', exact_permutation_oracle=True,
                quality_evaluated=False, interpretation=interpretation('held'),
                scope='fixed historical held supports; synthetic native-stride QKV; full canvas kept; alias2',
                exclusions=['offline search', 'selector', 'first-use map/list/split construction', 'KV copies', 'model', 'request lifecycle', 'accuracy'],
                fp32_reference='IEEE FP32; TF32 explicitly disabled',
                permutation_cost='standalone estimate; not necessarily additive with consumer',
                cuda_graphs='none requested; eager warmed; six same-host routes rotate together')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_need_dir')
    parser.add_argument('new_report')
    parser.add_argument('--mode', choices=('permutation', 'held'), default='permutation')
    parser.add_argument('--repeats', type=int, default=None,
                        help='default: permutation 100; held 32')
    args = parser.parse_args(argv)
    dest = Path(args.new_report)
    repeats = args.repeats if args.repeats is not None else (100 if args.mode == 'permutation' else 32)
    if dest.exists() or repeats < 4:
        raise ValueError('new report and at least four repeats required')
    if args.mode == 'permutation':
        report = permutation_report(args.private_need_dir, repeats)
    else:
        report = held_report(args.private_need_dir, repeats)
    report.update(allocation_policy='each call allocates both gather and scatter outputs; same lifecycle for torch and Triton',
                  timing_parameters=dict(repeats=repeats, warm=8, rotated_order=True, mode=args.mode))
    with dest.open('x', encoding='utf8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    return report


if __name__ == '__main__':
    main()
