"""Same-state GLOBAL cost controls; independent diagnostic, never request speed.

Reuse official varlen FA4 and existing adapter copy/alias2/merge implementations.
Synthetic QKV is the default, not evidence of the cause of a real-model slowdown.
Optional --state accepts an explicitly supplied PRIVATE torch tensor snapshot;
this tool does not capture a model or certify that snapshot's provenance.
Only numeric measurements/settings are exported. CUDA imports are lazy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from scripts.v29_component_timing import (rotate, zero_counter_deltas,
                                          measure_components, vllm_compile_guard)


@dataclass(frozen=True)
class Settings:
    prefixes: tuple = (32768,)
    keep_fraction: float = .25
    repeats: int = 32
    warm: int = 8
    seed: int = 2914
    max_model_len: int = 106496
    # Qualified BEFORE execution; no post-result tuning in this diagnostic.
    output_atol: float = .02
    output_rtol: float = .02
    summary_atol: float = .02
    summary_rtol: float = .02
    dp_atol: float = .0002
    dp_rtol: float = .0002
    merge_atol: float = .01
    merge_rtol: float = .01

    def validate(self):
        if (not self.prefixes or len(set(self.prefixes)) != len(self.prefixes)
                or any(type(p) is not int or p < 64 or p+256 > self.max_model_len for p in self.prefixes)
                or type(self.repeats) is not int or self.repeats < 4
                or type(self.warm) is not int or self.warm < 1
                or type(self.seed) is not int or self.seed < 0
                or type(self.max_model_len) is not int or self.max_model_len < 320
                or not math.isfinite(self.keep_fraction) or not 0 < self.keep_fraction <= 1):
            raise ValueError('bounded prefix, positive density/warm and at least four repeats required')
        if any(not math.isfinite(v) or v <= 0 for k, v in asdict(self).items() if k.endswith(('atol', 'rtol'))):
            raise ValueError('positive fixed tolerances required')
        return self


def frozen_settings(settings, input_mode):
    settings.validate()
    if input_mode not in ('synthetic', 'supplied_private_snapshot'):
        raise ValueError('unsupported input mode')
    values = asdict(settings)
    values['prefixes'] = list(settings.prefixes)
    return dict(schema='v29_homogeneous_cost_spec_v1', diagnostic_only=True,
                input_mode=input_mode, settings=values,
                geometry=dict(query_rows=256, query_heads=16, kv_heads=2, head_dim=512,
                              cache_page_tokens=64, query_block=128, alias_splits=2, rank=32),
                scale=1.0, native=dict(fa_version=4, causal=True, dynamic_causal=False,
                                      num_splits=0, max_seqlen_q=256, softcap=0.0),
                sparse=dict(causal=False, inner_num_splits=1, alias_splits=2),
                full_canvas_and_boundary_kept=True, support_construction_timed=False,
                selector_control=dict(sensitivity=1.0, log_threshold=-3.8735877018001794,
                                      clock='static same-state control; no sampling or adaptive history'),
                cuda_graphs_requested=False, selector_stream='serialized diagnostic control',
                quality_evaluated=False, request_speed_claim_allowed=False)


def fixed_support(prefix, fraction, seed, heads=16, rows=256, q_block=128):
    if (type(prefix) is not int or prefix < 64 or rows != 256 or q_block != 128
            or heads < 1 or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('positive prefix and fixed Q256/Q128 support required')
    pt, kt = prefix//64, (prefix+rows+63)//64
    rng = np.random.default_rng(seed)
    kept = rng.random((1, heads, rows//q_block, kt)) < fraction
    kept[..., 0] = True                         # first legal support is never empty
    kept[..., pt:] = True                      # partial boundary plus full canvas
    return kept


def check_split_builds(before, after):
    if before != after:
        raise RuntimeError('held alias lists rebuilt in timed region')


def snapshot_contract(query, cache, table, prefix, scale, dynamic):
    """CPU array/tensor metadata contract. Device transfer happens only afterwards."""
    if (tuple(query.shape) != (256, 16, 512) or tuple(cache.shape[1:]) != (2, 64, 1024)
            or len(cache.shape) != 4 or type(prefix) is not int or prefix < 64
            or scale != 1.0 or dynamic is not False):
        raise ValueError('private snapshot must be denoising GLOBAL at official scale1 and native geometry')
    pages = np.asarray(table)
    required = (prefix+256+63)//64
    if (pages.ndim != 1 or pages.dtype.kind not in 'iu' or len(pages) != required
            or (pages < 0).any() or (pages >= cache.shape[0]).any()
            or len(set(pages.tolist())) != len(pages)):
        raise ValueError('distinct in-range physical pages for the exact logical token span required')


def dense_prefix_reference(logmass, mu):
    """Independent float64 dense-prefix equations, vectorized over rows/tiles.

    Returns log(delta norm), dense cumulative log mass and mean. Used on actual
    emitted summaries to qualify DP-build, separate from summary FP32 tolerance.
    """
    z, mu = np.asarray(logmass, np.float64), np.asarray(mu, np.float64)
    if z.ndim != 2 or mu.shape[:2] != z.shape or mu.ndim != 3 or not np.isfinite(z).all() or not np.isfinite(mu).all():
        raise ValueError('finite fully active [rows,tiles] summaries required')
    maximum = z.max(axis=1, keepdims=True)
    weights = np.exp(z-maximum)
    cumulative = np.cumsum(weights, axis=1)
    if (cumulative <= 0).any():
        raise ValueError('unstable independent DP oracle; fail rather than infer')
    means = np.cumsum(weights[..., None]*mu, axis=1)/cumulative[..., None]
    previous = np.concatenate((np.zeros_like(means[:, :1]), means[:, :-1]), axis=1)
    delta = weights[..., None]/cumulative[..., None]*(mu-previous)
    with np.errstate(divide='ignore'):
        risk = np.log(np.linalg.norm(delta, axis=-1))
    risk[:, 0] = np.inf
    return risk, np.log(cumulative)+maximum, means


def tail_decision_reference(scores, sketch, previous_mass, previous_mu, reference, threshold):
    """Independent float64 dense-state tail scan, including a partial last tile."""
    scores, sketch = np.asarray(scores, np.float64), np.asarray(sketch, np.float64)
    previous, mean = np.asarray(previous_mass, np.float64).copy(), np.asarray(previous_mu, np.float64).copy()
    if (scores.ndim != 2 or scores.shape[0] != 256 or scores.shape[1] < 1
            or sketch.shape != (scores.shape[1], 32) or previous.shape != (256,)
            or mean.shape != (256, 32) or reference <= 0 or not math.isfinite(reference)
            or not math.isfinite(threshold) or not all(np.isfinite(x).all() for x in (scores,sketch,previous,mean))):
        raise ValueError('finite actual tail and dense-prefix state required')
    skipped = []
    for lo in range(0, scores.shape[1], 64):
        block = scores[:, lo:lo+64]
        maximum = block.max(-1)
        weights = np.exp(block-maximum[:, None])
        total = weights.sum(-1)
        mu = (weights/total[:, None]) @ sketch[lo:lo+64]
        z = maximum+np.log(total)
        combined = np.logaddexp(previous, z)
        alpha = np.exp(z-combined)
        delta = alpha[:, None]*(mu-mean)
        with np.errstate(divide='ignore'):
            risk = np.log(np.linalg.norm(delta, axis=-1))-math.log(reference)
        skipped.append(risk.reshape(2,128).max(-1) < threshold)
        mean = np.exp(previous-combined)[:, None]*mean+alpha[:, None]*mu
        previous = combined
    return np.stack(skipped, axis=-1)


def _close(torch, got, want, atol, rtol):
    # Legal LSE/log-risk infinities are checked separately. Attention is finite.
    if not torch.isfinite(got).all().item() or not torch.isfinite(want).all().item():
        raise RuntimeError('nonfinite attention/reference before timing')
    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=rtol)
    return dict(max_abs=(got.float()-want.float()).abs().max().item(), atol=atol, rtol=rtol)


def _case(torch, fa, prefix, settings, generator, snapshot, supplied=None):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe
    from experiments.numerical_qk_reuse import v27_dense_prefix as dp
    from experiments.diffusion_gemma_jl_output_aware.projections import Projections

    n, h, hk, d, page, scale = 256, 16, 2, 512, 64, 1.0
    nk, pt, kt = prefix+n, prefix//64, (prefix+n+63)//64
    if supplied is None:
        cache = torch.randn(kt+3, hk, page, 2*d, device='cuda', dtype=torch.bfloat16, generator=generator)*.1
        query = torch.randn(n, h, d, device='cuda', dtype=torch.bfloat16, generator=generator)*.1
        table = torch.randperm(kt+3, device='cuda', generator=generator)[:kt].to(torch.int32)
    else:
        query, cache, table = [supplied[key].to('cuda') for key in ('query', 'kv_cache', 'block_table')]
    kc, vc = cache.transpose(1, 2).split(d, -1)
    q = query.transpose(0, 1)[None]
    k = kc[table.long()].reshape(-1, hk, d)[:nk].transpose(0, 1)[None].contiguous()
    v = vc[table.long()].reshape(-1, hk, d)[:nk].transpose(0, 1)[None].contiguous()
    support = fixed_support(prefix, settings.keep_fraction, settings.seed+prefix)
    masks = dict(allkept=torch.ones_like(torch.from_numpy(support), device='cuda'),
                 sparse=torch.from_numpy(support).to('cuda'))
    lists = {name: fa.block_sparse_tensors(mask, q_block=128) for name, mask in masks.items()}
    adapters = {(copy, merge): VllmMethodAdapter(['full_attention'], arm='allkept', lifecycle='request_clear',
                  kv_copy_backend=copy, merge_backend=merge) for copy in ('torch', 'triton') for merge in ('torch', 'triton')}
    for adapter in adapters.values():
        adapter.paged = dict(k=kc, v=vc, table=table, nk=nk)
        for value in lists.values():
            adapter._split(value)
    cu = torch.tensor([0, n], device='cuda', dtype=torch.int32)
    used = torch.tensor([nk], device='cuda', dtype=torch.int32)
    dynamic = torch.tensor([False], device='cuda')
    descale = torch.ones(1, hk, device='cuda', dtype=torch.float32)
    native_output = torch.empty_like(query)

    def native():
        flash_attn_varlen_func(query, kc, vc, cu_seqlens_q=cu, max_seqlen_q=n,
            max_seqlen_k=settings.max_model_len, seqused_k=used, block_table=table[None],
            softmax_scale=scale, causal=True, dynamic_causal=dynamic, num_splits=0,
            fa_version=4, softcap=0.0, q_descale=descale, k_descale=descale,
            v_descale=descale, out=native_output)
        return native_output[None]

    fwd = fa.load()
    leaf_adapter = adapters['torch', 'torch']

    def partials(name):
        split = leaf_adapter._split(lists[name])
        tables = table[None].expand(2, -1).contiguous()
        lengths = torch.full((2,), nk, device='cuda', dtype=torch.int32)
        return fwd(q.transpose(1, 2).expand(2, -1, -1, -1), kc, vc, softmax_scale=scale,
                   causal=False, page_table=tables, seqused_k=lengths,
                   block_sparse_tensors=split, num_splits=1, return_lse=True)[:2]

    def merge_standard(o, lse):
        weights = torch.softmax(lse, dim=0).permute(0, 2, 1)[..., None]
        return (o.float()*weights).sum(0, keepdim=True).to(o.dtype)

    errors = {}
    # Independent IEEE FP32 attention oracle, one head at a time; no full GQA copy.
    outputs = {'native': native().clone()}
    for name in lists:
        for (copy, merge), adapter in adapters.items():
            outputs[f'{name}_{copy}_{merge}'] = adapter.sparse_lists(None, q, k, v, lists[name], scale)
    max_errors = {name: 0. for name in outputs}
    for head in range(h):
        scores = q[0, head].float() @ k[0, head//8].float().T
        dense_ref = scores.softmax(-1) @ v[0, head//8].float()
        for name, output in outputs.items():
            if name.startswith('sparse'):
                mask = masks['sparse'][0, head].repeat_interleave(128, 0).repeat_interleave(64, 1)[:, :nk]
                reference = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v[0, head//8].float()
            else:
                reference = dense_ref
            error = _close(torch, output[0, :, head], reference, settings.output_atol, settings.output_rtol)
            max_errors[name] = max(max_errors[name], error['max_abs'])
    errors['attention_max_abs'] = max_errors
    for name in lists:
        o, lse = partials(name)
        if not torch.isfinite(o).all().item() or torch.isnan(lse).any().item() or torch.isposinf(lse).any().item() or (~torch.isfinite(lse)).all(0).any().item():
            raise RuntimeError('invalid alias partial/LSE before timing')
        errors[name+'_merge'] = _close(torch, adapters['torch', 'triton']._merge_alias2(o, lse),
                                        merge_standard(o, lse), settings.merge_atol, settings.merge_rtol)
    del outputs

    functions = {'native_varlen_dynamic': native}
    for copy in ('torch', 'triton'):
        adapter = adapters[copy, 'torch']
        def copy_buffer(full, adapter=adapter):
            if full:
                adapter.buffers.clear()
            return adapter._buffers(0, kc, vc, table, prefix, n)
        full_buffer = copy_buffer(True)
        for key, expected in (('k', k), ('v', v)):
            if not torch.equal(full_buffer[key], expected):
                raise RuntimeError('bit-exact copy oracle failed before timing')
        copy_buffer(False)
        for key, expected in (('k', k), ('v', v)):
            if not torch.equal(full_buffer[key], expected):
                raise RuntimeError('bit-exact tail copy oracle failed before timing')
        functions[f'copy_full_{copy}'] = lambda fn=copy_buffer: fn(True)
        functions[f'copy_tail_{copy}'] = lambda fn=copy_buffer: fn(False)

    writebacks = {key: torch.empty_like(native_output)[None] for key in adapters}
    for (copy, merge), adapter in adapters.items():
        for name in lists:
            def component(full, adapter=adapter, name=name, output=writebacks[copy, merge]):
                if full:
                    adapter.buffers.clear()
                b = adapter._buffers(0, kc, vc, table, prefix, n)
                output.copy_(adapter.sparse_lists(None, q, b['k'], b['v'], lists[name], scale))
                return output
            # Exercise the complete path (including writeback) before timing.
            got = component(True)
            expected = merge_standard(*partials(name))
            _close(torch, got, expected, settings.merge_atol, settings.merge_rtol)
            functions[f'full_{name}_{copy}_{merge}'] = lambda fn=component: fn(True)
            functions[f'tail_{name}_{copy}_{merge}'] = lambda fn=component: fn(False)
    stored_partials = {name: partials(name) for name in lists}
    for name in lists:
        functions[f'partial_fa4_{name}'] = lambda name=name: partials(name)
        functions[f'merge_torch_{name}'] = lambda name=name: merge_standard(*stored_partials[name])
        functions[f'merge_triton_{name}'] = lambda name=name: adapters['torch', 'triton']._merge_alias2(*stored_partials[name])

    # Same native V, actual Gaussian32 matrix convention; no artificial need data.
    matrix = Projections().get(5, hk, d, 'gaussian', 32, 1729, q.device)
    sketch = torch.empty(1, hk, nk, 32, device='cuda', dtype=torch.float32)
    norm = torch.empty(1, hk, nk, device='cuda', dtype=torch.float32)
    valid = torch.ones(1, hk, nk, device='cuda', dtype=torch.bool)
    def projection(start):
        x = v[..., start:, :].float()
        torch.matmul(x, matrix, out=sketch[..., start:, :])
        norm[..., start:] = x.norm(dim=-1).square()
        return (norm.masked_fill(~valid, 0.).sum(-1)/valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
    reference = projection(0)
    expected_sketch = torch.matmul(v.float(), matrix)
    _close(torch, sketch, expected_sketch, settings.dp_atol, settings.dp_rtol)
    projection(pt*64)
    _close(torch, sketch, expected_sketch, settings.dp_atol, settings.dp_rtol)
    functions['projection_full'] = lambda: projection(0)
    functions['projection_tail'] = lambda: projection(pt*64)
    summary = allocate_summary(1, h, 2, kt, pt, 32, 'cuda', ('homogeneous',))
    def observe(output=True):
        return fused_observe(q, k, v, sketch, scale, pt, summary, splits=2,
                             mu=True, mu_precision='bf16', output=output)
    observed, tail = observe()
    sensitivity = torch.ones(1, n, device='cuda', dtype=torch.float32)
    state = dp.build(summary, n, kt, hk)
    threshold = -3.8735877018001794
    def route():
        return dp.route(tail, sketch, reference, state, sensitivity=sensitivity,
                        log_threshold=threshold, key_offset=pt*64)
    routing = route()
    if routing.invalid_tiles.any().item() or not routing.eligible.all().item():
        raise RuntimeError('invalid/unexpected structurally inactive DP route')
    for head in range(h):
        scores = q[0, head].float() @ k[0, head//8].float().T
        ref_output = scores.softmax(-1) @ v[0, head//8].float()
        _close(torch, observed[0, :, head], ref_output, settings.output_atol, settings.output_rtol)
        _close(torch, tail[0, head], scores[:, pt*64:], settings.dp_atol, settings.dp_rtol)
        block_scores = scores[:, :pt*64].reshape(2, 128, pt, 64).permute(0, 2, 1, 3)
        wanted_z = block_scores.logsumexp(-1)
        wanted_mu = block_scores.softmax(-1) @ sketch[0, head//8, :pt*64].reshape(pt, 64, 32)
        _close(torch, summary.z[0, head], wanted_z, settings.summary_atol, settings.summary_rtol)
        _close(torch, summary.mu[0, head], wanted_mu, settings.summary_atol, settings.summary_rtol)
        # Exact DP-build equations on emitted summaries; transfer is untimed.
        zs = summary.z[0, head].permute(0, 2, 1).reshape(n, pt).cpu().numpy()
        mus = summary.mu[0, head].permute(0, 2, 1, 3).reshape(n, pt, 32).cpu().numpy()
        lognorm, mass, means = dense_prefix_reference(zs, mus)
        expected_logn = torch.from_numpy(lognorm.reshape(2, 128, pt).transpose(0, 2, 1).copy()).to('cuda').float()
        torch.testing.assert_close(state.lognorm[0, head], expected_logn, atol=settings.dp_atol, rtol=settings.dp_rtol)
        _close(torch, state.previous[0, head], torch.from_numpy(mass[:, -1].reshape(2, 128)).to('cuda'), settings.dp_atol, settings.dp_rtol)
        _close(torch, state.projected[0, head], torch.from_numpy(means[:, -1].reshape(2, 128, 32)).to('cuda'), settings.dp_atol, settings.dp_rtol)
        # Independent decision controls use the emitted, already qualified state.
        risk = lognorm-math.log(reference[0, head//8].item())
        expected_skip = (risk.reshape(2, 128, pt).max(1) < threshold)
        actual_skip = routing.skipped[0, head, :, :pt].cpu().numpy()
        if not np.array_equal(expected_skip, actual_skip):
            raise RuntimeError('DP prefix decision oracle mismatch before timing')
        expected_tail = tail_decision_reference(tail[0,head].cpu().numpy(),
            sketch[0,head//8,pt*64:].cpu().numpy(), state.previous[0,head].reshape(n).cpu().numpy(),
            state.projected[0,head].reshape(n,32).cpu().numpy(), reference[0,head//8].item(), threshold)
        if not np.array_equal(expected_tail, routing.skipped[0,head,:,pt:].cpu().numpy()):
            raise RuntimeError('DP tail decision oracle mismatch before timing')
    if not summary.active.eq(1).all().item() or not summary.bad.eq(0).all().item():
        raise RuntimeError('summary finite-input flags failed')
    # Observe-only must agree, not silently benchmark a different operation.
    no_output, tail_only = observe(False)
    if no_output is not None or not torch.equal(tail_only, tail):
        raise RuntimeError('observation-only tail mismatch')
    functions['fused_observe_full_output'] = observe
    functions['fused_observe_only'] = lambda: observe(False)
    functions['dp_build_allocating'] = lambda: dp.build(summary, n, kt, hk)
    functions['dp_route_allocating'] = route
    def observation_dp_serial():
        output, current_tail = observe()
        current_state = dp.build(summary, n, kt, hk)
        dp.route(current_tail, sketch, reference, current_state, sensitivity=sensitivity,
                 log_threshold=threshold, key_offset=pt*64)
        return output
    functions['observation_dp_serial_control'] = observation_dp_serial
    torch.cuda.synchronize()
    before = {key: adapter.calls['split_list_builds'] for key, adapter in adapters.items()}
    timing_evidence = measure_components(torch, functions, repeats=settings.repeats,
                                         warm=settings.warm, snapshot=snapshot)
    timings = timing_evidence['timings']
    check_split_builds(before, {key: adapter.calls['split_list_builds'] for key, adapter in adapters.items()})
    native_ms = timings['native_varlen_dynamic']['medians']['cuda_event_ms']
    obs_ms = timings['fused_observe_full_output']['medians']['cuda_event_ms']
    return dict(prefix_tokens=prefix, canvas_tokens=n, mask_keep_fraction_actual=float(support.mean()),
                native_query_stride=list(query.stride()), native_cache_stride=list(kc.stride()),
                random_page_order_requested=supplied is None,
                summary_bytes=summary.nbytes, geometry=dict(q=256, h=16, hk=2, d=512, page=64, query_block=128),
                numerical_checks=dict(passed=True, attention_errors=errors, copy_bit_exact=True,
                    observer_full_output_and_all_prefix_summaries=True, dp_build_and_prefix_decision_oracle=True,
                    dp_tail_decision_oracle=True),
                settings_receipt=frozen_settings(settings, 'synthetic' if supplied is None else 'supplied_private_snapshot'),
                timing_evidence=timing_evidence,
                counters={f'{c}_{m}':dict(a.calls) for (c,m),a in adapters.items()},
                native_dispatch_receipt=dict(requested_num_splits=0, resolved_heuristic_splits=None,
                    resolved_split_note='not inspected or inferred from durations; native chooses its actual heuristic',
                    varlen=True, dynamic_causal_tensor_value=False, causal_dispatch=True,
                    max_seqlen_q=256, max_seqlen_k=settings.max_model_len, seqused_k=nk,
                    softmax_scale=1.0, dtype='bfloat16', fa_version=4, softcap=0.0,
                    descales='ones FP32 passed as serving; BF16 path ignores FP8 descales'),
                same_input_observer_full_over_native=obs_ms/native_ms,
                same_input_observer_minus_native_event_ms=obs_ms-native_ms,
                same_input_difference_note='different dense implementations with identical QKV/full output contract; includes custom dense-output implementation differences, not a pure added-statistics cost')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('new_report', nargs='?')
    p.add_argument('--prefix', type=int, nargs='+', default=[32768])
    p.add_argument('--keep-fraction', type=float, default=.25)
    p.add_argument('--repeats', type=int, default=32)
    p.add_argument('--warm', type=int, default=8)
    p.add_argument('--seed', type=int, default=2914)
    p.add_argument('--state', type=Path, help='private .pt tensor snapshot; no capture is performed')
    p.add_argument('--spec', type=Path, help='frozen public spec matching these CLI settings exactly')
    p.add_argument('--source-commit', help='exact deployed 40-hex source commit (externally pinned)')
    p.add_argument('--print-spec', action='store_true', help='CPU-only spec template; does not run CUDA')
    args = p.parse_args(argv)
    settings = Settings(prefixes=tuple(args.prefix), keep_fraction=args.keep_fraction,
                        repeats=args.repeats, warm=args.warm, seed=args.seed).validate()
    spec = frozen_settings(settings, 'supplied_private_snapshot' if args.state else 'synthetic')
    if args.print_spec:
        print(json.dumps(spec, indent=2, allow_nan=False))
        return spec
    if not args.new_report or not args.source_commit or not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('new report and exact externally pinned source commit required')
    dest = Path(args.new_report)
    if dest.exists() or not dest.parent.is_dir():
        raise ValueError('report must be new with an existing parent')
    if args.spec is None or json.loads(args.spec.read_text(encoding='utf-8')) != spec:
        raise ValueError('a frozen spec identical to CLI parameters is required before GPU imports')
    supplied = None
    import torch
    if args.state:
        supplied = torch.load(args.state, map_location='cpu', weights_only=True)
        if not isinstance(supplied, dict) or set(supplied) != {'query','kv_cache','block_table','prefix_tokens','softmax_scale','dynamic_causal'}:
            raise ValueError('private snapshot requires exact tensor/geometry-only schema')
        for key in ('query','kv_cache'):
            if not isinstance(supplied[key], torch.Tensor) or supplied[key].dtype != torch.bfloat16 or not supplied[key].is_contiguous():
                raise ValueError('contiguous native BF16 snapshot tensors required')
            if not torch.isfinite(supplied[key]).all().item():
                raise ValueError('finite private snapshot tensors required')
        table = supplied['block_table']
        if not isinstance(table, torch.Tensor) or table.dtype != torch.int32 or not table.is_contiguous():
            raise ValueError('contiguous CPU int32 page table required')
        snapshot_contract(supplied['query'],supplied['kv_cache'],table.numpy(),
                          supplied['prefix_tokens'],supplied['softmax_scale'],supplied['dynamic_causal'])
        if tuple(args.prefix) != (supplied['prefix_tokens'],):
            raise ValueError('frozen prefix must match the single supplied snapshot')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        raise RuntimeError('qualified SM90 GPU required')
    torch.set_float32_matmul_precision('highest')
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision='ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    generator=torch.Generator(device='cuda').manual_seed(settings.seed)
    # Activate official monitor before importing component FA4/consumer modules.
    with vllm_compile_guard() as (snapshot, monitor_receipt):
        from experiments.numerical_qk_reuse import v27_fa4 as fa
        import vllm
        import triton
        records=[_case(torch, fa, prefix, settings, generator, snapshot, supplied) for prefix in settings.prefixes]
    report=dict(schema='v29_homogeneous_cost_diagnostic_v1', source_commit=args.source_commit,
                spec=spec, records=records, body_wall_seconds=time.monotonic()-started,
                software=dict(torch=torch.__version__, cuda=torch.version.cuda, vllm=vllm.__version__, triton=triton.__version__),
                gpu=dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability())),
                scope='standalone same-input GLOBAL components; no model, sampler, request, graph replay or quality run',
                snapshot_provenance_verified=False, causal_slowdown_explanation_claim_allowed=False,
                timing='rotated independent CUDA event spans and synchronized host boundaries; host_call may include implicit blocking and is not pure launch cost',
                allocation='native out preallocated as serving; alias allocates partial/merged output plus common preallocated writeback as adapter; copy_full includes original buffer allocation, tail reuses prefix',
                controls='all-kept and fixed sparse support use both shared copy/merge backends; native heuristic num_splits=0 is not forced to S1; alias2 uses inner S1 and a generic LSE merge',
                exclusions=['QKV production/model work','sampler','online support construction and identity/clock logic',
                            'first alias-list/map construction','real async selector overlap',
                            'observation leaf summary allocation (preallocated); projection reported separately'],
                selector_scope='serialized DP controls; NOT runtime asynchronous scheduling or critical path',
                monitor_activation=monitor_receipt,
                jit_counter_scope='official monitor activated before component imports; handler entry events and real vLLM compilation/capture counters, with documented monitor coverage limits',
                aggregation_rule='never sum leaf/full/overlapping scopes into W, S or per-request overhead',
                quality_evaluated=False, request_speed_claim_allowed=False)
    with dest.open('x', encoding='utf-8') as out:
        json.dump(report,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(dict(cases=len(records), numerical_checks_passed=True, report_bytes=dest.stat().st_size)))
    return report


if __name__=='__main__':
    main()
