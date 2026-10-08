"""Synthetic H100 diagnostics, deliberately separate from model/accuracy results."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from experiments.numerical_qk_reuse.v31_value_selectors import SELECTORS, select, full_support, masked_output
from experiments.numerical_qk_reuse.v31_value_kernels import statistics_cuda


def elapsed(operation, repeats=5):
    operation()  # compile/warm outside timing
    torch.cuda.synchronize()
    measurements = []
    result = None
    for _ in range(repeats):
        before = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        measurements.append(1000*(time.perf_counter()-before))
    return sorted(measurements)[len(measurements)//2], result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--prefixes', default='2048,32768,65536,131072')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(1729)
    rows = []
    for prefix in map(int, args.prefixes.split(',')):
        n, h, hk, d = 256, 16, 2, 512
        q = torch.randn(1, h, n, d, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(1, hk, prefix+n, d, dtype=torch.bfloat16, device='cuda')
        v = torch.randn_like(k)
        # A deterministic Gaussian32 bank, identical across variants in this cell.
        from experiments.diffusion_gemma_jl_output_aware.projections import Projections
        projections = Projections()
        r = projections.get(5, hk, d, 'gaussian', 32, 1729, 'cuda')
        z = torch.matmul(v.float(), r)
        nu = v.float().square().sum(-1).mean(-1).sqrt()[0]
        ms, stats = elapsed(lambda: statistics_cuda(q, k, z, nu, d**-.5, prefix//64), 3)
        alpha, c, full, _ = full_support(stats)
        k_tiles = min(8192//64, prefix//64)
        exact_mask = None
        exact_error = None
        for selector in SELECTORS:
            # Threshold is a DIAGNOSTIC ONLY, never a calibrated target threshold.
            start = time.perf_counter()
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            select_ms, (mask, receipt) = elapsed(lambda: select(stats, selector, budget=k_tiles, threshold=.01), 1 if 'v3b' in selector else 3)
            selector_peak_bytes = torch.cuda.max_memory_allocated()
            selected = mask.reshape(h*2, -1)
            estimate = masked_output(alpha, c, selected)
            error = (estimate-full).norm(dim=-1)/stats.nu
            if selector == SELECTORS[3]:
                exact_mask = selected
                exact_error = float(error.max())
            row = dict(prefix=prefix, selector=selector, stats_ms=ms, select_ms=select_ms,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                selector_incremental_peak_bytes=selector_peak_bytes-baseline_bytes,
                kept_prefix_tiles=int(mask[..., :prefix//64].sum()),
                eligible_prefix_tiles=h*2*(prefix//64), max_relative_sketch_error=float(error.max()),
                mean_retained_mass=float((alpha*selected[..., None]).sum(1).mean()),
                projection_manifest=projections.manifest, gpu_seconds=time.perf_counter()-start, **receipt)
            if exact_mask is not None and selector == SELECTORS[4]:
                row['mask_jaccard_vs_exact'] = float((selected & exact_mask).sum()/(selected | exact_mask).sum())
                row['sketch_error_ratio_vs_exact'] = float(error.max())/exact_error
            rows.append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(dict(kind='synthetic_selector_diagnostic', records=rows), indent=2)+'\n')
            print(json.dumps({k: row[k] for k in ('prefix', 'selector', 'stats_ms', 'select_ms', 'max_relative_sketch_error')}), flush=True)
        del stats, alpha, c, full, q, k, v, z
    # A marker for this diagnostic only, never the whole scientific study.
    output.with_suffix('.complete.json').write_text(json.dumps({'status': 'complete', 'kind': 'synthetic_only'})+'\n')


if __name__ == '__main__':
    main()
