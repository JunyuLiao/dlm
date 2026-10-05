"""v27 in-model correctness of the FA4 sparse consumer: during real requests of a BOUND panel (piecewise substrate),
the first N FA4 consume calls of each arm are re-evaluated with (a) the 64-row Triton consumer and (b) an FP32
masked-softmax reference on the SAME q/k/v and the SAME method keep map (kept = eligible & ~skipped, Q128 x KV64
tiles). Reports max/mean abs error of FA4 and Triton vs the reference, and that the FA4 block lists hold exactly
the kept tile count. Numbers only.
usage: python -m scripts.v27_consume_check --run-dir DIR --host IP --gpu-uuid UUID --stage S --dataset D
           --index I --budget N --checks K --arm A [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--run-dir', '--host', '--gpu-uuid', '--stage', '--dataset'):
        p.add_argument(name, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--budget', type=int, default=512)
    p.add_argument('--checks', type=int, default=6)
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args(argv)
    run = Path(a.run_dir)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       a.host, a.gpu_uuid, stage=a.stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import integration, v27_fa4
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_consumer64 import consume64
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    if protocol.get('substrate', 'eager') != 'eager':
        from experiments.numerical_qk_reuse import v27_substrate
        v27_substrate.install(adapter.model)
    results = []
    original = integration.Attention._consume

    def checked(self, q, k, v, skipped, eligible, scale, window, causal):
        result = original(self, q, k, v, skipped, eligible, scale, window, causal)
        if self.consumer == 'fa4' and len(results) < a.checks and q.shape[-1] == 512:
            with torch.no_grad():
                kept = (eligible & ~skipped)
                h, nq, nk = q.shape[1], q.shape[2], k.shape[2]
                rep = h // k.shape[1]
                mask = kept.repeat_interleave(128, 2)[:, :, :nq].repeat_interleave(64, 3)[..., :nk]
                s = torch.einsum('bhqd,bhkd->bhqk', q.float(), k.float().repeat_interleave(rep, 1)) * scale
                ref = torch.einsum('bhqk,bhkd->bhqd', s.masked_fill(~mask, float('-inf')).softmax(-1),
                                   v.float().repeat_interleave(rep, 1))
                del s
                fa4 = result.output.float()                                   # [1,H,Q,D]
                tri = consume64(q, k, v, skipped, eligible, scale, splits=2).float().transpose(1, 2)
                lists = v27_fa4.block_sparse_tensors(kept)
                results.append(dict(layer=int(getattr(self, '_last_layer', -1)), keys=nk,
                                    kept_fraction=round(float(kept.float().mean()), 4),
                                    fa4_max_abs=round(float((fa4 - ref).abs().max()), 5),
                                    fa4_mean_abs=round(float((fa4 - ref).abs().mean()), 7),
                                    triton_max_abs=round(float((tri - ref).abs().max()), 5),
                                    fa4_vs_triton_max_abs=round(float((fa4 - tri).abs().max()), 5),
                                    list_count_matches=bool(int(lists.full_block_cnt.sum()) == int(kept.sum()))))
                del ref, mask
        return result
    integration.Attention._consume = checked
    row = dict(rows[protocol['ids'][a.dataset][a.index]], generation_budget=a.budget)
    try:
        for arm in a.arm:
            results.clear()
            with prefill_dense64(adapter.model, os.environ.get('V27_PREFILL_DENSE64') == '1'):
                _one(adapter, row, protocol['seeds'][0], configs[a.dataset][arm])
            worst = max((r['fa4_max_abs'] for r in results), default=None)
            print(json.dumps(dict(arm=arm, checks=len(results), worst_fa4_max_abs=worst,
                                  all_lists_match=all(r['list_count_matches'] for r in results), detail=results)),
                  flush=True)
    finally:
        integration.Attention._consume = original


if __name__ == '__main__':
    main()
