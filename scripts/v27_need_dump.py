"""Scratch diagnostic (no timing): dump the per-row need matrices of real dense-prefix decisions for offline
query-row grouping analysis (``scripts/v27_regroup_offline.py``).

At every EVERY-th decision (cap MAX per request) of the given arm, the per-row need of every prefix tile is
recomputed from the stored risk table under the router's worst-row rule (risk >= threshold or +inf), intersected
with the router's eligibility, and saved bit-packed: need [H, 256, PT] (np.packbits along PT), plus the executed
128-row keep map's prefix part [H, 2, PT]. No prompts, tokens or outputs are saved.
usage: python -m scripts.v27_need_dump RUN_DIR HOST UUID SEED STAGES ITEM_INDICES ARM OUT_DIR [EVERY] [MAX]
"""
import json
import os
import sys
from pathlib import Path


def main():
    run, host, uuid, seed, stages = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(',')
    items = [int(x) for x in sys.argv[6].split(',')]
    arm, out_dir = sys.argv[7], Path(sys.argv[8])
    every = int(sys.argv[9]) if len(sys.argv) > 9 else 4
    cap = int(sys.argv[10]) if len(sys.argv) > 10 else 16
    out_dir.mkdir(parents=True, exist_ok=True)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       host, uuid, stage=stages[0])
    import numpy as np
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_substrate, v27_dense_prefix as dp
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    v27_substrate.install(model, name=protocol['substrate'])
    kernel = v27_substrate.prefill_kernel(protocol['substrate'])
    original = dp.route
    ctx = dict(n=0, saved=0, tag='')

    def hooked(scores, z, reference, state, *, sensitivity=None, log_threshold, **kw):
        r = original(scores, z, reference, state, sensitivity=sensitivity, log_threshold=log_threshold, **kw)
        b, h, qb, pt, _ = state.lognorm.shape
        if b != 1 or qb != 2 or scores.shape[2] != 256 or pt == 0 or kw.get('risk_topk') is not None:
            return r
        ctx['n'] += 1
        if (ctx['n'] - 1) % every or ctx['saved'] >= cap:
            return r
        hk = reference.shape[1]
        kh = torch.arange(h, device=reference.device) // (h // hk)
        risk = state.lognorm - torch.log(reference.float().clamp_min(1e-12))[:, kh][:, :, None, None, None]
        if sensitivity is not None:
            risk = risk + torch.log(sensitivity.float()).view(b, 1, qb, 1, 128)
        need = (risk >= float(log_threshold)) & state.eligible.bool()[..., None]       # [1,H,2,PT,128]
        need = need[0].permute(0, 1, 3, 2).reshape(h, 256, pt)
        kept128 = (r.eligible & ~r.skipped)[0, :, :, :pt]
        np.savez_compressed(out_dir / f"{ctx['tag']}_d{ctx['n']:03d}.npz",
                            need=np.packbits(need.cpu().numpy(), axis=-1), pt=pt,
                            kept128=np.packbits(kept128.cpu().numpy(), axis=-1),
                            eligible=np.packbits(state.eligible[0].bool().cpu().numpy(), axis=-1))
        ctx['saved'] += 1
        return r

    dp.route = hooked
    log = open(out_dir / 'dump.jsonl', 'a', encoding='utf-8')
    for stage in stages:
        protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json',
                                                           run / 'manifests', host, uuid, stage=stage)
        ids, config = protocol['ids'][stage], configs[stage][arm]
        for index in items:
            row = rows[ids[index]]
            v27_substrate.set_local(model, v27_substrate.local_mode_for(config))
            ctx.update(n=0, saved=0, tag=f'{stage.split("_")[-1]}_i{index}_s{seed}')
            with prefill_dense64(model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel=kernel):
                receipt = _one(adapter, row, seed, config)
            log.write(json.dumps(dict(stage=stage, index=index, seed=seed, arm=arm, decisions=ctx['n'],
                                      saved=ctx['saved'], calls=receipt.get('total_decoder_calls'))) + '\n')
            log.flush()


if __name__ == '__main__':
    main()
