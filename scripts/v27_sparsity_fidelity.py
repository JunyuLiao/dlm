"""v27: how much does a held sparse map change GLOBAL attention? (diagnostic; numbers only, no text)

Every FA4 block-sparse GLOBAL call that skips tiles (held M1/M2/M3/B calls) is shadowed, on the SAME q/k/v, by
  * FA4 with the same lists and return_lse, and FA4 with every tile kept (the exact dense reference);
  * HF-path SDPA (enable_gqa), whose difference to FA4 dense is the exact-kernel-swap envelope, i.e. the
    numerical noise any dense baseline change already introduces.
Per call it records the kept tile fraction, the attention mass the kept tiles carry per (head, query)
(exp(lse_kept - lse_all): 1.0 means nothing skipped), and the relative output error ||o_sparse - o_dense|| /
||o_dense|| next to ||o_sdpa - o_fa4|| / ||o_fa4||. The request's own output is the unshadowed sparse output, so
generation is unchanged. Dense and fused-observation calls are not shadowed.
Oracle (--oracle-every N): on every N-th shadowed call the exact per-tile attention mass is computed from the
current Q/K, and the oracle keeps, per (head, query block), the SAME NUMBER of tiles as the method but the ones with
the largest mean mass over the block's rows. oracle_mass_mean vs mass_mean separates selector quality from how
concentrated the model's attention is (a small gap: the selector is near the best any block-sparse map of that size
can do; a large gap: a better ranking would pay).
Trajectory rows (--out FILE.jsonl, numbers only): per shadowed call its canvas, denoising step within the canvas,
GLOBAL layer position, and the map's age (steps since that layer's current keep map was first used), so kept mass
can be read against step and staleness.
usage: python -m scripts.v27_sparsity_fidelity --run-dir DIR --host IP --gpu-uuid UUID --stage S --dataset D
           [--index I ...] [--budget N] [--out FILE] --arm A [--arm ...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics


def quantiles(xs, qs=(0.01, 0.1, 0.5, 0.9, 0.99)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f'p{int(q * 100):02d}': round(xs[min(len(xs) - 1, int(q * len(xs)))], 5) for q in qs}


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('--run-dir', '--host', '--gpu-uuid', '--stage', '--dataset'):
        p.add_argument(name, required=True)
    p.add_argument('--index', type=int, action='append')
    p.add_argument('--budget', type=int)
    p.add_argument('--arm', action='append', required=True)
    p.add_argument('--out')
    p.add_argument('--oracle-every', type=int, default=0)
    a = p.parse_args(argv)
    from pathlib import Path
    run = Path(a.run_dir)
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       a.host, a.gpu_uuid, stage=a.stage)
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_fa4, v27_substrate
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][a.host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    substrate = protocol.get('substrate', 'eager')
    if substrate != 'eager':
        v27_substrate.install(model, name=substrate)
    kernel = v27_substrate.prefill_kernel(substrate)
    records = []
    shadow_count = [0]
    original = v27_fa4.sparse_lists

    def oracle_mass(q, k, lists, scale, kt, qb):
        """(method mass from the same exact tile masses, oracle mass) per row, averaged; exact FP32 softmax."""
        heads, kv = q.shape[1], k.shape[1]
        group, nq, nk = heads // kv, q.shape[2], k.shape[2]
        cnt = lists.full_block_cnt[0].long()                              # [H, QB]
        idx = lists.full_block_idx[0].long()                              # [H, QB, KT]
        method_rows, oracle_rows = [], []
        for g in range(kv):
            scores = torch.matmul(q[0, g * group:(g + 1) * group].float(), k[0, g].float().T) * scale
            probs = torch.softmax(scores, -1)                             # [G, NQ, NK]
            pad = kt * 64 - nk
            if pad:
                probs = torch.nn.functional.pad(probs, (0, pad))
            tile_rows = probs.view(group, nq, kt, 64).sum(-1)            # [G, NQ, KT] mass per row per tile
            del scores, probs
            for b in range(qb):
                rows = tile_rows[:, b * 128:(b + 1) * 128]                # [G, R, KT]
                block = rows.mean(1)                                      # [G, KT]
                for j in range(group):
                    h = g * group + j
                    n = int(cnt[h, b])
                    kept = idx[h, b, :n]
                    method_rows.append(rows[j][:, kept].sum(-1))
                    top = block[j].topk(n).indices
                    oracle_rows.append(rows[j][:, top].sum(-1))
        method = torch.cat(method_rows)
        oracle = torch.cat(oracle_rows)
        return float(method.mean()), float(oracle.mean()), float(torch.quantile(oracle, 0.01))

    where = dict(canvas=-1, step=0, layer=0)
    first_seen = {}                       # layer position -> (lists object, step index when first used)
    encoder = model.model.encoder
    enc_forward, dec_forward = encoder.forward, model.forward

    def on_encoder(*x, **kw):
        where.update(canvas=where['canvas'] + 1, step=0)
        first_seen.clear()
        return enc_forward(*x, **kw)

    def on_decoder(*x, **kw):
        where['layer'] = 0
        try:
            return dec_forward(*x, **kw)
        finally:
            where['step'] += 1
    encoder.forward, model.forward = on_encoder, on_decoder

    def shadowed(q, k, v, lists, scale):
        out = original(q, k, v, lists, scale)
        heads, qb = lists.full_block_cnt.shape[1:]
        kt = -(-k.shape[2] // 64)
        kept = float(lists.full_block_cnt.sum()) / (heads * qb * kt)
        layer = where['layer']
        where['layer'] += 1
        if kept >= 1.0:
            return out
        seen = first_seen.get(layer)
        if seen is None or seen[0] is not lists:
            first_seen[layer] = seen = (lists, where['step'])
        fwd = v27_fa4.load()
        qs, ks, vs = v27_fa4._layout(q, k, v)
        tile = {} if q.shape[-1] > 256 else dict(tile_mn=(128, 64))
        o_s, lse_s = fwd(qs, ks, vs, softmax_scale=scale, causal=False, block_sparse_tensors=lists,
                         return_lse=True, **tile)[:2]
        full = v27_fa4._allkept(q.shape[1], qb, kt, q.device)
        o_d, lse_d = fwd(qs, ks, vs, softmax_scale=scale, causal=False, block_sparse_tensors=full,
                         return_lse=True, **tile)[:2]
        mass = torch.exp(lse_s.float() - lse_d.float()).flatten()
        o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
        dense = o_d.float().transpose(1, 2)
        denom = dense.norm()
        rec = dict(canvas=where['canvas'], step=where['step'], layer=layer, age=where['step'] - seen[1],
                   nk=int(k.shape[2]), kept=round(kept, 5),
                   mass_mean=float(mass.mean()), mass_p01=float(torch.quantile(mass, 0.01)),
                   mass_min=float(mass.min()),
                   err=float((o_s.float().transpose(1, 2) - dense).norm() / denom),
                   env=float((o_ref.float() - dense).norm() / denom),
                   same_as_production=bool(torch.equal(o_s, out)))
        shadow_count[0] += 1
        if a.oracle_every and shadow_count[0] % a.oracle_every == 0:
            exact, best, best_p01 = oracle_mass(q, k, lists, scale, kt, qb)
            rec.update(exact_method_mass=exact, oracle_mass_mean=best, oracle_mass_p01=best_p01)
        records.append(rec)
        return out
    v27_fa4.sparse_lists = shadowed
    try:
        for index in a.index or [0]:
            row = rows[protocol['ids'][a.dataset][index]]
            if a.budget:
                row = dict(row, generation_budget=a.budget)
            for arm in a.arm:
                config = configs[a.dataset][arm]
                if substrate != 'eager':
                    v27_substrate.set_local(model, v27_substrate.local_mode_for(config))
                records.clear()
                where.update(canvas=-1, step=0, layer=0)
                first_seen.clear()
                with prefill_dense64(model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel=kernel):
                    receipt = _one(adapter, row, protocol['seeds'][0], config)
                torch.cuda.synchronize()
                summary = dict(dataset=a.dataset, index=index, arm=arm, prompt_tokens=row.get('prompt_token_count'),
                               decoder_calls=receipt['total_decoder_calls'], output_tokens=receipt['output_tokens'],
                               shadowed_layer_calls=len(records))
                if records:
                    summary.update(
                        kept=quantiles([r['kept'] for r in records]),
                        mass_mean=quantiles([r['mass_mean'] for r in records]),
                        mass_p01=quantiles([r['mass_p01'] for r in records]),
                        mass_min=round(min(r['mass_min'] for r in records), 5),
                        err=quantiles([r['err'] for r in records]),
                        env=quantiles([r['env'] for r in records]),
                        err_over_env_median=round(statistics.median(r['err'] / max(r['env'], 1e-12)
                                                                    for r in records), 3),
                        shadow_equals_production=all(r['same_as_production'] for r in records))
                    sampled = [r for r in records if 'oracle_mass_mean' in r]
                    if sampled:
                        summary.update(
                            oracle_calls=len(sampled),
                            oracle_method_mass=quantiles([r['exact_method_mass'] for r in sampled]),
                            oracle_mass=quantiles([r['oracle_mass_mean'] for r in sampled]),
                            oracle_gap_median=round(statistics.median(r['oracle_mass_mean'] - r['exact_method_mass']
                                                                      for r in sampled), 4))
                if records:
                    by_age, by_step = {}, {}
                    for r in records:
                        by_age.setdefault(min(r['age'], 12), []).append(r['mass_mean'])
                        by_step.setdefault(min(r['step'] // 4 * 4, 32), []).append(r['mass_mean'])
                    summary.update(
                        mass_mean_by_age={k: round(statistics.mean(v), 5) for k, v in sorted(by_age.items())},
                        mass_mean_by_step4={k: round(statistics.mean(v), 5) for k, v in sorted(by_step.items())})
                print(json.dumps(summary), flush=True)
                if a.out:
                    with open(a.out, 'a', encoding='utf-8') as f:
                        for r in records:
                            f.write(json.dumps(dict(r, dataset=a.dataset, index=index, arm=arm)) + '\n')
    finally:
        v27_fa4.sparse_lists = original
        encoder.forward, model.forward = enc_forward, dec_forward


if __name__ == '__main__':
    main()
