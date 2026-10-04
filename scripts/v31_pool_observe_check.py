"""GPU check of round 5's pool observation (VllmMethodAdapter._pool_observe) on synthetic GLOBAL calls (DiffusionGemma
geometry: 16 query / 2 KV heads, head_dim 512, a 256-row canvas, 64-key tiles, vLLM's paged cache layout).

Checks, per context length:
  z      the block-sparse observation's prefix-tile log-mass equals the dense FA4 observation's on every pool tile
         (both are FA4's own FP32 scores of the same tile), and is -inf outside the pool;
  output the pool call's output matches an FP32 reference attention over the pool tiles (+ canvas tiles);
  time   median CUDA-event time of the dense observation (observe_dense), the pool observation, and the held sparse
         call on a k-tile list (the step's ordinary cost), over warm repetitions.
Prints one JSON row per check. usage: python v31_pool_observe_check.py OUT.jsonl [--keys 65536,131072] [--pool 4]
"""
import argparse
import json
import statistics

import torch

H, HK, D, CANVAS, TILE, PAGE = 16, 2, 512, 256, 64, 64
G, QB = H // HK, CANVAS // 128


def geometry(keys, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed + keys)
    nk, prefix = keys, keys - CANVAS
    kt, pt = -(-keys // TILE), (keys - CANVAS) // TILE
    nblocks = -(-keys // PAGE) + 16
    kv = torch.randn(nblocks, HK, PAGE, 2 * D, device='cuda', generator=g).to(torch.bfloat16)
    key_cache, value_cache = kv.transpose(1, 2).split(D, dim=-1)                          # [NB, PAGE, HK, D]
    table = torch.randperm(nblocks, device='cuda', generator=g)[:-(-keys // PAGE)].to(torch.int32)
    q = torch.randn(1, H, CANVAS, D, device='cuda', generator=g).to(torch.bfloat16)
    k = key_cache[table.long()].reshape(-1, HK, D)[:keys].transpose(0, 1).unsqueeze(0).contiguous()
    v = value_cache[table.long()].reshape(-1, HK, D)[:keys].transpose(0, 1).unsqueeze(0).contiguous()
    return dict(nk=nk, prefix=prefix, kt=kt, pt=pt, key_cache=key_cache, value_cache=value_cache, table=table, q=q, k=k,
                v=v, scale=D ** -0.5)


def reference(geo, kept):
    """FP32 attention of every head over its kept tiles; kept [H, QB, KT] -> [1, CANVAS, H, D]."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.empty(CANVAS, H, D, device='cuda')
        for h in range(H):
            kf, vf = geo['k'][0, h // G].float(), geo['v'][0, h // G].float()
            s = (geo['q'][0, h].float() @ kf.T) * geo['scale']
            m = kept[h].repeat_interleave(128, 0).repeat_interleave(TILE, 1)[:, :geo['nk']]
            out[:, h] = torch.softmax(s.masked_fill(~m, float('-inf')), -1) @ vf
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    return out[None]


def timed(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return round(statistics.median(ts), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--keys', default='65536,131072')
    ap.add_argument('--pool', type=int, default=4)
    ap.add_argument('--k', type=int, default=4096)
    a = ap.parse_args()
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse import vllm_adapter as va
    from experiments.numerical_qk_reuse.v31_fa4_observe import observe_dense
    v27_fa4.load()
    rows = []
    for keys in (int(x) for x in a.keys.split(',')):
        geo = geometry(keys)
        ad = va.VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='mage', mage_select='fa4',
                                  mage_granularity='qblock_max', mage_k=a.k, mage_select_step=1,
                                  mage_reselect_trigger=0.5, mage_pool=a.pool, kv_copy_backend='triton',
                                  merge_backend='triton')
        ad.begin_request()
        ad.paged = dict(k=geo['key_cache'], v=geo['value_cache'], nk=geo['nk'], prefix=geo['prefix'], table=geo['table'])
        pt, kt = geo['pt'], geo['kt']
        gen = torch.Generator(device='cpu').manual_seed(keys)
        ptiles = min(pt, a.pool * a.k // TILE)
        pool = torch.zeros(1, H, QB, kt, dtype=torch.bool)
        for h in range(H):
            for b in range(QB):
                pool[0, h, b, torch.randperm(pt, generator=gen)[:ptiles]] = True
        pool[..., pt:] = True
        pool = pool.cuda()
        z_dense = torch.empty(H, QB, pt, 128, device='cuda')
        qd, kd, vd = geo['q'].transpose(1, 2), geo['k'].transpose(1, 2), geo['v'].transpose(1, 2)
        observe_dense(qd, kd, vd, geo['scale'], z_dense)
        out, z = ad._pool_observe(geo['q'], geo['scale'], pool, CANVAS, geo['prefix'])
        inpool = pool[0, :, :, :pt][..., None].expand(H, QB, pt, 128)
        dz = (z - z_dense).abs()[inpool]
        outside = z[~inpool]
        ref = reference(geo, pool[0])
        err = (out.float() - ref).abs()
        kept = torch.zeros_like(pool)
        kept[..., pt:] = True
        kept[0, :, :, :pt] = torch.rand(H, QB, pt, device='cuda').argsort(-1).argsort(-1) < a.k // TILE
        held = v27_fa4.block_sparse_tensors(kept)
        row = dict(keys=keys, pool_tiles_per_unit=ptiles, k_tiles=a.k // TILE,
                   z_max_abs_diff=float(dz.max()), z_outside_all_neg_inf=bool(torch.isneginf(outside).all()),
                   out_max_abs_err=float(err.max()), out_mean_abs_err=float(err.mean()),
                   out_finite=bool(torch.isfinite(out).all()),
                   ms_dense_observe=timed(lambda: observe_dense(qd, kd, vd, geo['scale'], z_dense)),
                   ms_pool_observe=timed(lambda: ad._pool_observe(geo['q'], geo['scale'], pool, CANVAS, geo['prefix'])),
                   ms_held_sparse=timed(lambda: ad.sparse_lists(None, geo['q'], geo['k'], geo['v'], held, geo['scale'])))
        row['ok'] = row['z_max_abs_diff'] < 1e-3 and row['z_outside_all_neg_inf'] and row['out_max_abs_err'] < 5e-2
        print(json.dumps(row), flush=True)
        rows.append(row)
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
