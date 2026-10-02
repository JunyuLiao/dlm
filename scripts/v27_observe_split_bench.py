"""Kernel-level cost of the observation call (canvas call 1) at the GLOBAL decode geometry (256 queries, 16 heads,
2 KV heads, head_dim 512, bidirectional, bf16), no end-to-end claim:
  fused       the current fused observation (dense output + summaries + tail, our Triton kernel)
  obs_only    the same kernel with output=False (summaries + tail only; v27 observe_carried)
  fa4_sparse  the official FA4 block-sparse call on a keep map with the given prefix keep fraction (64-row lists)
  fa4_dense   FA4 all-kept (the dense baseline's call)
CUDA-event medians of ROUNDS interleaved rounds after warm-up; random inputs (these kernels do no data-dependent
skipping). usage: python -m scripts.v27_observe_split_bench OUT.json KEYS[,KEYS...] KEEP[,KEEP...]
"""
import json
import math
import statistics
import sys

import torch

ROUNDS, WARM = 20, 3


def main():
    out_path = sys.argv[1]
    keys_list = [int(x) for x in sys.argv[2].split(',')]
    keeps = [float(x) for x in sys.argv[3].split(',')]
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary
    from experiments.numerical_qk_reuse.v27_consumer64 import fused_observe
    rows = []
    for keys, keep in zip(keys_list, keeps):
        g = torch.Generator(device='cuda').manual_seed(keys)
        nq, h, hk, d = 256, 16, 2, 512
        with torch.inference_mode():
            q = torch.randn(1, nq, h, d, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
            k = torch.randn(1, keys, hk, d, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
            v = torch.randn(1, keys, hk, d, device='cuda', dtype=torch.bfloat16, generator=g).transpose(1, 2)
            z = torch.randn(1, hk, keys, 32, device='cuda', generator=g)
            pt = (keys - nq) // 64
            qb, kt = math.ceil(nq / 128), math.ceil(keys / 64)
            summary = allocate_summary(1, h, qb, kt, pt, 32, 'cuda', ('bench',))
            kept = torch.rand(1, h, 2 * qb, kt, device='cuda', generator=g) < keep
            kept[..., 0] = True
            kept[..., pt:] = True                                  # canvas / boundary tail always kept
            lists = v27_fa4.block_sparse_tensors(kept, q_block=64)
            scale = d ** -.5
            fns = {
                'fused': lambda: fused_observe(q, k, v, z, scale, pt, summary, splits=2, mu=True, mu_precision='bf16'),
                'obs_only': lambda: fused_observe(q, k, v, z, scale, pt, summary, splits=2, mu=True,
                                                  mu_precision='bf16', output=False),
                'fa4_sparse': lambda: v27_fa4.sparse_lists(q, k, v, lists, scale),
                'fa4_dense': lambda: v27_fa4.dense(q, k, v, scale),
            }
            names = list(fns)
            for _ in range(WARM):
                for n in names:
                    fns[n]()
            ev = {n: [] for n in names}
            for r in range(ROUNDS):
                for n in names[r % len(names):] + names[:r % len(names)]:
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    fns[n]()
                    b.record()
                    ev[n].append((a, b))
            torch.cuda.synchronize()
            ms = {n: round(statistics.median(a.elapsed_time(b) for a, b in ev[n]), 4) for n in names}
        row = dict(keys=keys, prefix_keep=keep, kept_fraction=round(float(kept.float().mean()), 4), ms=ms,
                   call1_current=ms['fused'], call1_observe_carried=round(ms['obs_only'] + ms['fa4_sparse'], 4))
        row['ratio'] = round(row['call1_observe_carried'] / row['call1_current'], 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rounds=ROUNDS, rows=rows),
              open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
