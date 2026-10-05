"""Scratch diagnostic (not a panel, no timing claims): how many more GLOBAL prefix tiles could be skipped if the
256 canvas query rows were regrouped (per head) into the two 128-row FA4 query tiles by their tile needs.

Idea from chw/value_aware (query-row regrouping before a tile-ANDing kernel); this measures its potential on the
M3 R6 DP decisions of our pipeline. At every dense-prefix decision call it recomputes, from the stored risk table,
each row's need set (worst-row rule: a tile is needed by a row if its risk is >= threshold or +inf) and counts kept
eligible prefix tiles under:
  natural   rows 0-127 / 128-255 (what the kernel runs; checked against the router's own skip map),
  sort      rows sorted per head by need count, split in halves,
  svd       rows sorted per head by the leading singular direction of the need matrix, split in halves,
  per_row   mean per-row need (a lower bound no 128-row tile can reach).
Ported idea (credit chw, branch chw/value_aware); run in a deploy dir with the host env.
usage: python -m scripts.v27_regroup_diag RUN_DIR HOST UUID SEED STAGE ITEM_INDICES ARM OUT_JSONL
"""
import json
import sys
from pathlib import Path


def main():
    run, host, uuid, seed, stages = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(',')
    items = [int(x) for x in sys.argv[6].split(',')]
    arm = sys.argv[7]
    out = open(sys.argv[8], 'a', encoding='utf-8')
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       host, uuid, stage=stages[0])
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_substrate, v27_dense_prefix as dp
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    import os
    adapter = create_adapter('diffusion_gemma', binding['host_models'][host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    v27_substrate.install(model, name=protocol['substrate'])
    kernel = v27_substrate.prefill_kernel(protocol['substrate'])
    acc = []
    original = dp.route

    def union_count(need, order, size=128):
        # need [R, PT] bool for one head; order: row permutation; tiles of `size` rows
        g = need[order]
        return int(sum(g[i:i + size].any(0).sum() for i in range(0, g.shape[0], size)))

    def hooked(scores, z, reference, state, *, sensitivity=None, log_threshold, **kw):
        routing = original(scores, z, reference, state, sensitivity=sensitivity, log_threshold=log_threshold, **kw)
        try:
            b, h, qb, pt, _ = state.lognorm.shape
            nq = scores.shape[2]
            if b != 1 or qb != 2 or nq != 256 or pt == 0 or kw.get('risk_topk') is not None:
                return routing
            hk = reference.shape[1]
            log_ref = torch.log(reference.float().clamp_min(1e-12))[0]               # [HK]
            kh = torch.arange(h, device=log_ref.device) // (h // hk)
            ln = state.lognorm[0]                                                     # [H,2,PT,128]
            risk = ln - log_ref[kh][:, None, None, None]
            if sensitivity is not None:
                risk = risk + torch.log(sensitivity.float()[0]).view(1, 2, 1, 128)
            need = risk >= float(log_threshold)          # +inf (first support / bad rows) counts as needed
            elig = state.eligible[0].bool()                                           # [H,2,PT]
            need = need & elig[..., None]
            need = need.permute(0, 1, 3, 2).reshape(h, 256, pt)                      # rows 0..255
            elig_any = elig.any(1)                                                    # [H,PT] candidates
            nat = sor = svd = 0
            fine = {64: [0, 0, 0], 32: [0, 0, 0]}   # natural, sort, svd at finer query tiles
            per_row = 0.0
            for i in range(h):
                n = need[i]
                cnt = n.sum(1)
                ident = torch.arange(256, device=n.device)
                o_sort = torch.argsort(cnt, stable=True)
                m = n.float()
                m = m - m.mean(0, keepdim=True)
                try:
                    _, _, v = torch.linalg.svd(m, full_matrices=False)
                    o_svd = torch.argsort(m @ v[0], stable=True)
                except Exception:
                    o_svd = o_sort
                nat += union_count(n, ident)
                sor += union_count(n, o_sort)
                svd += union_count(n, o_svd)
                for size in (64, 32):
                    fine[size][0] += union_count(n, ident, size) * size / 128
                    fine[size][1] += union_count(n, o_sort, size) * size / 128
                    fine[size][2] += union_count(n, o_svd, size) * size / 128
                per_row += float(cnt.float().mean()) * 2
            eligible_slots = int(elig.sum())                                          # (head, qtile, tile) slots
            sk = getattr(routing, 'skipped', None)
            if sk is None:
                sk = getattr(routing, 'skip', None)
            router_kept = int((~sk[0, :, :, :pt].bool() & elig).sum()) if sk is not None else None
            acc.append(dict(pt=pt, eligible=eligible_slots, natural=nat, sort=sor, svd=svd, per_row=round(per_row, 1),
                            q64_natural=fine[64][0], q64_sort=fine[64][1], q64_svd=fine[64][2],
                            q32_natural=fine[32][0], q32_sort=fine[32][1], q32_svd=fine[32][2],
                            router_kept=router_kept))
        except Exception as e:  # never break generation
            acc.append(dict(error=repr(e)[:200]))
        return routing

    dp.route = hooked
    for stage in stages:
      protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                         host, uuid, stage=stage)
      ids = protocol['ids'][stage]
      config = configs[stage][arm]
      for index in items:
          row = rows[ids[index]]
          v27_substrate.set_local(model, v27_substrate.local_mode_for(config))
          acc.clear()
          with prefill_dense64(model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel=kernel):
              receipt = _one(adapter, row, seed, config)
          ok = [a for a in acc if 'error' not in a]
          tot = {k: sum(a[k] for a in ok) for k in ('eligible', 'natural', 'sort', 'svd', 'q64_natural', 'q64_sort', 'q64_svd', 'q32_natural', 'q32_sort', 'q32_svd')}
          tot['per_row'] = round(sum(a['per_row'] for a in ok), 1)
          check = sum(1 for a in ok if a['router_kept'] is not None and a['router_kept'] == a['natural'])
          out.write(json.dumps(dict(stage=stage, index=index, seed=seed, arm=arm, decisions=len(ok),
                                    errors=[a['error'] for a in acc if 'error' in a][:3],
                                    natural_matches_router=f'{check}/{len(ok)}', totals=tot,
                                    kept_fraction={k: round(tot[k] / max(tot['eligible'], 1), 4)
                                                   for k in ('natural', 'sort', 'svd', 'q64_natural', 'q64_sort', 'q64_svd', 'q32_natural', 'q32_sort', 'q32_svd', 'per_row')},
                                    calls=receipt.get('total_decoder_calls'))) + '\n')
          out.flush()


if __name__ == '__main__':
    main()
