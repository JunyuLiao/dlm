"""Kernel-level timing on identical real states (no end-to-end claim): GLOBAL sparse FA4 calls under the 128-row
M3 map, the q64 map and the q64r (within-block regrouped) map, each at FA4 num_splits 1 and 2, plus the dense
all-kept call and the per-decision map/list costs.

A real request runs in the given arm; at every dense-prefix decision the route's risk state is kept, and at the
first FA4 consume of that decision the same Q/K/V are timed under every map (CUDA events, interleaved rotated
order, median of ROUNDS after warm-up). The generation itself is untouched (the timed calls' outputs are discarded).
q64r includes the Q gather and output scatter. Idea of regrouping credited to chw/value_aware.
usage: python -m scripts.v27_q64_bench RUN_DIR HOST UUID SEED STAGES ITEM_INDICES ARM OUT_JSONL [MAX_PER_ITEM]
"""
import json
import os
import statistics
import sys
from pathlib import Path

ROUNDS, WARM = 15, 2


def main():
    run, host, uuid, seed, stages = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(',')
    items = [int(x) for x in sys.argv[6].split(',')]
    arm = sys.argv[7]
    out = open(sys.argv[8], 'a', encoding='utf-8')
    cap = int(sys.argv[9]) if len(sys.argv) > 9 else 60
    from scripts.v21_run import validate_inputs
    protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json', run / 'manifests',
                                                       host, uuid, stage=stages[0])
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_substrate, v27_fa4, v27_dense_prefix as dp
    from experiments.numerical_qk_reuse.integration import Attention
    from experiments.numerical_qk_reuse.runner import _one
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    adapter = create_adapter('diffusion_gemma', binding['host_models'][host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    v27_substrate.install(model, name=protocol['substrate'])
    kernel = v27_substrate.prefill_kernel(protocol['substrate'])
    fwd = v27_fa4.load()
    bst = v27_fa4.block_sparse_tensors
    pending, records, ctx = {}, [], {}
    original_route, original_consume = dp.route, Attention._consume

    def route(scores, z, reference, state, *, sensitivity=None, log_threshold, **kw):
        r = original_route(scores, z, reference, state, sensitivity=sensitivity, log_threshold=log_threshold, **kw)
        if kw.get('risk_topk') is None and kw.get('risk_budget') is None:
            pending[id(r.skipped)] = dict(skipped=r.skipped, eligible=r.eligible, state=state, ref=reference,
                                          sens=sensitivity, thr=float(log_threshold), nq=scores.shape[2])
            while len(pending) > 32:
                pending.pop(next(iter(pending)))
        return r

    def timed(fns):
        names = list(fns)
        for _ in range(WARM):
            for n in names:
                fns[n]()
        ev = {n: [] for n in names}
        for r in range(ROUNDS):
            rot = names[r % len(names):] + names[:r % len(names)]
            for n in rot:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fns[n]()
                b.record()
                ev[n].append((a, b))
        torch.cuda.synchronize()
        return {n: round(statistics.median(a.elapsed_time(b) for a, b in ev[n]), 4) for n in names}

    def imbalance(kept):
        cnt = kept.sum(-1).float()                    # per (head, query tile)
        return round(float(cnt.max() / cnt.mean().clamp_min(1)), 3)

    def measure(q, k, v, p, scale):
        st, ref, sens, thr, nq = p['state'], p['ref'], p['sens'], p['thr'], p['nq']
        skipped, eligible = p['skipped'], p['eligible']
        kept128 = eligible & ~skipped
        kept64 = dp.refine_q64(st, skipped, eligible, ref, sens, thr, nq)
        rr = dp.refine_q64(st, skipped, eligible, ref, sens, thr, nq, regroup=True)
        if kept64 is None or rr is None:
            return None
        kept64r, order = rr
        lists = {'128': bst(kept128), '64': bst(kept64, q_block=64), '64r': bst(kept64r, q_block=64)}
        allkept = v27_fa4._allkept(q.shape[1], -(-q.shape[2] // 128), -(-k.shape[2] // 64), q.device)
        qs, ks, vs = v27_fa4._layout(q, k, v)
        idx = order[..., None].expand(-1, -1, -1, q.shape[-1])

        def call(lst, splits, perm=False):
            if not perm:
                return fwd(qs, ks, vs, softmax_scale=scale, causal=False, block_sparse_tensors=lst, num_splits=splits)[0]
            qp = torch.gather(q, 2, idx).transpose(1, 2)
            o = fwd(qp, ks, vs, softmax_scale=scale, causal=False, block_sparse_tensors=lst, num_splits=splits)[0]
            return torch.empty_like(o).transpose(1, 2).scatter_(2, idx, o.transpose(1, 2)).transpose(1, 2)

        def perm_only():
            qp = torch.gather(q, 2, idx)
            return torch.empty_like(qp).scatter_(2, idx, qp)

        fns = {'dense_s1': lambda: call(allkept, 1)}
        for s in (1, 2):
            fns[f'k128_s{s}'] = lambda s=s: call(lists['128'], s)
            fns[f'k64_s{s}'] = lambda s=s: call(lists['64'], s)
            fns[f'k64r_s{s}'] = lambda s=s: call(lists['64r'], s, perm=True)
        fns['perm_only'] = perm_only
        fns['build128'] = lambda: bst(kept128)
        fns['refine64_build'] = lambda: bst(dp.refine_q64(st, skipped, eligible, ref, sens, thr, nq), q_block=64)
        fns['refine64r_build'] = lambda: bst(dp.refine_q64(st, skipped, eligible, ref, sens, thr, nq,
                                                           regroup=True)[0], q_block=64)
        times = timed(fns)
        dense = call(allkept, 1).float()
        o128, o64, o64r = call(lists['128'], 1).float(), call(lists['64'], 1).float(), call(lists['64r'], 1, True).float()
        o64s2 = call(lists['64'], 2).float()
        pt = st.lognorm.shape[3]
        elig = int(eligible[..., :pt].sum()) * 2 if pt else 0
        frac = lambda m: round(float(m[..., :pt].sum()) / max(elig, 1), 4)
        return dict(nk=int(k.shape[2]), pt=pt, kept_prefix={'128': round(float(kept128[..., :pt].sum()) * 2 / max(elig, 1), 4),
                                                            '64': frac(kept64), '64r': frac(kept64r)},
                    kept_all={'128': round(float(kept128.float().mean()), 4), '64': round(float(kept64.float().mean()), 4),
                              '64r': round(float(kept64r.float().mean()), 4)},
                    imbalance={'128': imbalance(kept128), '64': imbalance(kept64), '64r': imbalance(kept64r)},
                    ms=times,
                    max_abs_vs_dense={'128': round(float((o128 - dense).abs().max()), 5),
                                      '64': round(float((o64 - dense).abs().max()), 5),
                                      '64r': round(float((o64r - dense).abs().max()), 5)},
                    split2_vs_split1_64=round(float((o64s2 - o64).abs().max()), 6))

    def consume(self, q, k, v, skipped, eligible, scale, window, causal):
        p = pending.pop(id(skipped), None)
        if p is not None and p['skipped'] is skipped and self.consumer == 'fa4' and ctx['n'] < cap:
            torch.cuda.synchronize()
            try:
                with torch.inference_mode():
                    rec = measure(q, k, v, p, scale)
                if rec is not None:
                    records.append(rec)
                    ctx['n'] += 1
            except Exception as e:   # never break generation
                records.append(dict(error=repr(e)[:300]))
            torch.cuda.synchronize()
        return original_consume(self, q, k, v, skipped, eligible, scale, window, causal)

    dp.route, Attention._consume = route, consume
    for stage in stages:
        protocol, binding, rows, configs = validate_inputs(run / 'protocol.json', run / 'binding.json',
                                                           run / 'manifests', host, uuid, stage=stage)
        ids, config = protocol['ids'][stage], configs[stage][arm]
        for index in items:
            row = rows[ids[index]]
            v27_substrate.set_local(model, v27_substrate.local_mode_for(config))
            records.clear()
            pending.clear()
            ctx['n'] = 0
            with prefill_dense64(model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel=kernel):
                receipt = _one(adapter, row, seed, config)
            ok = [r for r in records if 'error' not in r]
            summary = {}
            for key in ok[0]['ms'] if ok else ():
                summary[key] = round(statistics.median(r['ms'][key] for r in ok), 4)
            out.write(json.dumps(dict(stage=stage, index=index, seed=seed, arm=arm, decisions=len(ok),
                                      errors=[r['error'] for r in records if 'error' in r][:3],
                                      median_ms=summary,
                                      kept_prefix={m: round(statistics.mean(r['kept_prefix'][m] for r in ok), 4)
                                                   for m in ('128', '64', '64r')} if ok else {},
                                      calls=receipt.get('total_decoder_calls'), records=ok)) + '\n')
            out.flush()


if __name__ == '__main__':
    main()
