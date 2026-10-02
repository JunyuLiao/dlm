"""Scratch-style diagnostic (no timing claims): how much does the held map change between re-decisions within a canvas?

For the dense-prefix selector (M3 R6 DP), every decision call of one GLOBAL layer within one canvas shares the same
prefix risk table (state.identity). This records, per (layer, canvas), the skip map of the first decision and compares
every later decision of that canvas with it: changed eligible prefix tiles (either direction), as a fraction of
eligible tiles and of tiles kept by the first decision, plus the kept-set Jaccard similarity. It tests the inference
that re-decisions at calls 8, 14, ... barely change the map (docs/RESEARCH_CONTEXT.md §3).
usage: python -m scripts.v27_map_drift_diag RUN_DIR HOST UUID SEED STAGES ITEM_INDICES ARM OUT_JSONL
"""
import json
import os
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
    adapter = create_adapter('diffusion_gemma', binding['host_models'][host], device='cuda',
                             precision='bfloat16', revision=protocol['model_revision']).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    v27_substrate.install(model, name=protocol['substrate'])
    kernel = v27_substrate.prefill_kernel(protocol['substrate'])
    first, stats = {}, []
    original = dp.route

    def hooked(scores, z, reference, state, *, sensitivity=None, log_threshold, **kw):
        routing = original(scores, z, reference, state, sensitivity=sensitivity, log_threshold=log_threshold, **kw)
        try:
            sk = getattr(routing, 'skipped', None)
            if sk is None:
                sk = getattr(routing, 'skip', None)
            pt = state.prefix_tiles
            if sk is None or pt == 0:
                return routing
            elig = state.eligible.bool()
            kept = (~sk[..., :pt].bool()) & elig
            key = state.identity
            log_ref = torch.log(reference.float().clamp_min(1e-12)).reshape(-1)
            if key not in first:
                first[key] = (kept.clone(), log_ref.clone())
                stats.append(dict(kind='first'))
            else:
                k0, r0 = first[key]
                changed = int((k0 ^ kept).sum())
                inter, union = int((k0 & kept).sum()), int((k0 | kept).sum())
                stats.append(dict(kind='redecision', eligible=int(elig.sum()), kept_first=int(k0.sum()),
                                  kept_now=int(kept.sum()), changed=changed, inter=inter, union=union,
                                  dlogref=float((log_ref - r0).mean())))
        except Exception as e:
            stats.append(dict(kind='error', error=repr(e)[:200]))
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
            first.clear()
            stats.clear()
            with prefill_dense64(model, os.environ.get('V27_PREFILL_DENSE64') == '1', kernel=kernel):
                receipt = _one(adapter, row, seed, config)
            red = [s for s in stats if s['kind'] == 'redecision']
            tot = {k: sum(s[k] for s in red) for k in ('eligible', 'kept_first', 'kept_now', 'changed', 'inter', 'union')}
            out.write(json.dumps(dict(
                stage=stage, index=index, seed=seed, arm=arm, first_decisions=sum(s['kind'] == 'first' for s in stats),
                redecisions=len(red), errors=[s['error'] for s in stats if s['kind'] == 'error'][:3], totals=tot,
                changed_of_eligible=round(tot['changed'] / max(tot['eligible'], 1), 5),
                changed_of_kept=round(tot['changed'] / max(tot['kept_first'], 1), 5),
                kept_jaccard=round(tot['inter'] / max(tot['union'], 1), 5),
                redecisions_with_any_change=sum(1 for s in red if s['changed'] > 0),
                mean_dlogref=round(sum(s['dlogref'] for s in red) / max(len(red), 1), 4),
                min_dlogref=round(min((s['dlogref'] for s in red), default=0), 4),
                max_dlogref=round(max((s['dlogref'] for s in red), default=0), 4),
                calls=receipt.get('total_decoder_calls'))) + '\n')
            out.flush()


if __name__ == '__main__':
    main()
