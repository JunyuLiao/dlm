"""v14 gate 5.2: adaptive behavior from IDENTICAL real canvas-start states.

Capture (untouched native model, fast-T controller observing): the StepSnapshot at
cur_step=48 of a chosen canvas (prefix KV, noise canvas, CPU/CUDA RNG, native
sampler/stopping/processors, controller). Per arm: restore that snapshot once and
let the NATIVE loop run to the native stable&confident stop or the 48-step cap
(same code path as generate(): each step's outputs feed the next). Not scored.

Recorded per step: native stop components (stable, confident, mean processed
entropy), accepted/renoised positions, deterministic top-1 flips, the T weights
used (s>1 rows), GLOBAL executed-tile fractions (prefix-prunable region and all
eligible tiles), support churn per layer, restored tiles (CVM), and divergence of
the arm's argmax from the native run at the same step with the s the arm had
CAUSALLY available for that row at that step. Diagnostic syncs are allowed here
(no timing is reported from this script).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch


class RecordingStop:
    """Proxy for the native stable&confident criterion; returns exactly its decision."""

    def __init__(self, inner):
        self.inner, self.rows = inner, []

    def __call__(self, argmax_canvas, logits, **kwargs):
        hist = self.inner.argmax_canvas_history
        stable = bool((hist == argmax_canvas[None]).all()) if hist is not None else False
        ent = torch.distributions.Categorical(logits=logits.float()).entropy()
        mean_ent = float(ent.mean())
        out = self.inner(argmax_canvas, logits, **kwargs)
        confident = mean_ent < self.inner.confidence_threshold
        if self.inner.stability_threshold != 1 or bool(out.all()) != (stable and confident):
            raise RuntimeError('stop proxy disagrees with the native criterion')
        self.rows.append(dict(stable=stable, confident=confident, mean_entropy=mean_ent,
                              rows_entropy_gt_0p1=int((ent > .1).sum())))
        return out

    def reset(self):
        self.inner.reset()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Tap:
    """Wraps the routed GLOBAL attention override and records executed-tile maps."""

    def __init__(self, guard, router, kind):
        self.guard, self.router, self.kind = guard, router, kind
        self.rows, self.prev, self.last_t = [], {}, None
        if kind == 'T_G':                                     # Junyu's router: capture its kernel result masks
            tap = self
            inner = router.kernel

            class K:
                def __call__(self, *a, **k):
                    r = inner(*a, **k)
                    tap.last_t = (r.skipped, r.eligible)
                    return r

                def __getattr__(self, name):
                    return getattr(inner, name)
            router.kernel = K()

    def __call__(self, module, q, k, v, mask, **kwargs):
        r = self.router
        before = dict(r.counts) if hasattr(r, 'counts') else None
        out = self.guard(module, q, k, v, mask, **kwargs)
        nq, nk = q.shape[-2], k.shape[-2]
        protect = (nk - nq) // 64
        layer = int(module.layer_idx)
        phase, skipped, eligible = 'fresh', None, None
        if self.kind == 'T_G':
            skipped, eligible = self.last_t
        else:
            d = {key: r.counts[key] - before[key] for key in ('anchors', 'ordinary', 'fallback_native')}
            phase = 'fallback' if d['fallback_native'] else 'anchor' if d['anchors'] else 'ordinary'
            if phase == 'anchor':
                skipped, eligible = r.kernel.last_masks
            elif phase == 'ordinary':
                st = r.layers[layer]
                skipped, eligible = st['skipped'], st['eligible']
        row = dict(layer=layer, phase=phase, protect=protect, nk=nk)
        if skipped is not None:
            ex = (~skipped.bool()) & eligible.bool()
            el = eligible.bool()
            row.update(executed_frac=float(ex.sum() / el.sum().clamp_min(1)),
                       prefix_executed_frac=float(ex[..., :protect].sum() / el[..., :protect].sum().clamp_min(1))
                       if protect > 0 else None)
            prev = self.prev.get(layer)
            if prev is not None and prev.shape == ex.shape:
                row['churn'] = float((prev ^ ex).sum() / el.sum().clamp_min(1))
            self.prev[layer] = ex.clone()
        self.rows.append(row)
        return out

    def __getattr__(self, name):
        return getattr(self.guard, name)


def run_to_stop(model, snap, state, runtime_router, reference):
    from experiments.value_direction_hopper.query_adaptive import observe
    from scripts.replay_harness import unwrap_sampler
    ctx = observe(model, state) if state is not None else None
    if ctx:
        ctx.__enter__()
    try:
        if runtime_router is not None and hasattr(runtime_router, 'invalidate'):
            runtime_router.invalidate()
        kw = snap.prepare(controller=state)
        stop = RecordingStop(kw['diffusion_stopping_criteria'])
        kw['diffusion_stopping_criteria'] = stop
        sampler = unwrap_sampler(kw['sampler'])
        rows, first_div, s_at_div, argmaxes = [], {}, {}, []
        prev_top = None
        for i in range(48):
            cur = 48 - i
            kw['cur_step'] = cur
            res = model._denoising_step(**kw)
            new_cur, new_arg, sc, fin = res
            acc = sampler.accepted_token_mask
            top = new_arg[0]
            s = None if state is None or state.used_weights is None else state.used_weights[0].float()
            row = dict(cur_step=cur, accepted=int(acc.sum()), renoised=int((~acc).sum()),
                       flips=None if prev_top is None else int((top != prev_top).sum()),
                       s_gt1_rows=None if s is None else int((s > 1).sum()),
                       s_mean=None if s is None else float(s.mean()), **stop.rows[-1])
            if reference is not None and i < len(reference):
                div = top != reference[i]
                row['diverged_rows_vs_native'] = int(div.sum())
                for p in div.nonzero().flatten().tolist():
                    if p not in first_div:
                        first_div[p] = i
                        s_at_div[p] = 1.0 if s is None else float(s[p])
            argmaxes.append(top.clone())
            rows.append(row)
            prev_top = top
            if bool(fin.all()):
                break
            kw.update(current_canvas=new_cur, argmax_canvas=new_arg, self_conditioning_logits=sc, finished_denoising=fin)
        final = argmaxes[-1]
        return dict(rows=rows, argmaxes=argmaxes, native_stop=bool(fin.all()), calls=len(rows), first_div=first_div,
                    s_at_div=s_at_div, final_digest=hashlib.sha256(final.cpu().numpy().tobytes()).hexdigest()[:16],
                    final=final)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--states', nargs='+', required=True, help='id:seed:canvas')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse.runner import GOLD_FIELDS, _rows
    from experiments.value_direction_hopper.query_adaptive import State
    from scripts.v14_forward_profile import arm_runtime, capture_sequence
    arms = json.loads(a.arms)
    rows_by_id = {r['id']: {k: v for k, v in r.items() if k not in GOLD_FIELDS} for r in _rows(a.manifest)}
    adapter = create_adapter('diffusion_gemma', str(a.model), device='cuda', precision='bfloat16', revision=a.revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = adapter.model
    report = dict(schema='v14_adaptive_canvas_diagnostic_v1', arms=[x['name'] for x in arms], states={})
    for spec in a.states:
        qid, seed, canvas = spec.rsplit(':', 2)
        seed, canvas = int(seed), int(canvas)
        snap = capture_sequence(adapter, rows_by_id[qid], canvas, 1, seed)[0]
        ns = SimpleNamespace(phase='v14adaptive', ids=[qid], manifest=a.manifest, policy=a.policy, model=a.model,
                             revision=a.revision, seeds=[seed])
        reference, out = None, dict(prefix_tokens=snap['absolute'], arms={})
        for arm in arms:
            with torch.inference_mode(), arm_runtime(adapter, arm, ns) as runtime:
                state = runtime['state'] if runtime else None
                router = runtime['router'] if runtime else None
                tap = None
                if runtime is not None:
                    tap = Tap(runtime['binding'].runtime.attention_override, router,
                              'T_G' if arm['condition'] == 'global_T' else 'CVM')
                    runtime['binding'].runtime.attention_override = tap
                    if hasattr(router, 'audit'):
                        router.audit, router.audit_rows = True, []
                if state is None:                             # D_native: an observing fast-T controller (no router)
                    state = State('T', None, m_ref=14.258454322814941, beta=3., gamma=.5, diagnostics=False, fast_t=True)
                r = run_to_stop(model, snap['snapshot'], state, router, reference)
                if runtime is not None:
                    runtime['binding'].runtime.attention_override = tap.guard
            if arm['name'] == 'D_native':
                reference, ref_final = r['argmaxes'], r['final']
            calls = tap.rows if tap is not None else []
            prefix_fr = [c['prefix_executed_frac'] for c in calls if c.get('prefix_executed_frac') is not None]
            churn = [c['churn'] for c in calls if 'churn' in c]
            audit = getattr(router, 'audit_rows', []) if router is not None else []
            div_s = list(r['s_at_div'].values())
            summary = dict(
                calls_to_finish=r['calls'], native_stop=r['native_stop'], cap=not r['native_stop'],
                accepted_total=sum(x['accepted'] for x in r['rows']), renoised_total=sum(x['renoised'] for x in r['rows']),
                flips_total=sum(x['flips'] or 0 for x in r['rows']),
                steps_stable_not_confident=sum(x['stable'] and not x['confident'] for x in r['rows']),
                steps_confident_not_stable=sum(x['confident'] and not x['stable'] for x in r['rows']),
                final_mean_entropy=r['rows'][-1]['mean_entropy'],
                global_calls=len(calls), phases={ph: sum(c['phase'] == ph for c in calls) for ph in sorted({c['phase'] for c in calls})},
                mean_prefix_executed_frac=statistics.mean(prefix_fr) if prefix_fr else None,
                mean_executed_frac=statistics.mean(c['executed_frac'] for c in calls if 'executed_frac' in c) if calls else None,
                mean_churn=statistics.mean(churn) if churn else None,
                restored_tiles=sum(x['restored'] for x in audit), restorations_by_step=[x['restored'] for x in audit],
                final_digest=r['final_digest'],
                final_agreement_with_native=None if reference is None else float((r['final'] == ref_final).float().mean()),
                rows_ever_diverged_from_native=len(r['first_div']) if reference is not None else None,
                diverged_rows_with_protection_available=sum(s > 1 for s in div_s),
                diverged_rows_without_protection=sum(s <= 1 for s in div_s),
                first_divergence_steps=sorted(set(r['first_div'].values())))
            out['arms'][arm['name']] = dict(summary=summary, steps=r['rows'])
            print(spec, arm['name'], {k: summary[k] for k in ('calls_to_finish', 'native_stop', 'mean_prefix_executed_frac',
                                                             'restored_tiles', 'final_agreement_with_native',
                                                             'steps_stable_not_confident')}, flush=True)
        report['states'][spec] = out
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
