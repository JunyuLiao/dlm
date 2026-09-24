"""Phase A2 + Phase B: same-state/same-support diagnostic on REAL layer states.

Runs one real native-dense trajectory (the actual generation output is
untouched -- every attention call still returns the true dense answer so the
model's own decoding proceeds normally) and, as a side channel at two
instrumented layers (one local/sliding, one global/full -- bounded breadth,
not all 30 layers, per the v6 instruction that small captures suffice), shadow
-evaluates the numerical-reuse machinery on the SAME real q/k/v/mask/cache
seen by that call. This is diagnostics only: capped generation length, never
an AIME quality result, and the real dense trajectory is never teacher-forced
or fed back a degraded state.

Zero-temporal-approximation checks (Phase A2):
  - all-kept: cached_executor/reference oracle with log_threshold=-inf must
    reproduce the plain dense softmax over native-legal support exactly
    (Cell A, below) -- this is a kernel/oracle correctness check, not a new
    predictor.
  - kernel-vs-oracle at score_period=1 (real threshold): resampled from the
    SAME real captures, run through both the Triton cached_executor and the
    Torch reference oracle; the existing GPU tests only cover synthetic QKV,
    this extends the same comparison to real activations.

Phase B five cells, per instrumented (layer, call) with a natural stale-score
age (a shadow ScoreCache with score_period=8, matching production M1):
  A: all native-legal keys,      current scores
  B: legacy (Junyu) support,     current scores
  C: M1 stale-score/current-V retained support, CURRENT scores
  D: same support as C,          CACHED (stale) scores  [== the M1 routing
     call's own output; production M1 does exactly this]
  E: all native-legal keys,      CACHED (stale) scores
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from dllm.attention.blasst.core import _attention_validity
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from experiments.numerical_qk_reuse.cache import Identity, ScoreCache
from experiments.numerical_qk_reuse import reference as oracle


def _rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def true_dense(q, k, v, mask, *, scaling, is_causal, **_kwargs):
    """Faithful reimplementation of the installed sdpa_attention_forward:
    GQA repeat, then torch SDPA with the same is_causal derivation. Verified
    against transformers/integrations/sdpa_attention.py (Transformers 5.11)
    on the remote host; used ONLY as this diagnostic's untouched pass-through
    so the real generation is bit-faithful to a plain dense run."""
    h, hk = q.shape[1], k.shape[1]
    if h != hk:
        k = k.repeat_interleave(h // hk, dim=1)
        v = v.repeat_interleave(h // hk, dim=1)
    causal = q.shape[2] > 1 and mask is None and bool(is_causal)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.,
                                         scale=scaling, is_causal=causal)
    return out.transpose(1, 2).contiguous(), None


def raw_scores(q, k, mask, scale):
    hk = k.shape[1]
    repeated = k.repeat_interleave(q.shape[1] // hk, dim=1)
    scores = torch.matmul(q, repeated.transpose(-1, -2)) * scale
    if mask is not None and mask.dtype != torch.bool:
        scores = scores + mask[..., :k.shape[-2]]
    return scores.float()


def never_skip_bitmap(scores, q_tile=128, k_tile=64):
    b, h, nq, nk = scores.shape
    qb, kt = (nq + q_tile - 1) // q_tile, (nk + k_tile - 1) // k_tile
    return torch.zeros((b, h, qb, kt), dtype=torch.bool, device=scores.device)


class Diagnostic:
    def __init__(self, adapter, thresholds, layers, score_period=8):
        self.thresholds = thresholds
        self.layers = set(layers)
        self.native = ScoreCache(score_period, 1)
        self.legacy = ScoreCache(score_period, 1)
        self.projections = Projections()
        self.canvas = self.step = self.epoch = -1
        self.sources, self.handles = {}, []
        self.rows: list[dict[str, Any]] = []
        self.kernel_samples: list[dict[str, Any]] = []
        for name, module in adapter.model.named_modules():
            if type(module).__name__ == 'DiffusionGemmaEncoderModel':
                self.handles.append(module.register_forward_pre_hook(self._invalidate))
            if adapter.is_blasst_attention_module(name, module):
                self.handles.append(module.register_forward_pre_hook(self._identify, with_kwargs=True))
        if not self.handles:
            raise ValueError('No qualified native DG hooks')

    def begin_step(self, canvas, step):
        if canvas != self.canvas:
            self.native.clear(); self.legacy.clear()
        self.canvas, self.step = canvas, step

    def _invalidate(self, *_):
        self.epoch += 1
        self.native.clear(); self.legacy.clear(); self.sources.clear()

    def _identify(self, module, args, kwargs):
        cache = kwargs.get('past_key_values', args[3] if len(args) > 3 else None)
        if cache is None or getattr(cache, 'is_compileable', False):
            return
        layer = int(module.layer_idx)
        source = cache.layers[layer]
        self.sources[layer] = (source.values, cache.get_seq_length(), source.keys.shape[-2])

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        output = true_dense(q, k, v, mask, scaling=scaling, is_causal=is_causal)
        layer = int(module.layer_idx)
        if layer not in self.layers or layer not in self.sources or mask is not None:
            return output
        prefix_v, absolute, prefix = self.sources[layer]
        b, h, nq, d = q.shape
        hk = k.shape[1]
        scale = float(scaling) if scaling is not None else d ** -.5
        scores_raw = raw_scores(q, k, mask, scale)
        native_valid = _attention_validity(mask, q, k, is_causal=bool(is_causal), sliding_window=None)
        legacy_valid = _attention_validity(mask, q, k, is_causal=bool(is_causal), sliding_window=sliding_window)
        current_native = scores_raw.masked_fill(~native_valid, -torch.inf).contiguous()
        current_legacy = scores_raw.masked_fill(~legacy_valid, -torch.inf).contiguous()

        identity = Identity(0, self.canvas, self.epoch, layer, b, h, hk, nq, k.shape[-2], d,
                            absolute, absolute - prefix, k.shape[-2], scale,
                            str(q.dtype), str(q.device), ('diag',), 0)
        native_plan = self.native.plan(identity, self.step)
        legacy_plan = self.legacy.plan(identity, self.step)
        if native_plan.score_refresh:
            self.native.publish_scores(identity, self.step, current_native)
        if legacy_plan.score_refresh:
            self.legacy.publish_scores(identity, self.step, current_legacy)
        stale_native = self.native.get(identity).scores
        stale_legacy = self.legacy.get(identity).scores
        score_age = self.step - self.legacy.get(identity).score_step

        matrix = self.projections.get(layer, hk, d, 'gaussian', 32, 1729, v.device)
        current_v = v.float()
        projected = torch.matmul(current_v, matrix)
        valid_any = legacy_valid.reshape(b, hk, h // hk, nq, k.shape[-2]).any((2, 3))
        ref = (current_v.square().sum(-1).masked_fill(~valid_any, 0.).sum(-1) /
               valid_any.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)

        threshold = float(self.thresholds['local' if sliding_window else 'global']['log_threshold'])
        m1_route = oracle.attention(stale_legacy, v, projected, ref, log_threshold=threshold)
        never_skip = never_skip_bitmap(current_native)

        # skipped=all-False means "never drop"; oracle.attention derives its
        # own per-tile eligibility from each score tensor's own finite
        # pattern, so this alone recovers the "keep every legal key" cells.
        cell_a = oracle.attention(current_native, v, skipped=never_skip)
        cell_b = oracle.attention(current_legacy, v, skipped=never_skip)
        # Same retained tiles as the M1 routing decision (legality is a
        # function of the shared validity mask, not the score value, so
        # oracle.attention's internally-recomputed eligibility over
        # current_legacy agrees with the one over stale_legacy).
        cell_c = oracle.attention(current_legacy, v, skipped=m1_route.skipped)
        cell_d = m1_route  # identical support as C, cached scores: production M1's own output.
        cell_e = oracle.attention(stale_native, v, skipped=never_skip)

        def rel_l2(x, y):
            diff = (x.float() - y.float()).flatten()
            base = y.float().flatten().norm()
            return float(diff.norm() / base.clamp_min(1e-12))

        self.rows.append(dict(
            canvas=self.canvas, decoder_call=self.step, layer=layer,
            kind='local' if sliding_window else 'global',
            prefix=prefix, absolute=int(absolute), nq=nq, nk=int(k.shape[-2]),
            score_age=int(score_age),
            native_legal_pairs=int(native_valid.sum()), legacy_legal_pairs=int(legacy_valid.sum()),
            a_vs_e_rel_l2=rel_l2(cell_a.output, cell_e.output),
            b_vs_c_rel_l2=rel_l2(cell_b.output, cell_c.output),
            c_vs_d_rel_l2=rel_l2(cell_c.output, cell_d.output),
            retained_tiles_c=int((m1_route.eligible & ~m1_route.skipped).sum()),
            eligible_tiles_c=int(m1_route.eligible.sum()),
        ))
        if len(self.kernel_samples) < 6 and score_age == 0:
            self.kernel_samples.append(dict(scores=stale_legacy.clone(), v=v.clone(),
                                            projected=projected.clone(), ref=ref.clone(),
                                            threshold=threshold, layer=layer, canvas=self.canvas, call=self.step))
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def kernel_vs_oracle(samples):
    from experiments.numerical_qk_reuse.cached_executor import attention as triton_attention
    rows = []
    for sample in samples:
        triton_out = triton_attention(sample['scores'], sample['v'].contiguous(),
                                      sample['projected'].contiguous(), sample['ref'].contiguous(),
                                      log_threshold=sample['threshold'])
        oracle_out = oracle.attention(sample['scores'], sample['v'], sample['projected'], sample['ref'],
                                      log_threshold=sample['threshold'])
        diff = (triton_out.output.float() - oracle_out.output.float()).flatten()
        base = oracle_out.output.float().flatten().norm()
        rows.append(dict(layer=sample['layer'], canvas=sample['canvas'], call=sample['call'],
                         rel_l2=float(diff.norm() / base.clamp_min(1e-12)),
                         skip_agree=bool(torch.equal(triton_out.skipped, oracle_out.skipped))))
    return rows


@contextmanager
def observe_steps(model, diag):
    """Feeds diag.begin_step(canvas_index, canvas_local_call) exactly like
    runner.CanvasCalls: a new canvas starts when the native schedule step
    (which counts DOWN within a canvas) increases relative to the previous
    call."""
    original = model._denoising_step
    had = '_denoising_step' in model.__dict__
    saved = model.__dict__.get('_denoising_step')
    state = {'canvas': -1, 'local_call': -1, 'last_schedule_step': None}

    def step(this, **kwargs):
        schedule_step = int(kwargs['cur_step'])
        if state['last_schedule_step'] is None or schedule_step >= state['last_schedule_step']:
            state['canvas'] += 1
            state['local_call'] = -1
        state['local_call'] += 1
        state['last_schedule_step'] = schedule_step
        diag.begin_step(state['canvas'], state['local_call'])
        return original(**kwargs)

    model._denoising_step = MethodType(step, model)
    try:
        yield
    finally:
        if had:
            model._denoising_step = saved
        else:
            del model._denoising_step


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-name", default="T_s50")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    from dllm.models import GenerationRequest, create_adapter

    manifest = {row['id']: row for row in _rows(args.manifest)}
    frozen = json.loads(args.policy.read_text(encoding='utf-8'))
    thresholds = frozen['policies'][args.policy_name]
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda',
                             precision='bfloat16', revision=args.revision).load()
    report = {'schema': 'numerical_reuse_phaseAB_diagnostic_v1', 'ids': {}, 'kernel_vs_oracle': []}
    for id_ in args.ids:
        row = manifest[id_]
        binding = _install_dense(adapter)
        diag = Diagnostic(adapter, thresholds, args.layers)
        binding.runtime.attention_override = diag
        try:
            request = GenerationRequest(prompt=row['prompt'], max_new_tokens=args.max_new_tokens,
                                        temperature=0.0, seed=42, extra={'thinking': True})
            with observe_steps(adapter.model, diag):
                adapter.generate(request)
        finally:
            binding.close()
            diag.close()
        report['ids'][id_] = diag.rows
        report['kernel_vs_oracle'].extend(kernel_vs_oracle(diag.kernel_samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding='utf-8')
    print(f"wrote {args.output} with {sum(len(v) for v in report['ids'].values())} instrumented calls")


if __name__ == "__main__":
    main()
