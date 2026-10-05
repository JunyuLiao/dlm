"""Bounded capture: separate the SUPPORT CHOICE from the STALE WEIGHTS.

The v6 five-cell diagnostic could not do this: its "all-kept" cell B meant
`B vs C` mixed "pruning at all" with "pruning chosen from history", and it
ran with uniform sensitivity. Here every cell shares ONE legal-mask support,
one captured state, the same current V/reference/threshold, and a REAL
nonuniform causal T history taken from completed prior logits.

Two selectors on the same state:
  K_f -- the selector run on CURRENT scores (what a fresh selector picks)
  K_h -- the selector run on CACHED scores (what M1 actually picks)

Four outputs, all on the same common legal mask:
  O_ff = current scores on K_f      O_fh = cached scores on K_f
  O_hf = current scores on K_h      O_hh = cached scores on K_h
plus full_current and full_cached (nothing pruned).

  O_hf vs O_ff  -- incremental cost of choosing the support from history
  O_hh vs O_hf  -- cost of stale final weights at the SAME support
  full_cached vs full_current -- stale weights with no pruning at all

error(candidate, reference) = ||candidate - reference|| / max(||reference||, eps)
with both norms and the absolute error stored, so a reader never has to guess
which side the normalizer came from. Support agreement is measured directly
(tile and per-key overlap, retained current mass), never inferred from an
output distance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from dllm.attention.blasst.core import _attention_type, _attention_validity
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from experiments.numerical_qk_reuse import reference as oracle
from experiments.numerical_qk_reuse.cache import Identity, ScoreCache
from experiments.numerical_qk_reuse.cached_executor import attention, route_only
from experiments.numerical_qk_reuse.integration import Attention


def error(candidate, reference, epsilon=1e-12):
    candidate, reference = candidate.float(), reference.float()
    absolute = float((candidate - reference).norm())
    norm_reference = float(reference.norm())
    return dict(absolute_error=absolute, norm_reference=norm_reference,
                norm_candidate=float(candidate.norm()),
                relative_error=absolute / max(norm_reference, epsilon))


def tile_overlap(left, right):
    both = int((left & right).sum())
    either = int((left | right).sum())
    return dict(retained_left=int(left.sum()), retained_right=int(right.sum()),
                intersection=both, union=either,
                jaccard=both / either if either else 1.0,
                disagreeing_tiles=int((left ^ right).sum()))


def retained_mass(scores, retained, nq, nk):
    """Fraction of the CURRENT softmax mass that the retained support keeps."""
    keep = retained.repeat_interleave(128, dim=-2).repeat_interleave(64, dim=-1)[..., :nq, :nk]
    finite = torch.isfinite(scores)
    rows = finite.any(-1, keepdim=True)
    probabilities = torch.softmax(torch.where(rows, scores, torch.zeros_like(scores)), dim=-1)
    probabilities = torch.where(rows & finite, probabilities, torch.zeros_like(probabilities))
    total = probabilities.sum()
    return float((probabilities * keep).sum() / total.clamp_min(1e-12))


class Cells:
    """Shadow evaluator: the dense trajectory it observes is never altered."""

    def __init__(self, adapter, thresholds, layers, score_period, steps):
        self.thresholds, self.layers = thresholds, set(layers)
        self.score_period, self.steps = score_period, set(steps)
        self.cache = ScoreCache(score_period, 1)
        self.projections = Projections()
        self.sources, self.handles, self.rows = {}, [], []
        self.canvas = self.step = self.epoch = -1
        self.sensitivity = None
        for name, module in adapter.model.named_modules():
            if type(module).__name__ == 'DiffusionGemmaEncoderModel':
                self.handles.append(module.register_forward_pre_hook(self._invalidate))
            if adapter.is_blasst_attention_module(name, module):
                self.handles.append(module.register_forward_pre_hook(self._identify, with_kwargs=True))

    def _invalidate(self, *_):
        self.epoch += 1
        self.cache.clear()
        self.sources.clear()

    def _identify(self, module, arguments, kwargs):
        cache = kwargs.get('past_key_values', arguments[3] if len(arguments) > 3 else None)
        if cache is None:
            return
        layer = int(module.layer_idx)
        self.sources[layer] = (cache.get_seq_length(), cache.layers[layer].keys.shape[-2])

    def begin_step(self, canvas, step):
        if canvas != self.canvas:
            self.cache.clear()
        self.canvas, self.step = canvas, step

    @torch.no_grad()
    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, sliding_window=None, **kwargs):
        from scripts.native_reuse_phaseAB_diagnostic import true_dense
        output = true_dense(q, k, v, mask, scaling=scaling, is_causal=is_causal)
        layer = int(module.layer_idx)
        if layer not in self.layers or layer not in self.sources or mask is not None:
            return output
        b, h, nq, d = q.shape
        hk, nk = k.shape[1], k.shape[-2]
        absolute, prefix = self.sources[layer]
        scale = float(scaling) if scaling is not None else d ** -.5
        kind = _attention_type(module, sliding_window)

        # One common legal-mask support for every cell in this row.
        current = Attention.observe_scores(q, k, mask, scale, bool(is_causal), sliding_window, 0)
        identity = Identity(0, self.canvas, self.epoch, layer, b, h, hk, nq, nk, d,
                            absolute, absolute - prefix, nk, scale,
                            str(q.dtype), str(q.device), ('cells',), 0)
        plan = self.cache.plan(identity, self.step)
        if plan.score_refresh:
            self.cache.publish_scores(identity, self.step, current)
        entry = self.cache.get(identity)
        cached = entry.scores
        age = self.step - entry.score_step
        if self.step not in self.steps:
            return output

        matrix = self.projections.get(layer, hk, d, 'gaussian', 32, 1729, v.device)
        current_v = v.float()
        projected = torch.matmul(current_v, matrix).contiguous()
        valid = torch.isfinite(current).reshape(b, hk, h // hk, nq, nk).any((2, 3))
        ref = (current_v.square().sum(-1).masked_fill(~valid, 0.).sum(-1) /
               valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12).contiguous()
        threshold = float(self.thresholds[kind]['log_threshold'])
        sensitivity = None
        if self.sensitivity is not None and self.sensitivity.shape == (b, nq):
            sensitivity = self.sensitivity.contiguous()

        fresh = route_only(current, projected, ref, sensitivity=sensitivity, log_threshold=threshold)
        historical = route_only(cached, projected, ref, sensitivity=sensitivity, log_threshold=threshold)
        keep_f, keep_h = fresh.eligible & ~fresh.skipped, historical.eligible & ~historical.skipped
        never = torch.zeros_like(fresh.skipped)
        vc = v.contiguous()

        o_ff = attention(current, vc, skipped=fresh.skipped, eligible=fresh.eligible).output
        o_fh = attention(cached, vc, skipped=fresh.skipped, eligible=fresh.eligible).output
        o_hf = attention(current, vc, skipped=historical.skipped, eligible=historical.eligible).output
        o_hh = attention(cached, vc, skipped=historical.skipped, eligible=historical.eligible).output
        full_current = attention(current, vc, skipped=never, eligible=fresh.eligible).output
        full_cached = attention(cached, vc, skipped=never, eligible=historical.eligible).output

        self.rows.append(dict(
            canvas=self.canvas, decoder_call=self.step, layer=layer, kind=kind,
            score_age=int(age), prefix=int(prefix), nq=nq, nk=nk,
            nonuniform_T=bool(sensitivity is not None),
            sensitivity_min=None if sensitivity is None else float(sensitivity.min()),
            sensitivity_max=None if sensitivity is None else float(sensitivity.max()),
            support_agreement=tile_overlap(keep_f, keep_h),
            retained_current_mass_fresh=retained_mass(current, keep_f, nq, nk),
            retained_current_mass_historical=retained_mass(current, keep_h, nq, nk),
            # Incremental cost of choosing the support from history.
            O_hf_vs_O_ff=error(o_hf, o_ff),
            # Cost of stale final weights at the SAME (historical) support.
            O_hh_vs_O_hf=error(o_hh, o_hf),
            # Same, at the fresh support.
            O_fh_vs_O_ff=error(o_fh, o_ff),
            # Stale weights with no pruning at all.
            full_cached_vs_full_current=error(full_cached, full_current),
            # Pruning at all, with current weights.
            O_ff_vs_full_current=error(o_ff, full_current),
        ))
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@contextmanager
def observe_steps(model, cells, state):
    original = model._denoising_step
    had = '_denoising_step' in model.__dict__
    saved = model.__dict__.get('_denoising_step')
    book = {'canvas': -1, 'call': -1, 'last': None}

    def step(this, **kwargs):
        schedule = int(kwargs['cur_step'])
        if book['last'] is None or schedule >= book['last']:
            book['canvas'] += 1
            book['call'] = -1
        book['call'] += 1
        book['last'] = schedule
        cells.begin_step(book['canvas'], book['call'])
        # Real nonuniform causal T from COMPLETED prior iterations only.
        state.begin(schedule, kwargs['current_canvas'])
        cells.sensitivity = state.used_weights
        from experiments.value_direction_hopper.query_adaptive import Sampler
        kwargs['sampler'] = Sampler(kwargs['sampler'], state)
        result = original(**kwargs)
        state.finish_step(result, schedule)
        return result

    model._denoising_step = MethodType(step, model)
    try:
        yield
    finally:
        state.finish_canvas()
        if had:
            model._denoising_step = saved
        else:
            del model._denoising_step


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-name', default='T_s50')
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 5])
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 1, 2, 7, 8],
                        help='canvas-local decoder calls to evaluate (0 and 8 are anchors)')
    parser.add_argument('--score-refresh-period', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=600)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    from dllm.models import GenerationRequest, create_adapter
    from experiments.value_direction_hopper.query_adaptive import State

    manifest = {row['id']: row for row in json.loads(args.manifest.read_text(encoding='utf-8'))}
    frozen = json.loads(args.policy.read_text(encoding='utf-8'))
    thresholds = frozen['policies'][args.policy_name]
    adapter = create_adapter('diffusion_gemma', str(args.model), device='cuda',
                             precision='bfloat16', revision=args.revision).load()
    report = {'schema': 'preqk_support_cells_v1', 'score_refresh_period': args.score_refresh_period,
              'ids': {}}
    for id_ in args.ids:
        row = manifest[id_]
        binding = _install_dense(adapter)
        cells = Cells(adapter, thresholds, args.layers, args.score_refresh_period, args.steps)
        binding.runtime.attention_override = cells
        state = State('T', None, m_ref=float(frozen['m_ref']), beta=float(frozen['beta']),
                      gamma=float(frozen['gamma']), diagnostics=False)
        try:
            request = GenerationRequest(prompt=row['prompt'], max_new_tokens=args.max_new_tokens,
                                        temperature=0.0, seed=42, extra={'thinking': True})
            with observe_steps(adapter.model, cells, state):
                adapter.generate(request)
        finally:
            binding.close()
            cells.close()
        report['ids'][id_] = cells.rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"wrote {args.output} with {sum(len(v) for v in report['ids'].values())} cell rows")


if __name__ == '__main__':
    main()
