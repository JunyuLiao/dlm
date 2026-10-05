"""Phase C: warm, decomposed forward-cost PROFILE (separate from any timed
generation run). Captures one real (layer, call) state from a short live
generation, then times each component in isolation with CUDA events, after
warmup iterations to let Triton/cuDNN kernel specializations settle. Never
runs inside an accepted E2E timing; this script is only ever invoked on its
own.

Components, matching what the v6 instruction asked to separate rather than
lump into one end-to-end number:
  - native dense: the true installed sdpa_attention_forward path (baseline).
  - score producer: observe_scores' QK matmul + validity + mask (the ONLY
    current-QK producer any numerical arm uses).
  - V lease (cold): Sketches.get() on a brand-new source (full reproject) --
    what production M1 paid on every call before the Phase C repair.
  - V lease (warm): Sketches.get() on the SAME source object/version/shape a
    second time -- what production M1 pays on every call after the repair.
  - decision+PV (fresh decision): cached_executor.attention() computing a new
    routing decision from scores (the _route + _pv kernels).
  - held PV only: cached_executor.attention() consuming an already-held
    bitmap (the _held_eligible + _pv kernels; M3's cheaper steady-state call).

All components use the SAME captured shape/dtype/support/self-conditioning
state; only what work is asked of them differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense
from experiments.numerical_qk_reuse.integration import Attention as NumericalAttention
from experiments.value_direction_hopper.integration import Sketches


def true_dense(q, k, v, mask, *, scaling, is_causal):
    import torch.nn.functional as F
    h, hk = q.shape[1], k.shape[1]
    if h != hk:
        k = k.repeat_interleave(h // hk, dim=1)
        v = v.repeat_interleave(h // hk, dim=1)
    causal = q.shape[2] > 1 and mask is None and bool(is_causal)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.,
                                         scale=scaling, is_causal=causal)
    return out.transpose(1, 2).contiguous(), None


class Capture:
    """Grabs one real (layer, call) attention input via a short live generation,
    then detaches everything so the PROFILE never re-touches the model."""

    def __init__(self, layer, after_call=2):
        self.layer, self.after_call = layer, after_call
        self.state = None
        self.handles = []
        self.sources = {}
        self.call = -1

    def install(self, adapter):
        for name, module in adapter.model.named_modules():
            if adapter.is_blasst_attention_module(name, module):
                self.handles.append(module.register_forward_pre_hook(self._identify, with_kwargs=True))

    def _identify(self, module, args, kwargs):
        cache = kwargs.get('past_key_values', args[3] if len(args) > 3 else None)
        if cache is None or getattr(cache, 'is_compileable', False):
            return
        layer = int(module.layer_idx)
        self.sources[layer] = (cache.layers[layer].keys, cache.layers[layer].values, cache.get_seq_length())

    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None, is_causal=None,
                sliding_window=None, **kwargs):
        out = true_dense(q, k, v, mask, scaling=scaling, is_causal=is_causal)
        layer = int(module.layer_idx)
        if layer == self.layer:
            self.call += 1
            if self.call == self.after_call and self.state is None and layer in self.sources:
                prefix_k, prefix_v, absolute = self.sources[layer]
                self.state = dict(
                    q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
                    mask=mask, scaling=float(scaling) if scaling is not None else q.shape[-1] ** -.5,
                    is_causal=bool(is_causal), sliding_window=sliding_window,
                    prefix_k=prefix_k.detach().clone(), prefix_v=prefix_v.detach().clone(),
                    prefix=int(prefix_k.shape[-2]), absolute=int(absolute), layer=layer,
                )
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def capture_state(model_path, revision, manifest_row, layer, after_call, max_new_tokens):
    from dllm.models import GenerationRequest, create_adapter
    adapter = create_adapter('diffusion_gemma', str(model_path), device='cuda',
                             precision='bfloat16', revision=revision).load()
    binding = _install_dense(adapter)
    capture = Capture(layer, after_call)
    capture.install(adapter)
    binding.runtime.attention_override = capture
    try:
        request = GenerationRequest(prompt=manifest_row['prompt'], max_new_tokens=max_new_tokens,
                                    temperature=0.0, seed=42, extra={'thinking': True})
        adapter.generate(request)
    finally:
        binding.close()
        capture.close()
    if capture.state is None:
        raise RuntimeError(f'Layer {layer} never reached call {after_call}')
    return capture.state, adapter


def bare_sketches(projections):
    """A Sketches instance with no installed hooks -- profile_state supplies
    `sources` directly instead of via a live forward_pre_hook, to isolate
    get()'s own cost from hook-installation/model-scan cost."""
    s = Sketches.__new__(Sketches)
    s.projections, s.entries, s.sources, s.handles = projections, {}, {}, []
    s.fused = False
    s.epoch = 0
    s.projected_tokens = s.reused_tokens = s.peak_storage_bytes = s.projection_madds = 0
    return s


def timed(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps  # ms/rep


@torch.no_grad()
def profile_state(state, thresholds, kind, *, warmup=10, reps=50):
    from experiments.numerical_qk_reuse.cached_executor import attention as triton_attention

    q, k, v = state['q'], state['k'], state['v']
    b, h, nq, d = q.shape
    hk = k.shape[1]
    threshold = float(thresholds[kind]['log_threshold'])
    projections = Projections()

    rows = {}
    rows['native_dense_ms'] = timed(
        lambda: true_dense(q, k, v, state['mask'], scaling=state['scaling'], is_causal=state['is_causal']),
        warmup, reps)

    rows['score_producer_ms'] = timed(
        lambda: NumericalAttention.observe_scores(q, k, state['mask'], state['scaling'],
                                                   state['is_causal'], state['sliding_window'], 0),
        warmup, reps)

    scores = NumericalAttention.observe_scores(q, k, state['mask'], state['scaling'],
                                               state['is_causal'], state['sliding_window'], 0)
    valid = torch.isfinite(scores).reshape(b, hk, h // hk, nq, k.shape[-2]).any((2, 3))

    def cold_lease():
        # A brand-new Sketches AND a brand-new source tensor every call: the
        # lease can never fire, matching what production M1 paid on every
        # single call before the Phase C repair. Sketches.version() calls
        # torch.is_inference/._version, so the stand-in source must be a real
        # tensor, not a bare object().
        s = bare_sketches(projections)
        s.sources[0] = torch.empty(1, device=v.device)
        return s.get(0, v, valid, state['prefix'])
    rows['v_lease_cold_ms'] = timed(cold_lease, warmup, reps)

    warm_sketches = bare_sketches(projections)
    warm_source = torch.empty(1, device=v.device)
    warm_sketches.sources = {0: warm_source}
    warm_sketches.get(0, v, valid, state['prefix'])  # first call: cold, primes the lease

    rows['v_lease_warm_ms'] = timed(lambda: warm_sketches.get(0, v, valid, state['prefix']), warmup, reps)

    projected, ref = warm_sketches.get(0, v, valid, state['prefix'])

    rows['decision_and_pv_fresh_ms'] = timed(
        lambda: triton_attention(scores, v, projected.contiguous(), ref.contiguous(), log_threshold=threshold),
        warmup, reps)

    route = triton_attention(scores, v, projected.contiguous(), ref.contiguous(), log_threshold=threshold)
    # Also the routing_only_current_output "held" call: kernel cost does not
    # depend on whether `scores` is stale or fresh, only on the retained
    # bitmap's shape -- so held_pv_only_ms IS routing_only's extra PV pass.
    rows['held_pv_only_ms'] = timed(
        lambda: triton_attention(scores, v, skipped=route.skipped, eligible=route.eligible), warmup, reps)

    return rows


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-name", default="T_s50")
    parser.add_argument("--id", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--after-call", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    manifest = {row['id']: row for row in json.loads(args.manifest.read_text(encoding='utf-8'))}
    thresholds = json.loads(args.policy.read_text(encoding='utf-8'))['policies'][args.policy_name]
    row = manifest[args.id]

    report: dict[str, Any] = {'schema': 'numerical_reuse_warm_profile_v1', 'id': args.id,
                              'warmup': args.warmup, 'reps': args.reps, 'layers': {}}
    for layer in args.layers:
        state, adapter = capture_state(args.model, args.revision, row, layer, args.after_call,
                                       args.max_new_tokens)
        kind = 'local' if state['sliding_window'] else 'global'
        report['layers'][str(layer)] = dict(
            kind=kind, prefix=state['prefix'], nq=state['q'].shape[2], nk=state['k'].shape[2],
            timings_ms_per_call=profile_state(state, thresholds, kind, warmup=args.warmup, reps=args.reps))
        del adapter
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
