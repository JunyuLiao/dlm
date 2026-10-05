"""Opt-in, untimed same-QKV support-consumer operator probe for v20.

Only the first A/D/H call per selected layer and canvas is inspected. The
already observed bitmap is supplied to the same numerical current consumer;
no score cache, sketch, selector, projection or full-forward cost is included.
"""
from __future__ import annotations

import statistics

LAYERS = frozenset((0, 5))
PHASES = frozenset(('A', 'D', 'H'))


def within_v11_envelope(new_error, triton_error, relative_error):
    """The pinned v11 same-support Hopper/Triton-vs-FP32 envelope."""
    return (new_error <= 1.5 * triton_error + 1e-3 and relative_error <= 1e-2)


def _reference(q, k, v, skipped, eligible, scale):
    import math
    import torch
    b, h, nq, _ = q.shape
    hk, nk = k.shape[1], k.shape[2]
    if h % hk:
        raise ValueError('unqualified grouped-query geometry')
    kk = k.repeat_interleave(h // hk, 1).float()
    vv = v.repeat_interleave(h // hk, 1).float()
    scores = torch.matmul(q.float(), kk.transpose(-1, -2))
    scores = (scores.to(torch.bfloat16).float() * scale).to(torch.bfloat16).float()
    keep = ((eligible & ~skipped).repeat_interleave(128, 2)[:, :, :nq]
            .repeat_interleave(64, 3)[..., :nk])
    scores = scores.masked_fill(~keep, -math.inf)
    rows = keep.any(-1, keepdim=True)
    probabilities = torch.softmax(scores.masked_fill(~rows, 0.), -1) * rows
    return torch.matmul(probabilities, vv)


class OperatorProbe:
    def __init__(self, *, repetitions=3, digest=None):
        if type(repetitions) is not int or repetitions < 3:
            raise ValueError('operator probe needs at least three measured repetitions')
        self.repetitions, self.digest = repetitions, digest
        self.seen, self.rows = set(), []

    def wants(self, *, layer, canvas, phase):
        return layer in LAYERS and phase in PHASES and (canvas, layer, phase) not in self.seen

    def observe(self, router, module, q, k, v, skipped, eligible, *, phase, canvas, step,
                scale, actual_output):
        import torch
        from experiments.numerical_qk_reuse.cached_executor import preqk_attention
        layer = int(module.layer_idx)
        if not self.wants(layer=layer, canvas=canvas, phase=phase):
            return
        if (router.support != 'native_mask' or router.output_mode !=
                'historical_route_preqk_current_output' or router._mask_present):
            raise ValueError('operator probe requires the pinned native current consumer')
        if scale is None:
            scale = q.shape[-1] ** -.5
        scale = float(scale)
        before = (skipped.clone(), eligible.clone())
        reference = _reference(q, k, v, skipped, eligible, scale)
        tri = preqk_attention(q, k, v, skipped, eligible, scale=scale,
                              window=None, is_causal=False, variant='generic').output.float()

        def consume():
            return router._consume(q, k, v, skipped, eligible, scale, None, False)

        consume()  # compilation/warmup outside all measured repetitions
        torch.cuda.synchronize()
        spans = []
        current = None
        for _ in range(self.repetitions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            current = consume()
            end.record()
            torch.cuda.synchronize()
            spans.append(start.elapsed_time(end))
        if not torch.equal(before[0], skipped) or not torch.equal(before[1], eligible):
            raise ValueError('operator probe mutated frozen support')
        if bool(current.invalid_scores.any()):
            raise ValueError('same-support consumer reported invalid scores')
        candidate = current.output.float()
        if not isinstance(actual_output, (tuple, list)) or not actual_output:
            raise ValueError('actual routed output unavailable for same-QKV comparison')
        actual = actual_output[0].transpose(1, 2).float()
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError('same-support consumer produced nonfinite output')
        if not bool(torch.isfinite(actual).all()):
            raise ValueError('actual routed output is nonfinite')
        new_error = float((candidate-reference).abs().max())
        triton_error = float((tri-reference).abs().max())
        relative_error = float((candidate-reference).norm() / reference.norm().clamp_min(1e-30))
        actual_error = float((actual-reference).abs().max())
        actual_relative = float((actual-reference).norm() / reference.norm().clamp_min(1e-30))
        if not within_v11_envelope(new_error, triton_error, relative_error):
            raise ValueError('same-QKV support consumer exceeded pinned v11 numeric envelope')
        if not within_v11_envelope(actual_error, triton_error, actual_relative):
            raise ValueError('actual routed output exceeded pinned v11 numeric envelope')
        self.seen.add((canvas, layer, phase))
        self.rows.append(dict(layer=layer, canvas=canvas, decoder_call=step, phase=phase,
                              kind='local' if bool(module.is_sliding) else 'global',
                              q=int(q.shape[-2]), k=int(k.shape[-2]),
                              qkv_digest=(self.digest(q, k, v) if self.digest else None),
                              bitmap_shape=list(skipped.shape),
                              new_max_abs_error=new_error, triton_max_abs_error=triton_error,
                              relative_l2_error=relative_error,
                              actual_output_max_abs_error=actual_error,
                              actual_output_relative_l2_error=actual_relative,
                              v11_envelope_pass=True,
                              prepared_consumer_event_median_ms=statistics.median(spans),
                              prepared_consumer_event_samples_ms=spans,
                              definition='same actual QKV and observed support; operator only; '
                              'no projection, route, score cache, model forward or generation cost'))
