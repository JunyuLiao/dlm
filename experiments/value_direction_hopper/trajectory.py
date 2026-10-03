"""Optional observation of the native decoder, without copying its generation loop.

Acceptance is reversible in DiffusionGemma. It is not permanent unmasking.
Only fixed_steps changes behavior: it records, but suppresses, adaptive stopping.
"""
from contextlib import contextmanager
from types import MethodType

import torch


def distribution(logits):
    x = logits.float()
    top, ids = x.topk(2, dim=-1)
    logz = torch.logsumexp(x, dim=-1)
    p = (top - logz[..., None]).exp()
    # Store only per-position statistics, never a vocabulary-sized tensor.
    # topk tie ordering differs from argmax, which is what native decoding uses.
    return dict(top1=x.argmax(-1).cpu().flatten().tolist(),
                confidence=p[..., 0].cpu().flatten().tolist(),
                margin=(p[..., 0]-p[..., 1]).cpu().flatten().tolist(),
                logit_margin=(top[..., 0]-top[..., 1]).cpu().flatten().tolist())


class SamplerProbe:
    def __init__(self, inner, record):
        self.inner, self.record = inner, record

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def accept_canvas(self, current, proposed, logits, cur_step):
        result = self.inner.accept_canvas(current, proposed, logits, cur_step)
        self.record.update(distribution(logits))
        # Match the exact dtype and formula used by sampler and stopper.
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        self.record['entropy'] = entropy.cpu().flatten().tolist()
        self.record['entropy_dtype'] = str(entropy.dtype)
        self.record['accepted'] = self.inner.accepted_token_mask.cpu().flatten().tolist()
        self.record['entropy_bound'] = self.inner.entropy_bound
        self.record['sampled_token'] = proposed.cpu().flatten().tolist()
        return result


class StopProbe:
    def __init__(self, inner, record, fixed):
        self.inner, self.record, self.fixed = inner, record, fixed

    def __call__(self, canvas, logits):
        history = self.inner.argmax_canvas_history
        stable = (torch.ones(canvas.shape[0], device=canvas.device, dtype=torch.bool)
                  if self.inner.stability_threshold == 0 else
                  torch.zeros(canvas.shape[0], device=canvas.device, dtype=torch.bool)
                  if history is None else (history == canvas[None]).all(-1).all(0))
        changed = (torch.ones_like(canvas, dtype=torch.bool) if history is None else
                   (history != canvas[None]).any(0))
        result = self.inner(canvas, logits)
        self.record.update(stable=bool(stable.item()),
                           unstable_positions=changed.cpu().flatten().tolist(),
                           confidence_threshold=self.inner.confidence_threshold,
                           stability_threshold=self.inner.stability_threshold,
                           would_stop=bool(result.item()))
        return torch.zeros_like(result) if self.fixed else result


@contextmanager
def observe(model, *, fixed_steps=False, draft_score=None):
    """Instrument just this model instance and restore it even after failure."""
    original = model._denoising_step
    had_instance = '_denoising_step' in model.__dict__
    saved = model.__dict__.get('_denoising_step')
    records = []

    def step(this, **kwargs):
        current = kwargs['current_canvas']
        if current.shape[0] != 1:
            raise ValueError('Trajectory diagnostics require batch size one')
        previous = kwargs['sampler'].accepted_token_mask
        record = dict(step=len(records)+1, remaining_schedule_step=int(kwargs['cur_step']),
                      input_length=int(kwargs['input_ids'].shape[-1]),
                      active_before=([True]*current.shape[-1] if previous is None else
                                     (~previous).cpu().flatten().tolist()))
        kwargs['sampler'] = SamplerProbe(kwargs['sampler'], record)
        stopper = kwargs['diffusion_stopping_criteria']
        if stopper is None:
            raise ValueError('Expected native stable-and-confident stopper')
        kwargs['diffusion_stopping_criteria'] = StopProbe(stopper, record, fixed_steps)
        forward = kwargs['decoder_forward']
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def measured_forward(*args, **kw):
            start.record()
            output = forward(*args, **kw)
            end.record()
            record['raw'] = distribution(output.logits)
            return output

        kwargs['decoder_forward'] = measured_forward
        output = original(**kwargs)
        if record['top1'] != output[1].cpu().flatten().tolist():
            raise AssertionError('Recorded top-1 differs from native argmax')
        end.synchronize()
        record['decoder_ms'] = start.elapsed_time(end)
        record['terminated'] = bool(output[3].item()) or int(kwargs['cur_step']) == 1
        if draft_score:
            record['draft_score'] = draft_score(output[1][0].cpu().tolist())
        records.append(record)
        return output

    model._denoising_step = MethodType(step, model)
    try:
        yield records
    finally:
        if had_instance:
            model._denoising_step = saved
        else:
            del model._denoising_step


class AttentionProbe:
    """Shared-QKV output errors on six queries, two heads and three layers."""
    def __init__(self, router, records):
        self.router, self.records, self.errors = router, records, []

    def __call__(self, module, q, k, v, mask, **kwargs):
        result = self.router(module, q, k, v, mask, **kwargs)
        layer = int(module.layer_idx)
        if layer not in (0, 5, 29):
            return result
        from dllm.attention.blasst.core import _attention_validity, _attention_type
        heads = torch.tensor([0, q.shape[1]//2], device=q.device)
        queries = torch.tensor(sorted({min(i, q.shape[-2]-1) for i in (0,31,63,127,191,255)}), device=q.device)
        valid = _attention_validity(mask, q, k, is_causal=bool(kwargs.get('is_causal', False)),
                                    sliding_window=kwargs.get('sliding_window'))
        valid = valid.expand(q.shape[0], q.shape[1], q.shape[-2], k.shape[-2])[:,heads][:,:,queries]
        qq = q[:,heads][:,:,queries]
        kk = k[:,heads//(q.shape[1]//k.shape[1])]
        vv = v[:,heads//(q.shape[1]//v.shape[1])]
        scores = (qq @ kk.transpose(-1,-2)) * kwargs.get('scaling', q.shape[-1]**-.5)
        if mask is not None and mask.dtype != torch.bool:
            bias = mask.expand(q.shape[0], q.shape[1], q.shape[-2], k.shape[-2])[:,heads][:,:,queries]
            scores = scores + bias
        p = scores.float().masked_fill(~valid, -torch.inf).softmax(-1).nan_to_num()
        dense = p @ vv.float()
        actual = result[0].transpose(1,2)[:,heads][:,:,queries].float()
        error = (actual-dense).square().sum()
        self.errors.append(dict(step=len(self.records)+1, layer=layer,
            attention_type=_attention_type(module,kwargs.get('sliding_window')),
            error_sq=float(error), dense_sq=float(dense.square().sum()),
            cosine_sum=float(torch.nn.functional.cosine_similarity(actual,dense,dim=-1).sum()),
            rows=int(dense.shape[0]*dense.shape[1]*dense.shape[2])))
        return result
